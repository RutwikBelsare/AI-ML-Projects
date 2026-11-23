
from uuid import uuid4
import traceback
from fastapi import APIRouter, HTTPException
from app.agent.graph import resume_research
import json
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.rag.dynamic_ingest import (
    ingest_uploaded_document,
    MAX_FILE_SIZE,
)
import logging

from app.rag.document_manager import (
    list_uploaded_documents,
    delete_uploaded_document,
)

import queue
import threading
from app.api.conversation_store import (
    create_conversation,
    list_conversations,
    get_conversation,
    add_message,
    delete_conversation,
)
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, ApprovalRequest
from app.agent.graph import run_research_streaming
from pydantic import BaseModel, Field



router = APIRouter(prefix="/api", tags=["Research Agent"])

logger = logging.getLogger(__name__)

def save_user_message(
    thread_id: str,
    user_id: str,
    question: str,
):
    """Create conversation if needed and save the user's question."""

    conversation = get_conversation(thread_id, user_id)

    if conversation is None:
        title = question.strip()[:60] or "New Chat"

        create_conversation(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
        )

    add_message(
        thread_id=thread_id,
        user_id=user_id,
        role="user",
        content=question,
    )


def save_assistant_message(
    thread_id: str,
    user_id: str,
    answer: str,
):
    """Save the agent's final response."""

    if answer and answer.strip():
        add_message(
            thread_id=thread_id,
            user_id=user_id,
            role="assistant",
            content=answer,
        )
@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Execute a research question using the LangGraph agent."""

    thread_id = request.thread_id or f"thread_{uuid4().hex[:12]}"

    try:
        # Save the user's question.
        save_user_message(
            thread_id=thread_id,
            user_id=request.user_id,
            question=request.question,
        )

        state, pending_interrupts = run_research_streaming(
            question=request.question,
            thread_id=thread_id,
            user_id=request.user_id,
        )

        approval_required = bool(pending_interrupts)

        status = (
            "approval_required"
            if approval_required
            else state.get("status", "completed")
        )

        final_answer = state.get("final_answer", "")

        # Save the final response only when approval is not pending.
        if not approval_required:
            save_assistant_message(
                thread_id=thread_id,
                user_id=request.user_id,
                answer=final_answer,
            )

        return ChatResponse(
            thread_id=thread_id,
            user_id=request.user_id,
            status=status,
            final_answer=final_answer,
            approval_required=approval_required,
            errors=state.get("errors", []),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Research execution failed. Check backend logs.",
        ) from exc

    
class ConversationCreateRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(default="New Chat", min_length=1, max_length=200)


class ConversationUserRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)


@router.post("/conversations")
def create_new_conversation(request: ConversationCreateRequest):
    try:
        create_conversation(
            thread_id=request.thread_id,
            user_id=request.user_id,
            title=request.title,
        )
        return {
            "message": "Conversation created successfully.",
            "thread_id": request.thread_id,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/conversations")
def get_conversation_list(user_id: str):
    return {
        "conversations": list_conversations(user_id)
    }


@router.get("/conversations/{thread_id}")
def get_conversation_details(thread_id: str, user_id: str):
    conversation = get_conversation(thread_id, user_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return conversation


@router.delete("/conversations/{thread_id}")
def remove_conversation(thread_id: str, user_id: str):
    deleted = delete_conversation(thread_id, user_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    return {"message": "Conversation deleted successfully."}


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream research agent events using Server-Sent Events."""

    thread_id = request.thread_id or f"thread_{uuid4().hex[:12]}"
    event_queue = queue.Queue()

    def publish_event(event: dict):
        event_queue.put(event)

    def execute_agent():
        try:
            # Save the user's question before executing the agent.
            save_user_message(
                thread_id=thread_id,
                user_id=request.user_id,
                question=request.question,
            )

            state, pending_interrupts = run_research_streaming(
                question=request.question,
                thread_id=thread_id,
                user_id=request.user_id,
                event_callback=publish_event,
            )

            # Save the final answer only if approval is not pending.
            if not pending_interrupts:
                save_assistant_message(
                    thread_id=thread_id,
                    user_id=request.user_id,
                    answer=state.get("final_answer", ""),
                )

        except Exception:
            publish_event({
                "type": "error",
                "message": (
                    "Research execution failed. "
                    "Check backend logs."
                ),
            })

        finally:
            event_queue.put(None)

    def event_generator():
        yield (
            "event: started\n"
            + "data: "
            + json.dumps({
                "thread_id": thread_id,
                "user_id": request.user_id,
            })
            + "\n\n"
        )

        worker = threading.Thread(
            target=execute_agent,
            daemon=True,
        )
        worker.start()

        while True:
            event = event_queue.get()

            if event is None:
                break

            event_type = event.get("type", "message")

            yield (
                f"event: {event_type}\n"
                f"data: {json.dumps(event)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
@router.post("/chat/approve")
def approve_tool_execution(request: ApprovalRequest):
    """Approve or reject pending tool execution."""

    try:
        state = resume_research(
            thread_id=request.thread_id,
            user_id=request.user_id,
            approved=request.approved,
        )

        final_answer = state.get("final_answer", "")

        # Save the agent's final response after approval/rejection.
        save_assistant_message(
            thread_id=request.thread_id,
            user_id=request.user_id,
            answer=final_answer,
        )

        return ChatResponse(
            thread_id=request.thread_id,
            user_id=request.user_id,
            status=state.get("status", "completed"),
            final_answer=final_answer,
            approval_required=False,
            errors=state.get("errors", []),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Approval processing failed. Check backend logs.",
        ) from exc



@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and permanently index it in ChromaDB.
    Supported formats: PDF, TXT, DOCX, CSV.
    """

    try:
        # Validate filename.
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required.",
            )

        # Read at most 20 MB + 1 byte to detect oversized files.
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size cannot exceed 20 MB.",
            )

        # Ingest and permanently index the document.
        result = ingest_uploaded_document(
            filename=file.filename,
            content=content,
        )

        return {
            "success": True,
            "message": result["message"],
            "status": result["status"],
            "document": {
                "document_id": result.get("document_id"),
                "filename": result.get("filename"),
                "chunks_indexed": result.get("chunks_indexed", 0),
            },
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document upload failed. Check backend logs.",
        ) from exc

    finally:
        await file.close()


# Document Management APIs
@router.get("/documents")
def get_uploaded_documents():
    """ Return all permanently uploaded documents. """

    try:
        documents = list_uploaded_documents()
        return{
            "sucess":True,
            "total_documents": len(documents),
            "documents":documents

        }
    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail = "Unable to retrieve documents.",
        )from exc


@router.delete("/documents/{document_id}")
def remove_uploaded_document(document_id: str):
    """Delete an uploaded document and its indexed chunks."""

    print(
        f"\n========== DOCUMENT DELETION STARTED =========="
    )
    print(f"Document ID: {document_id}")

    try:
        result = delete_uploaded_document(document_id)

        print("Document deletion completed successfully.")
        print("===============================================\n")

        return {
            "success": True,
            "message": "Document deleted successfully.",
            "document": result,
        }

    except ValueError as exc:
        print(f"VALIDATION ERROR: {exc}")
        raise HTTPException(
            status_code=404 if "not found" in str(exc).lower() else 400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print("\n========== DOCUMENT DELETION FAILED ==========")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception message: {exc}")
        traceback.print_exc()
        print("===============================================\n")

        logger.exception(
            "Document deletion failed for ID: %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {type(exc).__name__}: {exc}",
        ) from exc
