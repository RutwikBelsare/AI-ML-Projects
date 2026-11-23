from fastapi import APIRouter, Depends

from backend.schemas.requests import ChatRequest
from backend.schemas.responses import ChatResponse
from backend.services.dependencies import get_support_service, get_tool_calling_service
from backend.services.support_services import SupportService
from fastapi.responses import StreamingResponse

from backend.services.tool_calling_service import ToolCallingService



router = APIRouter(
    prefix="/api/v1/support",
    tags=["Customer Support"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    support_service: SupportService = Depends(
        get_support_service
    ),
):

    result = await support_service.process_message(
        request.message
    )

    return ChatResponse(
        success=True,
        data=result,
    )
@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    support_service: SupportService = Depends(
        get_support_service
    ),
):
    return StreamingResponse(
        support_service.stream_message(
            request.message
        ),
        media_type="text/plain",
    )

@router.post("/tool-chat")
async def tool_chat(
    request: ChatRequest,
    tool_service: ToolCallingService = Depends(
        get_tool_calling_service
    ),
):
    result = await tool_service.process_message(
        request.message
    )

    return {
        "success": True,
        "response": result,
    }

@router.post("/tool-chat/stream")
async def tool_chat_stream(
    request: ChatRequest,
    tool_service: ToolCallingService = Depends(
        get_tool_calling_service
    ),
):
    return StreamingResponse(
        tool_service.stream_message(
            request.message
        ),
        media_type="text/plain",
    )