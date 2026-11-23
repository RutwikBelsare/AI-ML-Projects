# app/agent/hitl_demo.py

import sys
import sqlite3

from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver



# 1. Define Graph State


class ApprovalState(TypedDict, total=False):
    action: str
    approved: bool
    execution_result: str
    status: str



# 2. Approval Node
def approval_node(state: ApprovalState) -> dict:
    """
    Pause execution and request human approval.
    """

    print("\nWaiting for human approval...")

    decision = interrupt({
        "type": "approval_required",
        "action": state["action"],
        "message": (
            f"Do you approve this action: "
            f"{state['action']}?"
        ),
    })

    return {
        "approved": decision.get("approved", False),
        "status": "approval_received",
    }



# 3. Conditional Routing
def route_after_approval(state: ApprovalState) -> str:

    if state.get("approved") is True:
        return "execute"

    return "reject"


# ==========================================
# 4. Execute Action Node
# ==========================================

def execute_node(state: ApprovalState) -> dict:
    """
    Execute the action only after approval.

    This is a simulated action for learning.
    """

    action = state["action"]

    print("\nExecuting approved action...")

    result = f"Successfully executed: {action}"

    return {
        "execution_result": result,
        "status": "executed",
    }


# ==========================================
# 5. Reject Action Node
# ==========================================

def reject_node(state: ApprovalState) -> dict:

    print("\nHuman rejected the action.")

    return {
        "execution_result": (
            f"Action cancelled: {state['action']}"
        ),
        "status": "rejected",
    }


# ==========================================
# 6. Create Graph
# ==========================================

builder = StateGraph(ApprovalState)

builder.add_node("approval", approval_node)
builder.add_node("execute", execute_node)
builder.add_node("reject", reject_node)

builder.add_edge(START, "approval")

builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "execute": "execute",
        "reject": "reject",
    },
)

builder.add_edge("execute", END)
builder.add_edge("reject", END)


# ==========================================
# 7. SQLite Checkpointer
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = CHECKPOINT_DIR / "hitl_demo.sqlite"


def create_graph(connection):

    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return builder.compile(
        checkpointer=checkpointer
    )


# ==========================================
# 8. Start Approval Workflow


def start_workflow(action: str, thread_id: str):

    with sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    ) as connection:

        graph = create_graph(connection)

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = graph.invoke(
            {
                "action": action,
                "status": "started",
            },
            config=config,
        )

        print("\n=== WORKFLOW STATUS ===")
        print("Status:", result.get("status"))

        if result.get("__interrupt__"):
            for item in result["__interrupt__"]:
                print("Approval request:", item.value)

        return result



# 9. Resume Workflow


def resume_workflow(thread_id: str, approved: bool):

    with sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    ) as connection:

        graph = create_graph(connection)

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        result = graph.invoke(
            Command(
                resume={
                    "approved": approved
                }
            ),
            config=config,
        )

        print("\n=== FINAL RESULT ===")
        print("Status:", result.get("status"))
        print(
            "Execution result:",
            result.get("execution_result")
        )

        return result



# 10. Command Line Entry Point
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "Start:  python -m app.agent.hitl_demo "
            "start <thread_id>\n"
            "Resume: python -m app.agent.hitl_demo "
            "resume <thread_id> <yes/no>"
        )
        raise SystemExit(1)

    command = sys.argv[1]
    thread_id = sys.argv[2]

    if command == "start":

        action = "Execute a sample research action"

        start_workflow(
            action=action,
            thread_id=thread_id,
        )

    elif command == "resume":

        if len(sys.argv) != 4:
            raise ValueError(
                "Provide approval decision: yes or no."
            )

        decision = sys.argv[3].lower()

        if decision not in ("yes", "no"):
            raise ValueError("Decision must be yes or no.")

        resume_workflow(
            thread_id=thread_id,
            approved=(decision == "yes"),
        )

    else:
        raise ValueError("Use start or resume.")
