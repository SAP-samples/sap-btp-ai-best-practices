"""TODO management tools for task planning and progress tracking.

Implements `write_todos` and `read_todos` tools compatible with LangGraph.
These allow the agent to maintain a running TODO list inside graph state and
recite/update it as tasks progress, inspired by the deep-agents-from-scratch
notebook lesson.
"""

from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


class TodoItem(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: One of: "pending", "in_progress", "completed"
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


# _WRITE_TODOS_DESCRIPTION = (
#     "Create or update a concise, high-signal TODO list to plan and track work.\n\n"
#     "Guidelines:\n"
#     "- Keep each item short (≤14 words) and action-oriented.\n"
#     "- Use statuses: pending | in_progress | completed (only one in_progress ideally).\n"
#     "- Re-write the full list when plans change; do not append duplicates.\n"
#     "- Check off completed items immediately; keep the list current.\n"
#     "- Include only meaningful, user-facing tasks; avoid low-level operational steps.\n"
#     "- If starting a new task, set it to in_progress and ensure only one is active.\n"
# )

_WRITE_TODOS_DESCRIPTION = """Create and manage structured task lists for tracking progress through complex workflows.

## When to Use
- Multi-step or non-trivial tasks requiring coordination
- When user provides multiple tasks or explicitly requests todo list
- Avoid for single, trivial actions unless directed otherwise

## Structure
- Maintain one list containing multiple todo objects (content, status, id)
- Use clear, actionable content descriptions
- Status must be: pending, in_progress, or completed

## Best Practices
- Only one in_progress task at a time
- Mark completed immediately when task is fully done
- Always send the full updated list when making changes
- Prune irrelevant items to keep list focused

## Progress Updates
- Call write_todos again to change task status or edit content
- Reflect real-time progress; don't batch completions
- If blocked, keep in_progress and add new task describing blocker

## Parameters
- todos: List of TODO items with content and status fields

## Returns
Updates agent state with new todo list."""


@tool(description=_WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: list[TodoItem], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    print("[TODO] Updated TODOs:", todos)
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list from the agent state.

    Retrieves and formats the TODO list so the model can refresh its working
    memory and stay focused on remaining tasks.

    Args:
        state: Injected agent state containing the current TODO list.
        tool_call_id: Injected tool call identifier for message tracking.

    Returns:
        A human-readable string representation of the current TODO list. If no
        todos exist, returns a string indicating the list is empty.
    """
    todos = state.get("todos", []) if isinstance(state, dict) else []
    print("[TODO] Read todos:", todos)
    if not todos:
        return "No todos currently in the list."

    status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
    lines = ["Current TODO List:"]
    for i, todo in enumerate(todos, 1):
        emoji = status_emoji.get(todo.get("status", ""), "❓")
        lines.append(f"{i}. {emoji} {todo.get('content')} ({todo.get('status')})")
    return "\n".join(lines)


__all__ = ["TodoItem", "write_todos", "read_todos"]
