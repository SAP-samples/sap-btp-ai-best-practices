from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

try:
    from .agent import run_agent
except ImportError:  # pragma: no cover
    from agent import run_agent
try:
    from ..security import get_api_key
except ImportError:  # pragma: no cover
    from security import get_api_key

router = APIRouter()


JSONRPC_VERSION = "2.0"

# Standard JSON-RPC Error Codes
ERROR_PARSE = -32700
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL = -32603

# A2A-specific Error Codes
ERROR_TASK_NOT_FOUND = -32001
ERROR_TASK_NOT_CANCELABLE = -32002
ERROR_PUSH_NOT_SUPPORTED = -32003
ERROR_UNSUPPORTED_OPERATION = -32004
ERROR_CONTENT_TYPE_NOT_SUPPORTED = -32005

TERMINAL_STATES = {"completed", "canceled", "failed", "rejected"}

# Storage for tasks and conversation contexts
_TASKS: Dict[str, Dict[str, Any]] = {}
_CONTEXTS: Dict[str, List[Dict[str, Any]]] = {}


# ==============================================================================
# JSON-RPC Response Helpers
# ==============================================================================

def _now_iso() -> str:
    """Returns current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _jsonrpc_error(
    req_id: Optional[Any],
    code: int,
    message: str,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates a JSON-RPC error response.

    Args:
        req_id: The request ID to include in the response.
        code: The error code (negative integer).
        message: Human-readable error message.
        data: Optional additional error data.

    Returns:
        JSON-RPC error response dictionary.
    """
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "error": error}


def _jsonrpc_result(req_id: Any, result: Any) -> Dict[str, Any]:
    """Creates a JSON-RPC success response.

    Args:
        req_id: The request ID to include in the response.
        result: The result payload.

    Returns:
        JSON-RPC result response dictionary.
    """
    return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": result}


def _tool_result_stats(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build compact log-safe stats for tool results."""
    stats: List[Dict[str, Any]] = []
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        stats.append(
            {
                "name": item.get("name"),
                "row_count_estimate": item.get("row_count_estimate"),
                "payload_bytes_estimate": item.get("payload_bytes_estimate"),
                "content_truncated": bool(item.get("content_truncated")),
            }
        )
    return stats




# ==============================================================================
# A2A Message/Task Building Helpers
# ==============================================================================

def _text_part(text: str) -> Dict[str, Any]:
    """Creates an A2A text part."""
    return {"kind": "text", "text": text}


def _message(
    role: str,
    text: str,
    message_id: Optional[str] = None,
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates an A2A message.

    Args:
        role: Message role ("user" or "agent").
        text: Message text content.
        message_id: Optional message ID (generated if not provided).
        task_id: Optional task ID to associate with the message.
        context_id: Optional context ID for conversation threading.

    Returns:
        A2A message dictionary.
    """
    msg_id = message_id or str(uuid.uuid4())
    msg: Dict[str, Any] = {
        "kind": "message",
        "role": role,
        "parts": [_text_part(text)],
        "messageId": msg_id,
    }
    if task_id:
        msg["taskId"] = task_id
    if context_id:
        msg["contextId"] = context_id
    return msg


def _artifact(text: str) -> Dict[str, Any]:
    """Creates an A2A artifact containing text content."""
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "answer.txt",
        "parts": [_text_part(text)],
    }


def _task(
    task_id: str,
    context_id: str,
    state: str,
    status_message: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Creates an A2A task object.

    Args:
        task_id: Unique task identifier.
        context_id: Conversation context identifier.
        state: Task state (e.g., "completed", "failed").
        status_message: Optional status message (typically the agent's response).
        history: Optional list of messages in the conversation.
        artifacts: Optional list of artifacts produced.

    Returns:
        A2A task dictionary.
    """
    task: Dict[str, Any] = {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": _now_iso()},
    }
    if status_message:
        task["status"]["message"] = status_message
    if history is not None:
        task["history"] = history
    if artifacts is not None:
        task["artifacts"] = artifacts
    return task


def _apply_history_length(
    task: Dict[str, Any], history_length: Optional[int]
) -> Dict[str, Any]:
    """Trims task history to the specified length.

    Args:
        task: The task dictionary to process.
        history_length: Maximum number of history entries to keep.
            None or negative means keep all.

    Returns:
        Task dictionary with trimmed history (or original if no trimming needed).
    """
    if history_length is None or history_length < 0:
        return task
    result = dict(task)
    history = task.get("history")
    if isinstance(history, list):
        result["history"] = history[-history_length:] if history_length else []
    return result


# ==============================================================================
# Validation Helpers
# ==============================================================================

def _validate_jsonrpc_request(payload: Any) -> Optional[Dict[str, Any]]:
    """Validates JSON-RPC request structure.

    Args:
        payload: The parsed JSON payload.

    Returns:
        Error response if validation fails, None if valid.
    """
    if not isinstance(payload, dict):
        return _jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request")
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        return _jsonrpc_error(
            payload.get("id"), ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request"
        )
    if "id" not in payload or payload.get("id") is None:
        return _jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request")
    if not isinstance(payload.get("method"), str):
        return _jsonrpc_error(
            payload.get("id"), ERROR_INVALID_REQUEST, "Invalid JSON-RPC Request"
        )
    return None


def _extract_text_parts(parts: List[Dict[str, Any]]) -> Optional[str]:
    """Extracts and concatenates text from A2A message parts.

    Args:
        parts: List of A2A message parts.

    Returns:
        Concatenated text if all parts are text, None if any part is non-text.
    """
    texts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            return None
        if part.get("kind") != "text":
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        texts.append(text)
    return "\n".join(texts)


def _validate_message_params(
    params: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Validates message/send parameters and extracts key fields.

    Validates the message structure according to A2A protocol requirements:
    - message must be a dict with role, parts, and messageId
    - role must be "user" or "agent"
    - parts must be a non-empty list of text parts

    Args:
        params: The params object from the JSON-RPC request.

    Returns:
        Tuple of (error_dict, text, message_id, context_id).
        If error_dict is not None, validation failed and it should be returned.
        Otherwise, text, message_id, and context_id contain the extracted values.
    """
    message = params.get("message")
    if not isinstance(message, dict):
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
        )

    role = message.get("role")
    if role not in {"user", "agent"}:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
        )

    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
        )

    text = _extract_text_parts(parts)
    if text is None:
        return (
            {"code": ERROR_CONTENT_TYPE_NOT_SUPPORTED, "message": "Incompatible content types"},
            None,
            None,
            None,
        )

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        return (
            {"code": ERROR_INVALID_PARAMS, "message": "Invalid method parameters"},
            None,
            None,
            None,
        )

    context_id = message.get("contextId")
    return (None, text, message_id, context_id)


# ==============================================================================
# Context Management
# ==============================================================================

def _get_or_create_context_history(context_id: str) -> List[Dict[str, Any]]:
    """Gets existing context history or creates a new empty one.

    Args:
        context_id: The conversation context identifier.

    Returns:
        List of A2A messages for this context (mutable reference).
    """
    if context_id not in _CONTEXTS:
        _CONTEXTS[context_id] = []
    return _CONTEXTS[context_id]


# ==============================================================================
# Task Response Building
# ==============================================================================

def _build_task_response(
    task_id: str,
    context_id: str,
    history: List[Dict[str, Any]],
    agent_text: str,
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
    include_tool_calls: bool,
) -> Dict[str, Any]:
    """Builds the complete task response object.

    Creates the agent message, artifacts, and assembles the full task
    structure for the JSON-RPC response.

    Args:
        task_id: Unique task identifier.
        context_id: Conversation context identifier.
        history: List of A2A messages in the conversation (will be modified).
        agent_text: The agent's response text.
        tool_calls: List of tool calls made by the agent.
        include_tool_calls: Whether to include tool calls in task metadata.

    Returns:
        Complete A2A task dictionary.
    """
    agent_msg = _message(
        role="agent",
        text=agent_text,
        task_id=task_id,
        context_id=context_id,
    )
    history.append(agent_msg)

    artifacts = [_artifact(agent_text)]
    task = _task(
        task_id=task_id,
        context_id=context_id,
        state="completed",
        status_message=agent_msg,
        history=history,
        artifacts=artifacts,
    )

    if include_tool_calls:
        metadata: Dict[str, Any] = {"toolCalls": tool_calls}
        if tool_results:
            metadata["toolResults"] = tool_results
        task["metadata"] = metadata

    return task


# ==============================================================================
# JSON-RPC Method Handlers
# ==============================================================================

async def _handle_message_send(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the message/send JSON-RPC method.

    This is the main entry point for sending messages to the agent. It:
    1. Validates the message parameters
    2. Creates or retrieves the conversation context
    3. Invokes the agent with the user's message
    4. Builds and stores the task response

    Args:
        req_id: The JSON-RPC request ID.
        params: The method parameters containing the message.

    Returns:
        JSON-RPC response (either result or error).
    """
    # Step 1: Validate parameters
    error, text, message_id, provided_context_id = _validate_message_params(params)
    if error:
        return _jsonrpc_error(req_id, error["code"], error["message"])

    # Step 2: Setup context and task identifiers
    context_id = provided_context_id or str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    # Step 3: Create user message and add to context history
    user_msg = _message(
        role="user",
        text=text,
        message_id=message_id,
        task_id=task_id,
        context_id=context_id,
    )
    history = _get_or_create_context_history(context_id)
    history.append(user_msg)

    # Step 4: Run the agent (LangGraph manages LLM context via thread_id)
    try:
        agent_result = await run_agent(text, context_id)
    except Exception as exc:
        return _jsonrpc_error(
            req_id, ERROR_INTERNAL, "Internal server error", {"detail": str(exc)}
        )

    # Step 5: Build the task response
    metadata = params.get("metadata") if isinstance(params, dict) else None
    include_tool_calls = bool(
        isinstance(metadata, dict) and metadata.get("includeToolCalls")
    )

    tool_results = agent_result.get("tool_results", []) or []
    final_text = agent_result.get("text", "")

    task = _build_task_response(
        task_id=task_id,
        context_id=context_id,
        history=history,
        agent_text=final_text,
        tool_calls=agent_result.get("tool_calls", []),
        tool_results=tool_results,
        include_tool_calls=include_tool_calls,
    )

    # Step 6: Log tool calls if enabled
    if os.getenv("A2A_LOG_TOOL_CALLS", "false").strip().lower() in {"1", "true", "yes"}:
        print(f"[a2a] tool_calls={agent_result.get('tool_calls', [])}")
        if tool_results:
            print(f"[a2a] tool_result_stats={_tool_result_stats(tool_results)}")

    # Step 7: Store task and return response
    _TASKS[task_id] = {"task": task, "context_id": context_id}

    config = params.get("configuration") or {}
    history_length = config.get("historyLength")
    task_response = _apply_history_length(
        task, history_length if isinstance(history_length, int) else None
    )
    return _jsonrpc_result(req_id, task_response)


def _handle_tasks_get(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the tasks/get JSON-RPC method.

    Retrieves a task by its ID, optionally trimming the history.

    Args:
        req_id: The JSON-RPC request ID.
        params: The method parameters containing the task ID.

    Returns:
        JSON-RPC response with the task or error.
    """
    task_id = params.get("id") if isinstance(params, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "Invalid method parameters")
    if task_id not in _TASKS:
        return _jsonrpc_error(req_id, ERROR_TASK_NOT_FOUND, "Task not found")

    task = _TASKS[task_id]["task"]
    history_length = params.get("historyLength") if isinstance(params, dict) else None
    task_response = _apply_history_length(
        task, history_length if isinstance(history_length, int) else None
    )
    return _jsonrpc_result(req_id, task_response)


def _handle_tasks_cancel(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handles the tasks/cancel JSON-RPC method.

    Attempts to cancel a task. Only non-terminal tasks can be canceled.

    Args:
        req_id: The JSON-RPC request ID.
        params: The method parameters containing the task ID.

    Returns:
        JSON-RPC response with the canceled task or error.
    """
    task_id = params.get("id") if isinstance(params, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return _jsonrpc_error(req_id, ERROR_INVALID_PARAMS, "Invalid method parameters")
    if task_id not in _TASKS:
        return _jsonrpc_error(req_id, ERROR_TASK_NOT_FOUND, "Task not found")

    task = _TASKS[task_id]["task"]
    state = task.get("status", {}).get("state")
    if state in TERMINAL_STATES:
        return _jsonrpc_error(req_id, ERROR_TASK_NOT_CANCELABLE, "Task cannot be canceled")

    task["status"] = {"state": "canceled", "timestamp": _now_iso()}
    return _jsonrpc_result(req_id, task)


# ==============================================================================
# FastAPI Endpoints
# ==============================================================================

@router.post("")
@router.post("/")
async def a2a_endpoint(request: Request, _: str = Depends(get_api_key)) -> JSONResponse:
    """Main A2A JSON-RPC endpoint.

    Routes JSON-RPC requests to the appropriate handler based on the method.

    Supported methods:
    - message/send: Send a message to the agent
    - tasks/get: Retrieve a task by ID
    - tasks/cancel: Cancel a running task
    """
    try:
        payload = await request.json()
    except Exception:
        error = _jsonrpc_error(None, ERROR_PARSE, "Invalid JSON payload")
        return JSONResponse(error, status_code=400)

    validation_error = _validate_jsonrpc_request(payload)
    if validation_error:
        return JSONResponse(validation_error, status_code=400)

    req_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "message/send":
        response = await _handle_message_send(req_id, params)
    elif method == "tasks/get":
        response = _handle_tasks_get(req_id, params)
    elif method == "tasks/cancel":
        response = _handle_tasks_cancel(req_id, params)
    elif method.startswith("tasks/pushNotificationConfig/"):
        response = _jsonrpc_error(
            req_id,
            ERROR_PUSH_NOT_SUPPORTED,
            "Push Notification is not supported",
        )
    else:
        response = _jsonrpc_error(req_id, ERROR_METHOD_NOT_FOUND, "Method not found")

    return JSONResponse(response, status_code=200)


# ==============================================================================
# Agent Card
# ==============================================================================

def _agent_card_base_url() -> str:
    """Gets the base URL for the agent card from environment or default."""
    base_url = os.getenv("A2A_BASE_URL") or os.getenv("API_BASE_URL") or "http://localhost:8000"
    return base_url.rstrip("/")


def _agent_card_endpoint_url() -> str:
    """Gets the full endpoint URL for the JSON-RPC interface."""
    endpoint = os.getenv("A2A_ENDPOINT_URL")
    if endpoint:
        return endpoint.rstrip("/")
    base_url = _agent_card_base_url()
    return f"{base_url}/api/a2a"


def _agent_card() -> Dict[str, Any]:
    """Builds the A2A Agent Card describing this agent's capabilities.

    The Agent Card is a standardized way to describe an agent's:
    - Identity (name, description, version)
    - Capabilities (streaming, push notifications)
    - Skills (what the agent can do)
    - Input/output modes supported

    Returns:
        Agent Card dictionary conforming to A2A protocol.
    """
    endpoint_url = _agent_card_endpoint_url()
    provider_org = os.getenv("A2A_PROVIDER_ORG")
    provider_url = os.getenv("A2A_PROVIDER_URL")
    provider = None
    if provider_org and provider_url:
        provider = {"organization": provider_org, "url": provider_url}

    card: Dict[str, Any] = {
        "protocolVersion": "0.3.0",
        "name": os.getenv("A2A_AGENT_NAME", "Eligibility A2A Agent"),
        "description": os.getenv(
            "A2A_AGENT_DESCRIPTION",
            "Eligibility assistant for invoice rules and customer log analysis.",
        ),
        "url": endpoint_url,
        "preferredTransport": "JSONRPC",
        "additionalInterfaces": [
            {"url": endpoint_url, "transport": "JSONRPC"},
        ],
        "version": os.getenv("A2A_AGENT_VERSION", "0.1.0"),
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "skills": [
            {
                "id": "eligibility-analysis",
                "name": "Eligibility Analysis",
                "description": "Explains eligibility rules and analyzes invoice outcomes.",
                "tags": ["eligibility", "invoices", "rules", "analytics"],
                "examples": [
                    "Why was invoice INV-100 rejected?",
                    "What are the most rejected debtors?",
                    "Which rules fail most often for seller S-123?",
                ],
            }
        ],
    }
    if provider:
        card["provider"] = provider
    docs_url = os.getenv("A2A_AGENT_DOCS_URL")
    if docs_url:
        card["documentationUrl"] = docs_url
    return card


@router.get("/.well-known/agent-card.json")
async def agent_card() -> Dict[str, Any]:
    """Returns the Agent Card for this A2A agent.

    The Agent Card is served at the well-known URL as per A2A protocol,
    allowing clients to discover the agent's capabilities.
    """
    return _agent_card()
