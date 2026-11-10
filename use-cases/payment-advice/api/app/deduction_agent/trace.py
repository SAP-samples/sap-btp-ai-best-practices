"""
Tool-call trace extraction for LangGraph message lists.

Walks a list of LangGraph messages and extracts structured records for every
AI tool call, matching each call to its corresponding ToolMessage result.

Usage example:
    from app.deduction_agent.trace import extract_tool_calls

    records = extract_tool_calls(state["messages"])
    # [{"tool": "get_deduction_rules", "args": {...}, "result_summary": "null"}]
"""

from __future__ import annotations

_SUMMARY_MAX = 20000  # maximum characters kept from a tool-result content string


def _get_attr(msg, attr: str, default=None):
    """
    Read an attribute from either an object-style message (getattr) or a
    dict-style message (msg.get).  Returns *default* if the attribute is
    absent on either form.
    """
    if isinstance(msg, dict):
        return msg.get(attr, default)
    return getattr(msg, attr, default)


def _stringify(value) -> str:
    """
    Convert a content value to a plain string.

    LangChain content can be a str, a list of content blocks, or None.
    We cast everything to str so the summary is always a plain string.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # lists of dicts / content-block objects — take their repr as a fallback
    return str(value)


def extract_tool_calls(messages: list) -> list[dict]:
    """
    Extract a flat list of tool-call records from a LangGraph message sequence.

    For every AI message that carries one or more tool_calls, one record is
    emitted per call:

        {
            "tool":           str,        # tool / function name
            "args":           dict,       # call arguments (may be {})
            "result_summary": str | None  # matching ToolMessage content,
                                          # truncated to _SUMMARY_MAX chars, or None
        }

    Results are matched to calls by ``tool_call_id`` (primary path).  When a
    call has no id, the next unused result message is consumed in order
    (best-effort fallback).

    The function is intentionally defensive:
    - Supports both object-style messages (LangChain dataclass / BaseMessage)
      and plain dict messages.
    - Individual tool_call entries may omit "id" without breaking extraction.
    - An empty or missing tool_calls list on an AI message is silently skipped.

    Parameters
    ----------
    messages:
        Sequence of LangGraph / LangChain messages (object or dict form).

    Returns
    -------
    list[dict]
        Ordered list of tool-call records.  Empty list when no AI message
        has tool_calls.
    """
    # ------------------------------------------------------------------
    # Pass 1: build an index of tool-result messages keyed by tool_call_id.
    # Also collect result messages that have NO id into an ordered list so
    # we can do positional matching as a fallback.
    # ------------------------------------------------------------------
    results_by_id: dict[str, str] = {}   # tool_call_id -> truncated summary

    for msg in messages:
        tcid = _get_attr(msg, "tool_call_id")
        if tcid is not None:
            # This is a ToolMessage / tool result
            raw = _get_attr(msg, "content", "")
            summary = _stringify(raw)[:_SUMMARY_MAX]
            results_by_id[tcid] = summary

    # Collect result messages that carry no tool_call_id in message order so
    # we can do positional matching as a best-effort fallback.
    # An unkeyed result is a message that has no tool_calls AND no tool_call_id
    # but does have a content field (heuristic: it looks like a result).
    unkeyed_results = [
        _stringify(_get_attr(msg, "content", ""))[:_SUMMARY_MAX]
        for msg in messages
        if _get_attr(msg, "tool_call_id") is None
        and not _get_attr(msg, "tool_calls")
        and _get_attr(msg, "content") is not None
    ]
    _unkeyed_idx = 0  # pointer into unkeyed_results for positional fallback

    # ------------------------------------------------------------------
    # Pass 2: walk AI messages and emit one record per tool call.
    # ------------------------------------------------------------------
    records: list[dict] = []

    for msg in messages:
        tool_calls = _get_attr(msg, "tool_calls")
        if not tool_calls:
            continue  # not an AI message with tool calls

        for call in tool_calls:
            # LangChain tool_call entries are plain dicts with keys
            # "name", "args", "id" (id may be absent).
            if isinstance(call, dict):
                name = call.get("name", "")
                args = call.get("args", {})
                call_id = call.get("id")
            else:
                # Defensive: handle object-style call entries (unusual but possible)
                name = getattr(call, "name", "")
                args = getattr(call, "args", {})
                call_id = getattr(call, "id", None)

            # Resolve result summary
            if call_id is not None and call_id in results_by_id:
                result_summary = results_by_id[call_id] or None
            else:
                # Positional fallback: consume next unkeyed result
                if _unkeyed_idx < len(unkeyed_results):
                    result_summary = unkeyed_results[_unkeyed_idx] or None
                    _unkeyed_idx += 1
                else:
                    result_summary = None

            records.append({
                "tool": name,
                "args": args,
                "result_summary": result_summary,
            })

    return records
