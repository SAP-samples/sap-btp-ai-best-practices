"""Utility helpers for safe tool-result previews and logging."""
from __future__ import annotations

import json
from typing import Any, Optional


def stringify_tool_content(content: Any) -> str:
    """Convert LangGraph tool message content into a printable string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "\n".join(text_parts)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def estimate_row_count(content: Any) -> Optional[int]:
    """Best-effort estimate of row count in a tool payload."""
    if isinstance(content, list):
        return len(content)
    if isinstance(content, dict):
        pagination = content.get("pagination")
        if isinstance(pagination, dict) and "returned" in pagination:
            try:
                return int(pagination["returned"])
            except Exception:
                pass
        for key in (
            "rows",
            "invoices",
            "entries",
            "processes",
            "matches",
            "groups",
            "debtors",
            "sellers",
            "data",
            "explanations",
        ):
            value = content.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def build_tool_result_preview(content: Any, max_chars: int) -> dict[str, Any]:
    """Create a bounded preview plus size/count metadata for tool results."""
    raw_text = stringify_tool_content(content)
    payload_bytes = len(raw_text.encode("utf-8", errors="ignore"))
    is_truncated = len(raw_text) > max_chars
    preview = raw_text[:max_chars] if is_truncated else raw_text
    if is_truncated:
        preview += "... [truncated]"
    return {
        "content_preview": preview,
        "content_truncated": is_truncated,
        "content_char_length": len(raw_text),
        "payload_bytes_estimate": payload_bytes,
        "row_count_estimate": estimate_row_count(content),
    }
