"""Robust JSON parsing helpers for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


class JsonParseError(ValueError):
    """Raised when model output cannot be parsed as JSON."""


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output with markdown fallbacks."""

    text = (raw or "").strip()
    if not text:
        raise JsonParseError("Empty model response")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    for pattern in (
        r"```json\s*([\s\S]+?)\s*```",
        r"```\s*([\s\S]+?)\s*```",
        r"(\{[\s\S]+\})",
    ):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    raise JsonParseError(f"Could not parse JSON object from response: {text[:300]}")
