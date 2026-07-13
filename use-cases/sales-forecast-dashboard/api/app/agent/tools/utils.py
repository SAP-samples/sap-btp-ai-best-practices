"""
Utility helpers for agent tools.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np


def sanitize_for_json(value: Any) -> Any:
    """
    Recursively normalize tool outputs so they can be serialized for LLM requests.

    - Converts NaN/inf to None
    - Converts numpy scalars/arrays to native Python types
    - Converts datetime objects to ISO-8601 strings
    """
    # Normalize missing or invalid numeric values early.
    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, np.ndarray):
        return [sanitize_for_json(v) for v in value.tolist()]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    return value
