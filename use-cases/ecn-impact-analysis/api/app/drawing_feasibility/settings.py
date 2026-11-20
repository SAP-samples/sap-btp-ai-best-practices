"""Validated settings for the always-on drawing-feasibility workflow."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, PositiveInt, StringConstraints


ENVIRONMENT_FIELDS: dict[str, str] = {
    "DRAWING_MAX_UPLOAD_BYTES": "max_upload_bytes",
}
"""Drawing-specific, non-secret environment-variable names.

Model selection and LLM runtime limits deliberately come from the primary ECN
LLM configuration.  This prevents a second deployment choice from drifting
away from the model used by the rest of the application.
"""


class DrawingFeasibilitySettings(BaseModel):
    """Hold immutable drawing upload limits and inherited LLM runtime settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    max_upload_bytes: PositiveInt = 20 * 1024 * 1024
    request_timeout_seconds: PositiveInt
    max_concurrency: PositiveInt


def load_drawing_feasibility_settings(
    environ: Mapping[str, str] | None = None,
    *,
    model_name: str,
    request_timeout_seconds: int,
    max_concurrency: int,
) -> DrawingFeasibilitySettings:
    """Load drawing-specific limits and inherit primary LLM runtime settings.

    Args:
        environ: Explicit values for deterministic tests; process environment
            is used when omitted. Empty drawing-specific values retain defaults.
        model_name: Deployment name selected by the primary ECN LLM settings.
        request_timeout_seconds: Primary ECN LLM request timeout.
        max_concurrency: Primary ECN LLM concurrency limit.

    Returns:
        Immutable settings for the standalone workflow.
    """
    source = os.environ if environ is None else environ
    values = {
        field: source[name]
        for name, field in ENVIRONMENT_FIELDS.items()
        if source.get(name, "") != ""
    }
    values.update(
        model_name=model_name,
        request_timeout_seconds=request_timeout_seconds,
        max_concurrency=max_concurrency,
    )
    return DrawingFeasibilitySettings.model_validate(values)
