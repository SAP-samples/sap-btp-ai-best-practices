"""Runtime settings and reproducibility configuration for the ECN impact pipeline.

Settings are loaded once at startup from a documented set of environment
variables and validated by Pydantic.  The model is frozen so that values
cannot be accidentally mutated after loading.

``canonical_model_config`` emits a deterministic, sorted, compact JSON string
containing only the four keys that must be pinned in stored analysis records for
reproducibility: ``max_prompt_characters``, ``max_prompt_evidence_items``,
``request_timeout_seconds``, and ``strict_structured_output``.

Typical usage::

    from app.ecn.settings import load_ecn_settings, canonical_model_config

    settings = load_ecn_settings()           # reads os.environ
    model_cfg_json = canonical_model_config(settings)
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    field_validator,
)


DEFAULT_LLM_MODEL = "gpt-5.4"
"""Default SAP AI Core text-generation deployment for new ECN jobs."""

LLM_TIMEOUT_FALLBACK_MODEL = "gpt-5.4"
"""Single fallback deployment used when the configured model times out."""

# ---------------------------------------------------------------------------
# Environment-variable name → field-name mapping (single source of truth)
# ---------------------------------------------------------------------------

ENVIRONMENT_FIELDS: dict[str, str] = {
    "MBOM_CHANGE_NUMBER_PATTERN": "mbom_change_number_pattern",
    "LLM_MAX_CONCURRENCY": "llm_max_concurrency",
    "LLM_REQUEST_TIMEOUT_SECONDS": "llm_request_timeout_seconds",
    "MAX_JSON_UPLOAD_BYTES": "max_json_upload_bytes",
    "MAX_WORKBOOK_UPLOAD_BYTES": "max_workbook_upload_bytes",
    "MAX_WORKBOOK_ROWS": "max_workbook_rows",
    "MAX_WORKBOOK_CELLS": "max_workbook_cells",
    "MAX_PROMPT_EVIDENCE_ITEMS": "max_prompt_evidence_items",
    "MAX_PROMPT_CHARACTERS": "max_prompt_characters",
    "LLM_MODEL": "llm_model_name",
}
"""Map every supported environment variable name to its ``EcnSettings`` field name.

This mapping is the only place where environment variable names are defined;
it is consumed by ``load_ecn_settings`` and may be used by deployment tooling
to verify that required variables are present.
"""


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class EcnSettings(BaseModel):
    """Validated, immutable runtime settings for the ECN impact pipeline.

    All fields are validated by Pydantic on construction.  The model is frozen
    so that the configuration cannot be mutated after startup.  Extra keys are
    forbidden to catch typos in callers.

    Args:
        llm_max_concurrency: Maximum number of concurrent LLM requests.
            Defaults to ``4``.
        llm_request_timeout_seconds: Per-request LLM timeout in seconds.
            Defaults to ``180``.
        max_json_upload_bytes: Maximum accepted byte size for an ECN JSON
            upload.  Defaults to 25 MiB.
        max_workbook_upload_bytes: Maximum accepted byte size for a BOM
            workbook upload.  Defaults to 100 MiB.
        max_workbook_rows: Maximum number of data rows allowed in a BOM
            workbook.  Defaults to 100 000.
        max_workbook_cells: Maximum number of non-empty cells allowed in a
            BOM workbook.  Defaults to 3 000 000.
        max_prompt_evidence_items: Maximum number of evidence items included
            in the final LLM evidence projection. Defaults to ``100``.
        max_prompt_characters: Maximum character count of a single LLM
            prompt. Must be >= 4096. Defaults to 120 000 for the final pass;
            the planner retains its separate 40 000-character hard limit.
        llm_model_name: The Gen AI Hub model deployment name to use for
            analysis. Defaults to ``DEFAULT_LLM_MODEL``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mbom_change_number_pattern: str = r"^[A-Z]{4}(?P<digits>\d{6})[0-9A-Z]{2}$"

    @field_validator("mbom_change_number_pattern")
    @classmethod
    def validate_change_pattern(cls, value: str) -> str:
        """Validate the configured MBOM regex and require a named digits group."""
        try:
            pattern = re.compile(value)
        except re.error as exc:
            raise ValueError("Invalid MBOM change number pattern") from exc
        if "digits" not in pattern.groupindex:
            raise ValueError("MBOM change number pattern needs a named digits group")
        return value

    llm_max_concurrency: PositiveInt = 4
    llm_request_timeout_seconds: PositiveInt = 180
    max_json_upload_bytes: PositiveInt = 128 * 1024 * 1024
    max_workbook_upload_bytes: PositiveInt = 100 * 1024 * 1024
    max_workbook_rows: PositiveInt = 100_000
    max_workbook_cells: PositiveInt = 3_000_000
    max_prompt_evidence_items: PositiveInt = 100
    max_prompt_characters: int = Field(default=120_000, ge=4_096)
    llm_model_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ] = DEFAULT_LLM_MODEL


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_ecn_settings(environ: Mapping[str, str] | None = None) -> EcnSettings:
    """Load and validate ECN settings from an explicit environment mapping.

    Only the variables listed in ``ENVIRONMENT_FIELDS`` are read.  Variables
    present in the source mapping but with an empty string value are silently
    ignored so that deployment tooling may set variables to ``""`` to indicate
    "use the default".

    Args:
        environ: An explicit mapping of environment variable names to values.
            Pass an empty dict ``{}`` to use all defaults.  When ``None``,
            ``os.environ`` is used.

    Returns:
        A validated, frozen ``EcnSettings`` instance.

    Raises:
        pydantic.ValidationError: If any documented variable carries an invalid
            value (e.g. a non-positive integer for ``LLM_MAX_CONCURRENCY``).
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    values: dict[str, str] = {
        field: source[name]
        for name, field in ENVIRONMENT_FIELDS.items()
        if name in source and source[name] != ""
    }
    return EcnSettings.model_validate(values)


# ---------------------------------------------------------------------------
# Canonical model configuration
# ---------------------------------------------------------------------------


def canonical_model_config(settings: EcnSettings) -> str:
    """Return a sorted, compact JSON string that pins the reproducibility keys.

    The returned string contains exactly four keys:

    - ``max_prompt_characters``: matches ``settings.max_prompt_characters``.
    - ``max_prompt_evidence_items``: matches ``settings.max_prompt_evidence_items``.
    - ``request_timeout_seconds``: matches ``settings.llm_request_timeout_seconds``.
    - ``strict_structured_output``: always ``true``.

    The keys are sorted alphabetically and the JSON is compact (no extra
    whitespace) so that the string is byte-identical across runs and safe to
    use as a cache key or to store in a HANA column.

    Args:
        settings: A validated ``EcnSettings`` instance.

    Returns:
        A compact, sorted JSON string containing the four reproducibility keys.
    """
    config: dict[str, object] = {
        "max_prompt_characters": settings.max_prompt_characters,
        "max_prompt_evidence_items": settings.max_prompt_evidence_items,
        "request_timeout_seconds": settings.llm_request_timeout_seconds,
        "strict_structured_output": True,
    }
    return json.dumps(config, sort_keys=True, separators=(",", ":"))
