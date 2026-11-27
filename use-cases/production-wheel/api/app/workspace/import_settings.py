"""Extraction settings for a workbook the planner is about to import.

On the Datasets page the planner chooses a production workbook before anything is
extracted. The page previews it (``POST /api/datasets/preview``) and sends that
preview with each chat turn as ``pending_import``. The Workspace Assistant reads it
and proposes extraction setting overrides; the planner applies them on a chat card
and the upload sends them in its unchanged ``settings`` form field.

Nothing here touches HANA: the pending import lives in the process-local page
session context, like the selected dataset or run IDs.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from production_wheel.extraction.datasets import DEFAULT_SETTINGS, _settings
from production_wheel.extraction.frequencies import ALL_AVAILABLE, is_all_available


class FrequencyCount(BaseModel):
    """One recognized weekly code of the workbook and its FINI row count."""

    model_config = ConfigDict(extra="forbid")
    code: str = Field(max_length=16)
    label: str | None = Field(default=None, max_length=200)
    rows: int = Field(ge=0)


class UnrecognizedFrequency(BaseModel):
    """One source frequency value without a weekly code (for example "999...")."""

    model_config = ConfigDict(extra="forbid")
    value: str = Field(max_length=200)
    rows: int = Field(ge=0)


class FrequencyInventory(BaseModel):
    """Frequency counts read from the workbook's FINI table by the preview."""

    model_config = ConfigDict(extra="forbid")
    codes: list[FrequencyCount] = Field(default_factory=list, max_length=100)
    unrecognized: list[UnrecognizedFrequency] = Field(default_factory=list, max_length=100)
    total_rows: int = Field(default=0, ge=0)


class PendingImport(BaseModel):
    """Workbook chosen on the Datasets page but not yet extracted.

    ``preview_id`` changes every time the planner picks a file, so a proposal made
    for an earlier file can be recognized as stale. ``overrides`` are the settings
    already applied in the page (empty means all defaults).
    """

    model_config = ConfigDict(extra="forbid")
    preview_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    primary_filename: str = Field(max_length=255)
    enrichment_filename: str | None = Field(default=None, max_length=255)
    plant: str | None = Field(default=None, max_length=64)
    frequency_inventory: FrequencyInventory
    overrides: dict = Field(default_factory=dict)


# Plain-language meaning of each setting, returned to the assistant with the context.
SETTING_HELP = {
    "modeled_frequencies": (
        "Weekly production frequency codes whose FINIs are modeled; other FINIs are "
        f"out of scope. Use the keyword '{ALL_AVAILABLE}' for every code in the workbook."
    ),
    "demand_days": (
        "Days that convert total demand to daily demand. When not overridden, the import "
        "uses the source horizon found in the workbook if there is one, otherwise 250."
    ),
    "productive_weeks": (
        "Productive weeks per year. When not overridden and the workbook gives a source "
        "horizon, the import uses that horizon divided by five demand days per week."
    ),
    "canonical_factor": "Share of the nominal production-version lot size used as the effective batch (0 to 1].",
}


def pending_import_summary(pending: dict | None) -> dict:
    """Describe the pending import and the settings vocabulary for the assistant.

    Args:
        pending: ``pending_import`` from the page session context, or None.
    Returns:
        Dict with ``pending_import`` (or None plus a ``note`` telling the assistant
        to ask for a workbook), ``defaults`` and ``setting_help``.
    """
    result = {"pending_import": pending, "defaults": DEFAULT_SETTINGS, "setting_help": SETTING_HELP}
    if not pending:
        result["note"] = (
            "No production workbook is chosen on the Datasets page. Ask the user to choose "
            "the production workbook first; its frequency codes are then read automatically."
        )
    return result


def build_settings_proposal(
    pending: dict | None,
    summary: str,
    changes: dict,
    reset_to_default: list[str] | None = None,
) -> dict:
    """Validate an override change against the pending workbook and build a proposal card.

    The change is a patch: keys in ``changes`` replace the current overrides, keys in
    ``reset_to_default`` are removed, and every other current override is kept.

    Args:
        pending: ``pending_import`` from the page session context.
        summary: Plain-language description of what the planner approves.
        changes: Subset of ``DEFAULT_SETTINGS`` keys with new values.
        reset_to_default: Keys whose override is removed.
    Returns:
        ``extraction_settings_proposal`` event with the full resulting ``overrides``
        (ready for the upload form), ``modeled_codes`` expanded for display and
        planner-facing ``warnings``.
    Raises:
        ValueError: No pending workbook, empty summary, unknown keys or invalid values.
    """
    if not pending:
        raise ValueError("Choose the production workbook on the Datasets page before proposing settings")
    if not summary or not summary.strip():
        raise ValueError("Provide a plain-language summary of the proposed settings for the user")
    reset = set(reset_to_default or [])
    unknown = (set(changes) | reset) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown extraction settings: {sorted(unknown)}")
    overrides = {key: value for key, value in (pending.get("overrides") or {}).items() if key not in reset}
    overrides.update(changes)
    # The same validator as the upload: rejects anything the import would refuse.
    resolved, _ = _settings(overrides)
    overrides = {key: resolved[key] for key in overrides}

    inventory = pending["frequency_inventory"]
    workbook_codes = [item["code"] for item in inventory.get("codes", [])]
    modeled = resolved["modeled_frequencies"]
    modeled_codes = list(workbook_codes) if modeled == ALL_AVAILABLE else list(modeled)
    warnings = []
    missing = [code for code in modeled_codes if code not in workbook_codes]
    if missing:
        warnings.append(f"No FINI in this workbook uses {', '.join(missing)}.")
    undetermined = sum(item["rows"] for item in inventory.get("unrecognized", []))
    if undetermined:
        warnings.append(
            f"{undetermined} FINI rows have no weekly frequency code "
            f"({', '.join(item['value'] for item in inventory['unrecognized'])}) and stay out of scope."
        )
    rows_by_code = {item["code"]: item["rows"] for item in inventory.get("codes", [])}
    return {
        "type": "extraction_settings_proposal",
        "proposal_id": uuid.uuid4().hex,
        "preview_id": pending["preview_id"],
        "primary_filename": pending["primary_filename"],
        "summary": summary.strip(),
        "overrides": overrides,
        "modeled_codes": modeled_codes,
        "modeled_rows": sum(rows_by_code.get(code, 0) for code in modeled_codes),
        "total_rows": inventory.get("total_rows", 0),
        "all_available": is_all_available(modeled),
        "warnings": warnings,
    }
