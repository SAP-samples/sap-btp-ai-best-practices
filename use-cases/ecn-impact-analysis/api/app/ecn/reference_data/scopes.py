"""Validated source scope and immutable assessment dataset selection."""
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class DatasetContext(BaseModel):
    """Describe declared source coverage without inferring plant from brand or filename."""
    model_config = ConfigDict(extra="forbid")
    model_range: str | None = Field(default=None, max_length=80)
    data_group: str = Field(default="customer", min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    environment: Literal["UNKNOWN", "QA", "PRODUCTION", "SYNTHETIC"] = "UNKNOWN"
    snapshot_date: date | None = None
    coverage_complete: bool = False
    comparison_dataset_id: str | None = None

def select_assessment_datasets(records, data_group):
    """Return immutable current/comparison metadata for one isolated dataset group."""
    ordered = sorted(records, key=lambda r: (r.publication_order, r.dataset_id), reverse=True)
    current, seen = [], set()
    by_id = {r.dataset_id: r for r in ordered}
    for record in ordered:
        if getattr(record, "context", {}).get("data_group", "customer") != data_group:
            continue
        key = scope_key(record)
        if key not in seen:
            seen.add(key)
            current.append(record)
    result = []
    for record in current:
        result.append(dataset_pin(record, False))
        previous_id = getattr(record, "context", {}).get("comparison_dataset_id")
        if previous_id:
            previous = by_id.get(previous_id)
            if previous is None or scope_key(previous) != scope_key(record) or previous.publication_order >= record.publication_order:
                raise ValueError("Comparison must name an older dataset in the same source scope")
            result.append(dataset_pin(previous, True))
    return tuple(result)


def scope_key(record):
    """Return the exact activation boundary including environment and model range."""
    context = getattr(record, "context", {})
    return (str(record.dataset_type), record.scope_code, context.get("model_range"),
            context.get("data_group", "customer"), context.get("environment", "UNKNOWN"))


def dataset_pin(record, comparison):
    """Copy immutable source metadata into the assessment evidence context."""
    return {"dataset_id": record.dataset_id, "dataset_type": str(record.dataset_type),
            "artifact_id": record.artifact_id, "file_name": record.file_name, "sha256": record.sha256,
            **getattr(record, "context", {}), "comparison": comparison}

