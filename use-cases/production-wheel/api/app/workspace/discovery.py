"""Shared, validated metadata and inclusive date filters for workspace discovery."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

_ALLOWED = {
    "datasets": {"plant", "name", "status", "created_from", "created_to"},
    "runs": {"dataset_id", "status", "parent_run_id", "created_from", "created_to"},
}


def _timestamp(value: str | datetime) -> datetime:
    """Parse an ISO datetime as UTC, interpreting an omitted timezone as UTC."""
    if isinstance(value, str):
        if "T" not in value and " " not in value:
            raise ValueError(
                "Date filters require ISO datetimes, for example 2026-09-07T00:00:00Z"
            )
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Date filters require valid ISO datetimes") from exc
    if not isinstance(value, datetime):
        raise ValueError("Date filters require ISO datetime strings")  # noqa: TRY004 - HTTP domain contract uses ValueError.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def filter_records(
    items: list[dict], filters: dict | None, kind: Literal["datasets", "runs"]
) -> list[dict]:
    """Filter discovery records by supported metadata and inclusive creation dates.

    Args:
        items: Repository records, preserved in their original discovery order.
        filters: Supported exact-match metadata, name substring, and ISO bounds.
        kind: Dataset or run vocabulary; unsupported keys are always rejected.

    Returns:
        Matching records. Unknown IDs match nothing. Missing/malformed creation
        timestamps cannot match bounded queries; unbounded discovery retains them.
    """
    criteria = dict(filters or {})
    unknown = set(criteria) - _ALLOWED[kind]
    if unknown:
        raise ValueError(f"Unsupported {kind} discovery filters: {sorted(unknown)}")
    lower = (
        _timestamp(criteria.pop("created_from"))
        if criteria.get("created_from") is not None
        else None
    )
    upper = (
        _timestamp(criteria.pop("created_to"))
        if criteria.get("created_to") is not None
        else None
    )
    criteria.pop("created_from", None)
    criteria.pop("created_to", None)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("created_from must not exceed created_to")
    for key, value in criteria.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} discovery filter must be a string")
    result = []
    for item in items:
        if lower is not None or upper is not None:
            try:
                created = _timestamp(item.get("created_at"))
            except ValueError:
                continue
            if (lower is not None and created < lower) or (
                upper is not None and created > upper
            ):
                continue
        matches = True
        for key, value in criteria.items():
            if value is None:
                continue
            if key == "plant":
                matches = (item.get("metadata") or {}).get("plant") == value
            elif key == "name":
                matches = value.casefold() in str(item.get("name") or "").casefold()
            else:
                matches = item.get(key) == value
            if not matches:
                break
        if matches:
            result.append(item)
    return result
