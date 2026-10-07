"""
Deterministic verification and confidence scoring for a canonical payload (UC-01).

Runs cheap deterministic checks and produces a confidence score plus a
``needs_review`` gate the CLI uses to decide whether to pause. Per the design, this
does NOT check per-line arithmetic (too fragile across real documents); it uses:

- required canonical header fields present (hard issue if missing);
- soft total reconciliation: sum of line ``net_amount`` vs header ``payment_amount``
  within tolerance (warning only, tolerant of partial advices / FX / rounding);
- suspected duplicates: line rows identical on a projection key (warning; the
  design's surgical LLM diagnosis of just these rows can be layered on later).

The confidence score is a simple, transparent weighted deduction, not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import REQUIRED_HEADER_FIELDS

# Fields used to detect suspected duplicate rows (a projection, not the full row).
_DUP_KEY_FIELDS = ("invoice_reference", "net_amount", "deduction_reason", "gross_amount")


@dataclass
class VerifyResult:
    """Outcome of verifying a canonical payload."""

    confidence: float
    needs_review: bool
    issues: list[str] = field(default_factory=list)          # hard problems
    warnings: list[str] = field(default_factory=list)        # soft signals
    suspected_duplicate_groups: list[list[int]] = field(default_factory=list)


def _as_number(value: Any) -> float | None:
    """Return a float if the value is numeric, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _suspected_duplicates(line_items: list[dict[str, Any]]) -> list[list[int]]:
    """Group row indexes that share the same projection key (possible duplicates)."""
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(line_items):
        key = tuple(row.get(name) for name in _DUP_KEY_FIELDS)
        if all(part is None for part in key):
            continue  # nothing distinctive extracted; skip
        groups.setdefault(key, []).append(index)
    return [indexes for indexes in groups.values() if len(indexes) > 1]


def verify_canonical(
    header: dict[str, Any],
    line_items: list[dict[str, Any]],
    *,
    tolerance: float = 0.01,
    confidence_threshold: float = 0.7,
) -> VerifyResult:
    """
    Verify a canonical payload and score confidence.

    Args:
        header: Canonical header mapping.
        line_items: Canonical line items.
        tolerance: Relative tolerance for the soft total reconciliation.
        confidence_threshold: Below this, ``needs_review`` is True.

    Returns:
        A ``VerifyResult`` with confidence, gate, issues, warnings, and suspected
        duplicate groups.
    """
    issues: list[str] = []
    warnings: list[str] = []
    confidence = 1.0

    # Hard: required header fields present and non-empty.
    for required in REQUIRED_HEADER_FIELDS:
        value = header.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(f"missing required header field: {required}")
            confidence -= 0.25

    # Hard-ish: no line items at all is suspicious for an advice.
    if not line_items:
        warnings.append("no line items extracted")
        confidence -= 0.2

    # Soft: total reconciliation (never a hard gate).
    payment_amount = _as_number(header.get("payment_amount"))
    nets = [n for n in (_as_number(row.get("net_amount")) for row in line_items) if n is not None]
    if payment_amount is not None and nets:
        total = sum(nets)
        if abs(total - payment_amount) > tolerance * max(1.0, abs(payment_amount)):
            warnings.append(
                f"line net total {total:.2f} does not reconcile to payment_amount "
                f"{payment_amount:.2f}"
            )
            confidence -= 0.15

    # Soft: suspected duplicate rows.
    duplicate_groups = _suspected_duplicates(line_items)
    if duplicate_groups:
        count = sum(len(group) for group in duplicate_groups)
        warnings.append(f"{count} rows in {len(duplicate_groups)} suspected-duplicate group(s)")
        confidence -= 0.1

    confidence = max(0.0, round(confidence, 3))
    needs_review = confidence < confidence_threshold or bool(issues)
    return VerifyResult(
        confidence=confidence,
        needs_review=needs_review,
        issues=issues,
        warnings=warnings,
        suspected_duplicate_groups=duplicate_groups,
    )
