"""
Offer file ingestion and canonical mapping.

The Offer file provides candidate invoices for optimization and has a schema
different from the extraction/event file. This loader maps Offer columns to the
optimizer canonical fields without any row-content joins against extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from ..model.canonical import (
    OFFER_TO_CANONICAL,
    SOURCE_PROFILE_OFFER,
    to_canonical,
)


@dataclass(frozen=True)
class OfferLoadReport:
    source_path: str
    sheet_name: str
    raw_rows: int
    loaded_rows: int
    dropped_missing_invoice_ref: int
    dropped_missing_candidate_amount: int
    missing_required_columns: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_path": self.source_path,
            "sheet_name": self.sheet_name,
            "raw_rows": self.raw_rows,
            "loaded_rows": self.loaded_rows,
            "dropped_missing_invoice_ref": self.dropped_missing_invoice_ref,
            "dropped_missing_candidate_amount": self.dropped_missing_candidate_amount,
            "missing_required_columns": list(self.missing_required_columns),
        }


class OfferLoadError(ValueError):
    """Raised when Offer input is invalid for optimizer workflow."""


def load_offer_file(
    input_path: str | Path,
    sheet_name: str = "Sheet1",
) -> tuple[pd.DataFrame, OfferLoadReport]:
    """Load Offer Excel and return canonicalized dataframe plus validation report."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Offer file not found: {source}")

    raw = pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
    canonical, canonical_report = to_canonical(
        raw,
        mapping=OFFER_TO_CANONICAL,
        source_profile=SOURCE_PROFILE_OFFER,
    )

    if canonical_report.missing_required_columns:
        raise OfferLoadError(
            "Missing required Offer columns: "
            + ", ".join(canonical_report.missing_required_columns)
        )

    # Candidate amount drives optimization objective in offer-first flows.
    if "candidate_amount" in canonical.columns:
        canonical["candidate_amount"] = pd.to_numeric(canonical["candidate_amount"], errors="coerce")
    if "invoice_amount" in canonical.columns:
        canonical["invoice_amount"] = pd.to_numeric(canonical["invoice_amount"], errors="coerce")

    dropped_missing_invoice_ref = int(canonical["invoice_reference"].isna().sum())
    dropped_missing_candidate_amount = int(canonical["candidate_amount"].isna().sum())

    canonical = canonical.dropna(subset=["invoice_reference", "candidate_amount"]).copy()
    canonical["candidate_amount"] = canonical["candidate_amount"].astype(float)

    # Compatibility aliases used by existing single-week modules/reporting.
    canonical["Invoice Reference"] = canonical["invoice_reference"]
    canonical["Customer"] = canonical.get("debtor_id")
    canonical["Company Code"] = canonical.get("seller_id_external")
    canonical["Purchase Price"] = canonical["candidate_amount"]
    canonical["Due Date"] = canonical.get("due_date")
    status_series = (
        canonical["status"]
        if "status" in canonical.columns
        else pd.Series([pd.NA] * len(canonical), index=canonical.index)
    )
    canonical["Status"] = status_series.fillna("Offer Candidate")
    canonical["Reason"] = (
        canonical["reason"]
        if "reason" in canonical.columns
        else pd.Series([pd.NA] * len(canonical), index=canonical.index)
    )

    report = OfferLoadReport(
        source_path=str(source),
        sheet_name=sheet_name,
        raw_rows=canonical_report.raw_rows,
        loaded_rows=len(canonical),
        dropped_missing_invoice_ref=dropped_missing_invoice_ref,
        dropped_missing_candidate_amount=dropped_missing_candidate_amount,
        missing_required_columns=canonical_report.missing_required_columns,
    )
    return canonical.reset_index(drop=True), report
