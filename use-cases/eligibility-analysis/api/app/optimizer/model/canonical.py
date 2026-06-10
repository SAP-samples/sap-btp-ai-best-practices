"""
Canonical schema contracts and source-to-canonical mappings for optimizer data.

The optimizer consumes multiple source files with different schemas:
  - Offer file (candidate invoices to optimize)
  - Extraction/event file (lifecycle and release history)

This module defines:
  - Canonical field names
  - Source-specific column mappings
  - Typed contracts used across optimizer components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Source Profiles
# ---------------------------------------------------------------------------

SOURCE_PROFILE_OFFER = "offer_file"
SOURCE_PROFILE_EXTRACTION = "extraction_file"
SOURCE_PROFILE_HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Canonical Fields
# ---------------------------------------------------------------------------

CANONICAL_DATE_FIELDS = (
    "offer_file_date",
    "summary_file_date",
    "credit_start",
    "credit_release",
    "due_date",
    "issuance_date",
    "dispatch_date",
    "clearing_date",
    "repurchase_date",
)

CANONICAL_NUMERIC_FIELDS = (
    "candidate_amount",
    "invoice_amount",
    "purchase_price",
    "exchange_rate",
    "margin",
    "repurchase_amount",
)

CANONICAL_STRING_FIELDS = (
    "invoice_reference",
    "document_number",
    "debtor_id",
    "seller_id_external",
    "company_code",
    "program_id",
    "status",
    "reason",
    "original_currency",
    "funding_currency",
    "clearing_document",
)


# ---------------------------------------------------------------------------
# Source Column Maps
# ---------------------------------------------------------------------------

OFFER_TO_CANONICAL: Dict[str, str] = {
    "PROGRAMA": "program_id",
    "ID SELLER": "seller_id_external",
    "ID DEBTOR": "debtor_id",
    "INVOICE REF": "invoice_reference",
    "TOTAL NET VALUE (ORIGINAL CCY)": "candidate_amount",
    "TOTAL INVOICE AMOUNT (ORIGINAL CCY)": "invoice_amount",
    "ORIGINAL CURRENCY": "original_currency",
    "FUNDING CURRENCY": "funding_currency",
    "EXCHANGE RATE": "exchange_rate",
    "ISSUANCE DATE": "issuance_date",
    "DESPATCH DATE": "dispatch_date",
    "DUE DATE": "due_date",
    "MARGIN": "margin",
}

EXTRACTION_TO_CANONICAL: Dict[str, str] = {
    "Offer File Date (UTC)": "offer_file_date",
    "Summary File Date (UTC)": "summary_file_date",
    "Reconciliation File Date (UTC)": "credit_release",
    "Due Date": "due_date",
    "Issuance date": "issuance_date",
    "Dispatch Date": "dispatch_date",
    "Invoice Reference": "invoice_reference",
    "Document Number": "document_number",
    "Customer": "debtor_id",
    "Company Code": "company_code",
    "Status": "status",
    "Reason": "reason",
    "Purchase Price": "purchase_price",
    "Amount": "invoice_amount",
    "Currency": "original_currency",
    "Funding Currency": "funding_currency",
    "Exchange Rate": "exchange_rate",
    "Reconciliation Date": "reconciliation_date",
    "Clearing Date": "clearing_date",
    "Clearing Document": "clearing_document",
    "Repurchase Date": "repurchase_date",
    "Repurchase": "repurchase_amount",
}


REQUIRED_OFFER_COLUMNS: Sequence[str] = (
    "INVOICE REF",
    "ID DEBTOR",
    "ID SELLER",
    "TOTAL NET VALUE (ORIGINAL CCY)",
    "FUNDING CURRENCY",
    "ISSUANCE DATE",
    "DUE DATE",
)

REQUIRED_EXTRACTION_COLUMNS: Sequence[str] = (
    "Offer File Date (UTC)",
    "Summary File Date (UTC)",
    "Reconciliation File Date (UTC)",
    "Invoice Reference",
    "Customer",
    "Company Code",
    "Purchase Price",
    "Status",
)


# ---------------------------------------------------------------------------
# Canonical Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalInvoice:
    invoice_reference: str
    debtor_id: str
    candidate_amount: float
    due_date: pd.Timestamp | None
    funding_currency: str | None = None
    invoice_amount: float | None = None
    seller_id_external: str | None = None
    company_code: str | None = None
    program_id: str | None = None


@dataclass(frozen=True)
class CanonicalLifecycleEvent:
    invoice_reference: str
    credit_start: pd.Timestamp | None
    credit_release: pd.Timestamp | None
    status: str | None = None
    clearing_date: pd.Timestamp | None = None
    clearing_document: str | None = None
    repurchase_date: pd.Timestamp | None = None


@dataclass(frozen=True)
class CanonicalLimit:
    entity_type: str
    entity_id: str
    limit_amount: float


@dataclass(frozen=True)
class WeeklyExposurePoint:
    week_start: pd.Timestamp
    entity_type: str
    entity_id: str
    exposure_amount: float


@dataclass(frozen=True)
class CanonicalLoadReport:
    source_profile: str
    raw_rows: int
    loaded_rows: int
    missing_required_columns: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_profile": self.source_profile,
            "raw_rows": self.raw_rows,
            "loaded_rows": self.loaded_rows,
            "missing_required_columns": list(self.missing_required_columns),
        }


def detect_source_profile(columns: Iterable[str]) -> str:
    """Detect source profile based on available column names."""
    column_set = {str(col).strip() for col in columns}

    offer_hits = sum(col in column_set for col in REQUIRED_OFFER_COLUMNS)
    extraction_hits = sum(col in column_set for col in REQUIRED_EXTRACTION_COLUMNS)

    if offer_hits >= max(3, len(REQUIRED_OFFER_COLUMNS) // 2) and extraction_hits == 0:
        return SOURCE_PROFILE_OFFER
    if extraction_hits >= max(4, len(REQUIRED_EXTRACTION_COLUMNS) // 2) and offer_hits == 0:
        return SOURCE_PROFILE_EXTRACTION
    if offer_hits > 0 and extraction_hits > 0:
        return SOURCE_PROFILE_HYBRID
    return SOURCE_PROFILE_EXTRACTION


def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common extraction variants before canonical mapping."""
    out = df.copy()
    rename_map: Dict[str, str] = {}

    for col in out.columns:
        clean = str(col).strip()
        if clean != col:
            rename_map[col] = clean

    if rename_map:
        out = out.rename(columns=rename_map)

    if "Customer " in out.columns and "Customer" not in out.columns:
        out = out.rename(columns={"Customer ": "Customer"})

    return out


def to_canonical(
    df: pd.DataFrame,
    mapping: Mapping[str, str],
    source_profile: str,
) -> tuple[pd.DataFrame, CanonicalLoadReport]:
    """Map a source DataFrame into canonical fields and normalize types."""
    normalized = normalize_source_columns(df)
    raw_rows = len(normalized)

    reverse: Dict[str, str] = {src: dst for src, dst in mapping.items() if src in normalized.columns}

    canonical = pd.DataFrame(index=normalized.index)
    for src_col, dst_col in reverse.items():
        canonical[dst_col] = normalized[src_col]

    # Keep raw source columns for downstream compatibility/auditing.
    for col in normalized.columns:
        if col not in canonical.columns:
            canonical[col] = normalized[col]

    for col in CANONICAL_DATE_FIELDS:
        if col in canonical.columns:
            canonical[col] = pd.to_datetime(canonical[col], errors="coerce")

    for col in CANONICAL_NUMERIC_FIELDS:
        if col in canonical.columns:
            canonical[col] = pd.to_numeric(canonical[col], errors="coerce")

    for col in CANONICAL_STRING_FIELDS:
        if col in canonical.columns:
            series = canonical[col]
            canonical[col] = series.where(series.isna(), series.astype(str).str.strip())
            canonical[col] = canonical[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Canonical lifecycle default: credit starts at summary file date.
    if "credit_start" not in canonical.columns and "summary_file_date" in canonical.columns:
        canonical["credit_start"] = canonical["summary_file_date"]

    required = REQUIRED_OFFER_COLUMNS if source_profile == SOURCE_PROFILE_OFFER else REQUIRED_EXTRACTION_COLUMNS
    missing = tuple(col for col in required if col not in normalized.columns)

    canonical = canonical.reset_index(drop=True)
    canonical["optimizer_row_id"] = canonical.index.astype(str)
    canonical["source_profile"] = source_profile

    report = CanonicalLoadReport(
        source_profile=source_profile,
        raw_rows=raw_rows,
        loaded_rows=len(canonical),
        missing_required_columns=missing,
    )
    return canonical, report


def week_start(date_series: pd.Series) -> pd.Series:
    """Normalize timestamps to Monday week start (00:00)."""
    dates = pd.to_datetime(date_series, errors="coerce")
    return dates.dt.to_period("W-MON").dt.start_time


def expected_canonical_columns() -> List[str]:
    """Return canonical field names expected by the optimizer pipeline."""
    return sorted(set(CANONICAL_DATE_FIELDS + CANONICAL_NUMERIC_FIELDS + CANONICAL_STRING_FIELDS))
