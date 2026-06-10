"""
Data ingestion and normalization for optimizer inputs.

This module reads the BTP extraction Excel file and produces a clean DataFrame
suitable for the rule engine and optimizer. Key responsibilities:

  1. Read the specified sheet from the extraction workbook.
  2. Strip whitespace from column names (the extraction sometimes has trailing spaces).
  3. Normalize the "Customer " column (trailing space variant) to "Customer".
  4. Handle duplicate "Purchase Price" columns (the extraction sometimes exports
     "Purchase Price" and "Purchase Price.1" -- the loader coalesces them).
  5. Parse date columns to pandas Timestamps.
  6. Normalize identifier columns to stripped strings, replacing empty/NaN with pd.NA.
  7. Drop rows with missing Offer File Date or Purchase Price (these cannot be
     used by the cohort filter or the optimizer objective).
  8. Assign a stable ``optimizer_row_id`` for traceability through the pipeline.

The ``LoadReport`` captures statistics about the raw input, normalization, and
which rows were dropped, supporting auditability of the data preparation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from ..model.canonical import (
    EXTRACTION_TO_CANONICAL,
    SOURCE_PROFILE_EXTRACTION,
    to_canonical,
)

# Columns that must exist in the extraction for the optimizer to function.
REQUIRED_COLUMNS = (
    "Offer File Date (UTC)",
    "Summary File Date (UTC)",
    "Reconciliation File Date (UTC)",
    "Paid On (Europe, Madrid)",
    "Repurchase Date",
    "Repurchase",
    "Company Code",
    "Customer",
    "Invoice Reference",
    "Document Number",
    "Due Date",
    "Purchase Price",
    "Status",
    "Reason",
)

# Columns that should be parsed as datetime.
DATE_COLUMNS = (
    "Offer File Date (UTC)",
    "Summary File Date (UTC)",
    "Reconciliation File Date (UTC)",
    "Paid On (Europe, Madrid)",
    "Repurchase Date",
    "Due Date",
    "Reconciliation Date",
)

# Columns that should be normalized as stripped strings.
IDENTIFIER_COLUMNS = (
    "Company Code",
    "Customer",
    "Invoice Reference",
    "Document Number",
    "Status",
    "Reason",
    "Payment Block",
)

SUMMARY_FALLBACK_SHEETS = (
    "Funded Invoices",
    "Funded invoices",
)

SUMMARY_REQUIRED_COLUMNS = (
    "Seller Name",
    "Seller Client ID",
    "Debtor Client ID",
    "Doc Ref",
    "Issue Date",
    "Due Date",
    "Reconciliation Date",
    "Purchase Date",
    "Purchase Amount",
)

SUMMARY_HEADER_SENTINELS = (
    "Seller Name",
    "Debtor Client ID",
    "Doc Ref",
    "Issue Date",
    "Purchase Amount",
)


@dataclass(frozen=True)
class LoadReport:
    """Statistics from the data loading step, used for auditing and diagnostics.

    NOTE: ``dropped_missing_offer_date`` and ``dropped_missing_purchase_price``
    count rows where each respective column is NA. A row missing BOTH columns
    is counted in both fields, so their sum may exceed the actual number of
    dropped rows. The actual drop count is ``raw_rows - loaded_rows``.
    """
    source_path: str
    sheet_name: str
    raw_rows: int                            # Total rows in the Excel sheet
    loaded_rows: int                          # Rows surviving after cleanup
    dropped_missing_offer_date: int           # Count of rows with NA Offer File Date
    dropped_missing_purchase_price: int       # Count of rows with NA Purchase Price
    missing_required_columns: Tuple[str, ...] # Required columns absent from the sheet

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_path": self.source_path,
            "sheet_name": self.sheet_name,
            "raw_rows": self.raw_rows,
            "loaded_rows": self.loaded_rows,
            "dropped_missing_offer_date": self.dropped_missing_offer_date,
            "dropped_missing_purchase_price": self.dropped_missing_purchase_price,
            "missing_required_columns": list(self.missing_required_columns),
        }


class ExtractionLoadError(ValueError):
    """Raised when extraction input is invalid for optimizer workflow."""


def _resolve_sheet_name(source: Path, requested_sheet_name: str) -> str:
    """Resolve sheet name using exact, case-insensitive, and summary fallbacks."""
    workbook = pd.ExcelFile(source, engine="openpyxl")
    available = workbook.sheet_names
    if requested_sheet_name in available:
        return requested_sheet_name

    by_lower = {name.lower(): name for name in available}
    requested_lower = requested_sheet_name.lower()
    if requested_lower in by_lower:
        return by_lower[requested_lower]

    if requested_sheet_name == "SAPUI5 Export":
        for candidate in SUMMARY_FALLBACK_SHEETS:
            match = by_lower.get(candidate.lower())
            if match:
                return match

    raise ExtractionLoadError(
        f"Sheet '{requested_sheet_name}' not found in workbook. "
        f"Available sheets: {', '.join(available)}"
    )


def _strip_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Return rename map with whitespace-trimmed column names."""
    rename_map: Dict[str, str] = {}
    for col in columns:
        clean = str(col).strip()
        if clean != col:
            rename_map[col] = clean
    return rename_map


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in IDENTIFIER_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column]
        normalized = series.where(series.isna(), series.astype(str).str.strip())
        normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        df[column] = normalized
    return df


def _normalize_purchase_price(df: pd.DataFrame) -> pd.DataFrame:
    if "Purchase Price" in df.columns and "Purchase Price.1" in df.columns:
        # Prefer canonical Purchase Price, then fill with fallback duplicate column.
        df["Purchase Price"] = df["Purchase Price"].fillna(df["Purchase Price.1"])

    if "Purchase Price" in df.columns:
        df["Purchase Price"] = pd.to_numeric(df["Purchase Price"], errors="coerce")

    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _looks_like_summary_sheet(df: pd.DataFrame) -> bool:
    columns = {str(col).strip().lower() for col in df.columns}
    return "offer summary" in columns or any(
        str(col).strip().lower().startswith("unnamed:")
        for col in df.columns
    )


def _find_summary_header_row(summary_raw: pd.DataFrame) -> int | None:
    expected = {token.lower() for token in SUMMARY_HEADER_SENTINELS}
    max_rows = min(len(summary_raw), 60)
    for idx in range(max_rows):
        row_values = {
            str(cell).strip().lower()
            for cell in summary_raw.iloc[idx].tolist()
            if pd.notna(cell) and str(cell).strip()
        }
        if expected.issubset(row_values):
            return idx
    return None


def _extract_summary_metadata_value(
    summary_raw: pd.DataFrame,
    label: str,
) -> object | None:
    """Extract metadata value from top summary rows (e.g., 'Funding date:')."""
    target = label.strip().lower().rstrip(":")
    max_rows = min(len(summary_raw), 60)
    max_cols = summary_raw.shape[1]

    for row_idx in range(max_rows):
        row = summary_raw.iloc[row_idx]
        for col_idx in range(max_cols):
            cell = row.iloc[col_idx]
            if pd.isna(cell):
                continue
            token = str(cell).strip().lower().rstrip(":")
            if token != target:
                continue

            # Prefer immediate next cell when available.
            if col_idx + 1 < max_cols:
                next_cell = row.iloc[col_idx + 1]
                if pd.notna(next_cell) and str(next_cell).strip():
                    return next_cell

            # Otherwise use first non-empty cell to the right.
            for probe_idx in range(col_idx + 1, max_cols):
                probe_cell = row.iloc[probe_idx]
                if pd.notna(probe_cell) and str(probe_cell).strip():
                    return probe_cell
            return None

    return None


def _normalize_summary_funded_sheet(source: Path, sheet_name: str) -> tuple[pd.DataFrame, int]:
    """Parse the Summary 'Funded Invoices' layout into extraction-compatible rows."""
    summary_raw = pd.read_excel(source, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_idx = _find_summary_header_row(summary_raw)
    if header_idx is None:
        raise ExtractionLoadError(
            "Could not locate Summary funded table header row "
            f"in sheet '{sheet_name}'."
        )

    header = [
        str(value).strip() if pd.notna(value) else ""
        for value in summary_raw.iloc[header_idx].tolist()
    ]
    table = summary_raw.iloc[header_idx + 1 :].copy()
    table.columns = header
    table = table.dropna(how="all")
    table = table.loc[:, [col for col in table.columns if col]]
    raw_rows = len(table)

    missing_summary = tuple(col for col in SUMMARY_REQUIRED_COLUMNS if col not in table.columns)
    if missing_summary:
        raise ExtractionLoadError(
            "Missing required Summary columns: " + ", ".join(missing_summary)
        )

    doc_ref = table["Doc Ref"].where(table["Doc Ref"].notna(), "").astype(str).str.strip()
    table = table.loc[doc_ref != ""].copy()

    if "Original CCY" in table.columns:
        original_ccy = (
            table["Original CCY"]
            .where(table["Original CCY"].notna(), "")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        table = table.loc[original_ccy != "totals:"].copy()

    funding_date_meta = _extract_summary_metadata_value(summary_raw, "Funding date")
    offer_id_meta = _extract_summary_metadata_value(summary_raw, "Offer ID")
    program_name_meta = _extract_summary_metadata_value(summary_raw, "Program Name")
    client_name_meta = _extract_summary_metadata_value(summary_raw, "Client Name")

    funding_date_meta_ts = pd.to_datetime(funding_date_meta, errors="coerce")

    mapped = pd.DataFrame(index=table.index)
    # Summary files do not carry explicit Offer File Date per invoice.
    # We use file-level Funding date when available, otherwise Purchase Date.
    mapped["Offer File Date (UTC)"] = (
        funding_date_meta_ts if pd.notna(funding_date_meta_ts) else table["Purchase Date"]
    )
    mapped["Summary File Date (UTC)"] = table["Purchase Date"]
    mapped["Reconciliation File Date (UTC)"] = table["Reconciliation Date"]
    mapped["Paid On (Europe, Madrid)"] = pd.NaT
    mapped["Repurchase Date"] = pd.NaT
    mapped["Repurchase"] = 0
    mapped["Company Code"] = table["Seller Client ID"]
    mapped["Customer"] = table["Debtor Client ID"]
    mapped["Invoice Reference"] = table["Doc Ref"]
    mapped["Document Number"] = table["Doc Ref"]
    mapped["Due Date"] = table["Due Date"]
    mapped["Purchase Price"] = table["Purchase Amount"]
    mapped["Status"] = "Accepted"
    mapped["Reason"] = pd.NA

    # Optional extraction/canonical fields kept for traceability and downstream metrics.
    optional_map = {
        "Issue Date": "Issuance date",
        "Reconciliation Date": "Reconciliation Date",
        "Total Invoice Amount (O. CCY)": "Amount",
        "Original CCY": "Currency",
        "Funding CCY": "Funding Currency",
        "Base Rate": "Base Rate",
        "Credit/flat fee": "Credit Fee/ Flat Fee",
        "Margin": "Margin",
        "All in rate": "All in Rate",
        "Interest": "Interests",
        "Seller Client ID": "Seller Client ID",
        "Debtor Name": "Debtor Name",
        "Seller Name": "Seller Name",
        "Purchase Date": "Purchase Date",
    }
    for source_col, target_col in optional_map.items():
        if source_col in table.columns:
            mapped[target_col] = table[source_col]

    if pd.notna(funding_date_meta_ts):
        mapped["Funding date"] = funding_date_meta_ts
    if pd.notna(offer_id_meta):
        mapped["Offer ID"] = offer_id_meta
    if pd.notna(program_name_meta):
        mapped["Program Name"] = program_name_meta
    if pd.notna(client_name_meta):
        mapped["Client Name"] = client_name_meta

    return mapped.reset_index(drop=True), raw_rows


def load_extraction(
    input_path: str | Path,
    sheet_name: str = "SAPUI5 Export",
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load extraction Excel and return normalized dataframe plus validation report.

    Drops rows with missing cohort date or missing purchase price.
    """
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Extraction file not found: {source}")

    resolved_sheet_name = _resolve_sheet_name(source, sheet_name)
    df = pd.read_excel(source, sheet_name=resolved_sheet_name, engine="openpyxl")
    raw_rows = len(df)

    rename_map = _strip_columns(df.columns)
    if rename_map:
        df = df.rename(columns=rename_map)

    # Standardize known variant column names.
    if "Customer " in df.columns and "Customer" not in df.columns:
        df = df.rename(columns={"Customer ": "Customer"})

    missing_required = tuple(col for col in REQUIRED_COLUMNS if col not in df.columns)
    if missing_required and _looks_like_summary_sheet(df):
        df, raw_rows = _normalize_summary_funded_sheet(source, resolved_sheet_name)
        missing_required = tuple(col for col in REQUIRED_COLUMNS if col not in df.columns)

    if missing_required:
        raise ExtractionLoadError(
            "Missing required columns: " + ", ".join(missing_required)
        )

    df = _normalize_purchase_price(df)
    df = _parse_dates(df)
    df = _normalize_text_columns(df)

    # Attach canonical columns for multi-source optimizer paths.
    canonical_df, canonical_report = to_canonical(
        df,
        mapping=EXTRACTION_TO_CANONICAL,
        source_profile=SOURCE_PROFILE_EXTRACTION,
    )
    for col in canonical_df.columns:
        if col not in df.columns:
            df[col] = canonical_df[col]

    dropped_missing_offer_date = int(df["Offer File Date (UTC)"].isna().sum())
    dropped_missing_purchase_price = int(df["Purchase Price"].isna().sum())

    df = df.dropna(subset=["Offer File Date (UTC)", "Purchase Price"]).copy()
    df["Purchase Price"] = df["Purchase Price"].astype(float)
    df = df.reset_index(drop=True)
    df["optimizer_row_id"] = df.index.astype(str)

    report = LoadReport(
        source_path=str(source),
        sheet_name=resolved_sheet_name,
        raw_rows=raw_rows,
        loaded_rows=len(df),
        dropped_missing_offer_date=dropped_missing_offer_date,
        dropped_missing_purchase_price=dropped_missing_purchase_price,
        missing_required_columns=tuple(sorted(set(missing_required) | set(canonical_report.missing_required_columns))),
    )
    return df, report
