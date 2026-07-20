"""Service layer that adapts the doc_extraction CLI flow for FastAPI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from doc_extraction.embedding.matcher import run_community_code_matching
from doc_extraction.main import _extract_with_llm
from .reference_data import ReferenceDataError, load_reference_data
API_EXPORT_COLUMNS = [
    "file",
    "doc_type",
    "line_index",
    "header_documentDate",
    "header_deliveryDate",
    "header_senderAddress",
    "header_receiverID",
    "header_shipToName",
    "header_shipToAddress",
    "header_currencyCode",
    "header_netAmount",
    "header_vendorName",
    "Business_Partner_ID",
    "Original_Vendor_Name",
    "Supplier_Match_Score",
    "Supplier_Match_Method",
    "description",
    "netAmount",
    "quantity",
    "unitPrice",
    "materialNumber",
    "itemNumber",
    "usageSummary",
    "Codes_Desc_Top5",
    "LLM_Suggestion_Desc",
    "LLM_Confidence_Desc",
    "LLM_Reason_Desc",
    "Block_By_LLM_Desc",
]
_LINE_ITEM_VALUE_COLUMNS = (
    "description",
    "netAmount",
    "quantity",
    "unitPrice",
    "materialNumber",
    "itemNumber",
    "usageSummary",
)
_NOT_DETECTED = "Not detected"


def _normalize_label(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _normalize_candidate_names(names: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for name in names:
        normalized = _normalize_label(name)
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


_HEADER_VENDOR_CANDIDATES = _normalize_candidate_names(
    [
        "vendor",
        "vendor_name",
        "vendorName",
        "vendorCompany",
        "company_name",
        "companyName",
        "senderName",
        "sender_name",
        "supplierName",
        "supplier_name",
    ]
)
_LINE_VENDOR_CANDIDATES = _normalize_candidate_names(
    [
        "Vendor",
        "vendor",
        "vendorName",
        "header_vendor",
        "header_vendorName",
        "header_vendor_name",
        "header_company_name",
        "header_companyName",
        "header_senderName",
        "header_supplierName",
    ]
)


def _resolve_candidate_columns(columns: Sequence[object], normalized_candidates: Sequence[str]) -> list[str]:
    column_map = {_normalize_label(str(column)): str(column) for column in columns}
    resolved: list[str] = []
    for candidate in normalized_candidates:
        column = column_map.get(candidate)
        if column and column not in resolved:
            resolved.append(column)
    return resolved


def _coerce_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _build_vendor_series(df: pd.DataFrame, normalized_candidates: Sequence[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    target_columns = _resolve_candidate_columns(df.columns, normalized_candidates)
    if not target_columns:
        return None

    def _pick_vendor(row: pd.Series) -> str:
        for column in target_columns:
            text = _coerce_text_value(row.get(column))
            if text:
                return text
        return ""

    return df.apply(_pick_vendor, axis=1)


def _build_header_vendor_map(headers_df: pd.DataFrame, vendor_series: pd.Series | None) -> dict[str, str]:
    if vendor_series is None or "file" not in headers_df.columns:
        return {}
    file_series = headers_df["file"].apply(_coerce_text_value)
    vendor_values = vendor_series.apply(_coerce_text_value)
    return {
        file_value: vendor_value
        for file_value, vendor_value in zip(file_series, vendor_values)
        if file_value and vendor_value
    }


def _annotate_vendor(headers_df: pd.DataFrame, line_items_df: pd.DataFrame) -> None:
    header_vendor_series = _build_vendor_series(headers_df, _HEADER_VENDOR_CANDIDATES)
    if headers_df is not None:
        if header_vendor_series is None:
            headers_df["Vendor"] = "" if headers_df.size else pd.Series(dtype=str)
        else:
            headers_df["Vendor"] = header_vendor_series.apply(_coerce_text_value)

    header_vendor_map: dict[str, str] = {}
    if headers_df is not None and not headers_df.empty:
        header_vendor_map = _build_header_vendor_map(headers_df, header_vendor_series)

    if line_items_df is None or line_items_df.empty:
        if line_items_df is not None:
            line_items_df["Vendor"] = "" if line_items_df.size else pd.Series(dtype=str)
        return

    line_vendor_series = _build_vendor_series(line_items_df, _LINE_VENDOR_CANDIDATES)
    if line_vendor_series is None:
        line_vendor_series = pd.Series([""] * len(line_items_df), index=line_items_df.index, dtype=object)

    if header_vendor_map and "file" in line_items_df.columns:
        file_series = line_items_df["file"].apply(_coerce_text_value)
        fallback = file_series.map(header_vendor_map).fillna("")
        current = line_vendor_series.astype(str).str.strip()
        replace_mask = (current == "") & (fallback != "")
        line_vendor_series = line_vendor_series.where(~replace_mask, fallback)

    line_items_df["Vendor"] = line_vendor_series.apply(_coerce_text_value)


def _normalize_scalar(value: object) -> object | None:
    """Return a JSON-safe scalar or ``None`` for a missing pandas value.

    Args:
        value: Scalar extracted from an enriched pandas row.

    Returns:
        A native Python scalar, or ``None`` when the value is missing.
    """

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _detected_value(value: object) -> object:
    """Preserve a detected scalar and label a missing value for Joule.

    Args:
        value: Scalar value from an enriched matcher row.

    Returns:
        The native scalar or the configured missing-value label.
    """

    normalized = _normalize_scalar(value)
    if normalized is None or (isinstance(normalized, str) and not normalized.strip()):
        return _NOT_DETECTED
    return normalized


def _detected_text(value: object) -> str:
    """Return detected text without replacing meaningful fallback labels.

    Args:
        value: Text-like scalar from an enriched matcher row.

    Returns:
        Detected text, retained fallback text, or the missing-value label.
    """

    normalized = _detected_value(value)
    return normalized if isinstance(normalized, str) else str(normalized)


def _confidence_percentage(value: object) -> str:
    """Format a zero-to-one confidence score as a whole percentage string.

    Args:
        value: Numeric matcher confidence or an existing text label.

    Returns:
        Whole percentage text, retained text, or the missing-value label.
    """

    normalized = _normalize_scalar(value)
    if normalized is None or (isinstance(normalized, str) and not normalized.strip()):
        return _NOT_DETECTED
    if isinstance(normalized, str):
        return normalized
    return f"{float(normalized) * 100:.0f}%"


def _genuine_line_items(line_items_df: pd.DataFrame) -> pd.DataFrame:
    """Remove extractor placeholder rows that contain no genuine item values.

    Args:
        line_items_df: Raw line-item rows returned by document extraction.

    Returns:
        A copy containing only rows with at least one detected item value.
    """

    if line_items_df is None or line_items_df.empty:
        return pd.DataFrame()
    available = [column for column in _LINE_ITEM_VALUE_COLUMNS if column in line_items_df.columns]
    if not available:
        return line_items_df.iloc[0:0].copy()
    genuine_mask = line_items_df[available].apply(
        lambda row: any(_normalize_scalar(value) not in (None, "") for value in row),
        axis=1,
    )
    return line_items_df.loc[genuine_mask].reset_index(drop=True)


def _serialize_joule_line_items(line_items_df: pd.DataFrame) -> list[dict[str, object]]:
    """Serialize every enriched row into the exact seven-field Joule contract.

    Args:
        line_items_df: Commodity-code matcher output containing enriched rows.

    Returns:
        JSON-safe dictionaries for deterministic Joule rendering.
    """

    return [
        {
            "description": _detected_text(row.get("description")),
            "net_amount": _detected_value(row.get("netAmount")),
            "quantity": _detected_value(row.get("quantity")),
            "unit_price": _detected_value(row.get("unitPrice")),
            "ai_suggested_commodity_code": _detected_text(row.get("LLM_Suggestion_Desc")),
            "ai_confidence_score": _confidence_percentage(row.get("LLM_Confidence_Desc")),
            "ai_reasoning": _detected_text(row.get("LLM_Reason_Desc")),
        }
        for _, row in line_items_df.iterrows()
    ]

@dataclass(slots=True)
class ExtractionConfig:
    llm_verify: bool = False
    llm_model: str | None = None
    llm_min_confidence: float = 0.6
    top_k: int = 5
    merge_headers: bool = False
    output_name: str | None = None

    embedding_model: str | None = None

    enable_supplier_filtering: bool = True
    supplier_match_threshold: float = 70.0
    retry_confidence_threshold: float = 0.45

    # Line item extraction retry settings
    retry_multipage_on_empty: bool = True  # Enable Tier 2 fallback with multi-page images
    add_placeholder_columns: bool = True   # Enable Tier 3 consistent column structure

    show_preview: bool = False


class ExtractionResult:
    """Normalization of the pipeline output used by the API layer."""

    def __init__(
        self,
        *,
        output_path: Path,
        headers_df: pd.DataFrame,
        line_items_df: pd.DataFrame,
        runtime_seconds: float,
        reference_data_version: str,
        errors: list[str] | None = None,
    ) -> None:
        self.output_path = output_path
        self.headers_df = headers_df
        self.line_items_df = line_items_df
        self.runtime_seconds = runtime_seconds
        self.reference_data_version = reference_data_version
        self.errors = errors or []


def _ensure_outputs_dir() -> Path:
    base = Path("outputs") / "api"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _slugify(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)
    cleaned = cleaned.strip("_")
    return cleaned or "documents"


def _resolve_output_path(pdf_paths: Sequence[Path], desired_name: str | None) -> Path:
    outputs_dir = _ensure_outputs_dir()
    if desired_name:
        base_name = _slugify(desired_name)
    elif len(pdf_paths) == 1:
        base_name = pdf_paths[0].stem
    else:
        base_name = pdf_paths[0].parent.name or "documents"

    candidate = outputs_dir / f"{base_name}.xlsx"
    if candidate.exists():
        timestamp = int(time.time())
        candidate = outputs_dir / f"{base_name}_{timestamp}.xlsx"
    return candidate


def _preview(df: pd.DataFrame, limit: int = 20) -> list[dict[str, object]]:
    """Return a JSON-serializable preview of a DataFrame.

    Args:
        df: DataFrame to preview.
        limit: Maximum number of rows included in the output.

    Returns:
        A list of row dictionaries with missing values converted to blanks.
    """

    if df is None or df.empty:
        return []
    sample = df.head(limit)
    # Ensure JSON serialisable payload from pandas types
    return sample.fillna("").to_dict(orient="records")


def _run_embedding_pipeline(
    pdf_paths: Sequence[Path],
    config: ExtractionConfig,
) -> ExtractionResult:
    start_time = time.time()
    reference_data = load_reference_data()

    headers_df, line_items_df = _extract_with_llm(
        pdf_paths,
        retry_multipage=config.retry_multipage_on_empty,
        add_placeholders=config.add_placeholder_columns
    )

    line_items_df = _genuine_line_items(line_items_df)
    if line_items_df.empty:
        raise RuntimeError("No genuine line items were extracted from the provided PDFs.")

    _annotate_vendor(headers_df, line_items_df)

    headers_arg = headers_df if config.merge_headers else None
    output_path = _resolve_output_path(pdf_paths, config.output_name)

    final_output, enriched_df = run_community_code_matching(
        line_items=line_items_df,
        headers=headers_arg,
        community_codes_path=reference_data.catalog_df,
        unspsc_context_path=reference_data.unspsc_df,
        supplier_groups_path=reference_data.supplier_groups_df,
        output_path=output_path,
        embedding_model=config.embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        llm_verify=config.llm_verify,
        llm_model=config.llm_model,
        llm_min_confidence=config.llm_min_confidence,
        enable_supplier_filtering=config.enable_supplier_filtering,
        supplier_match_threshold=config.supplier_match_threshold,
        retry_confidence_threshold=config.retry_confidence_threshold,
        top_k_codes=config.top_k,
        show_preview=config.show_preview,
        export_columns=API_EXPORT_COLUMNS,
    )

    runtime = time.time() - start_time
    return ExtractionResult(
        output_path=Path(final_output),
        headers_df=headers_df,
        line_items_df=enriched_df,
        runtime_seconds=runtime,
        reference_data_version=reference_data.data_version,
    )


def run_extraction_for_paths(
    pdf_paths: Sequence[Path],
    config: ExtractionConfig,
) -> dict:
    """Run extraction for already-materialized PDF paths.

    Args:
        pdf_paths: Resolved paths to PDF files available on local disk.
        config: Runtime options controlling LLM verification and matching.

    Returns:
        A JSON-serializable payload containing output metadata and previews.

    Raises:
        RuntimeError: If reference data cannot be loaded or extraction fails.
    """

    try:
        result = _run_embedding_pipeline(pdf_paths, config)
    except ReferenceDataError as exc:
        raise RuntimeError(str(exc)) from exc

    return {
        "output_path": str(result.output_path),
        "output_exists": result.output_path.exists(),
        "file_count": len(pdf_paths),
        "llm_verify": config.llm_verify,
        "top_k": config.top_k,
        "runtime_seconds": result.runtime_seconds,
        "reference_data_version": result.reference_data_version,
        "headers_preview": _preview(result.headers_df),
        "line_items_preview": _preview(result.line_items_df),
        "joule_line_items": _serialize_joule_line_items(result.line_items_df),
        "errors": result.errors,
        "warnings": [],
    }
