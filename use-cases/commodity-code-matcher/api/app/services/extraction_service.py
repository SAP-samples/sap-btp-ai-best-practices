"""Service layer that adapts the doc_extraction CLI flow for FastAPI."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
from fastapi import UploadFile

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


async def _save_uploads(upload_files: Sequence[UploadFile]) -> list[Path]:
    if not upload_files:
        raise ValueError("At least one PDF must be uploaded.")

    temp_dir = Path(tempfile.mkdtemp(prefix="api_uploads_", dir=str(_ensure_outputs_dir())))
    saved: list[Path] = []
    for upload in upload_files:
        filename = upload.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Unsupported file type for '{filename}'. Only PDF is allowed.")
        destination = temp_dir / filename
        data = await upload.read()
        destination.write_bytes(data)
        saved.append(destination.resolve())
    return saved


def _preview(df: pd.DataFrame, limit: int = 20) -> list[dict[str, object]]:
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

    if line_items_df.empty:
        raise RuntimeError("No line items were extracted from the provided PDFs.")

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


def resolve_download_path(relative_path: str) -> Path:
    base = _ensure_outputs_dir().resolve()
    candidate = (base / relative_path).resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("Invalid download path.")
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")
    return candidate


async def run_extraction_pipeline(
    upload_files: Sequence[UploadFile],
    config: ExtractionConfig,
) -> dict:
    pdf_paths = await _save_uploads(upload_files)

    # Heavy lifting runs in a thread to keep the event loop responsive.
    try:
        result = await asyncio.to_thread(_run_embedding_pipeline, pdf_paths, config)
    except ReferenceDataError as exc:
        raise RuntimeError(str(exc)) from exc

    outputs_dir = _ensure_outputs_dir()
    download_path = str(result.output_path.relative_to(outputs_dir))

    return {
        "output_path": str(result.output_path),
        "download_path": download_path,
        "output_exists": result.output_path.exists(),
        "file_count": len(pdf_paths),
        "llm_verify": config.llm_verify,
        "top_k": config.top_k,
        "runtime_seconds": result.runtime_seconds,
        "reference_data_version": result.reference_data_version,
        "headers_preview": _preview(result.headers_df),
        "line_items_preview": _preview(result.line_items_df),
        "errors": result.errors,
        "warnings": [],
    }
