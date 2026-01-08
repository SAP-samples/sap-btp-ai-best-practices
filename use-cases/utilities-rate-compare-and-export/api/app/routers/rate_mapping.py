import logging
from pathlib import Path
import csv
from io import StringIO
from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from gen_ai_hub.orchestration.service import OrchestrationService

from ..security import get_api_key
from ..utils.document_intelligence import DocumentTranscriber
from ..utils.rate_mapping_llm import map_rates, extract_csv_from_llm_output
from ..utils.prompt_registry import get_available_types

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])


class CompareRequest(BaseModel):
    old_csv: str
    new_csv: str


# Shared Orchestration Service instance (can be reused across the app)
ORCHESTRATION_SERVICE = OrchestrationService()

# Reusable transcriber instance with reduced payload size for rate PDFs
TRANSCRIBER = DocumentTranscriber(
    orchestration_service=ORCHESTRATION_SERVICE,
    dpi=200,  # lower DPI to reduce bytes
    image_format="jpg",  # JPEG instead of PNG
    jpeg_quality=75,  # adjust if you want stronger compression
    grayscale=False,  # keep color by default
    auto_grayscale_megapixels=4.0,  # optionally grayscale very large pages
)

_MASTER_CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Rate and Price Keys Master Template.csv"
)


def _load_master_csv() -> str:
    try:
        with _MASTER_CSV_PATH.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Master CSV file not found at %s", _MASTER_CSV_PATH)
        raise HTTPException(
            status_code=500,
            detail="Master CSV template not found on server.",
        )
    except Exception as e:
        logger.exception("Failed to load master CSV: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to load master CSV template.",
        )


def _parse_csv(csv_text: str):
    """Parse semicolon-delimited CSV into headers list and rows list."""
    # Normalize newlines
    text = csv_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return [], []
    reader = csv.reader(StringIO(text), delimiter=";")
    rows = list(reader)
    if not rows:
        return [], []
    headers = rows[0]
    data_rows = rows[1:]
    return headers, data_rows


def _usage_to_dict(usage):
    if not usage:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "_asdict"):
        return usage._asdict()
    if hasattr(usage, "model_dump"):  # Pydantic v2
        return usage.model_dump()
    if hasattr(usage, "dict"):  # Pydantic v1
        return usage.dict()
    # Fallback
    return {
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _sum_usages(usages):
    total = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    for u in usages:
        u_dict = _usage_to_dict(u)
        if u_dict:
            total["completion_tokens"] += u_dict.get("completion_tokens", 0)
            total["prompt_tokens"] += u_dict.get("prompt_tokens", 0)
            total["total_tokens"] += u_dict.get("total_tokens", 0)
    return total


@router.get("/types")
def get_file_types():
    """Return list of available file types for rate mapping."""
    return get_available_types()


@router.post("/pdf")
async def pdf_rate_mapping(
    file: UploadFile = File(...),
    file_type: str = "nc-rates",
):
    """Accept a PDF, transcribe to Markdown, map rates via LLM, and return CSV + table data."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file.")

        # Step 1: PDF → Markdown using transcriber
        transcription = await TRANSCRIBER.transcribe_pdf_to_markdown(content)
        markdown = transcription.get("markdown", "")
        transcription_usage = transcription.get("token_usage", {})

        if not markdown:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract markdown from PDF.",
            )

        # Step 2: Load master CSV
        master_csv = _load_master_csv()
        master_header = master_csv.splitlines()[0] if master_csv else ""

        # Step 3: Call LLM to map rates
        llm_output, map_usage = await map_rates(
            markdown, master_csv, prompt_key=file_type
        )
        csv_text = extract_csv_from_llm_output(llm_output, master_header)

        # Step 4: Parse CSV into headers + rows
        headers, rows = _parse_csv(csv_text)

        token_usage = {
            "transcription": _usage_to_dict(transcription_usage),
            "mapping": _usage_to_dict(map_usage),
            "total": _sum_usages([transcription_usage, map_usage]),
        }

        return JSONResponse(
            {
                "markdown": markdown,
                "csvString": csv_text,
                "headers": headers,
                "rows": rows,
                "tokenUsage": token_usage,
            }
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("PDF rate mapping failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/markdown")
async def md_rate_mapping(
    file: UploadFile = File(...),
    file_type: str = "nc-rates",
):
    """Accept a Markdown file, map rates via LLM, and return CSV + table data.

    This bypasses the PDF transcription step and is intended for testing and
    debugging the rate mapping logic directly with prepared Markdown.
    """
    if not file.filename or not file.filename.lower().endswith(
        (".md", ".markdown", ".txt")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .md, .markdown, or .txt files are supported.",
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        try:
            markdown = content_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            markdown = content_bytes.decode("utf-8", "ignore").strip()

        if not markdown:
            raise HTTPException(
                status_code=400,
                detail="Markdown file contained no readable content.",
            )

        # Load master CSV
        master_csv = _load_master_csv()
        master_header = master_csv.splitlines()[0] if master_csv else ""

        # Call LLM to map rates
        llm_output, map_usage = await map_rates(
            markdown, master_csv, prompt_key=file_type
        )
        csv_text = extract_csv_from_llm_output(llm_output, master_header)

        # Parse CSV into headers + rows
        headers, rows = _parse_csv(csv_text)

        token_usage = {
            "mapping": _usage_to_dict(map_usage),
            "total": _sum_usages([map_usage]),
        }

        return JSONResponse(
            {
                "markdown": markdown,
                "csvString": csv_text,
                "headers": headers,
                "rows": rows,
                "tokenUsage": token_usage,
            }
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Markdown rate mapping failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
def compare_rates_endpoint(request: CompareRequest):
    """Compare two rate CSVs and return differences.

    Note: defined as sync function (def) so FastAPI runs it in a threadpool,
    preventing pandas from blocking the main event loop.
    """
    logger.info(
        "Received rate comparison request. Old CSV len: %d, New CSV len: %d",
        len(request.old_csv),
        len(request.new_csv),
    )
    try:
        # Load CSVs
        # Using string type to avoid inference issues
        df_old = pd.read_csv(StringIO(request.old_csv), sep=";", dtype=str)
        df_new = pd.read_csv(StringIO(request.new_csv), sep=";", dtype=str)

        # Normalize columns
        df_old.columns = df_old.columns.str.strip()
        df_new.columns = df_new.columns.str.strip()

        # Ensure key columns exist
        key_cols = ["PREIS", "VONZONE_1", "BISZONE_1", "PREISBTR_1"]
        for col in key_cols:
            if col not in df_old.columns:
                df_old[col] = ""
            if col not in df_new.columns:
                df_new[col] = ""

        # Helper to create key
        def make_key(df):
            return (
                df["PREIS"].fillna("").astype(str)
                + "||"
                + df["VONZONE_1"].fillna("").astype(str)
            )

        df_old["_KEY"] = make_key(df_old)
        df_new["_KEY"] = make_key(df_new)

        # Merge with indicator to find source
        merged = pd.merge(
            df_old,
            df_new,
            on="_KEY",
            how="outer",
            suffixes=("_OLD", "_NEW"),
            indicator=True,
        )

        # Base headers from new file (excluding _KEY)
        base_cols = [c for c in df_new.columns if c != "_KEY"]
        if not base_cols:
            base_cols = [c for c in df_old.columns if c != "_KEY"]

        final_rows = []

        # Prepare headers: We want standard columns, but insert OLD columns next to their counterparts
        # We know we only have PREISBTR_1_OLD and BISZONE_1_OLD
        final_headers = ["RowStatus"]
        for col in base_cols:
            final_headers.append(col)
            if col == "PREISBTR_1":
                final_headers.append("PREISBTR_1_OLD")
            elif col == "BISZONE_1":
                final_headers.append("BISZONE_1_OLD")

        for _, row in merged.iterrows():
            merge_status = row["_merge"]
            status = "Unchanged"

            # Map of changed columns for this row: col_name -> True
            changed_cells = {}

            # Extract previous values for tracking changes
            preis_old = (
                row.get("PREISBTR_1_OLD") if pd.notna(row.get("PREISBTR_1_OLD")) else ""
            )
            bis_old = (
                row.get("BISZONE_1_OLD") if pd.notna(row.get("BISZONE_1_OLD")) else ""
            )

            # Handle removed/added specifically for getting values
            if merge_status == "left_only":
                status = "Removed"
                # Ensure we have the old values even if columns didn't have suffix (should have due to merge, but safety check)
                if not preis_old and "PREISBTR_1" in row:
                    preis_old = row["PREISBTR_1"]
                if not bis_old and "BISZONE_1" in row:
                    bis_old = row["BISZONE_1"]

            elif merge_status == "right_only":
                status = "Added"
            else:
                # Check for specific field changes (Price and Tier End)
                # Ignore ABDATUM changes but show new value in table

                is_changed = False

                # Check PREISBTR_1 (Price)
                preis_new = row.get("PREISBTR_1_NEW")
                if pd.isna(preis_new) and "PREISBTR_1" in row:
                    preis_new = row["PREISBTR_1"]

                try:
                    p1 = float(str(preis_old).replace(",", ".")) if preis_old else 0.0
                    p2 = (
                        float(str(preis_new).replace(",", "."))
                        if pd.notna(preis_new)
                        else 0.0
                    )
                    if abs(p1 - p2) > 1e-8:
                        is_changed = True
                        changed_cells["PREISBTR_1"] = True
                except:
                    if str(preis_old) != str(preis_new):
                        is_changed = True
                        changed_cells["PREISBTR_1"] = True

                # Check BISZONE_1 (Tier End)
                bis_new = row.get("BISZONE_1_NEW")
                if pd.isna(bis_new) and "BISZONE_1" in row:
                    bis_new = row["BISZONE_1"]

                try:
                    b1 = float(str(bis_old).replace(",", ".")) if bis_old else 0.0
                    b2 = (
                        float(str(bis_new).replace(",", "."))
                        if pd.notna(bis_new)
                        else 0.0
                    )
                    # Tier limits can be large integers, float comparison safety
                    if abs(b1 - b2) > 1e-8:
                        is_changed = True
                        changed_cells["BISZONE_1"] = True
                except:
                    if str(bis_old) != str(bis_new):
                        is_changed = True
                        changed_cells["BISZONE_1"] = True

                if is_changed:
                    status = "Changed"

            # Construct row data
            out_row = [status]

            for col in base_cols:
                val = ""
                # Logic to pick correct value from merged row
                if status == "Removed":
                    if f"{col}_OLD" in row:
                        val = row[f"{col}_OLD"]
                    elif col in row:
                        val = row[col]
                else:
                    if f"{col}_NEW" in row:
                        val = row[f"{col}_NEW"]
                    elif col in row:
                        val = row[col]

                out_row.append(str(val) if pd.notna(val) else "")

                if col == "PREISBTR_1":
                    out_row.append(str(preis_old))
                elif col == "BISZONE_1":
                    out_row.append(str(bis_old))

            final_rows.append(
                {"data": out_row, "changes": [k for k, v in changed_cells.items()]}
            )

        return JSONResponse(
            {
                "headers": final_headers,
                "rows": [r["data"] for r in final_rows],
                "row_changes": [r["changes"] for r in final_rows],
            }
        )

    except Exception as e:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(e))
