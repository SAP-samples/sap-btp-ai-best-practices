import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from ..utils.excel_validation.utils import (
    markdown_table_from_rows as util_markdown_table_from_rows,
    summarize_sheet_for_llm as util_summarize_sheet_for_llm,
    build_column_profiles as util_build_column_profiles,
    build_row_candidates as util_build_row_candidates,
    excel_col_letter as util_excel_col_letter,
)

from ..security import get_api_key


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/excel-validation",
    tags=["excel-validation"],
    dependencies=[Depends(get_api_key)],
)


# Shared orchestration service and model
ORCHESTRATION_SERVICE = OrchestrationService()
MODEL_NAME = "gpt-4.1"


def _build_validation_prompt(
    sample_rows_markdown: str, column_profiles_json: str, row_candidates_json: str
) -> str:
    """Compose a system prompt instructing the LLM to detect numeric outliers/mis-keys.

    The LLM receives a compact markdown table preview of the first N rows and column headers.
    It must identify suspicious numeric values (order-of-magnitude anomalies, impossible values, mis-typed zeros, extra digits).
    Respond in strict JSON with fields: issues (list), summary, ok (boolean).
    Each issue: { rowIndex, column, value, reason }.
    """
    return (
        "You are a data validation assistant. You analyze tabular numeric data and flag likely mis-keyed or outlier values.\n"
        "Rules:\n"
        "- Look for order-of-magnitude anomalies, misplaced decimals, extra digits, negative where impossible, zeros in place of expected non-zero, etc.\n"
        "- Consider column-wise distributions using the provided preview only (no outside knowledge).\n"
        "- Use the provided per-column statistics and candidate outliers to guide detection (IQR + robust z-score). Validate candidates and add any obvious misses.\n"
        "- Prefer column-consistency: flag values that clearly deviate from that column's empirical range (mean/median/IQR).\n"
        "- Consider context of column names: treat identifier/categorical columns (e.g., 'Plant', 'PO', 'Material', 'UOM', 'Co-Product') as non-numeric targets — do not flag them. Focus on quantities/amounts/consumption/produced.\n"
        "- Cross-check within nearby rows: if a quantity column commonly holds ~2k-3k and a lone ~6k appears, consider it a likely digit duplication or mis-key.\n"
        "- Ignore summary/aggregate rows and cells (e.g., 'TOTAL', 'TOTALS', 'GRAND TOTAL(S)', 'SUBTOTAL', 'SUM', 'AVERAGE', 'TOTAL WEIGHT', 'hide', 'hide+Auto'). Treat whole-line totals (e.g., 'TOTALS / GRAND TOTALS | ...') as aggregates. Only flag if the total is clearly inconsistent. Case-insensitive.\n"
        "- If any cell in a row contains a summary keyword, treat the entire row as an aggregate, even if the numeric totals appear in later columns (e.g., 'TOTALS / GRAND TOTALS | | | 123 | 456 | 7'). Only flag for clear miscalculation.\n"
        "- If a row fragment has only one numeric under a column that is seldom populated in neighboring rows, treat it as likely misalignment and flag only when highly confident.\n"
        "- If no clear issues, set ok=true.\n\n"
        "Output strictly as minified JSON with keys: ok (bool), summary (string), issues (array of {rowIndex,column,value,reason}). No prose outside JSON. Limit to the top 5 highest-confidence issues.\n\n"
        f"Preview (Markdown table, small subset of rows):\n{sample_rows_markdown}\n\n"
        f"Column profiles (JSON with per-column stats and pre-flagged candidates):\n```json\n{column_profiles_json}\n```\n\n"
        f"Row candidates (rows with suspicious structure or singleton numerics):\n```json\n{row_candidates_json}\n```"
    )


"""Excel Validation Router - uses utilities from utils/excel_validation."""


# (helpers moved to utils/excel_validation)


def _summarize_sheet_for_llm(sheet) -> Dict[str, Any]:
    return util_summarize_sheet_for_llm(sheet)


def _is_number(value: Any) -> bool:
    # kept for backward compatibility if referenced; delegate
    from ..utils.excel_validation.utils import is_number as _util_is_number

    return _util_is_number(value)


def _percentile(sorted_values: List[float], p: float) -> float:
    from ..utils.excel_validation.utils import percentile as _util_percentile

    return _util_percentile(sorted_values, p)


def _excel_col_letter(zero_based_index: int) -> str:
    return util_excel_col_letter(zero_based_index)


def _build_column_profiles(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    return util_build_column_profiles(headers, rows)


def _build_row_candidates(headers: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    return util_build_row_candidates(headers, rows)


def _cell_contains_summary_label(value: Any) -> bool:
    """Heuristic: detect summary/aggregate markers in a single cell (case-insensitive).

    Covers variants like: TOTAL, TOTALS, GRAND TOTAL(S), SUBTOTAL, SUM, AVERAGE, AVG,
    domain-specific labels like 'TOTAL WEIGHT', and control markers like 'hide', 'hide+Auto'.
    """
    if value is None:
        return False
    try:
        text = str(value).strip().lower()
    except Exception:
        return False
    if not text:
        return False
    keywords = [
        "total",
        "totals",
        "grand total",
        "grand totals",
        "subtotal",
        "sum",
        "average",
        "avg",
        "total weight",
        "hide",
        "hide+auto",
    ]
    return any(k in text for k in keywords)


def _is_summary_row(row: List[Any]) -> bool:
    """Treat a row as an aggregate if any cell contains a summary keyword."""
    try:
        return any(_cell_contains_summary_label(cell) for cell in row)
    except Exception:
        return False


@router.post("/validate")
async def validate_excel(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Accept an Excel file, create a compact preview, and use LLM to flag numeric outliers.

    Returns a JSON with fields: ok, summary, issues, preview (headers/firstRows).
    """
    try:
        if not file.filename.lower().endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            raise HTTPException(
                status_code=400, detail="Only .xlsx/.xlsm Excel files supported"
            )

        content = await file.read()
        try:
            from openpyxl import load_workbook
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"openpyxl not available: {e}")

        wb = load_workbook(filename=io.BytesIO(content), data_only=True, read_only=True)
        sheet = wb.active
        data = util_summarize_sheet_for_llm(sheet)
        headers = data["headers"]
        rows = data["rows"]

        if not headers:
            raise HTTPException(
                status_code=400, detail="Excel appears to have no header row"
            )

        # Exclude summary/aggregate rows for LLM analysis inputs (but keep full preview for UI)
        rows_for_stats_indices = [
            i for i, r in enumerate(rows) if not _is_summary_row(r)
        ]
        if not rows_for_stats_indices:
            rows_for_stats_indices = list(range(len(rows)))
        rows_for_stats = [rows[i] for i in rows_for_stats_indices]
        # Build data aids for the LLM: preview table and per-column stats with candidate outliers
        md_table = util_markdown_table_from_rows(
            headers, rows_for_stats, max_rows=10000
        )
        profiles = util_build_column_profiles(headers, rows_for_stats)
        row_candidates = util_build_row_candidates(headers, rows_for_stats)
        import json

        profiles_json = json.dumps(profiles, separators=(",", ":"))
        row_candidates_json = json.dumps(row_candidates, separators=(",", ":"))

        # Build and log the compiled system prompt for observability
        system_prompt_text = _build_validation_prompt(
            md_table, profiles_json, row_candidates_json
        )
        logger.info(f"[excel-validation] Compiled system prompt:\n{system_prompt_text}")

        config = OrchestrationConfig(
            template=Template(
                messages=[
                    SystemMessage("{{?system_prompt}}"),
                    UserMessage(
                        "Analyze the data preview and return the JSON result as instructed."
                    ),
                ]
            ),
            llm=LLM(name=MODEL_NAME),
        )

        response = ORCHESTRATION_SERVICE.run(
            config=config,
            template_values=[
                TemplateValue(name="system_prompt", value=system_prompt_text)
            ],
        )

        llm_text: str = response.orchestration_result.choices[0].message.content or ""

        print("============llm_text============")
        print(llm_text)
        print("============llm_text============")

        # Try parse the JSON strictly; if fails, wrap as error
        import json

        try:
            result = json.loads(llm_text)
        except Exception:
            # Best-effort correction: attempt to find JSON block
            start = llm_text.find("{")
            end = llm_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(llm_text[start : end + 1])
                except Exception:
                    result = {
                        "ok": False,
                        "summary": "Failed to parse AI result",
                        "issues": [],
                    }
            else:
                result = {
                    "ok": False,
                    "summary": "Failed to parse AI result",
                    "issues": [],
                }

        # Enrich issues with Excel coordinates (column letter + row number)
        try:
            issues = result.get("issues", []) if isinstance(result, dict) else []
            col_letters = data.get("colLetters") or [
                util_excel_col_letter(i) for i in range(len(headers))
            ]
            row_numbers = data.get("rowNumbers") or [i + 2 for i in range(len(rows))]

            def _to_number(v: Any) -> Optional[float]:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    return float(v)
                try:
                    s = str(v).replace(",", "").strip()
                    return float(s)
                except Exception:
                    return None

            name_to_index = {str(h).lower(): idx for idx, h in enumerate(headers)}

            for it in issues:
                r_idx = it.get("rowIndex")
                # Map LLM rowIndex (based on filtered preview) back to original rows
                if isinstance(r_idx, int) and 0 <= r_idx < len(rows_for_stats_indices):
                    mapped_r_idx = rows_for_stats_indices[r_idx]
                    it["rowIndex"] = mapped_r_idx
                    r_idx = mapped_r_idx
                c_idx = it.get("columnIndex")
                # Try resolve by header name if provided
                if c_idx is None and it.get("column") is not None:
                    col_val = it.get("column")
                    # 1) If numeric index is provided, accept 0-based or 1-based
                    if isinstance(col_val, int):
                        if 0 <= col_val < len(headers):
                            c_idx = col_val
                        elif 1 <= col_val <= len(headers):
                            c_idx = col_val - 1
                    else:
                        # Try parse numeric-like strings as indices
                        try:
                            as_int = int(str(col_val).strip())
                            if 0 <= as_int < len(headers):
                                c_idx = as_int
                            elif 1 <= as_int <= len(headers):
                                c_idx = as_int - 1
                        except Exception:
                            # Fallback to header-name lookup
                            idx = name_to_index.get(str(col_val).lower())
                            if isinstance(idx, int):
                                c_idx = idx

                # Try resolve by matching value within the row
                if isinstance(r_idx, int) and c_idx is None and 0 <= r_idx < len(rows):
                    row_vals = rows[r_idx]
                    target_num = _to_number(it.get("value"))
                    # exact string match
                    if it.get("value") is not None:
                        val_str = str(it["value"])
                        for j, cell in enumerate(row_vals):
                            if str(cell) == val_str:
                                c_idx = j
                                break
                    # numeric tolerant match
                    if c_idx is None and target_num is not None:
                        best_j = None
                        best_err = None
                        for j, cell in enumerate(row_vals):
                            num = _to_number(cell)
                            if num is None:
                                continue
                            err = abs(num - target_num)
                            if best_err is None or err < best_err:
                                best_err = err
                                best_j = j
                        if best_j is not None and (best_err is not None):
                            # accept if sufficiently close (1e-6 relative tolerance)
                            if best_err <= max(1e-6, abs(target_num) * 1e-6):
                                c_idx = best_j

                # Write back indices for UI
                if isinstance(c_idx, int):
                    it["columnIndex"] = c_idx
                if isinstance(r_idx, int):
                    # Keep rowIndex as-is
                    pass

                # If both indices are present but the value at (r_idx, c_idx)
                # does not approximately match the reported value, try to
                # correct a potential off-by-one row by searching nearby rows.
                try:

                    def _approx_equal(a: Any, b: Any) -> bool:
                        a_num = _to_number(a)
                        b_num = _to_number(b)
                        if a_num is None or b_num is None:
                            return str(a) == str(b)
                        tol = max(1e-9, abs(b_num) * 1e-6)
                        return abs(a_num - b_num) <= tol

                    if (
                        isinstance(r_idx, int)
                        and isinstance(c_idx, int)
                        and 0 <= r_idx < len(rows)
                        and 0 <= c_idx < len(headers)
                    ):
                        current_row = rows[r_idx] if r_idx < len(rows) else []
                        current_val = (
                            current_row[c_idx] if c_idx < len(current_row) else None
                        )
                        target_val = it.get("value")
                        if not _approx_equal(current_val, target_val):
                            # search small neighborhood for a better row match
                            start = max(0, r_idx - 3)
                            end = min(len(rows) - 1, r_idx + 3)
                            found_row = None
                            for rr in range(start, end + 1):
                                row_vals = rows[rr]
                                cand = (
                                    row_vals[c_idx] if c_idx < len(row_vals) else None
                                )
                                if _approx_equal(cand, target_val):
                                    found_row = rr
                                    break
                            if found_row is None:
                                # broaden search across entire column to find exact/approx match
                                for rr in range(0, len(rows)):
                                    row_vals = rows[rr]
                                    cand = (
                                        row_vals[c_idx]
                                        if c_idx < len(row_vals)
                                        else None
                                    )
                                    if _approx_equal(cand, target_val):
                                        found_row = rr
                                        break
                            if found_row is not None and found_row != r_idx:
                                r_idx = found_row
                                it["rowIndex"] = r_idx
                except Exception:
                    # best-effort adjustment only
                    pass

                # Compute Excel coords
                excel_col = None
                excel_row = None
                if isinstance(c_idx, int) and 0 <= c_idx < len(col_letters):
                    excel_col = col_letters[c_idx]
                if isinstance(r_idx, int) and 0 <= r_idx < len(row_numbers):
                    excel_row = row_numbers[r_idx]
                if excel_col:
                    it["excelColumn"] = excel_col
                if excel_row is not None:
                    it["excelRow"] = excel_row
                if excel_col and excel_row is not None:
                    it["excelCell"] = f"{excel_col}{excel_row}"
        except Exception:
            # best-effort; ignore enrichment failures
            pass

        # Final safety filter: drop any issues that fall on summary/aggregate rows
        try:
            if isinstance(result, dict) and isinstance(result.get("issues"), list):
                summary_row_indexes = {
                    idx for idx, r in enumerate(rows) if _is_summary_row(r)
                }
                filtered_issues: List[Dict[str, Any]] = []
                for it in result["issues"]:
                    r_idx = it.get("rowIndex")
                    if isinstance(r_idx, int) and r_idx in summary_row_indexes:
                        # skip
                        continue
                    filtered_issues.append(it)
                result["issues"] = filtered_issues
        except Exception:
            # best-effort filtering only
            pass

        preview = {
            "headers": headers,
            "firstRows": rows,
            "rowNumbers": data.get("rowNumbers", []),
            "colLetters": data.get("colLetters", []),
        }
        return {"success": True, "result": result, "preview": preview}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Excel validation failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/save")
async def save_validated_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mock save endpoint: always returns success without persistence."""
    try:
        _ = payload  # no-op
        return {"success": True, "message": "data saved succesfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}
