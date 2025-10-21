"""
Credit Creation Page - Orchestrates end-to-end workflow to extract, validate, and evaluate credit.

This page follows docs/workflow.md to:
- Upload required PDFs/Excels (KYC, CSF, Vendor Comments, Calificacion_pagos.xlsx, AI_credito.xlsx)
- Extract key attributes from PDFs via backend extraction API
- Validate cross-document fields (legal name/Razón Social, RFC, address, etc.)
- Parse Excel sheets to construct invoices/behavior and credit request inputs
- Build the EvaluateRequest payload and call /api/credit/evaluate
- Display scores, checks and decision hints

Note: This first version scaffolds the UI and flow with placeholders where needed.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utilities and API client
import sys
sys.path.append(str(Path(__file__).parent.parent))

now = datetime.now()

CALIFICACION_PAGOS_PATH = Path(__file__).resolve().parents[2] / "data" / "Calificacion_pagos.xlsx"

from api_client import (
    extract_single_document,
    make_api_request,
    verify_documents,
    summarize_fields,
)
from utils import (
    load_css_files,
    initialize_session_state,
    create_results_dataframe,
    parse_api_error,
    markdown_to_pdf,
    sanitize_markdown_for_display,
)

# Page Configuration
st.set_page_config(
    page_title="Credit Creation",
    page_icon="static/images/SAP_logo_square.png",
    layout="wide"
)
st.logo("static/images/SAP_logo.svg")

# Load CSS
css_files = [
    os.path.join(Path(__file__).parent.parent.parent, "static", "styles", "variables.css"),
    os.path.join(Path(__file__).parent.parent.parent, "static", "styles", "style.css"),
]
load_css_files(css_files)

# Initialize session state
initialize_session_state()

# Persisted state for extracted key-values
if "cc_kyc_kv" not in st.session_state:
    st.session_state.cc_kyc_kv = {}
if "cc_csf_kv" not in st.session_state:
    st.session_state.cc_csf_kv = {}
if "cc_vendor_kv" not in st.session_state:
    st.session_state.cc_vendor_kv = {}
if "cc_cgv_kv" not in st.session_state:
    st.session_state.cc_cgv_kv = {}
if "cc_ic_kv" not in st.session_state:
    st.session_state.cc_ic_kv = {}
if "cc_il_kv" not in st.session_state:
    st.session_state.cc_il_kv = {}
if "cc_processed" not in st.session_state:
    st.session_state.cc_processed = False
if "cc_invoices" not in st.session_state:
    st.session_state.cc_invoices = []
if "cc_request_inputs" not in st.session_state:
    st.session_state.cc_request_inputs = {}
# New: store Calificacion_pagos fields for report header
if "cc_usuario_cxc_1" not in st.session_state:
    st.session_state.cc_usuario_cxc_1 = None
if "cc_vendedor_1" not in st.session_state:
    st.session_state.cc_vendedor_1 = None

# Title and description
st.title("Credit Creation")
st.markdown("Follow the guided workflow to assemble a credit evaluation payload and run the policy engine.")

# --- Helpers ---

def _kv_from_extraction(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Turn extraction response into key→value mapping for quick access."""
    kv: Dict[str, Any] = {}
    if resp and resp.get("success") and resp.get("results"):
        for item in resp["results"]:
            f = item.get("field")
            v = item.get("answer")
            if f:
                kv[f] = v
    return kv


def _safe_dt(date_str: Optional[str]) -> Optional[datetime]:
    """Try to parse a date string to datetime; return None on failure."""
    if not date_str:
        return None
    # Try Spanish natural language dates like "21 DE MAYO DE 2025"
    try:
        parsed = _parse_spanish_date(str(date_str))
        if isinstance(parsed, datetime):
            return parsed
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(date_str), fmt)
        except Exception:
            continue
    try:
        # Fallback: pandas parser with NaT check
        ts = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


# Robust parser for textual advance-purchase counts like "más de 3"
def _parse_adv_purchases(value: Any) -> int:
    """Parse textual/numeric advance-purchase counts into an integer.

    Examples:
    - "más de 3" → 4 (>=3)
    - "3" → 3
    - "2 compras" → 2
    - 1.0 → 1
    - None/unknown → 0
    """
    if value is None:
        return 0
    try:
        # If numeric (int/float as Excel cell), cast to int directly
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        text = str(value).strip().lower()
        # Handle accent-insensitive 'más'
        text_ascii = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        if ("mas" in text_ascii or "más" in text) and "3" in text_ascii:
            # Interpret as more than 3 → set to 4 to satisfy >=3 rule
            return 4
        # Extract first integer present
        import re
        m = re.search(r"(\d+)", text_ascii)
        if m:
            return int(m.group(1))
        return 0
    except Exception:
        return 0


# Normalize text to compare accent-insensitively
def _normalize_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    return (
        s.replace("á", "a")
         .replace("é", "e")
         .replace("í", "i")
         .replace("ó", "o")
         .replace("ú", "u")
    )


def _infer_use_case(operacion_value: Any) -> str:
    """Infer use_case (new|update|exception) from 'Operación' text with robust matching.

    Rules:
    - contains 'excepcion' → exception
    - contains 'actualizacion' → update
    - contains 'nuevo'/'nueva' → new
    Default: new
    """
    t = _normalize_text(operacion_value)
    if not t:
        return "new"
    # Exception first to avoid accidental overlaps
    if "excep" in t:
        return "exception"
    if "actualiz" in t:
        return "update"
    if ("nuevo" in t) or ("nueva" in t) or ("nuev" in t):
        return "new"
    return "new"


def _load_calificacion_pagos() -> Optional[BytesIO]:
    """Load Calificacion_pagos.xlsx into an in-memory stream for downstream parsing."""
    if not CALIFICACION_PAGOS_PATH.exists():
        return None
    data = CALIFICACION_PAGOS_PATH.read_bytes()
    file_like = BytesIO(data)
    file_like.name = CALIFICACION_PAGOS_PATH.name
    return file_like


# Parse Spanish dates like "21 de mayo de 2025" (case/accents-insensitive)
def _parse_spanish_date(value: Any) -> Optional[datetime]:
    text = _normalize_text(value)
    if not text:
        return None
    import re
    # Match: 1-2 digit day, optional "de", month name, optional "de", 4-digit year
    m = re.search(r"(\d{1,2})\s*(?:de\s*)?([a-záéíóú]+)\s*(?:de\s*)?(\d{4})", text)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2)
    year = int(m.group(3))
    month_map = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
        "noviembre": 11, "diciembre": 12,
    }
    # month_name is already normalized accents removed by _normalize_text
    month_num = month_map.get(month_name)
    if not month_num:
        return None
    try:
        return datetime(year, month_num, day)
    except Exception:
        return None


def _normalize_rfc(value: Any) -> str:
    """Normalize RFC codes for robust equality comparison.

    - Uppercase, trim whitespace
    - Remove spaces and hyphens
    - Keep only alphanumerics, Ñ and & (valid RFC characters)
    """
    import re
    s = str(value or "").strip().upper()
    s = s.replace(" ", "").replace("-", "")
    s = re.sub(r"[^A-Z0-9Ñ&]", "", s)
    return s


# ----- Payment Analysis Table helpers -----

def _years_window_from_latest(latest_year: int, span: int = 4) -> List[int]:
    """Return a contiguous range of `span` years ending at `latest_year`, oldest→newest.

    Example: latest_year=2024, span=4 → [2021, 2022, 2023, 2024]
    """
    return [latest_year - i for i in range(span - 1, -1, -1)]


def _label_display(en_label: str) -> str:
    """Return the display label for engine categories (English)."""
    mapping = {
        "Excellent": "Excellent",
        "Good": "Good",
        "Regular": "Regular",
        "Poor": "Poor",
        "Critical": "Critical",
    }
    return mapping.get(en_label or "", en_label or "")


def _build_payment_analysis_data(result: Dict[str, Any], payload: Dict[str, Any], now_dt: datetime) -> Dict[str, Any]:
    """Compute per-year counts by label and supporting metrics for the table.

    - Uses `scores.cp_by_invoice` from the evaluation result and maps each invoice to its due_date year
      using the last evaluated payload kept in session state.
    - Ensures exactly 4 columns (last 4 years). Missing years are shown with zeros.
    - Returns data ready to render, including CA% per year, C3M%, CH% and CAL.
    """
    scores = (result or {}).get("scores", {})
    cp_list: List[Dict[str, Any]] = scores.get("cp_by_invoice") or []

    # Build invoice_id → year/due_date map from payload invoices (due_date determines the year)
    inv_year_by_id: Dict[str, int] = {}
    inv_due_by_id: Dict[str, Optional[datetime]] = {}
    try:
        for inv in (payload or {}).get("behavior", {}).get("invoices", []) or []:
            inv_id = inv.get("invoice_id")
            due_raw = inv.get("due_date")
            due_dt = _safe_dt(due_raw)
            if inv_id and isinstance(due_dt, datetime):
                inv_year_by_id[inv_id] = due_dt.year
                inv_due_by_id[inv_id] = due_dt
            elif inv_id:
                inv_due_by_id[inv_id] = None
    except Exception:
        pass

    # Determine latest year present in data (prefer invoices/CA map); fallback to now
    years_in_data: List[int] = []
    try:
        years_in_data.extend(list(set(inv_year_by_id.values())))
    except Exception:
        pass
    try:
        ca_keys = [int(k) for k in (scores.get("CA_by_year_pct") or {}).keys()]
        years_in_data.extend(ca_keys)
    except Exception:
        pass
    latest_year = max(years_in_data) if years_in_data else now_dt.year
    years = _years_window_from_latest(latest_year, span=4)

    # Initialize counts per label per year
    en_labels = ["Excellent", "Good", "Regular", "Poor", "Critical"]
    counts: Dict[int, Dict[str, int]] = {y: {lbl: 0 for lbl in en_labels} for y in years}

    # Count labels for invoices within the window
    for e in cp_list:
        inv_id = e.get("invoice_id")
        lbl = e.get("label")
        y = inv_year_by_id.get(inv_id)
        if y in counts and lbl in counts[y]:
            counts[y][lbl] += 1

    # Compute totals per year
    totals = {y: sum(counts[y].values()) for y in years}

    # CA% by year; default 0 when missing (normalize keys to int)
    ca_map_raw: Dict[Any, Any] = scores.get("CA_by_year_pct") or {}
    try:
        ca_map_int = {int(k): float(v) for k, v in ca_map_raw.items()}
    except Exception:
        ca_map_int = {}
    ca_by_year = {int(y): float(ca_map_int.get(int(y), 0.0)) for y in years}

    # Other metrics
    c3m_pct = scores.get("C3M_pct")
    ch_pct = scores.get("CH_pct")
    cal_label = scores.get("CAL")

    # C3M counts by label (due within last 92 days relative to as_of)
    as_of_dt = _safe_dt((payload or {}).get("as_of")) or now_dt
    c3m_counts = {lbl: 0 for lbl in en_labels}
    try:
        for e in cp_list:
            inv_id = e.get("invoice_id")
            lbl = e.get("label")
            due_dt = inv_due_by_id.get(inv_id)
            if isinstance(due_dt, datetime):
                delta = (as_of_dt.date() - due_dt.date()).days
                if delta <= 92:
                    if lbl in c3m_counts:
                        c3m_counts[lbl] += 1
    except Exception:
        pass

    # Parámetros column derived from group thresholds
    group = (result or {}).get("group") or "A"
    def _param_for_label(g: str, lbl: str) -> str:
        if g == "B":
            if lbl == "Excellent":
                return "AV"
            if lbl == "Good":
                return "≤3"
            if lbl == "Regular":
                return "≤6"
            if lbl == "Poor":
                return "≤9"
            if lbl == "Critical":
                return "≥10"
        # Default group A
        if lbl == "Excellent":
            return "AV"
        if lbl == "Good":
            return "≤5"
        if lbl == "Regular":
            return "≤10"
        if lbl == "Poor":
            return "≤15"
        if lbl == "Critical":
            return ">15"
        return ""
    parametros_map = {lbl: _param_for_label(group, lbl) for lbl in en_labels}

    return {
        "years": years,
        "counts": counts,
        "totals": totals,
        "ca_by_year": ca_by_year,
        "c3m_pct": c3m_pct,
        "ch_pct": ch_pct,
        "cal_label": cal_label,
        "c3m_counts": c3m_counts,
        "parametros": parametros_map,
        "group": group,
    }


def _render_payment_analysis_table(data: Dict[str, Any]) -> str:
    """Render the payment analysis table as HTML with color bands and weights.

    Structure mirrors the provided image:
    - Header with "Payment Analysis" over four yearly columns and "Total Points" over "Last 3 Months".
    - First data row shows weights 3-6-8-10 under the 4 years; the last cell shows 27.
    - Category rows display yearly counts for labels: Excellent, Good, Regular, Poor, Critical.
    - "Total" row (counts sum) and "Rating" row with CA% per year and C3M% in the last column.
    - A caption below the table shows CH% and CAL.
    """
    years: List[int] = data["years"]
    counts: Dict[int, Dict[str, int]] = data["counts"]
    totals: Dict[int, int] = data["totals"]
    ca_by_year: Dict[int, float] = data["ca_by_year"]
    c3m_pct = data.get("c3m_pct")
    ch_pct = data.get("ch_pct")
    cal_label = data.get("cal_label")
    c3m_counts: Dict[str, int] = data.get("c3m_counts", {})
    parametros_map: Dict[str, str] = data.get("parametros", {})

    # Visual styles (inline to keep this self-contained in Streamlit)
    color_row = {
        "Excellent": "#4CAF50",  # green
        "Good": "#8BC34A",       # light green
        "Regular": "#FFC107",    # amber
        "Poor": "#FF9800",       # orange
        "Critical": "#F44336",   # red
    }

    table_css = """
    <style>
      .paytbl { border-collapse: collapse; width: 100%; font-size: 13px; }
      .paytbl th, .paytbl td { border: 1px solid #b0b0b0; padding: 6px 8px; text-align: center; background: #ffffff; }
      .paytbl th { background: #1f4e79; color: #fff; font-weight: 600; }
      .paytbl .subhead { background: #294f75; color: #fff; }
      .paytbl .labelcell { text-align: left; font-weight: 600; color: #333; }
      .paytbl .weight { background: #e6f0fa; font-weight: 600; }
      .paytbl .last3 { background: #c00000; color: #fff; font-weight: 600; }
      .paytbl .tphead { background: #385723; color: #fff; font-weight: 600; }
      .paytbl .totrow { background: #ececec; font-weight: 600; }
      .paytbl .calrow { background: #fff3cd; font-weight: 600; }
    </style>
    """

    # Build header
    y1, y2, y3, y4 = years
    html = [table_css]
    html.append("<table class='paytbl'>")
    html.append(
        f"<tr>"
        f"<th rowspan='2'>Weight</th>"
        f"<th class='subhead' colspan='4'>Payment Analysis</th>"
        f"<th class='tphead' colspan='1'>Total Points</th>"
        f"<th rowspan='2'>Parameters</th>"
        f"</tr>"
    )
    html.append(
        f"<tr>"
        f"<th>{y1}</th><th>{y2}</th><th>{y3}</th><th>{y4}</th>"
        f"<th class='last3'>Last 3 Months</th>"
        f"</tr>"
    )

    # Weights row (3-6-8-10) and total points 27 in the last column
    weights = [3, 6, 8, 10]
    html.append(
        "<tr>" +
        "<td class='labelcell weight'>Weight</td>" +
        "".join([f"<td class='weight'>{w}</td>" for w in weights]) +
        f"<td class='weight'>27</td>" +
        f"<td class='weight'></td>" +
        "</tr>"
    )

    # Category rows
    for en in ["Excellent", "Good", "Regular", "Poor", "Critical"]:
        display_label = _label_display(en)
        bg = color_row.get(en, "#ffffff")
        style_label = f" style=\"background:{bg}; color:#000;\""
        c1 = counts.get(y1, {}).get(en, 0)
        c2 = counts.get(y2, {}).get(en, 0)
        c3 = counts.get(y3, {}).get(en, 0)
        c4 = counts.get(y4, {}).get(en, 0)
        c_last = c3m_counts.get(en, 0)
        param_txt = parametros_map.get(en, "")
        html.append(
            f"<tr>"
            f"<td class='labelcell'{style_label}>{display_label}</td>"
            f"<td>{c1}</td><td>{c2}</td><td>{c3}</td><td>{c4}</td>"
            f"<td>{c_last}</td>"
            f"<td>{param_txt}</td>"
            f"</tr>"
        )

    # Totals row
    t1, t2, t3, t4 = totals.get(y1, 0), totals.get(y2, 0), totals.get(y3, 0), totals.get(y4, 0)
    html.append(
        f"<tr class='totrow'>"
        f"<td class='labelcell'>Total</td>"
        f"<td>{t1}</td><td>{t2}</td><td>{t3}</td><td>{t4}</td>"
        f"<td>{sum(c3m_counts.values())}</td>"
        f"<td></td>"
        f"</tr>"
    )

    # Calificación row: yearly CA% and C3M% in last column
    def fmt_pct(v: Any) -> str:
        try:
            return f"{float(v):.2f}%"
        except Exception:
            return "-"

    html.append(
        f"<tr class='calrow'>"
        f"<td class='labelcell'>Rating</td>"
        f"<td>{fmt_pct(ca_by_year.get(y1, 0.0))}</td>"
        f"<td>{fmt_pct(ca_by_year.get(y2, 0.0))}</td>"
        f"<td>{fmt_pct(ca_by_year.get(y3, 0.0))}</td>"
        f"<td>{fmt_pct(ca_by_year.get(y4, 0.0))}</td>"
        f"<td>{fmt_pct(c3m_pct)}</td>"
        f"<td>{('%.2f%%' % ch_pct) if isinstance(ch_pct, (int, float)) else '-'}</td>"
        f"</tr>"
    )

    html.append("</table>")

    # Caption with CAL label (keep minimal; CH already shown in table)
    caption = ""
    if cal_label is not None:
        caption = f"<div style='margin-top:6px; font-size:12px'><b>CAL:</b> {cal_label}</div>"

    return "".join(html) + caption

# --- Layout: Inputs ---

calificacion_pagos_available = CALIFICACION_PAGOS_PATH.exists()

with st.expander("Upload Required Documents", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        kyc_pdf = st.file_uploader("KYC (Know Your Customer)", type=["pdf"], key="cc_kyc")
        csf_pdf = st.file_uploader("CSF (Tax Status Certificate)", type=["pdf"], key="cc_csf")
        cgv_pdf = st.file_uploader("CGV (General Sales Conditions Form)", type=["pdf"], key="cc_cgv")
    with col2:
        vendor_pdf = st.file_uploader("Vendor Comments", type=["pdf"], key="cc_vendor")
        ic_pdf = st.file_uploader("Commercial Investigation", type=["pdf"], key="cc_ic")
        il_pdf = st.file_uploader("Legal Investigation", type=["pdf"], key="cc_il")
        if calificacion_pagos_available:
            st.markdown("Calificacion_pagos.xlsx: Automatically loaded")
        else:
            st.error("Calificacion_pagos.xlsx not found in api/data")
    with col3:
        ai_credito_xlsx = st.file_uploader("AI_credito (xlsx/csv)", type=["xlsx", "csv"], key="cc_ai_credito")
        st.caption("Required sheets: Facturado, TD2024, TD2025, TD3Month in Calificacion_pagos.xlsx; solicitud in AI_credito.xlsx")

st.markdown("---")

# --- Step 2: Extract PDFs ---
process_btn = st.button(
    "Process Documents",
    type="primary",
    use_container_width=True,
    disabled=not (kyc_pdf and csf_pdf and vendor_pdf and cgv_pdf and ic_pdf and il_pdf and calificacion_pagos_available and ai_credito_xlsx)
)

kyc_kv: Dict[str, Any] = {}
csf_kv: Dict[str, Any] = {}
vendor_kv: Dict[str, Any] = {}
cgv_kv: Dict[str, Any] = {}
ic_kv: Dict[str, Any] = {}
il_kv: Dict[str, Any] = {}

if process_btn:
    pagos_xlsx = _load_calificacion_pagos()
    if pagos_xlsx is None:
        st.error("Calificacion_pagos.xlsx not found in api/data")
        st.session_state.cc_processed = False
        st.session_state.cc_invoices = []
        st.stop()
    # Reset processed flag
    st.session_state.cc_processed = False
    progress_bar = st.progress(0, text="Processing PDFs...")

    # Map to backend document types from extraction router
    doc_map = {
        "kyc": "conoce_cliente",
        "csf": "constancia_fiscal",
        "vendor": "comentarios_vendedor",
        "cgv": "cgv",
        "ic": "investigacion_comercial",
        "il": "investigacion_legal",
    }

    def _extract(upload, doc_type_key: str) -> Dict[str, Any]:
        upload.seek(0)
        content = upload.read()
        return extract_single_document(
            file_content=content,
            filename=upload.name,
            document_type=doc_map[doc_type_key],
            temperature=0.1,
            language="es",
        )

    # Execute in parallel and update progress after each completion
    kyc_resp, csf_resp, vendor_resp, cgv_resp, ic_resp, il_resp = {}, {}, {}, {}, {}, {}
    try:
        completed = 0
        total_docs = 6
        with ThreadPoolExecutor(max_workers=total_docs) as executor:
            futures = {}
            futures[executor.submit(_extract, kyc_pdf, "kyc")] = "kyc"
            futures[executor.submit(_extract, csf_pdf, "csf")] = "csf"
            futures[executor.submit(_extract, vendor_pdf, "vendor")] = "vendor"
            futures[executor.submit(_extract, cgv_pdf, "cgv")] = "cgv"
            futures[executor.submit(_extract, ic_pdf, "ic")] = "ic"
            futures[executor.submit(_extract, il_pdf, "il")] = "il"

            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    resp = fut.result()
                except Exception as e:
                    resp = {"success": False, "error": str(e)}
                if key == "kyc":
                    kyc_resp = resp
                elif key == "csf":
                    csf_resp = resp
                elif key == "vendor":
                    vendor_resp = resp
                elif key == "cgv":
                    cgv_resp = resp
                elif key == "ic":
                    ic_resp = resp
                elif key == "il":
                    il_resp = resp
                completed += 1
                pct = int(min(100, (completed / total_docs) * 100))
                progress_bar.progress(pct, text=f"Processing PDFs... {completed}/{total_docs}")
    except Exception as e:
        st.error(f"Parallel extraction failed: {e}")

    # Persist key-value maps from PDF extraction
    kyc_kv = _kv_from_extraction(kyc_resp)
    csf_kv = _kv_from_extraction(csf_resp)
    vendor_kv = _kv_from_extraction(vendor_resp)
    cgv_kv = _kv_from_extraction(cgv_resp)
    ic_kv = _kv_from_extraction(ic_resp)
    il_kv = _kv_from_extraction(il_resp)
    # Summarize verbose Investigación Legal fields and replace originals
    try:
        # Map UI display labels -> DOX keys for the five target fields
        display_to_dox = {
            "Historial de Procesos Judiciales de la Empresa": "Historial_procesos_judiciales_empresa",
            "Factor de Riesgo de la Empresa": "Factor_riesgo_empresa",
            "Historial de Procesos Judiciales de la Persona": "Historial_procesos_judiciales_persona",
            "Factor de Riesgo de la Persona": "Factor_riesgo_persona",
            "Confirmación de Representación Social de la Persona": "Confirmacion_Representacion_social_persona",
        }
        dox_to_display = {v: k for k, v in display_to_dox.items()}

        # Build summarization input using DOX keys with values pulled from UI display keys
        il_for_summary = {}
        for display_key, dox_key in display_to_dox.items():
            if il_kv.get(display_key) not in (None, ""):
                il_for_summary[dox_key] = il_kv.get(display_key)

        il_summary_resp = summarize_fields(il_for_summary) if il_for_summary else {"success": False}
        if isinstance(il_summary_resp, dict) and il_summary_resp.get("success"):
            il_summaries = il_summary_resp.get("summaries", {})
            # Apply summaries back to UI keys
            for dox_key, summary_val in il_summaries.items():
                display_key = dox_to_display.get(dox_key, dox_key)
                il_kv[display_key] = summary_val
            # Persist summaries separately for later defaults and reporting
            st.session_state["cc_il_summary"] = il_summaries

        else:
            st.session_state["cc_il_summary"] = {}
    except Exception:
        st.session_state["cc_il_summary"] = {}
    st.session_state.cc_kyc_kv = kyc_kv
    st.session_state.cc_csf_kv = csf_kv
    st.session_state.cc_vendor_kv = vendor_kv
    st.session_state.cc_cgv_kv = cgv_kv
    st.session_state.cc_ic_kv = ic_kv
    st.session_state.cc_il_kv = il_kv

    # Persist IC → MMR parsed value for configuration default
    try:
        mmr_raw = ic_kv.get("Monto Máximo Recomendado") or ic_kv.get("Monto de crédito recomendado")
        mmr_val = None
        if mmr_raw is not None:
            txt = str(mmr_raw)
            txt = txt.replace("MX$", "$").replace("USD", "$")
            txt = txt.replace(",", "").replace(" ", "")
            import re as _re
            m = _re.search(r"-?\d+(?:\.\d+)?", txt)
            mmr_val = float(m.group(0)) if m else None
        if mmr_val is not None:
            st.session_state["cc_ic_mmr"] = float(mmr_val)
    except Exception:
        pass

    # Persist CGV signature date from extraction ('Fecha de firma')
    try:
        cgv_date_raw = cgv_kv.get("Fecha de firma")
        cgv_date_dt = _safe_dt(cgv_date_raw)
        if isinstance(cgv_date_dt, datetime):
            st.session_state["cc_cgv_date"] = cgv_date_dt
    except Exception:
        pass

    # Compute CSF date for tax certificate
    try:
        csf_date_raw = (
            csf_kv.get("Fecha de emisión")
            or csf_kv.get("Fecha de emision")
            or csf_kv.get("Fecha")
        )
        csf_date_dt = _safe_dt(csf_date_raw)
        st.session_state["cc_csf_date"] = csf_date_dt
    except Exception:
        st.session_state["cc_csf_date"] = None

    # Quick address verification for later payload construction
    try:
        kyc_address = kyc_kv.get("Domicilio Fiscal (Información Fiscal)")
        csf_address = csf_kv.get("Datos del Domicilio registrado")
        st.session_state["cc_address_match"] = False
        if kyc_address and csf_address:
            address_key_map = {
                "Domicilio Fiscal (Información Fiscal)": {
                    "to": "Datos del Domicilio registrado",
                    "comparator": "address",
                    "threshold": 0.60,
                }
            }
            addr_verify = verify_documents(kyc_kv, csf_kv, address_key_map)
            addr_match = False
            if addr_verify and isinstance(addr_verify, dict):
                summary = addr_verify.get("summary", {})
                addr_match = bool(summary.get("verified", 0) >= 1)
            st.session_state["cc_address_match"] = bool(addr_match)
    except Exception:
        pass

    # Parse AI_credito (xlsx/csv)
    try:
        ai_credito_xlsx.seek(0)
        name_ai = getattr(ai_credito_xlsx, "name", "").lower()
        if name_ai.endswith(".csv"):
            df_ai = pd.read_csv(ai_credito_xlsx)
        else:
            df_ai = pd.read_excel(ai_credito_xlsx, sheet_name="solicitud")
        row = df_ai.iloc[0].to_dict() if len(df_ai) > 0 else {}

        def _ai(label: str) -> Any:
            return row.get(label)

        use_case = _infer_use_case(_ai("Operación"))
        requested_amount = float(_ai("Monto solicitado de Crédito") or 0)
        requested_terms_days = int(_ai("Plazo solicitado de Crédito") or 30)
        current_credit_line = float(_ai("LC Actual") or 0)
        current_terms_days = int(_ai("Plazo actual (Dias)") or 0)
        adv_count = _parse_adv_purchases(_ai("# Ventas con pago anticipado:"))

        # Persist AI values
        st.session_state["cc_ai_rfc"] = _ai("RFC")
        st.session_state["cc_ai_customer_service"] = _ai("Customer Service")
        st.session_state["cc_ai_insurance_provider"] = _ai("Aseguradora a cubrir")
        st.session_state["cc_ai_business_unit"] = _ai("Unidad de Negocio:")
        st.session_state["cc_ai_sap_id"] = _ai("# SAP")

        st.session_state.cc_request_inputs = {
            "use_case": use_case,
            "requested_amount": requested_amount,
            "requested_currency": "MXN",
            "requested_terms_days": requested_terms_days,
            "last_update_date": None,
            "current_credit_line": current_credit_line,
            "current_credit_currency": "MXN",
            "advance_purchases_count": adv_count,
            "current_terms_days": current_terms_days,
            "cgv_signed_date": (st.session_state.get("cc_cgv_date").isoformat() if isinstance(st.session_state.get("cc_cgv_date"), datetime) else None),
        }
    except Exception as e:
        st.error(f"Failed to parse AI_credito (xlsx/csv): {e}")
        st.session_state.cc_request_inputs = {}

    # Parse Calificacion_pagos (xlsx/csv)
    invoices: List[Dict[str, Any]] = []
    try:
        pagos_xlsx.seek(0)
        name_pagos = getattr(pagos_xlsx, "name", "").lower()
        if name_pagos.endswith(".csv"):
            df_fact = pd.read_csv(pagos_xlsx)
        else:
            df_fact = pd.read_excel(pagos_xlsx, sheet_name="Facturado")

        kyc_ss = st.session_state.cc_kyc_kv
        csf_ss = st.session_state.cc_csf_kv
        vendor_ss = st.session_state.cc_vendor_kv
        target_name = kyc_ss.get("Razón Social") or csf_ss.get("Denominación / Razón Social") or vendor_ss.get("Nombre del Cliente")
        if target_name and "Nombre 1" in df_fact.columns:
            mask = df_fact["Nombre 1"].astype(str).str.contains(str(target_name), case=False, na=False)
            df_cust = df_fact[mask].copy()
        else:
            df_cust = df_fact.copy()

        # Persist customer ERP ID
        try:
            # Case-insensitive column mapping
            cols_map = {str(c).strip().lower(): c for c in df_fact.columns}
            src_df = df_cust if not df_cust.empty else df_fact

            # Persist ERP identifier from "Cuenta"
            cuenta_col = cols_map.get("cuenta")
            if cuenta_col is not None:
                cuenta_series = src_df[cuenta_col].dropna().astype(str)
                if not cuenta_series.empty:
                    st.session_state["cc_customer_id"] = cuenta_series.iloc[0].strip()
            # - "Usuario CxC 1"
            usuario_cxc_col = cols_map.get("usuario cxc 1")
            if usuario_cxc_col is not None and usuario_cxc_col in src_df.columns:
                serie = src_df[usuario_cxc_col].dropna().astype(str)
                if not serie.empty:
                    st.session_state["cc_usuario_cxc_1"] = serie.iloc[0].strip()

            # - "Vendedor 1"
            vendedor1_col = cols_map.get("vendedor 1")
            if vendedor1_col is not None and vendedor1_col in src_df.columns:
                serie2 = src_df[vendedor1_col].dropna().astype(str)
                if not serie2.empty:
                    st.session_state["cc_vendedor_1"] = serie2.iloc[0].strip()
        except Exception:
            # Ignore extraction errors; values are optional
            pass

        def to_dt(v):
            if pd.isna(v):
                return None
            if isinstance(v, datetime):
                return v
            return _safe_dt(str(v))

        for _, r in df_cust.iterrows():
            invoice_id = str(r.get("Nº documento") or r.get("#SAP") or f"INV_{_}")
            issue_date = to_dt(r.get("Fe.contabilización"))
            due_date = to_dt(r.get("Vencimiento neto"))
            paid_date = to_dt(r.get("Fecha compensación"))
            amount = float(r.get("Importe en moneda doc.") or 0)
            currency = str(r.get("Moneda del documento") or "MXN").upper()
            if issue_date and due_date:
                invoices.append({
                    "invoice_id": invoice_id,
                    "issue_date": issue_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "paid_date": paid_date.isoformat() if paid_date else None,
                    "amount": amount,
                    "currency": currency,
                })
        st.session_state.cc_invoices = invoices
    except Exception as e:
        st.error(f"Failed to parse Calificacion_pagos (xlsx/csv): {e}")
        st.session_state.cc_invoices = []

    # Mark done
    st.session_state.cc_processed = True
    progress_bar.progress(100, text="Processing complete")
    st.success("Documents processed")


if not st.session_state.get("cc_processed"):
    st.stop()

# --- Key Information ---
with st.expander("Key Information", expanded=True):
    kyc_ss = st.session_state.cc_kyc_kv
    csf_ss = st.session_state.cc_csf_kv
    vendor_ss = st.session_state.cc_vendor_kv
    cgv_ss = st.session_state.cc_cgv_kv
    ic_ss = st.session_state.cc_ic_kv
    il_ss = st.session_state.cc_il_kv

    # Client name from CSF ground truth when available
    client_name = csf_ss.get("Denominación / Razón Social") or kyc_ss.get("Razón Social") or vendor_ss.get("Nombre del Cliente")
    st.markdown(f"**Customer:** {client_name or 'NA'}")

    import re
    def _html_escape(s: Any) -> str:
        t = str(s) if s is not None else ""
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return t

    def _normalize_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def _norm_str(s: Any) -> str:
        return _normalize_spaces(_normalize_text(s))

    def _norm_addr(s: Any) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", _norm_str(s))

    def _parse_money(v: Any) -> Optional[float]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            txt = str(v)
            txt = txt.replace("MX$", "$").replace("USD", "$")
            txt = txt.replace(",", "").replace(" ", "")
            m = re.search(r"-?\d+(?:\.\d+)?", txt)
            return float(m.group(0)) if m else None
        except Exception:
            return None

    def _parse_int(v: Any) -> Optional[int]:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
            m = re.search(r"-?\d+", str(v))
            return int(m.group(0)) if m else None
        except Exception:
            return None

    def _verify_compare(
        a_label: str,
        a_val: Any,
        b_label: str,
        b_val: Any,
        comparator: str,
        ui_label: str,
        address_threshold: float = 0.6,
    ) -> bool:
        """Compare two values using the verification API comparators.

        - Maps UI comparator types to API comparators:
          address -> address (with threshold)
          int -> number
          numeric -> id (alphanumeric-only compare for amounts)
          string -> text (default)
        - Label-specific overrides:
          RFC/SAP ID/Customer ERP ID -> id
          Vendedor -> name
        """
        # Determine API comparator and threshold
        comp = "text"
        thr = None

        comp_lower = (comparator or "").lower()
        if comp_lower == "address":
            comp = "address"
            thr = address_threshold
        elif comp_lower == "int":
            comp = "number"
        elif comp_lower == "numeric":
            comp = "id"
        else:
            comp = "text"

        # Label-specific overrides
        lbl = (ui_label or "").strip().lower()
        if lbl in {"rfc", "sap id", "customer erp id", "customer erp", "legacy customer id", "sapid"}:
            comp = "id"
            thr = None
        elif lbl == "vendedor":
            comp = "name"
            thr = None

        # Compose and call verification API
        try:
            key_a = "A"
            key_b = "B"
            spec: Dict[str, Any] = {"to": key_b, "comparator": comp}
            if thr is not None:
                spec["threshold"] = float(thr)
            resp = verify_documents(doc_a={key_a: a_val}, doc_b={key_b: b_val}, key_map={key_a: spec})
            if isinstance(resp, dict) and resp.get("verified"):
                return bool(resp["verified"][0].get("match"))
            return False
        except Exception:
            return False

    def _editable_selectbox(
        label: str,
        session_key: str,
        base_value: Any,
        options: List[str],
        sources: Dict[str, Any],
        comparator: str = "string",
        prefer_csf: bool = False,
    ) -> Any:
        """Render an editable selectbox field with live validation summary.

        UI layout: "Label: [selectbox] — Validation text"
        - session_key: unique Streamlit session key to persist the edited value
        - base_value: initial value to prefill when no user edit exists
        - options: list of available options for the selectbox
        - sources: mapping of source name -> value to validate against
        - comparator: one of {string,int,numeric,address}
        - prefer_csf: currently informational; we validate against all sources
        """
        # Resolve current editable value
        edited_value = st.session_state.get(session_key, base_value)
        
        # Ensure the value is in the options list
        if edited_value not in options:
            edited_value = base_value if base_value in options else options[0]

        # Label
        st.markdown(f"**{_html_escape(label)}**")

        # Selectbox
        edited_value = st.selectbox(
            label="",
            options=options,
            index=options.index(edited_value) if edited_value in options else 0,
            key=session_key,
            label_visibility="collapsed",
        )

        # Build validation summary against provided sources
        present_sources = {k: v for k, v in sources.items() if v not in (None, "")}
        validation_text = "NA"
        color_red = False
        if len(present_sources) == 0:
            validation_text = "No sources available for validation"
            color_red = True
        else:
            # Check if edited value matches any source
            matches = []
            for source_name, source_value in present_sources.items():
                try:
                    is_match = _verify_compare("EDIT", edited_value, source_name, source_value, comparator, label)
                except Exception:
                    is_match = False
                if is_match:
                    matches.append(source_name)
            
            if matches:
                validation_text = f"Matches: {', '.join(matches)}"
                color_red = False
            else:
                validation_text = f"Does not match: {', '.join(present_sources.keys())}"
                color_red = True

        # Display validation summary
        color = "red" if color_red else "green"
        st.markdown(f"<div style='color:{color};margin-top:4px'>{validation_text}</div>", unsafe_allow_html=True)
        
        return edited_value

    def _editable_validate(
        label: str,
        session_key: str,
        base_value: Any,
        sources: Dict[str, Any],
        comparator: str = "string",
        prefer_csf: bool = False,
        is_multiline: bool = False,
    ) -> Any:
        """Render an editable field with live validation summary.

        UI layout: "Label: [editable input] — Validation text"
        - session_key: unique Streamlit session key to persist the edited value
        - base_value: initial value to prefill when no user edit exists
        - sources: mapping of source name -> value to validate against
        - comparator: one of {string,int,numeric,address}
        - prefer_csf: currently informational; we validate against all sources
        - is_multiline: when True, use a text_area for longer inputs like addresses
        """
        # Resolve current editable value
        edited_value = st.session_state.get(session_key, base_value)

        # Label
        st.markdown(f"**{_html_escape(label)}**")

        # Input
        if is_multiline:
            edited_value = st.text_area(
                label="",
                value=str(edited_value or ""),
                key=session_key,
                height=80,
                label_visibility="collapsed",
            )
        else:
            edited_value = st.text_input(
                label="",
                value=str(edited_value or ""),
                key=session_key,
                label_visibility="collapsed",
            )

        # Build validation summary against provided sources
        present_sources = {k: v for k, v in sources.items() if v not in (None, "")}
        validation_text = "NA"
        color_red = False
        if len(present_sources) == 0:
            validation_text = "No sources available for validation"
            color_red = True
        else:
            matched: List[str] = []
            mismatched: List[tuple] = []
            for src, val in present_sources.items():
                try:
                    is_match = _verify_compare("EDIT", edited_value, src, val, comparator, label)
                except Exception:
                    is_match = False
                if is_match:
                    matched.append(src)
                else:
                    mismatched.append((src, val))

            if matched and not mismatched:
                validation_text = f"Matches {' | '.join(matched)}"
            elif matched and mismatched:
                color_red = True
                mm = " | ".join([f"{src} - {_html_escape(val)}" for src, val in mismatched])
                validation_text = f"Partial: matches {' | '.join(matched)} — Do not match: {mm}"
            else:
                color_red = True
                mm = " | ".join([f"{src} - {_html_escape(val)}" for src, val in mismatched])
                validation_text = f"Do not match: {mm}"

        # Validation tag under the input
        st.markdown(
            f"<div style='margin-top:4px;color:{'red' if color_red else 'inherit'}'>{_html_escape(validation_text)}</div>",
            unsafe_allow_html=True,
        )

        return edited_value

    def _validate(label: str, sources: Dict[str, Any], comparator: str = "string", prefer_csf: bool = False):
        present = {k: v for k, v in sources.items() if v not in (None, "")}
        color_red = False
        text = "NA"

        # Prefer CSF as ground truth when requested
        if prefer_csf and ("CSF" in present):
            base_display = present.get("CSF")
            matched: List[str] = []
            mismatched: List[tuple] = []
            for src, val in present.items():
                if src == "CSF":
                    continue
                is_match = _verify_compare("CSF", base_display, src, val, comparator, label)
                if is_match:
                    matched.append(src)
                else:
                    mismatched.append((src, val))

            if matched:
                text = f"{_html_escape(base_display)} — CSF (matches {' | '.join(matched)})"
            else:
                text = f"{_html_escape(base_display)} — CSF"
            if mismatched:
                color_red = True
                mm = " | ".join([f"{src} - {_html_escape(val)}" for src, val in mismatched if val not in (None, "")])
                if mm:
                    text += f" — Do not match: {mm}"
        else:
            if len(present) == 0:
                color_red = True
                text = "NA"
            elif len(present) == 1:
                src, val = next(iter(present.items()))
                text = f"{_html_escape(val)} — {src}"
            else:
                # Use the first available source as base and compare others against it
                iter_items = list(present.items())
                base_src, base_val = iter_items[0]
                all_match = True
                for src, val in iter_items[1:]:
                    if not _verify_compare(base_src, base_val, src, val, comparator, label):
                        all_match = False
                        break

                if all_match:
                    text = f"{_html_escape(base_val)} — Validated ({' | '.join(present.keys())})"
                else:
                    color_red = True
                    parts = " | ".join([f"{src} - {_html_escape(present[src])}" for src in present.keys()])
                    text = f"Not validated ({parts})"

        st.markdown(
            f"<div style='margin-bottom:4px'><b>{_html_escape(label)}:</b> "
            f"<span style='color:{'red' if color_red else 'inherit'}'>{text}</span></div>",
            unsafe_allow_html=True,
        )

    def _country_from_nacionalidad(v: Any) -> Optional[str]:
        t = _norm_str(v)
        if not t:
            return None
        return "MX" if "mex" in t else t.upper()

    # Attributes (Editable + Validation)
    # Location (address) — prefer CSF as initial text, allow multiline editing
    ubicacion_base = (
        csf_ss.get("Datos del Domicilio registrado")
        or kyc_ss.get("Domicilio Fiscal (Información Fiscal)")
        or (cgv_ss.get("Domicilio Completo") if isinstance(cgv_ss, dict) else None)
        or (ic_ss.get("Domicilio Comercial") if isinstance(ic_ss, dict) else None)
        or (il_ss.get("Domicilio Fiscal") if isinstance(il_ss, dict) else None)
    )
    _editable_validate(
        "Location",
        "cc_edit_ubicacion",
        ubicacion_base,
        {
            "KYC": kyc_ss.get("Domicilio Fiscal (Información Fiscal)"),
            "CSF": csf_ss.get("Datos del Domicilio registrado"),
            "CGV": cgv_ss.get("Domicilio Completo") if isinstance(cgv_ss, dict) else None,
            "Commercial Investigation": ic_ss.get("Domicilio Comercial") if isinstance(ic_ss, dict) else None,
            "Legal Investigation": il_ss.get("Domicilio Fiscal") if isinstance(il_ss, dict) else None,
        },
        comparator="address",
        prefer_csf=True,
        is_multiline=True,
    )

    # SAP ID — editable, prefill from AI_credito or Vendor
    try:
        _vendor_sap = str(int(eval(vendor_ss.get("ID Cliente SAP")))) if vendor_ss.get("ID Cliente SAP") else None
    except Exception:
        _vendor_sap = None
    sap_base = st.session_state.get("cc_ai_sap_id") or _vendor_sap or ""
    _editable_validate(
        "SAP ID",
        "cc_edit_sap_id",
        sap_base,
        {
            "AI_credito": st.session_state.get("cc_ai_sap_id"),
            "Vendor": _vendor_sap,
        },
        comparator="string",
        prefer_csf=False,
    )

    # Customer ERP ID — editable, sourced from PaymentRating "Cuenta"
    erp_id_base = st.session_state.get("cc_customer_id") or ""
    _editable_validate(
        "Customer ERP ID",
        "cc_edit_erp_id",
        erp_id_base,
        {
            "PaymentRating": st.session_state.get("cc_customer_id"),
        },
        comparator="string",
        prefer_csf=False,
    )

    # Operation — editable selectbox with three options
    operacion_base = (st.session_state.get("cc_request_inputs", {}) or {}).get("use_case") or "new"
    operacion_options = ["new", "update", "exception"]
    _editable_selectbox(
        "Operation",
        "cc_edit_operacion",
        operacion_base,
        operacion_options,
        {
            "AI_credito": (st.session_state.get("cc_request_inputs", {}) or {}).get("use_case"),
        },
        comparator="string",
        prefer_csf=False,
    )

    # Seller — editable, prefer AI_credito
    vendedor_base = (str(st.session_state.get("cc_ai_customer_service")) or "").capitalize() or (str(vendor_ss.get("Responsable de customer service")) or "").capitalize()
    _editable_validate(
        "Seller",
        "cc_edit_vendedor",
        vendedor_base,
        {
            "AI_credito": (str(st.session_state.get("cc_ai_customer_service")) or "").capitalize(),
            "Vendor": (str(vendor_ss.get("Responsable de customer service")) or "").capitalize(),
        },
        comparator="string",
        prefer_csf=False,
    )

    # Insurance provider — editable, from AI_credito
    aseguradora_base = st.session_state.get("cc_ai_insurance_provider") or ""
    _editable_validate(
        "Insurance Provider",
        "cc_edit_aseguradora",
        aseguradora_base,
        {
            "AI_credito": st.session_state.get("cc_ai_insurance_provider"),
        },
        comparator="string",
        prefer_csf=False,
    )

    # Current credit line — editable, prefer AI_credito
    lc_actual_base = (st.session_state.get("cc_request_inputs") or {}).get("current_credit_line") or vendor_ss.get("Línea de Crédito Actual") or kyc_ss.get("Línea de Crédito Actual") or ""
    _editable_validate(
        "Current Credit Line",
        "cc_edit_lc_actual",
        lc_actual_base,
        {
            "KYC": kyc_ss.get("Línea de Crédito Actual"),
            "Vendor": vendor_ss.get("Línea de Crédito Actual"),
            "AI_credito": (st.session_state.get("cc_request_inputs") or {}).get("current_credit_line"),
        },
        comparator="numeric",
        prefer_csf=False,
    )

    # Requested credit line — editable, prefer AI_credito
    lc_solicitada_base = (st.session_state.get("cc_request_inputs") or {}).get("requested_amount") or vendor_ss.get("Línea de Crédito Solicitada") or kyc_ss.get("Línea de Crédito Solicitada") or ""
    _editable_validate(
        "Requested Credit Line",
        "cc_edit_lc_solicitada",
        lc_solicitada_base,
        {
            "KYC": kyc_ss.get("Línea de Crédito Solicitada"),
            "Vendor": vendor_ss.get("Línea de Crédito Solicitada"),
            "AI_credito": (st.session_state.get("cc_request_inputs") or {}).get("requested_amount"),
        },
        comparator="numeric",
        prefer_csf=False,
    )

    # Current term — editable, prefer AI_credito
    plazo_actual_base = (st.session_state.get("cc_request_inputs") or {}).get("current_terms_days") or vendor_ss.get("Plazo de Crédito Actual") or kyc_ss.get("Plazo de Crédito Actual") or ""
    _editable_validate(
        "Current Term (Days)",
        "cc_edit_plazo_actual",
        plazo_actual_base,
        {
            "KYC": kyc_ss.get("Plazo de Crédito Actual"),
            "Vendor": vendor_ss.get("Plazo de Crédito Actual"),
            "AI_credito": (st.session_state.get("cc_request_inputs") or {}).get("current_terms_days"),
        },
        comparator="int",
        prefer_csf=False,
    )

    # Requested term — editable, prefer AI_credito
    plazo_solicitado_base = (st.session_state.get("cc_request_inputs") or {}).get("requested_terms_days") or vendor_ss.get("Plazo de Crédito Solicitado") or kyc_ss.get("Plazo de Crédito Solicitado") or ""
    _editable_validate(
        "Requested Term (Days)",
        "cc_edit_plazo_solicitado",
        plazo_solicitado_base,
        {
            "KYC": kyc_ss.get("Plazo de Crédito Solicitado"),
            "Vendor": vendor_ss.get("Plazo de Crédito Solicitado"),
            "AI_credito": (st.session_state.get("cc_request_inputs") or {}).get("requested_terms_days"),
        },
        comparator="int",
        prefer_csf=False,
    )

    # RFC — editable, prefer AI_credito; include Investigación Legal RFC when present
    rfc_base = (
        st.session_state.get("cc_ai_rfc")
        or csf_ss.get("RFC")
        or kyc_ss.get("RFC")
        or (il_ss.get("RFC") if isinstance(il_ss, dict) else None)
        or ""
    )
    _editable_validate(
        "RFC",
        "cc_edit_rfc",
        rfc_base,
        {
            "KYC": kyc_ss.get("RFC"),
            "CSF": csf_ss.get("RFC"),
            "CGV": cgv_ss.get("RFC") if isinstance(cgv_ss, dict) else None,
            "Commercial Investigation": ic_ss.get("Identificador Fiscal") if isinstance(ic_ss, dict) else None,
            "Legal Investigation": il_ss.get("RFC") if isinstance(il_ss, dict) else None,
            "AI_credito": st.session_state.get("cc_ai_rfc"),
        },
        comparator="string",
        prefer_csf=True,
    )

    # Nacionalidad → MX when Mexican
    nac_raw = kyc_ss.get("Nacionalidad")
    nac_norm = _country_from_nacionalidad(nac_raw) if nac_raw else None
    st.markdown(
        f"<div style='margin-bottom:4px'><b>Nationality:</b> {nac_norm or 'NA'}</div>",
        unsafe_allow_html=True,
    )

    # Editable Calificacion_pagos supplemental fields (no cross-document validation)
    # Usuario CxC 1
    _editable_validate(
        "Usuario CxC 1",
        "cc_edit_usuario_cxc_1",
        st.session_state.get("cc_usuario_cxc_1"),
        {
            "PaymentRating": st.session_state.get("cc_usuario_cxc_1"),
        },
        comparator="string",
        prefer_csf=False,
        is_multiline=False,
    )
    # Vendedor 1
    _editable_validate(
        "Vendedor 1",
        "cc_edit_vendedor_1",
        st.session_state.get("cc_vendedor_1"),
        {
            "PaymentRating": st.session_state.get("cc_vendedor_1"),
        },
        comparator="string",
        prefer_csf=False,
        is_multiline=False,
    )

    # Sync editable key facts into the request payload persisted in session state
    req_inputs_raw = st.session_state.get("cc_request_inputs", {}) or {}
    req_inputs = dict(req_inputs_raw) if isinstance(req_inputs_raw, dict) else {}
    req_updated = False

    requested_amt_edit = _parse_money(st.session_state.get("cc_edit_lc_solicitada"))
    if requested_amt_edit is not None:
        if req_inputs.get("requested_amount") != requested_amt_edit:
            req_inputs["requested_amount"] = requested_amt_edit
            req_updated = True

    current_line_edit = _parse_money(st.session_state.get("cc_edit_lc_actual"))
    if current_line_edit is not None:
        if req_inputs.get("current_credit_line") != current_line_edit:
            req_inputs["current_credit_line"] = current_line_edit
            req_updated = True

    requested_terms_edit = _parse_int(st.session_state.get("cc_edit_plazo_solicitado"))
    if requested_terms_edit is not None:
        if req_inputs.get("requested_terms_days") != requested_terms_edit:
            req_inputs["requested_terms_days"] = requested_terms_edit
            req_updated = True

    current_terms_edit = _parse_int(st.session_state.get("cc_edit_plazo_actual"))
    if current_terms_edit is not None:
        if req_inputs.get("current_terms_days") != current_terms_edit:
            req_inputs["current_terms_days"] = current_terms_edit
            req_updated = True

    # Update use_case from the editable selectbox
    use_case_edit = st.session_state.get("cc_edit_operacion")
    if use_case_edit and req_inputs.get("use_case") != use_case_edit:
        req_inputs["use_case"] = use_case_edit
        req_updated = True

    if req_updated:
        st.session_state.cc_request_inputs = req_inputs

    # Persist ERP identifier override when user provides a new one
    erp_id_override = (st.session_state.get("cc_edit_erp_id") or "").strip()
    if erp_id_override:
        st.session_state["cc_customer_id_override"] = erp_id_override
    else:
        st.session_state.pop("cc_customer_id_override", None)


# --- Step 4: Parse Excel files to build inputs (UI removed; values sourced from session) ---

invoices: List[Dict[str, Any]] = st.session_state.get("cc_invoices", [])
request_inputs: Dict[str, Any] = st.session_state.get("cc_request_inputs", {})


st.markdown("---")

# --- Step 5: Configuration ---

_persona_options = [
    ("Corporate (Persona Moral)", "PM"),
    ("Individual (Persona Física)", "PF"),
]
with st.expander("Configuration", expanded=True):
    # Allow testing date-based checks by overriding the current date used in payload 'as_of'
    use_system_now = st.toggle(
        "Use system date as current date",
        value=True,
        help="When off, provide a custom date to simulate 'today' for document recency checks."
    )
    if use_system_now:
        # Use real system date
        now = datetime.now()
        st.caption(f"Current date in use: {now.strftime('%Y-%m-%d')}")
    else:
        # Let the user write a custom current date (supports multiple formats via _safe_dt)
        custom_now_input = st.text_input(
            "Custom current date (e.g., 2025-09-05, 05/09/2025, or '21 de mayo de 2025')",
            value=now.strftime("%Y-%m-%d"),
            help="This date will be used as 'as_of' to evaluate the freshness of KYC/CSF/address evidence (Spanish inputs such as '21 de mayo de 2025' are supported)."
        )
        parsed = _safe_dt(custom_now_input)
        if isinstance(parsed, datetime):
            now = parsed
            st.caption(f"Current date in use: {now.strftime('%Y-%m-%d')}")
        else:
            st.error("Invalid date. Using system date instead.")
            now = datetime.now()
    col1, col2, col3 = st.columns(3)
    with col1:
        persona_label = st.selectbox(
            "Customer type",
            options=[label for label, _ in _persona_options],
            index=0,
            help="Select the customer's legal form."
        )
        persona_map = dict(_persona_options)
        persona = persona_map.get(persona_label, "PM")
        # Autofill customer country from KYC nationality (MX when Mexican)
        def _country_default_from_kyc() -> str:
            val = (st.session_state.cc_kyc_kv or {}).get("Nacionalidad")
            t = _normalize_text(val)
            if t and "mex" in t:
                return "MX"
            return "MX" if not t else t.upper()
        country = st.text_input("Customer country", value=_country_default_from_kyc(), help="Country code, default MX")
        role = st.selectbox("Current role", options=["analyst", "coordinator"], index=1)
    with col2:
        exceptions_sem = st.number_input("Semester exception count", min_value=0, value=0, step=1)
        has_overdue = st.toggle("Overdue flag", value=False)
        active_credit = st.toggle("Has active credit", value=False)
        # Investigation settings
        # Default MMR from Commercial Investigation extraction when available
        _default_mmr = float(st.session_state.get("cc_ic_mmr") or 0.0)
        mmr_amount_input = st.number_input(
            "MMR (Maximum Recommended Amount)", min_value=0.0, value=_default_mmr, step=10000.0,
            help="Maximum recommended amount from commercial investigation (MXN)"
        )
        # Default legal risk from summarized 'Factor_riesgo_empresa' (low/medium/high)
        legal_options = ["Low", "Medium", "High"]
        # Prefer summarized DOX key; fallback to UI display label
        default_risk_raw = (
            (st.session_state.get("cc_il_summary", {}) or {}).get("Factor_riesgo_empresa")
            or (st.session_state.cc_il_kv or {}).get("Factor de Riesgo de la Empresa")
        )
        nr = _normalize_text(default_risk_raw) if default_risk_raw else ""
        if "alto" in nr or "high" in nr:
            default_idx = 2
        elif ("medio" in nr) or ("mediano" in nr) or ("medium" in nr):
            default_idx = 1
        elif "bajo" in nr or ("low" in nr):
            default_idx = 0
        else:
            default_idx = 0
        legal_risk_display = st.selectbox(
            "Legal risk", options=legal_options, index=default_idx,
            help="Result of legal investigation"
        )
        _legal_risk_map = {"Low": "low", "Medium": "medium", "High": "high"}
        legal_risk_value = _legal_risk_map.get(legal_risk_display, None)
    with col3:
        entity_name = st.text_input("Entity Name (derives Group A/B)", value="Sample Foods S.A. de C.V.")
        # Auto-filled from CGV 'Fecha de firma'
        cgv_dt = st.session_state.get("cc_cgv_date")
        cgv_val = cgv_dt.strftime("%Y-%m-%d") if isinstance(cgv_dt, datetime) else (str(cgv_dt) if cgv_dt else "")
        st.text_input("CGV date (from CGV)", value=cgv_val, disabled=True)
        pagare_signed = st.toggle("Promissory note signed", value=True)
        guarantors = st.number_input("Number of guarantors", min_value=0, value=1, step=1)
        insurance_full = st.toggle("Insurance covers 100% of credit", value=False)

# --- Step 6: Build payload ---

payload_preview: Dict[str, Any] = {}

with st.expander("Review request payload", expanded=True):
    # Use persisted extracted values
    kyc_ss = st.session_state.cc_kyc_kv
    csf_ss = st.session_state.cc_csf_kv
    vendor_ss = st.session_state.cc_vendor_kv
    legal_name = (kyc_ss.get("Razón Social") or csf_ss.get("Denominación / Razón Social") or vendor_ss.get("Nombre del Cliente") or "")
    rfc_val = (kyc_ss.get("RFC") or csf_ss.get("RFC") or "")

    # Behavior flags sourced from toggles and excel
    adv_purchases = _parse_adv_purchases(request_inputs.get("advance_purchases_count", 0))

    # Use customer_id from Calificacion_pagos['Cuenta']; allow manual override from Key Information
    erp_id_override = (
        (st.session_state.get("cc_customer_id_override") or "")
        or (st.session_state.get("cc_edit_erp_id") or "")
    ).strip()
    customer_id_val = erp_id_override or (st.session_state.get("cc_customer_id") or "CUST-UNKNOWN")
    customer_id_val = str(customer_id_val).strip() or "CUST-UNKNOWN"

    sap_id_val = (st.session_state.get("cc_edit_sap_id") or "").strip()
    vendedor_val = (st.session_state.get("cc_edit_vendedor") or "").strip()
    aseguradora_val = (st.session_state.get("cc_edit_aseguradora") or "").strip()
    ubicacion_val = (st.session_state.get("cc_edit_ubicacion") or "").strip()
    rfc_override = (st.session_state.get("cc_edit_rfc") or "").strip()
    rfc_display = rfc_override or (rfc_val or "")

    customer = {
        "customer_id": customer_id_val,
        "legal_name": legal_name or "Unknown Customer",
        "persona": persona,
        "country": country,
        "entity_name": entity_name or None,
        "customer_group": None,
        "rfc": rfc_display or None,
        # Prefer CGV PDF signature date when available; fallback to AI_credito
        "cgv_signed_date": (
            (st.session_state.get("cc_cgv_date").isoformat() if isinstance(st.session_state.get("cc_cgv_date"), datetime) else None)
            or request_inputs.get("cgv_signed_date")
        ),
        "pagare_signed": bool(pagare_signed),
        "guarantors": int(guarantors),
        "insurance_full_credit": bool(insurance_full),
        "sap_id": sap_id_val or None,
        "seller_contact": vendedor_val or None,
        "insurance_provider_name": aseguradora_val or None,
        "display_address": ubicacion_val or None,
    }

    # Documents: bind KYC Fecha → kyc_date if present in extracted KYC
    kyc_fecha_dt = _safe_dt(kyc_ss.get("Fecha")) if isinstance(kyc_ss, dict) else None
    docs = {
        "kyc_date": kyc_fecha_dt.isoformat() if kyc_fecha_dt else None,
        "seller_comments_present": True if (vendor_pdf or vendor_ss) else False,
        "address_proof_date": (
            st.session_state.get("cc_csf_date").isoformat()
            if (
                isinstance(st.session_state.get("cc_csf_date"), datetime)
                and bool(st.session_state.get("cc_address_match"))
            )
            else None
        ),
        # Always use the CSF date for tax_cert_date when available
        "tax_cert_date": (st.session_state.get("cc_csf_date").isoformat() if isinstance(st.session_state.get("cc_csf_date"), datetime) else None),
    }

    # Currency: take from KYC 'Moneda' (fallback MXN)
    kyc_currency_raw = kyc_ss.get("Moneda") if isinstance(kyc_ss, dict) else None
    kyc_currency = str(kyc_currency_raw).strip().upper() if kyc_currency_raw else "MXN"

    # CreditRequest: use parsed values but override currency with KYC 'Moneda'
    credit_req = {
        "use_case": request_inputs.get("use_case", "update"),
        "requested_amount": float(request_inputs.get("requested_amount", 0.0)),
        "requested_currency": kyc_currency,
        "requested_terms_days": int(request_inputs.get("requested_terms_days", 30)),
        "last_update_date": request_inputs.get("last_update_date"),
        "current_credit_line": float(request_inputs.get("current_credit_line", 0.0)),
        "current_credit_currency": kyc_currency,
    }

    investigation = {
        "mmr_amount": float(mmr_amount_input) if 'mmr_amount_input' in locals() else None,
        "mmr_currency": kyc_currency,
        "legal_risk": legal_risk_value if 'legal_risk_value' in locals() else None,
        "external_investigation_date": None,
        "onsite_visit_done": False,
        # Include summarized Investigación Legal fields for downstream report generation
        "il_summary": st.session_state.get("cc_il_summary") or {},
    }

    behavior = {
        "invoices": invoices or [],
        "has_overdue_invoices": bool(has_overdue),
        "advance_purchases_count": adv_purchases,
        "has_active_credit": bool(active_credit),
        "exceptions_in_semester": int(exceptions_sem),
    }

    role_ctx = {"role": role}

    payload_preview = {
        "customer": customer,
        "docs": docs,
        "request": credit_req,
        "investigation": investigation,
        "behavior": behavior,
        "role": role_ctx,
        "as_of": now.isoformat(),
    }

    # Persist payload in session to survive reruns
    st.session_state["cc_payload_preview"] = payload_preview

    st.json(payload_preview)

    st.caption("Review and adjust configuration if any field looks incorrect. This payload matches the API schema in api/routers/credit_policy.py.")

st.markdown("---")

# --- Step 7: Evaluate ---

run_eval = st.button("Run Credit Evaluation", type="primary", use_container_width=True, disabled=(len(payload_preview) == 0))

result: Dict[str, Any] = st.session_state.get("cc_eval_result", {})
if run_eval:
    with st.spinner("Calling credit policy engine..."):
        resp = make_api_request("/credit/evaluate", method="POST", payload=payload_preview)
        if isinstance(resp, dict) and (resp.get("success") is False) and resp.get("error"):
            st.error(parse_api_error(resp))
        else:
            result = resp or {}
            # Persist evaluation result and payload used for report generation
            st.session_state["cc_eval_result"] = result
            st.session_state["cc_eval_payload"] = payload_preview

if result:
    st.subheader("Evaluation Results")
    # Quick KPIs
    try:
        scores = result.get("scores", {})
        checks = result.get("checks", {})
        decision = result.get("decision_hint", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Group", result.get("group", "?") )
        with col2:
            st.metric("CAL", scores.get("CAL", "?") )
        with col3:
            c3m = scores.get("C3M_pct")
            st.metric("C3M %", f"{c3m:.1f}%" if isinstance(c3m, (int, float)) and c3m is not None else "-")
        with col4:
            st.metric("Needs Director?", "Yes" if decision.get("needs_director") else "No")
    except Exception:
        pass

    # Payment analysis table (above the expanders)
    try:
        payload_used = st.session_state.get("cc_eval_payload") or payload_preview
        table_data = _build_payment_analysis_data(result, payload_used, now)
        html_table = _render_payment_analysis_table(table_data)
        components.html(html_table, height=420, scrolling=False)
    except Exception as e:
        st.caption(f"Could not build payment analysis table: {e}")

    # Detailed JSONs
    with st.expander("Scores", expanded=False):
        st.json(result.get("scores", {}))
    with st.expander("Checks", expanded=False):
        st.json(result.get("checks", {}))
    with st.expander("Decision Hint", expanded=False):
        st.json(result.get("decision_hint", {}))

    st.download_button(
        label="Download Result JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2, default=str),
        file_name="evaluate_result.json",
        mime="application/json",
        use_container_width=True,
        key="dl_result",
    )

    # --- Generate Credit Report (LLM) ---
    st.markdown("---")
    st.subheader("Generate Credit Report")
    st.caption("Uses AI to generate an executive report.")

    gen_report = st.button(
        "Generate Credit Report",
        type="primary",
        use_container_width=True,
        key="btn_credit_report",
    )

    if gen_report:
        with st.spinner("Generating report..."):
            try:
                # Build report payload from the last evaluated or preview payload
                _base_payload = (
                    st.session_state.get("cc_eval_payload")
                    or st.session_state.get("cc_payload_preview")
                    or payload_preview
                )
                # Do not mutate the original payload used for evaluation; create shallow copies
                _base_customer = dict((_base_payload.get("customer") or {}))
                # Inject Calificacion_pagos supplemental fields (prefer edited values)
                _ux_val = (
                    (st.session_state.get("cc_edit_usuario_cxc_1") or "").strip()
                    or (st.session_state.get("cc_usuario_cxc_1") or "").strip()
                )
                _vend_val = (
                    (st.session_state.get("cc_edit_vendedor_1") or "").strip()
                    or (st.session_state.get("cc_vendedor_1") or "").strip()
                )
                _base_customer["usuario_cxc_1"] = _ux_val or None
                _base_customer["vendedor_1"] = _vend_val or None
                report_payload = dict(_base_payload)
                report_payload["customer"] = _base_customer

                report_resp = make_api_request(
                    "/credit/report",
                    method="POST",
                    payload={
                        "payload": report_payload,
                        "evaluation": st.session_state.get("cc_eval_result") or result,
                        "language": "en",
                    },
                )
                if isinstance(report_resp, dict) and report_resp.get("success"):
                    st.session_state["cc_credit_report"] = report_resp.get("report", "")
                else:
                    # Prefer backend error message when available
                    err = None
                    if isinstance(report_resp, dict):
                        err = report_resp.get("error") or parse_api_error(report_resp)
                    st.error(err or "Failed to generate report")
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    if st.session_state.get("cc_credit_report"):
        st.markdown("**Report**")
        # Render Markdown with sanitization to prevent LaTeX math interpretation
        # The original text in session_state remains unchanged for PDF generation
        sanitized_report = sanitize_markdown_for_display(st.session_state["cc_credit_report"])
        st.markdown(sanitized_report, unsafe_allow_html=False)

        # Generate PDF from Markdown locally and expose as a download button
        try:
            pdf_bytes = markdown_to_pdf(st.session_state["cc_credit_report"])
            st.download_button(
                label="Download Report (PDF)",
                data=pdf_bytes,
                file_name="credit_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_credit_report_pdf",
            )
        except Exception as e:
            st.warning(f"Could not render PDF from Markdown: {e}")

# Sidebar help
with st.sidebar:
    st.header("How this works")
    st.markdown(
        """
        1. Upload the KYC, CSF, Vendor Comments, CGV, and both Excel files (Calificacion_pagos and AI_credito).
        2. Extract the PDFs to populate the key attributes.
        3. Configure customer type plus legal and financial investigation results.
        4. Run the credit evaluation to obtain scores and recommendations.
        5. Generate the credit report in PDF format.
        """
    )
