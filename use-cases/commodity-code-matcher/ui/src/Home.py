import os
import time

import pandas as pd
import streamlit as st

from src.utils import load_css_files
from src.api_client import download_output, healthcheck, run_extraction

APP_TITLE = "Commodity Code Pipeline"

# --- Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="static/images/SAP_logo_square.png",
    layout="wide",
)
st.logo("static/images/SAP_logo.svg")
st.title(APP_TITLE)

# --- CSS ---
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

# --- Sidebar ---
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. **Upload PDFs**: Use the file uploader to select one or more PDF documents
    2. **Process**: Click the "Run Classification and Extract Fields" button to start processing
    3. **Download**: Once processing is complete, download the Excel file with the extracted fields and assigned reference codes
    
    The application uses LLM extraction, HANA-hosted synthetic reference data, and verification to classify procurement documents.
    """)
    # st.header("API")
    # if st.button("Check API Health", use_container_width=True):
    #     result = healthcheck()
    #     if result.get("status") == "healthy":
    #         st.success(f"API is healthy (service={result.get('service')})")
    #     else:
    #         st.error(f"API issue: {result.get('error') or result}")

    # st.divider()
    # st.markdown("**Defaults**")
    # st.caption("llm_extraction=true, llm_verify=true, llm_model=gpt-4.1")
    # st.caption("Other parameters use API defaults.")


def _render_error(result: dict):
    """Render an API error payload in Streamlit.

    Args:
        result: Dictionary returned by the API client containing an ``error``
            value and optional HTTP ``status``.
    """

    error = result.get("error")
    status = result.get("status")
    if not error:
        return
    message = error.get("detail") if isinstance(error, dict) else error
    prefix = f"Status {status}: " if status else ""
    st.error(f"{prefix}{message}")


def _render_result(result: dict):
    """Render a completed extraction job in Streamlit.

    Args:
        result: Final job status payload returned by the polling API client.
    """

    if result.get("errors"):
        st.warning("Some issues were raised during processing.")
        for err in result.get("errors", []):
            st.write(f"- {err}")

    col1, col2 = st.columns(2)
    col1.metric("PDFs processed", result.get("file_count", 0))
    col2.metric("Top-K codes", result.get("top_k", 5))
    st.caption(f"Runtime: {result.get('runtime_seconds', 0):0.1f}s")

    line_items = result.get("line_items_preview", [])
    if line_items:
        preview_columns = [
            "file",
            "description",
            "LLM_Suggestion_Desc",
            "header_vendorName",
            "quantity",
            "unitPrice",
        ]
        column_display_names = {
            "file": "File",
            "description": "Item Description",
            "LLM_Suggestion_Desc": "Reference Code (AI-powered)",
            "header_vendorName": "Vendor",
            "quantity": "Quantity",
            "unitPrice": "Unit Price",
        }
        df = pd.DataFrame(line_items)
        available_columns = [col for col in preview_columns if col in df.columns]
        if available_columns:
            preview_df = df[available_columns]
            preview_df = preview_df.rename(columns=column_display_names)
            st.subheader("Extracted Field Information")
            st.dataframe(preview_df, use_container_width=True)

    job_id = result.get("job_id")
    if job_id and result.get("status") == "SUCCEEDED":
        excel_bytes = download_output(job_id)
        if excel_bytes:
            st.success("Reference codes assigned.")
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="commodity_codes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        else:
            st.error("Unable to download the output file.")


# --- Main content ---
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Files will be processed with llm_extraction + llm_verify.",
)

if st.button("Run Classification and Extract Fields", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        def _update_job_status(status_payload: dict) -> None:
            """Update the visible progress indicators from a polling payload.

            Args:
                status_payload: Job status payload from the backend.
            """

            progress = int(status_payload.get("progress") or 0)
            progress_bar.progress(min(max(progress, 0), 100))
            message = status_payload.get("message") or status_payload.get("status") or "Waiting for job status..."
            if status_payload.get("error"):
                status_placeholder.error(status_payload["error"])
            else:
                status_placeholder.info(message)

        status_placeholder.info("Submitting extraction job...")
        with st.spinner("Processing..."):
            result = run_extraction(uploaded_files, on_status=_update_job_status)

        if result.get("status") == "SUCCEEDED":
            progress_bar.progress(100)
        elif result.get("progress") is not None:
            progress_bar.progress(min(max(int(result.get("progress") or 0), 0), 100))
        status_placeholder.empty()

        if result.get("error"):
            _render_error(result)
        elif result.get("status") == "FAILED":
            st.error("Extraction failed.")
            for err in result.get("errors", []):
                st.write(f"- {err}")
        else:
            _render_result(result)

# If a previous result exists in session_state, we could render it here (optional future work).
