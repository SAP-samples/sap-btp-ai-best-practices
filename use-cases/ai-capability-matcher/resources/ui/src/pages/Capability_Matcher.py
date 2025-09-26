"""
Streamlit comparator page that mimics the template styling and calls the backend
`/api/match` endpoint to match two CSV datasets by embeddings and LLM ranking.
"""

import io
import os
from typing import List

import pandas as pd
import streamlit as st

from src.utils import load_css_files
from src.api_client import make_api_request


APP_TITLE = "AI Capability Matcher"


def _concat_row(row: pd.Series, columns: List[str]) -> str:
    """Create text from selected columns (mirrors backend)."""

    parts = []
    for col in columns:
        val = row.get(col)
        if pd.isna(val):
            continue
        parts.append(f"{col}: {val}")
    return " - ".join(parts)


def main():
    """Render comparator UI page with template styling."""

    # Page config & styling
    st.set_page_config(
        page_title=APP_TITLE, page_icon="static/images/SAP_logo_square.png", layout="wide"
    )
    st.logo("static/images/SAP_logo.svg")

    css_files = [
        os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "variables.css"),
        os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "style.css"),
    ]
    load_css_files(css_files)

    st.title(APP_TITLE)
    st.markdown(
        "Upload base AI product catalog and client product data. Select the columns to generate text, then run matching."
    )

    # Inputs
    col_left, col_right = st.columns(2)
    with col_left:
        ai_file = st.file_uploader("Upload AI Catalog CSV", type=["csv"])
    with col_right:
        client_file = st.file_uploader("Upload Client CSV", type=["csv"])

    if not ai_file or not client_file:
        st.info("Please upload both CSV files to begin.")
        return

    df_ai = pd.read_csv(ai_file)
    df_client = pd.read_csv(client_file)

    # Make DataFrames JSON-safe for backend payload (convert NaN/NA -> None)
    df_ai = df_ai.where(pd.notnull(df_ai), None)
    df_client = df_client.where(pd.notnull(df_client), None)

    st.subheader("Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**AI Catalog (head)**")
        st.dataframe(df_ai.head(), use_container_width=True)
    with c2:
        st.markdown("**Client Data (head)**")
        st.dataframe(df_client.head(), use_container_width=True)

    st.subheader("Column Selection")
    ai_cols = list(df_ai.columns)
    client_cols = list(df_client.columns)
    selected_ai_cols = st.multiselect(
        "Select columns from AI catalog for text", options=ai_cols, default=ai_cols
    )
    selected_client_cols = st.multiselect(
        "Select columns from Client for text", options=client_cols, default=client_cols
    )

    matching_column = st.selectbox(
        "AI catalog display column (name in results)", options=ai_cols, index=0 if ai_cols else None
    )

    st.subheader("Settings")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        num_matches = st.slider("Matches per row", 1, 10, 5)
    with col_b:
        batch_size = st.slider("Batch size", 1, 10, 5)
    with col_c:
        use_llm = st.checkbox("Use LLM for ranking & reasoning", value=True)
    with col_d:
        read_timeout_minutes = st.number_input(
            "Timeout (minutes)", min_value=1, max_value=240, value=15, step=1,
            help="Maximum time to wait for the backend to complete processing large datasets."
        )

    batch_system_prompt = st.text_area(
        "Custom batch system prompt (optional)", value="", height=150
    )

    if st.button("Run Matching", type="primary"):
        # Prepare constant payload parts and UI elements for progress reporting
        # - We batch client rows and call the backend per batch to surface progress
        ai_rows = df_ai.to_dict(orient="records")
        total_clients = len(df_client)
        chunk_size = int(batch_size)

        # Progress UI
        progress = st.progress(0, text="Starting matching...")
        status = st.empty()

        # Output dataframe initialized from client data; columns added after first response
        df_out = df_client.copy()
        result_columns = None

        # Convert timeout minutes to seconds and pass as (connect, read)
        # Keep connect timeout short (10s) but allow a long read timeout per batch.
        timeout_seconds = float(read_timeout_minutes) * 60.0

        try:
            # Iterate over client rows in chunks to drive the progress bar
            for start in range(0, total_clients, chunk_size):
                end = min(start + chunk_size, total_clients)

                # Prepare batch-specific payload: only send the client subset
                client_rows_batch = df_client.iloc[start:end].to_dict(orient="records")
                payload = {
                    "ai_rows": ai_rows,
                    "client_rows": client_rows_batch,
                    "selected_ai_columns": selected_ai_cols,
                    "selected_client_columns": selected_client_cols,
                    "matching_column": matching_column,
                    "num_matches": int(num_matches),
                    "batch_size": int(len(client_rows_batch)),
                    "use_llm": bool(use_llm),
                    "batch_system_prompt": batch_system_prompt or None,
                }

                # Call backend for this batch
                resp = make_api_request(
                    "/api/match", payload=payload, timeout=(10.0, timeout_seconds)
                )
                if not resp or not resp.get("success"):
                    st.error(resp.get("error", f"Matching failed at rows {start}-{end-1}"))
                    break

                # Initialize result columns once from first response
                if result_columns is None:
                    result_columns = resp.get("result_columns", [])
                    for col in result_columns:
                        df_out[col] = None

                # Map batch results back to global indices
                per_client = resp.get("matches", [])
                for local_idx, row in enumerate(per_client):
                    global_idx = start + local_idx
                    results = row.get("results", {}) if isinstance(row, dict) else {}
                    for col in result_columns:
                        df_out.at[global_idx, col] = results.get(col)

                # Update progress bar and status text
                completed = end
                fraction = completed / max(total_clients, 1)
                progress.progress(min(max(int(fraction * 100), 0), 100), text=f"Processed {completed} / {total_clients} rows")
                status.caption(f"Completed batches: {end // chunk_size}{'+' if end % chunk_size else ''} / {((total_clients + chunk_size - 1) // chunk_size)}")

            else:
                # Executed only if loop wasn't broken -> success
                st.success("Matching completed")
                st.dataframe(df_out.head(), use_container_width=True)

                # Download button
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                out_name = f"{os.path.splitext(client_file.name)[0]}_matched_{os.path.splitext(ai_file.name)[0]}.csv"
                st.download_button(
                    "Download Results CSV", data=csv_bytes, file_name=out_name, mime="text/csv"
                )
        finally:
            # Ensure progress UI is completed or cleared appropriately
            progress.progress(100, text="Done")


main()


