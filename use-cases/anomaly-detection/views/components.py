from __future__ import annotations

"""Reusable Streamlit components for the refactored application."""

from typing import Optional

import pandas as pd
import streamlit as st


def anomaly_status_card(status_text: str, background_color: str) -> None:
    st.markdown(
        f"""
        <div style="background-color: {background_color}; padding: 15px; margin: 10px 0; border-radius: 8px;">
            <h4>{status_text}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


def shap_table(display_df: pd.DataFrame, caption: Optional[str] = None) -> None:
    if caption:
        st.caption(caption)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
