"""
Utility helpers for the Streamlit UI.
"""

import os
import streamlit as st


def load_css_files(file_paths: list[str]):
    """Load and inject CSS styles from multiple files."""
    full_css = ""
    for file_path in file_paths:
        with open(file_path, "r") as f:
            full_css += f.read()
    st.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)


def get_api_base_url() -> str:
    """Resolve the API base URL from env or defaults."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")
