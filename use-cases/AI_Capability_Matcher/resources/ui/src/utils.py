"""
Utility functions for the Streamlit application.
"""

import streamlit as st


def load_css_files(file_paths: list[str]):
    """Load and inject CSS styles from multiple files."""
    full_css = ""
    for file_path in file_paths:
        with open(file_path, "r") as f:
            full_css += f.read()
    st.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)
