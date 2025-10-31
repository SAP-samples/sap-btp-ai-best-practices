"""
Theme helpers for the Streamlit multipage application.

This module centralizes loading the shared CSS from resources/static/styles
and rendering a consistent logo-top banner used across all pages.
"""

import os
import streamlit as st


def load_css_files(file_paths: list[str]) -> None:
    """Load and inject CSS styles from multiple files.

    The function concatenates the contents of the given CSS files and injects
    them into the current Streamlit page. This allows us to reuse the same
    theme across all multipage views.
    """
    css_chunks: list[str] = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                css_chunks.append(f.read())
        except FileNotFoundError:
            # Fail soft if a style file is missing to avoid breaking the UI
            continue

    if css_chunks:
        st.markdown(f"<style>{''.join(css_chunks)}</style>", unsafe_allow_html=True)


def apply_template_theme() -> None:
    """Apply the shared CSS theme and show the logo banner.

    - Injects `variables.css` and `style.css` from `resources/static/styles/`
    - Renders the SAP-style logo using `st.logo` if available, otherwise falls
      back to a small image banner at the top of the page.
    """
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    styles_dir = os.path.join(static_dir, "styles")
    images_dir = os.path.join(static_dir, "images")

    load_css_files([
        os.path.join(styles_dir, "variables.css"),
        os.path.join(styles_dir, "style.css"),
    ])

    # Render logo (Streamlit >=1.32 provides st.logo)
    logo_svg = os.path.join(images_dir, "SAP_logo.svg")
    try:
        st.logo(logo_svg)
    except Exception:
        # Fallback for older Streamlit versions
        if os.path.exists(logo_svg):
            st.image(logo_svg, width=140)


