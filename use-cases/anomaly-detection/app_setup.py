from __future__ import annotations

"""Shared Streamlit setup utilities used across the UI pages."""

import sys
from pathlib import Path
from typing import Iterable

import streamlit as st

# Resolve important project paths once so all pages can import shared modules.
UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parent
STATIC_DIR = UI_ROOT / "static"
STYLES_DIR = STATIC_DIR / "styles"
IMAGES_DIR = STATIC_DIR / "images"

# Import order matters here – Streamlit needs to know about the page config
# before any UI is rendered, and our service modules live relative to the UI.
for path in (UI_ROOT, PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

from services.data_loader import LoadedData, load_dataset  # noqa: E402
from utils import state  # noqa: E402

# Branding assets reused in every page.
APP_ICON = IMAGES_DIR / "SAP_logo_square.png"
APP_LOGO = IMAGES_DIR / "SAP_logo.svg"
CSS_FILES = (
    STYLES_DIR / "variables.css",
    STYLES_DIR / "style.css",
)
APP_TITLE = "Real-Time Self-Learning Sales Order Anomaly Pattern Detection"


def configure_page(page_title: str = APP_TITLE) -> None:
    """Apply the common page configuration and logo."""
    st.set_page_config(
        page_title=page_title,
        page_icon=str(APP_ICON),
        layout="wide",
    )
    st.logo(str(APP_LOGO))


def _load_css_files(file_paths: Iterable[Path]) -> None:
    """Inject CSS snippets from the provided files."""
    css_chunks = []
    for file_path in file_paths:
        if not file_path.exists():
            continue
        css_chunks.append(file_path.read_text(encoding="utf-8"))
    if css_chunks:
        st.markdown(f"<style>{''.join(css_chunks)}</style>", unsafe_allow_html=True)


def apply_base_theme() -> None:
    """Load the shared CSS theme used across all pages."""
    _load_css_files(CSS_FILES)


def initialize_state(current_tab: str | None = None) -> None:
    """Ensure URL parameters and tab metadata are in sync."""
    state.initialize_url_params()
    if current_tab is not None:
        st.session_state.current_tab = current_tab
        state.sync_url_params()


def show_url_warnings() -> None:
    """Display and clear any URL validation warnings."""
    messages = st.session_state.get("url_validation_messages") or []
    for message in messages:
        st.warning(f"URL Parameter Issue: {message}")
    if messages:
        st.session_state.url_validation_messages = []


def ensure_data_loaded() -> bool:
    """Load the shared dataset into session state if needed."""
    if "loaded_data" in st.session_state:
        return True

    try:
        with st.spinner("Loading application data..."):
            st.session_state.loaded_data = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return False

    return True


def get_loaded_data() -> LoadedData | None:
    """Return the cached dataset if it has been loaded."""
    return st.session_state.get("loaded_data")
