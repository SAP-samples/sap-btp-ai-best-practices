from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ui/ and project roots are available for imports when Streamlit runs the page directly.
CURRENT_FILE = Path(__file__).resolve()
UI_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = UI_ROOT.parent
for path in (UI_ROOT, PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

import streamlit as st

import app_setup
from views import single_order

PAGE_NAME = "Select a Sales Order"


def main() -> None:
    """Render the single-order analysis page."""
    app_setup.configure_page(f"{PAGE_NAME} · {app_setup.APP_TITLE}")
    app_setup.apply_base_theme()
    app_setup.initialize_state(current_tab=PAGE_NAME)
    app_setup.show_url_warnings()

    if not app_setup.ensure_data_loaded():
        return

    loaded = app_setup.get_loaded_data()
    if not loaded:
        st.error("Application data failed to load.")
        return

    st.title(app_setup.APP_TITLE)
    st.caption("Search or sample an order to explore anomaly drivers and explanations.")

    single_order.render_single_order_view(
        loaded.results,
        loaded.features,
        loaded.results_directory,
    )


if __name__ == "__main__":
    main()
