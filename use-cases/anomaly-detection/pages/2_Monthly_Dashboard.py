from __future__ import annotations

import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
UI_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = UI_ROOT.parent
for path in (UI_ROOT, PROJECT_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)

import streamlit as st

import app_setup
from views import monthly_dashboard

PAGE_NAME = "Monthly Dashboard"


def main() -> None:
    """Render the aggregated anomaly dashboard page."""
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
    st.caption("Track monthly anomaly trends, drill into daily performance, and sync filters with the URL.")

    monthly_dashboard.render_monthly_dashboard(
        loaded.results,
        loaded.features,
    )


if __name__ == "__main__":
    main()
