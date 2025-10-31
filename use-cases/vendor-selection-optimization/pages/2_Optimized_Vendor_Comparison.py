"""
Optimized Vendor Comparison page.

Loads data for the selected material description and renders the
optimization-based comparison using existing standalone renderer.
"""

import pandas as pd
import streamlit as st

from ui import components
from ui.content import descriptions
from ui.theme import apply_template_theme
from core import utils, data_loader
from optimization.profile_manager import ProfileManager
from optimization import optimized_vendor_selection


def _load_context() -> tuple[ProfileManager, str]:
    pm = ProfileManager(".")
    pm.initialize_all_profiles()
    active = pm.get_active_profile()
    if "active_profile" not in st.session_state:
        st.session_state.active_profile = active
    elif st.session_state.active_profile != active:
        st.session_state.active_profile = active
        st.session_state.pop("main_vendor_data", None)
        st.session_state.last_material_filters = ""
    return pm, active


def _load_df(active_profile: str, filters: dict) -> pd.DataFrame:
    selected_maktx = filters["text_filters"].get("MAKTX", "All")
    if selected_maktx == "All":
        return pd.DataFrame()
    key = f"All_{selected_maktx}"
    if st.session_state.get("last_material_filters") != key:
        df = data_loader.load_vendor_data(
            force_reload=False, material_filters={"MAKTX": selected_maktx}, profile_id=active_profile
        )
        st.session_state.main_vendor_data = df if df is not None else pd.DataFrame()
        st.session_state.last_material_filters = key
    return st.session_state.get("main_vendor_data", pd.DataFrame())


def main() -> None:
    components.setup_page_config()
    apply_template_theme()
    components.apply_custom_styles()
    st.title("Optimized Vendor Comparison")

    pm, active_profile = _load_context()
    all_materials = data_loader.get_all_materials_profile_aware(active_profile)
    if "main_vendor_data" not in st.session_state:
        st.session_state.main_vendor_data = pd.DataFrame()
    filters = components.create_sidebar_filters(st.session_state.main_vendor_data, all_materials)

    df = _load_df(active_profile, filters)
    if df.empty:
        st.info("Please select a Material Description from the sidebar to view optimized comparisons.")
        return

    df_filtered = utils.filter_dataframe(df, filters)
    # Page intro: description + top metrics
    components.display_page_intro("Optimized Vendor Comparison", df_filtered)
    with st.spinner("Loading optimization data and generating visualizations..."):
        optimized_vendor_selection.render_optimized_selection_tab_standalone(df_filtered, filters=filters)


if __name__ == "__main__":
    main()


