"""
Settings page wrapper for the multipage app.
"""

import pandas as pd
import streamlit as st

from ui import components
from ui.theme import apply_template_theme
from ui.pages import settings_page
from ui.content import descriptions
from core import utils, data_loader
from optimization.profile_manager import ProfileManager


def main() -> None:
    components.setup_page_config()
    apply_template_theme()
    components.apply_custom_styles()

    # Load filters and, if selected, show page intro and metrics
    pm = ProfileManager(".")
    pm.initialize_all_profiles()
    active_profile = pm.get_active_profile()

    # Sidebar filters
    if "main_vendor_data" not in st.session_state:
        st.session_state.main_vendor_data = pd.DataFrame()
    all_materials = data_loader.get_all_materials_profile_aware(active_profile)
    filters = components.create_sidebar_filters(st.session_state.main_vendor_data, all_materials)

    selected_maktx = filters["text_filters"].get("MAKTX", "All")
    if selected_maktx != "All":
        key = f"All_{selected_maktx}"
        if st.session_state.get("last_material_filters") != key:
            df = data_loader.load_vendor_data(
                material_filters={"MAKTX": selected_maktx}, profile_id=active_profile
            )
            st.session_state.main_vendor_data = df if df is not None else pd.DataFrame()
            st.session_state.last_material_filters = key
        df = st.session_state.get("main_vendor_data", pd.DataFrame())
        df_filtered = utils.filter_dataframe(df, filters)
        components.display_page_intro("Settings", df_filtered)

    settings_page.render_settings_page_standalone()


if __name__ == "__main__":
    main()


