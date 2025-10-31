"""
Vendor Selection Assistant page.

Implements the new flow:
- Uses the sidebar material description (MAKTX) selection only
- Provides a button to obtain recommendations for the current material
- Shows the vendor comparison first, followed by AI recommendation
"""

import os
import time
import pandas as pd
import streamlit as st

from ui import components, vendor_analysis
from ui.content import descriptions
from ui.theme import apply_template_theme
from core import utils, data_loader
from optimization.profile_manager import ProfileManager


def load_context() -> tuple[ProfileManager, str]:
    """Initialize profiles and return (profile_manager, active_profile)."""
    profile_manager = ProfileManager(".")
    profile_manager.initialize_all_profiles()
    active_profile = profile_manager.get_active_profile()

    if "active_profile" not in st.session_state:
        st.session_state.active_profile = active_profile
    elif st.session_state.active_profile != active_profile:
        st.session_state.active_profile = active_profile
        st.session_state.pop("main_vendor_data", None)
        st.session_state.last_material_filters = ""
    return profile_manager, active_profile


def load_data_for_selected_material(active_profile: str, filters: dict, force_reload: bool = False) -> pd.DataFrame:
    """Load vendor data for the currently selected MAKTX only."""
    selected_maktx = filters["text_filters"].get("MAKTX", "All")
    if selected_maktx == "All":
        return pd.DataFrame()

    material_filters = {"MAKTX": selected_maktx}
    current_key = f"All_{selected_maktx}"
    if st.session_state.get("last_material_filters", "") != current_key or force_reload:
        df = data_loader.load_vendor_data(
            force_reload=force_reload, material_filters=material_filters, profile_id=active_profile
        )
        st.session_state.main_vendor_data = df if df is not None else pd.DataFrame()
        st.session_state.last_material_filters = current_key
    else:
        df = st.session_state.get("main_vendor_data", pd.DataFrame())
    return df if df is not None else pd.DataFrame()


def main() -> None:
    components.setup_page_config()
    apply_template_theme()
    components.apply_custom_styles()
    st.title("Vendor Selection Assistant")

    # Profiles and materials
    profile_manager, active_profile = load_context()
    all_materials = data_loader.get_all_materials_profile_aware(active_profile)

    # Sidebar filters (MATNR removed by components implementation)
    if "main_vendor_data" not in st.session_state:
        st.session_state.main_vendor_data = pd.DataFrame()
    filters = components.create_sidebar_filters(st.session_state.main_vendor_data, all_materials)

    # Load dataset for selected MAKTX
    df = load_data_for_selected_material(active_profile, filters)
    if df.empty:
        st.info("Please select a Material Description from the sidebar to begin analysis.")
        return

    # Apply the rest of the filters and load costs config
    df_filtered = utils.filter_dataframe(df, filters)
    costs_config = data_loader.load_costs_config(active_profile)
    components.display_sidebar_metrics(df_filtered)
    components.display_cost_config_status(costs_config)

    # Page intro: description + top metrics
    components.display_page_intro("Vendor Selection Assistant", df_filtered)

    st.markdown("---")
    st.subheader("AI-Powered Vendor Selection Assistant")

    selected_maktx = filters["text_filters"].get("MAKTX")
    material_df = df_filtered[df_filtered["MAKTX"] == selected_maktx].copy()

    if st.button("Obtain Supplier Recommendations for Current Material", type="primary"):
        if material_df.empty:
            st.warning("No vendors found for the selected material with current filters.")
        else:
            # Determine single MATNR (when unique) for display context
            matnr_value = None
            if "MATNR" in material_df.columns:
                unique_matnrs = material_df["MATNR"].dropna().unique()
                if len(unique_matnrs) == 1:
                    matnr_value = unique_matnrs[0]

            vendor_analysis.show_vendor_analysis(material_df, selected_maktx, None, costs_config, matnr=matnr_value)


if __name__ == "__main__":
    main()


