"""
Unified Settings Page for AI Supplier Sourcing Optimizer.

This module provides a unified interface for both Tariff Configuration and 
Economic Impact Parameter Settings (Optimization Settings).
"""

import streamlit as st
from optimization import tariff_configuration, optimization_settings
from optimization.profile_manager import ProfileManager


def render_profile_selection():
    """Render the optimization profile selection interface."""
    # Initialize ProfileManager
    profile_manager = ProfileManager(".")
    
    # Get current active profile
    if 'active_profile' not in st.session_state:
        st.session_state['active_profile'] = profile_manager.get_active_profile()
    
    # Get list of profiles
    profiles = profile_manager.get_profile_list()
    
    st.markdown("---")
    st.subheader("Optimization Profiles")
    st.markdown("Choose a profile to load pre-configured optimization settings:")
    
    # Create profile buttons in a row
    cols = st.columns(5)
    
    for i, profile in enumerate(profiles):
        with cols[i]:
            profile_id = profile['profile_id']
            profile_name = profile['name']
            
            # Determine button style based on active profile
            if profile_id == st.session_state['active_profile']:
                # Active profile - use primary button style
                button_label = f"{profile_name}"
                button_type = "primary"
            else:
                # Inactive profile - use secondary button style
                button_label = f"{profile_name}"
                button_type = "secondary"
            
            # Create button
            if st.button(button_label, key=f"profile_btn_{profile_id}", type=button_type):
                # Switch to selected profile
                st.session_state['active_profile'] = profile_id
                profile_manager.set_active_profile(profile_id)
                st.rerun()
    
    # Display current active profile info
    active_profile_info = next(p for p in profiles if p['profile_id'] == st.session_state['active_profile'])
    st.info(f"Active Profile: **{active_profile_info['name']}** - {active_profile_info['description']}")
    
    # Profile management expander
    with st.expander("Profile Management", expanded=False):
        # Profile renaming
        st.subheader("Rename Profile")
        selected_profile = st.selectbox(
            "Select profile to rename:",
            options=[p['profile_id'] for p in profiles],
            format_func=lambda x: next(p['name'] for p in profiles if p['profile_id'] == x)
        )
        
        new_name = st.text_input(
            "New name:",
            value=next(p['name'] for p in profiles if p['profile_id'] == selected_profile),
            key="profile_rename_input"
        )
        
        if st.button("Rename Profile", key="rename_profile_btn"):
            if new_name and new_name.strip():
                profile_manager.update_profile_name(selected_profile, new_name.strip())
                st.success(f"Profile renamed to: {new_name.strip()}")
                st.rerun()
            else:
                st.error("Please enter a valid name.")
    
    st.markdown("---")


def render_settings_page():
    """Render the unified Settings page with two tabs"""
    st.subheader("Settings")
    
    st.markdown("""
    Configure tariff values and economic impact parameters to customize the procurement optimization model.
    Use the tabs below to access different configuration options.
    """)
    
    # Render Profile Selection Interface
    render_profile_selection()
    
    # Create two tabs
    tab1, tab2 = st.tabs(["Tariffs Settings", "Economic Impact Parameter Settings"])
    
    # Get active profile for configuration
    active_profile = st.session_state.get('active_profile', 'profile_1')
    
    # Render Tariff Configuration in the first tab
    with tab1:
        # Call the existing tariff configuration rendering function
        # We pass the tab context and active profile
        tariff_configuration.render_tariff_configuration_tab(tab1, active_profile)
    
    # Render Economic Impact Parameter Settings in the second tab
    with tab2:
        # Call the existing optimization settings rendering function
        # We pass the tab context and active profile
        optimization_settings.render_optimization_settings_tab(tab2, active_profile)


def render_settings_page_standalone():
    """Standalone version for lazy loading from main.py"""
    render_settings_page()