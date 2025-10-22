"""
SAP RFQx Document Analysis – Streamlit entry point.

This home page adopts the SAP PoC template styling while routing users to the
dedicated feature pages that replaced the original tabbed layout.
"""

from __future__ import annotations
from datetime import datetime
from typing import List

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from app_context import apply_template_theme, ensure_session_state, get_project_manager


APP_TITLE = "SAP RFQx Document Analysis"

# Configure page once; other pages should not call set_page_config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✅",
    layout="wide",
)

apply_template_theme()

ensure_session_state()
project_manager = get_project_manager()

st.title(APP_TITLE)
st.caption(
    "Analyze, compare, and collaborate on RFQ documents with unified styling "
    "across the SAP Streamlit + FastAPI experience."
)

def _project_summary() -> List[str]:
    """Collect metadata about existing projects for quick display."""
    try:
        projects = project_manager.list_projects()
    except Exception:
        return []

    summary = []
    for project in projects:
        name = project.get("project_name", "Unnamed Project")
        updated = project.get("last_modified", "")[:10]
        providers = project.get("providers", [])
        provider_count = len(providers)
        summary.append(f"{name} · {provider_count} suppliers · updated {updated}")
    return summary


with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Get Started")
        st.write(
            "Navigate through the dedicated pages to manage projects, process suppliers, "
            "compare proposals, and chat with documents. The original Streamlit tabs "
            "have been reimagined as focused experiences that align with the SAP PoC "
            "template."
        )
        st.page_link(
            "pages/1_Project_Setup.py",
            label="**Project Setup & Management**",
        )
        st.page_link(
            "pages/2_Process_Documents.py",
            label="**Process Supplier Documents**",
        )
        st.page_link(
            "pages/3_Compare_Providers.py",
            label="**Compare Suppliers**",
        )
        st.page_link(
            "pages/4_Rfq_Recommender.py",
            label="**RFQ Insights & Recommendations**",
        )
        st.page_link(
            "pages/5_Supplier_Chat.py",
            label="**Supplier Document Chat**",
        )
    with col2:
        st.subheader("Current Session")
        current_project = st.session_state.get("current_project")
        if current_project:
            st.success(f"Active project: **{current_project}**")
        else:
            st.info("No project loaded. Visit Project Setup to begin.")

        providers = st.session_state.get("providers", [])
        processed = [
            provider for provider in providers if provider.get("status") == "completed"
        ]
        st.metric("Suppliers Configured", len(providers))
        st.metric("Suppliers Processed", len(processed))

st.divider()

st.subheader("Recent Projects")
summaries = _project_summary()
if summaries:
    for summary in summaries[:5]:
        st.write(f"- {summary}")
    if len(summaries) > 5:
        st.caption(f"+ {len(summaries) - 5} more projects available")
else:
    st.write("No saved projects yet. Create one from the Project Setup page.")

st.divider()

st.subheader("Release Notes")
st.write(
    f"""
- **{datetime.now():%Y-%m-%d}** · Adopted SAP PoC template styling and switched to a multi-page layout.
- Deprecated the advanced graph explorer page as part of the streamlined navigation.
- Backend processing, comparison, and chat capabilities remain unchanged.
"""
)
