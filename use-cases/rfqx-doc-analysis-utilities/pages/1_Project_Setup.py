"""
Project setup and management page.

Replaces the original Streamlit tab with a dedicated page that mirrors the SAP
template styling while keeping the existing project lifecycle behaviour.
"""

from __future__ import annotations

import streamlit as st

from app_context import (
    apply_template_theme,
    create_combined_graph_visualization,
    ensure_session_state,
    get_project_manager,
)

apply_template_theme()
ensure_session_state()

project_manager = get_project_manager()

st.title("Project Setup & Management")
st.caption(
    "Create new RFQ analysis projects or continue working with your existing ones."
)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create New Project")

        new_project_name = st.text_input(
            "Project Name",
            placeholder="Enter a descriptive name for your project",
            help="Use alphanumeric characters, spaces, hyphens, and underscores only.",
        )

        if st.button(
            "Create Project",
            type="primary",
            disabled=not new_project_name,
            use_container_width=True,
        ):
            try:
                project_manager.create_project(new_project_name)
                ensure_session_state()  # Reset to defaults for new project
                st.session_state.current_project = new_project_name
                st.session_state.project_loaded = True

                st.success(f"Project '{new_project_name}' created successfully!")
                st.info(
                    "Proceed to the Process Supplier Documents page to upload and analyze documents."
                )

                project_manager.save_project(new_project_name, st.session_state)
                st.rerun()
            except ValueError as err:
                st.error(str(err))
            except Exception as err:  # pragma: no cover - defensive for runtime issues
                st.error(f"Failed to create project: {err}")

    with col2:
        st.subheader("Load Existing Project")

        projects = project_manager.list_projects()

        if projects:
            project_names = [p["project_name"] for p in projects]
            selected_project = st.selectbox(
                "Select Project",
                options=[""] + project_names,
                format_func=lambda value: "-- Select a project --" if value == "" else value,
            )

            if selected_project:
                try:
                    project_info = project_manager.get_project_info(selected_project)
                except Exception as err:  # pragma: no cover
                    st.error(f"Error loading project details: {err}")
                else:
                    with st.expander("Project Details", expanded=True):
                        st.write(f"**Created:** {project_info['created_at'][:10]}")
                        st.write(f"**Last Modified:** {project_info['last_modified'][:10]}")

                        if project_info["providers"]:
                            st.write("**Suppliers:**")
                            for provider in project_info["providers"]:
                                st.write(
                                    f"- {provider['name']} ({provider['file_count']} files)"
                                )
                                for provider_file in provider.get("files", []):
                                    st.write(f"    {provider_file}")
                        else:
                            st.write("_No suppliers uploaded yet._")

                        if project_info.get("has_analysis"):
                            st.write("**Analysis report available.**")
                        if project_info.get("has_graph"):
                            st.write("**Knowledge graph available.**")

                    col_load, col_delete = st.columns(2)

                    with col_load:
                        if st.button(
                            "Load Project",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                project_state = project_manager.load_project(selected_project)
                                for key, value in project_state.items():
                                    if key not in {"comparison_report", "graph_image"}:
                                        st.session_state[key] = value

                                if "comparison_report" in project_state:
                                    st.session_state.comparison_report = project_state[
                                        "comparison_report"
                                    ]
                                if "graph_image" in project_state:
                                    st.session_state.graph_image = project_state["graph_image"]
                                    if (
                                        st.session_state.get("providers")
                                        and any(
                                            provider.get("extracted_data")
                                            for provider in st.session_state.providers
                                        )
                                    ):
                                        completed_providers = [
                                            provider
                                            for provider in st.session_state.providers
                                            if provider.get("extracted_data")
                                        ]
                                        providers_data = {
                                            provider["name"]: provider["extracted_data"]
                                            for provider in completed_providers
                                        }
                                        _, interactive_graph = create_combined_graph_visualization(
                                            providers_data
                                        )
                                        st.session_state.interactive_graph = interactive_graph

                                st.session_state.current_project = selected_project
                                st.session_state.project_loaded = True

                                st.success(f"Project '{selected_project}' loaded successfully!")
                                st.info(
                                    "Use the navigation links in the sidebar to continue your analysis."
                                )
                                st.rerun()
                            except Exception as err:
                                st.error(f"Failed to load project: {err}")

                    with col_delete:
                        if st.button(
                            "Delete Project",
                            type="secondary",
                            use_container_width=True,
                        ):
                            st.session_state[f"confirm_delete_{selected_project}"] = True

                    if st.session_state.get(f"confirm_delete_{selected_project}", False):
                        st.warning(
                            f"Are you sure you want to delete project '{selected_project}'? This action cannot be undone."
                        )
                        confirm_col, cancel_col = st.columns(2)

                        with confirm_col:
                            if st.button("Yes, Delete", type="primary"):
                                try:
                                    project_manager.delete_project(selected_project)
                                    st.success(f"Project '{selected_project}' deleted successfully!")
                                    if st.session_state.current_project == selected_project:
                                        ensure_session_state()
                                        st.session_state.current_project = None
                                        st.session_state.project_loaded = False
                                    del st.session_state[f"confirm_delete_{selected_project}"]
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Failed to delete project: {err}")

                        with cancel_col:
                            if st.button("Cancel"):
                                del st.session_state[f"confirm_delete_{selected_project}"]
                                st.rerun()
        else:
            st.info("No existing projects found. Create a new project to get started!")

st.divider()

if st.session_state.current_project:
    st.success(f"**Current Project:** {st.session_state.current_project}")
    if st.session_state.project_loaded and st.session_state.providers:
        if st.button("Save Current Progress", type="secondary"):
            try:
                project_manager.save_project(st.session_state.current_project, st.session_state)
                st.success("Project saved successfully!")
            except Exception as err:
                st.error(f"Failed to save project: {err}")
else:
    st.info("Create a project or load an existing one to begin.")
