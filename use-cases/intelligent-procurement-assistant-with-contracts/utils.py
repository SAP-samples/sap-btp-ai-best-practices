# utils.py
import streamlit as st
import pandas as pd
import os
from typing import Dict, Any  # Asegúrate de importar esto si safe_dataframe lo usa


def safe_dataframe(df, use_container_width=True, hide_index=True, **kwargs):
    """
    Compatible dataframe function that works with both old and new Streamlit versions
    """
    try:
        # Try new method first (Streamlit 1.25+)
        return st.dataframe(
            df, use_container_width=use_container_width, hide_index=hide_index, **kwargs
        )
    except TypeError:
        # Fallback for old versions - use pandas styling
        if hide_index:
            try:
                styled_df = df.style.hide(axis="index")
                return st.dataframe(
                    styled_df, use_container_width=use_container_width, **kwargs
                )
            except:
                try:
                    styled_df = df.style.hide_index()
                    return st.dataframe(
                        styled_df, use_container_width=use_container_width, **kwargs
                    )
                except Exception as e:
                    st.warning(f"Could not hide index for old Streamlit: {e}")
                    return st.dataframe(
                        df, use_container_width=use_container_width, **kwargs
                    )
        else:
            return st.dataframe(df, use_container_width=use_container_width, **kwargs)


def load_css_files(file_paths: list[str]):
    """Load and inject CSS styles from multiple files."""
    full_css = ""
    for file_path in file_paths:
        with open(file_path, "r") as f:
            full_css += f.read()
    st.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)


def load_catalog_file(catalog_filename: str, debug: bool = False):
    """
    Load catalog PDF file with robust path handling for deployment environments.

    Args:
        catalog_filename: Name of the catalog file (e.g., "GlobalTech_Supplies_Product_Catalog.pdf")
        debug: Whether to print debug information

    Returns:
        tuple: (success: bool, data: bytes or None, error_message: str or None)
    """
    if not catalog_filename or not catalog_filename.strip():
        return False, None, "No catalog filename provided"

    # Try multiple path strategies
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Strategy 1: Relative to script directory
    path_attempts = [
        os.path.join(script_dir, "static", "pdf", "catalogs", catalog_filename),
        os.path.join("static", "pdf", "catalogs", catalog_filename),  # Relative path
        os.path.join(
            os.getcwd(), "static", "pdf", "catalogs", catalog_filename
        ),  # Current working directory
    ]

    for i, catalog_path in enumerate(path_attempts):
        if debug:
            print(f"DEBUG: Attempt {i+1}: Trying path: {catalog_path}")
            print(f"DEBUG: Path exists: {os.path.exists(catalog_path)}")

        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                if debug:
                    print(
                        f"DEBUG: Successfully loaded {len(pdf_data)} bytes from {catalog_path}"
                    )
                return True, pdf_data, None
            except Exception as e:
                error_msg = f"Failed to read file: {str(e)}"
                if debug:
                    print(f"DEBUG: {error_msg}")
                return False, None, error_msg

    # If all attempts failed, provide debug info
    if debug:
        print(f"DEBUG: Script directory: {script_dir}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: All path attempts failed for: {catalog_filename}")

    return False, None, f"File not found: {catalog_filename}"


def create_catalog_download_button(
    pdf_data: bytes, filename: str, key: str, label: str = None
):
    """
    Create a simple download link that works reliably across all deployments.
    Uses only the direct download link approach for maximum compatibility.
    """
    import streamlit as st
    import base64

    # Use filename as label if no label provided
    if label is None:
        label = filename

    # Encode the PDF data as base64
    b64_data = base64.b64encode(pdf_data).decode()

    # Create a simple download link
    st.markdown(
        f"<a href='data:application/pdf;base64,{b64_data}' download='{filename}'>{label}</a>",
        unsafe_allow_html=True,
    )


def get_file_debug_info(catalog_filename: str = None):
    """
    Get comprehensive debug information about file paths and directory structure.
    This is useful for troubleshooting deployment issues.
    """
    import streamlit as st

    debug_info = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))

    debug_info["script_directory"] = script_dir
    debug_info["current_working_directory"] = os.getcwd()
    debug_info["python_path"] = os.path.dirname(os.path.abspath(__file__))

    # Check if static directory exists
    static_dir = os.path.join(script_dir, "static")
    debug_info["static_dir_exists"] = os.path.exists(static_dir)
    debug_info["static_dir_path"] = static_dir

    # Check if catalogs directory exists
    catalogs_dir = os.path.join(script_dir, "static", "pdf", "catalogs")
    debug_info["catalogs_dir_exists"] = os.path.exists(catalogs_dir)
    debug_info["catalogs_dir_path"] = catalogs_dir

    # List files in catalogs directory if it exists
    if debug_info["catalogs_dir_exists"]:
        try:
            debug_info["catalog_files"] = os.listdir(catalogs_dir)
        except Exception as e:
            debug_info["catalog_files_error"] = str(e)

    # If specific filename provided, check its existence
    if catalog_filename:
        debug_info["requested_file"] = catalog_filename
        possible_paths = [
            os.path.join(script_dir, "static", "pdf", "catalogs", catalog_filename),
            os.path.join("static", "pdf", "catalogs", catalog_filename),
            os.path.join(os.getcwd(), "static", "pdf", "catalogs", catalog_filename),
        ]

        debug_info["file_path_attempts"] = []
        for path in possible_paths:
            debug_info["file_path_attempts"].append(
                {"path": path, "exists": os.path.exists(path)}
            )

    return debug_info
