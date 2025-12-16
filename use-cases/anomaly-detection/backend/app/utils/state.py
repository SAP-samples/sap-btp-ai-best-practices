from __future__ import annotations

"""Session state and URL parameter helpers."""

from typing import Dict, List

import streamlit as st

TAB_OPTIONS = ["Select a Sales Order", "Monthly Dashboard", "Fine Tuning"]
TAB_MAPPING = {
    "SingleSOView": "Select a Sales Order",
    "DailyView": "Monthly Dashboard",
    "FineTuning": "Fine Tuning",
}
DISPLAY_TO_URL = {v: k for k, v in TAB_MAPPING.items()}


def initialize_url_params() -> None:
    """Initialize session state from URL parameters."""
    if "url_validation_messages" not in st.session_state:
        st.session_state.url_validation_messages = []

    if "current_tab" not in st.session_state:
        url_tab = st.query_params.get("tab", "")
        if url_tab and url_tab in TAB_MAPPING:
            st.session_state.current_tab = TAB_MAPPING[url_tab]
        elif url_tab and url_tab not in TAB_MAPPING:
            st.session_state.url_validation_messages.append(
                f"Invalid tab parameter '{url_tab}'. Valid options: SingleSOView, DailyView, FineTuning"
            )
            st.session_state.current_tab = TAB_OPTIONS[0]
        else:
            st.session_state.current_tab = TAB_OPTIONS[0]

    _initialize_numeric_param("url_doc_num", "DocNum")
    _initialize_numeric_param("url_item_num", "ItemNum")
    _initialize_numeric_param("url_year", "year", min_val=2020, max_val=2030)
    _initialize_numeric_param("url_month", "month", min_val=1, max_val=12)
    _initialize_numeric_param("url_day", "day", min_val=1, max_val=31)


def _initialize_numeric_param(state_key: str, query_key: str, min_val: int = 1, max_val: int | None = None) -> None:
    value = st.query_params.get(query_key, "")
    if value:
        try:
            int_value = int(value)
            is_below_min = (min_val is not None) and (int_value < min_val)
            is_above_max = (max_val is not None) and (int_value > max_val)
            if is_below_min or is_above_max:
                # Build a friendly validation message depending on provided bounds
                if max_val is None and min_val is not None:
                    msg = f"{query_key} must be >= {min_val}, got '{value}'"
                elif max_val is not None and min_val is not None:
                    msg = f"{query_key} must be between {min_val} and {max_val}, got '{value}'"
                else:
                    msg = f"Invalid {query_key} '{value}'"
                st.session_state.url_validation_messages.append(msg)
                st.session_state[state_key] = None if "year" in state_key or "month" in state_key or "day" in state_key else ""
            else:
                st.session_state[state_key] = int_value if "year" in state_key or "month" in state_key or "day" in state_key else str(int_value)
        except (ValueError, TypeError):
            st.session_state.url_validation_messages.append(f"Invalid {query_key} '{value}'. Must be a valid integer.")
            st.session_state[state_key] = None if "year" in state_key or "month" in state_key or "day" in state_key else ""
    else:
        st.session_state[state_key] = None if "year" in state_key or "month" in state_key or "day" in state_key else ""


def sync_url_params() -> None:
    """Synchronize URL parameters with current session state."""
    if hasattr(st.session_state, "current_tab"):
        st.query_params["tab"] = DISPLAY_TO_URL.get(st.session_state.current_tab, "SingleSOView")

    current_tab = st.session_state.get("current_tab", TAB_OPTIONS[0])

    if current_tab == "Select a Sales Order":
        _sync_param("DocNum", ["search_doc_number", "url_doc_num"])
        _sync_param("ItemNum", ["search_doc_item", "url_item_num"])
        _clean_params(["year", "month", "day"])
    elif current_tab == "Monthly Dashboard":
        _sync_param("year", ["selected_year", "url_year"], stringify=True)
        _sync_param("month", ["selected_month", "url_month"], stringify=True)
        _sync_param("day", ["selected_day_input", "url_day"], stringify=True)
        _clean_params(["DocNum", "ItemNum"])
    else:
        _clean_params(["DocNum", "ItemNum", "year", "month", "day"])


def _sync_param(param: str, state_keys: List[str], stringify: bool = False) -> None:
    for key in state_keys:
        value = st.session_state.get(key)
        if value:
            st.query_params[param] = str(value) if stringify else value
            return

    if param in st.query_params:
        del st.query_params[param]


def _clean_params(params: List[str]) -> None:
    for param in params:
        if param in st.query_params:
            del st.query_params[param]
