from __future__ import annotations

"""Helpers for synchronizing view state."""

import streamlit as st


def set_current_order(document_number: str, document_item: str) -> None:
    st.session_state["search_doc_number"] = document_number
    st.session_state["search_doc_item"] = document_item
