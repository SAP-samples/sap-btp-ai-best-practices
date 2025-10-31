from __future__ import annotations

"""Form components for user input."""

from typing import Optional

import streamlit as st

from services import order_selection


def order_search_form() -> Optional[order_selection.OrderKey]:
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        doc_num = st.text_input(
            "Sales Document Number:",
            placeholder="e.g., 12714306",
            key="sales_doc",
            value=st.session_state.get("url_doc_num", ""),
        )

    with col2:
        doc_item = st.text_input(
            "Sales Document Item:",
            placeholder="e.g., 10",
            key="sales_item",
            value=st.session_state.get("url_item_num", ""),
        )

    selected_key: Optional[order_selection.OrderKey] = None

    with col3:
        st.markdown("<div style='padding-top: 29px;'></div>", unsafe_allow_html=True)
        if st.button("Run AI Anomaly Detection", use_container_width=True, key="search_btn", type="primary"):
            if doc_num and doc_item:
                selected_key = order_selection.OrderKey(document_number=doc_num, document_item=doc_item)
                st.session_state["search_doc_number"] = doc_num
                st.session_state["search_doc_item"] = doc_item

    return selected_key
