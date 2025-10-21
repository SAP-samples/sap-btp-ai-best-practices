"""
RFQ Chatbot Page (6_RFQ_Chatbot.py)
This page provides a minimal Streamlit chat UI wired to the RFQ knowledge graphs.
It performs two parallel LLM calls (one per supplier) with each supplier's full
knowledge graph as context, then consolidates their answers into a single
response via a third LLM call.
"""

import os
from typing import Any, Dict
import streamlit as st

from src.api_client import list_suppliers, ask_chat

# Page configuration (match other pages: icon + SAP logo + CSS)
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title="RFQ Chatbot - RFQ Analysis",
    page_icon=PAGE_ICON_PATH,
    layout="wide"
)

try:
    st.logo(SAP_SVG_PATH)
except Exception:
    pass

from utils import load_css_files  # after sys.path add
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash"))


@st.cache_data
def _get_supplier_ids() -> Dict[str, Any]:
    """Fetch supplier list from API and return mapping of two default IDs and names."""
    data = list_suppliers()
    suppliers = data.get("suppliers", []) if isinstance(data, dict) else []
    # Fallback defaults
    s1 = next((s for s in suppliers if s.get("id") in (os.getenv("SUPPLIER1_ID", "supplier1"),)), suppliers[0] if suppliers else {"id": "supplier1", "name": "Supplier 1"})
    s2 = next((s for s in suppliers if s.get("id") in (os.getenv("SUPPLIER2_ID", "supplier2"),)), suppliers[1] if len(suppliers) > 1 else {"id": "supplier2", "name": "Supplier 2"})
    return {"supplier1": s1, "supplier2": s2}


def _render_sources(sources: list[Dict[str, Any]] | None) -> None:
    if not sources:
        return
    st.caption("Sources:")
    for s in sources:
        fn = str(s.get("filename", ""))
        ch = str(s.get("chunk_id", ""))
        st.write(f"- {fn}:{ch}")


# Old local-LLM helper functions removed: UI now calls API `/v1/chat/ask` exclusively.


# --- UI ---

# st.title("RFQ Chatbot")

# Initialize session state for chat history
if "chat_messages_chatbot" not in st.session_state:
    st.session_state["chat_messages_chatbot"] = [
        {"role": "assistant", "content": "Hi! Ask a question about the RFQ or suppliers."}
    ]

# Render chat history
for msg in st.session_state["chat_messages_chatbot"]:
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(msg.get("content", ""))

# Chat input
user_input = st.chat_input("Ask about suppliers, pricing, risks, RQDCE, etc.")

if user_input:
    # Echo user message in the chat
    st.session_state["chat_messages_chatbot"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Calling chat API..."):
                ids = _get_supplier_ids()
                s1 = ids.get("supplier1", {})
                s2 = ids.get("supplier2", {})
                s1_name = "SupplierA"
                s2_name = "SupplierB"
                model_name = CHAT_MODEL_NAME
                resp = ask_chat(user_input, s1.get("id", "supplier1"), s2.get("id", "supplier2"), model=model_name, supplier1_name=s1_name, supplier2_name=s2_name)
                final_text = resp.get("answer_markdown", "")
                st.markdown(final_text or "No answer.")
                st.session_state["chat_messages_chatbot"].append({
                    "role": "assistant",
                    "content": final_text or "No answer.",
                })
                with st.expander("Sources"):
                    _render_sources(resp.get("sources"))
    except Exception as e:
        err = f"RFQ Chatbot error: {e}"
        st.error(err)
        st.session_state["chat_messages_chatbot"].append({"role": "assistant", "content": err})

