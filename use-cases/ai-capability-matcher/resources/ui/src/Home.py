import streamlit as st
import os
from src.utils import load_css_files

# Constants
APP_TITLE = "AI Capability Matcher"

# --- Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="static/images/SAP_logo_square.png",
    layout="wide",
)
st.logo("static/images/SAP_logo.svg")
st.title(APP_TITLE)

# --- CSS ---
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

# --- App ---
# Intro description (high-level): explain what the application does and how to start
st.markdown(
    "Compare a client's product list against an AI capability catalog to find the best matches. "
    "Upload two CSV files, pick the columns that describe each item, and the app uses "
    "embeddings—with optional LLM-based ranking and brief reasoning—to suggest likely capabilities. "
    "Track progress as it runs and download the results as a CSV when finished."
)
st.markdown("Open the ‘Capability Matcher’ page from the sidebar to begin.")
