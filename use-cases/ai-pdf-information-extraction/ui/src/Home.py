import streamlit as st
import os
from src.utils import load_css_files

# Constants
APP_TITLE = "Document Extraction System"

# --- Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="static/images/SAP_logo_square.png",
    layout="wide",
)
st.logo("static/images/SAP_logo.svg")
st.title(f"Welcome to {APP_TITLE}")

# --- CSS ---
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

# --- Display API Status in Sidebar ---
# display_api_status()

# --- App ---

st.markdown("""
### AI-Powered PDF Document Information Extraction

This system enables structured information extraction from PDF documents using
advanced artificial intelligence with vision capabilities.

#### Key Features:

- **Custom Document Types**: Create and define your own document types with custom extraction fields
- **Flexible Extraction**: Define custom questions and attributes to extract from any document
- **Batch Processing**: Process multiple documents simultaneously
- **Export Results**: Download results in CSV format
- **Parallel Processing**: Optimized for maximum performance

#### How to use the application:

1. **Extraction Config**: Define custom document types and extraction questions
2. **Document Processing**: Upload and process documents using your custom types
3. **Results Dashboard**: View and export all processed results


---

Use the sidebar to navigate between different pages of the application.
""")

# Add connection status check
with st.sidebar:
    st.header("System Status")
    
    # Check API health
    from src.api_client import health_check
    
    if st.button("Check API Connection"):
        response = health_check()
        if response.get("status") == "healthy":
            st.success("API connected and working")
        else:
            st.error(f"Connection error: {response.get('error', 'API not available')}")
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    - **Extraction Config**: Define custom document types
    - **Document Processing**: Process PDFs with your types
    - **Results Dashboard**: View and export results
    """)
    

# Footer
st.markdown("---")
st.markdown(
    f"**{APP_TITLE}** | Built with Streamlit and FastAPI | Powered by OpenAI GPT-4 Vision",
    help="AI-powered document extraction system"
)