import streamlit as st
import os
from src.utils import load_css_files
from src.api_client import display_api_status

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
### AI-Powered Document Processing & Credit Evaluation System

This comprehensive system combines advanced AI document extraction with intelligent credit policy evaluation to streamline financial document processing and credit decision-making.

#### Key Features:

**Document Processing:**
- **AI-Powered Extraction**: Extract structured data from PDFs using SAP Document AI
- **Custom Document Types**: Support for KYC, CSF, Vendor Comments, and CGV documents
- **Cross-Document Validation**: Automatically verify consistency across multiple documents
- **Parallel Processing**: Process multiple documents simultaneously for maximum efficiency

**Credit Evaluation:**
- **Automated Credit Assessment**: AI-driven credit policy engine with configurable rules
- **Risk Scoring**: Generate CAL scores, C3M percentages, and risk assessments
- **Decision Support**: Get approval recommendations and director-level decision hints
- **Executive Reporting**: Generate comprehensive credit reports with AI analysis

**Data Integration:**
- **Excel Processing**: Parse payment history and credit request data from spreadsheets
- **Invoice Analysis**: Build payment behavior profiles from transaction history
- **Multi-Source Validation**: Cross-reference data from PDFs, Excel files, and system records

#### How to use the application:

**1. Credit Creation Workflow:**
   - Upload required documents: KYC, CSF, Vendor Comments, CGV, and Excel files
   - AI extracts key attributes (RFC, addresses, credit lines, etc.)
   - System validates cross-document consistency automatically
   - Configure credit parameters and risk settings
   - Run credit evaluation to get scores and recommendations
   - Generate executive reports and download results

**2. Document Extraction:**
   - Define custom document types and extraction fields
   - Upload and process documents using AI extraction
   - View and export structured results

**3. Results & Reporting:**
   - Review extraction results and validation status
   - Download credit evaluation reports as PDF
   - Export data in various formats (JSON, CSV)

---

Use the sidebar to navigate to the credit creation page.
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
    
    
