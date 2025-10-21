"""
API client for communicating with the Document Extraction backend.
"""

import os
import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st
import urllib3

# Get the API base URL from environment variables
# In production, this should be set to the deployed API URL
DEFAULT_PROD_API_URL = "https://sesajal_data_extraction_api.cfapps.eu10-004.hana.ondemand.com"
DEFAULT_LOCAL_API_URL = "http://localhost:8000"

# Try to get from environment, fallback to production URL if on cloud, local otherwise
API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    # Check if we're running in Cloud Foundry (PORT env var is set)
    if os.getenv("PORT"):
        API_BASE_URL = DEFAULT_PROD_API_URL
    else:
        API_BASE_URL = DEFAULT_LOCAL_API_URL

# SSL Verification setting - for Cloud Foundry internal apps, this can be disabled
# Set SSL_VERIFY=false in environment to disable SSL verification
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() not in ["false", "0", "no"]

# Suppress SSL warnings if verification is disabled
if not SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Store in session state for visibility
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = API_BASE_URL
if 'ssl_verify' not in st.session_state:
    st.session_state.ssl_verify = SSL_VERIFY


def make_api_request(
    endpoint: str, 
    method: str = "POST", 
    payload: Dict[str, Any] = None,
    files: Dict[str, Any] = None,
    data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Make a request to the backend API.

    Args:
        endpoint: The API endpoint to call (e.g., "/extraction/single").
        method: The HTTP method to use ("GET", "POST", etc.).
        payload: The JSON payload to send with the request.
        files: Files to upload (for multipart/form-data).
        data: Form data to send with files.

    Returns:
        A dictionary containing the JSON response from the API.
    """
    api_url = f"{API_BASE_URL}/api{endpoint}"

    try:
        # Get SSL verification setting from session state or environment
        verify_ssl = st.session_state.get('ssl_verify', SSL_VERIFY)
        
        if method.upper() == "POST":
            if files:
                # For file uploads, use multipart/form-data
                response = requests.post(
                    api_url,
                    files=files,
                    data=data,
                    timeout=600,  # 10 minutes for large file processing
                    verify=verify_ssl
                )
            else:
                # For regular JSON requests
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                    verify=verify_ssl
                )
        else:
            response = requests.get(api_url, timeout=10, verify=verify_ssl)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"API request failed with status {response.status_code}: {response.text}",
            }
    except requests.exceptions.SSLError as e:
        # Specific handling for SSL errors
        return {
            "success": False,
            "error": f"SSL Certificate verification failed. This is common in Cloud Foundry. "
                    f"Set SSL_VERIFY=false in environment variables to disable verification. "
                    f"Error: {str(e)}",
        }
    except requests.exceptions.ConnectionError as e:
        # Check if it's actually an SSL issue wrapped in ConnectionError
        if "SSL" in str(e) or "certificate" in str(e).lower():
            return {
                "success": False,
                "error": f"SSL/Certificate issue: {str(e)}. Try setting SSL_VERIFY=false in environment.",
            }
        return {
            "success": False,
            "error": f"Cannot connect to API server at {api_url}. Make sure the API service is running.",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The document might be too large or complex.",
        }
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


def health_check():
    """Performs a health check on the API server."""
    return make_api_request("/health", method="GET")


def get_api_status() -> Dict[str, Any]:
    """
    Get detailed API connection status for debugging.
    
    Returns:
        Dictionary with connection details and status
    """
    api_url = st.session_state.get('api_base_url', API_BASE_URL)
    
    try:
        response = health_check()
        if response and response.get("status") == "healthy":
            return {
                "connected": True,
                "api_url": api_url,
                "status": "✅ Connected",
                "message": "API is healthy and responding",
                "service": response.get("service", "extraction-api")
            }
        else:
            return {
                "connected": False,
                "api_url": api_url,
                "status": "❌ Connection Failed",
                "message": response.get("error", "API returned unhealthy status"),
                "response": response
            }
    except Exception as e:
        return {
            "connected": False,
            "api_url": api_url,
            "status": "❌ Connection Error",
            "message": str(e),
            "error_type": type(e).__name__
        }


def display_api_status():
    """Display API connection status in Streamlit sidebar."""
    status = get_api_status()
    
    with st.sidebar:
        st.markdown("### 🔌 API Connection")
        
        if status["connected"]:
            st.success(status["status"])
        else:
            st.error(status["status"])
            
        with st.expander("Connection Details", expanded=not status["connected"]):
            st.text(f"API URL: {status['api_url']}")
            st.text(f"SSL Verification: {'Enabled' if st.session_state.get('ssl_verify', SSL_VERIFY) else 'Disabled'}")
            st.text(f"Message: {status['message']}")
            
            if not status["connected"]:
                if "error_type" in status:
                    st.text(f"Error Type: {status['error_type']}")
                
                # Show SSL help if it's an SSL error
                if "SSL" in status.get("message", "") or "certificate" in status.get("message", ""):
                    st.warning(
                        "💡 **SSL Issue Detected**\n\n"
                        "To fix this, set the environment variable:\n"
                        "`SSL_VERIFY=false`\n\n"
                        "For local testing, add to your .env file:\n"
                        "`SSL_VERIFY=false`"
                    )
                
            # Add button to test connection
            if st.button("🔄 Test Connection"):
                st.rerun()


def extract_single_document(
    file_content: bytes,
    filename: str,
    document_type: str,
    questions: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    temperature: float = 0.1,
    language: str = "es"
) -> Dict[str, Any]:
    """
    Extract information from a single PDF document.
    
    Args:
        file_content: PDF file content as bytes
        filename: Name of the file
        document_type: Type of document
        questions: Optional custom questions
        fields: Optional custom field names
        temperature: LLM temperature
        language: Response language
        
    Returns:
        Extraction response
    """
    files = {
        "file": (filename, file_content, "application/pdf")
    }
    
    data = {
        "document_type": document_type,
        "temperature": str(temperature),
        "language": language
    }
    
    if questions:
        data["questions"] = json.dumps(questions)
    
    if fields:
        data["fields"] = json.dumps(fields)
    
    return make_api_request(
        "/extraction/single",
        method="POST",
        files=files,
        data=data
    )


def extract_batch_documents(
    files_list: List[tuple],
    document_types: List[str],
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    Process multiple documents in batch.
    
    Args:
        files_list: List of tuples (filename, content, content_type)
        document_types: List of document types
        max_concurrent: Maximum concurrent processing
        
    Returns:
        Batch extraction response with task ID
    """
    # Create files list with same field name for multiple files
    # FastAPI expects multiple files with the same field name
    files = [('files', file_tuple) for file_tuple in files_list]
    
    data = {
        "document_types": json.dumps(document_types),
        "max_concurrent": str(max_concurrent)
    }
    
    return make_api_request(
        "/extraction/batch",
        method="POST",
        files=files,
        data=data
    )


def get_document_schemas() -> Dict[str, Any]:
    """
    Get all document schemas with questions and fields.
    
    Returns:
        Document schemas response
    """
    return make_api_request("/extraction/schemas", method="GET")


def update_document_schema(
    document_type: str,
    questions: List[str],
    fields: List[str]
) -> Dict[str, Any]:
    """
    Update questions and fields for a document type.
    
    Args:
        document_type: Type of document
        questions: New questions list
        fields: New fields list
        
    Returns:
        Update response
    """
    payload = {
        "document_type": document_type,
        "questions": questions,
        "fields": fields
    }
    
    return make_api_request(
        "/extraction/schemas",
        method="POST",
        payload=payload
    )


def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    Check the status of a batch processing task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status response
    """
    return make_api_request(f"/extraction/status/{task_id}", method="GET")


# Verification API client helpers
def verify_documents(
    doc_a: Dict[str, Any],
    doc_b: Dict[str, Any],
    key_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify attributes between two documents using the verification API.

    Args:
        doc_a: First document as key-value pairs
        doc_b: Second document as key-value pairs
        key_map: Mapping of keys and comparators

    Returns:
        Verification response with verified, failed and summary
    """
    payload = {
        "doc_a": doc_a,
        "doc_b": doc_b,
        "key_map": key_map,
    }
    return make_api_request(
        "/verification/verify",
        method="POST",
        payload=payload,
    )


def summarize_fields(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize verbose Investigación Legal fields using the backend summary API.

    Args:
        values: Dictionary of fields to summarize. Only known target fields will be summarized.

    Returns:
        API response with keys: success(bool), summaries(dict) when success.
    """
    payload = {"values": values}
    return make_api_request(
        "/summary/summarize",
        method="POST",
        payload=payload,
    )


# Utility function to format extraction results for display
def format_extraction_results(response: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Format extraction results for display in Streamlit.
    
    Args:
        response: API extraction response
        
    Returns:
        List of formatted results
    """
    if not response.get("success") or not response.get("results"):
        return []
    
    formatted = []
    for result in response["results"]:
        formatted.append({
            "Campo": result.get("field", ""),
            "Valor": result.get("answer", "")
        })
    
    return formatted