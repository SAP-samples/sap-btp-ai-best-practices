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
DEFAULT_PROD_API_URL = ""
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