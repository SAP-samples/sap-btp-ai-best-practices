"""
Utility functions for the Document Extraction UI.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_css_files(file_paths: List[str]) -> None:
    """
    Load and inject CSS files into the Streamlit app.
    
    Args:
        file_paths: List of CSS file paths to load
    """
    css_content = ""
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists():
                with open(path, "r") as f:
                    css_content += f.read() + "\n"
        except Exception as e:
            st.warning(f"Could not load CSS file {file_path}: {e}")
    
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def export_results_to_json(results: Dict[str, Any], filename: str = None) -> str:
    """
    Export extraction results to JSON format.
    
    Args:
        results: Dictionary of results to export
        filename: Optional filename (without extension)
        
    Returns:
        JSON string
    """
    if filename is None:
        filename = f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return json.dumps(results, ensure_ascii=False, indent=2)


def export_results_to_csv(results_df: pd.DataFrame, filename: str = None) -> str:
    """
    Export extraction results to CSV format.
    
    Args:
        results_df: DataFrame with results
        filename: Optional filename (without extension)
        
    Returns:
        CSV string
    """
    if filename is None:
        filename = f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return results_df.to_csv(index=False)


def combine_results_for_export(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple DataFrames into one for export.
    
    Args:
        results_dict: Dictionary with document type as key and DataFrame as value
        
    Returns:
        Combined DataFrame
    """
    combined_data = []
    
    for doc_id, df in results_dict.items():
        df_copy = df.copy()
        # Document Type column should already be in the DataFrame from processing
        # If not, we'll add it based on the doc_id
        if 'Document Type' not in df_copy.columns:
            # Try to get the name from session state if available
            if 'custom_document_types' in st.session_state and doc_id in st.session_state.custom_document_types:
                df_copy['Document Type'] = st.session_state.custom_document_types[doc_id]['name']
            else:
                df_copy['Document Type'] = doc_id
        combined_data.append(df_copy)
    
    if combined_data:
        return pd.concat(combined_data, ignore_index=True)
    
    return pd.DataFrame()


def initialize_session_state():
    """
    Initialize session state variables for the application.
    """
    # Initialize extraction results storage
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = {}
    
    # Initialize batch results storage
    if "batch_extraction_results" not in st.session_state:
        st.session_state.batch_extraction_results = []
    
    # Initialize document questions storage
    if "document_questions" not in st.session_state:
        st.session_state.document_questions = {}
    
    # Initialize batch status
    if "batch_status" not in st.session_state:
        st.session_state.batch_status = {}
    
    # Initialize current task ID
    if "current_task_id" not in st.session_state:
        st.session_state.current_task_id = None


def display_processing_status(status: str, message: str = None):
    """
    Display a processing status message.
    
    Args:
        status: Status type ("success", "error", "warning", "info")
        message: Message to display
    """
    if status == "success":
        st.success(message or "Processing completed successfully")
    elif status == "error":
        st.error(message or "Error during processing")
    elif status == "warning":
        st.warning(message or "Warning")
    elif status == "info":
        st.info(message or "Processing...")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_pdf_file(file) -> tuple[bool, str]:
    """
    Validate that a file is a PDF.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file:
        return False, "No file uploaded"
    
    if not file.name.lower().endswith('.pdf'):
        return False, "File must be a PDF"
    
    # Check file size (max 10MB)
    if file.size > 10 * 1024 * 1024:
        return False, "File is too large (maximum 10MB)"
    
    return True, ""


def create_results_dataframe(extraction_response: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from extraction response.
    
    Args:
        extraction_response: Response from extraction API
        
    Returns:
        DataFrame with results
    """
    if not extraction_response.get("success") or not extraction_response.get("results"):
        return pd.DataFrame()
    
    data = []
    for result in extraction_response["results"]:
        data.append({
            "Field": result.get("field", ""),
            "Value": result.get("answer", "")
        })
    
    return pd.DataFrame(data)


def get_document_type_display_name(doc_type: str) -> str:
    """
    Get the display name for a document type.
    
    Args:
        doc_type: Internal document type name or ID
        
    Returns:
        Display name
    """
    # Check if it's a custom document type from session state
    if 'custom_document_types' in st.session_state and doc_type in st.session_state.custom_document_types:
        return st.session_state.custom_document_types[doc_type]['name']
    
    # Fallback to the doc_type itself
    return doc_type


def parse_api_error(error_response: Dict[str, Any]) -> str:
    """
    Parse API error response to get user-friendly message.
    
    Args:
        error_response: Error response from API
        
    Returns:
        User-friendly error message
    """
    if isinstance(error_response, dict):
        error = error_response.get("error", "")
        if "connect" in error.lower():
            return "Cannot connect to API server. Make sure it's running."
        elif "timeout" in error.lower():
            return "Request timed out. The document might be too large or complex."
        else:
            return error or "Unknown error"
    
    return str(error_response)


# --- Markdown → PDF helpers ---

try:
    # Python-Markdown: convert Markdown text to HTML
    import markdown as _md
except Exception:  # pragma: no cover
    _md = None

try:
    # xhtml2pdf (ReportLab-based) fallback with fewer OS dependencies
    from xhtml2pdf import pisa as _PISA
except Exception:  # pragma: no cover
    _PISA = None


def markdown_to_pdf(markdown_text: str) -> bytes:
    """
    Convert a Markdown string to a styled PDF (bytes) locally in the UI.

    This avoids backend calls by rendering Markdown → HTML (Python-Markdown)
    and then HTML → PDF (xhtml2pdf) entirely on the client side.

    Args:
        markdown_text: Markdown content to convert

    Returns:
        PDF file contents as bytes

    Raises:
        ImportError: When required packages are missing
        RuntimeError: On conversion errors
    """
    if not isinstance(markdown_text, str):
        markdown_text = str(markdown_text or "")

    if _md is None:
        raise ImportError("markdown package is not installed. Add 'markdown' to UI requirements.")

    # Convert markdown to HTML (enable common extensions)
    try:
        html_body = _md.markdown(
            markdown_text,
            extensions=[
                "extra",           # tables, etc.
                "fenced_code",     # ``` code blocks
                "sane_lists",
                "toc",             # basic support; okay if unused
            ],
            output_format="html5",
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Markdown conversion failed: {e}")

    # Basic printable styling suitable for A4
    css_text = """
    @page { size: A4; margin: 20mm 16mm; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      font-size: 12pt;
      line-height: 1.5;
      color: #111;
    }
    h1, h2, h3, h4, h5, h6 { font-weight: 600; margin: 0.7em 0 0.4em; }
    h1 { font-size: 22pt; }
    h2 { font-size: 18pt; }
    h3 { font-size: 16pt; }
    p { margin: 0.4em 0 0.8em; }
    ul, ol { margin: 0.4em 0 0.8em 1.4em; }
    blockquote {
      border-left: 4px solid #ddd;
      margin: 0.8em 0;
      padding: 0.1em 0 0.1em 1em;
      color: #555;
    }
    table { border-collapse: collapse; width: 100%; margin: 0.8em 0; }
    th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: left; }
    th { background: #f2f2f2; }
    code { background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }
    pre { background: #f6f8fa; padding: 10px; overflow-x: auto; }
    pre code { background: transparent; padding: 0; }
    """

    # Wrap into a minimal HTML document
    html_doc = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Credit Report</title>
      </head>
      <body>
        {html_body}
      </body>
    </html>
    """

    # Primary: xhtml2pdf (supports a subset of HTML/CSS; good enough for basic reports)
    if _PISA is not None:
        try:
            import io
            pdf_io = io.BytesIO()
            # xhtml2pdf expects simple HTML/CSS; complex CSS may be ignored
            result = _PISA.CreatePDF(src=html_doc, dest=pdf_io)
            if result.err:
                raise RuntimeError("xhtml2pdf failed to render PDF")
            return pdf_io.getvalue()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"PDF rendering failed (fallback): {e}")

    # If neither backend is available
    raise RuntimeError("No PDF backend available. Install 'xhtml2pdf'.")


def sanitize_markdown_for_display(text: str) -> str:
    """
    Sanitize markdown text for safe display in Streamlit by escaping characters
    that could trigger unwanted LaTeX math rendering or formatting.
    
    This function:
    - Escapes unescaped dollar signs ($) to prevent LaTeX math interpretation
    - Escapes underscores between word characters to prevent unwanted emphasis
    - Preserves code blocks (``` or ~~~) and inline code (``) unchanged
    
    Args:
        text: Raw markdown text to sanitize
        
    Returns:
        Sanitized markdown text safe for st.markdown() display
    """
    import re
    
    if not isinstance(text, str):
        text = str(text or "")
    
    if not text.strip():
        return text
    
    # Split text into segments: code blocks, inline code, and normal text
    segments = []
    current_pos = 0
    
    # Pattern to match code fences (``` or ~~~) and inline code (`)
    # This regex captures:
    # - Triple backticks with optional language: ```python\ncode\n```
    # - Triple tildes: ~~~\ncode\n~~~
    # - Inline code: `code`
    code_pattern = re.compile(
        r'(```[\w]*\n.*?\n```|~~~[\w]*\n.*?\n~~~|`[^`\n]+`)',
        re.DOTALL
    )
    
    for match in code_pattern.finditer(text):
        # Add normal text before this code block
        if match.start() > current_pos:
            normal_text = text[current_pos:match.start()]
            segments.append(('normal', normal_text))
        
        # Add the code block unchanged
        code_text = match.group(1)
        segments.append(('code', code_text))
        current_pos = match.end()
    
    # Add any remaining normal text
    if current_pos < len(text):
        normal_text = text[current_pos:]
        segments.append(('normal', normal_text))
    
    # If no code blocks found, treat entire text as normal
    if not segments:
        segments = [('normal', text)]
    
    # Process each segment
    result_parts = []
    for segment_type, segment_text in segments:
        if segment_type == 'code':
            # Keep code segments unchanged
            result_parts.append(segment_text)
        else:
            # Sanitize normal text segments
            sanitized = segment_text
            
            # Escape unescaped dollar signs to prevent LaTeX math rendering
            # Use negative lookbehind to avoid double-escaping already escaped dollars
            sanitized = re.sub(r'(?<!\\)\$', r'\\$', sanitized)
            
            # Escape underscores between word characters to prevent unwanted emphasis
            # This targets cases like RFC_ABC123 but preserves _emphasis_ and __bold__
            sanitized = re.sub(r'(?<=\w)_(?=\w)', r'\\_', sanitized)
            
            result_parts.append(sanitized)
    
    return ''.join(result_parts)
