"""
Text extraction node for PDF processing.

This node extracts text content from PDF files using pypdfium2,
preserving structure and handling various text encodings.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List

import pypdfium2 as pdfium

from ..schemas import (
    ExtractionState,
    TextExtractionResult,
    PageTextData
)

logger = logging.getLogger(__name__)


def extract_text_node(state: ExtractionState) -> Dict[str, Any]:
    """
    Extract text content from a PDF file.
    
    This node processes the PDF to extract all text content,
    preserving structure where possible and detecting tables/images.
    
    Args:
        state: Current extraction state containing pdf_path and other parameters
        
    Returns:
        Updated state with text_result field populated
    """
    start_time = time.time()
    pdf_path = Path(state["pdf_path"])
    max_pages = state.get("max_pages")
    
    # Initialize result
    text_result = TextExtractionResult(
        success=False,
        pdf_path=str(pdf_path),
        total_pages=0,
        pages=[],
        extraction_time_ms=0
    )
    
    try:
        # Validate file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")
        
        logger.info(f"Starting text extraction from: {pdf_path.name}")
        
        # Open PDF document with pypdfium2
        pdf_document = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(pdf_document)
        
        # Determine pages to process
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        # Extract text from each page
        pages_data: List[PageTextData] = []
        
        for page_num in range(pages_to_process):
            page = pdf_document[page_num]
            
            # Extract text content using pypdfium2
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            
            # Check for tables (simplified detection based on text patterns)
            has_tables = _detect_tables_from_text(text)
            
            # Check for images (pypdfium2 doesn't have direct image count, use heuristic)
            # We'll check if the page has significant content beyond text
            has_images = _detect_images(page, text)
            
            # Create page data
            page_data = PageTextData(
                page_number=page_num + 1,
                text=text,
                char_count=len(text),
                has_tables=has_tables,
                has_images=has_images
            )
            
            pages_data.append(page_data)
            
            logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
            
            # Close the textpage to free memory
            textpage.close()
            page.close()
        
        # Close PDF document
        pdf_document.close()
        
        # Update result with success
        text_result = TextExtractionResult(
            success=True,
            pdf_path=str(pdf_path),
            total_pages=total_pages,
            pages=pages_data,
            extraction_time_ms=(time.time() - start_time) * 1000
        )
        
        logger.info(f"Successfully extracted text from {len(pages_data)} pages")
        
    except Exception as e:
        # Log error and update result
        error_message = f"Text extraction failed: {str(e)}"
        logger.error(error_message)
        
        text_result = TextExtractionResult(
            success=False,
            pdf_path=str(pdf_path),
            total_pages=0,
            pages=[],
            extraction_time_ms=(time.time() - start_time) * 1000,
            error_message=error_message
        )
        
        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_message)
    
    # Return updated state
    return {"text_result": text_result}


def _detect_tables_from_text(text: str) -> bool:
    """
    Simple heuristic to detect if text likely contains tables.
    
    This uses basic detection looking for:
    - Multiple columns of aligned text
    - Presence of table-like structures (pipes, tabs)
    
    Args:
        text: Extracted text from page
        
    Returns:
        True if tables are likely present
    """
    try:
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Check for table indicators
        table_indicators = [
            "total", "suma", "subtotal", "precio", "cantidad", 
            "monto", "importe", "unit", "qty", "amount", 
            "$", "%", "tax", "iva", "descuento"
        ]
        
        # Count how many indicators are present
        indicator_count = sum(1 for indicator in table_indicators if indicator in text_lower)
        
        # Check for table-like structures
        lines = text.split('\n')
        
        # Look for lines with multiple columns (tabs or multiple spaces)
        columnar_lines = 0
        for line in lines:
            # Count significant gaps (multiple spaces or tabs)
            if '\t' in line or '  ' in line:
                columnar_lines += 1
            # Also check for pipe symbols (common in tables)
            if '|' in line and line.count('|') >= 2:
                return True
        
        # If we have multiple indicators and columnar structure, likely a table
        if indicator_count >= 3 or columnar_lines >= 3:
            return True
            
        return False
        
    except Exception as e:
        logger.debug(f"Error detecting tables: {e}")
        return False


def _detect_images(page: pdfium.PdfPage, text: str) -> bool:
    """
    Detect if a page likely contains images.
    
    This uses a heuristic approach since pypdfium2 doesn't directly expose image count.
    We check if the page has content that suggests images (low text density).
    
    Args:
        page: pypdfium2 page object
        text: Extracted text from the page
        
    Returns:
        True if images are likely present
    """
    try:
        # If the page has very little text, it might be mostly images
        if len(text.strip()) < 100:
            return True
            
        # Check for image-related keywords in the text
        image_keywords = ["figure", "fig.", "image", "photo", "diagram", "chart", "graph"]
        text_lower = text.lower()
        
        for keyword in image_keywords:
            if keyword in text_lower:
                return True
                
        return False
        
    except Exception as e:
        logger.debug(f"Error detecting images: {e}")
        return False


def extract_text_with_structure(page: pdfium.PdfPage, text: str) -> Dict[str, Any]:
    """
    Extract text with preserved structure including paragraphs and formatting.
    
    This is a simplified extraction method for pypdfium2.
    
    Args:
        page: pypdfium2 page object
        text: Extracted text from the page
        
    Returns:
        Dictionary containing structured text data
    """
    try:
        structured_content = {
            "paragraphs": [],
            "headers": [],
            "lists": [],
            "raw_text": text
        }
        
        # Split text into paragraphs (separated by double newlines)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Simple heuristic: if a line is all caps or starts with a number, might be a header
            lines = para.split('\n')
            if lines:
                first_line = lines[0].strip()
                
                # Check if it might be a header (all caps, short, or numbered)
                if (first_line.isupper() and len(first_line) < 100) or \
                   (first_line and first_line[0].isdigit() and '.' in first_line[:3]):
                    structured_content["headers"].append(first_line)
                
                # Check if it's a list item
                if first_line.startswith(('•', '-', '*', '○', '▪', '►')):
                    structured_content["lists"].append(para)
                else:
                    structured_content["paragraphs"].append(para)
        
        return structured_content
        
    except Exception as e:
        logger.error(f"Error extracting structured text: {e}")
        return {"raw_text": text, "paragraphs": [], "headers": [], "lists": []}