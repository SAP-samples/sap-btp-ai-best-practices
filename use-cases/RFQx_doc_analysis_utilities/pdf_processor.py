"""
PDF processing for the simplified RAG approach.

This module handles PDF text extraction optimized for full document processing
without chunking, ensuring maximum text preservation for large context models.
"""

import os
import re
import tiktoken
from typing import Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global tokenizer configuration for GPT-4
TOKENIZER_NAME = "cl100k_base"  # GPT-4 tokenizer


def extract_pdf_content(file_path: str, method: str = "pymupdf") -> Dict[str, Any]:
    """
    Extract complete content from a PDF document for large context processing.
    
    Args:
        file_path: Path to the PDF file
        method: Extraction method ("pymupdf" or "pdfplumber")
        
    Returns:
        Dictionary with extracted content and metadata
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a PDF or cannot be processed
    """
    file_path = Path(file_path)
    
    # Validate file exists and is a PDF
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF, got: {file_path.suffix}")
    
    print(f"Extracting complete content from {file_path.name} using {method}...")
    
    result = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "full_text": "",
        "page_count": 0,
        "word_count": 0,
        "token_count": 0,
        "extraction_method": method,
        "pages": []  # Store per-page content for debugging
    }
    
    try:
        if method == "pymupdf":
            result = _extract_with_pymupdf(file_path, result)
        elif method == "pdfplumber":
            result = _extract_with_pdfplumber(file_path, result)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
            
    except Exception as e:
        raise ValueError(f"Error processing PDF file {file_path}: {str(e)}")
    
    # Validate extraction
    if not result["full_text"].strip():
        raise ValueError(f"No text content extracted from {file_path}")
    
    # Calculate statistics
    result["word_count"] = len(re.findall(r'\b\w+\b', result["full_text"]))
    
    try:
        tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
        result["token_count"] = len(tokenizer.encode(result["full_text"]))
    except Exception as e:
        print(f"Warning: Could not count tokens: {e}")
        result["token_count"] = result["word_count"] * 1.3  # Rough estimation
    
    print(f"Extracted: {result['page_count']} pages, {result['word_count']} words, {result['token_count']} tokens")
    
    # Check if content might exceed context limits
    if result["token_count"] > 100000:  # Conservative limit for GPT-4
        print(f"⚠️  Warning: Document has {result['token_count']} tokens, which may exceed context limits")
    
    return result


def _extract_with_pymupdf(file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text using PyMuPDF with enhanced formatting preservation."""
    
    with fitz.open(str(file_path)) as pdf_doc:
        result["page_count"] = len(pdf_doc)
        
        all_text_parts = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            
            # Extract text with enhanced formatting
            text_dict = page.get_text("dict")
            page_text = _process_pymupdf_dict(text_dict)
            
            # Also get basic text as fallback
            basic_text = page.get_text()
            
            # Use the more comprehensive extraction
            final_page_text = page_text if len(page_text.strip()) > len(basic_text.strip()) else basic_text
            
            # Add page separator
            if page_num > 0:
                all_text_parts.append(f"\n\n--- PAGE {page_num + 1} ---\n\n")
            
            all_text_parts.append(final_page_text)
            
            # Store per-page content for debugging
            result["pages"].append({
                "page_number": page_num + 1,
                "text": final_page_text,
                "character_count": len(final_page_text)
            })
    
    result["full_text"] = "".join(all_text_parts)
    return result


def _extract_with_pdfplumber(file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text using pdfplumber with table detection."""
    
    with pdfplumber.open(str(file_path)) as pdf:
        result["page_count"] = len(pdf.pages)
        
        all_text_parts = []
        
        for page_num, page in enumerate(pdf.pages):
            page_parts = []
            
            # Add page separator
            if page_num > 0:
                all_text_parts.append(f"\n\n--- PAGE {page_num + 1} ---\n\n")
            
            # Extract regular text
            text = page.extract_text()
            if text:
                page_parts.append(text)
            
            # Extract tables as formatted text
            tables = page.extract_tables()
            for table_num, table in enumerate(tables):
                if table:
                    table_text = _format_table_as_text(table, f"Table {table_num + 1}")
                    page_parts.append(f"\n\n{table_text}\n\n")
            
            final_page_text = "\n".join(page_parts)
            all_text_parts.append(final_page_text)
            
            # Store per-page content for debugging
            result["pages"].append({
                "page_number": page_num + 1,
                "text": final_page_text,
                "character_count": len(final_page_text),
                "table_count": len(tables)
            })
    
    result["full_text"] = "".join(all_text_parts)
    return result


def _process_pymupdf_dict(text_dict: Dict) -> str:
    """Process PyMuPDF text dictionary to preserve formatting."""
    
    text_parts = []
    
    for block in text_dict.get("blocks", []):
        if "lines" in block:  # Text block
            block_parts = []
            
            for line in block["lines"]:
                line_parts = []
                
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        # Preserve some formatting cues
                        flags = span.get("flags", 0)
                        if flags & 16:  # Bold
                            text = f"**{text}**"
                        line_parts.append(text)
                
                if line_parts:
                    block_parts.append(" ".join(line_parts))
            
            if block_parts:
                text_parts.append("\n".join(block_parts))
    
    return "\n\n".join(text_parts)


def _format_table_as_text(table: list, table_title: str = "Table") -> str:
    """Format a table as readable text."""
    
    if not table:
        return ""
    
    # Remove None values and convert to strings
    clean_table = []
    for row in table:
        clean_row = [str(cell) if cell is not None else "" for cell in row]
        clean_table.append(clean_row)
    
    if not clean_table:
        return ""
    
    # Calculate column widths
    max_cols = max(len(row) for row in clean_table)
    col_widths = [0] * max_cols
    
    for row in clean_table:
        for i, cell in enumerate(row):
            if i < max_cols:
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Format table
    formatted_lines = [f"=== {table_title} ==="]
    
    for row_num, row in enumerate(clean_table):
        formatted_row = []
        for i in range(max_cols):
            cell = row[i] if i < len(row) else ""
            formatted_row.append(cell.ljust(col_widths[i]))
        
        formatted_lines.append(" | ".join(formatted_row))
        
        # Add separator after header
        if row_num == 0 and len(clean_table) > 1:
            separator = " | ".join(["-" * width for width in col_widths])
            formatted_lines.append(separator)
    
    return "\n".join(formatted_lines)


def check_context_limits(content: Dict[str, Any], max_tokens: int = 1000000) -> Dict[str, Any]:
    """
    Check if content fits within context limits and suggest optimizations.
    
    Args:
        content: Extracted content dictionary
        max_tokens: Maximum token limit
        
    Returns:
        Dictionary with limit check results and recommendations
    """
    
    token_count = content.get("token_count", 0)
    
    result = {
        "within_limits": token_count <= max_tokens,
        "token_count": token_count,
        "max_tokens": max_tokens,
        "utilization_percentage": (token_count / max_tokens) * 100,
        "recommendations": []
    }
    
    if not result["within_limits"]:
        excess_tokens = token_count - max_tokens
        result["recommendations"].extend([
            f"Document exceeds context limit by {excess_tokens} tokens",
            "Consider processing document in sections",
            "Or use a model with larger context window"
        ])
    elif token_count > max_tokens * 0.8:
        result["recommendations"].append(
            f"Document uses {result['utilization_percentage']:.1f}% of context - close to limit"
        )
    
    return result


def optimize_text_for_context(text: str, target_tokens: int = 800000) -> str:
    """
    Optimize text to fit within context limits while preserving key information.
    
    Args:
        text: Full text content
        target_tokens: Target token count
        
    Returns:
        Optimized text content
    """
    
    try:
        tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
        current_tokens = len(tokenizer.encode(text))
        
        if current_tokens <= target_tokens:
            return text
        
        # Calculate reduction ratio
        reduction_ratio = target_tokens / current_tokens
        
        # Split into paragraphs and preserve proportionally
        paragraphs = text.split('\n\n')
        optimized_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Keep important sections (headers, tables, short paragraphs)
                if (paragraph.startswith('===') or 
                    paragraph.startswith('---') or 
                    len(paragraph) < 200 or
                    'table' in paragraph.lower()):
                    optimized_paragraphs.append(paragraph)
                else:
                    # Truncate long paragraphs
                    target_length = int(len(paragraph) * reduction_ratio)
                    if target_length > 100:  # Keep minimum meaningful length
                        optimized_paragraphs.append(paragraph[:target_length] + "...")
        
        optimized_text = '\n\n'.join(optimized_paragraphs)
        
        # Verify final token count
        final_tokens = len(tokenizer.encode(optimized_text))
        print(f"Text optimization: {current_tokens} → {final_tokens} tokens ({reduction_ratio:.1%} reduction)")
        
        return optimized_text
        
    except Exception as e:
        print(f"Warning: Text optimization failed: {e}")
        # Fallback: simple truncation
        return text[:int(len(text) * 0.8)]