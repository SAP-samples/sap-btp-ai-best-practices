"""
Excel extraction module for the Knowledge Graph Creation Pipeline using markitdown.

This module handles Excel text extraction with enhanced metadata tracking,
converting tables to Markdown format and preserving sheet boundaries.
Part of Phase 1: Raw Text Extraction & Pre-processing.

This implementation uses the markitdown library for simplified and robust
Excel to Markdown conversion.
"""

import re
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from markitdown import MarkItDown

from ..models.kg_schema import SourceMetadata


def extract_excel_content(file_path: str, method: str = "markitdown", fast_mode: bool = True, 
                         export_csv: bool = False, empty_row_tolerance: int = 3, 
                         include_metadata: bool = True) -> Dict[str, Any]:
    """
    Extract complete content from an Excel document with enhanced metadata tracking.
    
    This function implements Phase 1 of the KG Creation Pipeline, converting
    Excel files into structured Markdown text with preserved sheet boundaries.
    
    Args:
        file_path: Path to the Excel file
        method: Extraction method (only "markitdown" is supported in this version)
        fast_mode: Ignored in this implementation (kept for compatibility)
        export_csv: Ignored in this implementation (CSV export not supported with markitdown)
        empty_row_tolerance: Ignored in this implementation
        include_metadata: Whether to include detailed metadata for each sheet
        
    Returns:
        Dictionary with extracted content and metadata:
        - filename: Name of the source file
        - file_path: Full path to the file
        - full_text: Complete extracted text with sheet headers
        - page_count: Number of sheets (treated as pages)
        - word_count: Total word count
        - extraction_method: Method used for extraction
        - pages: List of per-sheet content with metadata
        - extraction_timestamp: When the extraction occurred
        - source_metadata: List of SourceMetadata objects for each sheet
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not an Excel file or cannot be processed
    """
    file_path = Path(file_path)
    
    # Validate file exists and is Excel
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() not in ['.xlsx', '.xlsm', '.xls', '.xlsb']:
        raise ValueError(f"File must be Excel format, got: {file_path.suffix}")
    
    print(f"Extracting complete content from {file_path.name} using markitdown...")
    
    result = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "full_text": "",
        "page_count": 0,  # Will represent sheet count
        "word_count": 0,
        "extraction_method": "markitdown",
        "pages": [],  # Store per-sheet content (sheets = pages)
        "csv_exported": None,  # CSV export not supported with markitdown
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "source_metadata": []  # List of SourceMetadata objects
    }
    
    try:
        # Use markitdown to convert Excel to markdown
        md = MarkItDown()
        conversion_result = md.convert(str(file_path))
        full_text = conversion_result.text_content
        
        if not full_text.strip():
            raise ValueError(f"No text content extracted from {file_path}")
        
        # Process the markdown to extract sheet information
        result = _process_markitdown_output(full_text, result, include_metadata)
        
    except Exception as e:
        raise ValueError(f"Error processing Excel file {file_path}: {str(e)}")
    
    # Calculate statistics
    print("\nCalculating word count...")
    result["word_count"] = len(re.findall(r'\b\w+\b', result["full_text"]))
    
    print(f"\nExtraction complete: {result['page_count']} sheets, {result['word_count']} words")
    
    return result


def _clean_nan_only_lines(text: str) -> str:
    """Remove lines that consist only of '| NaN |' patterns repeated.
    
    This function uses regex to identify and remove lines that contain nothing
    but NaN values in table cells. Lines with other content alongside NaN values
    are preserved.
    
    Args:
        text: The markdown text to clean
        
    Returns:
        Cleaned text with NaN-only lines removed
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Regex pattern to match lines containing only | NaN | patterns
    # This matches lines that start with |, contain only NaN and spaces between pipes, and end with |
    nan_only_pattern = r'^\s*\|(\s*NaN\s*\|)+\s*$'
    
    for line in lines:
        if re.match(nan_only_pattern, line):
            # This line contains only NaN values, skip it
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _process_markitdown_output(markdown_text: str, result: Dict[str, Any], 
                              include_metadata: bool = True) -> Dict[str, Any]:
    """Process markitdown output to extract sheet information and metadata."""
    
    # Split by sheet headers (## SheetName format)
    sheet_pattern = r'^## (.+)$'
    
    # Find all sheet headers
    sheet_headers = []
    for match in re.finditer(sheet_pattern, markdown_text, re.MULTILINE):
        sheet_headers.append({
            'name': match.group(1),
            'start': match.start(),
            'end': match.end()
        })
    
    if not sheet_headers:
        # If no sheet headers found, treat entire content as one sheet
        sheet_headers = [{
            'name': 'Sheet1',
            'start': 0,
            'end': 0
        }]
        # Add sheet header to the beginning
        markdown_text = "## Sheet1\n\n" + markdown_text
    
    result["page_count"] = len(sheet_headers)
    print(f"Found {result['page_count']} sheets")
    
    # Convert markitdown format to our expected format
    # Replace ## SheetName with # Sheet X: SheetName for compatibility
    converted_text_parts = []
    
    for i, sheet_info in enumerate(sheet_headers):
        sheet_name = sheet_info['name']
        print(f"\nProcessing sheet {i + 1}/{result['page_count']}: {sheet_name}")
        
        # Determine content boundaries
        content_start = sheet_info['end']
        if i < len(sheet_headers) - 1:
            content_end = sheet_headers[i + 1]['start']
        else:
            content_end = len(markdown_text)
        
        # Extract sheet content
        sheet_content = markdown_text[content_start:content_end].strip()
        
        # Clean NaN-only lines from the sheet content
        sheet_content = _clean_nan_only_lines(sheet_content)
        
        # Add converted header
        if i > 0:
            converted_text_parts.append(f"\n\n# Sheet {i + 1}: {sheet_name}\n\n")
        else:
            converted_text_parts.append(f"# Sheet 1: {sheet_name}\n\n")
        
        # Add sheet content
        if sheet_content:
            converted_text_parts.append(sheet_content)
        else:
            converted_text_parts.append("(Empty sheet)")
        
        # Calculate statistics for this sheet
        full_sheet_text = sheet_content if sheet_content else "(Empty sheet)"
        char_count = len(full_sheet_text)
        word_count = len(re.findall(r'\b\w+\b', full_sheet_text))
        
        # Count tables (looking for markdown table separators)
        table_count = len(re.findall(r'^\|[\s\-\|]+\|$', full_sheet_text, re.MULTILINE))
        
        print(f"  Found {table_count} table(s), {char_count} characters")
        
        # Store per-sheet content with enhanced metadata
        sheet_data = {
            "page_number": i + 1,
            "sheet_name": sheet_name,
            "text": full_sheet_text,
            "character_count": char_count,
            "word_count": word_count,
            "table_count": table_count
        }
        
        if include_metadata:
            # Create SourceMetadata object for this sheet
            source_meta = SourceMetadata(
                filename=result["filename"],
                chunk_id=f"sheet_{sheet_name.replace(' ', '_')}"
            )
            result["source_metadata"].append(source_meta)
        
        result["pages"].append(sheet_data)
    
    # Join all converted parts
    result["full_text"] = "".join(converted_text_parts)
    
    return result


def extract_excel_for_kg_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Specialized extraction function for the KG Creation Pipeline.
    
    This is a convenience wrapper that ensures all necessary metadata
    is extracted for downstream processing phases.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Extraction result optimized for KG pipeline processing
    """
    return extract_excel_content(
        file_path=file_path,
        method="markitdown",
        fast_mode=True,  # Ignored but kept for compatibility
        export_csv=False,  # Not supported with markitdown
        include_metadata=True
    )


def get_sheet_separators(result: Dict[str, Any]) -> List[str]:
    """
    Extract the sheet header patterns from the full text.
    
    This is useful for the chunking phase to split the document
    by sheet boundaries.
    
    Args:
        result: Extraction result dictionary
        
    Returns:
        List of sheet header strings found in the text
    """
    # Find all sheet headers in the text
    header_pattern = r'# Sheet \d+: [^\n]+'
    headers = re.findall(header_pattern, result['full_text'])
    
    return headers


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = "data/sample.xlsx"
    
    try:
        # Extract content using the KG pipeline function
        result = extract_excel_for_kg_pipeline(excel_file)
        
        print("\n" + "="*50)
        print(f"Extraction complete for: {result['filename']}")
        print(f"Sheets: {result['page_count']}")
        print(f"Total words: {result['word_count']}")
        print(f"Source metadata objects: {len(result['source_metadata'])}")
        
        # Save to markdown file
        output_file = Path(excel_file).stem + "_extracted.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['full_text'])
        print(f"\nMarkdown saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)