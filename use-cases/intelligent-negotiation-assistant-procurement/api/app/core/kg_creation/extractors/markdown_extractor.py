"""
Markdown extraction module for the Knowledge Graph Creation Pipeline.

This module handles markdown file reading for direct markdown input,
bypassing the conversion step since the content is already in markdown format.
Part of Phase 1: Raw Text Extraction & Pre-processing.
"""

import os
import sys
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import markdown_processor
sys.path.append(str(Path(__file__).parent.parent.parent))

from markdown_processor import extract_markdown_content

from ..models.kg_schema import SourceMetadata


def extract_markdown_for_kg_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Extract content from a markdown file for the KG creation pipeline.
    
    This function is the interface between markdown files and the KG pipeline,
    ensuring compatibility with the existing extraction pipeline structure.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with extracted content compatible with KG pipeline:
        - filename: Name of the source file
        - file_path: Full path to the file
        - full_text: Complete markdown content
        - page_count: Number of "pages" (always 1 for markdown)
        - word_count: Total word count
        - token_count: Estimated token count for LLM processing
        - extraction_method: Method used (direct_markdown)
        - pages: List with single page entry
        - extraction_timestamp: When the extraction occurred
        - file_type: 'markdown'
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be processed
    """
    
    # Extract content using the markdown processor
    extraction_result = extract_markdown_content(file_path)
    
    # Add additional metadata for KG pipeline compatibility
    extraction_result['extraction_timestamp'] = datetime.now().isoformat()
    extraction_result['file_type'] = 'markdown'
    
    # Log extraction summary
    print(f"Markdown extraction complete for: {extraction_result['filename']}")
    print(f"  - Words: {extraction_result['word_count']}")
    print(f"  - Tokens: {extraction_result['token_count']}")
    
    return extraction_result


def extract_markdown_for_kg_pipeline_with_metadata(
    file_path: str, 
    chunk_prefix: str = "md"
) -> Dict[str, Any]:
    """
    Extract markdown content with enhanced metadata tracking.
    
    This variant provides additional metadata formatting consistent
    with PDF and Excel extractors.
    
    Args:
        file_path: Path to the markdown file
        chunk_prefix: Prefix for chunk IDs (default: "md")
        
    Returns:
        Enhanced extraction result with structured metadata
    """
    
    # Get base extraction
    result = extract_markdown_for_kg_pipeline(file_path)
    
    # Add chunk metadata structure
    file_stem = Path(file_path).stem
    result['chunk_metadata'] = {
        'prefix': chunk_prefix,
        'base_id': f"{file_stem}_{chunk_prefix}",
        'total_chunks': 1  # Markdown is treated as single chunk
    }
    
    return result