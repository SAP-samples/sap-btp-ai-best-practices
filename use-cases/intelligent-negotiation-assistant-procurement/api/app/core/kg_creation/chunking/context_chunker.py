"""
Context-aware chunking module for the Knowledge Graph Creation Pipeline.

This module implements Phase 2 of the pipeline, dividing documents into
logically coherent chunks based on structural boundaries (pages/sheets).
"""

import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..models.kg_schema import Chunk, SourceMetadata


logger = logging.getLogger(__name__)


class ContextAwareChunker:
    """
    Splits documents into chunks based on their structural boundaries.
    
    This class maintains contextual integrity by splitting along natural
    document boundaries (pages for PDFs, sheets for Excel files).
    """
    
    # Regex patterns for different document separators
    PDF_PAGE_PATTERN = r'--- PAGE (\d+) ---'
    EXCEL_SHEET_PATTERN = r'# Sheet (\d+): ([^\n]+)'
    
    def __init__(self):
        """Initialize the chunker."""
        self.pdf_regex = re.compile(self.PDF_PAGE_PATTERN)
        self.excel_regex = re.compile(self.EXCEL_SHEET_PATTERN)
    
    def chunk_document(self, 
                      extraction_result: Dict[str, Any], 
                      source_type: Optional[str] = None) -> List[Chunk]:
        """
        Chunk a document based on its structural boundaries.
        
        Args:
            extraction_result: The result from pdf_extractor or excel_extractor
            source_type: Optional hint about source type ('pdf' or 'excel')
                        If not provided, will be auto-detected
        
        Returns:
            List of Chunk objects with proper metadata
            
        Raises:
            ValueError: If the extraction result is invalid or empty
        """
        if not extraction_result or 'full_text' not in extraction_result:
            raise ValueError("Invalid extraction result: missing 'full_text'")
        
        full_text = extraction_result['full_text']
        filename = extraction_result.get('filename', 'unknown')
        
        if not full_text.strip():
            raise ValueError("Empty document text")
        
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(full_text, filename)
        
        logger.info(f"Chunking {source_type} document: {filename}")
        
        # Choose chunking strategy based on source type
        if source_type == 'pdf':
            chunks = self._chunk_pdf(full_text, filename)
        elif source_type == 'excel':
            chunks = self._chunk_excel(full_text, filename)
        else:
            # Fallback: treat entire document as single chunk
            logger.warning(f"Unknown source type '{source_type}', treating as single chunk")
            chunks = self._create_single_chunk(full_text, filename)
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def _detect_source_type(self, text: str, filename: str) -> str:
        """
        Auto-detect the source type based on content and filename.
        
        Args:
            text: The document text
            filename: The source filename
            
        Returns:
            'pdf', 'excel', or 'unknown'
        """
        # Check filename extension
        file_ext = Path(filename).suffix.lower()
        if file_ext == '.pdf':
            return 'pdf'
        elif file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            return 'excel'
        
        # Check content patterns
        if self.pdf_regex.search(text):
            return 'pdf'
        elif self.excel_regex.search(text):
            return 'excel'
        
        return 'unknown'
    
    def _chunk_pdf(self, text: str, filename: str) -> List[Chunk]:
        """
        Chunk PDF content by page boundaries.
        
        Args:
            text: The full PDF text with page separators
            filename: The source filename
            
        Returns:
            List of Chunk objects, one per page
        """
        chunks = []
        
        # Split by page separators
        parts = self.pdf_regex.split(text)
        
        # Handle case where no separators found
        if len(parts) == 1:
            return self._create_single_chunk(text, filename)
        
        # Process parts (format: ['pre-content', 'page_num', 'content', 'page_num', 'content', ...])
        current_page = 1
        
        # Handle any content before first page marker
        if parts[0].strip():
            chunk = self._create_chunk(
                text=parts[0].strip(),
                filename=filename,
                chunk_id=f"page_1"
            )
            chunks.append(chunk)
        
        # Process remaining parts
        i = 1
        while i < len(parts) - 1:
            page_num = int(parts[i])
            page_content = parts[i + 1] if i + 1 < len(parts) else ""
            
            if page_content.strip():
                chunk = self._create_chunk(
                    text=page_content.strip(),
                    filename=filename,
                    chunk_id=f"page_{page_num}"
                )
                chunks.append(chunk)
            
            i += 2
        
        return chunks
    
    def _chunk_excel(self, text: str, filename: str) -> List[Chunk]:
        """
        Chunk Excel content by sheet boundaries.
        
        Args:
            text: The full Excel text with sheet headers
            filename: The source filename
            
        Returns:
            List of Chunk objects, one per sheet
        """
        chunks = []
        
        # Find all sheet headers and their positions
        matches = list(self.excel_regex.finditer(text))
        
        if not matches:
            return self._create_single_chunk(text, filename)
        
        # Process each sheet
        for i, match in enumerate(matches):
            sheet_num = match.group(1)
            sheet_name = match.group(2).strip()
            
            # Determine content boundaries
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            # Extract sheet content (including the header)
            sheet_content = text[start_pos:end_pos].strip()
            
            if sheet_content:
                # Create chunk ID using sheet name for better readability
                safe_sheet_name = re.sub(r'[^\w\-_]', '_', sheet_name)
                chunk_id = f"sheet_{safe_sheet_name}"
                
                chunk = self._create_chunk(
                    text=sheet_content,
                    filename=filename,
                    chunk_id=chunk_id
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_single_chunk(self, text: str, filename: str) -> List[Chunk]:
        """
        Create a single chunk for the entire document.
        
        This is used as a fallback when no structural boundaries are found.
        
        Args:
            text: The full document text
            filename: The source filename
            
        Returns:
            List containing a single Chunk object
        """
        chunk = self._create_chunk(
            text=text.strip(),
            filename=filename,
            chunk_id="full_document"
        )
        return [chunk]
    
    def _create_chunk(self, text: str, filename: str, chunk_id: str) -> Chunk:
        """
        Create a Chunk object with proper metadata.
        
        Args:
            text: The chunk text
            filename: The source filename
            chunk_id: The unique chunk identifier
            
        Returns:
            Chunk object
        """
        # Create source metadata
        metadata = SourceMetadata(
            filename=filename,
            chunk_id=chunk_id
        )
        
        # Create and return chunk
        return Chunk(
            chunk_id=f"{filename}:{chunk_id}",
            chunk_text=text,
            metadata=metadata
        )
    
    def merge_chunks(self, chunks: List[Chunk], max_size: Optional[int] = None) -> List[Chunk]:
        """
        Optionally merge small chunks to optimize for LLM processing.
        
        This is useful when individual pages/sheets are very small and
        could be processed more efficiently together.
        
        Args:
            chunks: List of chunks to potentially merge
            max_size: Maximum size (in characters) for merged chunks
                     If None, no merging is performed
        
        Returns:
            List of potentially merged chunks
        """
        if max_size is None or not chunks:
            return chunks
        
        merged_chunks = []
        current_text = []
        current_size = 0
        current_ids = []
        
        for chunk in chunks:
            chunk_size = len(chunk.chunk_text)
            
            # Check if adding this chunk would exceed max size
            if current_text and current_size + chunk_size > max_size:
                # Create merged chunk from accumulated content
                merged_chunk = self._create_merged_chunk(
                    texts=current_text,
                    chunk_ids=current_ids,
                    original_chunks=chunks
                )
                merged_chunks.append(merged_chunk)
                
                # Reset accumulator
                current_text = [chunk.chunk_text]
                current_size = chunk_size
                current_ids = [chunk.chunk_id]
            else:
                # Add to current accumulator
                current_text.append(chunk.chunk_text)
                current_size += chunk_size
                current_ids.append(chunk.chunk_id)
        
        # Handle remaining content
        if current_text:
            merged_chunk = self._create_merged_chunk(
                texts=current_text,
                chunk_ids=current_ids,
                original_chunks=chunks
            )
            merged_chunks.append(merged_chunk)
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    
    def _create_merged_chunk(self, 
                           texts: List[str], 
                           chunk_ids: List[str],
                           original_chunks: List[Chunk]) -> Chunk:
        """
        Create a merged chunk from multiple text segments.
        
        Args:
            texts: List of text segments to merge
            chunk_ids: List of original chunk IDs
            original_chunks: Original chunk objects (for metadata)
            
        Returns:
            Merged Chunk object
        """
        # Combine texts with clear boundaries
        merged_text = "\n\n[CHUNK BOUNDARY]\n\n".join(texts)
        
        # Create merged chunk ID
        if len(chunk_ids) == 1:
            merged_id = chunk_ids[0]
        else:
            # Extract the base filename from first chunk
            base_parts = chunk_ids[0].split(':')
            filename = base_parts[0]
            
            # Create range indicator
            first_part = base_parts[1] if len(base_parts) > 1 else "1"
            last_part = chunk_ids[-1].split(':')[1] if ':' in chunk_ids[-1] else str(len(chunk_ids))
            
            merged_id = f"{filename}:merged_{first_part}_to_{last_part}"
        
        # Use metadata from first chunk (they should all have same filename)
        metadata = original_chunks[0].metadata
        
        return Chunk(
            chunk_id=merged_id,
            chunk_text=merged_text,
            metadata=metadata
        )


def chunk_extraction_result(extraction_result: Dict[str, Any], 
                          source_type: Optional[str] = None,
                          merge_small_chunks: bool = False,
                          max_chunk_size: Optional[int] = None) -> List[Chunk]:
    """
    Convenience function to chunk an extraction result.
    
    Args:
        extraction_result: Result from pdf_extractor or excel_extractor
        source_type: Optional source type hint ('pdf' or 'excel')
        merge_small_chunks: Whether to merge small chunks
        max_chunk_size: Maximum size for merged chunks (if merging enabled)
        
    Returns:
        List of Chunk objects
    """
    chunker = ContextAwareChunker()
    chunks = chunker.chunk_document(extraction_result, source_type)
    
    if merge_small_chunks and max_chunk_size:
        chunks = chunker.merge_chunks(chunks, max_chunk_size)
    
    return chunks