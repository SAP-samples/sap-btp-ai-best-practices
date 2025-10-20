"""
PDF extraction module for the Knowledge Graph Creation Pipeline.

This module handles PDF text extraction using an image-based approach with
Gemini transcription for better accuracy with tables and complex formatting.
Preserves structural context through page separators for downstream processing.
Part of Phase 1: Raw Text Extraction & Pre-processing.
"""

import re
import base64
import tempfile
import tiktoken
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timezone

import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from ..models.kg_schema import SourceMetadata
from ..llm import create_llm
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Global tokenizer configuration for GPT-4
TOKENIZER_NAME = "cl100k_base"  # GPT-4 tokenizer


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
        backoff_factor: Factor to multiply delay by after each retry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"All {max_retries} attempts failed. Last error: {str(e)}")
            
            raise last_exception
        
        return wrapper
    return decorator


def extract_pdf_content(file_path: str, method: str = "pymupdf", include_metadata: bool = True) -> Dict[str, Any]:
    """
    Extract complete content from a PDF document with enhanced metadata tracking.
    
    This function implements Phase 1 of the KG Creation Pipeline, converting
    PDF files into structured Markdown text with preserved page boundaries.
    Now uses image-based extraction with Gemini transcription for better accuracy.
    
    Args:
        file_path: Path to the PDF file
        method: Extraction method (kept for compatibility, always uses image-based)
        include_metadata: Whether to include detailed metadata for each page
        
    Returns:
        Dictionary with extracted content and metadata:
        - filename: Name of the source file
        - file_path: Full path to the file
        - full_text: Complete extracted text with page separators
        - page_count: Number of pages
        - word_count: Total word count
        - token_count: Estimated token count for LLM processing
        - extraction_method: Method used for extraction
        - pages: List of per-page content with metadata
        - extraction_timestamp: When the extraction occurred
        - source_metadata: List of SourceMetadata objects for each page
        
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
    
    print(f"Extracting complete content from {file_path.name} using image-based transcription...")
    
    result = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "full_text": "",
        "page_count": 0,
        "word_count": 0,
        "token_count": 0,
        "extraction_method": "image-transcription",  # Always use image-based method
        "pages": [],  # Store per-page content with metadata
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_metadata": []  # List of SourceMetadata objects
    }
    
    try:
        # Use temporary directory for image files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Convert PDF to images
            print(f"Converting PDF to images...")
            image_paths = _convert_pdf_to_images(file_path, temp_path)
            result["page_count"] = len(image_paths)
            
            # Transcribe images in parallel with batch processing
            print(f"Transcribing {len(image_paths)} pages using batch processing...")
            transcriptions = _transcribe_images_parallel(image_paths)
            
            # Process transcriptions into expected format
            all_text_parts = []
            
            for page_num, (_, transcribed_text) in enumerate(transcriptions):
                # Convert transcription format to match expected format
                # Remove the transcription header if present
                page_text = transcribed_text
                if page_text.startswith("---\n## Page"):
                    # Extract just the content after the header
                    lines = page_text.split('\n')
                    # Find where the actual content starts (after the --- line)
                    content_start = 0
                    for i, line in enumerate(lines):
                        if i > 0 and line.strip() == '---':
                            content_start = i + 1
                            break
                    if content_start > 0 and content_start < len(lines):
                        page_text = '\n'.join(lines[content_start:]).strip()
                
                # Add page separator in the expected format
                if page_num > 0:
                    all_text_parts.append(f"\n\n--- PAGE {page_num + 1} ---\n\n")
                
                all_text_parts.append(page_text)
                
                # Store per-page content with metadata
                page_data = {
                    "page_number": page_num + 1,
                    "text": page_text,
                    "character_count": len(page_text),
                    "word_count": len(re.findall(r'\b\w+\b', page_text))
                }
                
                if include_metadata:
                    # Create SourceMetadata object for this page
                    source_meta = SourceMetadata(
                        filename=result["filename"],
                        chunk_id=f"page_{page_num + 1}"
                    )
                    result["source_metadata"].append(source_meta)
                
                result["pages"].append(page_data)
    
    except Exception as e:
        raise ValueError(f"Error processing PDF file {file_path}: {str(e)}")
    
    result["full_text"] = "".join(all_text_parts)
    
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


def _convert_pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    """
    Convert PDF pages to PNG images using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        
    Returns:
        List of paths to the created image files
    """
    try:
        # Open PDF
        pdf_document = fitz.open(str(pdf_path))
        total_pages = len(pdf_document)
        
        # Calculate zoom factor from DPI (200 DPI for good quality)
        dpi = 200
        zoom = dpi / 72.0  # Default PDF resolution is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        saved_files = []
        
        for page_num in range(total_pages):
            # Get page
            page = pdf_document[page_num]
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Create filename with zero-padding
            padding = len(str(total_pages))
            filename = f"page_{str(page_num + 1).zfill(padding)}.png"
            output_path = output_dir / filename
            
            # Save the image
            pix.save(str(output_path))
            saved_files.append(output_path)
        
        # Close the PDF
        pdf_document.close()
        
        return saved_files
        
    except Exception as e:
        raise Exception(f"Error converting PDF to images: {e}")


def _transcribe_images_parallel(image_paths: List[Path], max_workers: int = 10) -> List[Tuple[int, str]]:
    """
    Transcribe multiple images in parallel using Gemini.
    
    Args:
        image_paths: List of paths to image files
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of (page_number, transcribed_text) tuples, sorted by page number
    """
    # Initialize Gemini model using the project's LLM factory
    llm = create_llm(model_name="gemini-2.5-flash", temperature=0.1)
    
    # Thread-safe counter for progress tracking
    progress_lock = threading.Lock()
    completed_pages = 0
    
    # Create list of (page_number, image_path) tuples
    image_list = [(i + 1, path) for i, path in enumerate(image_paths)]
    
    total_pages = len(image_list)
    print(f"Processing {total_pages} pages...")
    
    @retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
    def transcribe_single_image(page_num: int, image_path: Path) -> Tuple[int, str]:
        """Transcribe a single image and return its transcription."""
        nonlocal completed_pages
        
        print(f"Processing page {page_num}...")
        
        try:
            # Prepare image
            with Image.open(image_path) as img:
                # Resize if too large to reduce token usage
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Save to bytes
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                image_bytes = buffer.getvalue()
            
            # Encode to base64
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create transcription prompt
            transcription_prompt = """Please transcribe ALL text from this image with the following requirements:

1. Preserve the EXACT content - transcribe every word, number, and symbol you see
2. Maintain the original structure and formatting
3. Format tables properly in Markdown:
   - Use | for column separators
   - Add header separator row with |---|---|
   - Align columns for readability
4. Preserve hierarchical structure (headings, subheadings, bullet points)
5. Keep original line breaks where they appear meaningful
6. For any special formatting (bold, italic), use appropriate Markdown syntax
7. If there are any forms or structured data, represent them clearly
8. Include page numbers, headers, or footers if present

IMPORTANT: Focus on accuracy - every detail matters. If you're unsure about any text, indicate it with [unclear] but still attempt to transcribe it. Do not repeat text or add unnecessary spaces. The generated text should be clean and usable to an LLM.

Transcribe the image now:"""
            
            # Build content for HumanMessage
            content = [
                {"type": "text", "text": transcription_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ]
            
            # Create HumanMessage with multimodal content
            message = HumanMessage(content=content)
            
            # Invoke the model
            response = llm.invoke([message])
            
            # Get transcribed content
            transcribed_content = response.content
            
            # Format the response with page header
            page_header = f"\n\n---\n## Page {page_num}\n---\n\n"
            transcription = page_header + transcribed_content
            
            # Update progress counter thread-safely
            with progress_lock:
                completed_pages += 1
                print(f"Completed page {page_num} ({completed_pages}/{total_pages})")
            
            return (page_num, transcription)
            
        except Exception as e:
            # This exception will be caught by the retry decorator
            print(f"Error transcribing page {page_num}: {e}")
            raise
    
    # Process images in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image tasks
        future_to_page = {
            executor.submit(transcribe_single_image, page_num, image_path): (page_num, image_path)
            for page_num, image_path in image_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num, _ = future_to_page[future]
            
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Failed to transcribe page {page_num} after all retries: {e}")
                # Add error entry for failed page
                error_header = f"\n\n---\n## Page {page_num}\n---\n\n"
                error_content = f"[Error: Failed to transcribe this page after 3 attempts - {str(e)}]\n"
                all_results.append((page_num, error_header + error_content))
    
    # Sort results by page index to maintain order
    all_results.sort(key=lambda x: x[0])
    
    return all_results


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


def extract_pdf_for_kg_pipeline(file_path: str) -> Dict[str, Any]:
    """
    Specialized extraction function for the KG Creation Pipeline.
    
    This is a convenience wrapper that ensures all necessary metadata
    is extracted for downstream processing phases.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extraction result optimized for KG pipeline processing
    """
    return extract_pdf_content(
        file_path=file_path,
        method="image-transcription",  # Always use image-based method
        include_metadata=True
    )


def get_page_separators(result: Dict[str, Any]) -> List[str]:
    """
    Extract the page separator patterns from the full text.
    
    This is useful for the chunking phase to split the document
    by page boundaries.
    
    Args:
        result: Extraction result dictionary
        
    Returns:
        List of page separator strings found in the text
    """
    import re
    
    # Find all page separators in the text
    separator_pattern = r'--- PAGE \d+ ---'
    separators = re.findall(separator_pattern, result['full_text'])
    
    return separators