"""
Image extraction node for PDF processing.

This node converts PDF pages to images for visual processing by LLMs,
optimizing resolution and file sizes for efficient token usage.
"""

import time
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image

from ..schemas import (
    ExtractionState,
    ImageExtractionResult,
    PageImageData
)

logger = logging.getLogger(__name__)


def extract_image_node(state: ExtractionState) -> Dict[str, Any]:
    """
    Convert PDF pages to images for visual processing.
    
    This node processes the PDF to create high-quality images of each page,
    suitable for multimodal LLM processing.
    
    Args:
        state: Current extraction state containing pdf_path and configuration
        
    Returns:
        Updated state with image_result field populated
    """
    start_time = time.time()
    pdf_path = Path(state["pdf_path"])
    max_pages = state.get("max_pages")
    
    # Default configuration
    dpi = 200  # Good balance between quality and file size
    max_dimension = 2048  # Limit for token optimization
    
    # Initialize result
    image_result = ImageExtractionResult(
        success=False,
        pdf_path=str(pdf_path),
        total_pages=0,
        pages=[],
        extraction_time_ms=0
    )
    
    temp_dir = None
    
    try:
        # Validate file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")
        
        logger.info(f"Starting image extraction from: {pdf_path.name}")
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp(prefix=f"pdf_images_{pdf_path.stem}_")
        temp_path = Path(temp_dir)
        logger.debug(f"Created temp directory: {temp_dir}")
        
        # Open PDF document with pypdfium2
        pdf_document = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(pdf_document)
        
        # Determine pages to process
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
        
        # Calculate scale factor from DPI (72 DPI is default for PDFs)
        scale = dpi / 72.0
        
        # Extract images from each page
        pages_data: List[PageImageData] = []
        
        for page_num in range(pages_to_process):
            try:
                page = pdf_document[page_num]
                
                # Render page to PIL Image using pypdfium2
                # pypdfium2 renders directly to PIL with the scale factor
                img = page.render(scale=scale).to_pil()
                
                # Store original dimensions
                original_width, original_height = img.size
                
                # Resize if needed to control token usage
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    logger.debug(f"Resized page {page_num + 1} from {(original_width, original_height)} to {img.size}")
                
                # Create filename with zero-padding
                padding = len(str(total_pages))
                filename = f"page_{str(page_num + 1).zfill(padding)}.png"
                output_path = temp_path / filename
                
                # Save optimized image
                img.save(str(output_path), "PNG", optimize=True)
                
                # Get file size
                file_size_kb = output_path.stat().st_size / 1024
                
                # Create page data
                page_data = PageImageData(
                    page_number=page_num + 1,
                    image_path=str(output_path),
                    width=img.width,
                    height=img.height,
                    file_size_kb=round(file_size_kb, 2)
                )
                
                pages_data.append(page_data)
                
                logger.debug(f"Created image for page {page_num + 1}: {file_size_kb:.2f} KB")
                
                # Close the page to free memory
                page.close()
                
            except Exception as e:
                logger.warning(f"Failed to process page {page_num + 1}: {e}")
                # Try with lower scale as fallback
                if scale > 1.5:
                    try:
                        logger.info(f"Retrying page {page_num + 1} with lower DPI")
                        page = pdf_document[page_num]
                        fallback_scale = 1.5  # ~108 DPI
                        img = page.render(scale=fallback_scale).to_pil()
                        
                        # Save with fallback settings
                        padding = len(str(total_pages))
                        filename = f"page_{str(page_num + 1).zfill(padding)}.png"
                        output_path = temp_path / filename
                        img.save(str(output_path), "PNG", optimize=True)
                        
                        file_size_kb = output_path.stat().st_size / 1024
                        page_data = PageImageData(
                            page_number=page_num + 1,
                            image_path=str(output_path),
                            width=img.width,
                            height=img.height,
                            file_size_kb=round(file_size_kb, 2)
                        )
                        pages_data.append(page_data)
                        page.close()
                    except Exception as e2:
                        logger.error(f"Failed to process page {page_num + 1} even with fallback: {e2}")
        
        # Close PDF document
        pdf_document.close()
        
        # Calculate total size
        total_size_mb = sum(p.file_size_kb for p in pages_data) / 1024
        logger.info(f"Successfully converted {len(pages_data)} pages to images ({total_size_mb:.2f} MB total)")
        
        # Update result with success
        image_result = ImageExtractionResult(
            success=True,
            pdf_path=str(pdf_path),
            total_pages=total_pages,
            pages=pages_data,
            extraction_time_ms=(time.time() - start_time) * 1000,
            temp_dir=temp_dir
        )
        
    except Exception as e:
        # Log error and update result
        error_message = f"Image extraction failed: {str(e)}"
        logger.error(error_message)
        
        # Clean up temp directory on error
        if temp_dir:
            cleanup_temp_directory(temp_dir)
        
        image_result = ImageExtractionResult(
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
    return {"image_result": image_result}


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up temporary directory and its contents.
    
    Args:
        temp_dir: Path to temporary directory to remove
    """
    try:
        import shutil
        shutil.rmtree(temp_dir)
        logger.debug(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


def optimize_images_for_llm(
    pages: List[PageImageData],
    max_batch_size: int = 5,
    max_batch_size_mb: float = 10.0
) -> List[List[PageImageData]]:
    """
    Group images into batches optimized for LLM processing.
    
    This function groups pages into batches that respect:
    - Maximum number of images per batch
    - Maximum total size per batch
    
    Args:
        pages: List of page image data
        max_batch_size: Maximum number of images per batch
        max_batch_size_mb: Maximum total size per batch in MB
        
    Returns:
        List of page batches optimized for processing
    """
    batches = []
    current_batch = []
    current_size_mb = 0.0
    
    for page in pages:
        page_size_mb = page.file_size_kb / 1024
        
        # Check if adding this page would exceed limits
        if (len(current_batch) >= max_batch_size or 
            (current_size_mb + page_size_mb > max_batch_size_mb and current_batch)):
            # Start new batch
            batches.append(current_batch)
            current_batch = [page]
            current_size_mb = page_size_mb
        else:
            # Add to current batch
            current_batch.append(page)
            current_size_mb += page_size_mb
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Grouped {len(pages)} images into {len(batches)} batches")
    return batches


def create_image_with_annotations(
    image_path: str,
    annotations: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Create an annotated version of an image with highlights or boxes.
    
    Useful for highlighting extracted information in the original document.
    
    Args:
        image_path: Path to original image
        annotations: List of annotation dictionaries with bbox and labels
        output_path: Path to save annotated image
        
    Returns:
        Path to annotated image
    """
    try:
        from PIL import ImageDraw, ImageFont
        
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Apply annotations
        for annotation in annotations:
            bbox = annotation.get("bbox")  # [x1, y1, x2, y2]
            label = annotation.get("label", "")
            color = annotation.get("color", "red")
            
            if bbox:
                # Draw rectangle
                draw.rectangle(bbox, outline=color, width=2)
                
                # Add label if provided
                if label:
                    draw.text((bbox[0], bbox[1] - 20), label, fill=color)
        
        # Save annotated image
        img.save(output_path)
        logger.debug(f"Created annotated image: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return image_path