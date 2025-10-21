"""
PDF Image Extraction Module for Direct Image-to-Node Processing

This module handles PDF conversion to images for direct knowledge graph extraction,
bypassing the text transcription step. It provides efficient image-based processing
for better accuracy with tables, diagrams, and complex layouts.
"""

import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from ..models.kg_schema import SourceMetadata

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def extract_pdf_content_for_nodes(
    file_path: str, 
    dpi: int = 200,
    max_dimension: int = 2048,
    parallel_processing: bool = True,
    max_workers: int = 10
) -> Dict[str, Any]:
    """
    Extract PDF content as images for direct node extraction.
    
    This function converts PDF pages to images without transcription,
    preparing them for direct knowledge graph extraction using multimodal LLMs.
    
    Args:
        file_path: Path to the PDF file
        dpi: DPI for image conversion (default: 200 for good quality)
        max_dimension: Maximum dimension for images to control token usage
        parallel_processing: Enable parallel image processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with image data and metadata:
        - filename: Name of the source file
        - file_path: Full path to the file
        - page_count: Number of pages
        - extraction_method: Always "image-direct"
        - pages: List of page data with image paths and metadata
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
    
    logger.info(f"Extracting PDF as images for direct node extraction: {file_path.name}")
    
    result = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "page_count": 0,
        "extraction_method": "image-direct",
        "pages": [],  # Store per-page image data with metadata
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_metadata": [],  # List of SourceMetadata objects
        "temp_dir": None  # Will store temporary directory path
    }
    
    try:
        # Create a persistent temporary directory for this document
        temp_dir = tempfile.mkdtemp(prefix=f"kg_images_{file_path.stem}_")
        result["temp_dir"] = temp_dir
        temp_path = Path(temp_dir)
        
        # Convert PDF to images
        logger.info(f"Converting PDF to images at {dpi} DPI...")
        image_data = _convert_pdf_to_images_with_metadata(
            file_path, 
            temp_path, 
            dpi=dpi,
            max_dimension=max_dimension
        )
        
        result["page_count"] = len(image_data)
        
        # Process each page's metadata
        for page_info in image_data:
            page_num = page_info["page_number"]
            
            # Create page data structure
            page_data = {
                "page_number": page_num,
                "image_path": str(page_info["image_path"]),
                "dimensions": page_info["dimensions"],
                "file_size_kb": page_info["file_size_kb"],
                "has_text": page_info.get("has_text", True),
                "has_images": page_info.get("has_images", False),
                "rotation": page_info.get("rotation", 0)
            }
            
            # Create SourceMetadata object for this page
            source_meta = SourceMetadata(
                filename=result["filename"],
                chunk_id=f"page_{page_num}"
            )
            result["source_metadata"].append(source_meta)
            
            result["pages"].append(page_data)
        
        logger.info(f"Successfully converted {result['page_count']} pages to images")
        
        # Log summary statistics
        total_size_mb = sum(p["file_size_kb"] for p in result["pages"]) / 1024
        logger.info(f"Total image size: {total_size_mb:.2f} MB")
        
        return result
        
    except Exception as e:
        # Clean up temp directory on error
        if "temp_dir" in result and result["temp_dir"]:
            import shutil
            try:
                shutil.rmtree(result["temp_dir"])
            except:
                pass
        raise ValueError(f"Error processing PDF file {file_path}: {str(e)}")


def _convert_pdf_to_images_with_metadata(
    pdf_path: Path, 
    output_dir: Path,
    dpi: int = 200,
    max_dimension: int = 2048
) -> List[Dict[str, Any]]:
    """
    Convert PDF pages to PNG images with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        dpi: DPI for conversion
        max_dimension: Maximum dimension for images
        
    Returns:
        List of dictionaries with image metadata
    """
    try:
        # Open PDF
        pdf_document = fitz.open(str(pdf_path))
        total_pages = len(pdf_document)
        
        # Calculate zoom factor from DPI
        zoom = dpi / 72.0  # Default PDF resolution is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        image_data = []
        
        for page_num in range(total_pages):
            # Get page
            page = pdf_document[page_num]
            
            # Extract page metadata
            page_info = {
                "page_number": page_num + 1,
                "rotation": page.rotation,
                "has_text": len(page.get_text()) > 0,
                "has_images": len(page.get_images()) > 0
            }
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Create filename with zero-padding
            padding = len(str(total_pages))
            filename = f"page_{str(page_num + 1).zfill(padding)}.png"
            output_path = output_dir / filename
            
            # Convert to PIL Image for processing
            img_data = pix.pil_tobytes(format="PNG")
            img = Image.open(BytesIO(img_data))
            
            # Resize if needed to control token usage
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                logger.debug(f"Resized page {page_num + 1} from {pix.width}x{pix.height} to {img.size}")
            
            # Save optimized image
            img.save(str(output_path), "PNG", optimize=True)
            
            # Get file size
            file_size_kb = output_path.stat().st_size / 1024
            
            # Store metadata
            page_info.update({
                "image_path": output_path,
                "dimensions": {"width": img.width, "height": img.height},
                "original_dimensions": {"width": pix.width, "height": pix.height},
                "file_size_kb": round(file_size_kb, 2)
            })
            
            image_data.append(page_info)
        
        # Close the PDF
        pdf_document.close()
        
        return image_data
        
    except Exception as e:
        raise Exception(f"Error converting PDF to images: {e}")


def cleanup_temp_images(extraction_result: Dict[str, Any]) -> None:
    """
    Clean up temporary image files after processing.
    
    Args:
        extraction_result: Result from extract_pdf_content_for_nodes
    """
    if "temp_dir" in extraction_result and extraction_result["temp_dir"]:
        import shutil
        try:
            shutil.rmtree(extraction_result["temp_dir"])
            logger.info(f"Cleaned up temporary images from {extraction_result['temp_dir']}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


def get_image_batch_groups(
    pages: List[Dict[str, Any]], 
    max_batch_size: int = 5,
    max_batch_size_mb: float = 10.0
) -> List[List[Dict[str, Any]]]:
    """
    Group pages into batches for efficient processing.
    
    Args:
        pages: List of page data from extraction
        max_batch_size: Maximum number of pages per batch
        max_batch_size_mb: Maximum total size per batch in MB
        
    Returns:
        List of page batches
    """
    batches = []
    current_batch = []
    current_size_mb = 0.0
    
    for page in pages:
        page_size_mb = page["file_size_kb"] / 1024
        
        # Check if adding this page would exceed limits
        if (len(current_batch) >= max_batch_size or 
            current_size_mb + page_size_mb > max_batch_size_mb):
            # Start new batch
            if current_batch:
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
    
    logger.info(f"Grouped {len(pages)} pages into {len(batches)} batches")
    return batches


# Re-export the base image conversion function for compatibility
from .pdf_extractor_llm import _convert_pdf_to_images