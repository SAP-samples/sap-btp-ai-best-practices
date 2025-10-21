"""
Simplified PDF Extractor using direct image and text processing.

This module provides a streamlined PDF extraction system that converts
PDF pages to images and also extracts text using pypdfium2. Both the
extracted text and images are then provided to a vision-capable LLM as
additional context for improved accuracy.

No LangGraph, no parallel nodes, just simple and fast extraction.
"""

import logging
import base64
import tempfile
import shutil
import gc
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pypdfium2 as pdfium
from PIL import Image
from langchain.schema import HumanMessage, SystemMessage

from .common import make_llm

logger = logging.getLogger(__name__)


class SimpleImageExtractor:
    """
    Simplified PDF extractor that uses image-only processing.
    
    This class provides a direct path from PDF to answer without
    the complexity of LangGraph nodes and state management.
    """
    
    def __init__(
        self,
        max_dimension: int = 2048,
        dpi: int = 200,
        provider: str = "openai",
        model_name: str = "gpt-4.1",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_text_pages: int = 10,
        max_chars_per_page: int = 50000
    ):
        """
        Initialize the simple image extractor.
        
        Args:
            max_dimension: Maximum dimension for image resizing
            dpi: DPI for PDF to image conversion
            provider: LLM provider (default: openai)
            model_name: Model name (default: gpt-4.1)
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
            max_text_pages: Maximum number of pages of text to include in prompt
            max_chars_per_page: Max characters per page to include in prompt
        """
        self.max_dimension = max_dimension
        self.dpi = dpi
        self.max_text_pages = max_text_pages
        self.max_chars_per_page = max_chars_per_page
        
        # Initialize LLM with specified configuration
        self.llm = make_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"Initialized SimpleImageExtractor with {provider}/{model_name}")
    
    def extract(
        self,
        pdf_path: str,
        question: str,
        language: str = "es",
        max_pages: Optional[int] = None,
        pages_per_batch: int = 5
    ) -> Dict[str, Any]:
        """
        Extract information from PDF by answering a question.
        
        Args:
            pdf_path: Path to the PDF file
            question: Question to answer from the PDF
            language: Language for response (es/en)
            max_pages: Maximum number of pages to process
            pages_per_batch: Number of pages to process in each LLM call
            
        Returns:
            Dictionary with extraction results
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        
        # Validate input
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {pdf_path.suffix}")
        
        logger.info(f"Starting simple extraction from: {pdf_path.name}")
        logger.info(f"Question: {question}")
        
        temp_dir = None
        
        try:
            # Create temporary directory for images
            temp_dir = tempfile.mkdtemp(prefix=f"simple_extract_{pdf_path.stem}_")
            logger.debug(f"Created temp directory: {temp_dir}")
            
            # Convert PDF to images
            image_paths = self._pdf_to_images(
                pdf_path=str(pdf_path),
                temp_dir=temp_dir,
                max_pages=max_pages
            )
            
            if not image_paths:
                raise ValueError("No images could be extracted from PDF")
            
            logger.info(f"Converted {len(image_paths)} pages to images")
            
            # Extract text for the same set of pages (best-effort)
            page_texts: List[str] = []
            try:
                page_texts = self._extract_text(
                    pdf_path=str(pdf_path),
                    max_pages=len(image_paths)
                )
                logger.info(f"Extracted text from {len(page_texts)} pages")
            except Exception as e:
                logger.warning(f"Text extraction failed, proceeding with images only: {e}")
            
            # Process images in batches if needed
            if len(image_paths) <= pages_per_batch:
                # Process all pages in one call
                answer = self._process_images(
                    image_paths=image_paths,
                    question=question,
                    language=language,
                    page_texts_full=page_texts if page_texts else None
                )
            else:
                # Process in batches and combine results
                answer = self._process_images_in_batches(
                    image_paths=image_paths,
                    question=question,
                    language=language,
                    batch_size=pages_per_batch,
                    page_texts_full=page_texts if page_texts else None
                )
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return successful result
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "pages_processed": len(image_paths),
                "processing_time_ms": round(processing_time_ms, 2),
                "extraction_method": "simple_text_image",
                "model": "gpt-4.1"
            }
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": False,
                "question": question,
                "answer": "",
                "error": str(e),
                "processing_time_ms": round(processing_time_ms, 2)
            }
            
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
    
    def _pdf_to_images(
        self,
        pdf_path: str,
        temp_dir: str,
        max_pages: Optional[int] = None
    ) -> List[str]:
        """
        Convert PDF pages to images using pypdfium2.
        
        Args:
            pdf_path: Path to PDF file
            temp_dir: Directory to save images
            max_pages: Maximum pages to convert
            
        Returns:
            List of image file paths
        """
        image_paths = []
        pdf_document = None
        
        try:
            # Open PDF document with pypdfium2
            pdf_document = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf_document)
            
            # Determine pages to process
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            # Calculate scale factor from DPI (72 DPI is default for PDFs)
            scale = self.dpi / 72.0
            
            logger.info(f"Converting {pages_to_process} pages at {self.dpi} DPI (scale={scale:.2f})")
            
            for page_num in range(pages_to_process):
                try:
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Render page to PIL Image with lower memory usage
                    # pypdfium2 renders directly to PIL with the scale factor
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    
                    # If max_dimension is set, resize if needed
                    if self.max_dimension and max(pil_image.size) > self.max_dimension:
                        pil_image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                        logger.debug(f"Resized page {page_num + 1} to fit within {self.max_dimension}px")
                    
                    # Save image to temp directory
                    image_path = Path(temp_dir) / f"page_{page_num + 1:03d}.png"
                    pil_image.save(str(image_path), "PNG", optimize=True)
                    
                    image_paths.append(str(image_path))
                    
                    logger.debug(f"Saved page {page_num + 1} as image: {image_path.name}")
                    
                    # Clean up to free memory immediately
                    del pil_image
                    del bitmap
                    page.close()
                    gc.collect()  # Force garbage collection
                    
                except Exception as e:
                    logger.warning(f"Failed to render page {page_num + 1}: {e}")
                    # Try with lower DPI as fallback
                    if scale > 1.5:
                        try:
                            logger.info(f"Retrying page {page_num + 1} with lower DPI")
                            page = pdf_document[page_num]
                            fallback_scale = 1.5  # ~108 DPI
                            pil_image = page.render(scale=fallback_scale).to_pil()
                            
                            image_path = Path(temp_dir) / f"page_{page_num + 1:03d}.png"
                            pil_image.save(str(image_path), "PNG", optimize=True)
                            image_paths.append(str(image_path))
                            page.close()
                        except Exception as e2:
                            logger.error(f"Failed to render page {page_num + 1} even with fallback: {e2}")
            
        except Exception as e:
            logger.error(f"Error opening PDF document: {e}")
            raise ValueError(f"Failed to process PDF: {e}")
        
        finally:
            # Close the PDF document to free resources
            if pdf_document:
                pdf_document.close()
        
        return image_paths
    
    def _prepare_image(self, image_path: str) -> str:
        """
        Prepare image for LLM processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with Image.open(image_path) as img:
            # Store original size for logging
            original_size = img.size
            
            # Resize if too large
            if max(img.size) > self.max_dimension:
                img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {original_size} to {img.size}")
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            image_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _extract_text(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None
    ) -> List[str]:
        """
        Extract text from the PDF using pypdfium2.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            
        Returns:
            List of extracted text per page (1:1 with processed pages)
        """
        texts: List[str] = []
        pdf_document = None
        
        try:
            # Open PDF document
            pdf_document = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf_document)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            logger.info(f"Extracting text from {pages_to_process} pages")
            
            for page_num in range(pages_to_process):
                try:
                    page = pdf_document[page_num]
                    textpage = page.get_textpage()
                    text = textpage.get_text_bounded()
                    texts.append(text or "")
                    textpage.close()
                    page.close()
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    texts.append("")
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            raise
        finally:
            if pdf_document:
                pdf_document.close()
        
        return texts
    
    def _build_multimodal_message(
        self,
        question: str,
        image_data_list: List[str],
        language: str,
        page_text_map: Optional[Dict[int, str]] = None
    ) -> HumanMessage:
        """
        Build multimodal message for LLM with optional text context.
        
        Args:
            question: Question to answer
            image_data_list: List of base64 encoded images
            language: Response language
            page_text_map: Optional mapping of page_number -> extracted text
            
        Returns:
            HumanMessage with text and images
        """
        # Create prompt
        lang_instruction = {
            "es": "Responde ÚNICAMENTE en español.",
            "en": "Respond ONLY in English."
        }.get(language, "Respond in the specified language.")
        
        prompt = f"""You are a precise data extraction system. Extract and return ONLY the exact information requested.

{lang_instruction}

CRITICAL INSTRUCTIONS:
1. Answer DIRECTLY with the exact information requested - nothing more, nothing less
2. DO NOT add explanations, context, or additional commentary
3. DO NOT say "The answer is..." or "According to the document..." - just provide the answer
4. If the information is not found, respond ONLY with: "No encontrado" (Spanish) or "Not found" (English)
5. For multiple values, list them separated by commas
6. Return numbers and dates exactly as they appear in the document
7. For names, return the complete name as it appears

Question: {question}

First, use the extracted text content below. If insufficient, corroborate with the images. Provide ONLY the final answer."""
        
        # Build multimodal content
        text_parts: List[str] = [prompt]
        
        # Append extracted text content if available
        if page_text_map:
            text_parts.append("\n\n=== EXTRACTED TEXT CONTENT ===")
            # Sort by page number and limit pages
            items = sorted(page_text_map.items(), key=lambda kv: kv[0])
            for idx, (page_num, page_text) in enumerate(items):
                if idx >= self.max_text_pages:
                    break
                if page_text and page_text.strip():
                    snippet = page_text[: self.max_chars_per_page]
                    text_parts.append(f"\n--- Page {page_num} ---\n{snippet}")
        
        # Add a lead-in before images
        text_parts.append("\n\n=== DOCUMENT IMAGES ===\nPlease analyze these document images to answer the question:")
        
        content = [{"type": "text", "text": "".join(text_parts)}]
        
        for i, image_data in enumerate(image_data_list):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })
            logger.debug(f"Added image {i + 1} to message")
        
        return HumanMessage(content=content)
    
    def _process_images(
        self,
        image_paths: List[str],
        question: str,
        language: str,
        page_texts_full: Optional[List[str]] = None
    ) -> str:
        """
        Process images with LLM to answer question.
        
        Args:
            image_paths: Paths to image files
            question: Question to answer
            language: Response language
            page_texts_full: Optional list of text per page aligned to original PDF order
            
        Returns:
            Answer from LLM
        """
        # Prepare images
        image_data_list = []
        page_text_map: Dict[int, str] = {}
        for image_path in image_paths:
            image_data = self._prepare_image(image_path)
            image_data_list.append(image_data)
            # Derive page number from filename like page_001.png
            if page_texts_full:
                try:
                    match = re.search(r"page_(\d+)", Path(image_path).stem)
                    if match:
                        page_num = int(match.group(1))
                        # Map to extracted text (1-based index)
                        if 0 < page_num <= len(page_texts_full):
                            page_text_map[page_num] = page_texts_full[page_num - 1]
                except Exception as e:
                    logger.debug(f"Failed to map text for {image_path}: {e}")
        
        # Build multimodal message
        message = self._build_multimodal_message(
            question=question,
            image_data_list=image_data_list,
            language=language,
            page_text_map=page_text_map if page_text_map else None
        )
        
        # Get response from LLM
        logger.info(f"Sending {len(image_data_list)} images to LLM")
        response = self.llm.invoke([message])
        
        # Extract answer
        answer = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Received answer: {answer[:100]}...")
        
        return answer
    
    def _process_images_in_batches(
        self,
        image_paths: List[str],
        question: str,
        language: str,
        batch_size: int = 5,
        page_texts_full: Optional[List[str]] = None
    ) -> str:
        """
        Process images in batches for large documents.
        
        Args:
            image_paths: Paths to image files
            question: Question to answer
            language: Response language
            batch_size: Number of images per batch
            page_texts_full: Optional list of text per page aligned to original PDF order
            
        Returns:
            Combined answer from all batches
        """
        answers = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_paths)} images")
            
            answer = self._process_images(
                image_paths=batch_paths,
                question=question,
                language=language,
                page_texts_full=page_texts_full
            )
            
            # Only add non-empty answers that aren't "not found"
            if answer and answer.lower() not in ["no encontrado", "not found"]:
                answers.append(answer)
        
        # Combine answers
        if not answers:
            return "No encontrado" if language == "es" else "Not found"
        elif len(answers) == 1:
            return answers[0]
        else:
            # If multiple batches found answers, combine them
            # This might need refinement based on the type of question
            return ", ".join(set(answers))  # Use set to avoid duplicates
    
    def extract_batch(
        self,
        pdf_path: str,
        questions: List[str],
        language: str = "es",
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract multiple pieces of information from a single PDF.
        
        This is more efficient than calling extract multiple times
        as it reuses the converted images.
        
        Args:
            pdf_path: Path to the PDF file
            questions: List of questions to answer
            language: Language for responses
            max_pages: Maximum pages to process
            
        Returns:
            Dictionary with all answers
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        
        logger.info(f"Starting batch extraction for {len(questions)} questions")
        
        temp_dir = None
        results = {}
        
        try:
            # Create temporary directory for images
            temp_dir = tempfile.mkdtemp(prefix=f"batch_extract_{pdf_path.stem}_")
            
            # Convert PDF to images once
            image_paths = self._pdf_to_images(
                pdf_path=str(pdf_path),
                temp_dir=temp_dir,
                max_pages=max_pages
            )
            
            # Prepare images once
            image_data_list = []
            for image_path in image_paths:
                image_data = self._prepare_image(image_path)
                image_data_list.append(image_data)
            
            logger.info(f"Prepared {len(image_data_list)} images for batch processing")
            
            # Extract text once for all pages (best-effort)
            page_texts: List[str] = []
            try:
                page_texts = self._extract_text(
                    pdf_path=str(pdf_path),
                    max_pages=len(image_paths)
                )
                logger.info(f"Extracted text from {len(page_texts)} pages (batch)")
            except Exception as e:
                logger.warning(f"Text extraction failed for batch, proceeding with images only: {e}")
            
            # Process each question
            for question in questions:
                try:
                    # Build message for this question
                    # Map page number -> text for prompt inclusion
                    page_text_map: Dict[int, str] = {
                        (idx + 1): txt for idx, txt in enumerate(page_texts)
                        if txt and txt.strip()
                    } if page_texts else None
                    message = self._build_multimodal_message(
                        question=question,
                        image_data_list=image_data_list,
                        language=language,
                        page_text_map=page_text_map
                    )
                    
                    # Get answer
                    response = self.llm.invoke([message])
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                    results[question] = {
                        "answer": answer,
                        "success": True
                    }
                    
                    logger.debug(f"Answered: {question[:50]}... -> {answer[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to answer '{question}': {e}")
                    results[question] = {
                        "answer": "",
                        "success": False,
                        "error": str(e)
                    }
            
            # Calculate total processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "results": results,
                "pages_processed": len(image_paths),
                "questions_processed": len(questions),
                "processing_time_ms": round(processing_time_ms, 2)
            }
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }
            
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")