"""
Reducer node for combining text and image extraction results.

This node uses an LLM to process both text and image data to answer
specific questions about the PDF content.
"""

import time
import logging
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.messages import BaseMessage

from ..common import make_llm
from ..schemas import (
    ExtractionState,
    FinalExtractionResult,
    TextExtractionResult,
    ImageExtractionResult,
    PageImageData
)

logger = logging.getLogger(__name__)


def reducer_node(state: ExtractionState) -> Dict[str, Any]:
    """
    Combine text and image extraction results to answer questions.
    
    This node processes both text and visual data using an LLM to provide
    accurate answers to user questions, even when text extraction is unreliable.
    
    Args:
        state: Current extraction state with text_result and image_result
        
    Returns:
        Updated state with final_result field populated
    """
    start_time = time.time()
    
    # Extract inputs from state
    question = state["question"]
    language = state.get("language", "es")
    temperature = state.get("temperature", 0.2)
    text_result: Optional[TextExtractionResult] = state.get("text_result")
    image_result: Optional[ImageExtractionResult] = state.get("image_result")
    
    # Initialize final result
    final_result = FinalExtractionResult(
        success=False,
        question=question,
        answer="",
        source_pages=[],
        extraction_metadata={},
        processing_time_ms=0
    )
    
    try:
        # Check if we have any data to work with
        if not text_result and not image_result:
            raise ValueError("No extraction results available from either text or image nodes")
        
        # Determine which extraction method to prioritize
        use_vision = False
        if image_result and image_result.success and image_result.pages:
            use_vision = True
            logger.info("Using vision-capable LLM for multimodal processing")
        elif text_result and text_result.success and text_result.pages:
            logger.info("Using text-only processing")
        else:
            raise ValueError("Both extraction methods failed to produce usable data")
        
        # Initialize LLM based on capabilities needed
        if use_vision:
            # Use vision-capable model
            llm = make_llm(
                provider="openai",  # OpenAI GPT-4o supports vision
                model_name="gpt-4.1",
                temperature=temperature,
                max_tokens=2048
            )
        else:
            # Use standard text model
            llm = make_llm(
                provider="openai",
                model_name="gpt-4.1",
                temperature=temperature,
                max_tokens=2048
            )
        
        # Prepare messages for LLM
        messages = _prepare_messages(
            question=question,
            language=language,
            text_result=text_result,
            image_result=image_result,
            use_vision=use_vision
        )
        
        # Invoke LLM
        logger.info(f"Invoking LLM to answer: {question}")
        response = llm.invoke(messages)
        
        # Parse response
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Determine source pages
        source_pages = _extract_source_pages(text_result, image_result)
        
        # Prepare metadata
        extraction_metadata = {
            "extraction_method": "multimodal" if use_vision else "text_only",
            "text_extraction_success": text_result.success if text_result else False,
            "image_extraction_success": image_result.success if image_result else False,
            "pages_processed": len(source_pages),
            "llm_model": "gpt-4o" if use_vision else "gpt-4o-mini",
            "language": language
        }
        
        # Create successful result
        final_result = FinalExtractionResult(
            success=True,
            question=question,
            answer=answer,
            source_pages=source_pages,
            extraction_metadata=extraction_metadata,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        logger.info(f"Successfully answered question from {len(source_pages)} pages")
        
    except Exception as e:
        # Log error and create failed result
        error_message = f"Reducer processing failed: {str(e)}"
        logger.error(error_message)
        
        final_result = FinalExtractionResult(
            success=False,
            question=question,
            answer="",
            source_pages=[],
            extraction_metadata={
                "error": error_message
            },
            processing_time_ms=(time.time() - start_time) * 1000,
            error_message=error_message
        )
        
        # Add error to state
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error_message)
    
    # Clean up temporary images if configured
    if image_result and image_result.temp_dir:
        try:
            import shutil
            shutil.rmtree(image_result.temp_dir)
            logger.info("Cleaned up temporary image files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")
    
    # Return updated state
    return {"final_result": final_result}


def _prepare_messages(
    question: str,
    language: str,
    text_result: Optional[TextExtractionResult],
    image_result: Optional[ImageExtractionResult],
    use_vision: bool
) -> List[BaseMessage]:
    """
    Prepare messages for LLM invocation.
    
    Args:
        question: User's question to answer
        language: Language for response
        text_result: Text extraction results
        image_result: Image extraction results
        use_vision: Whether to use vision capabilities
        
    Returns:
        List of messages for LLM
    """
    messages = []
    
    # System prompt
    lang_instruction = {
        "es": "Responde ÚNICAMENTE en español.",
        "en": "Respond ONLY in English."
    }.get(language, "Respond in the specified language.")
    
    system_prompt = f"""You are a precise data extraction system. Extract and return ONLY the exact information requested.

{lang_instruction}

CRITICAL INSTRUCTIONS:
1. Answer DIRECTLY with the exact information requested - nothing more, nothing less
2. DO NOT add explanations, context, or additional commentary
3. DO NOT say "The answer is..." or "According to the document..." - just provide the answer
4. If the information is not found, respond ONLY with: "No encontrado" (Spanish) or "Not found" (English)
5. For multiple values, list them separated by commas
6. Return numbers and dates exactly as they appear in the document
7. For names, return the complete name as it appears

Examples:
Question: "¿Cuál es el nombre del cliente?"
Answer: "Juan Pérez García"

Question: "¿Cuál es el RFC?"
Answer: "ABC123456789"

Question: "¿Cuál es el monto total?"
Answer: "$15,234.50"

REMEMBER: Your response should contain ONLY the requested data, no explanations."""
    
    messages.append(SystemMessage(content=system_prompt))
    
    # Prepare human message content
    human_content_parts = []
    
    # Add question
    human_content_parts.append(f"Question: {question}\n\n")
    
    # Add text content if available
    if text_result and text_result.success and text_result.pages:
        human_content_parts.append("=== EXTRACTED TEXT CONTENT ===\n")
        
        for page_data in text_result.pages[:10]:  # Limit to first 10 pages for token management
            if page_data.text.strip():
                human_content_parts.append(f"\n--- Page {page_data.page_number} ---\n")
                human_content_parts.append(page_data.text[:5000])  # Limit text per page
                
                if page_data.has_tables:
                    human_content_parts.append("\n[Note: This page contains tables]\n")
                if page_data.has_images:
                    human_content_parts.append("\n[Note: This page contains images]\n")
    
    # For vision mode, prepare multimodal content
    if use_vision and image_result and image_result.success and image_result.pages:
        # Create multimodal message with images
        content = []
        
        # Add text instruction
        content.append({
            "type": "text",
            "text": "".join(human_content_parts) + "\n\n=== DOCUMENT IMAGES ===\nPlease analyze these document images to answer the question:"
        })
        
        # Add images (limit to prevent token overflow)
        max_images = 5
        for page_data in image_result.pages[:max_images]:
            try:
                # Read and encode image
                image_base64 = _encode_image(page_data.image_path)
                if image_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"  # Use high detail for document analysis
                        }
                    })
                    logger.debug(f"Added image from page {page_data.page_number}")
            except Exception as e:
                logger.warning(f"Failed to encode image for page {page_data.page_number}: {e}")
        
        messages.append(HumanMessage(content=content))
    else:
        # Text-only message
        messages.append(HumanMessage(content="".join(human_content_parts)))
    
    return messages


def _encode_image(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def _extract_source_pages(
    text_result: Optional[TextExtractionResult],
    image_result: Optional[ImageExtractionResult]
) -> List[int]:
    """
    Extract list of source pages used for answer.
    
    Args:
        text_result: Text extraction results
        image_result: Image extraction results
        
    Returns:
        List of page numbers that were processed
    """
    pages = set()
    
    if text_result and text_result.pages:
        pages.update(p.page_number for p in text_result.pages)
    
    if image_result and image_result.pages:
        pages.update(p.page_number for p in image_result.pages)
    
    return sorted(list(pages))