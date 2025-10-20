"""
Data schemas for the PDF extraction pipeline.

Defines Pydantic models for type safety and validation throughout
the extraction workflow.
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path


# Input schemas
class PDFExtractionRequest(BaseModel):
    """Request model for PDF extraction"""
    pdf_path: str = Field(..., description="Path to the PDF file to extract from")
    question: str = Field(..., description="Question to answer from the PDF content")
    language: str = Field(default="es", description="Language for the response (default: Spanish)")
    max_pages: Optional[int] = Field(None, description="Maximum number of pages to process")
    temperature: float = Field(default=0.2, description="LLM temperature for response generation")
    

# Node output schemas
class PageTextData(BaseModel):
    """Text data extracted from a single page"""
    page_number: int
    text: str
    char_count: int
    has_tables: bool = False
    has_images: bool = False
    

class TextExtractionResult(BaseModel):
    """Output from the text extraction node"""
    success: bool
    pdf_path: str
    total_pages: int
    pages: List[PageTextData]
    extraction_method: str = "pymupdf_text"
    extraction_time_ms: float
    error_message: Optional[str] = None
    

class PageImageData(BaseModel):
    """Image data for a single page"""
    page_number: int
    image_path: str
    width: int
    height: int
    file_size_kb: float
    

class ImageExtractionResult(BaseModel):
    """Output from the image extraction node"""
    success: bool
    pdf_path: str
    total_pages: int
    pages: List[PageImageData]
    extraction_method: str = "pymupdf_image"
    extraction_time_ms: float
    temp_dir: Optional[str] = None
    error_message: Optional[str] = None
    

# Final output schema
class FinalExtractionResult(BaseModel):
    """Final result after LLM processing"""
    success: bool
    question: str
    answer: str
    source_pages: List[int] = Field(default_factory=list)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float
    error_message: Optional[str] = None
    

# LangGraph state schema
class ExtractionState(TypedDict):
    """
    State schema for LangGraph workflow.
    
    This TypedDict defines the state that flows through the graph nodes.
    """
    # Input
    pdf_path: str
    question: str
    language: str
    temperature: float
    max_pages: Optional[int]
    
    # Node outputs
    text_result: Optional[TextExtractionResult]
    image_result: Optional[ImageExtractionResult]
    
    # Final output
    final_result: Optional[FinalExtractionResult]
    
    # Metadata
    start_time: datetime
    errors: List[str]
    

# Configuration schemas
class NodeConfig(BaseModel):
    """Configuration for individual nodes"""
    timeout_seconds: int = Field(default=30, description="Timeout for node execution")
    retry_attempts: int = Field(default=3, description="Number of retry attempts on failure")
    

class TextNodeConfig(NodeConfig):
    """Configuration for text extraction node"""
    extract_tables: bool = Field(default=True, description="Attempt to extract table structures")
    preserve_formatting: bool = Field(default=False, description="Preserve text formatting")
    

class ImageNodeConfig(NodeConfig):
    """Configuration for image extraction node"""
    dpi: int = Field(default=200, description="DPI for image conversion")
    max_dimension: int = Field(default=2048, description="Maximum image dimension")
    image_format: str = Field(default="PNG", description="Output image format")
    

class ReducerNodeConfig(NodeConfig):
    """Configuration for LLM reducer node"""
    llm_provider: str = Field(default="openai", description="LLM provider to use")
    model_name: str = Field(default="gpt-4o", description="Model name for extraction")
    max_tokens: int = Field(default=2048, description="Maximum tokens for response")
    use_vision: bool = Field(default=True, description="Use vision capabilities if available")
    

class ExtractorConfig(BaseModel):
    """Overall configuration for the PDF extractor"""
    text_node: TextNodeConfig = Field(default_factory=TextNodeConfig)
    image_node: ImageNodeConfig = Field(default_factory=ImageNodeConfig)
    reducer_node: ReducerNodeConfig = Field(default_factory=ReducerNodeConfig)
    parallel_processing: bool = Field(default=True, description="Run nodes in parallel")
    cleanup_temp_files: bool = Field(default=True, description="Clean up temporary files after processing")