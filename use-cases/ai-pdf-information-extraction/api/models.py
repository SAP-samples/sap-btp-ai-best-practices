"""
Data models for the Document Extraction API.

Defines Pydantic models for request/response validation and
document schema configuration.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types for extraction."""
    CONOCE_CLIENTE = "conoce_cliente"
    COMENTARIOS_VENDEDOR = "comentarios_vendedor"
    CONSTANCIA_FISCAL = "constancia_fiscal"
    INE = "ine"
    CGV = "cgv"
    CUSTOM = "custom"


class ExtractionRequest(BaseModel):
    """
    Request model for single document extraction.
    
    Attributes:
        document_type: Type of document being processed
        questions: Optional custom questions (uses defaults if not provided)
        temperature: LLM temperature for response generation
        language: Language for responses
    """
    document_type: DocumentType
    questions: Optional[List[str]] = None
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)
    language: Optional[str] = Field(default="es")


class ExtractionResult(BaseModel):
    """
    Single extraction result for a question.
    
    Attributes:
        question: The question that was asked
        answer: The extracted answer
        field: The field name for display
        confidence: Optional confidence score
    """
    question: str
    answer: str
    field: str
    confidence: Optional[float] = None


class ExtractionResponse(BaseModel):
    """
    Response model for document extraction.
    
    Attributes:
        success: Whether extraction was successful
        document_type: Type of document processed
        results: List of extraction results
        processing_time_ms: Time taken to process
        error: Error message if failed
        metadata: Additional metadata
    """
    success: bool
    document_type: str
    results: List[ExtractionResult]
    processing_time_ms: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

