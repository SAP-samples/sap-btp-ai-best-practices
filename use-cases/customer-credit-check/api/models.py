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
    INVESTIGACION_COMERCIAL = "investigacion_comercial"
    INVESTIGACION_LEGAL = "investigacion_legal"
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


class BatchExtractionRequest(BaseModel):
    """
    Request model for batch document processing.
    
    Attributes:
        max_concurrent: Maximum concurrent processing
        documents: List of document configurations
    """
    max_concurrent: Optional[int] = Field(default=10, ge=1, le=50)
    documents: List[Dict[str, Any]] = Field(
        ..., 
        description="List of documents with their types and optional questions"
    )


class BatchExtractionResponse(BaseModel):
    """
    Response model for batch extraction.
    
    Attributes:
        task_id: Unique identifier for the batch task
        total_documents: Total number of documents
        status: Current processing status
        results: List of extraction results (when complete)
        created_at: Timestamp when batch was created
    """
    task_id: str
    total_documents: int
    status: str  # "processing", "completed", "failed"
    results: Optional[List[ExtractionResponse]] = None
    created_at: float
    completed_at: Optional[float] = None


class DocumentSchema(BaseModel):
    """
    Schema configuration for a document type.
    
    Attributes:
        document_type: Type of document
        title: Display title
        questions: List of questions to ask
        fields: List of field names for display
    """
    document_type: DocumentType
    title: str
    questions: List[str]
    fields: List[str]


class DocumentSchemasResponse(BaseModel):
    """
    Response containing all document schemas.
    
    Attributes:
        schemas: Dictionary of document schemas by type
    """
    schemas: Dict[str, DocumentSchema]


class UpdateSchemaRequest(BaseModel):
    """
    Request to update a document schema.
    
    Attributes:
        document_type: Document type to update
        questions: New questions list
        fields: New fields list
    """
    document_type: DocumentType
    questions: List[str]
    fields: List[str]


class TaskStatusResponse(BaseModel):
    """
    Response for task status check.
    
    Attributes:
        task_id: Task identifier
        status: Current status
        progress: Progress percentage (0-100)
        message: Status message
        result: Final result if completed
    """
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int = Field(ge=0, le=100)
    message: Optional[str] = None
    result: Optional[Any] = None