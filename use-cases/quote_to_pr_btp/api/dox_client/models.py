"""
Data models for SAP Document AI (DOX) extraction.

Defines Pydantic models for request/response validation and
document schema configuration.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List, Literal
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


class ExtractionResult(BaseModel):
    """
    Single extraction result for a question.

    Attributes:
        question: The question that was asked
        answer: The extracted answer
        field: The field name for display
        confidence: Optional confidence score from DOX (0.0-1.0)
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
        metadata: Additional metadata (e.g., filename)
    """
    success: bool
    document_type: str
    results: List[ExtractionResult]
    processing_time_ms: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None




class FieldSetup(BaseModel):
    """
    Setup configuration for a field.

    Attributes:
        type: Setup type - "auto" (using AI) or "manual" (using templates)
        priority: Priority level (typically 1)
    """
    type: Literal["auto", "manual"]
    priority: int = 1


class NumberFormatting(BaseModel):
    """
    Formatting configuration for number fields.

    Attributes:
        length: Maximum length
        precision: Number of decimal places
        decimalSeparator: Character used for decimal separation
        thousandSeparator: Character used for thousand separation
    """
    length: Optional[str] = None
    precision: Optional[str] = None
    decimalSeparator: Optional[str] = None
    thousandSeparator: Optional[str] = None


class FieldDefinition(BaseModel):
    """
    Complete field definition for schema fields.

    Attributes:
        name: Field name (required, unique within schema)
        description: Field description
        label: Display label for the field
        setupType: Setup type identifier (e.g., "static")
        setupTypeVersion: Version of setup type (e.g., "2.0.0")
        setup: Setup configuration (type and priority) - defaults to "auto" for AI-based extraction
        formattingType: Type of formatting - "string", "number", "date", etc.
        formatting: Formatting configuration (depends on formattingType)
        formattingTypeVersion: Version of formatting type (e.g., "1.0.0")
        defaultExtractor: Default extractor configuration (can map to predefined extractors)

    Note: The default setup type is "auto" which uses AI-based extraction (requires premium plan).
    """
    name: str
    description: Optional[str] = None
    label: Optional[str] = None
    setupType: str = "static"
    setupTypeVersion: str = "2.0.0"
    setup: FieldSetup = Field(default_factory=lambda: FieldSetup(type="auto", priority=1))
    formattingType: Literal["string", "number", "date", "currency"] = "string"
    formatting: Dict[str, Any] = Field(default_factory=dict)
    formattingTypeVersion: str = "1.0.0"
    defaultExtractor: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by the API."""
        return self.model_dump(exclude_none=False)


class UploadExtractionOptions(BaseModel):
    """
    Ad-hoc extraction options used inside the upload document payload.

    Attributes:
        headerFields: Header field names to extract.
        lineItemFields: Line item field names to extract.
    """
    headerFields: Optional[List[str]] = None
    lineItemFields: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return only options that were explicitly set."""
        return self.model_dump(exclude_none=True)


class UploadOptions(BaseModel):
    """
    Document upload options sent as the multipart ``options`` JSON part.

    This model mirrors the core options used by the easy client flow and keeps
    field names aligned with the SAP REST API payload.
    """
    clientId: str
    documentType: Optional[str] = None
    receivedDate: Optional[str] = None
    customLabel: Optional[str] = None
    enrichment: Optional[Dict[str, Any]] = None
    schemaId: Optional[str] = None
    schemaName: Optional[str] = None
    schemaVersion: Optional[str] = None
    templateId: Optional[str] = None
    candidateTemplateIds: Optional[List[str]] = None
    extraction: Optional[UploadExtractionOptions] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return an SAP-compatible upload options dictionary."""
        return self.model_dump(exclude_none=True)


class CatalogOptions(BaseModel):
    """
    Document catalog search options for POST /document/catalog.
    """
    clientId: Optional[str] = None
    filter: Optional[str] = None
    likeFilter: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return an SAP-compatible catalog options dictionary."""
        return self.model_dump(exclude_none=True)


class ExtractedField(BaseModel):
    """
    A normalized view of an extracted SAP Document AI field.

    Extra attributes are preserved because SAP may return field-specific
    metadata such as barcode attributes, group IDs, or enrichment details.
    """
    model_config = ConfigDict(extra="allow")

    name: str
    value: Any = None
    rawValue: Any = None
    confidence: Optional[float] = None
    page: Optional[int] = None
    coordinates: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None


class ParsedExtraction(BaseModel):
    """
    Parsed extraction payload with header fields and line item rows.
    """
    header_fields: List[ExtractedField] = Field(default_factory=list)
    line_items: List[List[ExtractedField]] = Field(default_factory=list)

    @classmethod
    def from_job(cls, job_payload: Dict[str, Any]) -> "ParsedExtraction":
        """Build parsed extraction fields from a Get Result response payload."""
        extraction = job_payload.get("extraction") or {}
        header_fields = [ExtractedField(**field) for field in extraction.get("headerFields", [])]
        line_items = [
            [ExtractedField(**field) for field in row]
            for row in extraction.get("lineItems", [])
        ]
        return cls(header_fields=header_fields, line_items=line_items)
