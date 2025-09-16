"""PDF extraction-related Pydantic models for document processing endpoints."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union


class PDFUploadRequest(BaseModel):
    """Request model for PDF upload and processing.

    Attributes:
        file_content: Base64 encoded PDF file content
        filename: Original filename of the uploaded PDF
        extraction_model: LLM model to use for extraction ("anthropic", "openai", "gemini")
        temperature: Controls response randomness (0.0-1.0). Defaults to 0.1 for structured extraction.
        max_tokens: Maximum number of tokens in the response. Defaults to 2000.
    """

    file_content: str
    filename: str
    extraction_model: Optional[str] = "anthropic"
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 2000


class HeaderData(BaseModel):
    """Header data extracted from PDF.

    Attributes:
        client: Client name or identifier
        date: Date from the document
    """

    client: Optional[str] = None
    date: Optional[str] = None


class LineItem(BaseModel):
    """Line item data extracted from PDF.

    Attributes:
        material: Material name or description
        quantity: Quantity of the material (can be string or number)
        unit_price: Unit price of the material (can be string or number)
    """

    material: Optional[str] = None
    quantity: Optional[Union[str, int, float]] = None
    unit_price: Optional[Union[str, int, float]] = None


class ExtractedData(BaseModel):
    """Complete extracted data from PDF.

    Attributes:
        header: Header information (client, date)
        line_items: List of line items with material, quantity, and unit price
    """

    header: HeaderData
    line_items: List[LineItem]


class PDFExtractionResponse(BaseModel):
    """Response model for PDF extraction endpoints.

    Attributes:
        success: Whether the extraction completed successfully
        extracted_data: The extracted structured data from the PDF
        filename: Original filename of the processed PDF
        model_used: The LLM model used for extraction
        usage: Token usage statistics from the LLM
        error: Error message if extraction failed
    """

    success: bool
    extracted_data: Optional[ExtractedData] = None
    filename: str
    model_used: str
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
