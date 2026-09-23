"""
SAP Document AI (DOX) Client Package

A Python client for integrating with SAP Document Information Extraction service.
Supports document upload, schema-based extraction, and ad-hoc field extraction.
"""

from .sap_dox_client import DoxApiError, SapDoxClient, ServiceKey
from .models import (
    CatalogOptions,
    DocumentType,
    ExtractedField,
    ExtractionResult,
    ExtractionResponse,
    FieldSetup,
    NumberFormatting,
    FieldDefinition,
    ParsedExtraction,
    UploadExtractionOptions,
    UploadOptions,
)

__version__ = "1.0.0"
__all__ = [
    "SapDoxClient",
    "ServiceKey",
    "DoxApiError",
    "CatalogOptions",
    "DocumentType",
    "ExtractedField",
    "ExtractionResult",
    "ExtractionResponse",
    "FieldSetup",
    "NumberFormatting",
    "FieldDefinition",
    "ParsedExtraction",
    "UploadExtractionOptions",
    "UploadOptions",
]
