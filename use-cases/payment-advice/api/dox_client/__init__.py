"""Product-used SAP Document AI client and schema field models."""

from .sap_dox_client import DoxApiError, SapDoxClient, ServiceKey
from .models import FieldDefinition, FieldSetup

__version__ = "1.0.0"
__all__ = [
    "SapDoxClient",
    "ServiceKey",
    "DoxApiError",
    "FieldSetup",
    "FieldDefinition",
]
