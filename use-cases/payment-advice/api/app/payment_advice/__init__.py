"""
Payment Advice Extractor (UC-01) core package.

Turns a supplier payment/remittance advice (any supported format, any size) into a
validated canonical JSON payload in the ``payment_advice_canonical`` shape.

Public entry points are added incrementally.
"""

from __future__ import annotations

from .config import PaymentAdviceSettings
from .customers import (
    Customer,
    CustomerSchema,
    normalize_client_key,
)

__all__ = [
    "PaymentAdviceSettings",
    "Customer",
    "CustomerSchema",
    "normalize_client_key",
]
