from typing import List, Optional
from pydantic import BaseModel


class BusinessPartner(BaseModel):
    """Minimal S/4 Business Partner mock model."""

    id: str
    name: str
    city: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None


class PurchaseOrderItem(BaseModel):
    position: int
    material: Optional[str] = None
    description: Optional[str] = None
    quantity: float
    unit: str
    price: float


class PurchaseOrder(BaseModel):
    id: str
    vendorId: str
    date: str
    currency: str
    status: str
    items: List[PurchaseOrderItem]


class InvoiceItem(BaseModel):
    position: int
    description: Optional[str] = None
    amount: float


class Invoice(BaseModel):
    id: str
    vendorId: str
    poId: Optional[str] = None
    date: str
    currency: str
    amount: float
    status: str
    items: List[InvoiceItem]


__all__ = [
    "BusinessPartner",
    "PurchaseOrderItem",
    "PurchaseOrder",
    "InvoiceItem",
    "Invoice",
]
