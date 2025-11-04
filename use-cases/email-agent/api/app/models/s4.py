from typing import List, Optional
from pydantic import BaseModel


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
    "InvoiceItem",
    "Invoice",
]
