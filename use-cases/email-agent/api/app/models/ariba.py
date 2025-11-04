from typing import Optional
from pydantic import BaseModel


class AribaInvoice(BaseModel):
    invoiceNumber: str
    id: str
    invoiceDate: Optional[str] = None
    supplier: Optional[str] = None
    status: Optional[str] = None
    totalAmount: float
    currency: str
    scheduledDate: Optional[str] = None
    requester: Optional[str] = None
    matchedOrderId: Optional[str] = None


__all__ = ["AribaInvoice"]
