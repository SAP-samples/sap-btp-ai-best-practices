from pydantic import BaseModel
from typing import Optional


class VendorListEntry(BaseModel):
    relishBpNo: Optional[str] = None
    supplierName: Optional[str] = None
    aribaId: Optional[str] = None
    taxId: Optional[str] = None
    streetAddress: Optional[str] = None
    zipCode: Optional[str] = None
    telephone: Optional[str] = None
    commsEmail: Optional[str] = None
    commsDomain: Optional[str] = None
    remitEmail: Optional[str] = None
    remitDomain: Optional[str] = None
    enabled: Optional[bool] = None


__all__ = ["VendorListEntry"]
