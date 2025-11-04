from typing import Optional
from pydantic import BaseModel


class RelishRecord(BaseModel):
    process: Optional[str] = None
    processingDate: Optional[str] = None
    subStatus: Optional[str] = None
    status: Optional[str] = None
    supplierName: Optional[str] = None
    supplierEmail: Optional[str] = None
    supplierId: Optional[str] = None
    invoiceOrigin: Optional[str] = None
    orderId: Optional[str] = None
    invoiceDate: Optional[str] = None
    invoiceType: Optional[str] = None
    externalUrl: Optional[str] = None
    invoiceNumber: Optional[str] = None
    relishId: Optional[str] = None
    uuid: Optional[str] = None
    reason: Optional[str] = None
    submitterEmail: Optional[str] = None
    rfc: Optional[str] = None
    validated: Optional[bool] = None
    paymentReceiptCount: Optional[int] = None
    paymentMethod: Optional[str] = None
    highPriority: Optional[bool] = None
    historicalInvoice: Optional[bool] = None
    humanInterventionFlag: Optional[bool] = None
    processedBy: Optional[str] = None
    companyCode: Optional[str] = None
    vendorTaxId: Optional[str] = None
    buyerTaxId: Optional[str] = None
    totalAmount: float
    currency: Optional[str] = None
    aribaUniqueName: Optional[str] = None
    cfdiStatus: Optional[str] = None
    reconciledDate: Optional[str] = None
    scheduledPaymentDate: Optional[str] = None
    reconciliationStatus: Optional[str] = None
    irNumber: Optional[str] = None
    paymentDate: Optional[str] = None
    invStatus: Optional[str] = None
    appName: Optional[str] = None
    customAccountNumber: Optional[str] = None
    customApClerk: Optional[str] = None
    manualApproval: Optional[bool] = None
    account: Optional[str] = None
    apClerk: Optional[str] = None


__all__ = ["RelishRecord"]
