from pydantic import BaseModel, Field
from typing import Optional


class ColConfig(BaseModel):
    invoiceNumber: str = "Invoice#"
    invoiceAmount: str = "Invoice Amt"
    textCols: list[str] = Field(default_factory=lambda: ["BY_ORD_OF_NAME", "BY_ORD_OF_ADDR", "REMIT_NAME"])
    tolerance: float = 0.01


class MatchRequest(BaseModel):
    invoiceCSV: str
    paymentCSV: str
    ruleMatchedInvoices: list[str] = Field(default_factory=list)
    tolerance: float = 0.01
    colConfig: Optional[ColConfig] = None


class MatchProgress(BaseModel):
    phase: str = ""
    message: str = ""
    totalPayers: int = 0
    completedPayers: int = 0
    matchCount: int = 0


class MatchJobResponse(BaseModel):
    jobId: str


class MatchStatusResponse(BaseModel):
    status: str
    progress: Optional[MatchProgress] = None
    results: Optional[list[dict]] = None
    error: Optional[str] = None
