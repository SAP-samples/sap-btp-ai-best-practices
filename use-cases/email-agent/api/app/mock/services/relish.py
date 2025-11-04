import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_SEED_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "relish_seed.csv"


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Keep only digits and optional leading minus
    s_clean = re.sub(r"[^0-9\-]", "", s)
    if s_clean in {"", "-", "--"}:
        return None
    try:
        return int(s_clean)
    except Exception:
        return None


def _parse_amount(value: Optional[str]) -> float:
    """Parse strings like '51682.76', '29,160.00 USD', '-$1,293.64 USD' into float."""
    if value is None:
        return 0.0
    s = str(value).strip().replace('"', "")
    if not s:
        return 0.0
    s = s.replace("–", "-")
    # Remove currency words/symbols and spaces, preserve digits, dot, minus, comma for thousands
    s_clean = re.sub(r"[A-Za-z$\s]+", "", s)
    if not s_clean:
        return 0.0
    try:
        return float(s_clean.replace(",", ""))
    except Exception:
        return 0.0


def _read_rows() -> List[Dict[str, str]]:
    try:
        with _SEED_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception:
        return []


def load_records() -> List[Dict[str, Any]]:
    """
    Load Relish CSV and normalize column names/types into our RelishRecord schema.
    """
    rows = _read_rows()
    records: List[Dict[str, Any]] = []
    for row in rows:
        rec: Dict[str, Any] = {
            "process": (row.get("Process") or "").strip() or None,
            "processingDate": (row.get("ProcessingDate") or "").strip() or None,
            "subStatus": (row.get("SubStatus") or "").strip() or None,
            "status": (row.get("Status") or "").strip() or None,
            "supplierName": (row.get("SupplierName") or "").strip() or None,
            "supplierEmail": (row.get("SupplierEmail") or "").strip() or None,
            "supplierId": (row.get("SupplierId") or "").strip() or None,
            "invoiceOrigin": (row.get("InvoiceOrigin") or "").strip() or None,
            "orderId": (row.get("OrderId") or "").strip() or None,
            "invoiceDate": (row.get("InvoiceDate") or "").strip() or None,
            "invoiceType": (row.get("InvoiceType") or "").strip() or None,
            "externalUrl": (row.get("ExternalUrl") or "").strip() or None,
            "invoiceNumber": (row.get("InvoiceNumber") or "").strip() or None,
            "relishId": (row.get("RelishId") or "").strip() or None,
            "uuid": (row.get("UUID") or "").strip() or None,
            "reason": (row.get("Reason") or "").strip() or None,
            "submitterEmail": (row.get("SubmitterEmail") or "").strip() or None,
            "rfc": (row.get("Rfc") or "").strip() or None,
            "validated": _parse_bool(row.get("Validated")),
            "paymentReceiptCount": _parse_int(row.get("PaymentReceiptCount")),
            "paymentMethod": (row.get("PaymentMethod") or "").strip() or None,
            "highPriority": _parse_bool(row.get("HighPriority")),
            "historicalInvoice": _parse_bool(row.get("HistoricalInvoice")),
            "humanInterventionFlag": _parse_bool(row.get("HumanInterventionFlag")),
            "processedBy": (row.get("ProcessedBy") or "").strip() or None,
            "companyCode": (row.get("CompanyCode") or "").strip() or None,
            "vendorTaxId": (row.get("VendorTaxId") or "").strip() or None,
            "buyerTaxId": (row.get("BuyerTaxId") or "").strip() or None,
            "totalAmount": _parse_amount(row.get("TotalAmount")),
            "currency": (row.get("Currency") or "").strip() or None,
            "aribaUniqueName": (row.get("AribaUniqueName") or "").strip() or None,
            "cfdiStatus": (row.get("CfdiStatus") or "").strip() or None,
            "reconciledDate": (row.get("ReconciledDate") or "").strip() or None,
            "scheduledPaymentDate": (row.get("ScheduledPaymentDate") or "").strip()
            or None,
            "reconciliationStatus": (row.get("ReconciliationStatus") or "").strip()
            or None,
            "irNumber": (row.get("IrNumber") or "").strip() or None,
            "paymentDate": (row.get("PaymentDate") or "").strip() or None,
            "invStatus": (row.get("InvStatus") or "").strip() or None,
            "appName": (row.get("appName") or "").strip() or None,
            "customAccountNumber": (row.get("CUSTOM_ACCOUNT_NUMBER") or "").strip()
            or None,
            "customApClerk": (row.get("CUSTOM_ap_clerk") or "").strip() or None,
            "manualApproval": _parse_bool(row.get("MANUAL APPROVAL")),
            "account": (row.get("ACCOUNT") or "").strip() or None,
            "apClerk": (row.get("AP_CLERK") or "").strip() or None,
        }
        # Skip completely empty records
        if not any(v not in (None, "", 0, 0.0) for v in rec.values()):
            continue
        records.append(rec)
    return records
