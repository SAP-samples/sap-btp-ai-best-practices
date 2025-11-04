import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_SEED_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "ariba_seed.csv"


def _parse_amount(value: Optional[str]) -> float:
    """
    Parse money strings like "$7,809.05 USD", "-$915.11 USD", "917.28 USD" into float.
    Tolerant to extra spaces and quotes.
    """
    if value is None:
        return 0.0
    s = str(value).strip().replace('"', "")
    if not s:
        return 0.0
    s = s.replace("–", "-")  # normalize en dash
    # Remove currency words and symbols except signs, digits, dot, and comma
    # Keep a leading minus if present with or without a space
    s_clean = re.sub(r"[A-Za-z$,\s]+", " ", s).strip().replace("- ", "-")
    s_clean = re.sub(r"[^0-9\.\-]", "", s_clean)
    if not s_clean or s_clean == "-":
        return 0.0
    try:
        return float(s_clean)
    except Exception:
        try:
            return float(s_clean.replace(",", ""))
        except Exception:
            return 0.0


def _read_rows() -> List[Dict[str, str]]:
    """
    Read Ariba CSV rows. The seed includes an extra \'Company Code: ...\' line after the header;
    we will skip rows that look like header annotations or empty IDs.
    """
    try:
        with _SEED_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, str]] = []
            for row in reader:
                invoice_num = (row.get("Invoice #") or "").strip()
                # Skip annotation line such as "Company Code: 1010 ..."
                if invoice_num.lower().startswith("company code:"):
                    continue
                # Skip rows without any meaningful fields
                if not any(v and str(v).strip() for v in row.values()):
                    continue
                rows.append(dict(row))
            return rows
    except Exception:
        return []


def load_invoices() -> List[Dict[str, Any]]:
    """
    Load Ariba invoices from CSV and normalize into a simple list of dicts.
    Fields exposed:
      - invoiceNumber, id, invoiceDate, supplier, status, totalAmount, currency,
        scheduledDate, requester, matchedOrderId
    """
    rows = _read_rows()
    invoices: List[Dict[str, Any]] = []
    for row in rows:
        invoice_number = (row.get("Invoice #") or "").strip()
        inv_id = (row.get("ID") or "").strip()
        supplier = (row.get("Supplier") or "").strip()
        status = (row.get("Status") or "").strip()
        total_raw = row.get("Total") or ""
        scheduled = (row.get("Scheduled Date") or "").strip()
        requester = (row.get("Requester") or "").strip()
        matched_order = (row.get("Matched Order #") or "").strip()
        invoice_date = (row.get("Invoice Date") or "").strip()

        # Currency: default to USD if present in string, otherwise try to detect a 3-letter code.
        currency = "USD"
        m = re.search(r"\b([A-Z]{3})\b", str(total_raw))
        if m:
            currency = m.group(1)

        invoices.append(
            {
                "invoiceNumber": invoice_number,
                "id": inv_id or f"{invoice_number}",
                "invoiceDate": invoice_date,
                "supplier": supplier,
                "status": status,
                "totalAmount": _parse_amount(total_raw),
                "currency": currency,
                "scheduledDate": scheduled,
                "requester": requester,
                "matchedOrderId": matched_order,
            }
        )
    return invoices
