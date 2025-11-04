import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_SEED_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "s4_seed.csv"


def _parse_amount(value: Optional[str]) -> float:
    """Parse amount strings like '- 7,809.05    USD' or '917.28    USD' into float."""
    if value is None:
        return 0.0
    s = str(value).strip().replace('"', "")
    if not s:
        return 0.0
    s = s.replace("–", "-")  # normalize en dash to minus if present
    # Remove currency words and extra spaces, normalize minus with optional space
    s_clean = re.sub(r"[A-Za-z\s]+", " ", s).strip().replace("- ", "-")
    # Keep only digits, dot and minus
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
    """Read S/4 CSV rows with a tolerant CSV reader."""
    try:
        with _SEED_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception:
        return []


def load_seed() -> Dict[str, Any]:
    """
    Load mock S/4 seed data from api/app/mock/data/s4_seed.csv and
    synthesize business partners, purchase orders, and invoices to match existing models.
    """
    rows = _read_rows()
    partners_map: Dict[str, Dict[str, Any]] = {}
    purchase_orders_map: Dict[str, Dict[str, Any]] = {}
    invoices: List[Dict[str, Any]] = []

    for row in rows:
        supplier_id = str(row.get("Supplier") or "").strip()
        supplier_name = str(row.get("Supplier Name") or "").strip()

        # Business Partners (unique by Supplier ID)
        if supplier_id:
            if supplier_id not in partners_map:
                partners_map[supplier_id] = {
                    "id": supplier_id,
                    "name": supplier_name or supplier_id,
                    "city": None,
                    "country": None,
                    "email": None,
                }

        # Purchase Orders (unique by Purchasing Doc List)
        po_id_raw = row.get("Purchasing Doc List") or ""
        po_id = str(po_id_raw).strip()
        if po_id:
            if po_id not in purchase_orders_map:
                purchase_orders_map[po_id] = {
                    "id": po_id,
                    "vendorId": supplier_id,
                    "date": str(
                        row.get("Posting Date") or row.get("Journal Entry Date") or ""
                    ).strip(),
                    "currency": str(row.get("Transaction Currency") or "").strip()
                    or "USD",
                    # Use Journal Entry Type (e.g., ZP, XX, RE) as a simple status placeholder
                    "status": str(row.get("Journal Entry Type") or "").strip(),
                    "items": [],  # No item-level detail available in the CSV
                }

        # Invoices (one per row)
        amt = _parse_amount(
            row.get("Amount (CoCode Crcy)") or row.get("Amount (Tran Cur.)")
        )
        inv_id = str(row.get("Journal Entry") or "").strip()
        invoices.append(
            {
                "id": inv_id
                or f"{supplier_id}-{row.get('Posting Date') or row.get('Journal Entry Date') or ''}",
                "vendorId": supplier_id,
                "poId": po_id or None,
                "date": str(
                    row.get("Journal Entry Date") or row.get("Posting Date") or ""
                ).strip(),
                "currency": str(row.get("Transaction Currency") or "").strip() or "USD",
                "amount": amt,
                "status": str(row.get("Journal Entry Type") or "").strip(),
                "items": [],  # No item-level detail available in the CSV
            }
        )

    return {
        "businessPartners": list(partners_map.values()),
        "purchaseOrders": list(purchase_orders_map.values()),
        "invoices": invoices,
    }


def load_records() -> List[Dict[str, str]]:
    """
    Return raw S/4 CSV rows for debugging/parity with Ariba/Relish.
    """
    return _read_rows()
