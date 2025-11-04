import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SEED_CSV_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "vendor_list_seed.csv"
)


def _read_rows() -> List[Dict[str, str]]:
    try:
        with _SEED_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except Exception:
        return []


def _normalize_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return None


_EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE)


def _extract_domain(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    s = str(email).strip()
    if not s or s in {"0", "NA", "N/A", "NULL", "NONE"}:
        return None
    m = _EMAIL_PATTERN.search(s)
    if not m:
        return None
    return m.group(1).lower()


def load_vendor_list() -> List[Dict[str, Any]]:
    """
    Load Vendor List CSV and normalize fields.

    Output fields:
      - relishBpNo (string)
      - supplierName (string)
      - aribaId (string)
      - taxId (string)
      - streetAddress (string)
      - zipCode (string)
      - telephone (string)
      - commsEmail (string)
      - commsDomain (string | None)
      - remitEmail (string)
      - remitDomain (string | None)
      - enabled (bool | None)
    """
    rows = _read_rows()
    vendors: List[Dict[str, Any]] = []
    for row in rows:
        relish_bp = (row.get("Relish BP #") or "").strip()
        supplier_name = (row.get("Name of Supplier") or "").strip()
        ariba_id = (row.get("Ariba ID") or "").strip()
        tax_id = (row.get("Tax ID") or "").strip()
        street = (row.get("Street Adress") or "").strip()
        zip_code = (row.get("Zip Code") or "").strip()
        telephone = (row.get("Telephone") or "").strip()
        comms = (row.get("Communications Email Address") or "").strip()
        remit = (row.get("Remittance Email Address") or "").strip()
        enabled_raw = row.get("Enabled Vendor")

        # Skip completely empty lines
        if not (relish_bp or supplier_name or comms or remit):
            continue

        vendors.append(
            {
                "relishBpNo": relish_bp,
                "supplierName": supplier_name,
                "aribaId": ariba_id,
                "taxId": tax_id,
                "streetAddress": street,
                "zipCode": zip_code,
                "telephone": telephone,
                "commsEmail": comms or None,
                "commsDomain": _extract_domain(comms),
                "remitEmail": remit or None,
                "remitDomain": _extract_domain(remit),
                "enabled": _normalize_bool(enabled_raw),
            }
        )
    return vendors


def build_domain_index(
    vendors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Index vendors by email domain for both communications and remittance emails."""
    data = vendors if vendors is not None else load_vendor_list()
    index: Dict[str, List[Dict[str, Any]]] = {}
    for v in data:
        for dom in (v.get("commsDomain"), v.get("remitDomain")):
            if not dom:
                continue
            index.setdefault(dom, []).append(v)
    return index


def find_by_domain(
    domain: str, vendors: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Return vendors whose communications/remittance email domain matches the given domain."""
    if not domain:
        return []
    dom = domain.lower().strip()
    idx = build_domain_index(vendors)
    return list(idx.get(dom, []))


__all__ = [
    "load_vendor_list",
    "build_domain_index",
    "find_by_domain",
]
