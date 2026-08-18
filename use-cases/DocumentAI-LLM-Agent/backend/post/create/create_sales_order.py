"""
Chapter 05: Creating Sales Orders in S/4HANA
Demonstrates POST operations with deep-insert and CSRF token handling.
"""

import datetime as dt
import json
import requests
from urllib.parse import quote
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent / "chapter-03-advanced-queries"))
from config import BASE_URL, CLIENT, USER, PWD, VERIFY, API_SO


def sess() -> requests.Session:
    """Create an authenticated session with S4 system."""
    s = requests.Session()
    s.auth = (USER, PWD)
    s.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    s.verify = VERIFY
    return s


def sap_date(d: dt.date) -> str:
    """Convert Python date to SAP OData date format."""
    ms = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    return f"/Date({ms})/"


def fetch_csrf(session: requests.Session) -> tuple[str, requests.cookies.RequestsCookieJar]:
    """Fetch CSRF token and cookies for modifying requests."""
    # Try #1: GET entity set
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/json"}
    p = {"sap-client": CLIENT, "$top": "0"}
    r = session.get(f"{API_SO}/A_SalesOrder", params=p, headers=h, timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies

    # Try #2: GET $metadata
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/xml"}
    r = session.get(f"{API_SO}/$metadata", params={"sap-client": CLIENT}, headers=h, timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies

    raise RuntimeError(f"CSRF token fetch failed: status={r.status_code}")


def get_template_so(session: requests.Session, so_number: str) -> dict:
    """Read an existing Sales Order to use as a template."""
    so_key = quote(so_number.zfill(10), safe="")
    url = f"{API_SO}/A_SalesOrder('{so_key}')"
    
    params = {
        "sap-client": CLIENT,
        "$format": "json",
        "$expand": "to_Item,to_Partner"
    }
    
    r = session.get(url, params=params, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    return (data.get("d") or data)


def build_minimal_payload(src: dict) -> dict:
    """Build a minimal Sales Order payload from a template."""
    # Header
    hdr = {
        "SalesOrderType":        src.get("SalesOrderType") or "OR",
        "SalesOrganization":     src.get("SalesOrganization"),
        "DistributionChannel":   src.get("DistributionChannel"),
        "OrganizationDivision":  src.get("OrganizationDivision"),
        "SoldToParty":           src.get("SoldToParty"),
        "TransactionCurrency":   src.get("TransactionCurrency") or "EUR",
        "PricingDate":           sap_date(dt.date.today()),
        "RequestedDeliveryDate": sap_date(dt.date.today() + dt.timedelta(days=7)),
    }

    # Items
    items = (src.get("to_Item") or {}).get("results", [])
    if not items:
        raise ValueError("Template has no items")
    
    it = items[0]
    
    new_item = {
        "Material":                it.get("Material"),
        "RequestedQuantity":       it.get("RequestedQuantity") or "1",
        "RequestedQuantityUnit":   it.get("RequestedQuantityUnit") or it.get("OrderQuantityUnit") or "PC",
        "ProductionPlant":         it.get("ProductionPlant") or "C000",
        "ShippingPoint":           it.get("ShippingPoint") or "VS01",
        "PricingDate":             sap_date(dt.date.today()),
        "IncotermsClassification": it.get("IncotermsClassification") or "EXW",
        "IncotermsLocation1":      it.get("IncotermsLocation1") or "ex works",
    }

    payload = hdr
    payload["to_Item"] = {"results": [new_item]}

    return payload


def create_sales_order(session: requests.Session, payload: dict) -> dict:
    """Create a new Sales Order via POST deep-insert."""
    token, cookies = fetch_csrf(session)

    url = f"{API_SO}/A_SalesOrder"
    params = {"sap-client": CLIENT}
    
    headers = {
        "X-CSRF-Token": token,
        "X-Requested-With": "XMLHttpRequest",
        "Prefer": "return=representation",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    r = session.post(
        url, 
        params=params, 
        headers=headers, 
        cookies=cookies,
        data=json.dumps(payload), 
        timeout=120
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"\nCREATE FAILED: HTTP {e.response.status_code}")
        try:
            print("Error:", json.dumps(r.json(), indent=2))
        except:
            print("Response:", r.text[:500])
        raise

    data = r.json()
    return (data.get("d") or data)


def main():
    """Demonstrate creating a Sales Order."""
    print("="*60)
    print("Chapter 05: Creating Sales Orders")
    print("="*60)
    
    session = sess()

    # Step 1: Read template
    print("\n=== Reading Template Sales Order ===")
    try:
        template = get_template_so(session, "1")
        print(f"Template SO: {template.get('SalesOrder')}")
        print(f"  Type: {template.get('SalesOrderType')}")
        print(f"  Org: {template.get('SalesOrganization')}")
        print(f"  Customer: {template.get('SoldToParty')}")
    except Exception as e:
        print(f"Could not read template: {e}")
        print("Adjust the template SO number in the script")
        return

    # Step 2: Build payload
    print("\n=== Building Payload ===")
    try:
        payload = build_minimal_payload(template)
        print(f"Payload created successfully")
        print(f"  Sales Org: {payload.get('SalesOrganization')}")
        print(f"  Customer: {payload.get('SoldToParty')}")
        print(f"  Items: {len(payload.get('to_Item', {}).get('results', []))}")
    except Exception as e:
        print(f"Could not build payload: {e}")
        return

    # Step 3: Create Sales Order
    print("\n=== Creating Sales Order ===")
    try:
        created = create_sales_order(session, payload)
        so_number = created.get("SalesOrder")
        print(f"\nSUCCESS! Created Sales Order: {so_number}")
        print(f"  Type: {created.get('SalesOrderType')}")
        print(f"  Customer: {created.get('SoldToParty')}")
        print(f"  Total: {created.get('TotalNetAmount')} {created.get('TransactionCurrency')}")
    except Exception as e:
        print(f"Could not create Sales Order: {e}")
        print("\nCheck:")
        print("  - Authorization for creating Sales Orders")
        print("  - Material and customer validity")
        print("  - Organizational data configuration")
    
    print("\n" + "="*60)
    print("Chapter 05 completed!")
    print("Next: Chapter 06 - Updating Data (PATCH Operations)")
    print("="*60)


if __name__ == "__main__":
    main()

