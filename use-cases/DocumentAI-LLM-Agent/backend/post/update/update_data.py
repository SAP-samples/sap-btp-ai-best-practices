"""
Chapter 06: Updating Data with PATCH Operations
Demonstrates how to update Business Partners and Sales Orders.
"""

import json
import requests
from urllib.parse import quote
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent / "chapter-03-advanced-queries"))
from config import BASE_URL, CLIENT, USER, PWD, VERIFY, API_BP, API_SO


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


def bp10(x: str) -> str:
    """Ensure Business Partner ID is 10 digits with leading zeros."""
    return x.zfill(10) if x.isdigit() and len(x) != 10 else x


def fetch_csrf(session: requests.Session) -> tuple[str, requests.cookies.RequestsCookieJar]:
    """Fetch CSRF token for write operations."""
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/json"}
    p = {"sap-client": CLIENT, "$top": "0"}
    r = session.get(f"{API_SO}/A_SalesOrder", params=p, headers=h, timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies
    raise RuntimeError("CSRF token fetch failed")


def get_bp(s: requests.Session, bp_id: str) -> dict:
    """Read a Business Partner."""
    bid = quote(bp10(bp_id), safe="")
    url = f"{API_BP}/A_BusinessPartner('{bid}')"
    params = {"sap-client": CLIENT, "$format": "json"}
    r = s.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return (data.get("d") or data)


def update_bp(s: requests.Session, bp_id: str, updates: dict) -> dict:
    """Update a Business Partner using PATCH."""
    token, cookies = fetch_csrf(s)
    
    bid = quote(bp10(bp_id), safe="")
    url = f"{API_BP}/A_BusinessPartner('{bid}')"
    
    params = {"sap-client": CLIENT}
    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    r = s.patch(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        data=json.dumps(updates),
        timeout=60
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"PATCH failed: {e}")
        try:
            print("Error:", json.dumps(r.json(), indent=2))
        except:
            print("Response:", r.text[:500])
        raise
    
    if r.status_code == 204:
        return {"message": "Updated successfully (no content returned)"}
    
    data = r.json()
    return (data.get("d") or data)


def update_sales_order(s: requests.Session, so_number: str, updates: dict) -> dict:
    """Update Sales Order header fields."""
    token, cookies = fetch_csrf(s)
    
    so_key = quote(so_number.zfill(10), safe="")
    url = f"{API_SO}/A_SalesOrder('{so_key}')"
    
    params = {"sap-client": CLIENT}
    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    r = s.patch(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        data=json.dumps(updates),
        timeout=60
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"PATCH failed: {e}")
        try:
            print("Error:", json.dumps(r.json(), indent=2))
        except:
            print("Response:", r.text[:500])
        raise
    
    if r.status_code == 204:
        return {"message": "Updated successfully"}
    
    data = r.json()
    return (data.get("d") or data)


def main():
    """Demonstrate updating data with PATCH."""
    print("="*60)
    print("Chapter 06: Updating Data (PATCH Operations)")
    print("="*60)
    
    session = sess()
    
    # Example 1: Update Business Partner
    print("\n=== Update Business Partner (Demo) ===")
    print("Note: BP updates may fail if field is read-only or BP is locked")
    print("This is for demonstration purposes only.")
    
    # Uncomment to actually update (be careful!)
    # try:
    #     result = update_bp(
    #         session,
    #         "1000090",
    #         {"BusinessPartnerFullName": "Twister Updated"}
    #     )
    #     print(f"Update result: {result}")
    #     
    #     # Verify
    #     bp = get_bp(session, "1000090")
    #     print(f"New name: {bp.get('BusinessPartnerFullName')}")
    # except Exception as e:
    #     print(f"Update failed: {e}")
    
    # Example 2: Update Sales Order
    print("\n=== Update Sales Order (Demo) ===")
    print("Note: SO updates may fail based on status and authorizations")
    print("This demonstrates the PATCH operation structure.")
    
    # Uncomment to actually update (be careful!)
    # try:
    #     result = update_sales_order(
    #         session,
    #         "1415",
    #         {"PurchaseOrderByCustomer": "PO-2024-TEST"}
    #     )
    #     print(f"Update result: {result}")
    # except Exception as e:
    #     print(f"Update failed: {e}")
    
    print("\nIMPORTANT NOTES:")
    print("- PATCH operations modify existing data")
    print("- Always fetch a fresh CSRF token")
    print("- Test updates in development environment first")
    print("- Not all fields are updatable")
    print("- Document status may prevent updates")
    
    print("\n" + "="*60)
    print("Chapter 06 completed!")
    print("Next: Chapter 07 - Batch Operations")
    print("="*60)


if __name__ == "__main__":
    main()

