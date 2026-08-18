#!/usr/bin/env python3
"""
Integration test for FI Supplier Invoice endpoint.

Run with:
    cd /Users/joel/Library/CloudStorage/OneDrive-SAPSE/Projects/AI4U/docai/backend
    python test_fi_post.py

Requirements: requests (pip install requests)
Server must be running at http://localhost:8001
"""

import json
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8001"
PASS = "PASS"
FAIL = "FAIL"

results = []


def separator(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_request(method, url, payload=None):
    print(f"\n--> {method} {url}")
    if payload:
        print("    Request payload:")
        print(json.dumps(payload, indent=6))


def print_response(resp):
    print(f"\n<-- Status: {resp.status_code}")
    try:
        body = resp.json()
        print("    Response body:")
        print(json.dumps(body, indent=6))
    except Exception:
        print(f"    Response text: {resp.text[:500]}")


def record(label, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((label, status, detail))
    print(f"\n  [{status}] {label}" + (f" — {detail}" if detail else ""))


# ──────────────────────────────────────────────
# 1. HEALTH CHECK
# ──────────────────────────────────────────────
separator("1. HEALTH CHECK  —  GET /health")

try:
    url = f"{BASE_URL}/health"
    print_request("GET", url)
    resp = requests.get(url, timeout=5)
    print_response(resp)

    ok = resp.status_code == 200
    record("Health check", ok, f"HTTP {resp.status_code}")

except requests.exceptions.ConnectionError:
    print(f"\n  ERROR: Cannot connect to {BASE_URL}")
    print("  Make sure the FastAPI server is running on port 8001.")
    record("Health check", False, "Connection refused — server not running")
    separator("SUMMARY")
    for label, status, detail in results:
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    print()
    sys.exit(1)

except requests.exceptions.Timeout:
    record("Health check", False, "Request timed out")


# ──────────────────────────────────────────────
# 2. BUSINESS PARTNERS  —  GET /api/business-partners
# ──────────────────────────────────────────────
separator("2. S4 CONNECTION  —  GET /api/business-partners")

url = f"{BASE_URL}/api/business-partners"
print_request("GET", url)

try:
    resp = requests.get(url, timeout=15)
    print_response(resp)

    ok = resp.status_code == 200
    detail = f"HTTP {resp.status_code}"
    if ok:
        try:
            data = resp.json()
            count = len(data) if isinstance(data, list) else "n/a"
            detail += f", {count} business partner(s) returned"
        except Exception:
            pass
    record("Business partners endpoint", ok, detail)

except requests.exceptions.Timeout:
    record("Business partners endpoint", False, "Request timed out (S4 may be unreachable)")
except requests.exceptions.RequestException as exc:
    record("Business partners endpoint", False, str(exc))


# ──────────────────────────────────────────────
# 3. CUSTOMER SEARCH  —  GET /api/customers/search?q=SAP
# ──────────────────────────────────────────────
separator("3. CUSTOMER SEARCH  —  GET /api/customers/search?q=SAP")

url = f"{BASE_URL}/api/customers/search"
params = {"q": "SAP"}
print_request("GET", f"{url}?q=SAP")

try:
    resp = requests.get(url, params=params, timeout=15)
    print_response(resp)

    ok = resp.status_code == 200
    detail = f"HTTP {resp.status_code}"
    if ok:
        try:
            data = resp.json()
            count = len(data) if isinstance(data, list) else "n/a"
            detail += f", {count} customer(s) matched"
        except Exception:
            pass
    record("Customer search (q=SAP)", ok, detail)

except requests.exceptions.Timeout:
    record("Customer search (q=SAP)", False, "Request timed out")
except requests.exceptions.RequestException as exc:
    record("Customer search (q=SAP)", False, str(exc))


# ──────────────────────────────────────────────
# 4. POST INVOICE  —  POST /api/v1/fi/post-invoice
# ──────────────────────────────────────────────
separator("4. POST FI SUPPLIER INVOICE  —  POST /api/v1/fi/post-invoice")

payload = {
    "supplier_name": "SAP SE",
    "invoice_number": "TEST-INV-001",
    "invoice_date": "2026-07-29",
    "total_amount": 1500.00,
    "currency": "USD",
    "business_partner": "",   # empty — let the backend auto-match
}

url = f"{BASE_URL}/api/v1/fi/post-invoice"
print_request("POST", url, payload)

try:
    resp = requests.post(url, json=payload, timeout=30)
    print_response(resp)

    # Consider 200 and 201 as success; 422 means validation error in payload
    ok = resp.status_code in (200, 201)
    detail = f"HTTP {resp.status_code}"

    if resp.status_code == 422:
        detail += " (Unprocessable Entity — check payload schema)"
    elif resp.status_code == 404:
        detail += " (Endpoint not found — check route registration)"
    elif resp.status_code == 500:
        detail += " (Internal Server Error — check server logs)"

    record("POST /api/v1/fi/post-invoice", ok, detail)

    # Additional assertion: response should contain a document number or status
    if ok:
        try:
            body = resp.json()
            has_doc = any(
                k in body
                for k in ("document_number", "doc_number", "posting_id", "id", "status")
            )
            record(
                "Response contains expected field",
                has_doc,
                "expected one of: document_number / posting_id / id / status",
            )
        except Exception:
            record("Response is valid JSON", False, "Could not parse response body")

except requests.exceptions.Timeout:
    record("POST /api/v1/fi/post-invoice", False, "Request timed out (30 s)")
except requests.exceptions.RequestException as exc:
    record("POST /api/v1/fi/post-invoice", False, str(exc))


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
separator("SUMMARY")

total = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = total - passed

for label, status, detail in results:
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))

print()
print(f"  Result: {passed}/{total} checks passed")
print()

if failed == 0:
    print("  OVERALL: SUCCESS")
else:
    print(f"  OVERALL: FAILURE ({failed} check(s) failed)")

print()
sys.exit(0 if failed == 0 else 1)
