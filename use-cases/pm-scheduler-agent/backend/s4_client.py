"""S/4HANA OData client — BTP / on-prem connection layer.

Replaces the former direct SAP HANA (hdbcli) access. All order and work-center
data now comes from standard S/4HANA OData v2 services:

  * API_MAINTENANCEORDER_SRV  — PM orders + operations
  * API_WORK_CENTER_SRV       — work-center capacity

Authentication is HTTP basic against the S/4 gateway (credentials in .env).
"""
import os
import re
from functools import lru_cache

import requests

# ── OData service paths ───────────────────────────────────────────────────────
SRV_MAINT_ORDER = "/sap/opu/odata/sap/API_MAINTENANCEORDER_SRV"
SRV_WORK_CENTER = "/sap/opu/odata/sap/API_WORK_CENTER_SRV"

_ODATA_DATE_RE = re.compile(r"/Date\((-?\d+)")


def is_configured() -> bool:
    """True when S/4 connection parameters are present in the environment."""
    return bool(os.environ.get("S4_BASE_URL"))


def _base_url() -> str:
    return os.environ.get("S4_BASE_URL", "").rstrip("/")


def _verify():
    """Resolve the `verify` argument for requests (bool or CA bundle path)."""
    ca_bundle = os.environ.get("S4_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    return os.environ.get("S4_VERIFY", "false").lower() not in ("0", "false", "no")


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    """Build a reusable authenticated session."""
    user = os.environ.get("S4_USERNAME")
    pwd = os.environ.get("S4_PASSWORD")
    session = requests.Session()
    session.auth = (user, pwd)
    session.verify = _verify()
    session.headers.update({"Accept": "application/json"})
    if session.verify is False:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def parse_odata_date(val):
    """Parse an OData v2 `/Date(ms)/` string into milliseconds (int) or None.

    Returns the epoch-millisecond integer so callers can hand it straight to
    ``pd.to_datetime(x, unit="ms")``. Non-date values pass through unchanged.
    """
    if val is None:
        return None
    if isinstance(val, str):
        m = _ODATA_DATE_RE.search(val)
        if m:
            return int(m.group(1))
    return val


def odata_get(service: str, entity: str, params: dict, max_pages: int = 100) -> list[dict]:
    """Run a paginated GET against an OData v2 collection.

    Follows the ``d.__next`` link until the result set is exhausted (or the
    ``max_pages`` guard trips). Returns the flattened list of result records.
    """
    session = _session()
    client = os.environ.get("S4_CLIENT", "550")
    base_params = {"sap-client": client, "$format": "json", **params}

    url = f"{_base_url()}{service}/{entity}"
    out: list[dict] = []
    page = 0
    next_params = base_params

    while url and page < max_pages:
        resp = session.get(url, params=next_params, timeout=60)
        resp.raise_for_status()
        payload = resp.json().get("d", {})
        results = payload.get("results", payload if isinstance(payload, list) else [])
        if isinstance(results, dict):
            results = [results]
        out.extend(results)

        # OData v2 server-driven paging: absolute URL in d.__next
        next_link = payload.get("__next")
        if next_link:
            url = next_link if next_link.startswith("http") else f"{_base_url()}{next_link}"
            next_params = None  # the __next URL already carries all query options
        else:
            url = None
        page += 1

    return out


def update_order_basic_start(order_no: str, date_iso: str) -> None:
    """PATCH MaintOrdBasicStartDate on a maintenance order. Raises on failure."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(date_iso).replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    token, session = fetch_csrf_token()
    client = os.environ.get("S4_CLIENT", "550")
    url = f"{_base_url()}{SRV_MAINT_ORDER}/MaintenanceOrder('{order_no}')"
    resp = session.patch(
        url,
        params={"sap-client": client},
        headers={"X-CSRF-Token": token, "Content-Type": "application/json"},
        json={"MaintOrdBasicStartDate": f"/Date({ms})/"},
        timeout=30,
    )
    resp.raise_for_status()


def _fmt_oper(oper_no) -> str:
    """Normalise an operation number to the 4-digit S/4 format (e.g. 10 → '0010')."""
    s = str(oper_no or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(4) if s.isdigit() else s


def _date_iso_to_ms(date_iso: str) -> int:
    """Convert an ISO date (date or datetime) to epoch milliseconds (UTC midnight)."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(str(date_iso)[:10]).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def confirm_operations(items: list[dict], start_time: str = "PT06H00M00S") -> list[dict]:
    """Write scheduled dates back to S/4 by pinning each operation's constraint date.

    Sets OpEarliestSchedldExecStrtDte (+ time) on every MaintenanceOrderOperation
    via OData v2 MERGE. The CSRF token is fetched once and reused across all writes.

    `items` — [{"order_no", "oper_no", "date_iso"}]. Returns one result dict per
    item: {"order_no", "oper_no", "ok": bool, "status"/"error"}.
    """
    token, session = fetch_csrf_token()
    client = os.environ.get("S4_CLIENT", "550")
    results: list[dict] = []

    for it in items:
        order_no = str(it.get("order_no", "")).strip()
        oper = _fmt_oper(it.get("oper_no"))
        res = {"order_no": order_no, "oper_no": oper, "ok": False}
        if not order_no or not oper or not it.get("date_iso"):
            res["error"] = "missing order_no, oper_no or date"
            results.append(res)
            continue
        try:
            ms = _date_iso_to_ms(it["date_iso"])
            url = (
                f"{_base_url()}{SRV_MAINT_ORDER}/MaintenanceOrderOperation"
                f"(MaintenanceOrder='{order_no}',MaintenanceOrderOperation='{oper}',"
                f"MaintenanceOrderSubOperation='')"
            )
            resp = session.request(
                "MERGE", url,
                params={"sap-client": client},
                headers={"X-CSRF-Token": token, "Content-Type": "application/json"},
                json={
                    "OpEarliestSchedldExecStrtDte": f"/Date({ms})/",
                    "OpEarliestSchedldExecStrtTme": start_time,
                },
                timeout=30,
            )
            resp.raise_for_status()
            res["ok"] = True
            res["status"] = resp.status_code
        except Exception as e:
            res["error"] = str(e)
        results.append(res)

    return results


def fetch_csrf_token() -> tuple[str, requests.Session]:
    """Fetch a CSRF token for a subsequent write (PATCH/POST) request."""
    session = _session()
    client = os.environ.get("S4_CLIENT", "550")
    resp = session.get(
        f"{_base_url()}{SRV_MAINT_ORDER}/",
        params={"sap-client": client, "$format": "json"},
        headers={"X-CSRF-Token": "Fetch"},
        timeout=30,
    )
    return resp.headers.get("X-CSRF-Token", ""), session


# ── Standalone connectivity test (mirrors "Chapter 01") ───────────────────────
def test_connection() -> bool:
    """Verify the S/4 connection and print a human-readable report."""
    from dotenv import load_dotenv
    load_dotenv()

    print("Step 1: Checking configuration...")
    if not is_configured() or not os.environ.get("S4_USERNAME"):
        print("  ERROR: Missing S4_BASE_URL / S4_USERNAME / S4_PASSWORD in .env")
        return False
    print("  ✓ Configuration loaded")

    client = os.environ.get("S4_CLIENT", "550")
    print(f"\nStep 2: Session (client {client}, verify={_verify()})")
    session = _session()
    print("  ✓ Session created")

    print("\nStep 3: Maintenance Order API ($count)...")
    try:
        resp = session.get(
            f"{_base_url()}{SRV_MAINT_ORDER}/MaintenanceOrder/$count",
            params={"sap-client": client},
            headers={"Accept": "text/plain"},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"  ✓ Connected — MaintenanceOrder count: {resp.text.strip()}")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        hint = {401: "check S4_USERNAME/S4_PASSWORD", 404: "check S4_BASE_URL / service activation"}.get(code, "")
        print(f"  ERROR: HTTP {code} {('— ' + hint) if hint else ''}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print("\nStep 4: Work Center API ($count)...")
    try:
        resp = session.get(
            f"{_base_url()}{SRV_WORK_CENTER}/WorkCenterCapacity/$count",
            params={"sap-client": client},
            headers={"Accept": "text/plain"},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"  ✓ WorkCenterCapacity count: {resp.text.strip()}")
    except Exception as e:
        print(f"  WARNING: Work Center API test failed: {e}")

    print("\n" + "=" * 50)
    print("SUCCESS! S/4HANA connection is ready.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_connection() else 1)
