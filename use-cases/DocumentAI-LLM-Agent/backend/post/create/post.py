import os
import json
import datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

BASE_URL = os.getenv("S4_BASE_URL").rstrip("/")
CLIENT = os.getenv("S4_CLIENT")
USER = os.getenv("S4_USERNAME")
PWD = os.getenv("S4_PASSWORD")

VERIFY = False

API_SO = f"{BASE_URL}/sap/opu/odata/sap/API_SALES_ORDER_SRV"

# =========================================================
# SESSION
# =========================================================

def create_session():

    s = requests.Session()

    s.auth = (USER, PWD)

    s.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })

    s.verify = VERIFY

    return s


# =========================================================
# SAP DATE FORMAT
# =========================================================

def sap_date(d: dt.date):

    ms = int(
        dt.datetime(
            d.year,
            d.month,
            d.day,
            tzinfo=dt.timezone.utc
        ).timestamp() * 1000
    )

    return f"/Date({ms})/"


# =========================================================
# FETCH CSRF TOKEN
# =========================================================

def fetch_csrf(session):

    headers = {
        "X-CSRF-Token": "Fetch"
    }

    params = {
        "sap-client": CLIENT,
        "$top": "1"
    }

    response = session.get(
        f"{API_SO}/A_SalesOrder",
        headers=headers,
        params=params,
        timeout=60
    )

    token = response.headers.get("X-CSRF-Token")

    if not token:
        raise Exception(
            f"Could not fetch CSRF token. Status={response.status_code}"
        )

    return token, response.cookies


# =========================================================
# PAYLOAD
# =========================================================

def build_payload():

    today = dt.date.today()

    payload = {

        # =================================================
        # HEADER
        # =================================================

        "SalesOrderType": "OR",

        "SalesOrganization": "200",

        "DistributionChannel": "10",

        "OrganizationDivision": "00",

        "SoldToParty": "10008470",

        "TransactionCurrency": "USD",

        "RequestedDeliveryDate": sap_date(
            today + dt.timedelta(days=7)
        ),

        "PricingDate": sap_date(today),

        # =================================================
        # ITEMS
        # =================================================

        "to_Item": {
            "results": [
                {
                    "SalesOrderItem": "10",

                    "Material": "MXA920W-S",

                    "RequestedQuantity": "1",

                    "RequestedQuantityUnit": "EA",

                    "PricingDate": sap_date(today)
                }
            ]
        }
    }

    return payload


# =========================================================
# CREATE SALES ORDER
# =========================================================

def create_sales_order(session, payload):

    token, cookies = fetch_csrf(session)

    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    params = {
        "sap-client": CLIENT
    }

    response = session.post(
        f"{API_SO}/A_SalesOrder",
        headers=headers,
        cookies=cookies,
        params=params,
        data=json.dumps(payload),
        timeout=120
    )

    print("\nSTATUS:", response.status_code)

    try:
        response.raise_for_status()

    except Exception:

        print("\nERROR RESPONSE:\n")

        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)

        raise

    return response.json()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("CREATE SALES ORDER")
    print("=" * 60)

    session = create_session()

    payload = build_payload()

    print("\nPAYLOAD:\n")
    print(json.dumps(payload, indent=2))

    print("\nCreating Sales Order...\n")

    result = create_sales_order(session, payload)

    data = result.get("d") or result

    print("\nSUCCESS!")
    print(f"Sales Order: {data.get('SalesOrder')}")
    print(f"Customer:    {data.get('SoldToParty')}")
    print(f"Type:        {data.get('SalesOrderType')}")


if __name__ == "__main__":
    main()