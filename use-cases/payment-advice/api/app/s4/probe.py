"""Read-only S/4 discovery: services, company codes, demo customers and seed parameters.

Run with ``python -m app.s4.probe [COMPANY_CODE]`` from ``api/``. Every check is a GET, so the probe is safe to run
against any system. Each check returns a row `{check, ok, detail}`; a failed
check never stops the others, so one run produces the full prerequisites list.

Besides prerequisites it reports the master-data conventions already used in
the company code (reconciliation accounts, payment terms, business partner
groupings, revenue accounts of customer invoices with their tax code, tax
jurisdiction and profit center). Demo customers and invoices are created with
those values, so nobody has to guess them.

S/4 configuration that no API can read or create (reason codes in OBBE, payment
advice type number ranges, customer tolerances) is listed in `MANUAL_CHECKS`
for the S/4 functional expert.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from .client import S4Client, S4HTTPError, odata_rows, odata_str
from .lookup import existing_customers, pad_account

SERVICES = (
    ("API_PAYMENT_ADVICE_SRV", "create payment advices"),
    ("API_OPLACCTGDOCITEMCUBE_SRV", "find open AR items by invoice reference"),
    ("API_BUSINESS_PARTNER", "check and create demo customers"),
    ("API_COMPANYCODE_SRV", "read company codes and currencies"),
)
MANUAL_CHECKS = (
    "Reason codes 316, 319, 321, 323 exist for the company code (OBBE / T053R).",
    "Payment advice type 04 has a number range (T053G, FBE1 numbering).",
    "Customer tolerance group allows the demo deductions (OBA3), or residuals go to manual processing.",
    "User has F_AVIK_BUK / F_AVIK_AVA authorization for the company code and advice type.",
)


def _hint(exc: S4HTTPError) -> str:
    """Turn common failures into an actionable one-line hint."""
    if exc.status_code == 0:
        reason = re.search(r"Caused by (.*)", exc.message)
        reason_text = (reason.group(1) if reason else exc.message)[-200:]
        if "local issuer" in exc.message or "self-signed" in exc.message:
            return (f"TLS check failed ({reason_text}); the server certificate is signed by a CA this Python does not "
                    "trust (SAP internal SAPNetCA_G2). Set S4_VERIFY=false in api/.env for the VPN-internal test system, "
                    "or point REQUESTS_CA_BUNDLE to a bundle containing that CA")
        if "SSL" in exc.message or "certificate" in exc.message:
            return (f"TLS check failed ({reason_text}); the certificate must match the S4_BASE_URL host name, "
                    "or set S4_VERIFY=false for a VPN-internal test system")
        return f"not reachable ({reason_text}); in direct mode connect to the SAP VPN"
    if exc.status_code in {401, 403}:
        return f"HTTP {exc.status_code}: user lacks authorization"
    if exc.status_code == 404:
        return "HTTP 404: OData service not activated (/IWFND/MAINT_SERVICE)"
    return f"HTTP {exc.status_code}: {exc.detail[:200]}"


def _check(name: str, action: Callable[[], str]) -> dict[str, Any]:
    """Run one read-only check and capture success or the failure hint (`reachable` False on network errors)."""
    try:
        return {"check": name, "ok": True, "detail": action()}
    except S4HTTPError as exc:
        return {"check": name, "ok": False, "detail": _hint(exc), "reachable": exc.status_code != 0}


def _top(values: Iterable[Any], limit: int = 3) -> str:
    """Most frequent non-empty values with counts, e.g. "12100000 (40), 12100010 (3)"."""
    counts = Counter(str(v) for v in values if v not in (None, ""))
    return ", ".join(f"{value} ({count})" for value, count in counts.most_common(limit)) or "none"


def run_probe(client: S4Client, company_code: str = "CA01",
              customers: Iterable[str] = ("10053628",)) -> list[dict[str, Any]]:
    """Run all discovery checks against S/4.

    Args:
        client: Configured S/4 client.
        company_code: Company code the demo should post into.
        customers: Demo customer numbers to look for in that company code.

    Returns:
        Ordered list of `{check, ok, detail}` rows.
    """
    def metadata(name: str, why: str) -> str:
        """Confirm the OData service is activated and readable."""
        client.get_text(name, "/$metadata")
        return f"$metadata ok ({why})"

    first = _check(f"service {SERVICES[0][0]}", lambda: metadata(*SERVICES[0]))
    if first.get("reachable") is False:  # every other check would fail (and wait) the same way
        return [{k: v for k, v in first.items() if k != "reachable"},
                {"check": "remaining checks", "ok": False, "detail": "skipped until S/4 is reachable"}]
    rows = [first] + [_check(f"service {name}", lambda name=name, why=why: metadata(name, why)) for name, why in SERVICES[1:]]
    code = odata_str(company_code)
    sample_customers: list[str] = []

    def company_codes() -> str:
        """List the target company code plus every Canadian / CAD company code."""
        filt = f"CompanyCode eq {code} or Country eq 'CA' or Currency eq 'CAD'"
        found = odata_rows(client.get_json("API_COMPANYCODE_SRV", "/A_CompanyCode", {
            "$filter": filt, "$select": "CompanyCode,CompanyCodeName,Country,Currency", "$top": "50"}))
        listed = ", ".join(f"{r['CompanyCode']} {r['CompanyCodeName']} ({r['Country']}, {r['Currency']})" for r in found)
        target = "present" if any(r["CompanyCode"] == company_code for r in found) else "MISSING"
        return f"{company_code} {target}; candidates: {listed or 'none'}"

    def open_items() -> str:
        """Confirm read access to customer items and show one sample."""
        sample = odata_rows(client.get_json("API_OPLACCTGDOCITEMCUBE_SRV", "/A_OperationalAcctgDocItemCube", {
            "$filter": f"FinancialAccountType eq 'D' and CompanyCode eq {code}",
            "$select": "CompanyCode,AccountingDocument,FiscalYear,Customer,DocumentReferenceID", "$top": "3"}))
        return f"{len(sample)} sample customer item(s) in {company_code}" + (
            f", e.g. {sample[0]['AccountingDocument']}/{sample[0]['FiscalYear']} ref {sample[0]['DocumentReferenceID']!r}" if sample else "")

    def advices() -> str:
        """Confirm read access to payment advices in the company code."""
        found = odata_rows(client.get_json("API_PAYMENT_ADVICE_SRV", "/A_PaymentAdvice", {
            "$filter": f"CompanyCode eq {code}", "$select": "PaymentAdvice,PaymentAdviceType", "$top": "5"}))
        return f"{len(found)} existing advice(s) readable in {company_code}"

    def customer_conventions() -> str:
        """Reconciliation accounts and payment terms used by existing customers of the company code."""
        found = odata_rows(client.get_json("API_BUSINESS_PARTNER", "/A_CustomerCompany", {
            "$filter": f"CompanyCode eq {code}", "$select": "Customer,ReconciliationAccount,PaymentTerms", "$top": "200"}))
        sample_customers.extend(row["Customer"] for row in found[:20])
        return (f"{len(found)} customer(s); reconciliation accounts: {_top(r.get('ReconciliationAccount') for r in found)}; "
                f"payment terms: {_top(r.get('PaymentTerms') for r in found)}")

    def groupings() -> str:
        """Business partner groupings (number ranges) of those customers."""
        if not sample_customers:
            return "no existing customers to learn from"
        found = odata_rows(client.get_json("API_BUSINESS_PARTNER", "/A_BusinessPartner", {
            "$filter": " or ".join(f"Customer eq {odata_str(c)}" for c in sample_customers),
            "$select": "BusinessPartner,Customer,BusinessPartnerGrouping", "$top": "20"}))
        same = sum(1 for r in found if pad_account(r.get("BusinessPartner")) == pad_account(r.get("Customer")))
        return (f"groupings: {_top(r.get('BusinessPartnerGrouping') for r in found)}; "
                f"{same}/{len(found)} customers share their BP number")

    def revenue_accounts() -> str:
        """G/L accounts, tax codes, jurisdictions and profit centers on the non-customer lines of DR invoices."""
        found = odata_rows(client.get_json("API_OPLACCTGDOCITEMCUBE_SRV", "/A_OperationalAcctgDocItemCube", {
            "$filter": f"CompanyCode eq {code} and AccountingDocumentType eq 'DR' and FinancialAccountType eq 'S'",
            "$select": "CompanyCode,FiscalYear,AccountingDocument,AccountingDocumentItem,GLAccount,TaxCode,"
                       "TaxJurisdiction,ProfitCenter", "$top": "200"}))
        return (f"revenue/offset G/L accounts on DR invoices: {_top((r.get('GLAccount') for r in found), 5)}; "
                f"tax codes: {_top(r.get('TaxCode') for r in found)}; "
                f"jurisdictions: {_top(r.get('TaxJurisdiction') for r in found)}; "
                f"profit centers: {_top(r.get('ProfitCenter') for r in found)}")

    def demo_customers() -> str:
        """Whether the demo customer numbers exist in the company code."""
        wanted = [pad_account(c) for c in customers if pad_account(c)]
        present = existing_customers(client, wanted, company_code)
        missing = [c for c in wanted if c not in present]
        return (f"present: {', '.join(sorted(present)) or 'none'}; "
                f"missing (to create): {', '.join(missing) or 'none'}")

    rows += [
        _check("company codes", company_codes),
        _check("open item read", open_items),
        _check("payment advice read", advices),
        _check("customer master conventions", customer_conventions),
        _check("business partner groupings", groupings),
        _check("revenue accounts", revenue_accounts),
        _check("demo customers", demo_customers),
    ]
    return [{k: v for k, v in row.items() if k != "reachable"} for row in rows]


if __name__ == "__main__":  # Cloud Foundry entry point (only api/ is pushed): python -m app.s4.probe [COMPANY_CODE]
    import json
    import sys

    from .client import load_s4_config

    report = run_probe(S4Client(load_s4_config()), *sys.argv[1:2])
    print(json.dumps({"checks": report, "manual_checks": list(MANUAL_CHECKS)}, indent=2))
    sys.exit(0 if all(row["ok"] for row in report) else 2)
