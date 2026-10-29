"""Offline checks for the read-only S/4 discovery probe."""
import unittest

from app.s4.client import S4HTTPError
from app.s4.probe import run_probe


class FakeClient:
    """Answers probe reads from a small table; unknown services fail like S/4 would."""
    def __init__(self, inactive=(), network_error=None):
        self.inactive, self.network_error, self.calls = set(inactive), network_error, 0

    def get_text(self, service, path=""):
        self.calls += 1
        if self.network_error:
            raise S4HTTPError("GET", service, 0, self.network_error)
        if service in self.inactive:
            raise S4HTTPError("GET", service, 404, "Not Found")
        return "<edmx/>"

    def get_json(self, service, path="", params=None):
        if service in self.inactive:
            raise S4HTTPError("GET", service, 404, "Not Found")
        filt = (params or {}).get("$filter", "")
        if path == "/A_CustomerCompany" and "Customer eq" in filt:
            rows = [{"Customer": "0010053628", "CompanyCode": "Z291"}] if "'0010053628'" in filt else []
        elif path == "/A_CustomerCompany":
            rows = [{"Customer": "0090000001", "ReconciliationAccount": "12100000", "PaymentTerms": "0001"},
                    {"Customer": "0090000002", "ReconciliationAccount": "12100000", "PaymentTerms": "0002"}]
        elif path == "/A_OperationalAcctgDocItemCube" and "AccountingDocumentType" in filt:
            rows = [{"GLAccount": "41000000", "TaxCode": "V0", "TaxJurisdiction": "CA00", "ProfitCenter": "YB700"},
                    {"GLAccount": "41000000", "TaxCode": "V0", "TaxJurisdiction": "CA00", "ProfitCenter": "YB700"},
                    {"GLAccount": "22000000", "TaxCode": "", "TaxJurisdiction": "", "ProfitCenter": ""}]
        else:
            rows = {
                "/A_CompanyCode": [{"CompanyCode": "Z291", "CompanyCodeName": "Demo Canada", "Country": "CA", "Currency": "CAD"}],
                "/A_OperationalAcctgDocItemCube": [],
                "/A_PaymentAdvice": [],
                "/A_BusinessPartner": [{"BusinessPartner": "90000001", "Customer": "0090000001", "BusinessPartnerGrouping": "BP02"}],
            }[path]
        return {"d": {"results": rows}}


class ProbeTests(unittest.TestCase):
    """Every check reports independently so one run lists all gaps."""
    def test_prepared_system_reports_conventions_and_demo_customers(self):
        rows = {r["check"]: r for r in run_probe(FakeClient(), "Z291", ["10053628", "10059152"])}
        self.assertTrue(all(row["ok"] for row in rows.values()), rows)
        self.assertIn("Z291 present", rows["company codes"]["detail"])
        self.assertIn("reconciliation accounts: 12100000 (2)", rows["customer master conventions"]["detail"])
        self.assertIn("groupings: BP02 (1)", rows["business partner groupings"]["detail"])
        self.assertIn("41000000 (2), 22000000 (1)", rows["revenue accounts"]["detail"])
        self.assertIn("tax codes: V0 (2); jurisdictions: CA00 (2); profit centers: YB700 (2)", rows["revenue accounts"]["detail"])
        self.assertEqual(rows["demo customers"]["detail"], "present: 0010053628; missing (to create): 0010059152")

    def test_failures_are_collected_with_actionable_hints(self):
        rows = run_probe(FakeClient(inactive={"API_PAYMENT_ADVICE_SRV"}))
        failed = {row["check"]: row["detail"] for row in rows if not row["ok"]}
        self.assertEqual(set(failed), {"service API_PAYMENT_ADVICE_SRV", "payment advice read"})
        self.assertIn("not activated", failed["service API_PAYMENT_ADVICE_SRV"])

    def test_unreachable_system_stops_after_first_check_and_explains_tls(self):
        error = ("HTTPSConnectionPool(host='10.47.32.49', port=44301): Max retries exceeded with url: /x "
                 "(Caused by SSLError(SSLCertVerificationError(\"hostname '10.47.32.49' doesn't match\")))")
        client = FakeClient(network_error=error)
        rows = run_probe(client)
        self.assertEqual(client.calls, 1)
        self.assertEqual([r["check"] for r in rows], ["service API_PAYMENT_ADVICE_SRV", "remaining checks"])
        self.assertIn("S4_VERIFY=false", rows[0]["detail"])
        self.assertIn("doesn't match", rows[0]["detail"])

    def test_untrusted_ca_is_explained_as_trust_problem(self):
        error = ("HTTPSConnectionPool(host='10.47.32.49', port=44301): Max retries exceeded with url: /x (Caused by "
                 "SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                 "unable to get local issuer certificate (_ssl.c:1010)')))")
        detail = run_probe(FakeClient(network_error=error))[0]["detail"]
        self.assertIn("SAPNetCA_G2", detail)
        self.assertIn("S4_VERIFY=false", detail)


if __name__ == "__main__":
    unittest.main()
