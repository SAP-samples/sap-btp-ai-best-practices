"""Offline checks for the S/4 check/post service, posted-advice immutability and the HTTP surface."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.email_ingestion import corrections
from app.email_ingestion.domain import email_status
from app.routers import s4 as s4_router
from app.s4 import service
from app.s4.client import S4HTTPError
from app.security import get_api_key
from tests.unit.test_s4_validation import GOLDEN_ADVICE
from tests.unit.workspace_support import MemoryStore

INVOICE_ROW = {"CompanyCode": "CA01", "FiscalYear": "2025", "AccountingDocument": "1400000001", "AccountingDocumentItem": "001",
               "Customer": "0010053628", "DocumentReferenceID": "8700276154", "AmountInTransactionCurrency": "10596.28",
               "TransactionCurrency": "CAD", "IsCleared": False, "ClearingAccountingDocument": ""}
KEY = {"CompanyCode": "CA01", "PaymentAdviceAccountType": "D", "PaymentAdviceAccount": "0010053628", "PaymentAdvice": "0425092800000001"}


class FakeS4:
    """Serves the open-item cube, the S/4 duplicate search, creation and read-back."""
    def __init__(self, cleared=False, existing=False, create_error=None):
        self.config = SimpleNamespace(mode="direct")
        self.cleared, self.existing, self.create_error, self.created = cleared, existing, create_error, []

    def get_json(self, service_name, path, params=None):
        if path == "/A_OperationalAcctgDocItemCube":
            rows = [dict(INVOICE_ROW, IsCleared=self.cleared)] if "'8700276154'" in params["$filter"] else []
            return {"d": {"results": rows}}
        if path == "/A_PaymentAdvice":
            return {"d": {"results": [KEY] if self.existing else []}}
        if path == "/A_CustomerCompany":
            return {"d": {"results": [{"Customer": "0010053628", "CompanyCode": "CA01"}]}}
        return {"d": {**KEY, "to_PaymentAdviceItem": {"results": [{"PaymentAdviceItem": "00001"}]}}}

    def post_json(self, service_name, path, payload):
        if self.create_error:
            raise self.create_error
        self.created.append(payload)
        return {"d": KEY}


def workspace(status="reviewed", **s4):
    """One email with one interpreted Contoso advice."""
    store = MemoryStore()
    store.insert("email", {}, status="ready", record_id="e1")
    result = copy.deepcopy(GOLDEN_ADVICE)
    advice = {"result": result, "verification": {"suspected_duplicate_groups": []}, "corrections": []}
    if s4:
        advice["s4"] = s4
    store.insert("advice", advice, parent="e1", status=status, record_id="a1")
    return store


class CheckTests(unittest.TestCase):
    """The draft check is stored on the advice and never writes to S/4."""
    def test_check_stores_ready_preview_and_trace(self):
        store, s4 = workspace(), FakeS4()
        saved = service.check(store, s4, "a1", 0)
        self.assertTrue(saved["s4"]["ready"])
        self.assertEqual(saved["s4"]["payload"]["PaymentAdviceAccount"], "0010053628")
        self.assertEqual(saved["s4"]["result_hash"], service.result_hash(saved["result"]))
        self.assertEqual(len(saved["s4"]["trace"]), 1)
        self.assertEqual(s4.created, [])


class AliasTests(unittest.TestCase):
    """Company code aliases come from configuration and are only meant for demo systems."""
    def test_alias_parsing(self):
        self.assertEqual(service.company_code_aliases("ca01=Z291, US01 = 1710,broken"), {"CA01": "Z291", "US01": "1710"})
        self.assertEqual(service.company_code_aliases(""), {})
        self.assertEqual(service.reason_code_aliases("316=060, 319=030"), {"316": "060", "319": "030"})


class PostTests(unittest.TestCase):
    """Posting is explicit, fresh, single and final."""
    def test_post_creates_advice_marks_posted_and_keeps_audit(self):
        store, s4 = workspace(), FakeS4()
        saved = service.post(store, s4, "a1", 0)
        self.assertEqual(saved["status"], "posted")
        self.assertEqual(saved["s4"]["posted"]["key"]["PaymentAdvice"], "0425092800000001")
        self.assertEqual(len(s4.created), 1)
        audit = store.list("s4_posting", parent="a1")
        self.assertEqual(audit[0]["payload"]["CompanyCode"], "CA01")
        self.assertEqual(store.get("e1")["status"], "posted")

    def test_posted_advice_is_immutable(self):
        store = workspace()
        saved = service.post(store, FakeS4(), "a1", 0)
        with self.assertRaisesRegex(ValueError, "posted to S/4HANA"):
            corrections.edit(store, "a1", saved["revision"], "header", "payment_amount", 1, "late change")
        with self.assertRaisesRegex(ValueError, "posted to S/4HANA"):
            service.check(store, FakeS4(), "a1", saved["revision"])

    def test_unreviewed_advice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            service.post(workspace(status="ready"), FakeS4(), "a1", 0)

    def test_fresh_check_blocks_when_invoice_was_cleared_meanwhile(self):
        store, s4 = workspace(), FakeS4(cleared=True)
        with self.assertRaisesRegex(ValueError, "blocking"):
            service.post(store, s4, "a1", 0)
        self.assertEqual(s4.created, [])
        self.assertEqual(store.get("a1")["status"], "reviewed")
        self.assertFalse(store.get("a1")["s4"]["ready"])

    def test_existing_s4_advice_is_refused_as_duplicate(self):
        s4 = FakeS4(existing=True)
        with self.assertRaisesRegex(ValueError, "already holds"):
            service.post(workspace(), s4, "a1", 0)
        self.assertEqual(s4.created, [])

    def test_interrupted_post_adopts_the_advice_s4_already_created(self):
        s4 = FakeS4(existing=True)
        saved = service.post(workspace(posting_started_at="2026-09-28T10:00:00+00:00"), s4, "a1", 0)
        self.assertEqual(s4.created, [])
        self.assertTrue(saved["s4"]["posted"]["adopted"])

    def test_s4_rejection_is_saved_and_advice_stays_reviewed(self):
        store = workspace()
        error = S4HTTPError("POST", "/A_PaymentAdvice", 400, "Bad Request", '{"error":{"message":{"value":"Reason code 316 not defined"}}}')
        with self.assertRaisesRegex(service.S4PostError, "Reason code 316"):
            service.post(store, FakeS4(create_error=error), "a1", 0)
        advice = store.get("a1")
        self.assertEqual(advice["status"], "reviewed")
        self.assertEqual(advice["s4"]["post_error"], "Reason code 316 not defined")

    def test_email_rollup_with_posted_advices(self):
        self.assertEqual(email_status(["posted", "posted"]), "posted")
        self.assertEqual(email_status(["posted", "reviewed"]), "reviewed")
        self.assertEqual(email_status(["posted", "needs_review"]), "needs_review")


class HttpTests(unittest.TestCase):
    """Domain and S/4 failures map to clear HTTP statuses."""
    def client(self, store, s4):
        app = FastAPI()
        app.include_router(s4_router.router, prefix="/api/payment-advice/advices")
        app.dependency_overrides[get_api_key] = lambda: "ok"
        app.state.workspace = store
        self.addCleanup(patch.stopall)
        patch.object(service, "s4_client", lambda: s4).start()
        return TestClient(app)

    def test_check_then_post_over_http(self):
        with self.client(workspace(), FakeS4()) as http:
            checked = http.post("/api/payment-advice/advices/a1/s4/check", json={"revision": 0})
            self.assertEqual(checked.status_code, 200)
            posted = http.post("/api/payment-advice/advices/a1/s4/post", json={"revision": checked.json()["revision"]})
            self.assertEqual(posted.json()["status"], "posted")

    def test_s4_rejection_is_bad_gateway_and_stale_revision_conflicts(self):
        error = S4HTTPError("POST", "/A_PaymentAdvice", 400, "Bad Request", '{"error":{"message":{"value":"No number range"}}}')
        with self.client(workspace(), FakeS4(create_error=error)) as http:
            self.assertEqual(http.post("/api/payment-advice/advices/a1/s4/post", json={"revision": 5}).status_code, 409)
            response = http.post("/api/payment-advice/advices/a1/s4/post", json={"revision": 0})
            self.assertEqual(response.status_code, 502)
            self.assertIn("No number range", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
