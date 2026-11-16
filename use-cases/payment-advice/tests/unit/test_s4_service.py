"""Offline checks for the S/4 check/post/release service, the edit gate of unfinished posts and the HTTP surface.

Every S/4 call goes to the in-file `FakeS4`; no real S/4 system is contacted.
"""
import copy
import os
import re
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.email_ingestion import corrections
from app.email_ingestion.domain import email_status
from app.routers import s4 as s4_router
from app.s4 import posting, service
from app.s4.client import S4HTTPError, odata_str
from app.s4.payment_advice import KEY_FIELDS
from app.security import get_api_key
from tests.unit.test_s4_validation import GOLDEN_ADVICE
from tests.unit.workspace_support import MemoryStore

INVOICE_ROW = {"CompanyCode": "CA01", "FiscalYear": "2025", "AccountingDocument": "1400000001", "AccountingDocumentItem": "001",
               "Customer": "0010053628", "DocumentReferenceID": "8700276154", "AmountInTransactionCurrency": "10596.28",
               "TransactionCurrency": "CAD", "IsCleared": False, "ClearingAccountingDocument": ""}
# Start of an interrupted post, and "now" 30 seconds (young claim) or 10 minutes (old claim) later.
STARTED, YOUNG, OLD = "2026-09-28T10:00:00+00:00", "2026-09-28T10:00:30+00:00", "2026-09-28T10:10:00+00:00"


def timeout(status=0):
    """A create failure without a definite answer (status 0 = no response)."""
    return S4HTTPError("POST", "/A_PaymentAdvice", status, "Read timed out")


class FakeS4:
    """Serves the open-item cube, master data, the payment advice search, creation and read-back.

    ``existing`` lists payloads S/4 already stores (numbered 0425092800000001, 0425092800000002, ...);
    ``create_error`` is raised by the create, after S/4 stored the advice when ``completes`` is set.
    """
    def __init__(self, cleared=False, existing=(), create_error=None, completes=False):
        self.config = SimpleNamespace(mode="direct")
        self.cleared, self.create_error, self.completes = cleared, create_error, completes
        self.stored, self.created = {}, []
        for payload in existing:
            self._store(payload)

    def _store(self, payload):
        """Keep a created advice under the next advice number and return its key."""
        number = f"04250928{len(self.stored) + 1:08d}"
        self.stored[number] = {**copy.deepcopy(payload), "PaymentAdvice": number}
        return {field: self.stored[number][field] for field in KEY_FIELDS}

    def get_json(self, service_name, path, params=None):
        if path == "/A_OperationalAcctgDocItemCube":
            rows = [dict(INVOICE_ROW, IsCleared=self.cleared)] if "'8700276154'" in params["$filter"] else []
            return {"d": {"results": rows}}
        if path == "/A_PaymentAdvice":
            rows = [{field: advice[field] for field in KEY_FIELDS} for advice in self.stored.values()
                    if f"PaymentAdviceHeaderText eq {odata_str(advice['PaymentAdviceHeaderText'])}" in params["$filter"]]
            return {"d": {"results": rows, "__count": str(len(rows))}}
        if path == "/A_CustomerCompany":
            return {"d": {"results": [{"Customer": "0010053628", "CompanyCode": "CA01"}]}}
        if path == "/A_Customer":
            return {"d": {"results": [{"Customer": "0010053628", "PostingIsBlocked": False}]}}
        if path == "/I_PaytDifferenceReasonVH":
            return {"d": {"results": [{"CompanyCode": "CA01", "PaymentDifferenceReason": code} for code in ("316", "319")]}}
        # Read-back of one advice: echo what is stored, items numbered as S/4 does.
        advice = self.stored[re.search(r"PaymentAdvice='(\d+)'", path).group(1)]
        items = [{**item, "PaymentAdviceItem": f"{number:05d}"} for number, item in enumerate(advice["to_PaymentAdviceItem"], 1)]
        header = {field: value for field, value in advice.items() if field != "to_PaymentAdviceItem"}
        return {"d": {**header, "to_PaymentAdviceItem": {"results": items}}}

    def post_json(self, service_name, path, payload):
        if self.create_error and not self.completes:
            raise self.create_error
        self.created.append(payload)
        key = self._store(payload)
        if self.create_error:
            raise self.create_error
        return {"d": key}


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


def ready_payload():
    """The payload the check builds for the workspace advice."""
    return service.run_check(FakeS4(), workspace().get("a1"))["payload"]


def claim():
    """Claim keys left by a post whose create ended without an answer."""
    return {"posting_started_at": STARTED, "claimed_payload": ready_payload(), "post_outcome": "unknown"}


@contextmanager
def at(now):
    """Run with the service clock at ``now`` and the S/4 request timeout at 120 seconds."""
    with patch.object(service, "_now", lambda: now), patch.dict(os.environ, {"S4_TIMEOUT_SECONDS": "120"}):
        yield


def claim_keys(advice):
    """The claim keys stored on an advice."""
    return {key: advice["s4"][key] for key in ("posting_started_at", "claimed_payload", "post_outcome") if key in advice["s4"]}


class CheckTests(unittest.TestCase):
    """The draft check is stored on the advice and never writes to S/4."""
    def test_check_stores_ready_preview_and_trace(self):
        store, s4 = workspace(), FakeS4()
        saved = service.check(store, s4, "a1", 0)
        self.assertTrue(saved["s4"]["ready"])
        self.assertEqual(saved["s4"]["payload"]["PaymentAdviceAccount"], "0010053628")
        self.assertEqual(saved["s4"]["result_hash"], service.result_hash(saved["result"]))
        # One reference pass finds the invoice; then linked items, customer master and reason codes are read.
        self.assertEqual([(call["entity"], call["field"]) for call in saved["s4"]["trace"]],
                         [("/A_OperationalAcctgDocItemCube", "DocumentReferenceID"),
                          ("/A_OperationalAcctgDocItemCube", "InvoiceReference"),
                          ("/A_CustomerCompany", "Customer"), ("/A_Customer", "Customer"),
                          ("/I_PaytDifferenceReasonVH", "CompanyCode")])
        self.assertEqual(saved["s4"]["summary"]["resolution"], {"unique": 1, "auto": 0, "reviewer": 0})
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
        self.assertEqual(saved["s4"]["posted"]["verification"]["items_stored"], 2)
        self.assertEqual(saved["s4"]["posted"]["verification"]["differences"], [])
        self.assertNotIn("posting_started_at", saved["s4"])
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
        s4 = FakeS4(existing=[ready_payload()])
        with self.assertRaisesRegex(ValueError, "already holds payment advice 0425092800000001"):
            service.post(workspace(), s4, "a1", 0)
        self.assertEqual(s4.created, [])

    def test_interrupted_post_adopts_the_advice_s4_already_created(self):
        s4 = FakeS4(existing=[ready_payload()])
        store = workspace(**claim())
        saved = service.post(store, s4, "a1", 0)
        self.assertEqual(s4.created, [])
        self.assertTrue(saved["s4"]["posted"]["adopted"])
        self.assertEqual(saved["s4"]["posted"]["key"]["PaymentAdvice"], "0425092800000001")
        self.assertEqual(claim_keys(saved), {})

    def test_s4_rejection_is_saved_and_advice_stays_reviewed(self):
        store = workspace()
        error = S4HTTPError("POST", "/A_PaymentAdvice", 400, "Bad Request", '{"error":{"message":{"value":"Reason code 316 not defined"}}}')
        with self.assertRaisesRegex(service.S4PostError, "Reason code 316") as caught:
            service.post(store, FakeS4(create_error=error), "a1", 0)
        self.assertFalse(caught.exception.unknown)
        advice = store.get("a1")
        self.assertEqual(advice["status"], "reviewed")
        self.assertEqual(advice["s4"]["post_error"], "Reason code 316 not defined")
        self.assertIn("post_failed_at", advice["s4"])
        # A definite rejection means nothing exists in S/4: the claim is cleared and editing is allowed again.
        self.assertEqual(claim_keys(advice), {})
        corrections.edit(store, "a1", advice["revision"], "header", "payment_amount", 1, "fix after rejection")

    def test_email_rollup_with_posted_advices(self):
        self.assertEqual(email_status(["posted", "posted"]), "posted")
        self.assertEqual(email_status(["posted", "reviewed"]), "reviewed")
        self.assertEqual(email_status(["posted", "needs_review"]), "needs_review")


class UnknownOutcomeTests(unittest.TestCase):
    """A create without a definite answer keeps its claim; the next post reconciles it with S/4."""
    def test_no_answer_and_gateway_error_keep_the_claim(self):
        for status in (0, 503):
            with self.subTest(status=status):
                store = workspace()
                with self.assertRaisesRegex(service.S4PostError, "outcome is unknown") as caught:
                    service.post(store, FakeS4(create_error=timeout(status)), "a1", 0)
                self.assertTrue(caught.exception.unknown)
                advice = store.get("a1")
                self.assertEqual(advice["status"], "reviewed")
                self.assertEqual(advice["s4"]["post_outcome"], "unknown")
                self.assertEqual(advice["s4"]["claimed_payload"], ready_payload())
                self.assertTrue(advice["s4"]["posting_started_at"])
                self.assertEqual(advice["s4"]["post_error"], "Read timed out")

    def test_check_keeps_the_claim(self):
        store = workspace(**claim())
        saved = service.check(store, FakeS4(), "a1", 0)
        self.assertTrue(saved["s4"]["ready"])
        self.assertEqual(claim_keys(saved), claim())

    def test_blocking_fresh_check_keeps_the_claim(self):
        store = workspace(**claim())
        with self.assertRaisesRegex(ValueError, "blocking"):
            service.post(store, FakeS4(cleared=True), "a1", 0)
        self.assertFalse(store.get("a1")["s4"]["ready"])
        self.assertEqual(claim_keys(store.get("a1")), claim())

    def test_post_again_adopts_the_single_equal_advice(self):
        store, s4 = workspace(), FakeS4(create_error=timeout(), completes=True)
        with self.assertRaises(service.S4PostError):
            service.post(store, s4, "a1", 0)
        s4.create_error = None
        saved = service.post(store, s4, "a1", store.get("a1")["revision"])
        self.assertEqual(len(s4.created), 1)  # only the interrupted request reached S/4
        self.assertEqual(saved["status"], "posted")
        self.assertTrue(saved["s4"]["posted"]["adopted"])
        self.assertEqual(saved["s4"]["posted"]["verification"]["differences"], [])
        self.assertEqual(store.list("s4_posting", parent="a1")[0]["payload"], ready_payload())

    def test_advice_s4_created_is_adopted_even_when_s4_has_cleared_the_invoice_since(self):
        # The interrupted create succeeded in S/4, and S/4 has already used the advice to clear the invoice:
        # a fresh check now blocks, but the advice S/4 holds is the one that was sent.
        store = workspace(**claim())
        s4 = FakeS4(cleared=True, existing=[ready_payload()])
        saved = service.post(store, s4, "a1", 0)
        self.assertEqual((saved["status"], saved["s4"]["posted"]["adopted"], s4.created), ("posted", True, []))
        self.assertEqual(store.list("s4_posting", parent="a1")[0]["payload"], ready_payload())

    def test_only_a_clear_client_error_counts_as_a_rejection(self):
        outcome = {status: posting.is_definite_rejection(S4HTTPError("POST", "/A_PaymentAdvice", status, "x"))
                   for status in (0, 400, 403, 404, 408, 429, 500, 503)}
        self.assertEqual(outcome, {0: False, 400: True, 403: True, 404: True, 408: False, 429: False, 500: False, 503: False})

    def test_different_existing_advice_is_not_adopted(self):
        s4 = FakeS4(existing=[dict(ready_payload(), PaidAmountInPaytCurrency="1.00")])
        store = workspace(**claim())
        with self.assertRaisesRegex(ValueError, "differs from this advice: Header PaidAmountInPaytCurrency"):
            service.post(store, s4, "a1", 0)
        self.assertEqual(s4.created, [])
        self.assertEqual(store.get("a1")["status"], "reviewed")

    def test_several_existing_advices_are_refused(self):
        s4 = FakeS4(existing=[ready_payload(), ready_payload()])
        with self.assertRaisesRegex(ValueError, "several payment advices.*0425092800000001, 0425092800000002"):
            service.post(workspace(**claim()), s4, "a1", 0)
        self.assertEqual(s4.created, [])

    def test_young_claim_waits_and_old_claim_creates_once(self):
        store, s4 = workspace(**claim()), FakeS4()
        with at(YOUNG), self.assertRaisesRegex(ValueError, "try again in 90 seconds"):
            service.post(store, s4, "a1", 0)
        self.assertEqual(s4.created, [])
        with at(OLD):
            saved = service.post(store, s4, "a1", 0)
        self.assertEqual(len(s4.created), 1)
        self.assertFalse(saved["s4"]["posted"]["adopted"])


class ReleaseTests(unittest.TestCase):
    """An unfinished post blocks edits until it is posted again or released."""
    def test_edit_is_refused_while_claimed_and_allowed_after_release(self):
        store = workspace(**claim())
        with self.assertRaisesRegex(ValueError, "unfinished"):
            corrections.edit(store, "a1", 0, "header", "payment_amount", 1, "late change")
        with at(OLD):
            released = service.release(store, FakeS4(), "a1", 0)
        self.assertEqual(claim_keys(released), {})
        edited = corrections.edit(store, "a1", released["revision"], "header", "payment_amount", 1, "late change")
        self.assertEqual(edited["status"], "needs_review")

    def test_release_is_refused_when_s4_holds_the_advice(self):
        with at(OLD), self.assertRaisesRegex(ValueError, "0425092800000001.*post the advice again"):
            service.release(workspace(**claim()), FakeS4(existing=[ready_payload()]), "a1", 0)

    def test_release_is_refused_for_a_young_claim(self):
        with at(YOUNG), self.assertRaisesRegex(ValueError, "try again in 90 seconds"):
            service.release(workspace(**claim()), FakeS4(), "a1", 0)

    def test_release_is_refused_without_a_claim(self):
        with self.assertRaisesRegex(ValueError, "no unfinished"):
            service.release(workspace(), FakeS4(), "a1", 0)


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


    def test_release_over_http_maps_errors_like_the_other_routes(self):
        url = "/api/payment-advice/advices/a1/s4/release"
        with at(OLD), self.client(workspace(**claim()), FakeS4(existing=[ready_payload()])) as http:
            self.assertEqual(http.post(url, json={"revision": 0}).status_code, 400)
        with at(OLD), self.client(workspace(**claim()), FakeS4()) as http:
            self.assertEqual(http.post(url, json={"revision": 5}).status_code, 409)
            response = http.post(url, json={"revision": 0})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("posting_started_at", response.json()["s4"])

    def test_unknown_outcome_is_bad_gateway_without_calling_it_a_rejection(self):
        with self.client(workspace(), FakeS4(create_error=timeout())) as http:
            response = http.post("/api/payment-advice/advices/a1/s4/post", json={"revision": 0})
            self.assertEqual(response.status_code, 502)
            self.assertIn("outcome is unknown", response.json()["detail"])
            self.assertNotIn("rejected", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
