"""Verify exact candidate scope, durable uploads, and optimistic run revisions."""

import io
import tempfile
import unittest
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from app.models.workspace import CandidateScope
from app.services.database.backend import BackendType, DatabaseBackend
from app.services.optimizer.artifact_store import OptimizerArtifactStore
from app.services.workspace.analysis_store import AnalysisStore
from app.services.workspace.candidates import choose_candidates, canonical_candidates
from app.services.workspace.run_store import RunStore
from app.services.workspace.schema import transaction
from app.services.workspace.service import WorkspaceService


def offer_bytes(invalid=False):
    """Return an uploaded workbook with two distinct rows sharing a reference."""
    book = Workbook()
    sheet = book.active
    sheet.title = "Offer"
    sheet.append(["PROGRAMA", "ID SELLER", "SELLER", "ID DEBTOR", "DEBTOR",
                  "INVOICE REF", "ORIGINAL CURRENCY", "ISSUANCE DATE", "DUE DATE",
                  "AMOUNT ORIGINAL", "DOC NUMBER", "FISCAL YEAR"])
    for number in (1, 2):
        sheet.append(["P1", "S1", "Seller", "C1", "Customer", "same", "EUR",
                      "bad" if invalid and number == 2 else "2026-02-01",
                      "2026-02-12", 50, str(number), "2026"])
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


class CandidateScopeTests(unittest.TestCase):
    """Reject accidental changes of population at the solver boundary."""

    def setUp(self):
        """Create two eligible source rows and one rule rejection."""
        self.rows = [dict(row_id="a", eligible=True, invoice={"invoice_ref": "same"}),
                     dict(row_id="b", eligible=False, invoice={"invoice_ref": "other"}),
                     dict(row_id="c", eligible=True, invoice={"invoice_ref": "same"})]

    def test_all_eligible_preserves_duplicate_references(self):
        """Reference text must never replace source-row identity."""
        actual = choose_candidates(self.rows, CandidateScope(mode="all_eligible"))
        self.assertEqual([row["row_id"] for row in actual], ["a", "c"])

    def test_selected_scope_is_exact(self):
        """Only explicitly requested eligible rows reach the candidate adapter."""
        actual = choose_candidates(self.rows, CandidateScope(mode="selected", row_ids=["c"]))
        self.assertEqual([row["row_id"] for row in actual], ["c"])

    def test_invalid_selections_are_rejected(self):
        """Unknown, rejected, duplicated and empty selections are not guessed."""
        for ids in ([], ["foreign"], ["b"], ["a", "a"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                choose_candidates(self.rows, CandidateScope(mode="selected", row_ids=ids))
        with self.assertRaises(ValueError):
            choose_candidates(self.rows, CandidateScope(mode="all_eligible", row_ids=["a"]))


class WorkspaceAnalysisTests(unittest.TestCase):
    """Exercise real persistence through an explicitly isolated SQL test backend."""

    def setUp(self):
        """Create independent stores without consulting environment credentials."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.db"
        self.backend = DatabaseBackend(BackendType.SQLITE)
        self.analyses = AnalysisStore(self.backend, self.path)
        self.runs = RunStore(self.backend, self.path)
        self.service = WorkspaceService(self.analyses, self.runs)
        self.content = offer_bytes()

    def analyze(self, key="request-1", **settings):
        """Analyze the small workbook with a fixed, reproducible purchase date."""
        return self.service.analyze(self.content, "offer.xlsx", date(2026, 2, 2), settings, key)

    def test_retry_reopens_same_analysis_with_original_bytes(self):
        """A network retry must not append duplicate historical observations."""
        first = self.analyze()
        second = self.analyze()
        reopened = AnalysisStore(self.backend, self.path)
        self.assertEqual(first["analysis_id"], second["analysis_id"])
        self.assertEqual(reopened.list_analyses(20, 0)["total"], 1)
        self.assertEqual(reopened.original_content(first["analysis_id"]), self.content)
        self.assertEqual(reopened.get(first["analysis_id"])["eligible_count"], 2)

    def test_batch_failure_rolls_back_upload_and_source_rows(self):
        """A duplicate source ID inside a batch must not leave a partial saved offer."""
        rows=[{'row_id':'duplicate','source_row_number':number,'eligible':True} for number in (2,3)]
        with self.assertRaises(Exception):
            self.analyses.create(self.content,'broken.xlsx',date(2026,2,2),{},rows,'broken-batch')
        self.assertEqual(self.analyses.list_analyses()['total'],0)
        with self.backend.get_connection(self.path) as connection:
            count=connection.execute('SELECT COUNT(*) FROM RECEIVABLES_SOURCE_ROWS').fetchone()[0]
        self.assertEqual(count,0)

    def test_hana_scalar_read_preserves_oversized_diagnostics(self):
        """Small rows use bulk text, while a large record is read whole without truncation."""
        import json
        from contextlib import contextmanager
        small={'row_id':'small'};large={'row_id':'large','diagnostic':'é'*6000}
        cursor=Mock();cursor.fetchall.return_value=[(json.dumps(small),19,'small'),('truncated',7000,'large')]
        cursor.fetchone.return_value=(json.dumps(large),)
        @contextmanager
        def connection(*args):
            """Inject a SQL cursor with small and oversized result representations."""
            yield cursor
        store=object.__new__(AnalysisStore);store.backend=Mock(is_hana=True);store.db_path=None
        with patch.object(store,'get',return_value={}),patch('app.services.workspace.analysis_store.transaction',connection):
            self.assertEqual(store.all_rows('offer'),[small,large])
        self.assertEqual(cursor.execute.call_count,2)

    def test_key_reuse_with_changed_input_is_rejected(self):
        """An idempotency key cannot quietly substitute a different rule snapshot."""
        self.analyze()
        with self.assertRaises(ValueError):
            self.analyze(nddt=20)

    def test_reanalysis_preserves_source_identity(self):
        """Changed settings create a new analysis but retain physical source identities."""
        first, second = self.analyze(), self.analyze("request-2", nddt=20)
        a = self.analyses.list_rows(first["analysis_id"], {}, 20, 0)["items"]
        b = self.analyses.list_rows(second["analysis_id"], {}, 20, 0)["items"]
        self.assertNotEqual(first["analysis_id"], second["analysis_id"])
        self.assertEqual([r["row_id"] for r in a], [r["row_id"] for r in b])
        self.assertEqual(len(set(r["row_id"] for r in a)), 2)
        self.assertEqual([r["source_row_number"] for r in a], [2, 3])
        self.assertEqual(second["eligible_count"], 0)

    def test_malformed_row_cannot_silently_disappear(self):
        """A partial parse must produce a source-specific error and no saved analysis."""
        with self.assertRaisesRegex(ValueError, "Row 3"):
            self.service.analyze(offer_bytes(True), "offer.xlsx", date(2026, 2, 2), {}, "invalid")
        self.assertEqual(self.analyses.list_analyses(20, 0)["total"], 0)

    def test_run_creation_requires_no_second_file(self):
        """A run snapshots eligible source IDs directly from the saved analysis."""
        analysis = self.analyze()
        run = self.service.create_run(analysis["analysis_id"], CandidateScope(mode="all_eligible"))
        reopened = RunStore(self.backend, self.path).get(run["run_id"])
        self.assertEqual(len(reopened["row_ids"]), 2)
        self.assertEqual(reopened["status"], "draft")
        self.assertTrue(reopened["readiness_issues"])

    def test_delete_analyses_removes_related_hana_data_only_for_selected_uploads(self):
        """Deleting one upload must remove every child record without touching another upload."""
        deleted_analysis = self.analyze("delete-request")
        retained_analysis = self.analyze("retain-request")
        deleted_run = self.service.create_run(
            deleted_analysis["analysis_id"], CandidateScope(mode="all_eligible")
        )
        retained_run = self.service.create_run(
            retained_analysis["analysis_id"], CandidateScope(mode="all_eligible")
        )
        artifacts = OptimizerArtifactStore(db_path=self.path, backend=self.backend)
        artifacts.put_binary(deleted_run["run_id"], "all-files", b"delete")
        artifacts.put_binary(retained_run["run_id"], "all-files", b"retain")
        with transaction(self.backend, self.path) as cursor:
            for analysis_id, token in (
                (deleted_analysis["analysis_id"], "delete-token"),
                (retained_analysis["analysis_id"], "retain-token"),
            ):
                cursor.execute(
                    "INSERT INTO RECEIVABLES_ANALYSIS_EXPORTS "
                    "(scope_token,analysis_id,eligibility,insights_excel,insights_pdf) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (token, analysis_id, b"eligibility", b"insights", b"pdf"),
                )

        result = self.service.delete_analyses([deleted_analysis["analysis_id"]])

        self.assertEqual(
            result,
            {"deleted_analysis_ids": [deleted_analysis["analysis_id"]], "deleted_count": 1},
        )
        with self.assertRaises(LookupError):
            self.analyses.get(deleted_analysis["analysis_id"])
        with self.assertRaises(LookupError):
            self.runs.get(deleted_run["run_id"])
        self.assertIsNone(artifacts.get_binary(deleted_run["run_id"], "all-files"))
        self.assertEqual(self.analyses.get(retained_analysis["analysis_id"])["filename"], "offer.xlsx")
        self.assertEqual(self.runs.get(retained_run["run_id"])["analysis_id"], retained_analysis["analysis_id"])
        self.assertEqual(artifacts.get_binary(retained_run["run_id"], "all-files"), b"retain")
        with transaction(self.backend, self.path) as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM RECEIVABLES_SOURCE_ROWS WHERE analysis_id = ?",
                (deleted_analysis["analysis_id"],),
            )
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute(
                "SELECT COUNT(*) FROM RECEIVABLES_ANALYSIS_EXPORTS WHERE analysis_id = ?",
                (deleted_analysis["analysis_id"],),
            )
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute(
                "SELECT COUNT(*) FROM RECEIVABLES_ANALYSIS_EXPORTS WHERE analysis_id = ?",
                (retained_analysis["analysis_id"],),
            )
            self.assertEqual(cursor.fetchone()[0], 1)
            cursor.execute(
                "SELECT COUNT(*) FROM optimizer_process_artifacts WHERE process_id = ?",
                (deleted_run["run_id"],),
            )
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_revision_conflict_and_terminal_immutability(self):
        """Stale writes and edits to completed results must not overwrite saved inputs."""
        analysis = self.analyze()
        run = self.service.create_run(analysis["analysis_id"], CandidateScope(mode="all_eligible"))
        updated = self.runs.compare_and_swap(run["run_id"], 0, {"settings": {"horizon_weeks": 12}})
        self.assertEqual(updated["revision"], 1)
        with self.assertRaises(ValueError):
            self.runs.compare_and_swap(run["run_id"], 0, {"settings": {}})
        self.runs.compare_and_swap(run["run_id"], 1, {"status": "completed"})
        with self.assertRaises(ValueError):
            self.runs.compare_and_swap(run["run_id"], 2, {"settings": {}})

    def test_adapter_requires_explicit_mapping_and_preserves_amount(self):
        """Facility and FX values must come from settings, never guessed defaults."""
        analysis = self.analyze()
        rows = self.analyses.list_rows(analysis["analysis_id"], {}, 20, 0)["items"]
        with self.assertRaises(ValueError):
            canonical_candidates(analysis, rows, {})
        frame = canonical_candidates(analysis, rows, {"seller_to_facility": {"S1": "F1"}})
        self.assertEqual(frame["Company Code"].tolist(), ["F1", "F1"])
        self.assertEqual(frame["Customer"].tolist(), ["C1", "C1"])
        self.assertEqual(frame["Purchase Price"].tolist(), [50, 50])
        self.assertEqual(frame["row_id"].tolist(), [row["row_id"] for row in rows])

    def test_http_upload_to_run_and_structured_selection_error(self):
        """Real HTTP multipart analysis hands rows to a run with no second upload."""
        from app.routers.workspace import router, get_workspace_service
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_workspace_service] = lambda: self.service
        with patch("app.security.API_KEY", "test-only"), TestClient(app) as client:
            self.assertEqual(client.get("/api/workspace/analyses").status_code, 403)
            client.headers["X-API-Key"] = "test-only"
            response = client.post("/api/workspace/analyses", files={"file": ("offer.xlsx", self.content)},
                                   data={"analysis_date": "2026-02-02", "request_key": "http-1", "settings": "{}"})
            self.assertEqual(response.status_code, 200, response.text)
            analysis_id = response.json()["analysis_id"]
            result = client.post("/api/workspace/runs", json={"analysis_id": analysis_id})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(len(result.json()["row_ids"]), 2)
            run_id=result.json()["run_id"]
            draft={"planning_start":"2026-02-02","horizon_weeks":3,
                   "seller_to_facility":{"S1":"F1"},"customer_limits":{"C1":"100"},
                   "facility_limits_by_company_code":{"F1":"100"},"opening_confirmed_zero":True}
            preview=client.post(f"/api/workspace/runs/{run_id}/settings/preview",
                                json={"expected_revision":0,"settings":draft})
            self.assertEqual(preview.status_code,200,preview.text)
            self.assertEqual(preview.json()["readiness_issues"],[])
            self.assertEqual(client.get(f"/api/workspace/runs/{run_id}").json()["revision"],0)
            saved=client.put(f"/api/workspace/runs/{run_id}/settings",json={"expected_revision":0,"settings":draft})
            self.assertEqual(saved.json()["revision"],1)
            stale=client.put(f"/api/workspace/runs/{run_id}/settings",json={"expected_revision":0,"settings":draft})
            self.assertEqual(stale.status_code,409)
            invalid = client.post("/api/workspace/runs", json={"analysis_id": analysis_id,
                                  "candidate_scope": {"mode": "selected", "row_ids": ["foreign"]}})
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(invalid.json()["detail"]["fields"][0]["row_ids"], ["foreign"])
            insights = client.post(f"/api/workspace/analyses/{analysis_id}/insights", json={"row_ids": []})
            self.assertEqual(insights.status_code, 200, insights.text)
            snapshot = insights.json()
            self.assertEqual(snapshot["current_metrics"]["total"], 0)
            exported = client.get(f"/api/workspace/analyses/{analysis_id}/exports/insights-pdf",
                                  params={"scope_token":snapshot["scope_token"]})
            self.assertEqual(exported.status_code, 200)
            self.assertTrue(exported.content.startswith(b"%PDF"))

    def test_http_bulk_delete_requires_selection_and_removes_the_selected_upload(self):
        """The authenticated bulk endpoint rejects empty requests and deletes selected IDs."""
        from app.routers.workspace import router, get_workspace_service
        analysis = self.analyze("delete-through-http")
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_workspace_service] = lambda: self.service
        with patch("app.security.API_KEY", "test-only"), TestClient(app) as client:
            client.headers["X-API-Key"] = "test-only"
            empty = client.request("DELETE", "/api/workspace/analyses", json={"analysis_ids": []})
            self.assertEqual(empty.status_code, 422)
            invalid = client.request(
                "DELETE", "/api/workspace/analyses", json={"analysis_ids": ["not-an-analysis-id"]}
            )
            self.assertEqual(invalid.status_code, 422)
            response = client.request(
                "DELETE",
                "/api/workspace/analyses",
                json={"analysis_ids": [analysis["analysis_id"]]},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["deleted_analysis_ids"], [analysis["analysis_id"]])
            self.assertEqual(client.get(f"/api/workspace/analyses/{analysis['analysis_id']}").status_code, 404)
