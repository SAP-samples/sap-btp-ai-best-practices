"""
Unit tests for recombination (concat mode) and extraction orchestration (step 3).

Offline: a fake Document AI client stands in for SAP, so no network or credentials
are used. Covers the concat-vs-dedupe distinction, header dedup/conflict, split
recombination order, and job polling (success, timeout, failure).

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice.aggregation import aggregate_documents  # noqa: E402
from app.payment_advice.extract import (  # noqa: E402
    ExtractionError,
    extract_document,
)


def _fields(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": k, "value": v} for k, v in mapping.items()]


def _extraction(header: dict[str, Any], lines: list[dict[str, Any]]) -> dict[str, Any]:
    return {"headerFields": _fields(header), "lineItems": [_fields(line) for line in lines]}


class FakeDoxClient:
    """Returns a queued extraction per upload, in call order."""

    def __init__(self, extractions: list[dict[str, Any]]) -> None:
        self._extractions = extractions
        self._i = 0
        self.jobs: dict[str, dict[str, Any]] = {}
        self.uploads: list[tuple[str, dict[str, Any]]] = []

    def upload_document_stream(self, file_name, file_obj, mime_type, **options):
        job_id = f"job{self._i}"
        self.jobs[job_id] = self._extractions[self._i]
        self.uploads.append((file_name, options))
        self._i += 1
        return {"id": job_id, "status": "PENDING"}

    def get_job(self, job_id, *, extracted_values=None, return_null_values=None):
        return {"status": "DONE", "extraction": self.jobs[job_id]}


class RawView(unittest.TestCase):
    """The raw view keeps every printed cell, aligned with the normalized line items."""

    def test_cell_that_fails_normalization_survives_as_printed_text(self) -> None:
        line = [{"name": "invoice", "value": "8700276154"},
                {"name": "region", "value": None, "rawValue": " A "},
                {"name": "amount", "value": "10596.28"}]
        empty = [{"name": "region", "value": None, "rawValue": None}]
        docs = [{"id": "a", "file_name": "a", "extraction": {
            "headerFields": [{"name": "currency", "value": None, "rawValue": "CAD"}],
            "lineItems": [line, empty, _fields({"invoice": "924610168612UL"})]}}]
        result = aggregate_documents({}, docs, dedupe_line_items=False)
        self.assertEqual(result["line_items"][0], {"invoice": "8700276154", "amount": "10596.28"})
        self.assertEqual(result["raw"]["line_items"], [
            {"invoice": "8700276154", "region": "A", "amount": "10596.28"},
            {"invoice": "924610168612UL"}])
        self.assertEqual(result["raw"]["headers"], {"currency": "CAD"})
        self.assertNotIn("currency", result["headers"])


class ConcatVersusDedupe(unittest.TestCase):
    """The dedupe flag decides whether identical lines merge or repeat."""

    def _two_docs_with_identical_line(self) -> list[dict[str, Any]]:
        line = {"invoice": "8700", "amount": "-160.23"}
        return [
            {"id": "a", "file_name": "a", "extraction": _extraction({"payer": "ACME"}, [line])},
            {"id": "b", "file_name": "b", "extraction": _extraction({"payer": "ACME"}, [line])},
        ]

    def test_concat_keeps_duplicate_rows(self) -> None:
        docs = self._two_docs_with_identical_line()
        result = aggregate_documents({}, docs, dedupe_line_items=False)
        self.assertEqual(len(result["line_items"]), 2, "concat must keep both identical rows")

    def test_dedupe_merges_duplicate_rows(self) -> None:
        docs = self._two_docs_with_identical_line()
        result = aggregate_documents({}, docs, dedupe_line_items=True)
        self.assertEqual(len(result["line_items"]), 1, "dedupe must merge identical rows")

    def test_header_dedup_no_conflict_when_equal(self) -> None:
        docs = self._two_docs_with_identical_line()  # both payer=ACME
        result = aggregate_documents({}, docs, dedupe_line_items=False)
        self.assertEqual(result["headers"]["payer"], "ACME")
        self.assertEqual(result["conflicts"], [])

    def test_header_conflict_when_different(self) -> None:
        docs = [
            {"id": "a", "file_name": "a", "extraction": _extraction({"payer": "ACME"}, [])},
            {"id": "b", "file_name": "b", "extraction": _extraction({"payer": "OTHER"}, [])},
        ]
        result = aggregate_documents({}, docs, dedupe_line_items=False)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["conflicts"][0]["field"], "payer")


class ExtractOrchestration(unittest.TestCase):
    """extract_document splits, uploads, polls, and recombines in order."""

    def _write_csv(self, path: Path, n: int) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "amount"])
            writer.writerows([[i, i] for i in range(n)])

    def test_single_part_no_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "advice.csv"
            self._write_csv(src, 10)
            client = FakeDoxClient([_extraction({"payer": "ACME"}, [{"seq": 0}, {"seq": 1}])])
            result = extract_document(
                src,
                client=client,
                schema={},
                dox_client_id="ai4u_payment_advice",
                out_dir=Path(tmp) / "out",
                max_line_items=100,
            )
            self.assertEqual(len(result.job_ids), 1)
            self.assertFalse(result.oversized)
            self.assertEqual(len(result.aggregate["line_items"]), 2)

    def test_split_recombines_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "advice.csv"
            self._write_csv(src, 205)  # -> 3 parts at limit 100
            # One extraction per part; each carries a unique seq marker per line.
            extractions = [
                _extraction({"payer": "ACME"}, [{"seq": 3 * k}, {"seq": 3 * k + 1}])
                for k in range(3)
            ]
            client = FakeDoxClient(extractions)
            result = extract_document(
                src,
                client=client,
                schema={},
                dox_client_id="ai4u_payment_advice",
                out_dir=Path(tmp) / "out",
                max_line_items=100,
            )
            self.assertTrue(result.oversized)
            self.assertEqual(len(result.job_ids), 3)
            self.assertEqual(len(client.uploads), 3)
            seqs = [int(line["seq"]) for line in result.aggregate["line_items"]]
            self.assertEqual(seqs, [0, 1, 3, 4, 6, 7], "parts recombine in upload order")

    def test_timeout_raises(self) -> None:
        clock = {"t": 0.0}

        class PendingClient(FakeDoxClient):
            def get_job(self, job_id, *, extracted_values=None, return_null_values=None):
                return {"status": "PENDING"}

        def fake_now() -> float:
            clock["t"] += 5.0
            return clock["t"]

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "advice.csv"
            self._write_csv(src, 3)
            client = PendingClient([_extraction({}, [])])
            with self.assertRaises(ExtractionError):
                extract_document(
                    src,
                    client=client,
                    schema={},
                    dox_client_id="ai4u_payment_advice",
                    out_dir=Path(tmp) / "out",
                    poll_interval=0.0,
                    poll_timeout=10.0,
                    sleep=lambda _s: None,
                    now=fake_now,
                )

    def test_failed_job_raises(self) -> None:
        class FailingClient(FakeDoxClient):
            def get_job(self, job_id, *, extracted_values=None, return_null_values=None):
                return {"status": "FAILED"}

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "advice.csv"
            self._write_csv(src, 3)
            client = FailingClient([_extraction({}, [])])
            with self.assertRaises(ExtractionError):
                extract_document(
                    src,
                    client=client,
                    schema={},
                    dox_client_id="ai4u_payment_advice",
                    out_dir=Path(tmp) / "out",
                    sleep=lambda _s: None,
                )


if __name__ == "__main__":
    unittest.main()
