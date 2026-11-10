"""
Unit test for the pipeline wiring on the canonical path (offline, fake client).

The canonical path needs no LLM, so it is fully testable with a fake Document AI
client. The custom (mapped) path is covered by the mapper tests and the live
end-to-end run.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import pipeline  # noqa: E402
from app.payment_advice.schema_select import SchemaSelection  # noqa: E402


def _fields(mapping):
    return [{"name": k, "value": v} for k, v in mapping.items()]


class _FakeDox:
    def __init__(self, extraction):
        self._extraction = extraction

    def upload_document_stream(self, file_name, file_obj, mime_type, **options):
        return {"id": "job0", "status": "PENDING"}

    def get_job(self, job_id, *, extracted_values=None, return_null_values=None):
        return {"status": "DONE", "extraction": self._extraction}


class PipelineCanonicalPath(unittest.TestCase):
    def test_run_writes_canonical_and_run_log(self) -> None:
        extraction = {
            "headerFields": _fields({"payee_name": "Example Lighting", "payment_reference": "R1", "payment_amount": "100"}),
            "lineItems": [_fields({"invoice_reference": "A", "net_amount": "60"}),
                          _fields({"invoice_reference": "B", "net_amount": "40"})],
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "advice.csv"
            with src.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([["h"], ["x"]])  # content is irrelevant; fake client returns extraction

            settings = SimpleNamespace(
                dox_client_id="ai4u_payment_advice", max_pages=100, max_line_items=2000,
                max_columns=49, out_dir=str(Path(tmp) / "out"), mapper_model="gpt-5.6-luna",
            )
            selection = SchemaSelection(
                status="ready", schema_id="sid", schema_version="1", is_canonical=True, source="canonical",
            )
            result = pipeline.run(
                src, "contoso", selection, dox_client=_FakeDox(extraction), settings=settings,
            )

            self.assertTrue(result.canonical_path.is_file())
            self.assertTrue(result.run_log_path.is_file())
            self.assertIsNone(result.mapping)  # canonical path skips mapping
            self.assertEqual(result.canonical["header"]["payment_amount"], 100)  # normalized to number
            self.assertEqual(len(result.canonical["line_items"]), 2)
            self.assertFalse(result.verify.needs_review)

            written = json.loads(result.canonical_path.read_text())
            self.assertEqual(written["header"]["payee_name"], "Example Lighting")
            run_log = json.loads(result.run_log_path.read_text())
            self.assertEqual(run_log["schema"]["source"], "canonical")
            self.assertEqual(run_log["line_item_count"], 2)



class SchemaFieldLabels(unittest.TestCase):
    """Field labels (printed column headings) are read once per schema version."""

    def test_labels_are_read_and_cached_and_failures_are_tolerated(self) -> None:
        calls = []

        class Dox:
            def get_schema_version_details(self, schema_id, version, client_id="default"):
                calls.append((schema_id, version, client_id))
                if schema_id == "broken":
                    raise RuntimeError("unavailable")
                return {"headerFields": [{"name": "document_no", "label": "Document No.", "formattingType": "string"}],
                        "lineItemFields": [{"name": "region", "label": "Rg", "formattingType": "country/region"}]}

        fields = pipeline.schema_fields(Dox(), "s1", "1", "client")
        self.assertEqual(fields, [
            {"name": "document_no", "label": "Document No.", "type": "string", "scope": "header"},
            {"name": "region", "label": "Rg", "type": "country/region", "scope": "line"}])
        pipeline.schema_fields(Dox(), "s1", "1", "client")
        self.assertEqual(len(calls), 1)
        self.assertEqual(pipeline.schema_fields(Dox(), "broken", "1", "client"), [])


if __name__ == "__main__":
    unittest.main()
