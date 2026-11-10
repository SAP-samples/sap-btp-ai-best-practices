"""
Unit tests for the UC-01 extraction service (offline, patched dependencies).

``extract_to_canonical`` is pure wiring over ``select_schema`` + ``pipeline.run``,
so the tests patch those two seams and assert the wiring: the client key is
normalized, a ready schema flows straight into the pipeline, and a not-ready
(proposed) schema with ``assume_yes=False`` is rejected instead of silently
extracting.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import extraction_service  # noqa: E402
from app.payment_advice.schema_select import SchemaSelection  # noqa: E402


def _settings() -> SimpleNamespace:
    return SimpleNamespace(dox_client_id="ai4u_payment_advice", mapper_model="gpt-5.6-luna")


def _fake_pipeline_result() -> SimpleNamespace:
    return SimpleNamespace(
        canonical={"header": {"payee_name": "Example Lighting"}, "line_items": [{"invoice_reference": "A"}]},
        raw_extraction={"schema_fields": [], "header": {}, "line_items": [{"invoice": "A", "region": "W"}]},
        verify=SimpleNamespace(confidence=0.91, needs_review=False, issues=[], warnings=["w"]),
    )


class ExtractToCanonical(unittest.TestCase):
    def test_ready_schema_runs_pipeline_and_shapes_result(self) -> None:
        ready = SchemaSelection(
            status="ready", schema_id="sid", schema_version="1", is_canonical=True, source="canonical"
        )
        with mock.patch.object(extraction_service, "select_schema", return_value=ready) as sel, \
             mock.patch("app.payment_advice.pipeline.run", return_value=_fake_pipeline_result()) as run:
            out = extraction_service.extract_to_canonical(
                "/tmp/advice.pdf", "Contoso",
                engine=object(), dox=object(), settings=_settings(),
            )

        # Client key normalized before both the schema call and the result.
        self.assertEqual(out["client_key"], "contoso")
        self.assertEqual(sel.call_args.kwargs["assume_yes"], True)
        # Pipeline received the normalized key + the ready selection.
        self.assertEqual(run.call_args.args[1], "contoso")
        self.assertIs(run.call_args.args[2], ready)
        # Result carries the canonical payload + a serializable verify summary.
        self.assertEqual(out["canonical"]["header"]["payee_name"], "Example Lighting")
        self.assertEqual(out["raw_extraction"]["line_items"], [{"invoice": "A", "region": "W"}])
        self.assertEqual(out["verify"], {
            "confidence": 0.91, "needs_review": False, "issues": [], "warnings": ["w"],
        })

    def test_proposed_schema_without_assume_yes_raises(self) -> None:
        proposed = SchemaSelection(status="proposed", source="dedicated")
        with mock.patch.object(extraction_service, "select_schema", return_value=proposed), \
             mock.patch("app.payment_advice.pipeline.run") as run:
            with self.assertRaises(ValueError):
                extraction_service.extract_to_canonical(
                    "/tmp/advice.pdf", "acme",
                    engine=object(), dox=object(), settings=_settings(), assume_yes=False,
                )
            run.assert_not_called()  # never extract on an unconfirmed schema


if __name__ == "__main__":
    unittest.main()
