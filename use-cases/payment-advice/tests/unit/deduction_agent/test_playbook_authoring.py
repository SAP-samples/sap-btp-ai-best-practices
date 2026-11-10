"""Tests for HANA-compatible deduction-skill generation and golden comparison.

Run with::

    PYTHONPATH=api .venv/bin/python -m unittest \
        tests.unit.deduction_agent.test_playbook_authoring -v
"""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from app.deduction_agent.playbook_authoring import (
    PLAYBOOK_SCHEMA,
    compare_playbooks,
    generate_playbook,
)
from app.deduction_agent.rule_sources import RuleSourceError, read_rule_source


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "playbooks"
    / "contoso_golden.json"
)


def _write_minimal_docx(path: Path) -> None:
    """Write a valid one-paragraph DOCX used by the generation orchestration test."""

    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>Use reason code 323 by default.</w:t></w:r></w:p></w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


class _SourceAwareRuntime:
    """Return a fixed candidate only after observing the source bound for this run."""

    def __init__(self, candidate: dict) -> None:
        """Store the candidate and initialize the invocation capture."""

        self.candidate = candidate
        self.calls: list[dict] = []

    async def ainvoke(self, **kwargs):
        """Capture the invocation and prove the source context is visible while awaiting."""

        source_text = read_rule_source("rules.docx")["content"]
        self.calls.append({**kwargs, "source_text": source_text})
        return SimpleNamespace(
            output_parsed=self.candidate,
            output_text="Generated candidate",
            messages=[],
        )


class PlaybookAuthoringTests(unittest.TestCase):
    """Verify generation boundaries and semantic comparison behavior."""

    def setUp(self) -> None:
        """Load the independently authored Contoso golden playbook."""

        self.golden = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_structured_output_schema_has_a_provider_function_name(self) -> None:
        """Removing the top-level title must reproduce the provider adapter failure."""

        self.assertEqual(PLAYBOOK_SCHEMA["title"], "DeductionSkillCandidate")

    def test_compare_playbooks_detects_missing_and_unexpected_business_rules(self) -> None:
        """Dropping or hallucinating a business rule must make comparison fail."""

        exact = compare_playbooks(self.golden, self.golden)
        self.assertTrue(exact["matched"])
        self.assertEqual(exact["missing"], {})
        self.assertEqual(exact["unexpected"], {})

        changed = copy.deepcopy(self.golden)
        changed["anchors"]["deduction_rules"].pop()
        changed["anchors"]["deduction_rules"].append(
            {
                "priority": 99,
                "field": "invoice_number",
                "operator": "starts_with",
                "value": "MADE-UP",
                "reason_code": "999",
                "action": "assign",
                "description": "hallucinated",
            }
        )
        comparison = compare_playbooks(changed, self.golden)
        self.assertFalse(comparison["matched"])
        self.assertEqual(len(comparison["missing"]["deduction_rules"]), 1)
        self.assertEqual(len(comparison["unexpected"]["deduction_rules"]), 1)

    def test_compare_playbooks_normalizes_equivalent_business_field_labels(self) -> None:
        """Cosmetic source labels must not cause a false semantic mismatch."""

        equivalent = copy.deepcopy(self.golden)
        for rule in equivalent["anchors"]["customer_account_rules"]:
            rule["field"] = "Column Rg"
        replacements = {
            "advice_number": ("Advice no.", 'Concatenate "CONTOSO" & current date', "concatenate"),
            "payer": ("ILN sender", "10053628", "constant"),
            "payee": ("ILN recipient", "CA01", "constant"),
            "payment_currency": ("Payment currency", "CAD", "constant"),
            "invoice_currency": ("Inv. currency", "CAD", "constant"),
            "invoice_date": ("Inv. date", "Invoice Date", "column value"),
            "invoice_reference": ("Item characteristic, inv. number", "Your Invoice No", "column value"),
            "market_name": ("Market name", "Column Rg customer account mapping", "map"),
        }
        for mapping in equivalent["anchors"]["field_mappings"]:
            replacement = replacements.get(mapping["target_field"])
            if replacement:
                (
                    mapping["target_field"],
                    mapping["source_field"],
                    mapping["transformation"],
                ) = replacement

        comparison = compare_playbooks(equivalent, self.golden)
        self.assertTrue(comparison["matched"], comparison)

    def test_generate_playbook_binds_sources_and_resets_them_after_invocation(self) -> None:
        """Removing source binding or its finally-reset must break this test."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rules.docx"
            _write_minimal_docx(source)
            runtime = _SourceAwareRuntime(self.golden)

            candidate, report = asyncio.run(
                generate_playbook(
                    runtime=runtime,
                    client="CA01 Contoso",
                    source_paths=[source],
                    golden=self.golden,
                )
            )

            self.assertEqual(candidate["client_key"], "ca01_contoso")
            self.assertEqual(candidate["playbook_text"], self.golden["playbook_text"])
            self.assertTrue(report["comparison"]["matched"])
            self.assertEqual(report["sources"], ["rules.docx"])
            self.assertIn("Use reason code 323 by default.", runtime.calls[0]["source_text"])
            self.assertEqual(runtime.calls[0]["response_model"].get("type"), "object")

            with self.assertRaisesRegex(RuleSourceError, "No rule sources"):
                read_rule_source("rules.docx")

    def test_generate_playbook_rejects_a_malformed_agent_result(self) -> None:
        """Letting an incomplete model response reach HANA must break this test."""

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "rules.docx"
            _write_minimal_docx(source)
            runtime = _SourceAwareRuntime({"playbook_text": "missing anchors"})
            with self.assertRaisesRegex(ValueError, "anchors"):
                asyncio.run(
                    generate_playbook(
                        runtime=runtime,
                        client="CA01 Contoso",
                        source_paths=[source],
                    )
                )


if __name__ == "__main__":
    unittest.main()
