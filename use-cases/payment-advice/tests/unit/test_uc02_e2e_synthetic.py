"""End-to-end synthetic-playbook tests for UC-02 deduction interpretation.

These tests validate the core deduction pipeline WITHOUT any live LLM or network
calls. The Fabrikam test deterministically exercises the selective-analysis flow:
select_analysis_lines identifies the anomalous subset, the simulated agent entries
are tagged with line_index, assemble_interpretations merges them back into the full
index-aligned list, and enrich_payload produces the enriched output.

The playbook loaded here is explicitly marked "synthetic": true and uses
placeholder SAP reason codes — it is test scaffolding, not production rules.

Example run:
    PYTHONPATH=api python -m unittest tests.unit.test_uc02_e2e_synthetic -v
"""
import asyncio
import json
import os
import types
import unittest

from app.payment_advice.deductions import (
    assemble_interpretations,
    enrich_payload,
    select_analysis_lines,
)

# Base fixture directory, relative to repo root (tests are run from repo root)
FIX = os.path.join("tests", "unit", "fixtures")


class FabrikamSyntheticTest(unittest.TestCase):
    """Validate Fabrikam bracket-code mapping using the selective-analysis flow.

    Strategy (offline, deterministic):
    1. Load the real Fabrikam canonical fixture (extracted by the pipeline; not hand-edited).
    2. Load the synthetic playbook, which maps bracket codes to placeholder reason codes.
    3. Call select_analysis_lines to identify only the lines that require agent analysis
       (negatives / lines with a deduction_reason), each tagged with its line_index.
    4. Simulate agent entries for the analysis subset: parse the bracket code from
       deduction_reason and build an interpretation dict tagged with line_index for any
       line whose code appears in the playbook's anchors.codes map.
    5. Reassemble the full index-aligned list via assemble_interpretations.
    6. Call enrich_payload to produce the enriched output.
    7. Assert:
       a. At least one analyzed line received a non-default reason_code (not None, not "323"),
          proving the bracket-code path works end to end.
       b. A known plain positive invoice line (line index 2, gross=5298.52, no deduction_reason)
          has document_nature="invoice", reason_code=None, and empty flags — proving that plain
          lines pass through assemble_interpretations cleanly.
    """

    @classmethod
    def setUpClass(cls):
        """Load fixtures once for all tests in this class."""
        with open(os.path.join(FIX, "fabrikam_canonical.json"), encoding="utf-8") as fh:
            cls.payload = json.load(fh)
        with open(os.path.join(FIX, "playbooks", "fabrikam_synthetic.json"), encoding="utf-8") as fh:
            cls.playbook = json.load(fh)

    def test_fabrikam_codes_map_to_reason_codes(self):
        """Selective-analysis flow: analyzed lines map bracket codes; plain lines stay clean.

        The Fabrikam fixture mixes plain positive invoices with negative deduction lines.
        This test mirrors the route/CLI selective-analysis flow:
        - select_analysis_lines picks only the deduction lines.
        - The simulated agent maps each bracket code to a reason_code and tags the entry
          with the line's original line_index.
        - assemble_interpretations merges agent entries back into a full index-aligned list,
          defaulting plain lines to invoice/None/[] and unmatched analysis lines to 323.
        - enrich_payload applies the list to produce the final enriched payload.

        Assertion (a): real bracket-code-to-reason-code mapping is non-vacuous (at least one
        analysis line received a reason_code from the playbook, not the "323" catch-all).
        Assertion (b): line index 2 (gross=5298.52, no deduction_reason) is a plain invoice
        and must have document_nature="invoice", reason_code=None, and flags=[].
        """
        # anchors.codes maps bracket code strings to placeholder SAP reason codes
        # e.g. {"0059": "R0059", "0100": "R0100", "0057": "R0057", "0022": "R0022"}
        codes = self.playbook["anchors"]["codes"]

        # Step 1: identify the anomalous subset (negatives + deduction_reason lines).
        # Each returned dict is a shallow copy of the line augmented with line_index.
        analysis = select_analysis_lines(self.payload)

        # Step 2: build agent_entries for the analysis subset only.
        # Each entry is tagged with line_index so assemble_interpretations can merge it
        # back to the correct position in the full line_items list.
        # Lines whose bracket code is not in the playbook are omitted from agent_entries;
        # assemble_interpretations will default them to the 323 escalation path.
        agent_entries = []
        for line in analysis:
            reason = line.get("deduction_reason", "")
            # Bracket codes appear at the end of the description: "DESCRIPTION [0059]"
            code = reason.split("[")[-1].strip("]") if "[" in reason else None
            if code in codes:
                agent_entries.append({
                    "line_index": line["line_index"],
                    "document_nature": "chargeback",
                    "reason_code": codes[code],
                })

        # Step 3: reassemble the full index-aligned list.
        # Plain lines get invoice/None/[]; analysis lines with agent_entries get the
        # mapped interpretation; analysis lines with no matching entry get 323.
        full = assemble_interpretations(self.payload, agent_entries)

        # Step 4: enrich the payload.
        out = enrich_payload(
            self.payload,
            full,
            client_key="fabrikam",
            playbook_revision=self.playbook.get("revision", 1),
        )

        # Assertion (a): at least one analysis line received a real reason_code.
        # The fixture contains lines with codes 0059, 0057, 0100, and 0022, all in
        # the playbook, so this must be non-vacuous.
        mapped = [
            line for line in out["line_items"]
            if line.get("reason_code") not in (None, "323")
        ]
        self.assertTrue(
            mapped,
            msg=(
                "No Fabrikam line received a mapped reason_code. "
                "Codes found in playbook: {}. "
                "Check that the bracket-code parsing matches the actual fixture data."
            ).format(list(codes.keys())),
        )

        # Assertion (b): line index 2 is a plain positive invoice (gross=5298.52,
        # no deduction_reason). It must not be touched by the agent path at all.
        plain_line = out["line_items"][2]
        self.assertEqual(
            plain_line.get("document_nature"),
            "invoice",
            msg="Plain positive line (index 2, gross=5298.52) must have document_nature='invoice'",
        )
        self.assertIsNone(
            plain_line.get("reason_code"),
            msg="Plain positive line (index 2, gross=5298.52) must have reason_code=None",
        )
        self.assertEqual(
            plain_line.get("flags"),
            [],
            msg="Plain positive line (index 2, gross=5298.52) must have empty flags",
        )


class NdjsonStreamNoRuntimeTest(unittest.TestCase):
    """Guard the no-runtime + clean-payload contract for _ndjson_stream.

    A payload that contains ONLY plain positive invoice lines (no negatives, no
    deduction_reason) should produce the full three-envelope response even when
    ``request.app.state.runtime`` is None, because the LLM is never needed for
    plain lines.  Before the runtime-guard refactor the function returned a single
    error envelope in this case, which was incorrect.
    """

    def _make_request(self, runtime=None, engine=None):
        """Build a minimal fake FastAPI Request with controlled app.state."""
        state = types.SimpleNamespace(runtime=runtime, engine=engine)
        app = types.SimpleNamespace(state=state)
        return types.SimpleNamespace(app=app)

    def _collect_stream(self, client, payload, request):
        """Drive the async generator synchronously and return parsed envelopes."""
        from app.routers.payment_advice import _ndjson_stream

        async def _run():
            lines = []
            async for chunk in _ndjson_stream(client, payload, request):
                lines.append(json.loads(chunk.strip()))
            return lines

        return asyncio.run(_run())

    def test_clean_payload_no_runtime_streams_result(self):
        """A clean payload must emit analysis, tools, result — never error — with runtime=None.

        This directly tests the fix that moved the runtime=None guard inside the
        anomalous branch: plain-only payloads must not require a runtime.
        """
        # Construct a minimal payload with two plain positive lines only.
        payload = {
            "header": {},
            "line_items": [
                {"invoice_reference": "INV001", "gross_amount": 100.0, "net_amount": 98.0},
                {"invoice_reference": "INV002", "gross_amount": 200.0, "net_amount": 196.0},
            ],
        }
        request = self._make_request(runtime=None, engine=None)
        envelopes = self._collect_stream("fabrikam", payload, request)

        types_seen = [e["type"] for e in envelopes]

        # Must NOT emit an error envelope.
        self.assertNotIn(
            "error", types_seen,
            msg="Clean payload with runtime=None must not produce an error envelope.",
        )

        # Must emit the three expected envelopes in order.
        self.assertEqual(
            types_seen,
            ["analysis", "tools", "result"],
            msg="Clean payload must yield analysis → tools → result envelopes.",
        )

        # analysis envelope: no lines selected, both lines passed through.
        analysis_env = envelopes[0]
        self.assertEqual(analysis_env["analyzed_refs"], [])
        self.assertEqual(analysis_env["passthrough_count"], 2)

        # result envelope: both lines must be plain invoices with no flags.
        result_env = envelopes[2]
        for line in result_env["enriched"]["line_items"]:
            self.assertEqual(line.get("document_nature"), "invoice")
            self.assertIsNone(line.get("reason_code"))
            self.assertEqual(line.get("flags"), [])


if __name__ == "__main__":
    unittest.main()
