"""Tests for read-only optimizer A2A tools and prompt guardrails."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.a2a.system_prompt import SYSTEM_PROMPT
from app.a2a.tool_result_utils import build_tool_result_preview
from app.a2a.tools import optimizer_tools


class TestOptimizerToolRegistry(unittest.TestCase):
    def test_registry_contains_new_tools_only(self) -> None:
        names = [tool.name for tool in optimizer_tools.OPTIMIZER_TOOLS]
        self.assertEqual(
            names,
            [
                "list_optimizer_processes",
                "resolve_optimizer_process_id",
                "get_optimizer_limits",
                "get_optimizer_reason_legend",
                "get_optimizer_overview",
                "get_optimizer_exclusion_summary",
                "get_optimizer_utilization_summary",
                "get_optimizer_weekly_schedule_summary",
                "get_optimizer_invoice_decision",
                "get_optimizer_invoice_rows",
                "get_optimizer_weekly_exposure_rows",
            ],
        )
        self.assertNotIn("update_optimizer_limits", names)
        self.assertNotIn("get_optimizer_results", names)
        self.assertNotIn("get_optimizer_weekly_exposure", names)


class TestOptimizerTools(unittest.TestCase):
    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_resolve_optimizer_process_id_unique_prefix(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.resolve_process_id.return_value = {
            "match_type": "prefix_unique",
            "process_id": "8ac26cb4-8d16-4854-90d2-bdd3225d2e5f",
            "matches": [
                {
                    "process_id": "8ac26cb4-8d16-4854-90d2-bdd3225d2e5f",
                    "status": "completed",
                    "cohort": "2025-01-28",
                    "created_at": "2026-02-16T10:00:00",
                }
            ],
            "scanned": 20,
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.resolve_optimizer_process_id.invoke({"process_ref": "8ac26..."})
        self.assertEqual(result["summary"]["match_type"], "prefix_unique")
        self.assertEqual(result["summary"]["resolved_process_id"], "8ac26cb4-8d16-4854-90d2-bdd3225d2e5f")
        self.assertFalse(result["truncated"])

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_get_optimizer_overview_compact(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_chat_context.side_effect = AssertionError("get_chat_context should not be used")
        manager.get_overview_summary.return_value = {
            "cohort": "2025-01-28",
            "planning_mode": "multi_week",
            "source_profile": "offer_file",
            "horizon_weeks": 8,
            "kpis": {"optimizer_status": "OPTIMAL", "selected_amount_ratio_pct": 91.2},
            "deferred_reasons": {"CUSTOMER_CAP_BINDING": 10},
            "binding_constraints": [{"entity_type": "facility", "entity_id": "C001"}],
            "top_customers": [{"customer": "CUST1", "selected_amount": 1000.0, "share_pct": 25.0}],
            "weekly_schedule_summary": [{"week_start": "2025-01-28", "invoice_count": 3, "total_amount": 1000.0}],
            "row_counts": {"selected": 100, "excluded": 50, "weekly_plan": 100, "weekly_exposure": 500},
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_overview.invoke({"process_id": "proc-1"})
        self.assertIn("kpis", result["data"])
        self.assertNotIn("weekly_plan", result["data"])
        self.assertNotIn("weekly_exposure", result["data"])
        self.assertIn("binding_constraints_top", result["data"])

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_weekly_exposure_rows_pagination(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_weekly_exposure_rows.return_value = {
            "rows": [{"week_start": "2025-01-28", "entity_type": "facility", "entity_id": "C001"}] * 50,
            "total": 120,
            "limit": 50,
            "offset": 0,
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_weekly_exposure_rows.invoke({"process_id": "proc-1"})
        self.assertEqual(result["pagination"]["returned"], 50)
        self.assertTrue(result["pagination"]["has_more"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["next_offset"], 50)

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_invoice_rows_forwards_filters_and_pagination(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_invoice_rows.return_value = {
            "rows": [
                {
                    "invoice_ref": "INV-001",
                    "customer": "CUST-A",
                    "excluded_reason": "CUSTOMER_CAP_BINDING",
                }
            ],
            "total": 3,
            "limit": 1,
            "offset": 1,
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_invoice_rows.invoke(
            {
                "process_id": "proc-1",
                "bucket": "excluded",
                "customer": "CUST-A",
                "limit": 1,
                "offset": 1,
            }
        )
        self.assertEqual(result["summary"]["filters"]["customer"], "CUST-A")
        self.assertEqual(result["pagination"]["total"], 3)
        self.assertEqual(result["pagination"]["offset"], 1)
        self.assertEqual(result["pagination"]["returned"], 1)

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_invoice_decision_handles_multiple_matches(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.find_invoice_decisions.return_value = {
            "matches": [
                {"invoice_ref": "INV-001", "decision": "excluded"},
                {"invoice_ref": "INV-001-A", "decision": "selected"},
            ],
            "total": 3,
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_invoice_decision.invoke(
            {
                "process_id": "proc-1",
                "invoice_reference": "INV-001",
                "max_matches": 2,
            }
        )
        self.assertEqual(result["summary"]["matches_found"], 2)
        self.assertTrue(result["pagination"]["has_more"])

    def test_reason_legend_known_code(self) -> None:
        result = optimizer_tools.get_optimizer_reason_legend.invoke({"reason": "EXPIRED_WINDOW"})
        self.assertEqual(result["summary"]["returned"], 1)
        self.assertEqual(result["data"][0]["reason"], "EXPIRED_WINDOW")
        self.assertTrue(result["data"][0]["known"])
        self.assertEqual(result["data"][0]["stage"], "optimizer")
        self.assertIn("feasible planning week", result["data"][0]["meaning"])

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_exclusion_summary_includes_reason_definitions(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_chat_context.side_effect = AssertionError("get_chat_context should not be used")
        manager.get_exclusions_summary.return_value = {
            "rows": [
                {"reason": "planning_window_offer_file_date", "count": 10, "stage": "rule"},
                {"reason": "EXPIRED_WINDOW", "count": 5, "stage": "optimizer"},
            ]
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_exclusion_summary.invoke({"process_id": "proc-1"})
        defs = result["summary"]["reason_definitions"]
        self.assertEqual(len(defs), 2)
        self.assertIn("planning_window_offer_file_date", {row["reason"] for row in defs})
        self.assertIn("EXPIRED_WINDOW", {row["reason"] for row in defs})

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_utilization_summary_reads_summary_section(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_chat_context.side_effect = AssertionError("get_chat_context should not be used")
        manager.get_utilization_summary.return_value = {
            "available_entity_types": ["customer"],
            "rows": {
                "customer": [
                    {
                        "week_start": "2025-01-28",
                        "entity_type": "customer",
                        "entity_id": "CUST-A",
                        "used_total": 91.0,
                        "limit": 100.0,
                        "utilization_pct": 91.0,
                    },
                    {
                        "week_start": "2025-02-04",
                        "entity_type": "customer",
                        "entity_id": "CUST-A",
                        "used_total": 75.0,
                        "limit": 100.0,
                        "utilization_pct": 75.0,
                    },
                ]
            },
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_utilization_summary.invoke(
            {"process_id": "proc-1", "entity_type": "customer", "view": "peak"}
        )
        self.assertEqual(result["summary"]["entity_type"], "customer")
        self.assertEqual(result["summary"]["returned_rows"], 1)
        self.assertEqual(result["data"][0]["entity_id"], "CUST-A")

    @patch("app.a2a.tools.optimizer_tools._get_manager")
    def test_weekly_schedule_summary_reads_summary_section(self, mock_get_manager: MagicMock) -> None:
        manager = MagicMock()
        manager.get_chat_context.side_effect = AssertionError("get_chat_context should not be used")
        manager.get_schedule_summary.return_value = {
            "planning_mode": "multi_week",
            "rows": [
                {"week_start": "2025-01-28", "week_index": 1, "invoice_count": 3, "total_amount": 100.0},
                {"week_start": "2025-02-04", "week_index": 2, "invoice_count": 4, "total_amount": 150.0},
            ],
        }
        mock_get_manager.return_value = manager

        result = optimizer_tools.get_optimizer_weekly_schedule_summary.invoke({"process_id": "proc-1"})
        self.assertEqual(result["summary"]["planning_mode"], "multi_week")
        self.assertEqual(result["summary"]["weeks_available"], 2)
        self.assertEqual(result["summary"]["invoice_count_total"], 7)


class TestSystemPromptGuardrails(unittest.TestCase):
    def test_prompt_contains_optimizer_read_only_guidance(self) -> None:
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("optimizer chat is read-only", prompt_lower)
        self.assertIn("resolve_optimizer_process_id", SYSTEM_PROMPT)
        self.assertIn("get_optimizer_overview", SYSTEM_PROMPT)
        self.assertIn("get_optimizer_invoice_rows", SYSTEM_PROMPT)
        self.assertIn("get_optimizer_reason_legend", SYSTEM_PROMPT)
        self.assertNotIn("update_optimizer_limits", SYSTEM_PROMPT)
        self.assertNotIn("get_optimizer_results", SYSTEM_PROMPT)


class TestToolResultPreviewSafety(unittest.TestCase):
    def test_preview_is_truncated_and_has_row_estimate(self) -> None:
        payload = {"rows": [{"id": i, "value": "x" * 40} for i in range(80)]}
        preview = build_tool_result_preview(payload, max_chars=120)
        self.assertTrue(preview["content_truncated"])
        self.assertEqual(preview["row_count_estimate"], 80)
        self.assertGreater(preview["payload_bytes_estimate"], 120)
        self.assertIn("[truncated]", preview["content_preview"])


if __name__ == "__main__":
    unittest.main()
