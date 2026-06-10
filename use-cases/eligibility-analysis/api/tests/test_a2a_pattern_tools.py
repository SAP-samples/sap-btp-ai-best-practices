"""Tests for eligibility pattern analysis A2A tools."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.a2a.system_prompt import SYSTEM_PROMPT
from app.a2a.tools import eligibility_pattern_tools


class TestEligibilityPatternToolRegistry(unittest.TestCase):
    """Registry assertions for eligibility pattern tool exports."""

    def test_registry_contains_pattern_insights_tool(self) -> None:
        """Verify the combined pattern-insights tool is exposed."""
        names = [tool.name for tool in eligibility_pattern_tools.ELIGIBILITY_PATTERN_TOOLS]
        self.assertIn("get_eligibility_pattern_insights", names)


class TestEligibilityPatternInsightsTool(unittest.TestCase):
    """Tests for the combined pattern insights tool."""

    @patch("app.a2a.tools.eligibility_pattern_tools._get_analyzer")
    def test_default_args_return_all_sections(self, mock_get_analyzer: MagicMock) -> None:
        """Ensure defaults produce a full payload with all three sections."""
        analyzer = MagicMock()
        analyzer.analyze_all.return_value = {"total_patterns": 1, "patterns": [{"title": "x"}]}
        analyzer.get_debtor_profiles.return_value = [{"debtor_id": "D-1"}]
        analyzer.get_rejection_trend.return_value = [{"rejection_rate": 0.1}]
        mock_get_analyzer.return_value = analyzer

        result = eligibility_pattern_tools.get_eligibility_pattern_insights.invoke({})

        self.assertEqual(result["filters"]["lookback_days"], 180)
        self.assertIn("patterns", result)
        self.assertIn("debtor_profiles", result)
        self.assertIn("trend", result)
        self.assertEqual(result["filters"]["seller_id"], None)
        self.assertEqual(result["filters"]["debtor_id"], None)
        self.assertEqual(result["filters"]["programa"], None)
        self.assertEqual(result["filters"]["insurer_id"], None)

    @patch("app.a2a.tools.eligibility_pattern_tools._get_analyzer")
    def test_filters_and_min_invoices_are_forwarded(self, mock_get_analyzer: MagicMock) -> None:
        analyzer = MagicMock()
        analyzer.analyze_all.return_value = {"total_patterns": 0, "patterns": []}
        analyzer.get_debtor_profiles.return_value = []
        analyzer.get_rejection_trend.return_value = []
        mock_get_analyzer.return_value = analyzer

        result = eligibility_pattern_tools.get_eligibility_pattern_insights.invoke(
            {
                "seller_id": "S-123",
                "debtor_id": "D-456",
                "programa": "P1",
                "insurer_id": "I1",
                "lookback_days": 90,
            }
        )
        filters = analyzer.analyze_all.call_args.kwargs["filters"]

        self.assertEqual(result["filters"]["seller_id"], "S-123")
        self.assertEqual(filters.seller_id, "S-123")
        self.assertEqual(filters.debtor_id, "D-456")
        self.assertEqual(filters.programa, "P1")
        self.assertEqual(filters.insurer_id, "I1")
        self.assertEqual(filters.lookback_days, 90)
        self.assertEqual(filters.min_invoices, 3)

        analyzer.get_debtor_profiles.assert_called_once_with(filters=filters)
        analyzer.analyze_all.assert_called_once_with(filters=filters)

    @patch("app.a2a.tools.eligibility_pattern_tools._get_analyzer")
    def test_weekly_trend_granularity_used(self, mock_get_analyzer: MagicMock) -> None:
        analyzer = MagicMock()
        analyzer.analyze_all.return_value = {"total_patterns": 0, "patterns": []}
        analyzer.get_debtor_profiles.return_value = []
        analyzer.get_rejection_trend.return_value = []
        mock_get_analyzer.return_value = analyzer

        eligibility_pattern_tools.get_eligibility_pattern_insights.invoke({})
        filters = analyzer.analyze_all.call_args.kwargs["filters"]
        analyzer.get_rejection_trend.assert_called_once_with(granularity="week", filters=filters)


class TestSystemPromptGuideReferences(unittest.TestCase):
    """Prompt guidance coverage for new pattern-insights behavior."""

    def test_prompt_exposes_new_pattern_insights_tool(self) -> None:
        """Ensure the system prompt references the new tool and its use case."""
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("get_eligibility_pattern_insights", prompt_lower)
        self.assertIn("patterns across seller/debtor/program/insurer", prompt_lower)


if __name__ == "__main__":
    unittest.main()
