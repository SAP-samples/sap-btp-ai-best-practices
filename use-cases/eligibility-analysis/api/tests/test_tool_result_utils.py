"""Tests for assistant tool-result previews and the workspace-only tool registry.

Run from the repository root:
    cd api && PYTHONPATH=. ../.venv/bin/python -m unittest tests.test_tool_result_utils
"""

from __future__ import annotations

import asyncio
import unittest

from app.a2a.system_prompt import SYSTEM_PROMPT
from app.a2a.tool_result_utils import build_tool_result_preview
from app.a2a.tools import get_all_tools


class TestToolResultPreviewSafety(unittest.TestCase):
    """Large tool payloads are truncated before they reach the model context."""

    def test_preview_is_truncated_and_has_row_estimate(self) -> None:
        """Previews cap characters and still report the row count."""
        payload = {"rows": [{"id": i, "value": "x" * 40} for i in range(80)]}
        preview = build_tool_result_preview(payload, max_chars=120)
        self.assertTrue(preview["content_truncated"])
        self.assertEqual(preview["row_count_estimate"], 80)
        self.assertGreater(preview["payload_bytes_estimate"], 120)
        self.assertIn("[truncated]", preview["content_preview"])


class TestWorkspaceToolRegistry(unittest.TestCase):
    """The assistant binds only the read-only workspace tools."""

    def test_registry_and_prompt_are_workspace_only(self) -> None:
        """Every bound tool is described in the prompt and no legacy tool remains."""
        names = [tool.name for tool in asyncio.run(get_all_tools())]
        self.assertEqual(sorted(names), sorted([
            "get_workspace_overview",
            "get_workspace_invoice_rows",
            "get_workspace_pattern_insights",
            "inspect_saved_workspace",
            "list_saved_workspace_offers",
        ]))
        for name in names:
            self.assertIn(name, SYSTEM_PROMPT)
        for legacy in ("optimizer_process", "customer_logs", "get_seller_summary"):
            self.assertNotIn(legacy, SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
