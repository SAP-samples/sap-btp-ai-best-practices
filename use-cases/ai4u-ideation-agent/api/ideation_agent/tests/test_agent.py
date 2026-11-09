from __future__ import annotations

import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, SystemMessage

# Set up fake modules to avoid importing real tools/company_search
fake_tools_module = types.ModuleType("ideation_agent.tools")
fake_company_search = types.ModuleType("ideation_agent.tools.company_search")


async def _fake_get_all_tools():
    return []


fake_tools_module.get_all_tools = _fake_get_all_tools
fake_company_search.cse_enabled = lambda: False
fake_perplexity_background = types.ModuleType("ideation_agent.perplexity_background")
fake_perplexity_background.get_result = lambda ctx: None
fake_perplexity_background.get_problem_result = lambda ctx: None
fake_perplexity_background.start_enrichment = AsyncMock()
fake_perplexity_background.start_problem_enrichment = AsyncMock()
sys.modules.setdefault("ideation_agent.tools", fake_tools_module)
sys.modules.setdefault("ideation_agent.tools.company_search", fake_company_search)
sys.modules.setdefault("ideation_agent.perplexity_background", fake_perplexity_background)

from ideation_agent.agent import _coerce_structured_turn
from ideation_agent.response_contract import (
    DEFAULT_CLARIFICATION_ERROR,
    DEFAULT_FINAL_ERROR,
)


class AgentStructuredTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_success_on_first_pass(self) -> None:
        llm = SimpleNamespace(ainvoke=AsyncMock())
        raw = json.dumps(
            {
                "kind": "final",
                "app_type": "Application",
                "industry": "Procurement",
                "title": "Material Classification Copilot",
                "problem": "Clerks classify material groups manually.",
                "solution": "Extract quotation data from PDFs, match items against the material group taxonomy, and present top suggestions for clerk approval in S4.",
            }
        )

        turn = await _coerce_structured_turn(
            llm=llm,
            raw_text=raw,
            state_messages=[],
            sys_msg=SystemMessage(content="sys"),
            context_id="ctx-1",
            clarification_count=1,
            remaining_questions=2,
        )

        self.assertEqual(turn["kind"], "final")
        self.assertEqual(turn["clarification_count"], 1)
        llm.ainvoke.assert_not_awaited()

    async def test_generic_repair_retry_can_succeed(self) -> None:
        repaired = json.dumps(
            {
                "kind": "final",
                "app_type": "Application",
                "industry": "Manufacturing",
                "title": "Quotation Copilot",
                "problem": "Teams classify quotes manually.",
                "solution": "Extract quote content from PDFs, rank material group matches using semantic similarity, and write approved classifications back to S4.",
            }
        )
        llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content=repaired))
        )

        turn = await _coerce_structured_turn(
            llm=llm,
            raw_text="not valid json",
            state_messages=[],
            sys_msg=SystemMessage(content="sys"),
            context_id="ctx-2",
            clarification_count=0,
            remaining_questions=3,
        )

        self.assertEqual(turn["kind"], "final")
        self.assertEqual(turn["normalization_source"], "repair")
        self.assertEqual(llm.ainvoke.await_count, 1)

    async def test_forced_clarification_repair_can_succeed(self) -> None:
        generic_invalid = json.dumps(
            {
                "kind": "clarification",
                "message": "One more thing.",
                "question": "Which region should launch first?",
                "suggested_answers": ["EMEA", "North America"],
            }
        )
        forced_valid = json.dumps(
            {
                "kind": "clarification",
                "message": "That helps. One final detail will improve the fit.",
                "question": "Which region should launch first?",
                "suggested_answers": ["EMEA", "North America", "APJ"],
            }
        )
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=[
                    AIMessage(content=generic_invalid),
                    AIMessage(content=forced_valid),
                ]
            )
        )

        turn = await _coerce_structured_turn(
            llm=llm,
            raw_text="Which region should launch first?",
            state_messages=[],
            sys_msg=SystemMessage(content="sys"),
            context_id="ctx-3",
            clarification_count=1,
            remaining_questions=2,
        )

        self.assertEqual(turn["kind"], "clarification")
        self.assertEqual(turn["clarification_count"], 2)
        self.assertEqual(turn["normalization_source"], "repair")
        self.assertEqual(turn["suggested_replies"], ["EMEA", "North America", "APJ"])
        self.assertEqual(llm.ainvoke.await_count, 2)

    async def test_unrecoverable_invalid_clarification_returns_error(self) -> None:
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=[
                    AIMessage(content="Still not JSON"),
                    AIMessage(content='{"kind":"clarification","message":"x","question":"Which region?","suggested_answers":["EMEA"]}'),
                ]
            )
        )

        turn = await _coerce_structured_turn(
            llm=llm,
            raw_text="Which region should launch first?",
            state_messages=[],
            sys_msg=SystemMessage(content="sys"),
            context_id="ctx-4",
            clarification_count=1,
            remaining_questions=2,
        )

        self.assertEqual(turn["kind"], "error")
        self.assertIn("**I need one more try.**", turn["display_text"])
        self.assertIn(DEFAULT_CLARIFICATION_ERROR, turn["display_text"])
        self.assertEqual(turn["clarification_count"], 1)
        self.assertEqual(llm.ainvoke.await_count, 2)

    async def test_unrecoverable_invalid_final_returns_error(self) -> None:
        raw = json.dumps(
            {
                "kind": "final",
                "industry": "Manufacturing",
                "title": "Broken Output",
                "problem": "Missing solution field should fail.",
            }
        )
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=AIMessage(
                    content='{"kind":"final","industry":"Manufacturing","title":"Broken Output","problem":"Still broken"}'
                )
            )
        )

        turn = await _coerce_structured_turn(
            llm=llm,
            raw_text=raw,
            state_messages=[],
            sys_msg=SystemMessage(content="sys"),
            context_id="ctx-5",
            clarification_count=3,
            remaining_questions=0,
        )

        self.assertEqual(turn["kind"], "error")
        self.assertIn("**I need one more try.**", turn["display_text"])
        self.assertIn(DEFAULT_FINAL_ERROR, turn["display_text"])
        self.assertEqual(turn["clarification_count"], 3)
        self.assertEqual(llm.ainvoke.await_count, 1)


if __name__ == "__main__":
    unittest.main()
