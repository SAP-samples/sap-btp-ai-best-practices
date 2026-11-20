"""Agent wiring test: a scripted chat model drives the real runtime and the order tools.

No network and no HANA: the submission service is patched, so the test checks that
* the runtime binds the order tools next to load_skill,
* submit_order receives the order id from the graph state (not from the model),
* the nested field-map arguments are parsed into an OrderSubmission.
"""

import asyncio
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent_core import AgentRuntime
from app.agent_core.config import load_config
from app.agent_core.mcp import MCPManager
from app.agent_core.memory import NullConversationStore
from app.agent_core.skills import SkillLoader
from app.order_agent import tools as order_tools
from app.order_agent.runner import CONFIG_PATH

SUBMIT_ARGS = {
    "classification": "STANDARD",
    "classification_reason": "Customer purchase order with one line.",
    "header": {"po_number": {"value": "PO-77", "raw": "PO-77", "confidence": 0.97, "source": "A1 p.1"}},
    "lines": [{"line_type": "MATERIAL", "fields": {
        "material": {"value": "7000001", "raw": "XG1421/1GL", "confidence": 0.9},
        "quantity": {"value": 6, "raw": "6 Gallon", "confidence": 0.95, "note": "6 gallons = 6 cans"}}}],
    "review_flags": [{"line": 1, "field": "uom", "message": "unit column says gallon"}],
}


class ScriptedModel(BaseChatModel):
    """Calls submit_order once (without any order id), then answers."""

    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.calls.append(sorted(t.name for t in tools))
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if isinstance(messages[-1], ToolMessage):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Order PO-77 submitted."))])
        call = {"name": "submit_order", "args": SUBMIT_ARGS, "id": "call-1", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


class OrderAgentWiringTests(unittest.TestCase):
    """Runtime + tools integration without external services."""

    def test_submit_order_gets_injected_order_id(self):
        config = load_config(CONFIG_PATH)
        model = ScriptedModel()
        runtime = AgentRuntime(config, model, SkillLoader(config.skills.directory, 120000), MCPManager(),
                               NullConversationStore(), extra_tools=order_tools.build_order_tools())
        captured = {}

        def fake_apply(order_id, submission):
            captured["order_id"], captured["submission"] = order_id, submission
            return {"saved": True, "issues": []}

        with patch.object(order_tools, "apply_agent_submission", side_effect=fake_apply):
            result = asyncio.run(runtime.ainvoke("process order", context_id="order-123"))

        self.assertEqual(result.output_text, "Order PO-77 submitted.")
        self.assertEqual(model.calls[0], ["convert_quantities", "get_customer_profile", "load_skill",
                                          "search_customers", "search_materials", "submit_order"])
        self.assertEqual(captured["order_id"], "order-123")
        submission = captured["submission"]
        self.assertEqual(submission.header["po_number"].value, "PO-77")
        self.assertEqual(submission.lines[0].fields["quantity"].note, "6 gallons = 6 cans")
        self.assertEqual(submission.review_flags[0].line, 1)

    def test_config_matches_requirements(self):
        config = load_config(CONFIG_PATH)
        self.assertEqual((config.model.name, config.model.reasoning_effort), ("gpt-6-luna", "high"))
        self.assertIsNone(config.model.temperature)
        self.assertIsNone(config.model.max_tokens)
        self.assertFalse(config.summarization.enabled)
        names = {s.name for s in SkillLoader(Path(config.skills.directory), 120000).scan()}
        self.assertEqual(names, {"order-classification", "header-extraction", "line-item-extraction",
                                 "master-data-matching", "confidence-and-fixes"})


if __name__ == "__main__":
    unittest.main()
