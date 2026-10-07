"""Offline checks for the chat schema tools: plan registry, explicit-request guards, Advice-chat scope, real agent loop."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.deduction_agent.template_agent.config import AgentConfig
from app.deduction_agent.template_agent.mcp import MCPManager
from app.deduction_agent.template_agent.runtime import AgentRuntime
from app.deduction_agent.template_agent.skills import SkillLoader
from app.deduction_agent.tools import schema_tools as T
from app.payment_advice.schema_admin import CurrentSchema, SchemaPlan
from tests.unit.deduction_agent.test_runtime import RecordingStore

CURRENT = CurrentSchema("contoso", "schema-1", "2", [{"name": "document_no", "label": "Document No."}],
                        [{"name": "net", "label": "Net", "formattingType": "number"}])
SETTINGS = SimpleNamespace(dox_client_id="cid", mapper_model="model")


def build(request: list[str], bound_client=None):
    """Tools with a fake Document AI client; ``request[0]`` is the latest user message."""
    tools = T.build_schema_tools(None, lambda: object(), lambda: SETTINGS, lambda: request[0],
                                 lambda name: f"/tmp/{name}", bound_client=bound_client)
    return {t.name: t for t in tools}


class GuardTests(unittest.TestCase):
    """Writes need an explicit request naming the customer (Rules chat) and the version (publish)."""
    def setUp(self):
        for name, value in (("get_customer", SimpleNamespace(display_name="Contoso")),):
            patcher = mock.patch.object(T, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        current = mock.patch.object(T.A, "current_fields", return_value=CURRENT)
        current.start()
        self.addCleanup(current.stop)

    def plan(self, tools):
        """Plan adding payee_name and return the tool result."""
        return tools["plan_schema_changes"].invoke(
            {"client_key": "contoso", "add": [{"name": "payee_name", "scope": "header"}]})

    def test_plan_is_kept_by_id_and_prepare_needs_explicit_request(self):
        request = ["what fields are missing?"]
        tools = build(request)
        planned = self.plan(tools)
        self.assertEqual(planned["diff"]["added"][0]["name"], "payee_name")
        with mock.patch.object(T.A, "prepare") as prepare:
            with self.assertRaisesRegex(ValueError, "explicit request"):
                tools["prepare_schema_version"].invoke({"plan_id": planned["plan_id"]})
            request[0] = "yes, prepare it"
            with self.assertRaisesRegex(ValueError, "name the customer"):
                tools["prepare_schema_version"].invoke({"plan_id": planned["plan_id"]})
            request[0] = "yes, prepare it for Contoso"
            prepare.return_value = {"schema_id": "schema-1", "version": "3"}
            result = tools["prepare_schema_version"].invoke({"plan_id": planned["plan_id"]})
        self.assertEqual(result["version"], "3")
        plan = prepare.call_args.args[2]
        self.assertIsInstance(plan, SchemaPlan)
        self.assertEqual([f["name"] for f in plan.header], ["document_no", "payee_name"])

    def test_publish_needs_the_version_in_the_message(self):
        request = ["publish it for contoso"]
        tools = build(request)
        with mock.patch.object(T.A, "publish", return_value={"version": "3"}) as publish:
            with self.assertRaisesRegex(ValueError, "version 3"):
                tools["publish_schema_version"].invoke({"client_key": "contoso", "version": "3"})
            request[0] = "publish version 3 for contoso"
            tools["publish_schema_version"].invoke({"client_key": "contoso", "version": "3"})
        publish.assert_called_once()

    def test_unknown_plan_is_refused(self):
        tools = build(["prepare it for contoso"])
        with self.assertRaisesRegex(ValueError, "Unknown or expired plan"):
            tools["prepare_schema_version"].invoke({"plan_id": "missing"})


class AdviceScopeTests(unittest.TestCase):
    """An Advice chat can only work on its own customer's schema."""
    def test_other_customer_is_refused(self):
        tools = build(["show the schema"], bound_client="contoso")
        with mock.patch.object(T.A, "describe") as describe:
            with self.assertRaisesRegex(ValueError, "only work on the schema of 'contoso'"):
                tools["describe_extraction_schema"].invoke({"client_key": "acme"})
            tools["describe_extraction_schema"].invoke({})
        self.assertEqual(describe.call_args.args[2], "contoso")

    def test_publish_in_advice_chat_needs_no_customer_name(self):
        tools = build(["publish version 4"], bound_client="contoso")
        with mock.patch.object(T.A, "publish", return_value={}) as publish:
            tools["publish_schema_version"].invoke({"version": "4"})
        self.assertEqual(publish.call_args.args[2:4], ("contoso", "4"))


class PlanThenAnswerModel(BaseChatModel):
    """Calls plan_schema_changes once, then answers with the tool result it observed."""

    @property
    def _llm_type(self) -> str:
        """Fake model identifier."""
        return "schema-tools-test-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "PlanThenAnswerModel":
        """Keep the normal bind-tools interface."""
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        """First a tool call with list-of-dict arguments, then a final answer."""
        if isinstance(messages[-1], ToolMessage):
            answer = AIMessage(content="planned: " + messages[-1].content)
        else:
            answer = AIMessage(content="", tool_calls=[{
                "name": "plan_schema_changes", "id": "call-1", "type": "tool_call",
                "args": {"client_key": "contoso", "add": [{"name": "payee_name", "scope": "header", "type": "string"}]}}])
        return ChatResult(generations=[ChatGeneration(message=answer)])


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    """The tools run inside the real AgentRuntime graph (ToolNode argument parsing)."""
    async def test_agent_plans_through_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(T.A, "current_fields", return_value=CURRENT):
            config = AgentConfig.model_validate({"base_prompt": "Base", "model": {"provider": "openai", "name": "fake"},
                                                 "skills": {"directory": tmp}, "memory": {"enabled": False}})
            runtime = AgentRuntime(config, PlanThenAnswerModel(), SkillLoader(Path(tmp)), MCPManager(), RecordingStore(),
                                   extra_tools=list(build(["add payee_name"]).values()))
            result = await runtime.ainvoke("add payee_name to contoso", "ctx")
        self.assertIn("plan_id", result.output_text)
        self.assertIn("payee_name", result.output_text)


if __name__ == "__main__":
    unittest.main()
