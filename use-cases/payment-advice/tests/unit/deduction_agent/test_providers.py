"""Verify provider-factory compatibility details without live model calls.

Converted from the scaffold's pytest-based test to stdlib unittest so it runs
under ``unittest discover`` without requiring pytest.  The ``monkeypatch``
fixture is replaced by ``unittest.mock.patch``.
"""

import unittest
from unittest.mock import patch

from gen_ai_hub.proxy.langchain import ChatBedrockConverse

from app.deduction_agent.template_agent import providers
from app.deduction_agent.template_agent.config import ModelSettings
from app.deduction_agent.template_agent.providers import ResponsesCompatibleChatOpenAI


class TestProviders(unittest.TestCase):
    """Test SAP Gen AI Hub provider factory compatibility."""

    def _openai_kwargs(self, use_responses_api: bool) -> dict:
        """Build an OpenAI model with the class mocked and return its constructor kwargs."""
        settings = ModelSettings(provider="openai", name="gpt-5.4", reasoning_effort="low",
                                 use_responses_api=use_responses_api)
        with patch.object(providers, "ResponsesCompatibleChatOpenAI") as model_cls:
            providers.create_chat_model(settings)
        return model_cls.call_args.kwargs

    def test_responses_calls_are_stateless_with_inline_reasoning(self) -> None:
        """Responses models must not reference stored items (AI Core answers 404 on those)."""
        kwargs = self._openai_kwargs(use_responses_api=True)
        self.assertIs(kwargs["store"], False)
        self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_chat_completions_calls_get_no_responses_only_settings(self) -> None:
        """Chat Completions does not accept ``include``; leave both settings out."""
        kwargs = self._openai_kwargs(use_responses_api=False)
        self.assertNotIn("store", kwargs)
        self.assertNotIn("include", kwargs)

    def test_include_list_is_not_shared_between_models(self) -> None:
        """Each model gets its own ``include`` list, never the module constant itself."""
        kwargs = self._openai_kwargs(use_responses_api=True)
        self.assertIsNot(kwargs["include"], providers.RESPONSES_STATELESS_SETTINGS["include"])

    def test_responses_compatibility_removes_n_only_for_responses(self) -> None:
        """Ensure the SAP default ``n`` cannot reach the native Responses client."""
        # Replace the parent-class payload method and the routing predicate so
        # the ``ResponsesCompatibleChatOpenAI`` override can be exercised without
        # an actual model deployment.  Object is instantiated without __init__
        # (same pattern as the original scaffold test) to skip credential lookup.
        with patch(
            "gen_ai_hub.proxy.langchain.openai.ChatOpenAI._get_request_payload",
            new=lambda self, input_, stop=None, **kwargs: {"n": 1, "input": []},
        ), patch(
            "gen_ai_hub.proxy.langchain.openai.ChatOpenAI._use_responses_api",
            new=lambda self, payload: True,
        ):
            model = object.__new__(ResponsesCompatibleChatOpenAI)
            payload = model._get_request_payload("hello")
            self.assertEqual(payload, {"input": []})

    def test_claude_model_name_maps_to_non_empty_bedrock_model_id(self) -> None:
        """Map the SAP deployment name to the model ID required by Converse."""
        self.assertEqual(
            ChatBedrockConverse.get_corresponding_model_id(
                "anthropic--claude-4.6-sonnet"
            ),
            "anthropic.claude-sonnet-4-6",
        )


if __name__ == "__main__":
    unittest.main()
