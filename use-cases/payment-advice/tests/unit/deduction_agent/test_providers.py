"""Verify provider-factory compatibility details without live model calls.

Converted from the scaffold's pytest-based test to stdlib unittest so it runs
under ``unittest discover`` without requiring pytest.  The ``monkeypatch``
fixture is replaced by ``unittest.mock.patch``.
"""

import unittest
from unittest.mock import patch

from gen_ai_hub.proxy.langchain import ChatBedrockConverse

from app.deduction_agent.template_agent.providers import ResponsesCompatibleChatOpenAI


class TestProviders(unittest.TestCase):
    """Test SAP Gen AI Hub provider factory compatibility."""

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
