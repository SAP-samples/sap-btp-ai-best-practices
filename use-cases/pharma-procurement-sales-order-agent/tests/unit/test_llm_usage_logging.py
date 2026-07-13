"""Tests for Cloud Logging-compatible LLM usage event helpers."""

import hashlib
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from app.observability.llm_usage_logging import (
    emit_llm_usage_event,
    extract_token_usage,
    usage_dict_from_tokens,
)


class LLMUsageLoggingTests(unittest.TestCase):
    """Validate stdout event shape and token usage normalization."""

    def test_emit_llm_usage_event_prints_compact_cloud_logging_json(self) -> None:
        """Emit one event with VCAP metadata, hashed user, and token totals."""
        vcap = {
            "application_name": "template-api",
            "space_name": "Dev",
            "organization_name": "btp-ai-sandbox",
        }
        expected_hash = hashlib.sha256("pepper:alice@example.com".encode("utf-8")).hexdigest()[:24]

        with patch.dict(
            os.environ,
            {
                "VCAP_APPLICATION": json.dumps(vcap),
                "LOG_USER_HASH_SALT": "pepper",
            },
            clear=False,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                emit_llm_usage_event(
                    route="/api/llm-demo/chat",
                    method="POST",
                    user_id="alice@example.com",
                    client_host="Mozilla/5.0",
                    provider="sap-ai-core",
                    model="gpt-4.1",
                    llm_endpoint="chat.completions",
                    input_tokens=12,
                    output_tokens=8,
                    outcome="success",
                    latency_ms=123,
                    correlation_id="corr-123",
                )

        raw_event = output.getvalue().strip()
        self.assertNotIn(": ", raw_event)
        event = json.loads(raw_event)

        self.assertEqual(event["schema_version"], "btp.llm_usage.v1")
        self.assertEqual(event["event_type"], "llm_usage")
        self.assertEqual(event["app_name"], "template-api")
        self.assertEqual(event["space_name"], "Dev")
        self.assertEqual(event["org_name"], "btp-ai-sandbox")
        self.assertEqual(event["route"], "/api/llm-demo/chat")
        self.assertEqual(event["method"], "POST")
        self.assertEqual(event["user_hash"], expected_hash)
        self.assertEqual(event["actor_type"], "human")
        self.assertEqual(event["client_host"], "Mozilla/5.0")
        self.assertEqual(event["provider"], "sap-ai-core")
        self.assertEqual(event["model"], "gpt-4.1")
        self.assertEqual(event["llm_endpoint"], "chat.completions")
        self.assertEqual(event["input_tokens"], 12)
        self.assertEqual(event["output_tokens"], 8)
        self.assertEqual(event["total_tokens"], 20)
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["latency_ms"], 123)
        self.assertEqual(event["correlation_id"], "corr-123")

    def test_extract_token_usage_supports_native_chat_completion_usage(self) -> None:
        """Normalize OpenAI chat-completion usage fields into input/output tokens."""
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=21, completion_tokens=13, total_tokens=34)
        )

        usage = extract_token_usage(response)

        self.assertEqual(usage.input_tokens, 21)
        self.assertEqual(usage.output_tokens, 13)
        self.assertEqual(usage.total_tokens, 34)
        self.assertEqual(
            usage_dict_from_tokens(usage),
            {"prompt_tokens": 21, "completion_tokens": 13, "total_tokens": 34},
        )

    def test_extract_token_usage_supports_langchain_usage_metadata(self) -> None:
        """Normalize LangChain AIMessage usage metadata into input/output tokens."""
        message = SimpleNamespace(
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        )

        usage = extract_token_usage(message)

        self.assertEqual(usage.input_tokens, 5)
        self.assertEqual(usage.output_tokens, 3)
        self.assertEqual(usage.total_tokens, 8)


if __name__ == "__main__":
    unittest.main()
