"""Tests for SAP Cloud Logging LLM usage events."""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from app.observability.llm_usage_logging import (
    LlmUsageContext,
    emit_llm_usage_event,
    extract_client_host_from_request,
    extract_token_usage,
    extract_user_id_from_request,
)


class TestTokenUsageExtraction(unittest.TestCase):
    """Verify provider usage metadata is normalized into one shape."""

    def test_extracts_langchain_cached_token_metadata(self) -> None:
        """LangChain usage_metadata cache_read values are logged separately."""
        response = SimpleNamespace(
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cache_read": 35},
            }
        )

        usage = extract_token_usage(response)

        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 35)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 120)

    def test_extracts_openai_response_metadata_token_usage(self) -> None:
        """OpenAI/LangChain response_metadata token_usage values are supported."""
        response = SimpleNamespace(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 5},
                }
            }
        )

        usage = extract_token_usage(response)

        self.assertEqual(usage.input_tokens, 12)
        self.assertEqual(usage.cached_input_tokens, 5)
        self.assertEqual(usage.output_tokens, 8)
        self.assertEqual(usage.total_tokens, 20)


class TestLlmUsageEventEmission(unittest.TestCase):
    """Verify compact JSON stdout event behavior."""

    def test_emits_compact_json_with_hashed_user_and_vcap_metadata(self) -> None:
        """Events include stable schema fields and never print raw identities."""
        vcap = {
            "application_name": "eligibility-analysis-api",
            "space_name": "dev",
            "organization_name": "sap-demo",
            "uris": ["eligibility-analysis-api.cfapps.eu10-004.hana.ondemand.com"],
        }
        context = LlmUsageContext(
            route="/api/a2a",
            method="POST",
            user_id="person@example.com",
            actor_type="human",
            client_host="pytest",
            correlation_id="corr-123",
        )

        with patch.dict(
            os.environ,
            {
                "VCAP_APPLICATION": json.dumps(vcap),
                "LOG_USER_HASH_SALT": "unit-test-salt",
            },
        ):
            stream = io.StringIO()
            with redirect_stdout(stream):
                emit_llm_usage_event(
                    context=context,
                    model="gpt-4.1",
                    llm_endpoint="chat.completions",
                    input_tokens=42,
                    cached_input_tokens=11,
                    output_tokens=7,
                    outcome="success",
                    latency_ms=123,
                )

        output = stream.getvalue()
        self.assertEqual(output.count("\n"), 1)
        self.assertNotIn("person@example.com", output)
        event = json.loads(output)
        self.assertEqual(event["schema_version"], "btp.llm_usage.v1")
        self.assertEqual(event["event_type"], "llm_usage")
        self.assertEqual(event["app_name"], "eligibility-analysis-api")
        self.assertEqual(event["space_name"], "dev")
        self.assertEqual(event["org_name"], "sap-demo")
        self.assertEqual(event["route"], "/api/a2a")
        self.assertEqual(event["method"], "POST")
        self.assertEqual(event["actor_type"], "human")
        self.assertEqual(event["client_host"], "pytest")
        self.assertEqual(event["provider"], "sap-ai-core")
        self.assertEqual(event["model"], "gpt-4.1")
        self.assertEqual(event["llm_endpoint"], "chat.completions")
        self.assertEqual(event["input_tokens"], 42)
        self.assertEqual(event["cached_input_tokens"], 11)
        self.assertEqual(event["output_tokens"], 7)
        self.assertEqual(event["total_tokens"], 49)
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["latency_ms"], 123)
        self.assertEqual(event["correlation_id"], "corr-123")
        self.assertIsInstance(event["user_hash"], str)

    def test_extracts_request_identity_and_client_context(self) -> None:
        """Request helpers read user, host, and forwarded metadata safely."""
        request = SimpleNamespace(
            headers={
                "x-client-user-id": "person@example.com",
                "x-client-host": "browser-host",
                "x-correlation-id": "corr-456",
            },
            client=SimpleNamespace(host="127.0.0.1"),
        )

        self.assertEqual(extract_user_id_from_request(request), "person@example.com")
        self.assertEqual(extract_client_host_from_request(request), "browser-host")


if __name__ == "__main__":
    unittest.main()
