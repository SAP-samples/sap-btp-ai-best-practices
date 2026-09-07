"""Tests for LLM token usage stdout observability helpers."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from app.observability.llm_usage_logging import (
    emit_llm_usage_event,
    extract_token_usage,
)
from app.services.joule_knowledge_importer import GenAiHubEmbeddingClient


def test_emit_llm_usage_event_writes_compact_json_with_cf_metadata(
    capsys,
    monkeypatch,
) -> None:
    """Verify usage events are compact Cloud Logging-friendly JSON lines.

    Inputs:
        capsys: Pytest stdout capture fixture.
        monkeypatch: Pytest environment patching fixture.

    Outputs:
        None. Assertions confirm schema fields and Cloud Foundry metadata.
    """

    monkeypatch.setenv(
        "VCAP_APPLICATION",
        json.dumps(
            {
                "application_name": "assessment-api",
                "space_name": "dev",
                "organization_name": "assessment-org",
                "application_uris": ["assessment-api.cfapps.eu10-005.hana.ondemand.com"],
            }
        ),
    )

    emit_llm_usage_event(
        route="/api/ai-review/jobs",
        method="POST",
        user_id="alice@example.test",
        client_host="pytest",
        model="gpt-5.4",
        llm_endpoint="responses",
        input_tokens=12,
        output_tokens=5,
        outcome="success",
        latency_ms=42,
        correlation_id="corr-123",
    )

    output = capsys.readouterr().out.strip()
    assert "\n" not in output
    assert ": " not in output
    event = json.loads(output)
    assert event["schema_version"] == "btp.llm_usage.v1"
    assert event["event_type"] == "llm_usage"
    assert event["app_name"] == "assessment-api"
    assert event["space_name"] == "dev"
    assert event["org_name"] == "assessment-org"
    assert event["route"] == "/api/ai-review/jobs"
    assert event["model"] == "gpt-5.4"
    assert event["llm_endpoint"] == "responses"
    assert event["input_tokens"] == 12
    assert event["cached_input_tokens"] == 0
    assert event["cache_read_input_tokens"] == 0
    assert event["cache_write_input_tokens"] == 0
    assert event["input_total_tokens"] == 12
    assert event["output_tokens"] == 5
    assert event["total_tokens"] == 17
    assert event["correlation_id"] == "corr-123"
    assert event["user_hash"]
    assert "alice@example.test" not in output


def test_extract_token_usage_normalizes_provider_response_shapes() -> None:
    """Verify provider-specific usage metadata maps to stable token fields.

    Inputs:
        None. The test builds fake Responses, LangChain, and embedding objects.

    Outputs:
        None. Assertions confirm normalized input, output, and total counts.
    """

    responses_usage = extract_token_usage(
        SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=25, total_tokens=125)
        )
    )
    assert responses_usage.input_tokens == 100
    assert responses_usage.output_tokens == 25
    assert responses_usage.total_tokens == 125

    langchain_usage = extract_token_usage(
        SimpleNamespace(
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 10,
                    "total_tokens": 50,
                }
            }
        )
    )
    assert langchain_usage.input_tokens == 40
    assert langchain_usage.output_tokens == 10
    assert langchain_usage.total_tokens == 50

    embedding_usage = extract_token_usage(
        {"usage": {"prompt_tokens": 33, "total_tokens": 33}}
    )
    assert embedding_usage.input_tokens == 33
    assert embedding_usage.output_tokens == 0
    assert embedding_usage.total_tokens == 33


def test_extract_token_usage_separates_cached_provider_input() -> None:
    """Verify cached provider input is not double-counted as uncached usage."""

    usage = extract_token_usage(
        {
            "usage": {
                "prompt_token_count": 100,
                "cached_content_token_count": 60,
                "candidates_token_count": 20,
                "total_token_count": 120,
            }
        }
    )

    assert usage.input_tokens == 40
    assert usage.cached_input_tokens == 60
    assert usage.cache_read_input_tokens == 60
    assert usage.cache_write_input_tokens == 0
    assert usage.input_total_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120


def test_embedding_client_emits_embedding_usage_event(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SAP Gen AI Hub embedding calls emit token usage telemetry.

    Inputs:
        capsys: Pytest stdout capture fixture.
        monkeypatch: Pytest helper used to inject a fake SAP SDK module.

    Outputs:
        None. Assertions confirm embedding endpoint metadata and zero output
        tokens.
    """

    class FakeEmbeddingsApi:
        """Fake OpenAI-compatible embeddings API returning usage metadata."""

        def create(self, model_name: str, input: list[str]) -> SimpleNamespace:
            """Return deterministic vectors for the supplied texts.

            Inputs:
                model_name: Embedding deployment name.
                input: Text values to embed.

            Outputs:
                SimpleNamespace: Fake embedding response with usage metadata.
            """

            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[float(index), 0.0])
                    for index, _text in enumerate(input, start=1)
                ],
                usage={"prompt_tokens": 9, "total_tokens": 9},
            )

    monkeypatch.setitem(
        sys.modules,
        "gen_ai_hub.proxy.native.openai",
        SimpleNamespace(embeddings=FakeEmbeddingsApi()),
    )

    vectors = GenAiHubEmbeddingClient(model_name="text-embedding-test").embed_batch(
        ["first", "second"]
    )

    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "worker:document-processing:embeddings"
    assert event["model"] == "text-embedding-test"
    assert event["llm_endpoint"] == "embeddings"
    assert event["input_tokens"] == 9
    assert event["output_tokens"] == 0
    assert event["total_tokens"] == 9
    assert event["outcome"] == "success"
