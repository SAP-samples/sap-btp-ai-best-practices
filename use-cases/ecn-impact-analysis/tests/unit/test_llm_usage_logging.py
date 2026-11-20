"""Tests for Cloud Logging-compatible LLM usage event helpers."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from app.observability.llm_usage_logging import (
    TokenUsage,
    emit_llm_usage_event,
    extract_token_usage,
    usage_dict_from_tokens,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, TokenUsage()),
        (
            {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            TokenUsage(
                usage_available=True,
                input_tokens=20,
                input_total_tokens=20,
                output_tokens=5,
                total_tokens=25,
            ),
        ),
        (
            {
                "prompt_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens": 5,
                "total_tokens": 25,
            },
            TokenUsage(
                usage_available=True,
                input_tokens=13,
                cached_input_tokens=7,
                input_total_tokens=20,
                output_tokens=5,
                total_tokens=25,
            ),
        ),
        (
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            TokenUsage(
                usage_available=True,
                input_tokens=0,
                input_total_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        ),
    ],
)
def test_normalize_openai_usage_preserves_availability(
    raw: object, expected: TokenUsage
) -> None:
    """Normalize totals and cache reads without treating missing metadata as zero."""
    assert extract_token_usage(raw) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=30,
                    input_tokens_details=SimpleNamespace(cached_tokens=10),
                    output_tokens=6,
                    total_tokens=36,
                )
            ),
            TokenUsage(20, 10, 6, 36, None, 30, True),
        ),
        (
            SimpleNamespace(
                usage_metadata={
                    "input_tokens": 9,
                    "output_tokens": 4,
                    "total_tokens": 13,
                }
            ),
            TokenUsage(9, None, 4, 13, None, 9, True),
        ),
        (
            SimpleNamespace(
                usage_metadata={
                    "prompt_token_count": 12,
                    "cached_content_token_count": 2,
                    "candidates_token_count": 3,
                    "total_token_count": 15,
                }
            ),
            TokenUsage(10, 2, 3, 15, None, 12, True),
        ),
        (
            {
                "usage": {
                    "inputTokens": 14,
                    "outputTokens": 5,
                    "totalTokens": 21,
                    "cacheReadInputTokens": 3,
                    "cacheWriteInputTokens": 2,
                }
            },
            TokenUsage(11, 3, 5, 21, 2, 14, True),
        ),
        (
            SimpleNamespace(
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2,
                    }
                }
            ),
            TokenUsage(8, None, 2, 10, None, 8, True),
        ),
    ],
)
def test_extract_token_usage_supports_provider_shapes(
    source: object, expected: TokenUsage
) -> None:
    """Normalize Responses, LangChain, Gemini, and Bedrock usage metadata."""
    assert extract_token_usage(source) == expected


def test_invalid_cache_count_does_not_invent_uncached_input() -> None:
    """Keep an inclusive total but leave uncached input unknown for invalid cache data."""
    usage = extract_token_usage(
        {"prompt_tokens": 4, "prompt_tokens_details": {"cached_tokens": 8}}
    )

    assert usage.usage_available is True
    assert usage.input_total_tokens == 4
    assert usage.cached_input_tokens == 8
    assert usage.input_tokens is None


def test_usage_event_has_correlations_and_no_source_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Log correlations and counts without business source content."""
    emit_llm_usage_event(
        route="ecn-impact-analysis",
        method="INTERNAL",
        actor_type="batch",
        provider="sap-ai-core",
        model="gpt-5.4",
        llm_endpoint="responses",
        usage_available=True,
        input_tokens=11,
        cached_input_tokens=None,
        cache_read_input_tokens=None,
        cache_write_input_tokens=None,
        input_total_tokens=11,
        output_tokens=3,
        total_tokens=14,
        outcome="success",
        latency_ms=42,
        correlation_id="correlation-1",
        upload_id="upload-1",
        job_id="job-1",
        attempt_id="attempt-1",
        model_deployment=None,
        stage="FINAL",
    )

    raw_event = capsys.readouterr().out.strip()
    assert ": " not in raw_event
    event = json.loads(raw_event)
    assert event["schema_version"] == "btp.llm_usage.v1"
    assert event["actor_type"] == "batch"
    assert event["method"] == "INTERNAL"
    assert event["route"] == "ecn-impact-analysis"
    assert event["job_id"] == "job-1"
    assert event["usage_available"] is True
    assert event["cached_input_tokens"] is None
    assert event["cache_read_input_tokens"] is None
    assert event["model_deployment"] is None
    assert event["stage"] == "FINAL"
    assert "source_json" not in event
    assert "prompt" not in event


@pytest.mark.parametrize(("available", "value"), [(False, None), (True, 0)])
def test_usage_event_distinguishes_absent_from_measured_zero(
    available: bool,
    value: int | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retain nullable counts and the explicit provider-availability flag."""
    emit_llm_usage_event(
        route="ecn-impact-analysis",
        usage_available=available,
        input_tokens=value,
        cached_input_tokens=value,
        cache_read_input_tokens=value,
        cache_write_input_tokens=value,
        input_total_tokens=value,
        output_tokens=value,
        total_tokens=value,
    )

    event = json.loads(capsys.readouterr().out)
    assert event["usage_available"] is available
    for field in (
        "input_tokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "input_total_tokens",
        "output_tokens",
        "total_tokens",
    ):
        assert event[field] == value


def test_usage_event_hashes_user_and_reads_vcap(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Populate Cloud Foundry metadata and never print the raw user identity."""
    vcap = {
        "application_name": "template-api",
        "space_name": "Dev",
        "organization_name": "btp-ai-sandbox",
    }
    monkeypatch.setenv("VCAP_APPLICATION", json.dumps(vcap))
    monkeypatch.setenv("LOG_USER_HASH_SALT", "pepper")

    emit_llm_usage_event(route="/api/example", user_id="alice@example.com")

    raw = capsys.readouterr().out
    event = json.loads(raw)
    expected_hash = hashlib.sha256(
        "pepper:alice@example.com".encode("utf-8")
    ).hexdigest()[:24]
    assert "alice@example.com" not in raw
    assert event["user_hash"] == expected_hash
    assert event["app_name"] == "template-api"
    assert event["space_name"] == "Dev"
    assert event["org_name"] == "btp-ai-sandbox"


def test_demo_usage_projection_keeps_legacy_zero_response() -> None:
    """Keep the temporary demo API numeric even when provider usage is absent."""
    assert usage_dict_from_tokens(TokenUsage()) == {
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
