"""Unit tests for EcnSettings and canonical model configuration.

Tests cover:
- Default values for every setting field.
- Environment-variable overrides using the documented ENVIRONMENT_FIELDS mapping.
- Rejection of invalid values (non-positive integers, out-of-range strings).
- canonical_model_config output format: sorted compact JSON with the four
  mandated keys and strict_structured_output=True.
- Empty strings in the environment are ignored (treated as absent).

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_settings.py
"""

import json

import pytest
from app.ecn.settings import EcnSettings, canonical_model_config, load_ecn_settings
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Defaults and canonical model config
# ---------------------------------------------------------------------------


def test_settings_defaults_and_canonical_model_config() -> None:
    """Default settings must match the contract, and canonical JSON must contain exactly four keys."""
    settings = load_ecn_settings({})
    assert settings.llm_max_concurrency == 4
    assert settings.llm_request_timeout_seconds == 180
    assert settings.max_json_upload_bytes == 128 * 1024 * 1024
    assert settings.max_workbook_upload_bytes == 100 * 1024 * 1024
    assert settings.max_workbook_rows == 100_000
    assert settings.max_workbook_cells == 3_000_000
    assert settings.max_prompt_evidence_items == 100
    assert settings.max_prompt_characters == 120_000
    assert settings.llm_model_name == "gpt-5.4"
    assert json.loads(canonical_model_config(settings)) == {
        "max_prompt_characters": 120_000,
        "max_prompt_evidence_items": 100,
        "request_timeout_seconds": 180,
        "strict_structured_output": True,
    }


# ---------------------------------------------------------------------------
# canonical_model_config must be sorted compact JSON
# ---------------------------------------------------------------------------


def test_canonical_model_config_keys_are_sorted() -> None:
    """canonical_model_config must emit sorted keys for deterministic output."""
    settings = load_ecn_settings({})
    raw = canonical_model_config(settings)
    parsed = json.loads(raw)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_canonical_model_config_is_compact() -> None:
    """canonical_model_config must not contain any extra whitespace."""
    settings = load_ecn_settings({})
    raw = canonical_model_config(settings)
    # Compact JSON: re-encoding with no separators must yield identical string.
    parsed = json.loads(raw)
    expected = json.dumps(parsed, separators=(",", ":"), sort_keys=True)
    assert raw == expected


def test_canonical_model_config_has_exactly_four_keys() -> None:
    """canonical_model_config must contain exactly max_prompt_characters,
    max_prompt_evidence_items, request_timeout_seconds, and strict_structured_output.
    """
    settings = load_ecn_settings({})
    parsed = json.loads(canonical_model_config(settings))
    assert set(parsed.keys()) == {
        "max_prompt_characters",
        "max_prompt_evidence_items",
        "request_timeout_seconds",
        "strict_structured_output",
    }


def test_canonical_model_config_strict_structured_output_is_true() -> None:
    """canonical_model_config must always set strict_structured_output=True."""
    settings = load_ecn_settings({})
    parsed = json.loads(canonical_model_config(settings))
    assert parsed["strict_structured_output"] is True


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


def test_settings_override_via_environment_variables() -> None:
    """All documented environment names must override their settings fields."""
    env = {
        "LLM_MAX_CONCURRENCY": "8",
        "LLM_REQUEST_TIMEOUT_SECONDS": "60",
        "MAX_JSON_UPLOAD_BYTES": "10485760",
        "MAX_WORKBOOK_UPLOAD_BYTES": "52428800",
        "MAX_WORKBOOK_ROWS": "50000",
        "MAX_WORKBOOK_CELLS": "1500000",
        "MAX_PROMPT_EVIDENCE_ITEMS": "200",
        "MAX_PROMPT_CHARACTERS": "128000",
        "LLM_MODEL": "gpt-test-deployment",
    }
    settings = load_ecn_settings(env)
    assert settings.llm_max_concurrency == 8
    assert settings.llm_request_timeout_seconds == 60
    assert settings.max_json_upload_bytes == 10_485_760
    assert settings.max_workbook_upload_bytes == 52_428_800
    assert settings.max_workbook_rows == 50_000
    assert settings.max_workbook_cells == 1_500_000
    assert settings.max_prompt_evidence_items == 200
    assert settings.max_prompt_characters == 128_000
    assert settings.llm_model_name == "gpt-test-deployment"


def test_empty_string_environment_values_are_ignored() -> None:
    """Environment entries with empty string values must be treated as absent."""
    env = {"LLM_MAX_CONCURRENCY": ""}
    settings = load_ecn_settings(env)
    assert settings.llm_max_concurrency == 4  # default


def test_settings_are_frozen() -> None:
    """EcnSettings instances must be immutable (frozen=True)."""
    settings = load_ecn_settings({})
    with pytest.raises(ValidationError):
        settings.llm_max_concurrency = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Rejection of invalid values
# ---------------------------------------------------------------------------


def test_settings_reject_zero_concurrency() -> None:
    """PositiveInt fields must reject zero."""
    with pytest.raises(ValidationError):
        EcnSettings(llm_max_concurrency=0)


def test_settings_reject_negative_timeout() -> None:
    """PositiveInt fields must reject negative values."""
    with pytest.raises(ValidationError):
        EcnSettings(llm_request_timeout_seconds=-1)


def test_settings_reject_prompt_characters_below_minimum() -> None:
    """max_prompt_characters must be >= 4096."""
    with pytest.raises(ValidationError):
        EcnSettings(max_prompt_characters=4095)


def test_settings_reject_blank_model_name() -> None:
    """llm_model_name must not be blank."""
    with pytest.raises(ValidationError):
        EcnSettings(llm_model_name="   ")


def test_settings_reject_extra_fields() -> None:
    """extra='forbid' must reject any undocumented setting."""
    with pytest.raises(ValidationError):
        EcnSettings.model_validate({"llm_max_concurrency": 4, "unknown_field": "oops"})


def test_canonical_model_config_reflects_override() -> None:
    """canonical_model_config values must reflect the actual settings instance."""
    settings = load_ecn_settings(
        {"MAX_PROMPT_CHARACTERS": "100000", "MAX_PROMPT_EVIDENCE_ITEMS": "300"}
    )
    parsed = json.loads(canonical_model_config(settings))
    assert parsed["max_prompt_characters"] == 100_000
    assert parsed["max_prompt_evidence_items"] == 300
    assert parsed["request_timeout_seconds"] == 180  # default unchanged
