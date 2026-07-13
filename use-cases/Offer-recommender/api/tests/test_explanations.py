from __future__ import annotations

import sys
import types

import pandas as pd

from app.nbo.models import DecisionExplanation
from app.nbo.output import write_excel, write_json
from app.services.explanations import DeterministicExplanationPolisher
from app.services.recommendations import RecommendationService


def _clear_aicore_env(monkeypatch) -> None:
    """Remove GenAI Hub credentials so explanation tests use deterministic fallback."""
    for name in (
        "AICORE_AUTH_URL",
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_BASE_URL",
        "AICORE_RESOURCE_GROUP",
    ):
        monkeypatch.delenv(name, raising=False)


def _set_aicore_env(monkeypatch) -> None:
    """Populate fake GenAI Hub credentials to prove explanations ignore them."""
    monkeypatch.setenv("AICORE_AUTH_URL", "https://auth.example.invalid")
    monkeypatch.setenv("AICORE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("AICORE_BASE_URL", "https://api.example.invalid")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "default")


def _install_fake_native_chat(monkeypatch, parse_impl) -> None:
    """Install fake GenAI Hub native OpenAI modules that must not be called."""
    gen_ai_hub = types.ModuleType("gen_ai_hub")
    proxy = types.ModuleType("gen_ai_hub.proxy")
    native = types.ModuleType("gen_ai_hub.proxy.native")
    openai = types.ModuleType("gen_ai_hub.proxy.native.openai")
    openai.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse_impl)
    )

    monkeypatch.setitem(sys.modules, "gen_ai_hub", gen_ai_hub)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy", proxy)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native", native)
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.openai", openai)


def test_deterministic_polisher_never_calls_genai_hub(monkeypatch) -> None:
    """Explanation text should stay deterministic even if GenAI Hub is configured."""
    _set_aicore_env(monkeypatch)
    monkeypatch.setenv("EXPLANATION_MODEL", "gpt-4.1-mini")
    called = False

    def fake_parse(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("explanation code should not call GenAI Hub")

    _install_fake_native_chat(monkeypatch, fake_parse)
    audit = DecisionExplanation(
        summary="Plan Savings Review was selected.",
        details=["Estimated savings: $20.23."],
        facts_used=["Current rate plan: E21."],
        rules_used=["lower_estimated_bill"],
    )

    result = DeterministicExplanationPolisher().polish(audit)

    assert called is False
    assert result.summary == "Plan Savings Review was selected."
    assert result.details == ["Estimated savings: $20.23."]
    assert result.polish_status == "fallback"


def test_recommendation_service_attaches_nested_offer_and_question_explanations(monkeypatch) -> None:
    """Recommendations should explain final, blocked, and follow-up decisions."""
    _clear_aicore_env(monkeypatch)
    service = RecommendationService()

    result = service.evaluate_account("104")

    assert result.final_offer is not None
    assert result.final_offer.explanation is not None
    assert result.final_offer.explanation.polish_status == "fallback"
    assert "Plan Savings Review" in result.final_offer.explanation.summary
    assert "Demand Saver 5-10 P.M." in result.final_offer.explanation.summary
    assert "E16" not in result.final_offer.explanation.summary
    assert result.final_offer.explanation.facts_used
    explanation_text = " ".join(
        [
            result.final_offer.explanation.summary,
            *result.final_offer.explanation.details,
            *result.final_offer.explanation.facts_used,
        ]
    )
    assert "Current rate plan: Residential Time of Use 3-6." in explanation_text
    assert "Recommended rate plan: Demand Saver 5-10 P.M.." in explanation_text
    assert "Current rate plan: E21" not in explanation_text
    assert "Recommended rate plan: E16" not in explanation_text
    assert all(
        "Reason codes" not in detail
        and "lower_estimated_bill" not in detail
        and "_" not in detail
        for detail in result.final_offer.explanation.details
    )

    blocked_offer = next(
        offer for offer in result.blocked_offers if offer.program_id == "income_qualified_discount"
    )
    assert blocked_offer.explanation is not None
    assert blocked_offer.explanation.blockers
    assert "missing" in blocked_offer.explanation.summary.casefold()

    assert result.questions
    assert result.questions[0].explanation is not None
    assert result.questions[0].explanation.blockers == [result.questions[0].expected_fact]


def test_declined_offer_explanation_mentions_session_suppression(monkeypatch) -> None:
    """Session-declined offers should carry a suppression explanation."""
    _clear_aicore_env(monkeypatch)
    service = RecommendationService()

    result = service.evaluate_account(
        "104",
        declined_programs=["rate_plan_optimization"],
    )

    suppressed = next(
        offer for offer in result.blocked_offers if offer.program_id == "rate_plan_optimization"
    )
    assert suppressed.explanation is not None
    assert "declined" in suppressed.explanation.summary.casefold()
    assert "suppressed_session_decline" in suppressed.explanation.rules_used


def test_decision_outputs_include_nested_explanation_fields(tmp_path, monkeypatch) -> None:
    """Batch JSON and Excel artifacts should expose per-offer and per-question explanations."""
    _clear_aicore_env(monkeypatch)
    result = RecommendationService().evaluate_account("104")

    json_path = write_json([result], path=tmp_path / "recommendations.json")
    excel_path = write_excel([result], path=tmp_path / "recommendations.xlsx")

    payload = json_path.read_text(encoding="utf-8")
    assert '"explanation"' in payload
    assert '"polish_status": "fallback"' in payload

    offer_rows = pd.read_excel(excel_path, sheet_name="Offer Decisions")
    question_rows = pd.read_excel(excel_path, sheet_name="Questions")

    assert "EXPLANATION_SUMMARY" in offer_rows.columns
    assert "EXPLANATION_DETAILS" in offer_rows.columns
    assert "EXPLANATION_SUMMARY" in question_rows.columns
    assert "EXPLANATION_DETAILS" in question_rows.columns
