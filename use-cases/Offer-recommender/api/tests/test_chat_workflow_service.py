from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.chat.service import (
    ChatInvalidDeclineError,
    ChatThreadNotFoundError,
    ChatWorkflowService,
    _extract_llm_token_usage,
    _normalize_checkpoint_value,
)
from app.nbo.models import WorkflowStage
from app.services.saved_answers import InMemorySavedCustomerAnswerRepository, SavedCustomerAnswerService


def test_primary_offer_is_visible_before_followup_opt_in(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    updated = service.post_message(thread.thread_id, "my billing account number is 6001")

    assert updated.decision_result is not None
    assert updated.decision_result.final_offer is not None
    assert updated.decision_result.workflow_stage == "primary_offer_with_followup"
    assert updated.decision_result.explanation is not None
    assert updated.decision_result.explanation.summary.startswith(
        "Plan Savings Review is the current primary recommendation"
    )
    assert any(
        fact.startswith("Estimated monthly cost comparison:")
        for fact in updated.decision_result.explanation.facts_used
    )
    assert updated.current_question is not None
    assert updated.current_question.expected_fact == "customer_wants_followup"
    assert "I found a primary recommendation for this account." in updated.messages[-1].content
    assert (
        "Would you like to answer a few quick questions to check for more programs?"
        in updated.messages[-1].content
    )
    assert "Plan Savings Review" not in updated.messages[-1].content
    assert "6001" not in updated.messages[-1].content


def test_followup_opt_in_uses_plain_language_without_repeating_primary_summary(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    updated = service.post_message(thread.thread_id, "Yes")

    assert updated.decision_result is not None
    assert updated.decision_result.final_offer is not None
    assert updated.decision_result.workflow_stage == "primary_offer_with_followup"
    assert updated.current_question is not None
    assert updated.current_question.expected_fact == "account_name_type"
    assert updated.messages[-1].content.startswith("Next question:")
    assert "Next question:" in updated.messages[-1].content
    assert "Recorded answer:" not in updated.messages[-1].content
    assert "Thanks." not in updated.messages[-1].content
    assert "I'll keep checking for more programs." not in updated.messages[-1].content
    assert "Plan Savings Review" not in updated.messages[-1].content
    assert "bring_your_own_thermostat" not in updated.messages[-1].content
    assert "6001" not in updated.messages[-1].content


def test_followup_message_calls_out_newly_eligible_additional_offers(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    service.post_message(thread.thread_id, "Yes")
    service.post_message(thread.thread_id, "Personal / individual")
    service.post_message(thread.thread_id, "Yes")
    updated = service.post_message(thread.thread_id, "Yes")

    assert updated.decision_result is not None
    assert updated.decision_result.final_offer is not None
    additional_offers = [
        offer.display_name
        for offer in updated.decision_result.eligible_offers
        if offer.program_id != updated.decision_result.final_offer.program_id
    ]

    assert additional_offers == ["Household Assistance Discount"]
    assert "Newly eligible offer: Household Assistance Discount." in updated.messages[-1].content
    assert "rate_plan_optimization" not in updated.messages[-1].content
    assert "income_qualified_discount" not in updated.messages[-1].content


def test_followup_opt_out_unknown_ends_optional_discovery_for_session(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    updated = service.post_message(thread.thread_id, "Not sure / Prefer not to answer")

    assert updated.decision_result is not None
    assert updated.decision_result.final_offer is not None
    assert updated.decision_result.workflow_stage == "primary_offer_final"
    assert updated.current_question is None
    assert updated.status_phase == "complete"
    assert (
        "Would you like to continue with a few questions to explore additional programs?"
        not in updated.messages[-1].content
    )


def test_unknown_followup_answer_does_not_repeat_the_same_question(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    service.post_message(thread.thread_id, "Yes")
    service.post_message(thread.thread_id, "Yes")
    updated = service.post_message(thread.thread_id, "Other / Not sure")

    assert updated.user_answers["account_name_type"] is None
    assert updated.current_question is not None
    assert updated.current_question.expected_fact != "account_name_type"
    assert "Is the account held in an individual's name or in a company/business name?" not in updated.messages[-1].content


def test_declined_program_is_suppressed_from_future_recommendations(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    updated = service.post_message(thread.thread_id, "my billing account number is 104")
    assert updated.decision_result is not None
    assert updated.decision_result.final_offer is not None

    declined_program_id = updated.decision_result.final_offer.program_id
    declined = service.decline_program(thread.thread_id, declined_program_id)

    assert declined.declined_programs == [declined_program_id]
    assert declined.current_question is not None
    assert declined.current_question.source == "customer"
    assert "I removed Plan Savings Review from this session." in declined.messages[-1].content
    assert "Declined programs suppressed" not in declined.messages[-1].content
    assert "rate_plan_optimization" not in declined.messages[-1].content
    if declined.decision_result and declined.decision_result.final_offer:
        assert declined.decision_result.final_offer.program_id != declined_program_id


def test_decline_program_rejects_program_not_currently_eligible(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 104")

    with pytest.raises(ChatInvalidDeclineError):
        service.decline_program(thread.thread_id, "income_qualified_discount")


def test_unknown_thread_raises_not_found_error(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    with pytest.raises(ChatThreadNotFoundError):
        service.get_thread("missing-thread")


def test_system_blocker_does_not_stop_chat_when_customer_questions_remain(
    tmp_path: Path,
) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    updated = service.post_message(thread.thread_id, "my billing account number is 103")

    assert updated.decision_result is not None
    assert updated.decision_result.workflow_stage == "needs_core_facts"
    assert updated.current_question is not None
    assert updated.current_question.source == "customer"
    assert updated.status_phase == "questioning"
    assert updated.pending_questions
    assert updated.pending_questions[0].expected_fact == "has_current_snapshot"
    assert updated.messages[-1].content.startswith("Next question:")


def test_customer_answer_after_system_blocker_path_continues_question_flow(
    tmp_path: Path,
) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 103")
    updated = service.post_message(thread.thread_id, "Yes")

    assert updated.current_question is not None
    assert updated.status_phase == "questioning"
    assert updated.messages[-2].role == "user"
    assert updated.messages[-2].content == "Yes"
    assert updated.messages[-1].content.startswith("Next question:")


def test_newly_eligible_additional_offers_excludes_demoted_previous_primary() -> None:
    service = ChatWorkflowService.__new__(ChatWorkflowService)

    newly_eligible = service._newly_eligible_additional_offers(
        {
            "final_offer": {
                "program_id": "rate_plan_optimization",
                "display_name": "Plan Savings Review",
            },
            "eligible_offers": [
                {
                    "program_id": "rate_plan_optimization",
                    "display_name": "Plan Savings Review",
                }
            ],
        },
        {
            "final_offer": {
                "program_id": "income_qualified_discount",
                "display_name": "Household Assistance Discount",
            },
            "eligible_offers": [
                {
                    "program_id": "income_qualified_discount",
                    "display_name": "Household Assistance Discount",
                },
                {
                    "program_id": "rate_plan_optimization",
                    "display_name": "Plan Savings Review",
                },
                {
                    "program_id": "byot",
                    "display_name": "BYOT",
                },
            ],
        },
    )

    assert [offer["display_name"] for offer in newly_eligible] == ["BYOT"]


def test_completion_copy_distinguishes_primary_recommendation_from_additional_programs() -> None:
    service = ChatWorkflowService.__new__(ChatWorkflowService)

    message = service._format_completion_update(
        {
            "final_offer": {
                "program_id": "rate_plan_optimization",
                "display_name": "Plan Savings Review",
            },
            "eligible_offers": [
                {
                    "program_id": "rate_plan_optimization",
                    "display_name": "Plan Savings Review",
                },
                {
                    "program_id": "income_qualified_discount",
                    "display_name": "Household Assistance Discount",
                },
            ],
            "workflow_stage": "primary_offer_final",
        }
    )

    assert "primary recommendation remains in place" in message
    assert "additional program is now eligible" in message


def test_thread_state_resumes_from_sqlite_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "threads.sqlite"

    first_service = ChatWorkflowService(db_path=db_path)
    thread = first_service.create_thread()
    first_service.post_message(thread.thread_id, "my billing account number is 103")

    resumed_service = ChatWorkflowService(db_path=db_path)
    resumed = resumed_service.get_thread(thread.thread_id)

    assert resumed.thread_id == thread.thread_id
    assert resumed.billing_account == "103"
    assert resumed.current_question is not None
    assert resumed.current_question.source == "customer"
    assert resumed.status_phase == "questioning"

    checkpoint = resumed_service.graph.get_state(resumed_service._config(thread.thread_id))
    assert checkpoint.values["decision_result"]["workflow_stage"] == "needs_core_facts"
    assert isinstance(checkpoint.values["decision_result"]["workflow_stage"], str)


def test_checkpoint_normalizer_converts_enums_to_strings() -> None:
    normalized = _normalize_checkpoint_value(
        {
            "decision_result": {
                "workflow_stage": WorkflowStage.PRIMARY_OFFER_FINAL,
                "eligible_offers": [{"program_id": "rate_plan_optimization"}],
            },
            "pending_questions": [],
        }
    )

    assert normalized["decision_result"]["workflow_stage"] == "primary_offer_final"
    assert isinstance(normalized["decision_result"]["workflow_stage"], str)


@pytest.mark.parametrize(
    "answer_text",
    [
        "not sure",
        "prefer not to answer",
        "Not sure / Prefer not to answer",
        "don't know",
    ],
)
def test_plain_language_opt_out_variants_map_to_nullable_answer(
    tmp_path: Path,
    answer_text: str,
) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    updated = service.post_message(thread.thread_id, answer_text)

    assert updated.user_answers["customer_wants_followup"] is None
    assert updated.decision_result is not None
    assert updated.decision_result.workflow_stage == "primary_offer_final"


def test_llm_fallback_maps_free_text_affirmation_to_yes(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")
    service._match_answer_option_with_llm = lambda question, message: (True, True)  # type: ignore[method-assign]

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    updated = service.post_message(thread.thread_id, "Sure")

    assert updated.user_answers["customer_wants_followup"] is True
    assert updated.current_question is not None
    assert updated.current_question.expected_fact == "account_name_type"
    assert "Please answer using one of the available options" not in updated.messages[-1].content


def test_llm_fallback_reasks_when_free_text_is_ambiguous(tmp_path: Path) -> None:
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")
    service._match_answer_option_with_llm = lambda question, message: (False, None)  # type: ignore[method-assign]

    thread = service.create_thread()
    service.post_message(thread.thread_id, "my billing account number is 6001")
    updated = service.post_message(thread.thread_id, "Maybe")

    assert updated.current_question is not None
    assert updated.current_question.expected_fact == "customer_wants_followup"
    assert updated.user_answers == {}
    assert updated.messages[-1].content.startswith("Please answer using one of the available options:")


def test_extract_llm_token_usage_reads_langchain_usage_metadata() -> None:
    """Verify token usage extraction supports LangChain usage_metadata fields."""
    class Response:
        """Minimal LLM response carrying LangChain-style usage metadata."""

        usage_metadata = {
            "input_tokens": 17,
            "output_tokens": 5,
            "total_tokens": 22,
        }

    assert _extract_llm_token_usage(Response()) == {
        "input_tokens": 17,
        "output_tokens": 5,
        "total_tokens": 22,
    }


def test_answer_classifier_logs_llm_token_usage_when_logger_is_configured(
    tmp_path: Path,
) -> None:
    """Verify an injected token logger receives answer-classifier usage events."""
    class Response:
        """Minimal answer-classifier response with content and usage metadata."""

        content = "MATCH: 1"
        usage_metadata = {
            "input_tokens": 41,
            "output_tokens": 3,
            "total_tokens": 44,
        }

    class FakeLlm:
        """Fake answer-classifier LLM that returns a deterministic match response."""

        def invoke(self, prompt: str) -> Response:
            """Return a response after checking the expected user reply is in the prompt."""
            assert "User reply: 'Absolutely'" in prompt
            return Response()

    logged_events: list[dict[str, object]] = []
    service = ChatWorkflowService(
        db_path=tmp_path / "threads.sqlite",
        answer_classifier_token_logger=logged_events.append,
    )
    service._answer_classifier_llm = FakeLlm()
    service._answer_classifier_llm_initialized = True

    matched, value = service._match_answer_option_with_llm(
        {
            "prompt": "Would you like to continue?",
            "expected_fact": "customer_wants_followup",
            "answer_options": [
                {"label": "Yes", "value": True},
                {"label": "No", "value": False},
            ],
        },
        "Absolutely",
    )

    assert matched is True
    assert value is True
    assert logged_events == [
        {
            "llm_call": "chat_answer_classifier",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 41,
            "output_tokens": 3,
            "total_tokens": 44,
        }
    ]


def test_answer_classifier_can_log_token_usage_to_python_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify the environment flag enables token usage logging through Python logging."""
    class Response:
        """Minimal LLM response carrying token counts for the logger path."""

        usage_metadata = {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        }

    monkeypatch.setenv("CHAT_ANSWER_CLASSIFIER_LOG_TOKEN_USAGE", "true")
    service = ChatWorkflowService(db_path=tmp_path / "threads.sqlite")

    with caplog.at_level(logging.INFO, logger="app.chat.service"):
        service._log_answer_classifier_token_usage(Response())

    assert "chat_answer_classifier_token_usage" in caplog.text
    assert "input_tokens=12" in caplog.text
    assert "output_tokens=4" in caplog.text
    assert "total_tokens=16" in caplog.text


def test_chat_reuses_saved_answers_across_new_threads(tmp_path: Path) -> None:
    """Saved account answers should prevent repeated questions in a later thread."""
    saved_answers = SavedCustomerAnswerService(InMemorySavedCustomerAnswerRepository())
    service = ChatWorkflowService(
        db_path=tmp_path / "threads.sqlite",
        saved_answers=saved_answers,
    )

    first_thread = service.create_thread()
    service.post_message(first_thread.thread_id, "my billing account number is 6001")
    service.post_message(first_thread.thread_id, "Yes")
    service.post_message(first_thread.thread_id, "Personal / individual")

    assert saved_answers.get_answer_values("6001") == {
        "customer_wants_followup": True,
        "account_name_type": "PERSONAL",
    }

    second_thread = service.create_thread()
    updated = service.post_message(second_thread.thread_id, "my billing account number is 6001")
    question_facts = [
        question.expected_fact
        for question in updated.pending_questions
    ]

    assert updated.user_answers == {
        "customer_wants_followup": True,
        "account_name_type": "PERSONAL",
    }
    assert "customer_wants_followup" not in question_facts
    assert "account_name_type" not in question_facts
