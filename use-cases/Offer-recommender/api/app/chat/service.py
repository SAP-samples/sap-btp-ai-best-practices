from __future__ import annotations

import os
import re
import sqlite3
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

from app.models.nbo import ChatMessageResponse, ChatThreadStateResponse, serialize_question, serialize_recommendation
from app.nbo.config import OUTPUT_DIR
from app.nbo.models import WorkflowStage
from app.services.recommendations import RecommendationService
from app.services.saved_answers import SavedCustomerAnswerService, saved_answer_service

logger = logging.getLogger(__name__)

TokenUsageLogger = Callable[[dict[str, Any]], None]


class ChatGraphState(TypedDict, total=False):
    thread_id: str
    input_action: str
    input_message: str
    declined_program_id: str | None
    billing_account: str | None
    customer_type: str | None
    messages: list[dict[str, str]]
    user_answers: dict[str, Any]
    pending_questions: list[dict[str, Any]]
    current_question: dict[str, Any] | None
    decision_result: dict[str, Any] | None
    previous_decision_result: dict[str, Any] | None
    declined_programs: list[str]
    declined_offer_display_name: str | None
    status_phase: str
    assistant_reply: str | None
    parse_error: str | None
    parsed_answer: Any


class ChatThreadNotFoundError(LookupError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Chat thread '{thread_id}' was not found")
        self.thread_id = thread_id


class ChatInvalidDeclineError(ValueError):
    def __init__(self, program_id: str) -> None:
        super().__init__(f"Program '{program_id}' is not currently eligible for this chat thread")
        self.program_id = program_id


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag_enabled(name: str) -> bool:
    """Return whether an environment variable contains a truthy feature flag value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _append_message(messages: list[dict[str, str]], role: str, content: str) -> list[dict[str, str]]:
    updated = list(messages)
    updated.append({"role": role, "content": content, "timestamp": _timestamp()})
    return updated


def _default_state(thread_id: str) -> ChatGraphState:
    return {
        "thread_id": thread_id,
        "billing_account": None,
        "customer_type": None,
        "messages": [],
        "user_answers": {},
        "pending_questions": [],
        "current_question": None,
        "decision_result": None,
        "previous_decision_result": None,
        "declined_programs": [],
        "declined_offer_display_name": None,
        "status_phase": "awaiting_account",
        "assistant_reply": None,
        "parse_error": None,
        "parsed_answer": None,
    }


def _normalize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _normalize_checkpoint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_checkpoint_value(item) for item in value]
    return value


def _stringify_answer(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if value is None:
        return "Not sure"
    return str(value)


def _match_answer_option_deterministic(question: dict[str, Any], message: str) -> tuple[bool, Any]:
    answer_options = question.get("answer_options", [])
    normalized = message.strip().lower()
    has_nullable_option = any(option["value"] is None for option in answer_options)

    for index, option in enumerate(answer_options, start=1):
        if normalized == str(index):
            return True, option["value"]
        if normalized == option["label"].strip().lower():
            return True, option["value"]

        value = option["value"]
        if value is True and normalized in {"yes", "y", "true", "si", "sí"}:
            return True, True
        if value is False and normalized in {"no", "n", "false"}:
            return True, False
        if value is None and normalized in {
            "not sure",
            "prefer not to answer",
            "not sure / prefer not to answer",
            "don't know",
            "dont know",
            "prefer not",
            "skip",
            "unknown",
        }:
            return True, None

    if has_nullable_option and normalized in {
        "not sure",
        "prefer not to answer",
        "not sure / prefer not to answer",
        "don't know",
        "dont know",
        "prefer not",
        "skip",
        "unknown",
    }:
        return True, None

    return False, None


def _answer_label_for_value(question: dict[str, Any], value: Any) -> str | None:
    """Return the display label for a selected answer option."""
    for option in question.get("answer_options", []):
        if option.get("value") == value:
            return option.get("label")
    return None


def _llm_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return str(content).strip()


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping view for dict-like metadata values."""
    if isinstance(value, Mapping):
        return value
    return None


def _iter_llm_usage_sources(response: Any) -> list[Mapping[str, Any]]:
    """Collect usage metadata containers from common LangChain response shapes."""
    sources: list[Mapping[str, Any]] = []
    response_mapping = _as_mapping(response)
    if response_mapping is not None:
        sources.append(response_mapping)

    for attr_name in ("usage_metadata", "usage", "response_metadata"):
        attr_value = getattr(response, attr_name, None)
        attr_mapping = _as_mapping(attr_value)
        if attr_mapping is not None:
            sources.append(attr_mapping)

    # LangChain provider wrappers commonly place provider-native usage inside
    # response_metadata.token_usage, while newer responses may expose usage.
    for source in list(sources):
        for nested_key in ("token_usage", "usage", "usage_metadata"):
            nested_mapping = _as_mapping(source.get(nested_key))
            if nested_mapping is not None:
                sources.append(nested_mapping)

    return sources


def _first_token_value(
    sources: list[Mapping[str, Any]],
    candidate_keys: tuple[str, ...],
) -> int | None:
    """Find the first integer token count for any of the candidate metadata keys."""
    for source in sources:
        for key in candidate_keys:
            value = source.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
    return None


def _extract_llm_token_usage(response: Any) -> dict[str, int | None]:
    """Normalize input, output, and total token counts from an LLM response.

    Args:
        response: LangChain response object or provider response dictionary.

    Returns:
        A dictionary with input, output, and total token counts when supplied by
        the provider. Missing fields are returned as ``None``.
    """
    sources = _iter_llm_usage_sources(response)
    input_tokens = _first_token_value(
        sources,
        ("input_tokens", "prompt_tokens", "input_token_count"),
    )
    output_tokens = _first_token_value(
        sources,
        ("output_tokens", "completion_tokens", "output_token_count"),
    )
    total_tokens = _first_token_value(sources, ("total_tokens", "total_token_count"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _log_token_usage_event(event: dict[str, Any]) -> None:
    """Write a structured chat answer-classifier token usage event to application logs.

    Args:
        event: Token usage event produced by the answer-classifier logging hook.

    Returns:
        None.
    """
    logger.info(
        "chat_answer_classifier_token_usage provider=%s model=%s input_tokens=%s output_tokens=%s total_tokens=%s",
        event.get("provider"),
        event.get("model"),
        event.get("input_tokens"),
        event.get("output_tokens"),
        event.get("total_tokens"),
    )


def _offer_name_list(offers: list[dict[str, Any]]) -> str:
    names = [offer["display_name"] for offer in offers]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


class ChatWorkflowService:
    """Persistent LangGraph-backed chat workflow using deterministic evaluation."""

    def __init__(
        self,
        db_path: Path | None = None,
        recommendations: RecommendationService | None = None,
        saved_answers: SavedCustomerAnswerService | None = None,
        answer_classifier_token_logger: TokenUsageLogger | None = None,
    ) -> None:
        """Initialize a persistent chat workflow service.

        Args:
            db_path: Optional SQLite checkpoint path for chat thread state.
            recommendations: Optional recommendation service dependency.
            saved_answers: Optional saved-answer service dependency.
            answer_classifier_token_logger: Optional callback that receives
                structured token usage events for answer-classifier LLM calls.

        Returns:
            None.
        """
        self.recommendations = recommendations or RecommendationService()
        self.saved_answers = saved_answers or saved_answer_service
        self.answer_classifier_token_logger = answer_classifier_token_logger
        if (
            self.answer_classifier_token_logger is None
            and _env_flag_enabled("CHAT_ANSWER_CLASSIFIER_LOG_TOKEN_USAGE")
        ):
            self.answer_classifier_token_logger = _log_token_usage_event
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (OUTPUT_DIR / "chat_threads.sqlite")
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = RLock()
        self._answer_classifier_llm: Any | None = None
        self._answer_classifier_llm_initialized = False
        self.checkpointer = SqliteSaver(self.conn)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)
        graph.add_node("initialize_state", self._initialize_state)
        graph.add_node("route_input", self._route_input)
        graph.add_node("capture_account", self._capture_account)
        graph.add_node("capture_answer", self._capture_answer)
        graph.add_node("decline_offer", self._decline_offer)
        graph.add_node("evaluate_account", self._evaluate_account)
        graph.add_node("respond", self._respond)

        graph.add_edge(START, "initialize_state")
        graph.add_edge("initialize_state", "route_input")
        graph.add_conditional_edges(
            "route_input",
            self._route_after_input,
            {
                "respond": "respond",
                "capture_account": "capture_account",
                "capture_answer": "capture_answer",
                "decline_offer": "decline_offer",
                "evaluate_account": "evaluate_account",
            },
        )
        graph.add_conditional_edges(
            "capture_account",
            self._route_after_account_capture,
            {
                "respond": "respond",
                "evaluate_account": "evaluate_account",
            },
        )
        graph.add_conditional_edges(
            "capture_answer",
            self._route_after_answer_capture,
            {
                "respond": "respond",
                "evaluate_account": "evaluate_account",
            },
        )
        graph.add_edge("decline_offer", "evaluate_account")
        graph.add_edge("evaluate_account", "respond")
        graph.add_edge("respond", END)

        return graph.compile(checkpointer=self.checkpointer)

    def _initialize_state(self, state: ChatGraphState) -> ChatGraphState:
        thread_id = state.get("thread_id") or str(uuid4())
        base = _default_state(thread_id)
        merged = dict(base)
        merged.update({k: v for k, v in state.items() if v is not None})
        return merged

    def _route_input(self, state: ChatGraphState) -> ChatGraphState:
        action = state.get("input_action") or "initialize"
        message = (state.get("input_message") or "").strip()
        if (
            action == "message"
            and message
            and state.get("billing_account")
            and not state.get("current_question")
            and state.get("decision_result")
        ):
            extracted = self.recommendations.extract_account_from_text(message)
            current_account = state.get("billing_account")
            if not extracted or extracted == current_account:
                messages = _append_message(state.get("messages", []), "user", message)
                return {
                    "messages": messages,
                    "assistant_reply": self._format_terminal_reply(
                        state["decision_result"]
                    ),
                }
        return state

    def _route_after_input(self, state: ChatGraphState) -> str:
        action = state.get("input_action") or "initialize"
        message = (state.get("input_message") or "").strip()
        current_question = state.get("current_question")

        if state.get("assistant_reply"):
            return "respond"
        if action == "initialize":
            return "respond"
        if action == "decline":
            return "decline_offer"
        if message:
            extracted = self.recommendations.extract_account_from_text(message)
            current_account = state.get("billing_account")
            if extracted and extracted != current_account:
                return "capture_account"
        if not state.get("billing_account"):
            return "capture_account"
        if current_question:
            return "capture_answer"
        return "evaluate_account"

    def _capture_account(self, state: ChatGraphState) -> ChatGraphState:
        message = state.get("input_message") or ""
        matched = self.recommendations.extract_account_from_text(message)
        if not matched:
            return {
                "parse_error": "account_not_found",
                "assistant_reply": (
                    "I couldn't match that billing account number to the Customer Offer Advisor data. "
                    "Please verify the account number and try again."
                ),
                "status_phase": "awaiting_account",
            }

        messages = state.get("messages", [])
        if message:
            messages = _append_message(messages, "user", message)

        return {
            "billing_account": matched,
            "customer_type": None,
            "messages": messages,
            "user_answers": self.saved_answers.get_answer_values(matched),
            "pending_questions": [],
            "current_question": None,
            "decision_result": None,
            "declined_programs": [],
            "status_phase": "evaluating",
            "parse_error": None,
        }

    def _route_after_account_capture(self, state: ChatGraphState) -> str:
        if state.get("parse_error"):
            return "respond"
        return "evaluate_account"

    def _capture_answer(self, state: ChatGraphState) -> ChatGraphState:
        message = state.get("input_message") or ""
        question = state.get("current_question")
        messages = state.get("messages", [])
        if message:
            messages = _append_message(messages, "user", message)

        if not question:
            return {"messages": messages}

        matched, value = self._match_answer_option(question, message)
        if not matched:
            options = ", ".join(option["label"] for option in question.get("answer_options", []))
            return {
                "messages": messages,
                "assistant_reply": f"Please answer using one of the available options: {options}",
                "status_phase": "questioning",
            }

        updated_answers = dict(state.get("user_answers", {}))
        updated_answers[question["expected_fact"]] = value
        billing_account = state.get("billing_account")
        if billing_account:
            self.saved_answers.save_answer(
                billing_account,
                question["expected_fact"],
                value,
                answer_label=_answer_label_for_value(question, value),
                question_prompt=question.get("prompt"),
                source_surface="chat",
            )
        return {
            "messages": messages,
            "user_answers": updated_answers,
            "status_phase": "evaluating",
            "parsed_answer": value,
        }

    def _route_after_answer_capture(self, state: ChatGraphState) -> str:
        if state.get("assistant_reply"):
            return "respond"
        return "evaluate_account"

    def _decline_offer(self, state: ChatGraphState) -> ChatGraphState:
        program_id = state.get("declined_program_id")
        messages = state.get("messages", [])
        offer_name = self._lookup_offer_display_name(state.get("decision_result"), program_id)
        eligible_program_ids = {
            offer.get("program_id")
            for offer in (state.get("decision_result") or {}).get("eligible_offers", [])
        }
        if program_id and program_id not in eligible_program_ids:
            raise ChatInvalidDeclineError(program_id)
        if program_id:
            messages = _append_message(
                messages,
                "user",
                f"Not interested in {offer_name or 'that program'}",
            )
        declined = list(state.get("declined_programs", []))
        if program_id and program_id not in declined:
            declined.append(program_id)
        return {
            "messages": messages,
            "declined_programs": declined,
            "declined_offer_display_name": offer_name,
            "status_phase": "evaluating",
        }

    def _evaluate_account(self, state: ChatGraphState) -> ChatGraphState:
        billing_account = state.get("billing_account")
        if not billing_account:
            return {
                "assistant_reply": "Please provide the customer's billing account number to continue.",
                "status_phase": "awaiting_account",
            }

        result = self.recommendations.evaluate_account(
            billing_account,
            user_answers=state.get("user_answers", {}),
            declined_programs=state.get("declined_programs", []),
        )
        serialized = serialize_recommendation(result).model_dump(mode="json")
        pending = [
            serialize_question(question).model_dump(mode="json")
            for question in result.questions
        ]
        current = None
        if not self._chat_blocked_on_system_data(serialized, pending):
            current = next(
                (
                    question
                    for question in pending
                    if question.get("source") == "customer"
                    and question.get("answer_options")
                ),
                None,
            )
        phase = "questioning" if current else "complete"
        return {
            "billing_account": result.billing_account,
            "customer_type": result.customer_type,
            "previous_decision_result": state.get("decision_result"),
            "decision_result": serialized,
            "pending_questions": pending,
            "current_question": current,
            "status_phase": phase,
            "parse_error": None,
        }

    def _respond(self, state: ChatGraphState) -> ChatGraphState:
        messages = list(state.get("messages", []))
        assistant_reply = state.get("assistant_reply")
        if assistant_reply:
            return {
                "messages": _append_message(messages, "assistant", assistant_reply),
                "assistant_reply": None,
            }

        action = state.get("input_action") or "initialize"
        decision_result = state.get("decision_result")
        previous_decision_result = state.get("previous_decision_result")
        current_question = state.get("current_question")

        if action == "initialize" and not messages:
            reply = (
                "Welcome to the Customer Offer Advisor assistant. "
                "Please provide the customer's billing account number."
            )
        elif state.get("parse_error") == "account_not_found":
            reply = (
                "I couldn't match that billing account number to the Customer Offer Advisor data. "
                "Please verify the account number and try again."
            )
        elif action == "decline" and decision_result:
            reply = self._format_decline_update(
                decision_result,
                current_question,
                state.get("declined_offer_display_name"),
            )
        elif current_question:
            reply = self._format_question_turn(
                current_question,
                decision_result,
                previous_decision_result,
            )
        elif decision_result:
            reply = self._format_completion_update(decision_result)
        else:
            reply = "Please provide the customer's billing account number."

        return {
            "messages": _append_message(messages, "assistant", reply),
            "assistant_reply": None,
            "parsed_answer": None,
            "previous_decision_result": None,
            "declined_offer_display_name": None,
        }

    def _match_answer_option(self, question: dict[str, Any], message: str) -> tuple[bool, Any]:
        matched, value = _match_answer_option_deterministic(question, message)
        if matched:
            return matched, value

        if not message.strip() or not question.get("answer_options"):
            return False, None

        return self._match_answer_option_with_llm(question, message)

    def _get_answer_classifier_llm(self) -> Any | None:
        if self._answer_classifier_llm_initialized:
            return self._answer_classifier_llm

        self._answer_classifier_llm_initialized = True
        try:
            from app.utils.langgraph.common import make_llm

            provider = os.getenv("CHAT_ANSWER_CLASSIFIER_PROVIDER", "openai")
            model_name = os.getenv("CHAT_ANSWER_CLASSIFIER_MODEL", "gpt-4o")
            self._answer_classifier_llm = make_llm(
                provider=provider,
                model_name=model_name,
                temperature=0.0,
                max_tokens=32,
            )
        except Exception as exc:
            logger.warning(
                "Chat answer classifier LLM could not be initialized; falling back to strict option matching. provider=%s model=%s error=%s",
                os.getenv("CHAT_ANSWER_CLASSIFIER_PROVIDER", "openai"),
                os.getenv("CHAT_ANSWER_CLASSIFIER_MODEL", "gpt-4o"),
                exc,
            )
            self._answer_classifier_llm = None

        return self._answer_classifier_llm

    def _match_answer_option_with_llm(
        self,
        question: dict[str, Any],
        message: str,
    ) -> tuple[bool, Any]:
        llm = self._get_answer_classifier_llm()
        if llm is None:
            return False, None

        answer_options = question.get("answer_options", [])
        numbered_options = "\n".join(
            f"{index}. {option['label']} (value={option['value']!r})"
            for index, option in enumerate(answer_options, start=1)
        )
        prompt = (
            "You map a user's free-text reply to one of the allowed answer options for a "
            "customer-service chat question.\n"
            "Rules:\n"
            "- Choose only if the user's intent clearly matches one option.\n"
            "- Return UNRESOLVED if the intent is ambiguous.\n"
            "- Choose the uncertainty/refusal option only when the user explicitly expresses "
            "uncertainty, lack of knowledge, skipping, or refusal.\n"
            "- Return exactly one line: MATCH: <number> or MATCH: UNRESOLVED\n\n"
            f"Question: {question['prompt']}\n"
            f"Allowed options:\n{numbered_options}\n\n"
            f"User reply: {message!r}"
        )

        try:
            response = llm.invoke(prompt)
        except Exception:
            return False, None

        self._log_answer_classifier_token_usage(response)

        response_text = _llm_response_text(response)
        normalized = response_text.upper()
        if "UNRESOLVED" in normalized:
            return False, None

        match = re.search(r"MATCH:\s*(\d+)", response_text, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"\b(\d+)\b", response_text)
        if match is None:
            return False, None

        option_index = int(match.group(1)) - 1
        if option_index < 0 or option_index >= len(answer_options):
            return False, None

        return True, answer_options[option_index]["value"]

    def _log_answer_classifier_token_usage(self, response: Any) -> None:
        """Send token usage for a chat answer-classifier LLM response to the optional logger.

        Args:
            response: LLM response returned by the answer-classifier invocation.

        Returns:
            None. Logging failures are swallowed so observability cannot break
            chat answer parsing.
        """
        token_logger = self.answer_classifier_token_logger
        if token_logger is None:
            return

        provider = os.getenv("CHAT_ANSWER_CLASSIFIER_PROVIDER", "openai")
        model_name = os.getenv("CHAT_ANSWER_CLASSIFIER_MODEL", "gpt-4o")
        event = {
            "llm_call": "chat_answer_classifier",
            "provider": provider,
            "model": model_name,
            **_extract_llm_token_usage(response),
        }
        try:
            token_logger(event)
        except Exception as exc:
            logger.warning(
                "Chat answer classifier token logger failed; continuing without token telemetry. error=%s",
                exc,
            )

    def _lookup_offer_display_name(
        self,
        decision_result: dict[str, Any] | None,
        program_id: str | None,
    ) -> str | None:
        if not decision_result or not program_id:
            return None

        final_offer = decision_result.get("final_offer")
        if final_offer and final_offer.get("program_id") == program_id:
            return final_offer.get("display_name")

        for offer in decision_result.get("eligible_offers", []):
            if offer.get("program_id") == program_id:
                return offer.get("display_name")
        return None

    def _additional_offers(
        self,
        decision_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not decision_result:
            return []

        final_offer = decision_result.get("final_offer")
        final_program_id = final_offer.get("program_id") if final_offer else None
        return [
            offer
            for offer in decision_result.get("eligible_offers", [])
            if offer.get("program_id") != final_program_id
        ]

    def _newly_eligible_additional_offers(
        self,
        previous_decision_result: dict[str, Any] | None,
        decision_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        previous_offer_ids = {
            offer.get("program_id")
            for offer in (previous_decision_result or {}).get("eligible_offers", [])
        }
        return [
            offer
            for offer in self._additional_offers(decision_result)
            if offer.get("program_id") not in previous_offer_ids
        ]

    def _chat_blocked_on_system_data(
        self,
        decision_result: dict[str, Any] | None,
        pending_questions: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not decision_result or decision_result.get("final_offer"):
            return False

        pending = pending_questions if pending_questions is not None else decision_result.get("questions", [])
        if not pending:
            return False

        if decision_result.get("workflow_stage") == WorkflowStage.SYSTEM_BLOCKED.value:
            return True

        return not any(
            question.get("source") == "customer" and question.get("answer_options")
            for question in pending
        )

    def _format_options(self, question: dict[str, Any]) -> str:
        options = ", ".join(option["label"] for option in question.get("answer_options", []))
        if not options:
            return ""
        return f" Available options: {options}."

    def _format_question_turn(
        self,
        question: dict[str, Any],
        decision_result: dict[str, Any] | None,
        previous_decision_result: dict[str, Any] | None,
    ) -> str:
        if (
            decision_result
            and decision_result.get("final_offer")
            and question.get("expected_fact") == "customer_wants_followup"
        ):
            return (
                "I found a primary recommendation for this account. "
                "Would you like to answer a few quick questions to check for more programs?"
                + self._format_options(question)
            )

        newly_eligible = self._newly_eligible_additional_offers(
            previous_decision_result,
            decision_result,
        )
        if newly_eligible:
            noun = "offer" if len(newly_eligible) == 1 else "offers"
            intro = f"Newly eligible {noun}: {_offer_name_list(newly_eligible)}. "
        else:
            intro = ""
        if not intro and not decision_result:
            intro = "I still need a little more information. "

        return (
            f"{intro}Next question: {question['prompt']}"
            + self._format_options(question)
        )

    def _format_completion_update(self, decision_result: dict[str, Any]) -> str:
        final_offer = decision_result.get("final_offer")
        additional_offers = self._additional_offers(decision_result)
        workflow_stage = decision_result.get("workflow_stage")
        pending = decision_result.get("questions", [])

        if final_offer:
            if additional_offers:
                noun = "program is" if len(additional_offers) == 1 else "programs are"
                return (
                    "Discovery is complete. "
                    "The primary recommendation remains in place, and "
                    f"{len(additional_offers)} additional {noun} now eligible."
                )
            if workflow_stage == WorkflowStage.PRIMARY_OFFER_FINAL.value:
                return (
                    "Discovery is complete. "
                    "The current primary recommendation remains the best fit for this session."
                )
            return "Discovery is complete for now."

        if pending:
            if self._chat_blocked_on_system_data(decision_result, pending):
                return (
                    "I can’t move this recommendation further with customer answers alone. "
                    "The remaining blockers depend on system or profile data that needs to be updated first."
                )
            noun = "question" if len(pending) == 1 else "questions"
            return (
                "I still need more information before I can recommend a program. "
                f"There {'is' if len(pending) == 1 else 'are'} {len(pending)} {noun} left."
            )
        return "I couldn't identify an eligible program with the information currently available."

    def _format_terminal_reply(self, decision_result: dict[str, Any]) -> str:
        summary = self._format_completion_update(decision_result)
        if self._chat_blocked_on_system_data(decision_result):
            return (
                f"{summary} Enter a different billing account to evaluate another customer, "
                "or update the missing system data before retrying this one."
            )
        return (
            f"{summary} Enter a different billing account to start a new evaluation."
        )

    def _format_decline_update(
        self,
        decision_result: dict[str, Any],
        current_question: dict[str, Any] | None,
        declined_offer_display_name: str | None,
    ) -> str:
        removed_name = declined_offer_display_name or "that program"
        parts = [f"I removed {removed_name} from this session."]

        if decision_result.get("final_offer"):
            parts.append("I kept the best remaining option in view.")
        elif current_question:
            parts.append("I can keep checking for other programs with a few quick questions.")
        else:
            parts.append("There are no other eligible offers ready right now.")

        additional_offers = self._additional_offers(decision_result)
        if additional_offers:
            noun = "offer is" if len(additional_offers) == 1 else "offers are"
            parts.append(
                f"Additional eligible {noun} available: {_offer_name_list(additional_offers)}."
            )

        if current_question:
            parts.append(f"Next question: {current_question['prompt']}{self._format_options(current_question)}")

        return " ".join(parts)

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _get_existing_values(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.values:
            raise ChatThreadNotFoundError(thread_id)
        normalized = _normalize_checkpoint_value(snapshot.values)
        if normalized != snapshot.values:
            self.graph.update_state(self._config(thread_id), normalized)
        return normalized

    def _build_thread_state_response(
        self,
        thread_id: str,
        values: dict[str, Any],
    ) -> ChatThreadStateResponse:
        return ChatThreadStateResponse(
            thread_id=thread_id,
            billing_account=values.get("billing_account"),
            customer_type=values.get("customer_type"),
            messages=[
                ChatMessageResponse(**message)
                for message in values.get("messages", [])
            ],
            user_answers=dict(values.get("user_answers", {})),
            pending_questions=[
                serialize_question_dict(question)
                for question in values.get("pending_questions", [])
            ],
            current_question=serialize_question_dict(values["current_question"])
            if values.get("current_question")
            else None,
            decision_result=serialize_recommendation_dict(values["decision_result"])
            if values.get("decision_result")
            else None,
            declined_programs=list(values.get("declined_programs", [])),
            status_phase=values.get("status_phase", "awaiting_account"),
        )

    def create_thread(self) -> ChatThreadStateResponse:
        with self._lock:
            thread_id = str(uuid4())
            self.graph.invoke(
                {
                    "thread_id": thread_id,
                    "input_action": "initialize",
                    "input_message": "",
                },
                config=self._config(thread_id),
            )
            values = self._get_existing_values(thread_id)
            return self._build_thread_state_response(thread_id, values)

    def get_thread(self, thread_id: str) -> ChatThreadStateResponse:
        with self._lock:
            values = self._get_existing_values(thread_id)
            return self._build_thread_state_response(thread_id, values)

    def post_message(self, thread_id: str, message: str) -> ChatThreadStateResponse:
        with self._lock:
            self._get_existing_values(thread_id)
            self.graph.invoke(
                {
                    "thread_id": thread_id,
                    "input_action": "message",
                    "input_message": message,
                },
                config=self._config(thread_id),
            )
            values = self._get_existing_values(thread_id)
            return self._build_thread_state_response(thread_id, values)

    def decline_program(self, thread_id: str, program_id: str) -> ChatThreadStateResponse:
        with self._lock:
            self._get_existing_values(thread_id)
            self.graph.invoke(
                {
                    "thread_id": thread_id,
                    "input_action": "decline",
                    "declined_program_id": program_id,
                },
                config=self._config(thread_id),
            )
            values = self._get_existing_values(thread_id)
            return self._build_thread_state_response(thread_id, values)


def serialize_question_dict(question: dict[str, Any]):
    return serialize_question_dict_model(question)


def serialize_recommendation_dict(recommendation: dict[str, Any]):
    return serialize_recommendation_dict_model(recommendation)


def serialize_question_dict_model(question: dict[str, Any]):
    from app.models.nbo import QuestionResponse

    return QuestionResponse.model_validate(question)


def serialize_recommendation_dict_model(recommendation: dict[str, Any]):
    from app.models.nbo import RecommendationResponse

    return RecommendationResponse.model_validate(recommendation)


chat_workflow_service = ChatWorkflowService()
