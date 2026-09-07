"""SAP Gen AI Hub structured-output wrapper for assessment questionnaire review."""

import json
import os
import time
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from app.models.ai_review import QuestionReviewResult, RetrievalRoundAssessment
from app.services.document_extractors import ExtractedDocument
from app.models.language import DEFAULT_LANGUAGE, normalize_language
from app.observability.llm_usage_logging import (
    LlmUsageContext,
    TokenUsage,
    emit_llm_usage_event,
    extract_token_usage,
)
from app.utils.langgraph.common import (
    ReasoningEffort,
    get_reasoning_effort,
    make_openai_chat_completions_api,
    make_openai_responses_api,
    resolve_openai_chat_completions_api,
    resolve_openai_responses_api,
    validate_reasoning_effort,
)

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)
"""Pydantic structured output model returned by a Gen AI Hub parser."""


REVIEW_INSTRUCTIONS = """Validate assessment questionnaire answer selections using only the provided evidence.
Treat every answer item as one of the questionnaire's minimum guarantees or minimum requirements for maturity, not as an exact ceiling; stronger or more formal evidence satisfies weaker lower-level guarantees when the stronger practice clearly covers the lower requirement.
If evidence proves the organization performs a stronger, more structured, more formal, or more governed version of an answer item, select the lower answer item as satisfied even when the item's literal wording describes a less mature practice. Do not treat the stronger evidence as a contradiction of the weaker guarantee.
Strategic planning example: if evidence shows a structured and formally defined planning process, then an answer item saying the planning process is unstructured and not formally defined is still passed as a lower minimum guarantee; select the lower planning item and explain that the organization exceeds that minimum.
Return level-grouped decisions, evidence references, and reasoning for the question.
For each level_result.level_reasoning, merge reasoning and evidence into one user-facing paragraph instead of writing a separate evidence list. The paragraph must explain each selected, low_confidence, unsupported, and unclear answer item at that level.
When citing evidence, name the file name and any available page, section, sheet, table, row, column, or email header/body context. Treat DOCX/XLSX/EML extraction block IDs, for example docx-block-0004, xlsx-block-0001, or eml-block-0002, as internal navigation labels only: use them to find the cited content, but never cite, quote, paraphrase, or reproduce them in user-facing rationale, level_reasoning, warnings, status explanations, or evidence snippets. For image evidence, also cite the relevant image region and visible text, labels, charts, forms, diagrams, signs, or other visible content without inventing unreadable details. Quote or summarize the exact snippet/content used, and explain why it supports, contradicts, or fails to support the answer item.
When evidence is missing, explain the missing evidence that prevents selecting an answer item instead of only saying it is unsupported.
Do not emit deselect, unselect, remove, clear, or equivalent answer decisions. If evidence does not satisfy an answer item's minimum guarantee, mark it unsupported or unclear and explain the missing evidence.
Use low_confidence only for a minimal implied prerequisite: if stronger higher-level evidence supports a later-level answer but earlier-level dependencies are not directly evidenced, select only the prerequisite answer items logically implied by that stronger evidence and mark those decisions low_confidence. Do not use low_confidence to select every prior-level item.
Apply the cumulative maturity dependency strictly: if you select any answer at level X, where X is greater than 1, there must be at least one selectable answer in every lower level 1 through X-1. Selectable means decision keep_selected, select, or low_confidence.
When a higher-level answer is supported but one or more lower levels have no directly supported answer, choose the minimum non-contradicted lower-level answer item that best bridges the cumulative maturity dependency and mark it low_confidence. Explain that it is selected because the higher-level evidence implies the prerequisite maturity, while direct evidence is missing.
Do not mark an answer low_confidence when it is directly contradicted by evidence. If every candidate in a required lower level is directly contradicted by evidence, do not select the higher-level answer; mark the higher-level answer unclear or unsupported and explain that the dependency chain cannot be satisfied.
User-facing text hygiene: Do not include answer item IDs, evidence IDs, DOCX/XLSX/EML extraction block IDs, schema field names, enum values, or raw field names in rationale, level_reasoning, warnings, status explanations, or evidence snippets. Do not mention internal prompt rules, cumulative maturity dependency, minimum non-contradicted bridge, dependency chain, selectable answer, or JSON/schema policy in user-facing text. Explain these cases in plain business language, for example as a cautious prerequisite inference from stronger evidence with direct evidence missing.
Do not invent evidence, document content, selected answers, or unsupported maturity levels."""

RAG_ASSESSMENT_INSTRUCTIONS = """Assess retrieved evidence chunks for one assessment questionnaire question.
Do not choose final questionnaire answers yet.
Classify retrieved chunks as support, contradiction, background_only, unrelated, or incomplete_but_useful.
Put support, contradiction, or incomplete_but_useful chunks that may affect the final review in accepted_evidence.
Put background_only or unrelated chunks in rejected_evidence.
Do not put rejected chunks into final-review evidence.
Identify remaining evidence gaps in plain business language.
Generate refined retrieval queries only for gaps that could change the final answer decision.
Set sufficient_for_final_review to true only when the accepted evidence and explicit missing-evidence notes are enough to produce a grounded final review.
Set stop_reason to the short code sufficient_evidence when sufficient_for_final_review is true.
Set stop_reason to the short code needs_more_evidence when more retrieval is needed.
Use only supported short codes in stop_reason: sufficient_evidence or needs_more_evidence. Do not write prose, sentences, explanations, or rationale in stop_reason.
Do not invent evidence or infer facts beyond the retrieved chunks."""
"""Instructions for one RAG evidence assessment round."""

EXCEL_SUMMARY_INSTRUCTIONS = """Summarize one extracted Excel workbook for retrieval.
Use the supplied sheet names, ranges, row labels, formulas, and cell values only.
Return a compact business summary that helps semantic search find relevant workbook areas.
Do not invent facts, controls, owners, dates, or metrics not present in the extracted workbook text."""
"""Instructions for generating one spreadsheet summary retrieval chunk."""


class SpreadsheetSummaryResult(BaseModel):
    """Structured output for an Excel workbook summary.

    Inputs:
        summary: Compact grounded summary of the extracted workbook.

    Outputs:
        Validated summary object returned by the active model parser.
    """

    summary: str


LANGUAGE_REVIEW_INSTRUCTIONS = {
    "en": (
        "Write all user-facing rationale, level reasoning, evidence snippets, "
        "warnings, and status explanations in English. In structured fields "
        "only, keep schema field names, answer item IDs, evidence IDs, and "
        "enum values unchanged; do not include them in user-facing text."
    ),
    "it": (
        "Write all user-facing rationale, level reasoning, evidence snippets, "
        "warnings, and status explanations in Italian. In structured fields "
        "only, keep schema field names, answer item IDs, evidence IDs, and "
        "enum values unchanged; do not include them in user-facing text."
    ),
}
"""Language-specific instructions appended to every structured review call."""

CONTEXT_LIMIT_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window",
    "maximum context",
    "input exceeds",
    "payload too large",
    "request too large",
    "file too large",
)
"""Provider error markers that mean the task can retry through RAG."""


class ReviewClientError(RuntimeError):
    """Base error raised when the Gen AI review client cannot complete a call.

    Inputs:
        message: User-facing and loggable error message.
        error_code: Stable machine-readable failure category.
        original_error: Original provider or SDK exception.

    Outputs:
        Exception object with a stable ``error_code`` used by routing logic.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "review_model_failed",
        original_error: Exception | None = None,
    ) -> None:
        """Initialize the typed review client error.

        Inputs:
            message: Human-readable error detail.
            error_code: Stable failure category.
            original_error: Optional provider exception.

        Outputs:
            None. The exception stores error metadata for callers.
        """
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


class ReviewContextLimitError(ReviewClientError):
    """Error raised when a review request exceeds context or payload limits.

    Inputs:
        message: Human-readable context-limit detail.
        original_error: Original provider or SDK exception.

    Outputs:
        Exception object that callers can catch to trigger RAG fallback.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize the context-limit error.

        Inputs:
            message: Human-readable context-limit detail.
            original_error: Optional provider exception.

        Outputs:
            None. The exception carries ``error_code='context_limit'``.
        """
        super().__init__(
            message=message,
            error_code="context_limit",
            original_error=original_error,
        )


def is_context_limit_error(error: Exception) -> bool:
    """Return whether an exception represents a context or payload limit.

    Inputs:
        error: Provider or SDK exception raised by a Gen AI Hub model API.

    Outputs:
        bool: ``True`` when the error message includes a recognized limit marker.
    """
    error_text = str(error).lower()
    return any(marker in error_text for marker in CONTEXT_LIMIT_ERROR_MARKERS)


def review_instructions_for_language(language: str | None) -> str:
    """Build model instructions for the requested review output language.

    Inputs:
        language: Supported language code submitted with the review job.

    Outputs:
        str: Base review instructions plus language-specific generation rules.
    """

    normalized_language = normalize_language(language)
    return "\n".join(
        [
            REVIEW_INSTRUCTIONS,
            LANGUAGE_REVIEW_INSTRUCTIONS[normalized_language],
        ]
    )


def _responses_api_from_openai_module(openai_module: Any) -> Any | None:
    """Resolve a Responses API object from supported Gen AI Hub SDK shapes.

    Inputs:
        openai_module: Imported ``gen_ai_hub.proxy.native.openai`` module or a
            test double with equivalent attributes.

    Outputs:
        Any | None: Object exposing callable ``parse`` when found, otherwise
        ``None``. Current SDK versions may expose the Responses API either as a
        module-level ``responses`` object or as ``OpenAI().responses``.
    """

    return resolve_openai_responses_api(openai_module)


def _chat_completions_api_from_openai_module(openai_module: Any) -> Any | None:
    """Resolve a Chat Completions API object from supported SDK shapes.

    Inputs:
        openai_module: Imported ``gen_ai_hub.proxy.native.openai`` module or a
            test double with equivalent attributes.

    Outputs:
        Any | None: Object exposing callable ``parse`` when found, otherwise
        ``None``.
    """

    return resolve_openai_chat_completions_api(openai_module)


def _default_responses_api() -> Any:
    """Import and resolve the default SAP Gen AI Hub Responses API object.

    Inputs:
        None. Credentials are loaded from the environment before import.

    Outputs:
        Any: Resolved Responses API object exposing callable ``parse``.

    Raises:
        RuntimeError: Raised when the Gen AI Hub SDK is not installed or does
        not expose a usable Responses API implementation.
    """

    return make_openai_responses_api()


def _default_chat_completions_api() -> Any:
    """Import and resolve the default SAP Gen AI Hub Chat Completions object.

    Inputs:
        None. Credentials are loaded from the environment before import.

    Outputs:
        Any: Resolved Chat Completions object exposing callable ``parse``.

    Raises:
        RuntimeError: Raised when the Gen AI Hub SDK is not installed or does
        not expose a usable Chat Completions parser.
    """

    return make_openai_chat_completions_api()


def _model_uses_chat_completions(model: str) -> bool:
    """Return whether a model should use Chat Completions instead of Responses.

    Inputs:
        model: OpenAI-compatible model name configured for review calls.

    Outputs:
        bool: ``True`` for GPT-4 family model names such as ``gpt-4.1`` and
        ``gpt-4o``. GPT-5 and newer models keep the Responses API path.
    """

    return model.strip().lower().startswith("gpt-4")


def _responses_content_to_chat_content(content: Any) -> Any:
    """Convert Responses-style content blocks into Chat Completions content.

    Inputs:
        content: Message content using either a plain string or Responses API
            content blocks such as ``input_text`` and ``input_image``.

    Outputs:
        Any: Chat Completions-compatible message content. Text blocks become
        ``text`` blocks and image data URLs become ``image_url`` blocks.

    Raises:
        ValueError: Raised when raw ``input_file`` blocks are encountered
        because the Chat Completions fallback cannot safely attach PDFs.
    """

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    chat_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            chat_blocks.append({"type": "text", "text": str(block)})
            continue

        block_type = block.get("type")
        if block_type == "input_text":
            chat_blocks.append({"type": "text", "text": str(block.get("text", ""))})
            continue

        if block_type == "input_image":
            image_url = block.get("image_url")
            if not image_url:
                raise ValueError("Responses input_image block is missing image_url.")
            chat_blocks.append(
                {"type": "image_url", "image_url": {"url": str(image_url)}}
            )
            continue

        if block_type == "text":
            chat_blocks.append({"type": "text", "text": str(block.get("text", ""))})
            continue

        if block_type == "image_url":
            chat_blocks.append(block)
            continue

        if block_type == "input_file":
            filename = block.get("filename", "attached file")
            raise ValueError(
                "Chat Completions fallback cannot process Responses input_file "
                f"block for {filename!r}. Use text/RAG evidence or a GPT-5 "
                "Responses model for direct file review."
            )

        chat_blocks.append(
            {
                "type": "text",
                "text": json.dumps(block, ensure_ascii=True),
            }
        )
    return chat_blocks


def _responses_input_to_chat_messages(
    instructions: str,
    input_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build Chat Completions messages from Responses API input messages.

    Inputs:
        instructions: System/developer instructions originally sent through the
            Responses API ``instructions`` field.
        input_messages: Responses API input messages for the review task.

    Outputs:
        list[dict[str, Any]]: Chat Completions messages with the instructions
        as a leading system message and converted user content blocks.
    """

    chat_messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions}
    ]
    for message in input_messages:
        chat_messages.append(
            {
                "role": message.get("role", "user"),
                "content": _responses_content_to_chat_content(
                    message.get("content", "")
                ),
            }
        )
    return chat_messages


def _parsed_from_chat_completion(
    completion: Any,
    text_format: type[StructuredOutputT],
) -> StructuredOutputT:
    """Extract and validate the parsed object from a chat completion response.

    Inputs:
        completion: Object returned by ``chat.completions.parse``.
        text_format: Pydantic model expected in the parsed chat message.

    Outputs:
        StructuredOutputT: Parsed and validated Pydantic output.

    Raises:
        ValueError: Raised when the chat parser response does not include a
        parseable first choice.
    """

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("Chat Completions response did not include choices.")

    message = getattr(choices[0], "message", None)
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, text_format):
            return parsed
        return text_format.model_validate(parsed)

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return text_format.model_validate_json(content)

    raise ValueError(
        "Chat Completions parse response did not include parsed message content."
    )


class GenAiReviewClient:
    """Client for structured assessment question reviews through SAP Gen AI Hub.

    Inputs:
        model: Optional OpenAI-compatible model name to call through SAP Gen AI
            Hub. When omitted, ``GENAI_REVIEW_MODEL`` is read from the
            environment and defaults to ``gpt-5.4``.
        responses_api: Optional Responses API object used for dependency
            injection in tests.
        chat_completions_api: Optional Chat Completions API object used for
            GPT-4-family structured output fallback.

    Outputs:
        A reusable client that returns validated ``QuestionReviewResult``
        instances from structured Responses or Chat Completions calls.
    """

    def __init__(
        self,
        model: str | None = None,
        responses_api: Any | None = None,
        chat_completions_api: Any | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the review client.

        Inputs:
            model: Optional OpenAI-compatible model name. When omitted,
                ``GENAI_REVIEW_MODEL`` is read from the environment and defaults
                to ``gpt-5.4``.
            responses_api: Optional object with a ``parse`` method. When omitted,
                the SAP Gen AI Hub OpenAI-compatible Responses API is imported.
            chat_completions_api: Optional object with a ``parse`` method used
                when ``model`` is in the GPT-4 family.
            reasoning_effort: Optional Responses API reasoning effort. When
                omitted, ``GENAI_REASONING_EFFORT`` is read from the environment
                and defaults to ``low``. This setting is only validated and
                sent for the Responses API route.

        Outputs:
            None. The client stores the model and the active parser dependency.

        Raises:
            RuntimeError: If the selected SAP Gen AI Hub parser is unavailable
                when no test double is injected.
        """
        load_dotenv()
        self.model = model or os.getenv("GENAI_REVIEW_MODEL", "gpt-5.4")
        self.uses_chat_completions = _model_uses_chat_completions(self.model)
        self.responses_api: Any | None = None
        self.chat_completions_api: Any | None = None
        self.reasoning_effort: ReasoningEffort | None = None
        if self.uses_chat_completions:
            if chat_completions_api is None:
                chat_completions_api = _default_chat_completions_api()

            if chat_completions_api is None or not callable(
                getattr(chat_completions_api, "parse", None)
            ):
                raise RuntimeError(
                    "SAP Gen AI Hub Chat Completions API parse unavailable. "
                    "The resolved chat_completions_api must not be None and "
                    "must expose a callable parse method."
                )
            self.chat_completions_api = chat_completions_api
        else:
            if responses_api is None:
                responses_api = _default_responses_api()

            if responses_api is None or not callable(
                getattr(responses_api, "parse", None)
            ):
                raise RuntimeError(
                    "SAP Gen AI Hub Responses API parse unavailable. The "
                    "resolved responses_api must not be None and must expose a "
                    "callable parse method."
                )
            self.responses_api = responses_api
            self.reasoning_effort = (
                validate_reasoning_effort(reasoning_effort)
                if reasoning_effort is not None
                else get_reasoning_effort()
            )

    def _api_name(self) -> str:
        """Return the active SAP Gen AI Hub API name for error messages.

        Inputs:
            None. The method reads the model routing selected at construction.

        Outputs:
            str: Human-readable API name used in wrapped provider errors.
        """

        if self.uses_chat_completions:
            return "SAP Gen AI Hub Chat Completions API"
        return "SAP Gen AI Hub Responses API"

    def _parse_structured_output(
        self,
        instructions: str,
        input_messages: list[dict[str, Any]],
        text_format: type[StructuredOutputT],
        operation: str = "structured_output",
        usage_context: LlmUsageContext | None = None,
    ) -> StructuredOutputT:
        """Parse one structured model response through the active API surface.

        Inputs:
            instructions: System instructions for the review task.
            input_messages: Responses-style input messages built by the review
                graph and RAG code.
            text_format: Pydantic model expected in the structured response.
            operation: Short operation name included in the usage event route.
            usage_context: Optional caller-provided observability metadata. When
                omitted, the existing document-processing route is retained.

        Outputs:
            StructuredOutputT: Parsed and validated model output.
        """

        started_at = time.perf_counter()
        usage = TokenUsage()
        context = usage_context or LlmUsageContext(
            route=f"worker:document-processing:{operation}",
            actor_type="batch",
        )
        llm_endpoint = (
            "chat.completions" if self.uses_chat_completions else "responses"
        )
        if self.uses_chat_completions:
            if self.chat_completions_api is None:
                raise RuntimeError("Chat Completions API dependency is missing.")
            try:
                completion = self.chat_completions_api.parse(
                    model=self.model,
                    messages=_responses_input_to_chat_messages(
                        instructions=instructions,
                        input_messages=input_messages,
                    ),
                    response_format=text_format,
                )
                parsed = _parsed_from_chat_completion(completion, text_format)
                usage = extract_token_usage(completion)
                emit_llm_usage_event(
                    route=context.route,
                    method=context.method,
                    actor_type=context.actor_type,
                    model=self.model,
                    llm_endpoint=llm_endpoint,
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    cache_write_input_tokens=usage.cache_write_input_tokens,
                    input_total_tokens=usage.input_total_tokens,
                    total_tokens=usage.total_tokens,
                    output_tokens=usage.output_tokens,
                    outcome="success",
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    correlation_id=context.correlation_id,
                )
                return parsed
            except Exception:
                emit_llm_usage_event(
                    route=context.route,
                    method=context.method,
                    actor_type=context.actor_type,
                    model=self.model,
                    llm_endpoint=llm_endpoint,
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    cache_write_input_tokens=usage.cache_write_input_tokens,
                    input_total_tokens=usage.input_total_tokens,
                    total_tokens=usage.total_tokens,
                    output_tokens=usage.output_tokens,
                    outcome="error",
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    correlation_id=context.correlation_id,
                )
                raise

        if self.responses_api is None:
            raise RuntimeError("Responses API dependency is missing.")
        if self.reasoning_effort is None:
            raise RuntimeError("Responses API reasoning effort is missing.")
        try:
            response = self.responses_api.parse(
                model=self.model,
                instructions=instructions,
                input=input_messages,
                text_format=text_format,
                reasoning={"effort": self.reasoning_effort},
            )
            parsed = response.output_parsed
            usage = extract_token_usage(response)
            emit_llm_usage_event(
                route=context.route,
                method=context.method,
                actor_type=context.actor_type,
                model=self.model,
                llm_endpoint=llm_endpoint,
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                input_total_tokens=usage.input_total_tokens,
                total_tokens=usage.total_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                correlation_id=context.correlation_id,
            )
            return parsed
        except Exception:
            emit_llm_usage_event(
                route=context.route,
                method=context.method,
                actor_type=context.actor_type,
                model=self.model,
                llm_endpoint=llm_endpoint,
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                input_total_tokens=usage.input_total_tokens,
                total_tokens=usage.total_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                correlation_id=context.correlation_id,
            )
            raise

    def parse_structured_output(
        self,
        instructions: str,
        input_messages: list[dict[str, Any]],
        text_format: type[StructuredOutputT],
        operation: str = "structured_output",
        usage_context: LlmUsageContext | None = None,
    ) -> StructuredOutputT:
        """Return a validated structured result for a non-review AI workflow.

        Inputs:
            instructions: System instructions governing the model response.
            input_messages: Responses-style messages containing task inputs.
            text_format: Pydantic schema required from the model.
            operation: Short operation name used by default token logging.
            usage_context: Optional route and correlation metadata supplied by
                the caller, such as an assessment report job identifier.

        Outputs:
            StructuredOutputT: Provider output validated against ``text_format``.
        """

        return self._parse_structured_output(
            instructions=instructions,
            input_messages=input_messages,
            text_format=text_format,
            operation=operation,
            usage_context=usage_context,
        )

    def review_question(
        self,
        input_messages: list[dict[str, Any]],
        language: str = DEFAULT_LANGUAGE,
    ) -> QuestionReviewResult:
        """Review one assessment questionnaire question with structured output.

        Inputs:
            input_messages: Responses-style input messages containing the question,
                current answer selections, and evidence to evaluate.
            language: Supported language code for generated user-facing text.

        Outputs:
            ``QuestionReviewResult`` parsed by the active structured-output API.

        Raises:
            ReviewContextLimitError: If the review request exceeds the model
                context window or request payload limit.
            ReviewClientError: If the real model API call fails, including
                missing SAP Gen AI Hub credentials or provider configuration.
        """
        try:
            return self._parse_structured_output(
                instructions=review_instructions_for_language(language),
                input_messages=input_messages,
                text_format=QuestionReviewResult,
                operation="review_question",
            )
        except Exception as exc:
            if is_context_limit_error(exc):
                raise ReviewContextLimitError(
                    f"{self._api_name()} question review exceeded "
                    "the model context window or request payload limit. The "
                    "task can be retried through automatic evidence RAG.",
                    original_error=exc,
                ) from exc
            raise ReviewClientError(
                f"{self._api_name()} question review failed. "
                "Verify Gen AI Hub credentials, resource group, deployment, and "
                "network access before retrying.",
                error_code="review_model_failed",
                original_error=exc,
            ) from exc

    def assess_retrieved_evidence(
        self,
        input_messages: list[dict[str, Any]],
        language: str = DEFAULT_LANGUAGE,
    ) -> RetrievalRoundAssessment:
        """Assess retrieved RAG evidence before final question review.

        Inputs:
            input_messages: Responses-style input messages containing the
                question, answer items, candidate chunks, and previous gaps.
            language: Supported language code for generated assessment text.

        Outputs:
            RetrievalRoundAssessment: Structured relevance and sufficiency
            assessment for the current retrieval round.

        Raises:
            ReviewClientError: Raised for non-context provider failures.
            ReviewContextLimitError: Raised for context or payload limit failures.
        """
        instructions = "\n".join(
            [
                RAG_ASSESSMENT_INSTRUCTIONS,
                LANGUAGE_REVIEW_INSTRUCTIONS[normalize_language(language)],
            ]
        )
        try:
            return self._parse_structured_output(
                instructions=instructions,
                input_messages=input_messages,
                text_format=RetrievalRoundAssessment,
                operation="assess_retrieved_evidence",
            )
        except Exception as exc:
            if is_context_limit_error(exc):
                raise ReviewContextLimitError(
                    f"{self._api_name()} RAG evidence assessment exceeded "
                    "the model context window or request payload limit.",
                    original_error=exc,
                ) from exc
            raise ReviewClientError(
                f"{self._api_name()} RAG evidence assessment failed. "
                "Verify Gen AI Hub credentials, resource group, deployment, and "
                "network access before retrying.",
                error_code="review_model_failed",
                original_error=exc,
            ) from exc

    def summarize_excel_document(
        self,
        document: ExtractedDocument,
        language: str = DEFAULT_LANGUAGE,
    ) -> str:
        """Generate a compact retrieval summary for one extracted workbook.

        Inputs:
            document: Extracted XLSX/XLSM evidence document.
            language: Supported language code for the summary text.

        Outputs:
            str: Summary text used as an additional batch retrieval chunk.

        Raises:
            ReviewClientError: Raised when the summary model call fails.
            ReviewContextLimitError: Raised for context or payload limit failures.
        """

        workbook_text = "\n\n".join(
            [
                f"Workbook file: {document.file_name}",
                *[
                    "\n".join(
                        [
                            f"Block: {block.block_id}",
                            f"Sheet: {block.sheet_name or 'unknown'}",
                            f"Range: {block.table_name or 'unknown'}",
                            "Text:",
                            block.text,
                        ]
                    )
                    for block in document.blocks
                ],
            ]
        )
        instructions = "\n".join(
            [
                EXCEL_SUMMARY_INSTRUCTIONS,
                LANGUAGE_REVIEW_INSTRUCTIONS[normalize_language(language)],
            ]
        )
        try:
            response = self._parse_structured_output(
                instructions=instructions,
                input_messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": workbook_text,
                            }
                        ],
                    }
                ],
                text_format=SpreadsheetSummaryResult,
                operation="summarize_excel_document",
            )
        except Exception as exc:
            if is_context_limit_error(exc):
                raise ReviewContextLimitError(
                    f"{self._api_name()} Excel summary exceeded the "
                    "model context window or request payload limit.",
                    original_error=exc,
                ) from exc
            raise ReviewClientError(
                f"{self._api_name()} Excel summary failed. "
                "Verify Gen AI Hub credentials, resource group, deployment, and "
                "network access before retrying.",
                error_code="excel_summary_failed",
                original_error=exc,
            ) from exc
        return response.summary
