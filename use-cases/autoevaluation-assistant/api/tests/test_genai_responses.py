"""Tests for the SAP Gen AI Hub Responses API review client wrapper."""

import json
from types import SimpleNamespace

from pydantic import BaseModel

from app.models.ai_review import QuestionReviewResult
from app.observability.llm_usage_logging import LlmUsageContext
from app.services.genai_responses import (
    GenAiReviewClient,
    ReviewClientError,
    ReviewContextLimitError,
    _chat_completions_api_from_openai_module,
    _responses_api_from_openai_module,
    review_instructions_for_language,
)
import pytest


class FakeResponse:
    """Fake parsed Responses API response.

    Inputs:
        output_parsed: Structured model result that the client should return.

    Outputs:
        An object shaped like the parsed SAP Gen AI Hub Responses API response.
    """

    def __init__(self, output_parsed: object) -> None:
        """Store the structured parsed response returned by the fake API.

        Inputs:
            output_parsed: Structured review result to expose on the response.

        Outputs:
            None. The value is assigned to ``self.output_parsed``.
        """
        self.output_parsed = output_parsed
        self.usage = SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
        )


class FakeResponses:
    """Fake Responses API object that records parse call arguments.

    Inputs:
        None. The fake creates a deterministic ``QuestionReviewResult``.

    Outputs:
        A test double with ``last_kwargs`` populated after ``parse`` is called.
    """

    def __init__(self) -> None:
        """Create an empty fake with no recorded call arguments.

        Inputs:
            None.

        Outputs:
            None. Initializes ``last_kwargs`` for later assertions.
        """
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs: object) -> FakeResponse:
        """Record parse arguments and return a structured review result.

        Inputs:
            **kwargs: Arguments passed by ``GenAiReviewClient.review_question``.

        Outputs:
            ``FakeResponse`` carrying a deterministic ``QuestionReviewResult``.
        """
        self.last_kwargs = kwargs
        return FakeResponse(
            output_parsed=QuestionReviewResult(
                question_id="Q-001",
                model="gpt-5.4",
                overall_status="supported",
                highest_supported_level=1,
                current_selected_answer_item_ids=["Q-001-L1-1"],
                verified_selected_answer_item_ids=["Q-001-L1-1"],
            )
        )


class StructuredOutputFixture(BaseModel):
    """Minimal structured-output model used by the generic client test."""

    text: str


class FakeStructuredResponses:
    """Fake Responses API returning one generic structured result."""
    def __init__(self) -> None:
        """Initialize the parse-call recorder.

        Inputs:
            None.

        Outputs:
            None. ``last_kwargs`` starts empty.
        """
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs: object) -> FakeResponse:
        """Return a deterministic report draft.

        Inputs:
            **kwargs: Arguments passed by the public structured-output method.

        Outputs:
            FakeResponse: Parsed generic structured result.
        """
        self.last_kwargs = kwargs
        return FakeResponse(
            output_parsed=StructuredOutputFixture(
                text="The assessment is progressing."
            )
        )


class FakeChatCompletions:
    """Fake Chat Completions API object that records parse arguments.

    Inputs:
        None. The fake creates a deterministic ``QuestionReviewResult``.

    Outputs:
        A test double shaped like ``chat.completions`` with ``last_kwargs``
        populated after ``parse`` is called.
    """

    def __init__(self) -> None:
        """Create an empty fake with no recorded parse arguments.

        Inputs:
            None.

        Outputs:
            None. Initializes ``last_kwargs`` for later assertions.
        """
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs: object) -> SimpleNamespace:
        """Record parse arguments and return a parsed chat completion.

        Inputs:
            **kwargs: Arguments passed by ``GenAiReviewClient.review_question``.

        Outputs:
            SimpleNamespace: Chat completion-shaped object whose first choice
            carries a parsed ``QuestionReviewResult``.
        """
        self.last_kwargs = kwargs
        parsed_result = QuestionReviewResult(
            question_id="Q-001",
            model="gpt-4.1",
            overall_status="supported",
            highest_supported_level=1,
            current_selected_answer_item_ids=["Q-001-L1-1"],
            verified_selected_answer_item_ids=["Q-001-L1-1"],
        )
        return SimpleNamespace(
            usage={
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 100,
            },
            choices=[
                SimpleNamespace(message=SimpleNamespace(parsed=parsed_result))
            ]
        )


class FakeRagAssessmentResponses:
    """Fake Responses API returning a RAG assessment payload."""

    def __init__(self) -> None:
        """Initialize an empty call recorder."""
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs: object) -> FakeResponse:
        """Return a deterministic retrieval assessment.

        Inputs:
            **kwargs: Parse arguments from the client.

        Outputs:
            FakeResponse: Parsed retrieval assessment payload.
        """
        from app.models.ai_review import RetrievalRoundAssessment

        self.last_kwargs = kwargs
        return FakeResponse(
            output_parsed=RetrievalRoundAssessment(
                accepted_evidence=[],
                rejected_evidence=[
                    {
                        "source_chunk_id": "chunk-2",
                        "relevance": "background_only",
                        "related_answer_item_ids": [],
                        "rationale": "This chunk provides general context only.",
                        "confidence": 0.67,
                    }
                ],
                evidence_gaps=["Need board approval evidence."],
                refined_queries=["board approval strategic objectives"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            )
        )


class FailingResponses:
    """Fake Responses API object that raises from parse.

    Inputs:
        None. The fake always raises the same provider-like error.

    Outputs:
        A test double that exercises ``review_question`` error handling.
    """

    def parse(self, **kwargs: object) -> FakeResponse:
        """Raise a deterministic parse failure.

        Inputs:
            **kwargs: Arguments passed by ``GenAiReviewClient.review_question``.

        Outputs:
            None. This method always raises ``ValueError``.

        Raises:
            ValueError: Always raised to mimic a provider failure.
        """
        raise ValueError("model deployment unavailable")


class ContextLimitResponses:
    """Fake Responses API object that raises a context-window provider error.

    Inputs:
        None. The fake always raises a provider-shaped context error.

    Outputs:
        A test double that exercises context-limit error classification.
    """

    def parse(self, **kwargs: object) -> FakeResponse:
        """Raise a deterministic context-window failure.

        Inputs:
            **kwargs: Arguments passed by ``GenAiReviewClient.review_question``.

        Outputs:
            None. This method always raises ``ValueError``.

        Raises:
            ValueError: Always raised with a context-limit marker.
        """
        raise ValueError(
            "{'error': {'code': 'context_length_exceeded', "
            "'message': 'Your input exceeds the context window.'}}"
        )


def test_review_client_uses_structured_output() -> None:
    """Verify review calls use structured output and low reasoning effort.

    Inputs:
        None. The test injects ``FakeResponses`` and sends sample input
        messages.

    Outputs:
        None. Assertions confirm the returned result and parse call arguments.
    """
    fake = FakeResponses()
    client = GenAiReviewClient(model="gpt-5.4", responses_api=fake)
    input_messages = [
        {
            "role": "user",
            "content": "Review Q-001 using the attached policy evidence.",
        }
    ]

    result = client.review_question(input_messages)

    assert result.question_id == "Q-001"
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["text_format"] is QuestionReviewResult
    assert fake.last_kwargs["model"] == "gpt-5.4"
    assert fake.last_kwargs["reasoning"] == {"effort": "low"}


def test_public_structured_output_uses_supplied_usage_context(capsys) -> None:
    """Verify reusable structured calls retain their caller's usage context.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions cover output validation and report-specific logging.
    """
    fake = FakeStructuredResponses()
    client = GenAiReviewClient(model="gpt-5.4", responses_api=fake)
    usage_context = LlmUsageContext(
        route="worker:assessment-report:generate",
        method="POST",
        actor_type="batch",
        correlation_id="report-job-1",
    )

    result = client.parse_structured_output(
        instructions="Return the requested report section.",
        input_messages=[{"role": "user", "content": "Assessment snapshot"}],
        text_format=StructuredOutputFixture,
        operation="structured_fixture",
        usage_context=usage_context,
    )

    assert result.text == "The assessment is progressing."
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["text_format"] is StructuredOutputFixture
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "worker:assessment-report:generate"
    assert event["correlation_id"] == "report-job-1"


def test_review_client_emits_responses_usage_event(capsys) -> None:
    """Verify Responses review calls emit one token usage event.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions confirm token usage and endpoint metadata.
    """

    client = GenAiReviewClient(model="gpt-5.4", responses_api=FakeResponses())

    client.review_question(
        [
            {
                "role": "user",
                "content": "Review Q-001 using the attached policy evidence.",
            }
        ]
    )

    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["event_type"] == "llm_usage"
    assert event["route"] == "worker:document-processing:review_question"
    assert event["model"] == "gpt-5.4"
    assert event["llm_endpoint"] == "responses"
    assert event["input_tokens"] == 120
    assert event["output_tokens"] == 30
    assert event["outcome"] == "success"


def test_review_client_uses_chat_completions_for_gpt4_structured_output() -> None:
    """Verify GPT-4 review models use Chat Completions structured parsing.

    Inputs:
        None. The test injects ``FakeChatCompletions`` and sends a
        Responses-style text block.

    Outputs:
        None. Assertions confirm the final review result is parsed through the
        Chat Completions API without Responses-only reasoning parameters.
    """
    fake = FakeChatCompletions()
    client = GenAiReviewClient(model="gpt-4.1", chat_completions_api=fake)

    result = client.review_question(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Review Q-001 using accepted RAG evidence.",
                    }
                ],
            }
        ]
    )

    assert result.question_id == "Q-001"
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["model"] == "gpt-4.1"
    assert fake.last_kwargs["response_format"] is QuestionReviewResult
    assert "reasoning" not in fake.last_kwargs
    messages = fake.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "Validate assessment questionnaire" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [
        {
            "type": "text",
            "text": "Review Q-001 using accepted RAG evidence.",
        }
    ]


def test_review_client_emits_chat_completions_usage_event(capsys) -> None:
    """Verify Chat Completions review calls emit one token usage event.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions confirm chat endpoint usage is normalized.
    """

    client = GenAiReviewClient(
        model="gpt-4.1",
        chat_completions_api=FakeChatCompletions(),
    )

    client.review_question(
        [
            {
                "role": "user",
                "content": "Review Q-001 using accepted RAG evidence.",
            }
        ]
    )

    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "worker:document-processing:review_question"
    assert event["model"] == "gpt-4.1"
    assert event["llm_endpoint"] == "chat.completions"
    assert event["input_tokens"] == 80
    assert event["output_tokens"] == 20
    assert event["outcome"] == "success"


def test_review_client_ignores_reasoning_effort_for_gpt4_chat_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid Responses reasoning config does not block GPT-4 chat.

    Inputs:
        monkeypatch: Pytest helper used to isolate environment variables.

    Outputs:
        None. Assertions confirm the GPT-4 route can parse structured output
        without reading or sending Responses-only reasoning parameters.
    """
    monkeypatch.setenv("GENAI_REASONING_EFFORT", "slow")
    fake = FakeChatCompletions()
    client = GenAiReviewClient(model="gpt-4.1", chat_completions_api=fake)

    result = client.review_question(
        [
            {
                "role": "user",
                "content": "Review Q-001 using accepted RAG evidence.",
            }
        ]
    )

    assert result.model == "gpt-4.1"
    assert fake.last_kwargs is not None
    assert "reasoning" not in fake.last_kwargs


def test_review_client_reads_configured_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the final review model can be configured without code changes.

    Inputs:
        monkeypatch: Pytest helper used to isolate environment variables.

    Outputs:
        None. Assertions confirm the client uses the configured
        ``GENAI_REVIEW_MODEL`` for both GPT-5 Responses and GPT-4 Chat
        Completions routes.
    """

    monkeypatch.setenv("GENAI_REVIEW_MODEL", "gpt-5.4")
    default_client = GenAiReviewClient(responses_api=FakeResponses())
    assert default_client.model == "gpt-5.4"

    monkeypatch.setenv("GENAI_REVIEW_MODEL", "gpt-4.1")
    configured_client = GenAiReviewClient(chat_completions_api=FakeChatCompletions())
    assert configured_client.model == "gpt-4.1"


def test_review_client_assesses_retrieved_evidence_with_structured_output() -> None:
    """Verify RAG evidence assessment uses structured Responses parsing."""
    from app.models.ai_review import RetrievalRoundAssessment

    fake = FakeRagAssessmentResponses()
    client = GenAiReviewClient(model="gpt-5.4", responses_api=fake)

    assessment = client.assess_retrieved_evidence(
        input_messages=[
            {
                "role": "user",
                "content": "Assess retrieved chunks for Q.STR.02.01.",
            }
        ],
        language="en",
    )

    assert assessment.evidence_gaps == ["Need board approval evidence."]
    assert assessment.rejected_evidence[0].relevance == "background_only"
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["text_format"] is RetrievalRoundAssessment
    instructions = fake.last_kwargs["instructions"]
    assert "stop_reason" in instructions
    assert "sufficient_evidence" in instructions
    assert "needs_more_evidence" in instructions
    assert "Do not write prose" in instructions


def test_review_client_uses_configured_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ``GENAI_REASONING_EFFORT`` controls review call reasoning.

    Inputs:
        monkeypatch: Pytest helper used to isolate environment variables.

    Outputs:
        None. Assertions confirm the Responses API request uses the configured
        effort value instead of hard-coding it in the client.
    """
    monkeypatch.setenv("GENAI_REASONING_EFFORT", "none")
    fake = FakeResponses()
    client = GenAiReviewClient(model="gpt-5.4", responses_api=fake)

    client.review_question(
        [
            {
                "role": "user",
                "content": "Review Q-001 using the attached policy evidence.",
            }
        ]
    )

    assert fake.last_kwargs is not None
    assert fake.last_kwargs["reasoning"] == {"effort": "none"}


def test_review_client_adds_italian_generation_instructions() -> None:
    """Verify Italian jobs ask the model for Italian user-facing text.

    Inputs:
        None. The test injects ``FakeResponses`` and requests Italian output.

    Outputs:
        None. Assertions confirm only the natural-language result text is
        localized while structured IDs and enum values remain stable.
    """
    fake = FakeResponses()
    client = GenAiReviewClient(model="gpt-5.4", responses_api=fake)

    client.review_question(
        [
            {
                "role": "user",
                "content": "Review Q-001 using the attached policy evidence.",
            }
        ],
        language="it",
    )

    assert fake.last_kwargs is not None
    instructions = fake.last_kwargs["instructions"]
    assert "Italian" in instructions
    assert "structured fields" in instructions
    assert "unchanged" in instructions
    assert "user-facing text" in instructions


def test_review_instructions_require_evidence_rich_reasoning() -> None:
    """Verify review instructions require merged reasoning and evidence detail.

    Inputs:
        None. The test reads the English instruction text sent to the model.

    Outputs:
        None. Assertions confirm the prompt contract covers document-specific
        reasoning, missing evidence, and low-confidence prerequisite selections.
    """
    instructions = review_instructions_for_language("en")

    assert "merge reasoning and evidence" in instructions
    assert "file name" in instructions
    assert (
        "page, section, sheet, table, row, column, or email header/body"
        in instructions
    )
    assert "DOCX/XLSX/EML extraction block IDs" in instructions
    assert "docx-block-0004" in instructions
    assert "xlsx-block-0001" in instructions
    assert "eml-block-0002" in instructions
    assert "image" in instructions
    assert "missing evidence" in instructions
    assert "low_confidence" in instructions
    assert "minimal implied prerequisite" in instructions
    assert "at least one selectable answer in every lower level" in instructions
    assert "cumulative maturity dependency" in instructions
    assert "directly contradicted by evidence" in instructions
    assert "Do not include answer item IDs" in instructions
    assert "Do not mention internal prompt rules" in instructions
    assert "structured fields only" in instructions


def test_review_instructions_use_guarantee_based_selection() -> None:
    """Verify review instructions treat answers as minimum guarantees.

    Inputs:
        None. The test reads the English instruction text sent to the model.

    Outputs:
        None. Assertions confirm the prompt tells the model that stronger
        evidence satisfies weaker answer guarantees and forbids unselect output.
    """
    instructions = review_instructions_for_language("en")

    assert "minimum guarantees" in instructions
    assert "stronger or more formal evidence satisfies weaker" in instructions
    assert "Do not emit deselect" in instructions
    assert "unselect" in instructions
    assert "structured and formally defined planning process" in instructions
    assert "unstructured and not formally defined" in instructions
    assert "select the lower planning item" in instructions


def test_review_client_rejects_invalid_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid reasoning effort configuration fails clearly.

    Inputs:
        monkeypatch: Pytest helper used to isolate environment variables.

    Outputs:
        None. Assertions confirm invalid configuration is not silently ignored.
    """
    monkeypatch.setenv("GENAI_REASONING_EFFORT", "slow")

    with pytest.raises(ValueError) as error:
        GenAiReviewClient(model="gpt-5.4", responses_api=FakeResponses())

    assert "GENAI_REASONING_EFFORT" in str(error.value)


def test_review_client_rejects_responses_api_without_parse() -> None:
    """Verify initialization fails when the Responses API cannot parse.

    Inputs:
        None. The test injects a plain object without a ``parse`` method.

    Outputs:
        None. Assertions confirm a clear ``RuntimeError`` is raised before any
        review call can use the invalid dependency.
    """
    with pytest.raises(RuntimeError) as error:
        GenAiReviewClient(model="gpt-5.4", responses_api=object())

    assert "SAP Gen AI Hub Responses API" in str(error.value)
    assert "parse" in str(error.value)
    assert "unavailable" in str(error.value)


def test_review_client_rejects_chat_completions_api_without_parse() -> None:
    """Verify GPT-4 initialization fails without chat structured parsing.

    Inputs:
        None. The test injects a plain object without a ``parse`` method.

    Outputs:
        None. Assertions confirm a clear ``RuntimeError`` is raised before GPT-4
        review calls can use the invalid Chat Completions dependency.
    """
    with pytest.raises(RuntimeError) as error:
        GenAiReviewClient(model="gpt-4.1", chat_completions_api=object())

    assert "SAP Gen AI Hub Chat Completions API" in str(error.value)
    assert "parse" in str(error.value)
    assert "unavailable" in str(error.value)


def test_responses_api_resolver_supports_client_responses_shape() -> None:
    """Verify the wrapper supports SDKs exposing Responses on OpenAI clients.

    Inputs:
        None. The test supplies a fake OpenAI module whose module-level
        ``responses`` value is unavailable but whose ``OpenAI().responses``
        object exposes ``parse``.

    Outputs:
        None. Assertions confirm the client-scoped Responses object is used.
    """
    fake_responses = FakeResponses()
    fake_module = SimpleNamespace(
        responses=None,
        OpenAI=lambda: SimpleNamespace(responses=fake_responses),
    )

    assert _responses_api_from_openai_module(fake_module) is fake_responses


def test_chat_completions_api_resolver_supports_client_chat_shape() -> None:
    """Verify the wrapper supports SDKs exposing chat on OpenAI clients.

    Inputs:
        None. The test supplies a fake OpenAI module whose module-level chat
        value is unavailable but whose ``OpenAI().chat.completions`` object
        exposes ``parse``.

    Outputs:
        None. Assertions confirm the client-scoped Chat Completions object is
        used for GPT-4 structured output fallback.
    """
    fake_chat_completions = FakeChatCompletions()
    fake_module = SimpleNamespace(
        chat=None,
        OpenAI=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=fake_chat_completions)
        ),
    )

    assert (
        _chat_completions_api_from_openai_module(fake_module)
        is fake_chat_completions
    )


def test_responses_api_resolver_prefers_module_responses_shape() -> None:
    """Verify the wrapper keeps supporting module-level Responses objects.

    Inputs:
        None. The test supplies both SDK shapes.

    Outputs:
        None. Assertions confirm the module-level object wins when it exposes
        ``parse``.
    """
    module_responses = FakeResponses()
    client_responses = FakeResponses()
    fake_module = SimpleNamespace(
        responses=module_responses,
        OpenAI=lambda: SimpleNamespace(responses=client_responses),
    )

    assert _responses_api_from_openai_module(fake_module) is module_responses


def test_chat_completions_api_resolver_prefers_module_chat_shape() -> None:
    """Verify the wrapper keeps supporting module-level chat objects.

    Inputs:
        None. The test supplies both SDK shapes.

    Outputs:
        None. Assertions confirm the module-level Chat Completions object wins
        when it exposes ``parse``.
    """
    module_chat_completions = FakeChatCompletions()
    client_chat_completions = FakeChatCompletions()
    fake_module = SimpleNamespace(
        chat=SimpleNamespace(completions=module_chat_completions),
        OpenAI=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=client_chat_completions)
        ),
    )

    assert (
        _chat_completions_api_from_openai_module(fake_module)
        is module_chat_completions
    )


def test_review_question_chains_parse_failures() -> None:
    """Verify provider parse failures are wrapped with their original cause.

    Inputs:
        None. The test injects a fake Responses API whose ``parse`` raises.

    Outputs:
        None. Assertions confirm the wrapper raises ``RuntimeError`` and keeps
        the original ``ValueError`` as ``__cause__``.
    """
    client = GenAiReviewClient(model="gpt-5.4", responses_api=FailingResponses())

    with pytest.raises(RuntimeError) as error:
        client.review_question(
            [
                {
                    "role": "user",
                    "content": "Review Q-001 with evidence.",
                }
            ]
        )

    assert "SAP Gen AI Hub Responses API question review failed" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "model deployment unavailable"


def test_review_client_emits_error_usage_event(capsys) -> None:
    """Verify provider errors still emit an error token usage event.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions confirm error outcome and zero-token fallback.
    """

    client = GenAiReviewClient(model="gpt-5.4", responses_api=FailingResponses())

    with pytest.raises(ReviewClientError):
        client.review_question(
            [
                {
                    "role": "user",
                    "content": "Review Q-001 with evidence.",
                }
            ]
        )

    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "worker:document-processing:review_question"
    assert event["model"] == "gpt-5.4"
    assert event["llm_endpoint"] == "responses"
    assert event["input_tokens"] == 0
    assert event["output_tokens"] == 0
    assert event["outcome"] == "error"


def test_review_client_raises_context_limit_error_for_context_failures() -> None:
    """Verify provider context-window failures remain machine-readable.

    Inputs:
        None. The test injects a fake Responses API that raises a
        ``context_length_exceeded`` provider error.

    Outputs:
        None. Assertions confirm a typed ``ReviewContextLimitError`` is raised.
    """
    client = GenAiReviewClient(model="gpt-5.4", responses_api=ContextLimitResponses())

    with pytest.raises(ReviewContextLimitError) as error:
        client.review_question(
            [
                {
                    "role": "user",
                    "content": "Review Q-001 using oversized evidence.",
                }
            ]
        )

    assert error.value.error_code == "context_limit"
    assert "context window" in str(error.value)


def test_review_client_assess_retrieved_evidence_raises_context_limit_error() -> None:
    """Verify RAG evidence assessment preserves typed context-limit failures.

    Inputs:
        None. The test injects a fake Responses API that raises a
        ``context_length_exceeded`` provider error.

    Outputs:
        None. Assertions confirm a typed ``ReviewContextLimitError`` is raised.
    """
    client = GenAiReviewClient(model="gpt-5.4", responses_api=ContextLimitResponses())

    with pytest.raises(ReviewContextLimitError) as error:
        client.assess_retrieved_evidence(
            [
                {
                    "role": "user",
                    "content": "Assess oversized retrieved evidence.",
                }
            ]
        )

    assert error.value.error_code == "context_limit"
    assert "context window" in str(error.value)


def test_review_client_raises_generic_client_error_for_non_context_failures() -> None:
    """Verify non-context provider failures do not trigger RAG fallback.

    Inputs:
        None. The test injects the existing fake Responses API that raises a
        deployment-style failure.

    Outputs:
        None. Assertions confirm the raised error is not a context-limit error.
    """
    client = GenAiReviewClient(model="gpt-5.4", responses_api=FailingResponses())

    with pytest.raises(ReviewClientError) as error:
        client.review_question(
            [
                {
                    "role": "user",
                    "content": "Review Q-001 using policy evidence.",
                }
            ]
        )

    assert not isinstance(error.value, ReviewContextLimitError)
    assert error.value.error_code == "review_model_failed"
