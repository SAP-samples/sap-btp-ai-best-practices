from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm_usage_logging import LlmUsageContext, emit_llm_usage_for_response
from .model_clients import make_genai_hub_chat_model, should_use_openai_responses_api
from .token_usage import TokenUsageTracker


COMPLETION_SYSTEM_PROMPT = """Complete supplier price-change draft rows from the canonical extraction and S/4 context only.

Rules:
- Return exactly one item for each input item_index.
- S/4 context is authoritative for supplier, material, original price, currency, and UOM when present.
- Use canonical extraction for requested new price and dates only when it is explicit and consistent with the S/4 context.
- When price_context_candidates are present, select a candidate only when the context clearly supports it for that item.
- Leave any field null when the supplied context has no decisive option.
- List unresolved or ambiguous fields in unresolved_fields and explain the issue in notes.
- Do not approve, post, or finalize price changes. This stage only prepares draft rows for later human review.
- Return the strict CompletionResponse schema without markdown.
"""


class CompletionDecision(BaseModel):
    """One item-indexed Stage 2 completion decision for a draft price-change row."""

    item_index: int
    supplier_id: str | None = None
    supplier_name: str | None = None
    supplier_email: str | None = None
    material_number: str | None = None
    material_description: str | None = None
    original_price: str | None = None
    requested_new_price: str | None = None
    currency: str | None = None
    uom: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    confidence: float = 0.0
    notes: str = ""
    unresolved_fields: list[str] = Field(default_factory=list)


class CompletionResponse(BaseModel):
    """Structured Stage 2 completion response containing one decision per input item."""

    items: list[CompletionDecision] = Field(default_factory=list)


def coerce_completion_response(value: Any) -> CompletionResponse:
    """Normalize provider responses into a validated CompletionResponse.

    Args:
        value: CompletionResponse, dictionary, JSON string, parsed/raw wrapper,
            or message-like object with JSON string ``content``.

    Returns:
        Validated CompletionResponse.

    Raises:
        ValueError: If the response type cannot be converted to the schema.
    """
    if isinstance(value, CompletionResponse):
        return value

    if isinstance(value, dict) and "parsed" in value:
        parsed = value.get("parsed")
        if parsed is not None:
            return coerce_completion_response(parsed)
        raw = value.get("raw")
        if raw is not None:
            return coerce_completion_response(raw)

    parsed_attr = getattr(value, "parsed", None)
    if parsed_attr is not None:
        return coerce_completion_response(parsed_attr)

    output_parsed = getattr(value, "output_parsed", None)
    if output_parsed is not None:
        return coerce_completion_response(output_parsed)

    if isinstance(value, dict):
        return CompletionResponse.model_validate(value)

    content = getattr(value, "content", value)
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Completion response content is not valid JSON.") from exc
        return coerce_completion_response(payload)

    output_text = getattr(value, "output_text", None)
    if isinstance(output_text, str):
        return coerce_completion_response(output_text)

    raise ValueError(f"Unsupported completion response type: {type(value).__name__}")


class CompletionModelClient:
    """Adapter for Stage 2 batched completion over Gen AI Hub model clients."""

    def __init__(
        self,
        model_name: str,
        reasoning_effort: str = "low",
        client: Any | None = None,
    ) -> None:
        """Create a completion model adapter.

        Args:
            model_name: Gen AI Hub deployment name used for completion.
            reasoning_effort: OpenAI Responses reasoning effort for GPT-5-or-newer models.
            client: Optional fake or custom client exposing ``invoke(payload)``.

        Returns:
            None.
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.client = client

    def complete(
        self,
        payload: dict[str, Any],
        token_usage_tracker: TokenUsageTracker | None = None,
        gmail_message_id: str | None = None,
        extraction_id: str | None = None,
        llm_usage_context: LlmUsageContext | None = None,
    ) -> CompletionResponse:
        """Complete a batch of canonical extraction rows with supplied S/4 context.

        Args:
            payload: JSON-serializable completion input containing item-indexed rows and context.
            token_usage_tracker: Optional tracker that records provider usage metadata.
            gmail_message_id: Gmail message associated with the completion batch.
            extraction_id: Persisted extraction id associated with the completion batch.
            llm_usage_context: Optional Cloud Logging request context.

        Returns:
            Validated CompletionResponse with one decision per input item.
        """
        llm_endpoint = "responses" if should_use_openai_responses_api(self.model_name) else "chat.completions"
        started_at = time.perf_counter()
        response: Any | None = None
        try:
            response = self._invoke(payload)
            emit_llm_usage_for_response(
                context=llm_usage_context,
                model=self.model_name,
                llm_endpoint=llm_endpoint,
                response=response,
                outcome="success",
                started_at=started_at,
            )
        except Exception:
            emit_llm_usage_for_response(
                context=llm_usage_context,
                model=self.model_name,
                llm_endpoint=llm_endpoint,
                response=response,
                outcome="error",
                started_at=started_at,
            )
            raise
        if token_usage_tracker is not None:
            token_usage_tracker.record_response(
                stage="completion",
                model_name=self.model_name,
                response=response,
                gmail_message_id=gmail_message_id,
                extraction_id=extraction_id,
            )
        return coerce_completion_response(response)

    def _invoke(self, payload: dict[str, Any]) -> Any:
        """Invoke the configured completion model path.

        Args:
            payload: JSON-serializable completion input.

        Returns:
            Raw provider response, structured-output wrapper, or fake-client response.
        """
        if self.client is not None:
            return self.client.invoke(payload)

        user_payload = json.dumps(payload, ensure_ascii=True, default=str)
        if should_use_openai_responses_api(self.model_name):
            from gen_ai_hub.proxy.native.openai import responses

            return responses.parse(
                model=self.model_name,
                instructions=COMPLETION_SYSTEM_PROMPT,
                input=user_payload,
                text_format=CompletionResponse,
                reasoning={"effort": self.reasoning_effort},
            )

        chat_model = make_genai_hub_chat_model(self.model_name)
        structured_output_kwargs = {
            "method": "json_schema",
            "schema": CompletionResponse,
            "strict": True,
        }
        try:
            structured_llm = chat_model.with_structured_output(
                **structured_output_kwargs,
                include_raw=True,
            )
        except TypeError:
            structured_llm = chat_model.with_structured_output(**structured_output_kwargs)
        return structured_llm.invoke(
            [
                SystemMessage(content=COMPLETION_SYSTEM_PROMPT),
                HumanMessage(content=user_payload),
            ]
        )
