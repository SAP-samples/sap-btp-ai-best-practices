"""Native asynchronous SAP Generative AI Hub adapter for ECN impact analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from copy import deepcopy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gen_ai_hub.proxy.native.openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from app.ecn.analysis_models import AnalysisSections, ImpactFactors, ImpactSeverity
from app.ecn.llm.diagnostics import PROVIDER_ERROR_CODES, classify_provider_failure
from app.ecn.llm.prompts import (
    PromptEvidenceProjection,
    build_impact_input,
    build_impact_instructions,
)
from app.ecn.llm.staged import EXECUTIVE_SUMMARY_PATTERN, StagePrompt
from app.ecn.llm.two_pass import (
    EvidencePlan,
    FinalEvidenceProjection,
    PlannerPromptProjection,
    build_final_input,
    build_final_instructions,
    build_planner_input,
    build_planner_instructions,
)
from app.ecn.settings import EcnSettings, LLM_TIMEOUT_FALLBACK_MODEL
from app.observability import TokenUsage, extract_token_usage
from app.persistence.records import PinnedGenerationConfig


class ProviderCallStatus(StrEnum):
    """Discriminate returned provider responses from safe adapter errors."""

    RECEIVED = "RECEIVED"
    ERROR = "ERROR"


class ProviderCallStage(StrEnum):
    """Identify the bounded model stage represented by one provider call."""

    DISCOVERY_1 = "DISCOVERY_1"
    DISCOVERY_2 = "DISCOVERY_2"
    PLANNER = "PLANNER"
    FINAL = "FINAL"
    MANUFACTURING = "MANUFACTURING"
    PROCUREMENT = "PROCUREMENT"
    CONFIGURATION = "CONFIGURATION"
    DATA_READINESS = "DATA_READINESS"
    SUMMARY = "SUMMARY"


class LegacyAnalysisResponse(BaseModel):
    """Preserve the pre-staged provider contract for already pinned jobs."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sections: AnalysisSections
    impact_factors: ImpactFactors
    overall_impact: ImpactSeverity


@dataclass(frozen=True)
class ProviderCallResult:
    """Retain one provider call's safe output, metadata, usage, and latency.

    Args:
        status: Whether a provider response was received or the call errored.
        raw_json: Structured output text; nullable for incomplete/refused responses.
        response_status: Provider response status such as ``completed``.
        refusal_present: Whether refusal content was present, without retaining its text.
        usage: Availability-preserving normalized provider usage.
        model_name: Model used for this request, which may be the timeout fallback.
        model_deployment: Optional bounded actual model/deployment label.
        provider_request_id: Optional bounded provider response identifier.
        latency_ms: Nonnegative elapsed provider-call time in milliseconds.
        error_code: Fixed safe code for adapter errors; otherwise ``None``.

    Raises:
        ValueError: If the discriminated status or persistence bounds conflict.
    """

    status: ProviderCallStatus
    raw_json: str | None
    response_status: str | None
    refusal_present: bool
    usage: TokenUsage
    model_name: str
    model_deployment: str | None
    provider_request_id: str | None
    latency_ms: int
    error_code: str | None
    stage: ProviderCallStage = ProviderCallStage.FINAL
    http_status: int | None = None
    provider_code: str | None = None
    retry_after_seconds: int | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        """Enforce result discrimination, string bounds, and nonnegative counts."""
        if self.status is ProviderCallStatus.RECEIVED and self.error_code is not None:
            raise ValueError("a received provider result cannot contain an error code")
        if self.status is ProviderCallStatus.ERROR and (
            self.error_code not in PROVIDER_ERROR_CODES
            or self.raw_json is not None
            or self.refusal_present
        ):
            raise ValueError("an error provider result has contradictory fields")
        if not self.model_name or len(self.model_name) > 200:
            raise ValueError("model_name is blank or exceeds its persistence bound")
        for value in (
            self.response_status,
            self.model_deployment,
            self.provider_request_id,
        ):
            if value is not None and (not value or len(value) > 200):
                raise ValueError("optional provider metadata exceeds its bound")
        if self.provider_code is not None and (
            not self.provider_code or len(self.provider_code) > 120
        ):
            raise ValueError("provider_code exceeds its persistence bound")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be nonnegative")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("http_status is outside the HTTP status range")
        if self.retry_after_seconds is not None and not 0 <= self.retry_after_seconds <= 86_400:
            raise ValueError("retry_after_seconds is outside the supported bound")
        if self.status is ProviderCallStatus.RECEIVED and (
            self.http_status is not None
            or self.provider_code is not None
            or self.retry_after_seconds is not None
            or self.retryable
        ):
            raise ValueError("a received provider result cannot contain failure metadata")
        counts = (
            self.usage.input_tokens,
            self.usage.cached_input_tokens,
            self.usage.cache_write_input_tokens,
            self.usage.input_total_tokens,
            self.usage.output_tokens,
            self.usage.total_tokens,
        )
        if any(value is not None and value < 0 for value in counts):
            raise ValueError("token counts must be nonnegative")
        if not self.usage.usage_available and any(
            value is not None for value in counts
        ):
            raise ValueError("unavailable usage cannot contain measured counts")


class EcnAnalysisProvider(Protocol):
    """Define one stage call in the bounded two-pass generation workflow."""

    async def generate(
        self,
        projection: PromptEvidenceProjection | PlannerPromptProjection | FinalEvidenceProjection | StagePrompt,
        config: PinnedGenerationConfig,
        stage: ProviderCallStage = ProviderCallStage.FINAL,
    ) -> ProviderCallResult:
        """Perform one stage request with a single fallback on provider timeout."""

    async def close(self) -> None:
        """Close the long-lived asynchronous client."""


def _bounded_label(value: object, maximum: int = 200) -> str | None:
    """Return a bounded provider label or a deterministic digest marker."""
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    if len(rendered) <= maximum:
        return rendered
    return f"sha256:{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}"


def _read_value(source: object, key: str) -> object | None:
    """Read a key from provider mappings or response object attributes."""
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _contains_refusal(value: object) -> bool:
    """Detect provider refusal structure without copying the refusal text."""
    if isinstance(value, (list, tuple)):
        return any(_contains_refusal(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "refusal" or value.get("refusal") is not None:
            return True
        return any(_contains_refusal(item) for item in value.values())
    if (
        getattr(value, "type", None) == "refusal"
        or getattr(value, "refusal", None) is not None
    ):
        return True
    content = getattr(value, "content", None)
    return _contains_refusal(content) if content is not None else False


def _strict_provider_schema(schema: dict[str, object], *, concise: bool = False) -> dict[str, object]:
    """Make every object property required for OpenAI strict JSON schemas.

    Pydantic omits fields with application-side compatibility defaults from an
    object's ``required`` array. OpenAI strict structured output requires every
    declared property to be required, even when the Pydantic model can fill a
    missing value while reading historical records. It also rejects ``default``
    beside ``$ref``, so compatibility defaults are removed from the provider
    schema. The generated schema is a fresh dictionary, allowing safe in-place
    normalization without mutating model metadata.

    Args:
        schema: Fresh JSON schema returned by ``BaseModel.model_json_schema``.
        concise: Enforce snapshot-v2 bullet bounds during structured generation.

    Returns:
        The same schema with closed objects and complete ``required`` arrays.
    """

    def normalize(value: object) -> None:
        """Recursively normalize nested definitions, arrays, and unions."""
        if isinstance(value, dict):
            value.pop("default", None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["additionalProperties"] = False
                value["required"] = list(properties)
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(schema)
    if concise:
        # Constrain generation itself; a prompt alone can exceed the bullet cap.
        # Keep the persisted string contract and every generated qualification.
        definitions = schema["$defs"]
        executive = deepcopy(definitions["AnalysisSection"])
        executive["properties"]["assessment_markdown"]["pattern"] = r"^- [^\r\n]+(?:\n- [^\r\n]+){0,4}$"
        definitions["AnalysisSections"]["properties"]["executive_summary"] = executive
        definitions["ImpactFactorAssessment"]["properties"]["assessment"]["pattern"] = r"^- [^\r\n]+(?:\n- [^\r\n]+){0,2}$"
    return schema


class NativeImpactProvider:
    """Call SAP Gen AI Hub and retry a timed-out primary once on GPT-5.4."""

    def __init__(
        self,
        client: AsyncOpenAI,
        monotonic: Callable[[], float] = time.monotonic,
        *,
        max_attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 4.0,
        timeout_fallback_model: str = LLM_TIMEOUT_FALLBACK_MODEL,
    ) -> None:
        """Store provider retry policy, client, and latency clock.

        Args:
            client: Long-lived native Gen AI Hub ``AsyncOpenAI`` client.
            monotonic: Injectable monotonic clock used for deterministic tests.
            max_attempts: Bound for primary-model rate-limit and server-error retries.
            sleep: Async delay function used for bounded retry backoff.
            jitter: Random value source used to vary bounded retry delays.
            retry_base_seconds: Initial delay before transient retries.
            retry_max_seconds: Maximum transient retry delay.
            timeout_fallback_model: Model used for one retry after a primary timeout.
        """
        if max_attempts < 1 or max_attempts > 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if retry_base_seconds < 0 or retry_max_seconds < retry_base_seconds:
            raise ValueError("retry delay bounds are invalid")
        if not timeout_fallback_model or len(timeout_fallback_model) > 200:
            raise ValueError("timeout fallback model must be a bounded nonblank name")
        self._client = client
        self._monotonic = monotonic
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._timeout_fallback_model = timeout_fallback_model

    async def generate(
        self,
        projection: PromptEvidenceProjection | PlannerPromptProjection | FinalEvidenceProjection | StagePrompt,
        config: PinnedGenerationConfig,
        stage: ProviderCallStage = ProviderCallStage.FINAL,
    ) -> ProviderCallResult:
        """Perform a strict Responses call, falling back once after a timeout.

        Args:
            projection: Bounded deterministic evidence projection for one ECN.
            config: Immutable job-local model and prompt limits.

        Returns:
            A discriminated received/error result without provider exception text.

        Raises:
            asyncio.CancelledError: When the application timeout or shutdown cancels
                the active native request.
        """
        started = self._monotonic()
        if isinstance(projection, StagePrompt):
            instructions = projection.instructions
            provider_input = [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(projection.payload, ensure_ascii=False, sort_keys=True)}]}]
            schema_name = f"ecn_{stage.value.lower()}_v5"
            schema = _strict_provider_schema(projection.schema.model_json_schema())
            # Constrain IDs at generation time, then retain runtime validation.
            # This prevents the model inventing a source ID for a correct fact.
            for section, ids in (projection.citation_ids_by_section or {}).items():
                if section == "factor":
                    properties = schema["$defs"]["ImpactFactorAssessment"]["properties"]
                else:
                    definition = deepcopy(schema["$defs"]["AnalysisSection"])
                    schema["$defs"]["AnalysisSections"]["properties"][section] = definition
                    properties = definition["properties"]
                if ids:
                    properties["evidence_ids"]["items"]["enum"] = list(ids)
                else:
                    properties["evidence_ids"]["maxItems"] = 0
            if stage in {ProviderCallStage.MANUFACTURING, ProviderCallStage.PROCUREMENT, ProviderCallStage.CONFIGURATION}:
                schema["$defs"]["ImpactFactorAssessment"]["properties"]["assessment"]["pattern"] = r"^- [^\r\n]+(?:\n- [^\r\n]+){0,1}$"
                schema["$defs"]["ImpactFactorAssessment"]["properties"]["assessment"]["maxLength"] = 550
            elif stage is ProviderCallStage.SUMMARY:
                for key in projection.required_gap_sections:
                    section = schema["$defs"]["AnalysisSections"]["properties"][key]
                    if "$ref" in section:
                        section = deepcopy(schema["$defs"]["AnalysisSection"])
                    section["properties"]["status"] = {"type": "string", "enum": ["INSUFFICIENT_EVIDENCE"]}
                    section["properties"]["missing_evidence"]["minItems"] = 1
                    schema["$defs"]["AnalysisSections"]["properties"][key] = section
                executive = schema["$defs"]["AnalysisSections"]["properties"]["executive_summary"]
                if "$ref" in executive:
                    executive = deepcopy(schema["$defs"]["AnalysisSection"])
                executive["properties"]["assessment_markdown"]["pattern"] = EXECUTIVE_SUMMARY_PATTERN
                schema["$defs"]["AnalysisSections"]["properties"]["executive_summary"] = executive
        elif stage is ProviderCallStage.PLANNER:
            if not isinstance(projection, PlannerPromptProjection):
                raise TypeError("planner stage requires PlannerPromptProjection")
            instructions = build_planner_instructions()
            provider_input = build_planner_input(projection)
            schema_name = "ecn_evidence_plan"
            schema = _strict_provider_schema(EvidencePlan.model_json_schema())
        elif isinstance(projection, FinalEvidenceProjection):
            instructions = build_final_instructions()
            provider_input = build_final_input(projection)
            schema_name = "ecn_impact_analysis_v2"
            schema = _strict_provider_schema(LegacyAnalysisResponse.model_json_schema(),
                concise=projection.snapshot_context.get("version") == "ecn-snapshot-v2")
        elif isinstance(projection, PromptEvidenceProjection):
            # Compatibility path for direct adapter contract tests and historical
            # tools; the production generation service uses FinalEvidenceProjection.
            instructions = build_impact_instructions()
            provider_input = build_impact_input(projection)
            schema_name = "ecn_impact_analysis"
            schema = _strict_provider_schema(LegacyAnalysisResponse.model_json_schema())
        else:
            raise TypeError("unsupported provider projection")

        async def call_model(model_name: str) -> object:
            """Send the prepared strict-schema request to one deployment."""
            return await self._client.responses.create(
                model=model_name,
                instructions=instructions,
                input=provider_input,
                timeout=config.request_timeout_seconds,
                store=False,
                max_output_tokens=8000,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )

        response: object | None = None
        model_name = config.model_name
        failure = None
        failure_response: object | None = None
        for attempt_number in range(1, self._max_attempts + 1):
            try:
                response = await call_model(model_name)
                failure = None
                failure_response = None
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize every provider failure.
                failure = classify_provider_failure(exc)
                failure_response = getattr(exc, "response", None)
                if (
                    failure.error_code == "PROVIDER_TIMEOUT"
                    and model_name != self._timeout_fallback_model
                ):
                    # Switch immediately on timeout; do not spend the remaining
                    # retry budget repeating the same slow primary deployment.
                    model_name = self._timeout_fallback_model
                    try:
                        response = await call_model(model_name)
                        failure = None
                        failure_response = None
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as fallback_exc:  # noqa: BLE001 - normalize final failure.
                        failure = classify_provider_failure(fallback_exc)
                        failure_response = getattr(fallback_exc, "response", None)
                        break
                if failure.retryable and attempt_number < self._max_attempts:
                    exponential = self._retry_base_seconds * (2 ** (attempt_number - 1))
                    delay = min(self._retry_max_seconds, exponential)
                    delay += min(delay * 0.25, self._retry_max_seconds - delay) * max(
                        0.0, min(1.0, self._jitter())
                    )
                    if failure.retry_after_seconds is not None:
                        delay = max(delay, min(float(failure.retry_after_seconds), self._retry_max_seconds))
                    await self._sleep(delay)
                    continue
                break
        if failure is not None:
            latency_ms = max(0, round((self._monotonic() - started) * 1_000))
            return ProviderCallResult(
                status=ProviderCallStatus.ERROR,
                raw_json=None,
                response_status=None,
                refusal_present=False,
                usage=extract_token_usage(failure_response),
                model_name=model_name,
                model_deployment=None,
                provider_request_id=failure.request_id,
                latency_ms=latency_ms,
                error_code=failure.error_code,
                stage=stage,
                http_status=failure.http_status,
                provider_code=failure.provider_code,
                retry_after_seconds=failure.retry_after_seconds,
                retryable=failure.retryable,
            )
        if response is None:
            raise RuntimeError("provider retry loop completed without a result")
        latency_ms = max(0, round((self._monotonic() - started) * 1_000))
        usage = extract_token_usage(response)
        output = _read_value(response, "output") or []
        raw_output = _read_value(response, "output_text")
        return ProviderCallResult(
            status=ProviderCallStatus.RECEIVED,
            raw_json=str(raw_output) if raw_output is not None else None,
            response_status=_bounded_label(_read_value(response, "status")),
            refusal_present=_contains_refusal(output),
            usage=usage,
            model_name=model_name,
            model_deployment=(
                _bounded_label(_read_value(response, "model"))
                or (model_name if model_name != config.model_name else None)
            ),
            provider_request_id=_bounded_label(_read_value(response, "id")),
            latency_ms=latency_ms,
            error_code=None,
            stage=stage,
        )

    async def close(self) -> None:
        """Close the native asynchronous client and release its connections."""
        await self._client.close()


def create_impact_provider(settings: EcnSettings) -> EcnAnalysisProvider:
    """Create ``AsyncOpenAI(max_retries=0)`` after environment settings load.

    Args:
        settings: Validated application settings; model choice remains job-local.

    Returns:
        A long-lived native provider adapter ready for background generation.
    """
    del settings
    return NativeImpactProvider(AsyncOpenAI(max_retries=0))
