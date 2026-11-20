"""Timeout-bounded RDF discovery for one impact-analysis attempt."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.ecn.assistant.runtime import ResponsesChatModel, _trace_value
from app.ecn.assistant.skills import SkillLoader
from app.ecn.assistant.graph_tools import create_graph_tools
from app.ecn.llm.diagnostics import classify_provider_failure
from app.ecn.llm.provider import (
    ProviderCallResult,
    ProviderCallStage,
    ProviderCallStatus,
)
from app.ecn.llm.timeout_fallback import invoke_with_timeout_fallback
from app.ecn.models import JsonScalar
from app.ecn.rdf.models import GraphFact
from app.observability import TokenUsage, extract_token_usage
from app.ecn.settings import LLM_TIMEOUT_FALLBACK_MODEL
from app.ecn.services.blocking_io import run_blocking

if TYPE_CHECKING:
    from app.ecn.llm.two_pass import FinalEvidenceProjection
    from app.persistence.records import PinnedGenerationConfig


logger = logging.getLogger(__name__)


class DiscoveryStatus(StrEnum):
    """Describe whether graph facts are safe to expose to final synthesis."""

    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class GraphEvidenceFact:
    """Persistable source fact returned by a bounded RDF tool operation.

    Args:
        evidence_id: Deterministic graph-fact citation identifier.
        subject_ref: Projected RDF resource that owns the fact.
        subject_types: RDF types returned for the subject.
        predicate_name: Source JSON field or fixed RDF relation name.
        value: Scalar value for literal facts, otherwise ``None``.
        object_ref: Related projected resource for edge facts, otherwise ``None``.
        json_pointer: RFC 6901 coordinate in the canonical ECN JSON.
        artifact_id: Relational source-artifact identity.
        source_sha256: Hash of the canonical projected JSON.
        projection_version: Mapper version that produced the fact.
    """

    evidence_id: str
    subject_ref: str
    subject_types: tuple[str, ...]
    predicate_name: str
    value: JsonScalar | None
    object_ref: str | None
    json_pointer: str
    artifact_id: str
    source_sha256: str
    projection_version: str

    @classmethod
    def from_graph_fact(cls, ecn_row_id: str, fact: GraphFact) -> "GraphEvidenceFact":
        """Create one deterministic evidence record from a returned graph fact."""
        payload = {
            "ecn_row_id": ecn_row_id,
            "json_pointer": fact.json_pointer,
            "object_ref": fact.object_ref,
            "predicate_name": fact.predicate_name,
            "projection_version": fact.projection_version,
            "source_sha256": fact.source_sha256,
            "subject_ref": fact.subject_ref,
            "value": fact.value,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(
            evidence_id=f"E-{digest[:16]}",
            subject_ref=fact.subject_ref,
            subject_types=fact.subject_types,
            predicate_name=fact.predicate_name,
            value=fact.value,
            object_ref=fact.object_ref,
            json_pointer=fact.json_pointer,
            artifact_id=fact.artifact_id,
            source_sha256=fact.source_sha256,
            projection_version=fact.projection_version,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical, source-only representation retained with analysis."""
        return {
            "evidence_id": self.evidence_id,
            "subject_ref": self.subject_ref,
            "subject_types": list(self.subject_types),
            "predicate_name": self.predicate_name,
            "value": self.value,
            "object_ref": self.object_ref,
            "json_pointer": self.json_pointer,
            "artifact_id": self.artifact_id,
            "source_sha256": self.source_sha256,
            "projection_version": self.projection_version,
        }


@dataclass(frozen=True)
class DiscoveryCall:
    """Audit one actual discovery-model request before final synthesis."""

    sequence: int
    result: ProviderCallResult
    started_at: datetime
    finished_at: datetime

    @property
    def stage(self) -> str:
        """Return the unique persistence label for this sequential call."""
        return f"DISCOVERY_{self.sequence}"


@dataclass(frozen=True)
class DiscoveryResult:
    """Retain only complete graph evidence, or a safe failure diagnostic.

    Partial facts are intentionally discarded on any discovery failure. This
    makes the final model's graph-citation allowlist an all-or-nothing trust
    boundary and leaves baseline deterministic evidence usable.
    """

    status: DiscoveryStatus
    facts: tuple[GraphEvidenceFact, ...]
    calls: tuple[DiscoveryCall, ...]
    error_code: str | None

    @classmethod
    def complete(
        cls,
        facts: tuple[GraphEvidenceFact, ...],
        calls: tuple[DiscoveryCall, ...] = (),
    ) -> "DiscoveryResult":
        """Return a successful result with duplicate fact identities removed."""
        unique = {fact.evidence_id: fact for fact in facts}
        return cls(
            DiscoveryStatus.COMPLETE,
            tuple(unique[key] for key in sorted(unique)),
            calls,
            None,
        )

    @classmethod
    def failed(
        cls,
        error_code: str,
        calls: tuple[DiscoveryCall, ...] = (),
    ) -> "DiscoveryResult":
        """Return a safe failed result without exposing partial graph facts."""
        if not error_code or len(error_code) > 80:
            raise ValueError("discovery error code must be 1 to 80 characters")
        return cls(DiscoveryStatus.FAILED, (), calls, error_code)

    @property
    def discovery_hash(self) -> str:
        """Return the deterministic digest of exactly the facts final synthesis receives."""
        return hashlib.sha256(
            _canonical_json([fact.as_dict() for fact in self.facts]).encode("utf-8")
        ).hexdigest()

    def evidence_json(self) -> str:
        """Return canonical persisted source facts without model prose or tool payloads."""
        return _canonical_json([fact.as_dict() for fact in self.facts])


def _canonical_json(value: object) -> str:
    """Serialize trusted discovery metadata with a stable compact representation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _discovery_prompt(projection: "FinalEvidenceProjection") -> str:
    """Build the small mandatory context for tool-directed graph exploration."""
    return _canonical_json(
        {
            "ecn_row_id": projection.ecn_row_id,
            "ecn_header": dict(projection.ecn_header),
            "snapshot_counts": projection.snapshot_context.get("counts", {}),
            "grouped_evidence_count": len(projection.records),
            "instruction": (
                "Use graph tools only to retrieve source facts that could materially "
                "clarify this ECN's impact. Do not make impact conclusions or return "
                "prose. Do not infer Make/Buy from descriptions, replacements from "
                "co-occurrence, or scope beyond the supplied Q900V BOM population."
            ),
        }
    )


class ImpactDiscoveryService:
    """Run tool-enabled Responses calls until completion or the request timeout.

    The service has no generic query tool and never passes model prose to the
    final model. It captures only graph facts returned by the existing bounded
    search and local-expansion tools.
    """

    def __init__(
        self,
        rdf,
        model_name: str,
        model_factory: Callable[[], object] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        fallback_model_factory: Callable[[], object] | None = None,
    ) -> None:
        """Store RDF access and lazy primary/fallback Responses model factories.

        Args:
            rdf: Currentness gate and scoped RDF query service.
            model_name: Primary text-generation deployment for discovery.
            model_factory: Optional lazy factory for the primary model.
            monotonic: Injectable clock for provider latency measurements.
            fallback_model_factory: Optional lazy factory for the timeout fallback.
        """
        self._rdf = rdf
        self._model_name = model_name
        self._model_factory = model_factory or self._create_model
        self._fallback_model_factory = fallback_model_factory or (
            lambda: self._create_model(LLM_TIMEOUT_FALLBACK_MODEL)
        )
        self._monotonic = monotonic
        self._instance: object | None = None
        self._fallback_instance: object | None = None
        self._additional_instances: dict[str, object] = {}
        self._model_lock = asyncio.Lock()

    def _create_model(self, model_name: str | None = None) -> ResponsesChatModel:
        """Create a Responses-compatible model for the requested deployment."""
        return ResponsesChatModel(
            proxy_model_name=model_name or self._model_name,
            use_responses_api=True,
            max_completion_tokens=1500,
            timeout=120,
            max_retries=0,
            model_kwargs={"store": False},
        )

    async def _get_model(self, model_name: str) -> object:
        """Initialize and cache the requested primary, fallback, or pinned model.

        Args:
            model_name: Exact deployment pinned for this discovery request.

        Returns:
            A cached LangChain model for the requested deployment.
        """
        async with self._model_lock:
            if model_name == self._model_name:
                if self._instance is None:
                    self._instance = await run_blocking(self._model_factory)
                return self._instance
            if model_name == LLM_TIMEOUT_FALLBACK_MODEL:
                if self._fallback_instance is None:
                    self._fallback_instance = await run_blocking(
                        self._fallback_model_factory
                    )
                return self._fallback_instance
            if model_name not in self._additional_instances:
                self._additional_instances[model_name] = await run_blocking(
                    lambda: self._create_model(model_name)
                )
            return self._additional_instances[model_name]

    async def discover(
        self,
        projection: "FinalEvidenceProjection",
        config: "PinnedGenerationConfig",
    ) -> DiscoveryResult:
        """Capture graph facts until the model stops requesting tools."""
        try:
            await run_blocking(self._rdf.require_current, projection.ecn_row_id)
        except Exception:  # noqa: BLE001 - the final baseline remains safe without RDF.
            return DiscoveryResult.failed("RDF_PROJECTION_UNAVAILABLE")
        try:
            skill = SkillLoader(
                Path(__file__).parent / "agent_skills", 16_000
            ).load(["assess-impact"])
        except Exception:  # noqa: BLE001 - a malformed local skill must not leak into prompts.
            return DiscoveryResult.failed("RDF_DISCOVERY_SKILL_ERROR")
        primary_model = await self._get_model(config.model_name)
        captured: list[GraphFact] = []
        issued_nodes: set[str] = set()
        tools = create_graph_tools(
            self._rdf, projection.ecn_row_id, captured, issued_nodes
        )
        tool_by_name = {tool.name: tool for tool in tools}
        messages: list[object] = [
            SystemMessage(
                content=(
                    "You are a timeout-bounded ECN RDF discovery worker. Use only the supplied "
                    "tools. Return no prose; final synthesis receives only tool-returned "
                    f"source facts.\n\n{skill}"
                )
            ),
            HumanMessage(content=_discovery_prompt(projection)),
        ]
        calls: list[DiscoveryCall] = []
        tool_sequence = 0
        active_model_name = config.model_name
        pending_call_started_at: datetime | None = None
        pending_call_started: float | None = None
        try:
            bound = primary_model.bind_tools(tools)  # type: ignore[union-attr]
            async with asyncio.timeout(config.request_timeout_seconds * 2):
                while True:
                    sequence = len(calls) + 1
                    started_at = datetime.now(timezone.utc)
                    started = self._monotonic()
                    pending_call_started_at = started_at
                    pending_call_started = started

                    def note_fallback(model_name: str) -> None:
                        """Mark a discovery call as running on the fallback model."""
                        nonlocal active_model_name
                        active_model_name = model_name

                    async def fallback_runnable() -> object:
                        """Bind the fallback deployment to the same fixed RDF tools."""
                        fallback_model = await self._get_model(
                            LLM_TIMEOUT_FALLBACK_MODEL
                        )
                        return fallback_model.bind_tools(tools)  # type: ignore[union-attr]

                    response, used_model_name = await invoke_with_timeout_fallback(
                        bound,
                        messages,
                        config.model_name,
                        fallback_runnable,
                        note_fallback,
                    )
                    active_model_name = used_model_name
                    finished_at = datetime.now(timezone.utc)
                    pending_call_started_at = None
                    pending_call_started = None
                    stage = (
                        ProviderCallStage.DISCOVERY_1
                        if sequence == 1
                        else ProviderCallStage.DISCOVERY_2
                    )
                    calls.append(
                        DiscoveryCall(
                            sequence,
                            ProviderCallResult(
                                status=ProviderCallStatus.RECEIVED,
                                raw_json=None,
                                response_status="completed",
                                refusal_present=False,
                                usage=extract_token_usage(response),
                                model_name=used_model_name,
                                model_deployment=(
                                    used_model_name
                                    if used_model_name != config.model_name
                                    else None
                                ),
                                provider_request_id=None,
                                latency_ms=max(
                                    0,
                                    round(
                                        (self._monotonic() - started) * 1_000
                                    ),
                                ),
                                error_code=None,
                                stage=stage,
                            ),
                            started_at,
                            finished_at,
                        )
                    )
                    messages.append(response)
                    requested = tuple(getattr(response, "tool_calls", ()))
                    if not requested:
                        break
                    for call in requested:
                        tool_sequence += 1
                        tool = tool_by_name.get(call.get("name"))
                        if tool is None:
                            return DiscoveryResult.failed(
                                "RDF_DISCOVERY_TOOL_INVALID", tuple(calls)
                            )
                        tool_started = self._monotonic()
                        try:
                            result = await tool.ainvoke(call.get("args", {}))
                        except Exception:  # noqa: BLE001 - tool gaps remain non-authoritative.
                            logger.info(
                                "ECN discovery tool call %s",
                                _canonical_json(
                                    {
                                        "sequence": tool_sequence,
                                        "assessment_id": projection.ecn_row_id,
                                        "tool": tool.name,
                                        "arguments": _trace_value(
                                            call.get("args", {})
                                        ),
                                        "duration_ms": max(
                                            0,
                                            round(
                                                (self._monotonic() - tool_started)
                                                * 1_000
                                            ),
                                        ),
                                        "outcome": "error",
                                        "result": {
                                            "error_code": "RDF_DISCOVERY_TOOL_ERROR"
                                        },
                                    }
                                ),
                            )
                            return DiscoveryResult.failed(
                                "RDF_DISCOVERY_TOOL_ERROR", tuple(calls)
                            )
                        payload = result.get("result", result)
                        summary = {
                            key: payload[key]
                            for key in (
                                "total_matches",
                                "page",
                                "page_size",
                                "has_next",
                                "detail_omitted",
                                "error_code",
                            )
                            if isinstance(payload, dict) and key in payload
                        }
                        if isinstance(payload, dict) and isinstance(
                            payload.get("items"), list
                        ):
                            summary["item_count"] = len(payload["items"])
                        logger.info(
                            "ECN discovery tool call %s",
                            _canonical_json(
                                {
                                    "sequence": tool_sequence,
                                    "assessment_id": projection.ecn_row_id,
                                    "tool": tool.name,
                                    "arguments": _trace_value(
                                        call.get("args", {})
                                    ),
                                    "duration_ms": max(
                                        0,
                                        round(
                                            (self._monotonic() - tool_started) * 1_000
                                        ),
                                    ),
                                    "outcome": "success",
                                    "result": summary,
                                }
                            ),
                        )
                        messages.append(
                            ToolMessage(
                                content=_canonical_json(result),
                                tool_call_id=str(
                                    call.get("id", "discovery-tool-call")
                                ),
                                name=tool.name,
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize SDK failures without exposing details.
            failure = classify_provider_failure(exc)
            if (
                failure.error_code == "PROVIDER_TIMEOUT"
                and pending_call_started_at is None
            ):
                # The overall discovery deadline may expire between model calls,
                # such as while a bounded RDF tool is running.
                return DiscoveryResult.failed("RDF_DISCOVERY_TIMEOUT", tuple(calls))
            sequence = len(calls) + 1
            stage = (
                ProviderCallStage.DISCOVERY_1
                if sequence == 1
                else ProviderCallStage.DISCOVERY_2
            )
            now = datetime.now(timezone.utc)
            call_started_at = pending_call_started_at or now
            latency_ms = (
                max(0, round((self._monotonic() - pending_call_started) * 1_000))
                if pending_call_started is not None
                else 0
            )
            calls.append(
                DiscoveryCall(
                    sequence,
                    ProviderCallResult(
                        status=ProviderCallStatus.ERROR,
                        raw_json=None,
                        response_status=None,
                        refusal_present=False,
                        usage=TokenUsage(),
                        model_name=active_model_name,
                        model_deployment=None,
                        provider_request_id=failure.request_id,
                        latency_ms=latency_ms,
                        error_code=failure.error_code,
                        stage=stage,
                        http_status=failure.http_status,
                        provider_code=failure.provider_code,
                        retry_after_seconds=failure.retry_after_seconds,
                        retryable=failure.retryable,
                    ),
                    call_started_at,
                    now,
                )
            )
            discovery_error = (
                "RDF_DISCOVERY_TIMEOUT"
                if failure.error_code == "PROVIDER_TIMEOUT"
                else failure.error_code
            )
            return DiscoveryResult.failed(discovery_error, tuple(calls))
        return DiscoveryResult.complete(
            tuple(
                GraphEvidenceFact.from_graph_fact(projection.ecn_row_id, fact)
                for fact in captured
            ),
            tuple(calls),
        )

    async def close(self) -> None:
        """Close the lazily created discovery model client during application shutdown."""
        closed_clients = set()
        instances = (
            self._instance,
            self._fallback_instance,
            *self._additional_instances.values(),
        )
        for instance in instances:
            client = getattr(instance, "root_async_client", None)
            if client is None or id(client) in closed_clients:
                continue
            closed_clients.add(id(client))
            await client.close()
