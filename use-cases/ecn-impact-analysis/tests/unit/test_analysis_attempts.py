"""Tests for isolated, auditable processing of one claimed analysis attempt."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import UUID

import pytest
from app.ecn.analysis_models import derive_overall_impact
from app.ecn.assistant.discovery import DiscoveryCall, DiscoveryResult, GraphEvidenceFact
from app.ecn.llm.provider import (
    ProviderCallResult,
    ProviderCallStage,
    ProviderCallStatus,
)
from app.ecn.llm.two_pass import (
    EvidencePlan,
    EvidenceSlice,
    FinalEvidenceProjection,
    resolve_evidence_plan,
)
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
)
from app.ecn.services.generation import (
    AnalysisGenerationService,
    JobProcessingResult,
    build_analysis_completion,
)
from app.observability import TokenUsage
from app.persistence.records import (
    AttemptStart,
    ClaimedJob,
    PinnedGenerationConfig,
    ProviderCallOutcome,
    ValidationOutcome,
)

from tests.fixtures.builders import (
    make_evidence_bundle,
    make_evidence_bundle_sections,
    synthetic_uuid,
)


class RecordingAnalysisRepository:
    """Record repository writes and optionally fail selected operations."""

    def __init__(self) -> None:
        """Initialize an empty call log and failure selector."""
        self.calls: list[str] = []
        self.outcomes: list[object] = []
        self.stage_records: list[object] = []
        self.completions: list[object] = []
        self.failures: list[object] = []
        self.raise_on: set[str] = set()
        self.saved_stages: dict[str, dict] = {}

    def list_stage_results(self, job_id, evidence_hash, prompt_version, schema_version):
        """Return already validated outputs available for an exact retry."""
        del job_id, evidence_hash, prompt_version, schema_version
        return dict(self.saved_stages)

    def save_stage_result(self, result_id, job_id, attempt_id, stage, evidence_hash,
                          prompt_version, schema_version, content, created_at):
        """Retain one validated stage after its provider call."""
        del result_id, job_id, attempt_id, evidence_hash, prompt_version, schema_version, created_at
        self.saved_stages[stage] = content

    def record_provider_outcome(self, attempt_id: str, outcome: object) -> None:
        """Record a terminal provider outcome for the attempt."""
        self.calls.append("record_provider_outcome")
        self.outcomes.append((attempt_id, outcome))
        if "record_provider_outcome" in self.raise_on:
            raise RuntimeError("database contains source secret")

    def record_provider_call(self, record: object) -> None:
        """Record one immutable planner/final child row."""
        self.calls.append("record_provider_call")
        self.stage_records.append(record)
        if "record_provider_call" in self.raise_on:
            raise RuntimeError("database contains source secret")

    def complete_job(self, job_id: str, completion: object) -> None:
        """Record an immutable analysis completion."""
        self.calls.append("complete_job")
        self.completions.append((job_id, completion))
        if "complete_job" in self.raise_on:
            raise RuntimeError("database contains source secret")

    def fail_job(
        self,
        job_id: str,
        attempt_id: str,
        error: object,
        finished_at: datetime,
    ) -> None:
        """Record a safe job failure transition."""
        self.calls.append("fail_job")
        self.failures.append((job_id, attempt_id, error, finished_at))
        if "fail_job" in self.raise_on:
            raise RuntimeError("database contains source secret")


class RecordingProvider:
    """Return stage-specific provider outcomes and record call count."""

    def __init__(self, final_result: ProviderCallResult) -> None:
        """Store planner/final defaults and initialize behavior controls."""
        self.planner_result = replace(
            final_result,
            raw_json=json.dumps(
                {
                    "requested_slices": [item.value for item in EvidenceSlice],
                    "focus_material_numbers": [],
                }
            ),
            usage=TokenUsage(5, 1, 2, 8, 0, 6, True),
            model_deployment="deployment-planner",
            provider_request_id="response-planner",
            latency_ms=21,
            stage=ProviderCallStage.PLANNER,
        )
        self.final_result = replace(final_result, stage=ProviderCallStage.FINAL)
        self.error: BaseException | None = None
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.call_count = 0
        self.closed = False

    async def generate(
        self,
        projection: object,
        config: object,
        stage: ProviderCallStage = ProviderCallStage.FINAL,
    ) -> ProviderCallResult:
        """Record and resolve one fake provider request."""
        del config
        self.call_count += 1
        self.entered.set()
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        if stage is ProviderCallStage.PLANNER:
            return self.planner_result
        result = self.final_result
        if result.raw_json == "__DYNAMIC__":
            assert isinstance(projection, FinalEvidenceProjection)
            result = replace(result, raw_json=_valid_raw_json(projection))
        return result

    async def close(self) -> None:
        """Record provider shutdown."""
        self.closed = True


class RecordingDiscoveryService:
    """Return a fixed bounded discovery outcome without a network dependency."""

    def __init__(self, result: DiscoveryResult) -> None:
        """Store the pre-built discovery outcome for one service invocation."""
        self.result = result
        self.calls = 0
        self.closed = False

    async def discover(
        self,
        projection: FinalEvidenceProjection,
        config: PinnedGenerationConfig,
    ) -> DiscoveryResult:
        """Record the baseline projection and return the configured result."""
        del projection, config
        self.calls += 1
        return self.result

    async def close(self) -> None:
        """Record lifecycle cleanup requested by the generation service."""
        self.closed = True


class FixedClock:
    """Return a fixed timezone-aware timestamp for deterministic persistence."""

    def __init__(self) -> None:
        """Store the deterministic Phase 3 timestamp."""
        self.value = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        """Return the fixed timestamp."""
        return self.value


class FixedUuidFactory:
    """Return a deterministic UUID for immutable analysis records."""

    def __call__(self) -> UUID:
        """Return the fixed analysis UUID."""
        return UUID("00000000-0000-4000-8000-000000000099")


def _evidence(key: AnalysisSectionKey, index: int) -> EvidenceItem:
    """Return one evidence item that permits a grounded section response."""
    values: dict[str, object] = {"safe": key.value}
    if key is AnalysisSectionKey.EXECUTIVE_SUMMARY:
        values = {"ecn_number": "ECN-TEST", "title": "Test change"}
    elif key in {
        AnalysisSectionKey.CHANGE_INTENT,
        AnalysisSectionKey.AFFECTED_MATERIALS,
    }:
        values = {"role": "RESULTING", "number": "PART-100"}
    elif key is AnalysisSectionKey.EBOM_MBOM_IMPACT:
        values = {"bom_kind": "EBOM", "component_number": "PART-100"}
    elif key is AnalysisSectionKey.CONFIGURATION_SCOPE:
        values = {"plant": "PL10"}
    return EvidenceItem(
        evidence_id=f"E-{index:016x}",
        section_key=key,
        kind=EvidenceKind.SOURCE_FACT,
        tier=None,
        summary=f"Evidence for {key.value}",
        sources=(
            EvidenceSource(
                artifact_id=synthetic_uuid("attempt-artifact"),
                artifact_kind=ArtifactKind.ECN_JSON,
                json_path=f"$.{key.value}",
                sheet_name=None,
                bom_row_id=None,
                source_row_number=None,
            ),
        ),
        values=values,
        related_ids=(),
        quality_qualifiers=(),
    )


def _claimed_job() -> ClaimedJob:
    """Build one claimed job with complete projected evidence and pinned context."""
    items = {
        key: _evidence(key, index) for index, key in enumerate(AnalysisSectionKey, 1)
    }
    evidence = make_evidence_bundle(
        sections=make_evidence_bundle_sections(
            **{key.value: (items[key],) for key in AnalysisSectionKey}
        ),
        evidence_hash="a" * 64,
    )
    attempt = AttemptStart(
        attempt_id=synthetic_uuid("attempt-1"),
        correlation_id="correlation-1",
        started_at=datetime(2026, 9, 15, 9, 59, tzinfo=timezone.utc),
        provider="sap-ai-core",
        model_name="gpt-5.4",
        model_deployment=None,
        llm_endpoint="responses",
    )
    config = PinnedGenerationConfig(
        prompt_version="ecn-impact-v2",
        output_schema_version="ecn-impact-sections-v2",
        model_name="gpt-5.4",
        request_timeout_seconds=10,
        max_prompt_evidence_items=20,
        max_prompt_characters=20_000,
        strict_structured_output=True,
    )
    return ClaimedJob(
        job_id=synthetic_uuid("job-1"),
        upload_id=synthetic_uuid("upload-1"),
        ecn_row_id=evidence.ecn_row_id,
        attempt_number=1,
        attempt=attempt,
        evidence=evidence,
        generation_config=config,
    )


def _valid_raw_json(projection: FinalEvidenceProjection) -> str:
    """Return a strict response citing only the resolved final projection."""
    sections: dict[str, object] = {}
    for key in AnalysisSectionKey:
        allowed = sorted(projection.allowed_evidence_ids_by_section[key])
        requires_gap = key in projection.sections_requiring_gap or not allowed
        sections[key.value] = (
            {
                "status": "INSUFFICIENT_EVIDENCE",
                "assessment_markdown": f"More evidence is needed for {key.value}.",
                "evidence_ids": allowed[:1],
                "missing_evidence": ["Additional source evidence is required."],
            }
            if requires_gap
            else {
                "status": "SUFFICIENT",
                "assessment_markdown": f"Grounded {key.value} assessment.",
                "evidence_ids": allowed[:1],
                "missing_evidence": [],
            }
        )

    def factor(ids: list[str]) -> dict[str, object]:
        """Return a known low factor or an explicit unknown factor."""
        if ids:
            return {
                "severity": "LOW",
                "assessment": "The resolved evidence indicates bounded impact.",
                "evidence_ids": ids[:1],
                "missing_evidence": [],
            }
        return {
            "severity": "UNKNOWN",
            "assessment": "The evidence is insufficient for this factor.",
            "evidence_ids": [],
            "missing_evidence": ["Additional factor evidence is required."],
        }

    material_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section.value
            in {
                "added_materials",
                "removed_materials",
                "modified_materials",
            }
            for citation in record.citation_ids
        }
    )
    manufacturing_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section.value
            in {
                "added_materials",
                "removed_materials",
                "modified_materials",
                "ebom_matches",
                "mbom_matches",
            }
            for citation in record.citation_ids
        }
    )
    configuration_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section.value == "configuration"
            for citation in record.citation_ids
        }
    )
    factors = {
        "manufacturing": factor(manufacturing_ids),
        "procurement": factor(material_ids),
        "configuration": factor(configuration_ids),
        "data_readiness": {
            "severity": "UNKNOWN",
            "assessment": "The service derives readiness deterministically.",
            "evidence_ids": [],
            "missing_evidence": ["Use the deterministic readiness assessment."],
        },
    }
    overall = (
        "LOW"
        if all(factors[name]["severity"] == "LOW" for name in (
            "manufacturing",
            "procurement",
            "configuration",
        ))
        else "UNKNOWN"
    )
    return json.dumps(
        {
            "sections": sections,
            "impact_factors": factors,
            "overall_impact": overall,
        }
    )


def _provider_result(claimed: ClaimedJob) -> ProviderCallResult:
    """Return one successful provider response with cache-aware usage."""
    return ProviderCallResult(
        status=ProviderCallStatus.RECEIVED,
        raw_json="__DYNAMIC__",
        response_status="completed",
        refusal_present=False,
        usage=TokenUsage(11, 2, 3, 16, 1, 13, True),
        model_name=claimed.attempt.model_name,
        model_deployment="deployment-1",
        provider_request_id="response-1",
        latency_ms=42,
        error_code=None,
    )


class StagedProvider:
    """Return strict stage payloads and allow one isolated stage failure."""

    def __init__(self, claimed, projection):
        """Keep the pinned projection and stage call observations."""
        self.base = _provider_result(claimed)
        self.valid = json.loads(_valid_raw_json(projection))
        self.valid["sections"]["executive_summary"]["assessment_markdown"] = "- **What changes:** QKX4638810 is replaced by QKX463881A.\n- **What is impacted:** Assembly Q923VV2526401.\n- **Where:** Plant PL10."
        self.calls = []
        self.fail_once = None
        self.active = 0
        self.maximum_active = 0
        self.factor_evidence_id = None

    async def generate(self, prompt, config, stage):
        """Return one stage response and observe parallel provider execution."""
        del config
        self.calls.append(stage.value)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0)
            if self.fail_once == stage.value:
                self.fail_once = None
                raise RuntimeError("isolated provider failure")
            if stage.value == "DATA_READINESS":
                payload = {"descriptions": [
                    {"code": item["code"], "description": "SIM1100 lacks a unit of measure, so its stock quantity cannot be compared."}
                    for item in prompt.payload["questions"]
                ]}
            elif stage.value == "SUMMARY":
                payload = {
                    "change_brief": "The cab cover changes from 3D printing to injection plastic.",
                    "sections": self.valid["sections"],
                }
            else:
                factor = dict(self.valid["impact_factors"][stage.value.lower()])
                factor["assessment"] = "- Grounded impact assessment."
                if self.factor_evidence_id:
                    factor["evidence_ids"] = [self.factor_evidence_id]
                payload = {"factor": factor, "additional_next_steps": [
                    f"Confirm the {stage.value.lower()} disposition for SIM1100 in the reviewed source record."
                ]}
            return replace(self.base, stage=stage, raw_json=json.dumps(payload))
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_staged_failure_retains_sibling_outputs_and_retries_only_missing_stage() -> None:
    """Four parallel perspectives persist independently before sequential synthesis."""
    original = _claimed_job()
    claimed = replace(original, generation_config=replace(
        original.generation_config, prompt_version="ecn-snapshot-v3",
        output_schema_version="ecn-impact-sections-v3",
    ))
    projection = build_synthesis_projection(claimed.evidence, None)
    repository = RecordingAnalysisRepository()
    provider = StagedProvider(claimed, projection)
    provider.fail_once = "PROCUREMENT"
    service = AnalysisGenerationService(
        analysis_repository=repository, provider=provider,
        usage_emitter=lambda **kwargs: None, clock=FixedClock(),
        uuid_factory=FixedUuidFactory(),
    )
    snapshot = {"assessment_date": "2026-03-15", "questions": [{
        "code": "STOCK_CONTEXT_LIMITED", "question": "Confirm missing stock fields",
        "why": "Stock units are absent.",
        "subjects": [{"number": "SIM1100", "details": ["Missing: unit of measure"]}],
    }]}
    first = await service._run_staged_analysis(
        claimed, projection, snapshot, (), DiscoveryResult.failed("RDF_DISCOVERY_UNAVAILABLE"),
    )
    assert first.state == "FAILED"
    assert set(repository.saved_stages) == {"MANUFACTURING", "CONFIGURATION", "DATA_READINESS"}
    assert provider.maximum_active == 4
    assert "SUMMARY" not in provider.calls
    second = await service._run_staged_analysis(
        claimed, projection, snapshot, (), DiscoveryResult.failed("RDF_DISCOVERY_UNAVAILABLE"),
    )
    assert second.state == "COMPLETED", (second.error_code, provider.calls, repository.saved_stages)
    assert provider.calls.count("MANUFACTURING") == 1
    assert provider.calls.count("PROCUREMENT") == 2
    assert provider.calls[-1] == "SUMMARY"
    response = repository.completions[-1][1].response
    assert response.change_brief == "The cab cover changes from 3D printing to injection plastic."
    assert response.data_readiness_descriptions[0].code == "STOCK_CONTEXT_LIMITED"
    assert response.additional_next_steps.procurement[0].startswith("Confirm the procurement")


def test_readiness_prompt_names_affected_objects_and_missing_fields() -> None:
    """Supply the stock issue's actual objects and absent fields to its prose call."""
    from app.ecn.llm.staged import build_stage_prompt

    claimed = _claimed_job()
    projection = build_synthesis_projection(claimed.evidence, None)
    snapshot = {"questions": [{
        "code": "STOCK_CONTEXT_LIMITED", "question": "Confirm missing stock fields",
        "why": "Stock rows lack comparable units.",
        "subjects": [
            {"number": number, "details": ["Plant P101 · Missing: unit of measure"]}
            for number in ("SIM1100", "SIM3300", "SIM8700")
        ],
    }]}
    prompt = build_stage_prompt(projection, "DATA_READINESS", snapshot=snapshot)
    issue = prompt.payload["questions"][0]
    assert issue["title"] == "Confirm missing stock fields"
    assert issue["missing_fields"] == ["unit of measure"]
    assert [item["number"] for item in issue["affected_objects"]] == [
        "SIM1100", "SIM3300", "SIM8700",
    ]


@pytest.mark.asyncio
async def test_staged_factors_accept_citations_from_supplied_rdf_discovery() -> None:
    """Accept graph fact IDs that the stage prompt explicitly supplied."""
    original = _claimed_job()
    claimed = replace(original, generation_config=replace(
        original.generation_config, prompt_version="ecn-snapshot-v3",
        output_schema_version="ecn-impact-sections-v3",
    ))
    fact = GraphEvidenceFact(
        evidence_id="E-1234567890abcdef", subject_ref="subject", subject_types=("Part",),
        predicate_name="Name", value="Observed source value", object_ref=None,
        json_pointer="/Name", artifact_id=claimed.evidence.context.ecn_artifact_id,
        source_sha256="a" * 64, projection_version="rdf-v1",
    )
    discovery = DiscoveryResult.complete((fact,))
    projection = build_synthesis_projection(claimed.evidence, None, discovery=discovery)
    repository = RecordingAnalysisRepository()
    provider = StagedProvider(claimed, projection)
    provider.factor_evidence_id = fact.evidence_id
    service = AnalysisGenerationService(
        analysis_repository=repository, provider=provider,
        usage_emitter=lambda **kwargs: None, clock=FixedClock(),
        uuid_factory=FixedUuidFactory(),
    )
    outcome = await service._run_staged_analysis(claimed, projection, {}, (), discovery)
    assert outcome.state == "COMPLETED", outcome.error_code
    assert repository.completions[-1][1].response.impact_factors.manufacturing.evidence_ids == [fact.evidence_id]


@dataclass
class ServiceFixture:
    """Bundle the service and its recording collaborators for one test."""

    service: AnalysisGenerationService
    claimed_job: ClaimedJob
    repository: RecordingAnalysisRepository
    provider: RecordingProvider
    emitted_events: list[dict[str, object]]

    @property
    def emitted_event_count(self) -> int:
        """Return how many telemetry events the service emitted."""
        return len(self.emitted_events)


@pytest.fixture
def service_fixture() -> ServiceFixture:
    """Return one successful generation service with deterministic collaborators."""
    claimed = _claimed_job()
    repository = RecordingAnalysisRepository()
    provider = RecordingProvider(_provider_result(claimed))
    events: list[dict[str, object]] = []

    def emit(**kwargs: object) -> None:
        """Record one payload-free usage event."""
        events.append(kwargs)

    service = AnalysisGenerationService(
        analysis_repository=repository,
        provider=provider,
        usage_emitter=emit,
        clock=FixedClock(),
        uuid_factory=FixedUuidFactory(),
    )
    return ServiceFixture(service, claimed, repository, provider, events)


@pytest.mark.asyncio
async def test_success_persists_provider_outcome_before_validation_and_completion(
    service_fixture: ServiceFixture,
) -> None:
    """Keep the billable provider outcome distinct from application validation."""
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.state == "COMPLETED"
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "complete_job",
    ]
    assert service_fixture.provider.call_count == 1
    assert service_fixture.emitted_event_count == 1
    assert [record.stage for record in service_fixture.repository.stage_records] == [
        "FINAL",
    ]
    assert service_fixture.repository.stage_records[0].call_number == 1
    aggregate = service_fixture.repository.outcomes[0][1]
    assert aggregate.usage.input_tokens == 11
    assert aggregate.usage.output_tokens == 3
    assert aggregate.usage.total_tokens == 16
    assert aggregate.latency_ms == 42
    completion = service_fixture.repository.completions[0][1]
    assert completion.model_deployment == "deployment-1"
    event = service_fixture.emitted_events[0]
    assert event == {
        "route": "ecn-impact-analysis",
        "method": "INTERNAL",
        "actor_type": "batch",
        "provider": "sap-ai-core",
        "model": "gpt-5.4",
        "model_deployment": "deployment-1",
        "llm_endpoint": "responses",
        "usage_available": True,
        "input_tokens": 11,
        "cached_input_tokens": 2,
        "cache_read_input_tokens": 2,
        "cache_write_input_tokens": 1,
        "input_total_tokens": 13,
        "output_tokens": 3,
        "total_tokens": 16,
        "outcome": "success",
        "latency_ms": 42,
        "correlation_id": "correlation-1",
        "upload_id": service_fixture.claimed_job.upload_id,
        "job_id": service_fixture.claimed_job.job_id,
        "attempt_id": service_fixture.claimed_job.attempt.attempt_id,
        "stage": "FINAL",
    }


@pytest.mark.asyncio
async def test_failed_discovery_is_audited_but_final_synthesis_remains_authoritative(
    service_fixture: ServiceFixture,
) -> None:
    """Keep a graph gap separate from the final model outcome and completion state."""
    claimed = service_fixture.claimed_job
    started = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)
    discovery = RecordingDiscoveryService(
        DiscoveryResult.failed(
            "RDF_PROJECTION_UNAVAILABLE",
            (
                DiscoveryCall(
                    1,
                    replace(
                        _provider_result(claimed),
                        raw_json=None,
                        stage=ProviderCallStage.DISCOVERY_1,
                    ),
                    started,
                    started,
                ),
                DiscoveryCall(
                    2,
                    replace(
                        _provider_result(claimed),
                        status=ProviderCallStatus.ERROR,
                        raw_json=None,
                        response_status=None,
                        usage=TokenUsage(),
                        model_deployment=None,
                        provider_request_id=None,
                        error_code="PROVIDER_RESPONSE_INVALID",
                        stage=ProviderCallStage.DISCOVERY_2,
                    ),
                    started,
                    started,
                ),
            ),
        )
    )
    service_fixture.service._discovery = discovery

    result = await service_fixture.service.run_claimed_job(claimed)

    assert result.state == "COMPLETED"
    assert [record.stage for record in service_fixture.repository.stage_records] == [
        "DISCOVERY_1",
        "DISCOVERY_2",
        "FINAL",
    ]
    assert [record.call_number for record in service_fixture.repository.stage_records] == [
        1,
        2,
        3,
    ]
    assert (
        service_fixture.repository.outcomes[0][1].provider_outcome
        is ProviderCallOutcome.SUCCESS
    )
    completion = service_fixture.repository.completions[0][1]
    assert completion.discovery_status == "FAILED"
    assert completion.discovery_error_code == "RDF_PROJECTION_UNAVAILABLE"
    assert completion.discovered_evidence_json == "[]"


@pytest.mark.asyncio
async def test_final_call_follows_every_unbounded_discovery_call(
    service_fixture: ServiceFixture,
) -> None:
    """Assign the final audit position after all actual discovery iterations."""
    claimed = service_fixture.claimed_job
    started = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)
    calls = tuple(
        DiscoveryCall(
            sequence,
            replace(
                _provider_result(claimed),
                raw_json=None,
                stage=(
                    ProviderCallStage.DISCOVERY_1
                    if sequence == 1
                    else ProviderCallStage.DISCOVERY_2
                ),
            ),
            started,
            started,
        )
        for sequence in range(1, 4)
    )
    service_fixture.service._discovery = RecordingDiscoveryService(
        DiscoveryResult.complete((), calls)
    )

    result = await service_fixture.service.run_claimed_job(claimed)

    assert result.state == "COMPLETED"
    assert [record.stage for record in service_fixture.repository.stage_records] == [
        "DISCOVERY_1",
        "DISCOVERY_2",
        "DISCOVERY_3",
        "FINAL",
    ]
    assert [
        record.call_number for record in service_fixture.repository.stage_records
    ] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_received_invalid_output_persists_usage_then_fails_validation(
    service_fixture: ServiceFixture,
) -> None:
    """Retain a billable response and its usage even when structured output is invalid."""
    service_fixture.provider.final_result = replace(
        service_fixture.provider.final_result, raw_json="{private malformed payload"
    )
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.state == "FAILED"
    assert result.error_code == "ANALYSIS_VALIDATION_ERROR"
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "fail_job",
    ]
    outcome = service_fixture.repository.outcomes[0][1]
    assert outcome.provider_outcome is ProviderCallOutcome.SUCCESS
    error = service_fixture.repository.failures[0][2]
    assert error.validation_outcome is ValidationOutcome.INVALID
    assert "private malformed payload" not in repr(error)


@pytest.mark.asyncio
async def test_service_derives_overall_impact_and_data_readiness(
    service_fixture: ServiceFixture,
) -> None:
    """Keep deterministic fields outside model authority before validation."""
    projection = resolve_evidence_plan(
        service_fixture.claimed_job.evidence,
        EvidencePlan(
            requested_slices=list(EvidenceSlice),
            focus_material_numbers=[],
        ),
    )
    payload = json.loads(_valid_raw_json(projection))
    payload["overall_impact"] = "HIGH"
    payload["impact_factors"]["data_readiness"] = {
        "severity": "HIGH",
        "assessment": "Model-authored readiness must not be retained.",
        "evidence_ids": [],
        "missing_evidence": [],
    }
    service_fixture.provider.final_result = replace(
        service_fixture.provider.final_result,
        raw_json=json.dumps(payload),
    )

    result = await service_fixture.service.run_claimed_job(
        service_fixture.claimed_job
    )

    assert result.state == "COMPLETED"
    response = service_fixture.repository.completions[0][1].response
    assert response.overall_impact is derive_overall_impact(response.impact_factors)
    assert (
        response.impact_factors.data_readiness.assessment
        != "Model-authored readiness must not be retained."
    )


@pytest.mark.asyncio
async def test_final_factor_rejects_an_out_of_projection_citation(
    service_fixture: ServiceFixture,
) -> None:
    """Reject fabricated citations while allowing cross-category synthesis."""
    projection = resolve_evidence_plan(
        service_fixture.claimed_job.evidence,
        EvidencePlan(
            requested_slices=list(EvidenceSlice),
            focus_material_numbers=[],
        ),
    )
    payload = json.loads(_valid_raw_json(projection))
    payload["impact_factors"]["manufacturing"] = {
        "severity": "LOW",
        "assessment": "This factor cites a record outside the bounded view.",
        "evidence_ids": ["E-ffffffffffffffff"],
        "missing_evidence": [],
    }
    service_fixture.provider.final_result = replace(
        service_fixture.provider.final_result,
        raw_json=json.dumps(payload),
    )

    result = await service_fixture.service.run_claimed_job(
        service_fixture.claimed_job
    )

    assert result.state == "FAILED"
    assert result.error_code == "ANALYSIS_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_provider_error_is_persisted_and_failed_without_validation(
    service_fixture: ServiceFixture,
) -> None:
    """Persist the adapter's safe provider failure and emit exactly one call event."""
    service_fixture.provider.final_result = replace(
        service_fixture.provider.final_result,
        status=ProviderCallStatus.ERROR,
        raw_json=None,
        response_status=None,
        refusal_present=False,
        usage=TokenUsage(),
        model_deployment=None,
        provider_request_id=None,
        error_code="PROVIDER_AUTH_FAILED",
    )
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "PROVIDER_AUTH_FAILED"
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "fail_job",
    ]
    assert service_fixture.emitted_event_count == 1
    assert service_fixture.emitted_events[0]["outcome"] == "error"


@pytest.mark.asyncio
async def test_projection_failure_records_not_called_without_telemetry(
    service_fixture: ServiceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record pre-provider failure without implying that an LLM request happened."""
    monkeypatch.setattr(
        "app.ecn.services.generation.build_synthesis_projection",
        lambda _bundle, _snapshot: (_ for _ in ()).throw(
            ValueError("private prompt projection details")
        ),
    )
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "PROMPT_PROJECTION_ERROR"
    assert service_fixture.provider.call_count == 0
    assert service_fixture.emitted_event_count == 0
    outcome = service_fixture.repository.outcomes[0][1]
    assert outcome.provider_outcome is ProviderCallOutcome.NOT_CALLED


@pytest.mark.asyncio
async def test_application_timeout_cancels_provider_once_and_records_timeout(
    service_fixture: ServiceFixture,
) -> None:
    """Apply the pinned timeout with no provider retry and one usage event."""
    service_fixture.provider.block = True
    service_fixture.claimed_job = replace(
        service_fixture.claimed_job,
        generation_config=replace(
            service_fixture.claimed_job.generation_config,
            request_timeout_seconds=0.01,
        ),
    )
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "PROVIDER_TIMEOUT"
    assert service_fixture.provider.call_count == 1
    assert service_fixture.emitted_event_count == 1
    outcome = service_fixture.repository.outcomes[0][1]
    assert outcome.provider_outcome is ProviderCallOutcome.TIMEOUT


@pytest.mark.asyncio
async def test_shutdown_cancellation_persists_interruption_then_propagates(
    service_fixture: ServiceFixture,
) -> None:
    """Audit an initiated call as interrupted while preserving task cancellation."""
    service_fixture.provider.block = True
    task = asyncio.create_task(
        service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    )
    await service_fixture.provider.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "fail_job",
    ]
    assert (
        service_fixture.repository.outcomes[0][1].provider_outcome
        is ProviderCallOutcome.INTERRUPTED
    )
    assert service_fixture.emitted_event_count == 1


@pytest.mark.asyncio
async def test_provider_outcome_persistence_failure_skips_validation_and_completion(
    service_fixture: ServiceFixture,
) -> None:
    """Never insert an analysis when its provider outcome was not stored."""
    service_fixture.repository.raise_on.add("record_provider_outcome")
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "DB_PERSISTENCE_ERROR"
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "fail_job",
    ]
    assert service_fixture.repository.completions == []


@pytest.mark.asyncio
async def test_provider_call_persistence_failure_still_closes_the_running_job(
    service_fixture: ServiceFixture,
) -> None:
    """Preserve the aggregate provider outcome before exposing a terminal DB failure."""
    service_fixture.repository.raise_on.add("record_provider_call")

    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)

    assert result == JobProcessingResult(
        service_fixture.claimed_job.job_id,
        "FAILED",
        "DB_PERSISTENCE_ERROR",
    )
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "fail_job",
    ]
    assert (
        service_fixture.repository.outcomes[0][1].provider_outcome
        is ProviderCallOutcome.SUCCESS
    )
    assert service_fixture.repository.completions == []


@pytest.mark.asyncio
async def test_best_effort_failure_write_is_contained(
    service_fixture: ServiceFixture,
) -> None:
    """Return a safe failure when both the primary and best-effort database writes fail."""
    service_fixture.repository.raise_on.update({"record_provider_outcome", "fail_job"})
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "DB_PERSISTENCE_ERROR"
    assert "secret" not in repr(result)


@pytest.mark.asyncio
async def test_adapter_contract_exception_is_recorded_as_an_initiated_failed_call(
    service_fixture: ServiceFixture,
) -> None:
    """Treat an unexpected adapter escape as a safe failed stage with telemetry."""
    service_fixture.provider.error = RuntimeError(
        "source payload and credential secret"
    )
    result = await service_fixture.service.run_claimed_job(service_fixture.claimed_job)
    assert result.error_code == "PROVIDER_ADAPTER_CONTRACT_ERROR"
    assert service_fixture.emitted_event_count == 1
    assert service_fixture.repository.stage_records[0].outcome is ProviderCallOutcome.ERROR
    assert "secret" not in repr(result)


@pytest.mark.asyncio
async def test_telemetry_failure_does_not_strand_the_analysis_job(
    service_fixture: ServiceFixture,
) -> None:
    """Keep provider-call persistence authoritative when logging raises."""

    def broken_emitter(**kwargs: object) -> None:
        """Simulate a non-authoritative telemetry sink contract failure."""
        del kwargs
        raise TypeError("telemetry contract changed")

    service_fixture.service._usage_emitter = broken_emitter

    result = await service_fixture.service.run_claimed_job(
        service_fixture.claimed_job
    )

    assert result.state == "COMPLETED"
    assert service_fixture.repository.calls == [
        "record_provider_call",
        "record_provider_outcome",
        "complete_job",
    ]


@pytest.mark.asyncio
async def test_service_close_closes_the_provider(
    service_fixture: ServiceFixture,
) -> None:
    """Delegate long-lived native client cleanup to the provider adapter."""
    await service_fixture.service.close()
    assert service_fixture.provider.closed is True


def test_completion_builder_rejects_cross_job_or_bad_hash(
    service_fixture: ServiceFixture,
) -> None:
    """Validate immutable job, attempt, evidence, and version references before SQL."""
    from app.ecn.analysis_models import AnalysisResponse, derive_completeness
    from app.ecn.llm.validation import ValidatedAnalysis

    response = AnalysisResponse.model_validate_json(
        json.dumps(
            {
                "sections": {
                    key.value: {
                        "status": "INSUFFICIENT_EVIDENCE",
                        "assessment_markdown": "More evidence is required.",
                        "evidence_ids": [],
                        "missing_evidence": ["Missing test evidence."],
                    }
                    for key in AnalysisSectionKey
                }
            }
        )
    )
    validated = ValidatedAnalysis(response, derive_completeness(response), "b" * 64)
    with pytest.raises(ValueError, match="evidence hash"):
        build_analysis_completion(
            service_fixture.claimed_job,
            validated,
            UUID("00000000-0000-4000-8000-000000000099"),
        )


async def test_reused_factor_with_stale_citation_is_generated_again() -> None:
    """A saved factor citing an ID this attempt cannot accept is re-run, not reused."""
    original = _claimed_job()
    claimed = replace(original, generation_config=replace(
        original.generation_config, prompt_version="ecn-snapshot-v4",
        output_schema_version="ecn-impact-sections-v3",
    ))
    projection = build_synthesis_projection(claimed.evidence, None)
    repository = RecordingAnalysisRepository()
    provider = StagedProvider(claimed, projection)
    provider.fail_once = "PROCUREMENT"
    service = AnalysisGenerationService(
        analysis_repository=repository, provider=provider,
        usage_emitter=lambda **kwargs: None, clock=FixedClock(),
        uuid_factory=FixedUuidFactory(),
    )
    discovery = DiscoveryResult.failed("RDF_DISCOVERY_UNAVAILABLE")
    first = await service._run_staged_analysis(claimed, projection, {}, (), discovery)
    assert first.state == "FAILED"
    # Simulate a discovery fact ID minted by the earlier attempt only.
    repository.saved_stages["MANUFACTURING"]["factor"]["evidence_ids"] = ["E-DISCOVERY-FROM-ATTEMPT-1"]
    second = await service._run_staged_analysis(claimed, projection, {}, (), discovery)
    assert second.state == "COMPLETED", (second.error_code, provider.calls)
    assert provider.calls.count("MANUFACTURING") == 2
    assert provider.calls.count("CONFIGURATION") == 1
