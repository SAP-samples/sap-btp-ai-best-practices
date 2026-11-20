"""In-memory integration harness for the ECN workflow.

Run its consuming suite from the repository root:
    PYTHONPATH=api python -m pytest -q tests/integration/test_ecn_workflow.py

Repository adapters emulate HANA transaction/state behavior. Linked-ECN
resolution reuses the production source-reference parser.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.engine import Engine

from app.dependencies import ServiceContainer
from app.ecn.errors import PersistenceStateError
from app.ecn.analysis_models import AnalysisCompleteness
from app.ecn.api_models import (
    AnalysisFilterState,
    AnalysisProgressResponse as ApiAnalysisProgressResponse,
    BusinessEvidenceDetailResponse,
    BusinessEvidencePageResponse,
    EcnDetailResponse,
    EcnListItem,
    EcnListQuery,
    EcnListResponse,
    LinkedEcnReference,
    HistoryResetResponse,
    RetryResponse,
    StoredAnalysisView,
    UploadResponse,
)
from app.ecn.llm.provider import (
    ProviderCallResult,
    ProviderCallStage,
    ProviderCallStatus,
)
from app.ecn.linked_ecns import extract_linked_ecn_numbers, normalize_ecn_number
from app.ecn.evidence.snapshot import build_snapshot
from app.ecn.intake.json_parser import _normalize_ecn
from app.ecn.llm.staged import StagePrompt
from app.ecn.llm.two_pass import EvidenceSlice, FinalEvidenceProjection
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    BusinessEvidenceSection,
    JobState,
    ParseState,
    SourceArtifact,
    OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
)
from app.ecn.rdf import (
    PROJECTION_VERSION,
    ProjectionBatchResult,
    ProjectionExpectedCounts,
    ProjectionRecord,
    ProjectionState,
)
from app.ecn.reference_data.models import (
    ReferenceDatasetDetail,
    ReferenceDatasetPublication,
    ReferenceDatasetRecord,
    ReferenceDatasetType,
)
from app.ecn.reference_data.service import ReferenceDatasetService, ReferenceUploadRequest
from app.ecn.services.generation import AnalysisGenerationService, SystemClock
from app.ecn.services.history import HistoryService
from app.ecn.services.ingestion import IngestionService
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.services.retry import RetryService
from app.ecn.services.runtime import GenerationRuntime
from app.ecn.settings import EcnSettings
from app.main import create_app
from app.observability.llm_usage_logging import TokenUsage
from app.persistence.analysis_repository import parse_pinned_generation_config
from app.persistence.records import (
    AnalysisCompletion,
    AnalysisProgressResponse,
    AttemptSeed,
    AttemptStart,
    ClaimedJob,
    ClaimJobResult,
    ClaimJobStatus,
    EcnDetailRecord,
    HistoryDeletionCounts,
    ProviderCallOutcome,
    ProviderOutcome,
    ProviderStageRecord,
    RetryQueueResult,
    RetryQueueStatus,
    SafeJobError,
    StoredEcnRecord,
    UploadPublication,
    UploadPublishResult,
    UploadRecord,
    ValidationOutcome,
)
from app.security import get_api_key


PLAIN_SUMMARY = "- **What changes:** QKX4638810 is replaced by QKX463881A.\n- **What is impacted:** Assembly Q923VV2526401.\n- **Where:** Plant PL10."
"""Executive summary in the v4 what changes / what is impacted / where shape."""


def six_section_json(projection: Any) -> str:
    """Return strict JSON honoring projected-evidence and forced-gap rules."""
    evidence_ids_by_section = {
        str(key): list(value)
        for key, value in projection.allowed_evidence_ids_by_section.items()
    }
    forced_gaps = {str(key) for key in projection.sections_requiring_gap}
    sections: dict[str, dict[str, Any]] = {}
    for key, evidence_ids in evidence_ids_by_section.items():
        if key in forced_gaps or not evidence_ids:
            sections[key] = {
                "status": "INSUFFICIENT_EVIDENCE",
                "assessment_markdown": (
                    f"The projected evidence is insufficient for {key}."
                ),
                "evidence_ids": evidence_ids[:1],
                "missing_evidence": [f"Authoritative evidence for {key}"],
            }
        else:
            sections[key] = {
                "status": "SUFFICIENT",
                "assessment_markdown": f"Stored explanation for {key}.",
                "evidence_ids": [evidence_ids[0]],
                "missing_evidence": [],
            }
    material_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section
            in {
                BusinessEvidenceSection.ADDED_MATERIALS,
                BusinessEvidenceSection.REMOVED_MATERIALS,
                BusinessEvidenceSection.MODIFIED_MATERIALS,
            }
            for citation in record.citation_ids
        }
    )
    manufacturing_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section
            in {
                BusinessEvidenceSection.ADDED_MATERIALS,
                BusinessEvidenceSection.REMOVED_MATERIALS,
                BusinessEvidenceSection.MODIFIED_MATERIALS,
                BusinessEvidenceSection.EBOM_MATCHES,
                BusinessEvidenceSection.MBOM_MATCHES,
            }
            for citation in record.citation_ids
        }
    )
    configuration_ids = sorted(
        {
            citation
            for record in projection.records
            if record.section is BusinessEvidenceSection.CONFIGURATION
            for citation in record.citation_ids
        }
    )

    def factor(evidence_ids: list[str]) -> dict[str, Any]:
        """Return a low evidence-backed factor or an explicit unknown."""
        if evidence_ids:
            return {
                "severity": "LOW",
                "assessment": "The supplied evidence indicates bounded impact.",
                "evidence_ids": evidence_ids[:1],
                "missing_evidence": [],
            }
        return {
            "severity": "UNKNOWN",
            "assessment": "The supplied evidence is insufficient.",
            "evidence_ids": [],
            "missing_evidence": ["Additional factor evidence"],
        }

    factors = {
        "manufacturing": factor(manufacturing_ids),
        "procurement": factor(material_ids),
        "configuration": factor(configuration_ids),
        "data_readiness": {
            "severity": "UNKNOWN",
            "assessment": "Readiness is derived by deterministic application logic.",
            "evidence_ids": [],
            "missing_evidence": ["Deterministic readiness result"],
        },
    }
    overall = (
        "LOW"
        if all(
            factors[name]["severity"] == "LOW"
            for name in ("manufacturing", "procurement", "configuration")
        )
        else "UNKNOWN"
    )
    return json.dumps(
        {
            "sections": sections,
            "impact_factors": factors,
            "overall_impact": overall,
        }
    )


def staged_projection(payload: dict[str, Any]) -> SimpleNamespace:
    """Rebuild the fake provider's cited view from the actual staged input."""
    records = [
        SimpleNamespace(
            section=BusinessEvidenceSection(item["section"]),
            citation_ids=item["citation_ids"],
        ) for item in payload["grouped_evidence"]
    ]
    citations = set(payload["header_citation_ids"])
    citations.update(citation for record in records for citation in record.citation_ids)
    return SimpleNamespace(
        records=records,
        allowed_evidence_ids_by_section={key: sorted(citations) for key in AnalysisSectionKey},
        sections_requiring_gap=frozenset(
            AnalysisSectionKey(value) for value in payload["sections_requiring_gap"]
        ),
    )


class ControlledImpactProvider:
    """Return deterministic strict JSON and optionally fail selected calls."""

    def __init__(
        self,
        failing_calls: set[int] | None = None,
        response_factory: Callable[[Any], str] | None = None,
        hold: bool = False,
    ) -> None:
        """Configure deterministic responses, failures, and an optional barrier."""
        self.failing_calls = failing_calls or set()
        self.response_factory = response_factory
        self.call_count = 0
        self.maximum_active = 0
        self._active = 0
        self._condition = asyncio.Condition()
        self._release = asyncio.Event()
        if not hold:
            self._release.set()

    async def wait_until_active(self, count: int) -> None:
        """Wait until the requested provider concurrency has been observed."""
        async with asyncio.timeout(5):
            async with self._condition:
                await self._condition.wait_for(lambda: self._active >= count)

    def release_all(self) -> None:
        """Release every provider call held by the integration barrier."""
        self._release.set()

    def hold_calls(self) -> None:
        """Pause subsequent provider calls until the test releases the barrier."""
        self._release.clear()

    async def generate(
        self,
        projection: Any,
        config: Any,
        stage: ProviderCallStage = ProviderCallStage.FINAL,
    ) -> ProviderCallResult:
        """Record concurrency and return a result using only projected IDs."""
        async with self._condition:
            self.call_count += 1
            ordinal = self.call_count
            self._active += 1
            self.maximum_active = max(self.maximum_active, self._active)
            self._condition.notify_all()
        try:
            await asyncio.sleep(0)
            await self._release.wait()
            if ordinal in self.failing_calls:
                return ProviderCallResult(
                    status=ProviderCallStatus.ERROR,
                    raw_json=None,
                    response_status=None,
                    refusal_present=False,
                    usage=TokenUsage(),
                    model_name=config.model_name,
                    model_deployment=None,
                    provider_request_id=f"controlled-{ordinal}",
                    latency_ms=1,
                    error_code="PROVIDER_SERVER_ERROR",
                    stage=stage,
                )
            if isinstance(projection, StagePrompt):
                if stage is ProviderCallStage.DATA_READINESS:
                    raw_json = json.dumps({"descriptions": [
                        {"code": question["code"], "description": (
                            f'{question["title"]}: {question["deterministic_problem"]}'
                        )} for question in projection.payload["questions"]
                    ]})
                else:
                    context = projection.payload.get("evidence", projection.payload)
                    source = staged_projection(context)
                    response = json.loads(
                        self.response_factory(source) if self.response_factory else six_section_json(source)
                    )
                    if stage is ProviderCallStage.SUMMARY:
                        executive = response["sections"]["executive_summary"]
                        executive["assessment_markdown"] = PLAIN_SUMMARY
                        raw_json = json.dumps({
                            "change_brief": "The affected part change is ready for review.",
                            "sections": response["sections"],
                        })
                    else:
                        factor = response["impact_factors"][stage.value.lower()]
                        factor["assessment"] = "- " + factor["assessment"].strip()
                        raw_json = json.dumps({"factor": factor, "additional_next_steps": [
                            f"Confirm the {stage.value.lower()} decision for this ECN."
                        ]})
            elif stage is ProviderCallStage.PLANNER:
                raw_json = json.dumps(
                    {
                        "requested_slices": [item.value for item in EvidenceSlice],
                        "focus_material_numbers": [],
                    }
                )
            else:
                assert isinstance(projection, FinalEvidenceProjection)
                raw_json = (
                    self.response_factory(projection)
                    if self.response_factory
                    else six_section_json(projection)
                )
            return ProviderCallResult(
                status=ProviderCallStatus.RECEIVED,
                raw_json=raw_json,
                response_status="completed",
                refusal_present=False,
                usage=TokenUsage(
                    input_tokens=20,
                    cached_input_tokens=0,
                    output_tokens=10,
                    total_tokens=30,
                    input_total_tokens=20,
                    usage_available=True,
                ),
                model_name=config.model_name,
                model_deployment=None,
                provider_request_id=f"controlled-{ordinal}",
                latency_ms=1,
                error_code=None,
                stage=stage,
            )
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def close(self) -> None:
        """Satisfy the provider lifecycle without external resources."""
        self._release.set()


@dataclass
class MemoryEcnEntry:
    """Store one normalized ECN projection and its immutable evidence."""

    record: StoredEcnRecord
    evidence: Any
    job_id: str


@dataclass
class MemoryJob:
    """Store mutable job state while retaining the pinned generation config."""

    job_id: str
    upload_id: str
    ecn_row_id: str
    state: JobState
    prompt_version: str
    output_schema_version: str
    model_name: str
    model_config_json: str
    attempt_count: int = 0
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class MemoryAttempt:
    """Store one attempt and its terminal provider/validation outcomes."""

    job_id: str
    attempt_number: int
    start: AttemptStart
    provider_outcome: ProviderCallOutcome = ProviderCallOutcome.STARTED
    validation_outcome: ValidationOutcome = ValidationOutcome.NOT_RUN
    outcome: ProviderOutcome | None = None


@dataclass
class MemoryWorkflowStore:
    """Own all mutable records shared by the in-memory repository adapters."""

    artifacts: dict[str, SourceArtifact] = field(default_factory=dict)
    bom_rows: dict[str, list[Any]] = field(default_factory=dict)
    uploads: dict[str, UploadRecord] = field(default_factory=dict)
    request_to_upload: dict[str, str] = field(default_factory=dict)
    ecns: dict[str, MemoryEcnEntry] = field(default_factory=dict)
    jobs: dict[str, MemoryJob] = field(default_factory=dict)
    attempts: dict[str, MemoryAttempt] = field(default_factory=dict)
    stage_results: list[dict[str, Any]] = field(default_factory=list)
    analyses: dict[str, AnalysisCompletion] = field(default_factory=dict)
    rule_artifact_ids: set[str] = field(default_factory=set)
    reference_datasets: list[ReferenceDatasetRecord] = field(default_factory=list)
    upload_dataset_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    publication_order: int = 0
    reference_publication_order: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class MemoryArtifactRepository:
    """Emulate immutable artifact and stable BOM-row reads under one lock."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def get_artifact(self, artifact_id: str) -> SourceArtifact | None:
        """Return one immutable artifact, if present."""
        with self._store.lock:
            return self._store.artifacts.get(artifact_id)

    def artifact_exists(self, artifact_id: str) -> bool:
        """Return whether an artifact identity exists."""
        with self._store.lock:
            return artifact_id in self._store.artifacts

    def get_current_rule_artifact(self) -> SourceArtifact | None:
        """Return the newest rule artifact using production tie-breaking."""
        with self._store.lock:
            rules = [
                self._store.artifacts[artifact_id]
                for artifact_id in self._store.rule_artifact_ids
            ]
            return max(
                rules,
                key=lambda artifact: (artifact.created_at, artifact.artifact_id),
                default=None,
            )

    def publish_rule_artifact(self, artifact: SourceArtifact) -> SourceArtifact:
        """Append one unique conditional-rule artifact."""
        if artifact.artifact_kind is not ArtifactKind.CONDITIONAL_RULE_WORKBOOK:
            raise ValueError("rule publication requires a conditional-rule artifact")
        with self._store.lock:
            if artifact.artifact_id in self._store.artifacts:
                raise ValueError("artifact already exists")
            self._store.artifacts[artifact.artifact_id] = artifact
            self._store.rule_artifact_ids.add(artifact.artifact_id)
        return artifact

    def iter_bom_rows(
        self, artifact_id: str, batch_size: int = 5_000
    ) -> Iterator[list[Any]]:
        """Yield copied stable BOM-row pages in production sort order."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with self._store.lock:
            rows = sorted(
                self._store.bom_rows.get(artifact_id, ()),
                key=lambda row: (
                    row.bom_kind.value,
                    row.sheet_name,
                    row.source_row_number,
                    row.bom_row_id,
                ),
            )
        for offset in range(0, len(rows), batch_size):
            yield list(rows[offset : offset + batch_size])


class MemoryUploadRepository:
    """Emulate idempotent, atomic publication and latest-BOM lookup."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def find_by_request_id(self, request_id: str) -> UploadRecord | None:
        """Return the authoritative upload for one idempotency key."""
        with self._store.lock:
            upload_id = self._store.request_to_upload.get(request_id)
            return self._store.uploads.get(upload_id) if upload_id else None

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        """Return one upload by internal identity."""
        with self._store.lock:
            return self._store.uploads.get(upload_id)

    def get_latest_bom_snapshot(self) -> SourceArtifact | None:
        """Return the BOM referenced by the latest committed upload."""
        with self._store.lock:
            latest = max(
                self._store.uploads.values(),
                key=lambda upload: (upload.publication_order, upload.upload_id),
                default=None,
            )
            return (
                self._store.artifacts.get(latest.bom_artifact_id)
                if latest is not None
                else None
            )

    def publish_upload(self, publication: UploadPublication) -> UploadPublishResult:
        """Atomically append all records or replay an existing request."""
        with self._store.lock:
            existing_id = self._store.request_to_upload.get(publication.request_id)
            if existing_id is not None:
                return UploadPublishResult(self._store.uploads[existing_id], False)

            self._store.publication_order += 1
            upload = UploadRecord(
                upload_id=publication.upload_id,
                request_id=publication.request_id,
                ecn_artifact_id=publication.ecn_artifact.artifact_id,
                bom_artifact_id=publication.bom_artifact.artifact_id,
                rule_artifact_id=(
                    publication.rule_artifact.artifact_id
                    if publication.rule_artifact
                    else None
                ),
                bom_reused=publication.bom_reused,
                imported_ecn_count=len(publication.ecns),
                publication_order=self._store.publication_order,
                created_at=publication.created_at,
            )
            self._store.artifacts[publication.ecn_artifact.artifact_id] = (
                publication.ecn_artifact
            )
            if not publication.bom_reused:
                self._store.artifacts[publication.bom_artifact.artifact_id] = (
                    publication.bom_artifact
                )
                self._store.bom_rows[publication.bom_artifact.artifact_id] = list(
                    publication.new_bom_rows
                )
            self._store.uploads[upload.upload_id] = upload
            self._store.upload_dataset_ids[upload.upload_id] = tuple(
                link.dataset_id for link in publication.dataset_links
            )
            self._store.request_to_upload[upload.request_id] = upload.upload_id

            for item in publication.ecns:
                ecn = item.ecn
                record = StoredEcnRecord(
                    ecn_row_id=ecn.ecn_row_id,
                    upload_id=upload.upload_id,
                    source_ordinal=ecn.source_ordinal,
                    ecn_number=ecn.ecn_number,
                    title=ecn.title,
                    description_summary=ecn.description_summary,
                    description_text=ecn.description_text,
                    counts=item.counts,
                    source_json=ecn.source_json,
                    change_candidates=item.change_candidates,
                    evidence_hash=item.evidence.evidence_hash,
                    parser_version=publication.parser_version,
                    deterministic_logic_version=publication.deterministic_logic_version,
                    rule_evaluator_version=publication.rule_evaluator_version,
                    created_at=publication.created_at,
                )
                self._store.ecns[ecn.ecn_row_id] = MemoryEcnEntry(
                    record=record,
                    evidence=item.evidence,
                    job_id=item.job_id,
                )
                self._store.jobs[item.job_id] = MemoryJob(
                    job_id=item.job_id,
                    upload_id=upload.upload_id,
                    ecn_row_id=ecn.ecn_row_id,
                    state=JobState.READY,
                    prompt_version=item.prompt_version,
                    output_schema_version=item.output_schema_version,
                    model_name=item.model_name,
                    model_config_json=item.model_config_json,
                )
            return UploadPublishResult(
                upload,
                True,
                projection_ecn_ids=tuple(
                    item.ecn.ecn_row_id for item in publication.ecns
                ),
            )


class MemoryReferenceDatasetRepository:
    """Emulate atomic immutable Settings publications for workflow tests."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def publish(
        self, publication: ReferenceDatasetPublication
    ) -> ReferenceDatasetRecord:
        """Store one validated publication and its normalized source rows."""
        with self._store.lock:
            self._store.reference_publication_order += 1
            self._store.artifacts[publication.artifact.artifact_id] = publication.artifact
            if publication.bom_rows:
                self._store.bom_rows[publication.artifact.artifact_id] = list(
                    publication.bom_rows
                )
            record = ReferenceDatasetRecord(
                dataset_id=publication.dataset_id,
                dataset_type=publication.dataset_type,
                artifact_id=publication.artifact.artifact_id,
                publication_order=self._store.reference_publication_order,
                file_name=publication.artifact.file_name,
                sha256=publication.artifact.sha256,
                scope_code=publication.scope_code,
                scope_name=publication.scope_name,
                row_count=publication.row_count,
                warning_counts=publication.warning_counts,
                parser_version="ecn-reference-v1",
                created_at=publication.created_at,
                active=True,
            )
            self._store.reference_datasets.append(record)
            return record

    def list(self) -> tuple[ReferenceDatasetRecord, ...]:
        """List publications newest first with active type/scope flags."""
        with self._store.lock:
            ordered = sorted(
                self._store.reference_datasets,
                key=lambda item: (item.publication_order, item.dataset_id),
                reverse=True,
            )
            seen: set[tuple[ReferenceDatasetType, str | None]] = set()
            output = []
            for item in ordered:
                key = (item.dataset_type, item.scope_code)
                output.append(replace(item, active=key not in seen))
                seen.add(key)
            return tuple(output)

    def get(
        self, dataset_id: str, sample_size: int = 20
    ) -> ReferenceDatasetDetail | None:
        """Return metadata without row samples in the compact workflow fake."""
        del sample_size
        record = next(
            (item for item in self.list() if item.dataset_id == dataset_id), None
        )
        return None if record is None else ReferenceDatasetDetail(record, ())

    def delete(self, dataset_id: str) -> bool:
        """Remove an unused Settings publication under the shared store lock."""
        with self._store.lock:
            record = next(
                (item for item in self._store.reference_datasets if item.dataset_id == dataset_id),
                None,
            )
            if record is None:
                return False
            if any(dataset_id in ids for ids in self._store.upload_dataset_ids.values()) or any(
                upload.bom_artifact_id == record.artifact_id
                or upload.rule_artifact_id == record.artifact_id
                for upload in self._store.uploads.values()
            ):
                raise PersistenceStateError(
                    "REFERENCE_DATASET_IN_USE",
                    "This file is used by an ECN upload. Delete that analysis history before removing the file.",
                    status_code=409,
                )
            if any(
                item.context.get("comparison_dataset_id") == dataset_id
                for item in self._store.reference_datasets
            ):
                raise PersistenceStateError(
                    "REFERENCE_DATASET_COMPARISON_IN_USE",
                    "This file is the comparison baseline for another dataset. Delete the dependent version first.",
                    status_code=409,
                )
            self._store.reference_datasets.remove(record)
            self._store.artifacts.pop(record.artifact_id, None)
            self._store.bom_rows.pop(record.artifact_id, None)
            self._store.rule_artifact_ids.discard(record.artifact_id)
            return True


class MemoryEcnRepository:
    """Provide side-effect-free list, detail, evidence, and queued-job reads."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def _analysis_state(
        self, job: MemoryJob, analysis: AnalysisCompletion | None
    ) -> AnalysisFilterState:
        """Derive the API state without persisting the gaps projection."""
        if job.state is JobState.COMPLETED and analysis is not None:
            if analysis.completeness is AnalysisCompleteness.COMPLETED_WITH_GAPS:
                return AnalysisFilterState.COMPLETED_WITH_GAPS
        return AnalysisFilterState(job.state.value)

    def _stored_analysis(
        self, completion: AnalysisCompletion | None
    ) -> StoredAnalysisView | None:
        """Map one immutable completion to the public stored-analysis view."""
        if completion is None:
            return None
        return StoredAnalysisView(
            analysis_id=completion.analysis_id,
            completeness=completion.completeness,
            sections=completion.response.sections,
            impact_factors=completion.response.impact_factors,
            overall_impact=completion.response.overall_impact,
            data_readiness_descriptions=completion.response.data_readiness_descriptions,
            additional_next_steps=completion.response.additional_next_steps,
            created_at=completion.created_at,
            evidence_hash=completion.evidence_hash,
            prompt_version=completion.prompt_version,
            output_schema_version=completion.output_schema_version,
            model_name=completion.model_name,
            model_deployment=completion.model_deployment,
        )

    def list_ecns(self, query: EcnListQuery) -> EcnListResponse:
        """Return a filtered, sorted, stable page of stored ECN occurrences."""
        with self._store.lock:
            items: list[EcnListItem] = []
            for entry in self._store.ecns.values():
                record = entry.record
                upload = self._store.uploads[record.upload_id]
                artifact = self._store.artifacts[upload.ecn_artifact_id]
                job = self._store.jobs[entry.job_id]
                analysis = self._store.analyses.get(entry.job_id)
                state = self._analysis_state(job, analysis)
                item = EcnListItem(
                    ecn_row_id=record.ecn_row_id,
                    ecn_number=record.ecn_number,
                    display_ecn_number=record.ecn_number or "Missing ECN ID",
                    title=record.title,
                    description_summary=record.description_summary,
                    source_ordinal=record.source_ordinal,
                    uploaded_at=upload.created_at,
                    source_file_name=artifact.file_name,
                    counts=record.counts,
                    analysis_state=state,
                    has_partial_analysis=(
                        state is AnalysisFilterState.FAILED and bool(
                            MemoryAnalysisRepository(self._store).list_stage_results(
                                job.job_id, entry.evidence.evidence_hash,
                                job.prompt_version, job.output_schema_version,
                            )
                        )
                    ),
                    completeness=analysis.completeness if analysis else None,
                    overall_impact=(
                        analysis.response.overall_impact if analysis else None
                    ),
                )
                if query.query:
                    haystack = " ".join(
                        value
                        for value in (
                            item.ecn_number,
                            item.title,
                            item.description_summary,
                        )
                        if value
                    ).casefold()
                    if query.query.casefold() not in haystack:
                        continue
                if query.ecn_number and not (
                    item.ecn_number
                    and item.ecn_number.casefold().startswith(
                        query.ecn_number.casefold()
                    )
                ):
                    continue
                if query.analysis_state and state is not query.analysis_state:
                    continue
                if query.has_warnings is not None and (
                    (item.counts.warning_count > 0) is not query.has_warnings
                ):
                    continue
                items.append(item)

            sort_values: dict[str, Callable[[EcnListItem], Any]] = {
                "uploaded_at": lambda item: item.uploaded_at,
                "ecn_number": lambda item: item.ecn_number or "",
                "description": lambda item: item.description_summary or "",
                "affected_item_count": lambda item: item.counts.affected_item_count,
                "ebom_match_count": lambda item: item.counts.ebom_match_count,
                "mbom_match_count": lambda item: item.counts.mbom_match_count,
                "warning_count": lambda item: item.counts.warning_count,
                "analysis_state": lambda item: item.analysis_state.value,
            }
            items.sort(
                key=lambda item: (
                    sort_values[query.sort](item),
                    str(item.ecn_row_id),
                ),
                reverse=query.order == "desc",
            )
            total = len(items)
            start = (query.page - 1) * query.page_size
            page_items = items[start : start + query.page_size]
            return EcnListResponse(
                items=page_items,
                page=query.page,
                page_size=query.page_size,
                total_items=total,
                total_pages=0 if total == 0 else math.ceil(total / query.page_size),
            )

    def get_ecn_detail(self, ecn_row_id: str) -> EcnDetailRecord | None:
        """Return one aggregate detail snapshot by exact internal identity."""
        with self._store.lock:
            entry = self._store.ecns.get(ecn_row_id)
            if entry is None:
                return None
            upload = self._store.uploads[entry.record.upload_id]
            job = self._store.jobs[entry.job_id]
            completion = self._store.analyses.get(job.job_id)
            from app.ecn.api_models import AnalysisJobView

            return EcnDetailRecord(
                ecn=entry.record,
                upload=upload,
                ecn_artifact=self._store.artifacts[upload.ecn_artifact_id],
                bom_artifact=self._store.artifacts[upload.bom_artifact_id],
                rule_artifact=(
                    self._store.artifacts.get(upload.rule_artifact_id)
                    if upload.rule_artifact_id
                    else None
                ),
                job=AnalysisJobView(
                    job_id=job.job_id,
                    state=job.state,
                    attempt_count=job.attempt_count,
                    error_code=job.error_code,
                    error_message=job.error_message,
                ),
                analysis=self._stored_analysis(completion),
            )

    def get_linked_ecns(self, record: EcnDetailRecord) -> list[LinkedEcnReference]:
        """Emulate target selection using the shared production reference parser."""
        numbers = extract_linked_ecn_numbers(
            record.ecn.description_text, record.ecn.description_summary,
            current_ecn_number=record.ecn.ecn_number,
        )
        with self._store.lock:
            candidates = sorted(
                (entry.record for entry in self._store.ecns.values()
                 if entry.record.ecn_row_id != record.ecn.ecn_row_id),
                key=lambda item: (
                    item.upload_id != record.upload.upload_id,
                    -self._store.uploads[item.upload_id].publication_order,
                    item.source_ordinal, item.ecn_row_id,
                ),
            )
            targets = {}
            for item in candidates:
                number = normalize_ecn_number(item.ecn_number)
                if number in numbers:
                    targets.setdefault(number, item.ecn_row_id)
            return [LinkedEcnReference(ecn_number=number, ecn_row_id=targets.get(number)) for number in numbers]

    def get_evidence(self, ecn_row_id: str, evidence_id: str) -> Any | None:
        """Return evidence only when it belongs to the requested ECN."""
        with self._store.lock:
            entry = self._store.ecns.get(ecn_row_id)
            if entry is None or not evidence_id or len(evidence_id) > 200:
                return None
            for key in AnalysisSectionKey:
                for item in getattr(entry.evidence.sections, key.value):
                    if item.evidence_id == evidence_id:
                        return item
            return None

    def get_evidence_bundle(self, ecn_row_id: str) -> Any | None:
        """Return a bundle only for an explicit lazy evidence page request."""
        with self._store.lock:
            entry = self._store.ecns.get(ecn_row_id)
            return entry.evidence if entry is not None else None

    def list_queued_job_ids(self, upload_id: str) -> list[str]:
        """Return queued jobs in source occurrence order."""
        with self._store.lock:
            queued = [
                job
                for job in self._store.jobs.values()
                if job.upload_id == upload_id and job.state is JobState.QUEUED
            ]
            queued.sort(
                key=lambda job: (
                    self._store.ecns[job.ecn_row_id].record.source_ordinal,
                    job.job_id,
                )
            )
            return [job.job_id for job in queued]


class MemoryWorkspaceRepository:
    """Emulate selected scheduling; HANA transaction behavior has separate acceptance coverage."""

    def __init__(self, store):
        """Bind the shared in-memory job collection."""
        self.store = store

    def create_batch(self, batch_id, ids):
        """Queue selected idle jobs with current prompt and retain prior results."""
        scheduled = []
        with self.store.lock:
            for job in self.store.jobs.values():
                if job.ecn_row_id in ids and job.state in {
                    JobState.READY, JobState.FAILED, JobState.BLOCKED, JobState.COMPLETED
                }:
                    job.state = JobState.QUEUED
                    job.prompt_version = PROMPT_VERSION
                    job.output_schema_version = OUTPUT_SCHEMA_VERSION
                    job.error_code = None
                    job.error_message = None
                    scheduled.append(job.job_id)
        return {"batch_id":batch_id}, tuple(scheduled)


class MemoryAnalysisRepository:
    """Emulate atomic claims, attempts, terminal writes, retry, and recovery."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def save_stage_result(self, result_id, job_id, attempt_id, stage, evidence_hash,
                          prompt_version, schema_version, content, created_at) -> None:
        """Persist one validated stage with the owning attempt and version pins."""
        del result_id, created_at
        with self._store.lock:
            self._store.stage_results.append({
                "job_id": job_id, "attempt_id": attempt_id, "stage": stage,
                "evidence_hash": evidence_hash, "prompt_version": prompt_version,
                "schema_version": schema_version, "content": content,
            })

    def list_stage_results(self, job_id, evidence_hash, prompt_version, schema_version):
        """Reuse only validated stages after the last accepted attempt."""
        with self._store.lock:
            accepted = self._store.analyses.get(job_id)
            baseline = self._store.attempts[accepted.successful_attempt_id].attempt_number if accepted else 0
            rows = sorted(self._store.stage_results, key=lambda row:
                self._store.attempts[row["attempt_id"]].attempt_number, reverse=True)
            outputs = {}
            for row in rows:
                if (row["job_id"] == job_id and row["evidence_hash"] == evidence_hash
                        and row["prompt_version"] == prompt_version and row["schema_version"] == schema_version
                        and self._store.attempts[row["attempt_id"]].attempt_number > baseline):
                    outputs.setdefault(row["stage"], row["content"])
            return outputs

    def stage_results_for_ecn(self, ecn_row_id):
        """Return current partial outputs for the HTTP stage endpoint."""
        with self._store.lock:
            entry = self._store.ecns.get(ecn_row_id)
            if entry is None:
                return {}
            job = self._store.jobs[entry.job_id]
            return self.list_stage_results(
                job.job_id, entry.evidence.evidence_hash,
                job.prompt_version, job.output_schema_version,
            )

    def claim_job_and_start_attempt(
        self, job_id: str, seed: AttemptSeed
    ) -> ClaimJobResult:
        """Atomically claim one queued job and create its started attempt."""
        with self._store.lock:
            job = self._store.jobs.get(job_id)
            if job is None:
                return ClaimJobResult(ClaimJobStatus.NOT_FOUND, None)
            if job.state is not JobState.QUEUED:
                return ClaimJobResult(ClaimJobStatus.NOT_QUEUED, None)
            try:
                config = parse_pinned_generation_config(
                    job.prompt_version,
                    job.output_schema_version,
                    job.model_name,
                    job.model_config_json,
                )
            except Exception:
                job.state = JobState.FAILED
                job.error_code = "PINNED_CONFIG_INVALID"
                job.error_message = "The pinned generation configuration is invalid"
                return ClaimJobResult(ClaimJobStatus.PINNED_CONFIG_INVALID, None)
            job.state = JobState.RUNNING
            job.attempt_count += 1
            job.error_code = None
            job.error_message = None
            attempt = AttemptStart(
                attempt_id=seed.attempt_id,
                correlation_id=seed.correlation_id,
                started_at=seed.started_at,
                provider=seed.provider,
                model_name=config.model_name,
                model_deployment=seed.model_deployment,
                llm_endpoint=seed.llm_endpoint,
            )
            self._store.attempts[attempt.attempt_id] = MemoryAttempt(
                job_id=job_id,
                attempt_number=job.attempt_count,
                start=attempt,
            )
            entry = self._store.ecns[job.ecn_row_id]
            return ClaimJobResult(
                ClaimJobStatus.CLAIMED,
                ClaimedJob(
                    job_id=job.job_id,
                    upload_id=job.upload_id,
                    ecn_row_id=job.ecn_row_id,
                    attempt_number=job.attempt_count,
                    attempt=attempt,
                    evidence=entry.evidence,
                    generation_config=config,
                ),
            )

    def record_provider_outcome(
        self, attempt_id: str, outcome: ProviderOutcome
    ) -> None:
        """Store one terminal provider outcome exactly once."""
        with self._store.lock:
            attempt = self._store.attempts[attempt_id]
            if attempt.provider_outcome is not ProviderCallOutcome.STARTED:
                raise RuntimeError("attempt already finished")
            attempt.provider_outcome = outcome.provider_outcome
            attempt.outcome = outcome

    def record_provider_call(self, record: ProviderStageRecord) -> None:
        """Accept one immutable provider child row in the in-memory harness."""
        del record

    def complete_job(self, job_id: str, completion: AnalysisCompletion) -> None:
        """Store one analysis and transition its running job to completed."""
        with self._store.lock:
            job = self._store.jobs[job_id]
            attempt = self._store.attempts[completion.successful_attempt_id]
            if (
                job.state is not JobState.RUNNING
                or attempt.provider_outcome is not ProviderCallOutcome.SUCCESS
                or completion.job_id != job_id
            ):
                raise RuntimeError("analysis state conflict")
            attempt.validation_outcome = ValidationOutcome.VALID
            self._store.analyses[job_id] = completion
            job.state = JobState.COMPLETED

    def fail_job(
        self,
        job_id: str,
        attempt_id: str,
        error: SafeJobError,
        finished_at: datetime,
    ) -> None:
        """Persist a safe failure after a terminal provider outcome."""
        del finished_at
        with self._store.lock:
            job = self._store.jobs[job_id]
            attempt = self._store.attempts[attempt_id]
            if (
                job.state is not JobState.RUNNING
                or attempt.provider_outcome is ProviderCallOutcome.STARTED
            ):
                raise RuntimeError("analysis state conflict")
            attempt.validation_outcome = error.validation_outcome
            job.state = JobState.FAILED
            job.error_code = error.code
            job.error_message = error.message

    def queue_failed_job(self, ecn_row_id: str) -> RetryQueueResult:
        """Requeue a failed job while retaining any prior accepted analysis."""
        with self._store.lock:
            entry = self._store.ecns.get(ecn_row_id)
            if entry is None:
                return RetryQueueResult(RetryQueueStatus.ECN_NOT_FOUND, None)
            job = self._store.jobs[entry.job_id]
            if job.state is not JobState.FAILED:
                return RetryQueueResult(RetryQueueStatus.NOT_RETRYABLE, job.job_id)
            job.state = JobState.QUEUED
            job.error_code = None
            job.error_message = None
            return RetryQueueResult(RetryQueueStatus.QUEUED, job.job_id)

    def block_queued_job(self, job_id: str, error_code: str) -> bool:
        """Hold one queued job after a cached provider preflight failure."""
        with self._store.lock:
            job = self._store.jobs[job_id]
            if job.state is not JobState.QUEUED:
                return False
            job.state = JobState.BLOCKED
            job.error_code = error_code
            job.error_message = "AI generation is blocked by provider preflight"
            return True

    def queue_provider_blocked_jobs(self) -> tuple[str, ...]:
        """Requeue every provider-blocked job in stable identifier order."""
        queued: list[str] = []
        with self._store.lock:
            for job_id in sorted(self._store.jobs):
                job = self._store.jobs[job_id]
                if job.state is not JobState.BLOCKED:
                    continue
                job.state = JobState.QUEUED
                job.error_code = None
                job.error_message = None
                queued.append(job_id)
        return tuple(queued)

    def get_progress(self, upload_id: str | None = None) -> AnalysisProgressResponse:
        """Aggregate mutually exclusive job states, optionally by upload."""
        with self._store.lock:
            jobs = [
                job
                for job in self._store.jobs.values()
                if upload_id is None or job.upload_id == upload_id
            ]
            ready = sum(job.state is JobState.READY for job in jobs)
            queued = sum(job.state is JobState.QUEUED for job in jobs)
            blocked = sum(job.state is JobState.BLOCKED for job in jobs)
            running = sum(job.state is JobState.RUNNING for job in jobs)
            failed = sum(job.state is JobState.FAILED for job in jobs)
            completed = 0
            gaps = 0
            for job in jobs:
                if job.state is not JobState.COMPLETED:
                    continue
                analysis = self._store.analyses[job.job_id]
                if analysis.completeness is AnalysisCompleteness.COMPLETED_WITH_GAPS:
                    gaps += 1
                else:
                    completed += 1
            return AnalysisProgressResponse(
                ready_count=ready,
                queued_count=queued,
                blocked_count=blocked,
                running_count=running,
                completed_count=completed,
                completed_with_gaps_count=gaps,
                failed_count=failed,
                unfinished_count=queued + blocked + running,
                total_count=len(jobs),
            )

    def fail_interrupted_jobs(self, finished_at: datetime) -> int:
        """Fail queued/running jobs and interrupt only unfinished attempts."""
        del finished_at
        with self._store.lock:
            changed = 0
            for attempt in self._store.attempts.values():
                if attempt.provider_outcome is ProviderCallOutcome.STARTED:
                    attempt.provider_outcome = ProviderCallOutcome.INTERRUPTED
                    attempt.validation_outcome = ValidationOutcome.NOT_RUN
            for job in self._store.jobs.values():
                if job.state in (JobState.QUEUED, JobState.RUNNING):
                    job.state = JobState.FAILED
                    job.error_code = "PROCESS_RESTARTED"
                    job.error_message = "Analysis was interrupted by a process restart"
                    changed += 1
            return changed

    def seed_interrupted_jobs(self) -> None:
        """Turn two queued fixture jobs into canonical queued/running leftovers."""
        with self._store.lock:
            jobs = sorted(
                self._store.jobs.values(),
                key=lambda job: self._store.ecns[job.ecn_row_id].record.source_ordinal,
            )
            if len(jobs) != 2 or any(job.state is not JobState.READY for job in jobs):
                raise RuntimeError("restart seeding requires exactly two queued jobs")
            jobs[0].state = JobState.QUEUED
            running = jobs[1]
            config = parse_pinned_generation_config(
                running.prompt_version,
                running.output_schema_version,
                running.model_name,
                running.model_config_json,
            )
            running.state = JobState.RUNNING
            running.attempt_count = 1
            attempt = AttemptStart(
                attempt_id="77777777-7777-4777-8777-777777777777",
                correlation_id="restart-leftover",
                started_at=datetime(2026, 9, 14, 11, 0, tzinfo=timezone.utc),
                provider="sap-ai-core",
                model_name=config.model_name,
                model_deployment=None,
                llm_endpoint="responses",
            )
            self._store.attempts[attempt.attempt_id] = MemoryAttempt(
                job_id=running.job_id,
                attempt_number=1,
                start=attempt,
            )


class MemoryHistoryRepository:
    """Delete idle workflow history while preserving conditional rules."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind the shared workflow store."""
        self._store = store

    def clear_history_if_idle(self) -> HistoryDeletionCounts | None:
        """Return no result for active work or dependency-ordered counts."""
        with self._store.lock:
            if any(
                job.state in (JobState.QUEUED, JobState.RUNNING)
                for job in self._store.jobs.values()
            ):
                return None
            reference_artifact_ids = {
                item.artifact_id for item in self._store.reference_datasets
            }
            counts = HistoryDeletionCounts(
                attempts=len(self._store.attempts),
                analyses=len(self._store.analyses),
                jobs=len(self._store.jobs),
                ecns=len(self._store.ecns),
                uploads=len(self._store.uploads),
                bom_rows=sum(
                    len(rows)
                    for artifact_id, rows in self._store.bom_rows.items()
                    if artifact_id not in reference_artifact_ids
                ),
                source_artifacts=sum(
                    artifact.artifact_kind is ArtifactKind.ECN_JSON
                    or (
                        artifact.artifact_kind is ArtifactKind.BOM_WORKBOOK
                        and artifact_id not in reference_artifact_ids
                    )
                    for artifact_id, artifact in self._store.artifacts.items()
                ),
            )
            self._store.attempts.clear()
            self._store.stage_results.clear()
            self._store.analyses.clear()
            self._store.jobs.clear()
            self._store.ecns.clear()
            self._store.uploads.clear()
            self._store.upload_dataset_ids.clear()
            self._store.request_to_upload.clear()
            self._store.bom_rows = {
                artifact_id: rows
                for artifact_id, rows in self._store.bom_rows.items()
                if artifact_id in reference_artifact_ids
            }
            self._store.artifacts = {
                artifact_id: artifact
                for artifact_id, artifact in self._store.artifacts.items()
                if artifact_id in self._store.rule_artifact_ids
                or artifact_id in reference_artifact_ids
            }
            return counts


class MemoryEngine:
    """Provide the inert engine lifecycle shape required by ServiceContainer."""

    def dispose(self) -> None:
        """Perform no work because the integration harness owns no connections."""


class MemorySnapshotService:
    """Build the real deterministic snapshot from the harness's pinned sources."""

    def __init__(self, store: MemoryWorkflowStore) -> None:
        """Bind in-memory source and normalized BOM records."""
        self._store = store

    def get(self, ecn_row_id: str) -> dict:
        """Return complete snapshot facts and deterministic issues for one ECN."""
        with self._store.lock:
            entry = self._store.ecns[ecn_row_id]
            upload = self._store.uploads[entry.record.upload_id]
            ecn = replace(_normalize_ecn(
                entry.record.source_json, upload.ecn_artifact_id,
                entry.record.source_ordinal,
            ), ecn_row_id=ecn_row_id)
            rows = self._store.bom_rows.get(upload.bom_artifact_id, [])
            return build_snapshot(
                ecn, rows, evidence_hash=entry.record.evidence_hash,
            )

    def overview(self, ecn_row_id: str) -> dict:
        """Return the source snapshot without its detailed object graph."""
        return {key: value for key, value in self.get(ecn_row_id).items()
                if key not in {"objects", "changes"}}

    def clear(self) -> None:
        """Clear no state because this snapshot adapter has no cache."""


class MemoryRdfProjectionService:
    """Provide deterministic current/failed projection states for HTTP workflow tests.

    The harness intentionally does not emulate RDF storage. Unit and opt-in
    HANA tests own that concern. This fake verifies the production route and
    generation gates that consume durable projection state.
    """

    def __init__(self, fail_projections: bool = False) -> None:
        """Initialize empty state and optionally fail every scheduled rebuild."""
        self.fail_projections = fail_projections
        self.statuses: dict[str, ProjectionState] = {}
        self.project_many_calls: list[tuple[str, ...]] = []

    @staticmethod
    def _record(ecn_row_id: str, status: ProjectionState) -> ProjectionRecord:
        """Build the minimum safe status record expected by diagnostic routes."""
        is_current = status is ProjectionState.CURRENT
        is_failed = status is ProjectionState.FAILED
        now = datetime.now(timezone.utc) if is_current else None
        return ProjectionRecord(
            ecn_row_id=ecn_row_id,
            source_sha256="0" * 64,
            projection_version=PROJECTION_VERSION,
            status=status,
            expected_counts=ProjectionExpectedCounts(0, 0, 0, 0, 0),
            verified_node_count=0 if is_current else None,
            verified_triple_count=0 if is_current else None,
            error_code="RDF_PROJECTION_FAILED" if is_failed else None,
            error_message="The RDF projection could not be rebuilt and verified."
            if is_failed
            else None,
            last_attempt_at=now,
            current_at=now,
        )

    def project(self, ecn_row_id: str) -> ProjectionRecord:
        """Complete one trusted rebuild with its configured deterministic outcome."""
        status = (
            ProjectionState.FAILED
            if self.fail_projections
            else ProjectionState.CURRENT
        )
        self.statuses[ecn_row_id] = status
        return self._record(ecn_row_id, status)

    def project_many(self, ecn_row_ids: tuple[str, ...]) -> ProjectionBatchResult:
        """Record post-commit scheduling and project each supplied ECN independently."""
        self.project_many_calls.append(tuple(ecn_row_ids))
        records = tuple(self.project(ecn_row_id) for ecn_row_id in ecn_row_ids)
        return ProjectionBatchResult(
            requested=len(records),
            current=sum(record.status is ProjectionState.CURRENT for record in records),
            failed=sum(record.status is ProjectionState.FAILED for record in records),
        )

    def get_projection(self, ecn_row_id: str) -> ProjectionRecord | None:
        """Return the fake durable status only after a scheduled projection exists."""
        status = self.statuses.get(ecn_row_id)
        return None if status is None else self._record(ecn_row_id, status)

    def request_rebuild(self, ecn_row_id: str) -> ProjectionRecord:
        """Mark one existing fake projection pending before its route background task."""
        if ecn_row_id not in self.statuses:
            raise PersistenceStateError(
                "RDF_PROJECTION_NOT_FOUND",
                "The selected ECN projection was not found.",
                status_code=404,
            )
        self.statuses[ecn_row_id] = ProjectionState.PENDING
        return self._record(ecn_row_id, ProjectionState.PENDING)

    def require_current(self, ecn_row_id: str) -> ProjectionRecord:
        """Reject graph-dependent work unless the selected projection is current."""
        status = self.statuses.get(ecn_row_id, ProjectionState.PENDING)
        if status is not ProjectionState.CURRENT:
            raise PersistenceStateError(
                f"RDF_PROJECTION_{status.value}",
                "The ECN RDF projection is not ready for graph-dependent work.",
                status_code=409,
            )
        return self._record(ecn_row_id, status)

    def require_all_current(self, ecn_row_ids: tuple[str, ...]) -> None:
        """Apply the same gate to every summary-selection member."""
        for ecn_row_id in ecn_row_ids:
            self.require_current(ecn_row_id)

    def clear_all(self) -> None:
        """Remove fake statuses after the in-memory history reset completes."""
        self.statuses.clear()


class MemoryAssistant:
    """Return a harmless deterministic chat result after the real route gate passes."""

    async def ainvoke(
        self, message: str, ecn_row_id: str, history: list[dict[str, str]], actor_id: str = ""
    ) -> dict[str, object]:
        """Echo the selected scope without interpreting source content or using a model."""
        del message, history, actor_id
        return {
            "assessment_id": ecn_row_id,
            "answer": "Projection-backed chat is available.",
            "consulted_objects": [],
            "model_calls": 0,
            "correlation_id": "in-memory-assistant",
        }


def require_success(response: Response) -> Response:
    """Raise a deterministic test error carrying the API's safe envelope."""
    if response.status_code >= 400:
        payload = response.json()
        raise RuntimeError(f"{payload['code']}: {payload['message']}")
    return response


@dataclass
class WorkflowHarness:
    """Expose production service/API operations over in-memory repositories."""

    client: AsyncClient
    services: ServiceContainer
    provider: ControlledImpactProvider
    rdf: MemoryRdfProjectionService
    store: MemoryWorkflowStore
    initial_rule_artifact_ids: frozenset[str]

    async def upload(
        self, request_id: UUID, ecn_bytes: bytes, bom_bytes: bytes | None
    ) -> UploadResponse:
        """Publish an optional Settings BOM, then post JSON-only ECN intake."""
        if bom_bytes is not None:
            reference_response = await self.client.post(
                "/api/reference-datasets",
                data={"dataset_type": "BOM"},
                files={
                    "source_file": (
                        "bom.xlsx",
                        bom_bytes,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )
            require_success(reference_response)
        response = await self.client.post(
            "/api/uploads",
            data={"request_id": str(request_id)},
            files={"ecn_file": ("ecn.json", ecn_bytes, "application/json")},
        )
        return UploadResponse.model_validate_json(require_success(response).text)

    async def generate_selected(self):
        """Explicitly request generation for currently ready fixture jobs through the real route."""
        ids = [job.ecn_row_id for job in self.store.jobs.values() if job.state is JobState.READY]
        if ids:
            require_success(await self.client.post("/api/summary-batches", json={
                "request_id": str(uuid4()),
                "selection": {"mode": "explicit", "ecn_ids": ids},
            }))

    async def list_ecns(self, **query: Any) -> EcnListResponse:
        """Get one ECN list page through the real HTTP route."""
        response = await self.client.get("/api/ecns", params=query)
        return EcnListResponse.model_validate_json(require_success(response).text)

    async def get_detail(self, ecn_row_id: UUID | str) -> EcnDetailResponse:
        """Get one exact ECN detail through the real HTTP route."""
        response = await self.client.get(f"/api/ecns/{ecn_row_id}")
        return EcnDetailResponse.model_validate_json(require_success(response).text)

    async def get_evidence(
        self, ecn_row_id: UUID | str, evidence_id: str
    ) -> BusinessEvidenceDetailResponse:
        """Get one same-ECN evidence item through the real HTTP route."""
        response = await self.client.get(
            f"/api/ecns/{ecn_row_id}/evidence/{evidence_id}"
        )
        return BusinessEvidenceDetailResponse.model_validate_json(
            require_success(response).text
        )

    async def list_evidence(
        self,
        ecn_row_id: UUID | str,
        section: BusinessEvidenceSection,
        *,
        include_technical: bool = False,
    ) -> BusinessEvidencePageResponse:
        """Get one lazy business evidence page through the real HTTP route."""
        response = await self.client.get(
            f"/api/ecns/{ecn_row_id}/evidence",
            params={
                "section": section.value,
                "page": 1,
                "page_size": 100,
                "include_technical": str(include_technical).lower(),
            },
        )
        return BusinessEvidencePageResponse.model_validate_json(
            require_success(response).text
        )

    async def progress(
        self, upload_id: UUID | str | None = None
    ) -> ApiAnalysisProgressResponse:
        """Get global or upload-scoped generation progress through HTTP."""
        params = {"upload_id": str(upload_id)} if upload_id is not None else None
        response = await self.client.get("/api/analysis-progress", params=params)
        return ApiAnalysisProgressResponse.model_validate_json(
            require_success(response).text
        )

    async def retry(self, ecn_row_id: UUID | str) -> RetryResponse:
        """Post a failed-only retry through the real HTTP route."""
        response = await self.client.post(
            f"/api/ecns/{ecn_row_id}/analysis/retry"
        )
        return RetryResponse.model_validate_json(require_success(response).text)

    async def reset(self, confirmation: str) -> HistoryResetResponse:
        """Delete idle history through the real protected HTTP route."""
        response = await self.client.request(
            "DELETE", "/api/history", json={"confirmation": confirmation}
        )
        return HistoryResetResponse.model_validate_json(
            require_success(response).text
        )

    async def wait_until_idle(self) -> None:
        """Wait until the real generation runtime has no tracked tasks."""
        await self.services.runtime.wait_until_idle()

    async def recover_interrupted(self, finished_at: datetime) -> int:
        """Invoke the production restart-recovery repository contract."""
        return self.services.analysis_repository.fail_interrupted_jobs(finished_at)


@asynccontextmanager
async def build_workflow_harness(
    *,
    hold_provider: bool = False,
    failing_calls: set[int] | None = None,
    response_factory: Callable[[Any], str] | None = None,
    seed_restart_leftovers: bool = False,
    fail_rdf_projections: bool = False,
) -> AsyncIterator[WorkflowHarness]:
    """Build one-loop production services with in-memory persistence/provider."""
    store = MemoryWorkflowStore()
    normalized_rules = json.dumps(
        {"format_version": "ecn-rules-normalized-v1", "rules": []},
        separators=(",", ":"),
    )
    rule_bytes = normalized_rules.encode("utf-8")
    rule_artifact = SourceArtifact(
        artifact_id="66666666-6666-4666-8666-666666666666",
        artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
        file_name="conditional-rules.xlsx",
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        byte_size=len(rule_bytes),
        sha256=hashlib.sha256(rule_bytes).hexdigest(),
        content_bytes=rule_bytes,
        normalized_content=normalized_rules,
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
    )
    store.artifacts[rule_artifact.artifact_id] = rule_artifact
    store.rule_artifact_ids.add(rule_artifact.artifact_id)
    store.reference_datasets.append(
        ReferenceDatasetRecord(
            dataset_id="66666666-6666-4666-8666-666666666667",
            dataset_type=ReferenceDatasetType.CONDITIONAL_RULES,
            artifact_id=rule_artifact.artifact_id,
            publication_order=1,
            file_name=rule_artifact.file_name,
            sha256=rule_artifact.sha256,
            scope_code=None,
            scope_name=None,
            row_count=0,
            warning_counts={},
            parser_version="ecn-reference-v1",
            created_at=rule_artifact.created_at,
            active=True,
        )
    )
    store.reference_publication_order = 1
    provider = ControlledImpactProvider(
        failing_calls=failing_calls,
        response_factory=response_factory,
        hold=hold_provider,
    )
    rdf_projection_service = MemoryRdfProjectionService(fail_rdf_projections)
    settings = EcnSettings()
    upload_repository = MemoryUploadRepository(store)
    artifact_repository = MemoryArtifactRepository(store)
    reference_dataset_repository = MemoryReferenceDatasetRepository(store)
    ecn_repository = MemoryEcnRepository(store)
    analysis_repository = MemoryAnalysisRepository(store)
    snapshot_service = MemorySnapshotService(store)
    mutation_guard = HistoryMutationGuard()
    clock = SystemClock()
    ingestion_service = IngestionService(
        upload_repository=cast(Any, upload_repository),
        artifact_repository=cast(Any, artifact_repository),
        settings=settings,
        preparation_gate=IngestionPreparationGate(),
        mutation_guard=mutation_guard,
        clock=lambda: datetime.now(timezone.utc),
        uuid_factory=lambda: str(uuid4()),
        reference_repository=cast(Any, reference_dataset_repository),
    )
    reference_dataset_service = ReferenceDatasetService(
        cast(Any, reference_dataset_repository),
        settings,
        clock=lambda: datetime.now(timezone.utc),
        uuid_factory=lambda: str(uuid4()),
        mutation_guard=mutation_guard,
    )
    generation_service = AnalysisGenerationService(
        cast(Any, analysis_repository),
        provider,
        lambda **_fields: None,
        clock,
        uuid4,
        snapshot_service=snapshot_service,
        rdf_projection_service=rdf_projection_service,
    )
    runtime = GenerationRuntime(
        generation_service,
        cast(Any, analysis_repository),
        mutation_guard,
        min(settings.llm_max_concurrency, 2),
        clock,
        uuid4,
        provider_max_concurrency=settings.llm_max_concurrency,
    )
    services = ServiceContainer(
        settings=settings,
        ingestion_service=ingestion_service,
        history_service=HistoryService(
            cast(Any, MemoryHistoryRepository(store)), mutation_guard
        ),
        retry_service=RetryService(cast(Any, analysis_repository), mutation_guard),
        ecn_repository=cast(Any, ecn_repository),
        snapshot_service=cast(Any, snapshot_service),
        analysis_repository=cast(Any, analysis_repository),
        runtime=runtime,
        engine=cast(Engine, MemoryEngine()),
        assistant_service=MemoryAssistant(),
        workspace_repository=MemoryWorkspaceRepository(store),
        reference_dataset_service=reference_dataset_service,
        reference_dataset_repository=cast(Any, reference_dataset_repository),
        rdf_projection_service=cast(Any, rdf_projection_service),
    )
    if seed_restart_leftovers:
        from tests.fixtures.builders import (
            minimal_bom_workbook_bytes,
            minimal_ecn_json_bytes,
        )

        from app.ecn.services.ingestion import UploadBytesRequest, UploadFileBytes

        reference_dataset_service.publish(
            ReferenceUploadRequest(
                ReferenceDatasetType.BOM,
                "restart-bom.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                minimal_bom_workbook_bytes(),
            )
        )
        request = UploadBytesRequest(
            request_id=UUID("88888888-8888-4888-8888-888888888888"),
            ecn_file=UploadFileBytes(
                "restart-ecn.json", "application/json", minimal_ecn_json_bytes()
            ),
            bom_file=None,
            actor_id=None,
        )
        ingestion_service.ingest_upload(request)
        analysis_repository.seed_interrupted_jobs()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[ServiceContainer]:
        """Yield the prebuilt service graph and close tracked work afterward."""
        try:
            yield services
        finally:
            await runtime.close()

    application = create_app(service_lifespan_factory=lifespan)
    application.dependency_overrides[get_api_key] = lambda: None
    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application, raise_app_exceptions=False),
            base_url="http://integration.test",
        ) as client:
            yield WorkflowHarness(
                client=client,
                services=services,
                provider=provider,
                rdf=rdf_projection_service,
                store=store,
                initial_rule_artifact_ids=frozenset(store.rule_artifact_ids),
            )
