"""Prepare deterministic ECN evidence and publish upload aggregates atomically."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from uuid import UUID, uuid5

from app.ecn.errors import PersistenceStateError, SourceValidationError
from app.ecn.evidence.builder import (
    build_ecn_evidence,
    build_rule_index,
    derive_impact_counts,
    select_and_evaluate_relevant_rules,
)
from app.ecn.evidence.change_candidates import derive_change_candidates
from app.ecn.evidence.matching import build_bom_indexes, match_ecn_to_bom
from app.ecn.intake.json_parser import parse_ecn_json
from app.ecn.intake.workbook_parser import parse_bom_workbook
from app.ecn.plant_settings import DEFAULT_EXCLUDED_RULE_STATES, PlantSettings
from app.ecn.models import (
    DETERMINISTIC_LOGIC_VERSION,
    OUTPUT_SCHEMA_VERSION,
    PARSER_VERSION,
    PROMPT_VERSION,
    RULE_EVALUATOR_VERSION,
    ArtifactKind,
    BomRow,
    EvidenceContext,
    ParsedBomWorkbook,
    ParseState,
    SourceArtifact,
    canonical_uuid,
)
from app.ecn.rules.importer import load_parsed_rules
from app.ecn.rules.models import ParsedRule
from app.ecn.reference_data.models import ReferenceDatasetPin, ReferenceDatasetType
from app.ecn.services.lifecycle import HistoryMutationGuard, IngestionPreparationGate
from app.ecn.settings import EcnSettings, canonical_model_config
from app.persistence.artifact_repository import ArtifactRepository
from app.persistence.records import EcnPublication, UploadPublication, UploadRecord
from app.persistence.upload_repository import UploadRepository
from app.persistence.plant_settings_repository import PlantSettingsRepository
from app.persistence.reference_dataset_repository import ReferenceDatasetRepository


@dataclass(frozen=True)
class UploadFileBytes:
    """Hold one bounded browser-uploaded file after transport reading.

    Args:
        file_name: Client-supplied base file name retained as provenance.
        media_type: Validated transport media type.
        content: Complete bounded source bytes.
    """

    file_name: str
    media_type: str
    content: bytes


@dataclass(frozen=True)
class UploadBytesRequest:
    """Hold one idempotent ECN upload with an optional replacement BOM workbook.

    Args:
        request_id: Client idempotency UUID.
        ecn_file: Required ECN JSON upload.
        bom_file: Optional replacement BOM workbook; absent means reuse latest.
        actor_id: Optional opaque, already-hashed actor identifier.
    """

    request_id: UUID
    ecn_file: UploadFileBytes
    bom_file: UploadFileBytes | None
    actor_id: str | None
    registered_artifact: SourceArtifact | None = None
    source_positions: tuple[int, ...] | None = None
    assessment_revision: UUID | None = None
    assessment_date: date | None = None
    data_group: str = "customer"


@dataclass(frozen=True)
class IngestionResult:
    """Return a publication/replay and only newly committed jobs to schedule.

    Args:
        upload: Authoritative committed upload record.
        created: Whether this request created the upload.
        queued_job_ids: Newly committed job IDs in ECN source order, or empty
            for replay and race-loss results.
        projection_ecn_ids: Newly committed ECN IDs whose durable projection
            statuses are ready for post-commit graph scheduling, or empty for
            replay and race-loss results.
    """

    upload: UploadRecord
    created: bool
    queued_job_ids: tuple[str, ...]
    projection_ecn_ids: tuple[str, ...] = ()


def _source_artifact(
    *,
    artifact_id: str,
    kind: ArtifactKind,
    uploaded: UploadFileBytes,
    normalized_content: str,
    created_at: datetime,
) -> SourceArtifact:
    """Build one immutable valid source artifact after pure parsing succeeds.

    Args:
        artifact_id: Canonical generated artifact UUID.
        kind: Source artifact classification.
        uploaded: Original bounded upload bytes and metadata.
        normalized_content: Exact parser output or compact BOM manifest.
        created_at: Shared publication timestamp.

    Returns:
        Valid immutable source artifact ready for aggregate publication.
    """
    return SourceArtifact(
        artifact_id=artifact_id,
        artifact_kind=kind,
        file_name=uploaded.file_name,
        media_type=uploaded.media_type,
        byte_size=len(uploaded.content),
        sha256=hashlib.sha256(uploaded.content).hexdigest(),
        content_bytes=uploaded.content,
        normalized_content=normalized_content,
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=created_at,
    )


def _bom_manifest(parsed_bom: ParsedBomWorkbook) -> str:
    """Serialize workbook metadata while normalized rows remain in their table.

    Args:
        parsed_bom: Fully parsed replacement workbook.

    Returns:
        Compact deterministic JSON manifest without local paths or row copies.
    """
    return json.dumps(
        {
            "format_version": "ecn-bom-manifest-v1",
            "parser_version": PARSER_VERSION,
            "row_count": parsed_bom.row_count,
            "cell_count": parsed_bom.cell_count,
            "sheet_roles": {
                role.value: sheet_name
                for role, sheet_name in parsed_bom.sheet_roles.items()
            },
            "quality_qualifiers": list(parsed_bom.quality_qualifiers),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stored_bom_rows(
    repository: ArtifactRepository, artifact_id: str
) -> tuple[BomRow, ...]:
    """Load persisted normalized BOM rows in bounded repository batches.

    Args:
        repository: Artifact read repository.
        artifact_id: Immutable BOM snapshot identifier.

    Returns:
        All normalized rows for compact in-memory index construction.
    """
    return tuple(
        row
        for batch in repository.iter_bom_rows(artifact_id, batch_size=5_000)
        for row in batch
    )


def _load_rules(
    repository: ArtifactRepository,
) -> tuple[SourceArtifact | None, tuple[ParsedRule, ...]]:
    """Load the current configuration artifact and its stored ASTs, if present.

    Args:
        repository: Artifact read repository.

    Returns:
        Current rule artifact and strict decoded rules, or an empty rule tuple.
    """
    artifact = repository.get_current_rule_artifact()
    if artifact is None or artifact.normalized_content is None:
        return artifact, ()
    return artifact, load_parsed_rules(artifact.normalized_content)


class IngestionService:
    """Prepare evidence and atomically publish idempotent upload aggregates."""

    def __init__(
        self,
        *,
        upload_repository: UploadRepository,
        artifact_repository: ArtifactRepository,
        settings: EcnSettings,
        preparation_gate: IngestionPreparationGate,
        mutation_guard: HistoryMutationGuard,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], str],
        reference_repository: ReferenceDatasetRepository | None = None,
        plant_settings_repository: PlantSettingsRepository | None = None,
    ) -> None:
        """Bind repositories, immutable settings, guards, clock, and ID source.

        Args:
            upload_repository: Atomic upload publication boundary.
            artifact_repository: BOM-row and current-rule read boundary.
            settings: Validated runtime limits and model configuration.
            preparation_gate: Shared capacity-one parsing/indexing permit.
            mutation_guard: Shared short lifecycle mutation guard.
            clock: Timezone-aware publication clock.
            uuid_factory: Canonical UUID string factory.
            reference_repository: Settings-managed dataset publications.
            plant_settings_repository: Plant Code Mapping read at import; the
                value is frozen into every ECN assessment of the upload.
        """
        self._uploads = upload_repository
        self._artifacts = artifact_repository
        self._settings = settings
        self._preparation = preparation_gate
        self._guard = mutation_guard
        self._clock = clock
        self._uuid = uuid_factory
        self._references = reference_repository
        self._plant_settings = plant_settings_repository

    def prepare_upload(self, request: UploadBytesRequest) -> UploadPublication:
        """Parse and derive the exact publication aggregate without HANA writes.

        Args:
            request: Bounded upload bytes plus idempotency and actor metadata.

        Returns:
            Fully prepared immutable aggregate for one repository transaction.

        Raises:
            SourceValidationError: If a first upload omits the required BOM or
                a source parser rejects the supplied bytes.
        """
        created_at = self._clock()
        upload_id = self._uuid()
        ecn_artifact_id = request.registered_artifact.artifact_id if request.registered_artifact else self._uuid()
        parsed_ecn = parse_ecn_json(request.ecn_file.content, ecn_artifact_id, **({"source_positions": request.source_positions} if request.source_positions is not None else {}))
        normalized_ecns = parsed_ecn.ecns
        if request.registered_artifact:
            normalized_ecns = tuple(replace(e, ecn_row_id=str(uuid5(UUID(ecn_artifact_id), f"{e.source_ordinal}:{request.assessment_revision}" if request.assessment_revision else str(e.source_ordinal)))) for e in normalized_ecns)
        ecn_artifact = request.registered_artifact or _source_artifact(
            artifact_id=ecn_artifact_id,
            kind=ArtifactKind.ECN_JSON,
            uploaded=request.ecn_file,
            normalized_content=parsed_ecn.normalized_content,
            created_at=created_at,
        )
        dataset_links: tuple[ReferenceDatasetPin, ...] = ()
        dataset_context = ()
        if self._references is not None:
            if request.bom_file is not None:
                raise SourceValidationError(
                    "BOM_UPLOAD_MOVED_TO_SETTINGS",
                    "BOM workbooks must be published from Settings before ECN upload.",
                    status_code=400,
                )
            from app.ecn.reference_data.scopes import select_assessment_datasets
            records = self._references.list()
            try:
                dataset_context = select_assessment_datasets(records, request.data_group)
            except ValueError as exc:
                raise SourceValidationError("DATASET_SELECTION_INVALID", str(exc), status_code=400) from exc
            active_ids = {d["dataset_id"] for d in dataset_context if not d["comparison"]}
            active = tuple(item for item in records if item.dataset_id in active_ids)
            active_boms = tuple(item for item in active if item.dataset_type is ReferenceDatasetType.BOM)
            active_bom = active_boms[0] if active_boms else None
            if active_bom is None:
                raise SourceValidationError(
                    "NO_ACTIVE_BOM",
                    "Publish a validated BOM workbook in Settings before uploading ECNs.",
                    status_code=400,
                )
            bom_artifact = self._artifacts.get_artifact(active_bom.artifact_id)
            if bom_artifact is None:
                raise PersistenceStateError(
                    "REFERENCE_ARTIFACT_MISSING",
                    "The active BOM source is unavailable; publish it again in Settings.",
                    status_code=409,
                )
            new_bom_rows = ()
            bom_rows = tuple(row for item in active_boms
                             for row in _stored_bom_rows(self._artifacts, item.artifact_id))
            bom_reused = True
            rule_artifacts = [self._artifacts.get_artifact(item.artifact_id) for item in active
                              if item.dataset_type is ReferenceDatasetType.CONDITIONAL_RULES]
            rule_artifacts = [artifact for artifact in rule_artifacts if artifact is not None]
            rule_artifact = rule_artifacts[0] if rule_artifacts else None
            # Legacy upload columns retain one primary source per type. The full
            # current/comparison pin set is immutable inside the hashed evidence context.
            primary = {}
            for item in active:
                primary.setdefault(item.dataset_type, item)
            dataset_links = tuple(ReferenceDatasetPin(item.dataset_type, item.dataset_id) for item in primary.values())
        elif request.bom_file is not None:
            bom_artifact_id = self._uuid()
            parsed_bom = parse_bom_workbook(
                request.bom_file.content,
                bom_artifact_id,
                self._settings.max_workbook_rows,
                self._settings.max_workbook_cells,
            )
            bom_artifact = _source_artifact(
                artifact_id=bom_artifact_id,
                kind=ArtifactKind.BOM_WORKBOOK,
                uploaded=request.bom_file,
                normalized_content=_bom_manifest(parsed_bom),
                created_at=created_at,
            )
            new_bom_rows = parsed_bom.rows
            bom_rows = parsed_bom.rows
            bom_reused = False
        else:
            bom_artifact = self._uploads.get_latest_bom_snapshot()
            if bom_artifact is None:
                raise SourceValidationError(
                    "BOM_SNAPSHOT_REQUIRED",
                    "The first successful upload requires an EBOM/MBOM workbook.",
                    status_code=400,
                )
            new_bom_rows = ()
            bom_rows = _stored_bom_rows(
                self._artifacts, bom_artifact.artifact_id
            )
            bom_reused = True

        indexes = build_bom_indexes(bom_rows)
        if self._references is None:
            rule_artifact, _rules = _load_rules(self._artifacts)
            rule_artifacts = [rule_artifact] if rule_artifact else []
        rule_sources = {}
        rule_indexes = []
        for artifact in rule_artifacts:
            artifact_rules = load_parsed_rules(artifact.normalized_content) if artifact.normalized_content else ()
            rule_sources.update({rule.rule_id: artifact.artifact_id for rule in artifact_rules})
            pin = next((d for d in dataset_context if d["artifact_id"] == artifact.artifact_id), {})
            allowed_boms = {d["artifact_id"] for d in dataset_context if d["dataset_type"] == "BOM" and not d.get("comparison")
                            and all(not pin.get(key) or pin[key] == d.get(key) for key in ("model_range", "environment", "data_group"))}
            rule_indexes.append((build_rule_index(artifact_rules), allowed_boms if dataset_context else None))
        publications: list[EcnPublication] = []
        context_base = {
            "upload_id": upload_id,
            "ecn_artifact_id": ecn_artifact.artifact_id,
            "bom_artifact_id": bom_artifact.artifact_id,
            "rule_artifact_id": (
                rule_artifact.artifact_id if rule_artifact else None
            ),
            "parser_version": PARSER_VERSION,
            "deterministic_logic_version": DETERMINISTIC_LOGIC_VERSION,
            "rule_evaluator_version": RULE_EVALUATOR_VERSION,
            "analysis_as_of_date": request.assessment_date or created_at.date(),
            "dataset_context": dataset_context,
            "data_group": request.data_group,
            "rule_source_artifact_ids": rule_sources,
            # Frozen now: later Settings edits need "Reassess with current data".
            "plant_assignment": (
                self._plant_settings.get() if self._plant_settings is not None
                else PlantSettings((), DEFAULT_EXCLUDED_RULE_STATES)
            ).frozen_context(),
        }
        for ecn in normalized_ecns:
            matches = match_ecn_to_bom(ecn, indexes)
            candidates = derive_change_candidates(ecn)
            rule_results = tuple(result for scoped_index, allowed_boms in rule_indexes
                for result in select_and_evaluate_relevant_rules(
                    replace(matches, enriched_matches=tuple(m for m in matches.enriched_matches
                        if allowed_boms is None or m.occurrence.artifact_id in allowed_boms)), scoped_index, {}, None))
            evidence = build_ecn_evidence(
                ecn,
                matches,
                rule_results,
                EvidenceContext(**context_base),
            )
            counts = derive_impact_counts(ecn, matches, evidence)
            publications.append(
                EcnPublication(
                    ecn=ecn,
                    change_candidates=candidates,
                    evidence=evidence,
                    counts=counts,
                    job_id=self._uuid(),
                    prompt_version=PROMPT_VERSION,
                    output_schema_version=OUTPUT_SCHEMA_VERSION,
                    model_name=self._settings.llm_model_name,
                    model_config_json=canonical_model_config(self._settings),
                )
            )
        return UploadPublication(
            upload_id=upload_id,
            request_id=canonical_uuid(request.request_id),
            ecn_artifact=ecn_artifact,
            bom_artifact=bom_artifact,
            new_bom_rows=tuple(new_bom_rows),
            rule_artifact=rule_artifact,
            bom_reused=bom_reused,
            ecns=tuple(publications),
            actor_id=request.actor_id,
            parser_version=PARSER_VERSION,
            deterministic_logic_version=DETERMINISTIC_LOGIC_VERSION,
            rule_evaluator_version=RULE_EVALUATOR_VERSION,
            created_at=created_at,
            dataset_links=dataset_links,
            ecn_reused=request.registered_artifact is not None,
        )

    def ingest_upload(self, request: UploadBytesRequest) -> IngestionResult:
        """Replay or prepare and atomically publish one upload.

        Args:
            request: Bounded upload bytes and idempotency metadata.

        Returns:
            Authoritative publication result. Only a newly created result
            carries the exact committed job IDs for post-commit scheduling.
        """
        request_id = canonical_uuid(request.request_id)
        existing = self._uploads.find_by_request_id(request_id)
        if existing is not None:
            return IngestionResult(existing, False, ())
        with self._preparation.hold():
            existing = self._uploads.find_by_request_id(request_id)
            if existing is not None:
                return IngestionResult(existing, False, ())
            publication = self.prepare_upload(request)
        with self._guard.hold():
            existing = self._uploads.find_by_request_id(request_id)
            if existing is not None:
                return IngestionResult(existing, False, ())
            self._ensure_reused_bom_available(publication)
            result = self._uploads.publish_upload(publication)
            queued = (
                tuple(item.job_id for item in publication.ecns)
                if result.created
                else ()
            )
            return IngestionResult(
                result.upload,
                result.created,
                queued,
                result.projection_ecn_ids if result.created else (),
            )

    def _ensure_reused_bom_available(
        self, publication: UploadPublication
    ) -> None:
        """Reject a reused snapshot deleted while preparation was in progress.

        Args:
            publication: Fully prepared aggregate about to be published while
                the shared mutation guard is held.

        Raises:
            PersistenceStateError: If an idle history reset removed the reused
                BOM artifact after preparation selected it.
        """
        if (
            publication.bom_reused
            and not self._artifacts.artifact_exists(
                publication.bom_artifact.artifact_id
            )
        ):
            raise PersistenceStateError(
                "BOM_SNAPSHOT_INVALIDATED",
                "The reused BOM snapshot was reset during upload preparation; "
                "submit the upload again with an EBOM/MBOM workbook.",
                status_code=409,
            )
