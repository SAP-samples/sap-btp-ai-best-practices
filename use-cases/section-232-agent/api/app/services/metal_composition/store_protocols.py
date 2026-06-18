"""Typed contracts for the metal composition persistence stores."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from app.models.metal_composition import (
    MetalCompositionAppSettings,
    MetalCompositionResponse,
)
from app.utils.hana import HANAConnection

from .classification_jobs import (
    ClassificationJobItemSeed,
    ClassificationJobItemStatus,
    ClassificationJobType,
    PersistedClassificationJob,
    PersistedClassificationJobItem,
)
from .hts_catalog_sources import (
    HTSCatalogRefreshState,
    PersistedHTSCatalogSourceFile,
)
from .section_232_rulesets import (
    BatchStatus,
    ReviewDecision,
    Section232CandidatePage,
    Section232CodeDeleteOverride,
    Section232DraftBatch,
    Section232DraftBatchStats,
    Section232DraftRuleCandidate,
    Section232DeletedDraftBatch,
    Section232PublishedRuleset,
    Section232PublishedRulesetInfo,
)
from .section_232_sources import (
    ExtractedSection232Source,
    PersistedSection232Source,
    Section232SourceSnippet,
)
from .ui_state import (
    PersistedAppSettings,
    PersistedClassificationSnapshot,
    StoredDocumentReference,
)


class UIStateStoreProtocol(Protocol):
    def replace_document_assignments(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        document_refs: Sequence[StoredDocumentReference],
    ) -> List[StoredDocumentReference]:
        ...

    def get_document_assignments(
        self,
        item_id: str,
        *,
        dataset_scope: str,
    ) -> List[StoredDocumentReference]:
        ...

    def get_document_assignment_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], List[StoredDocumentReference]]:
        ...

    def get_document_assignment_keys(self) -> Set[Tuple[str, str]]:
        ...

    def save_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        result: MetalCompositionResponse,
        last_classified_at: Optional[str] = None,
    ) -> PersistedClassificationSnapshot:
        ...

    def get_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
    ) -> Optional[PersistedClassificationSnapshot]:
        ...

    def get_classification_snapshot_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], PersistedClassificationSnapshot]:
        ...

    def get_classification_snapshot_keys(self) -> Set[Tuple[str, str]]:
        ...

    def get_classification_stats(self) -> tuple[int, Optional[str]]:
        ...

    def clear_classification_snapshots(self) -> int:
        ...

    def get_app_settings(self) -> PersistedAppSettings:
        ...

    def update_app_settings(
        self,
        *,
        use_material_master_metal_composition: bool,
    ) -> PersistedAppSettings:
        ...


class ClassificationJobStoreProtocol(Protocol):
    def submit_job(
        self,
        *,
        job_type: ClassificationJobType,
        items: Sequence[ClassificationJobItemSeed],
    ) -> PersistedClassificationJob:
        ...

    def get_job(self, job_id: str) -> Optional[PersistedClassificationJob]:
        ...

    def get_job_items(self, job_id: str) -> List[PersistedClassificationJobItem]:
        ...

    def get_active_item_ids(self, keys: Iterable[Tuple[str, str]]) -> List[str]:
        ...

    def supersede_active_items(
        self,
        keys: Iterable[Tuple[str, str]],
        *,
        error_message: str,
    ) -> int:
        ...

    def is_item_owned_by_job(self, item_id: str, dataset_scope: str, job_id: str) -> bool:
        ...

    def claim_next_queued_job(self, *, worker_id: str) -> Optional[PersistedClassificationJob]:
        ...

    def mark_job_items_running(self, job_id: str) -> None:
        ...

    def record_job_item_result(
        self,
        job_id: str,
        *,
        position: int,
        status: ClassificationJobItemStatus,
        error_message: Optional[str] = None,
        last_classified_at: Optional[str] = None,
    ) -> None:
        ...

    def fail_job(self, job_id: str, *, error_message: str) -> None:
        ...

    def cancel_all_active_jobs(self) -> int:
        ...


class Section232SourceStoreProtocol(Protocol):
    connection: HANAConnection | None

    def save_source(
        self,
        *,
        filename: str,
        size_bytes: int,
        extracted: ExtractedSection232Source,
    ) -> PersistedSection232Source:
        ...

    def list_sources(self) -> List[PersistedSection232Source]:
        ...

    def clear_sources(self) -> None:
        ...

    def delete_sources(self, source_ids: Sequence[str]) -> int:
        ...

    def clear_eligible_hts_codes(self) -> None:
        ...

    def has_sources(self) -> bool:
        ...

    def list_eligible_hts_codes(self) -> List[str]:
        ...

    def replace_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        ...

    def append_eligible_hts_codes(self, codes: Sequence[str]) -> List[str]:
        ...

    def retrieve_snippets(
        self,
        *,
        hts_codes: Sequence[str],
        metal_keywords: Sequence[str],
        settings: object,
    ) -> List[Section232SourceSnippet]:
        ...


class Section232RulesetStoreProtocol(Protocol):
    connection: HANAConnection | None

    def create_draft_batch(
        self,
        *,
        source_ids: Sequence[str],
        source_filenames: Sequence[str],
    ) -> Section232DraftBatch:
        ...

    def replace_batch_candidates(
        self,
        batch_id: str,
        candidates: Sequence[Section232DraftRuleCandidate],
    ) -> None:
        ...

    def review_candidate(
        self,
        batch_id: str,
        candidate_id: str,
        *,
        decision: ReviewDecision,
    ) -> Section232DraftRuleCandidate:
        ...

    def review_candidates(
        self,
        batch_id: str,
        candidate_ids: Sequence[str],
        *,
        decision: ReviewDecision,
    ) -> List[Section232DraftRuleCandidate]:
        ...

    def review_all_candidates(
        self,
        batch_id: str,
        *,
        decision: ReviewDecision,
        excluded_candidate_ids: Sequence[str] = (),
    ) -> int:
        ...

    def delete_draft_candidates_by_hts_code(self, batch_id: str, *, hts_code: str) -> int:
        ...

    def delete_pending_draft_batch(self, batch_id: str) -> Section232DeletedDraftBatch:
        ...

    def publish_snapshot(
        self,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate],
    ) -> Section232PublishedRuleset:
        ...

    def publish_batch(
        self,
        batch_id: str,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate] | None = None,
    ) -> Section232PublishedRuleset:
        ...

    def list_active_rules(self) -> List[Section232DraftRuleCandidate]:
        ...

    def list_draft_candidates(self, *, batch_id: str) -> List[Section232DraftRuleCandidate]:
        ...

    def get_draft_batch_stats(self, *, batch_id: str) -> Section232DraftBatchStats:
        ...

    def list_draft_candidate_page(
        self,
        *,
        batch_id: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        ...

    def list_draft_batches(self, *, status: Optional[BatchStatus] = None) -> List[Section232DraftBatch]:
        ...

    def get_active_ruleset_version(self) -> str | None:
        ...

    def count_pending_batches(self) -> int:
        ...

    def get_last_published_at(self) -> str | None:
        ...

    def get_published_ruleset(self, version: str) -> Optional[Section232PublishedRuleset]:
        ...

    def get_published_ruleset_info(self, version: str) -> Optional[Section232PublishedRulesetInfo]:
        ...

    def list_published_ruleset_infos(self) -> List[Section232PublishedRulesetInfo]:
        ...

    def create_delete_override(self, *, hts_code: str, deleted_by: str) -> Section232CodeDeleteOverride:
        ...

    def get_delete_override(self, hts_code: str) -> Optional[Section232CodeDeleteOverride]:
        ...

    def list_delete_overrides(self) -> List[Section232CodeDeleteOverride]:
        ...

    def list_published_rule_page(
        self,
        *,
        version: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        ...

    def list_active_eligible_codes(self, *, on_date: object) -> List[str]:
        ...

    def clear_all(self) -> Dict[str, int]:
        ...

    def get_draft_batch(self, batch_id: str) -> Section232DraftBatch:
        ...


class HTSCatalogSourceStoreProtocol(Protocol):
    def list_sources(self) -> List[PersistedHTSCatalogSourceFile]:
        ...

    def get_source_texts(self) -> Dict[str, str]:
        ...

    def upsert_sources(
        self,
        uploads: Sequence[tuple[str, str]],
    ) -> tuple[List[PersistedHTSCatalogSourceFile], int]:
        ...

    def delete_source(self, filename: str) -> Optional[PersistedHTSCatalogSourceFile]:
        ...

    def get_refresh_state(self) -> HTSCatalogRefreshState:
        ...

    def set_refresh_state(
        self,
        *,
        status: str,
        last_refresh_at: Optional[str],
        last_refresh_error: Optional[str],
        catalog_row_count: Optional[int] = None,
        code_map_row_count: Optional[int] = None,
    ) -> HTSCatalogRefreshState:
        ...
