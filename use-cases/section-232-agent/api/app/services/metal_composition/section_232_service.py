"""Internal Section 232 and catalog administration manager."""

from __future__ import annotations

import logging
import uuid
from contextlib import nullcontext
from dataclasses import replace as dataclass_replace
from datetime import date
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from app.models.metal_composition import (
    HTSCatalogSourceDeleteResponse,
    HTSCatalogSourceFileSummary,
    HTSCatalogSourceListResponse,
    HTSCatalogSourceUploadResponse,
    Section232CancelDraftBatchResponse,
    Section232ClassificationResponse,
    Section232DirectClassificationRequest,
    Section232DraftBatchProcessResponse,
    Section232DraftRuleBulkReviewResponse,
    Section232DraftRuleDeleteResponse,
    Section232DraftRuleListResponse,
    Section232DraftRuleReviewResponse,
    Section232EligibleCodeDetail,
    Section232EligibleCodeDetailListResponse,
    Section232EligibleCodeListResponse,
    Section232PublishDraftBatchResponse,
    Section232PublishedRuleDeleteResponse,
    Section232ResetResponse,
    Section232ReviewHistoryItem,
    Section232ReviewWorkspaceResponse,
    Section232RulesetSummaryResponse,
    Section232SourceListResponse,
    Section232SourceSummary,
    Section232SourceUpdateMode,
    Section232SourceUploadResponse,
    Section232Assessment,
)

from .hts_catalog import CHAPTER_TITLES, canonicalize_hts_code
from .hts_catalog_sources import (
    CHAPTER_FILENAME_RE,
    decode_catalog_upload,
    normalize_hts_catalog_filename,
    validate_catalog_source_text,
)
from .section_232_rules_engine import (
    apply_section_232_weight_override,
    build_section_232_ruleset_assessment,
    build_section_232_ruleset_reasoner_output,
    build_skipped_weight_override,
    evaluate_section_232_ruleset,
    legal_coverage_established,
)
from .section_232_sources import extract_section_232_pdf
from .section_232_rulesets import project_current_rules_by_hts_code
from .timing import utc_now_iso
from .workflow.token_usage import TokenUsageRecorder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .service import MetalCompositionService


def _normalized_section_232_code(value: Any) -> str:
    return canonicalize_hts_code(getattr(value, "hts_code", value))


def _hts_query_digits(value: str | None) -> str:
    normalized = canonicalize_hts_code(value or "")
    return normalized.replace(".", "")


def _filter_section_232_candidates_by_hts_query(
    candidates: Sequence[Any],
    *,
    hts_query: str | None,
) -> List[Any]:
    query_digits = _hts_query_digits(hts_query)
    if not query_digits:
        return list(candidates)
    return [
        candidate
        for candidate in candidates
        if canonicalize_hts_code(getattr(candidate, "hts_code", "")).replace(".", "").startswith(query_digits)
    ]


def _page_section_232_candidates(candidates: Sequence[Any], *, limit: int, offset: int) -> List[Any]:
    return list(candidates)[offset : offset + limit]


class MetalCompositionSection232Service:
    """Own Section 232 review, publication, source, and catalog admin flows."""

    def __init__(self, service: MetalCompositionService) -> None:
        self.service = service

    def get_section_232_ruleset_summary(self) -> Section232RulesetSummaryResponse:
        service = self.service
        active_version = service.section_232_ruleset_store.get_active_ruleset_version()
        eligible_codes = service._list_active_section_232_eligible_codes()
        pending_batches = [
            service._build_section_232_draft_batch_summary(batch)
            for batch in service.section_232_ruleset_store.list_draft_batches(status="pending_review")
        ]
        return Section232RulesetSummaryResponse(
            active_ruleset_version=active_version,
            eligible_hts_code_count=len(eligible_codes),
            pending_draft_batch_count=len(pending_batches),
            pending_draft_batches=pending_batches,
            last_published_at=service.section_232_ruleset_store.get_last_published_at(),
        )

    def get_section_232_review_workspace(
        self,
        *,
        batch_id: str | None = None,
        version: str | None = None,
        hts_query: str | None = None,
        limit: int,
        offset: int,
    ) -> Section232ReviewWorkspaceResponse:
        service = self.service
        if bool((batch_id or "").strip()) == bool((version or "").strip()):
            raise ValueError("Provide exactly one of batch_id or version.")

        if batch_id:
            batch = next(
                (
                    item
                    for item in service.section_232_ruleset_store.list_draft_batches(status="pending_review")
                    if item.batch_id == batch_id
                ),
                None,
            )
            if batch is None:
                raise KeyError(f"draft batch {batch_id} not found")
            all_candidates = service.section_232_ruleset_store.list_draft_candidates(batch_id=batch_id)
            filtered_candidates = _filter_section_232_candidates_by_hts_query(
                service._sort_section_232_review_candidates(all_candidates),
                hts_query=hts_query,
            )
            page_candidates = _page_section_232_candidates(filtered_candidates, limit=limit, offset=offset)
            page_source_ids = [
                source_id
                for candidate in page_candidates
                for source_id in list(candidate.source_document_ids or [])
            ]
            source_filename_by_id = service._build_section_232_source_filename_map([batch.batch_id])
            source_by_id = service._build_section_232_source_map(source_ids=page_source_ids)
            batch_created_at_by_id = {batch.batch_id: batch.created_at}
            rows = service._build_section_232_review_rows_from_candidates(
                page_candidates,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
                batch_created_at_by_id=batch_created_at_by_id,
            )
            return Section232ReviewWorkspaceResponse(
                mode="draft",
                batch=service._build_section_232_draft_batch_summary(batch),
                source_filenames=list(batch.source_filenames),
                total=len(filtered_candidates),
                limit=limit,
                offset=offset,
                rows=rows,
            )

        published = service.section_232_ruleset_store.get_published_ruleset_info(version or "")
        if published is None:
            raise KeyError(f"published ruleset {version} not found")
        snapshot = service.section_232_ruleset_store.get_published_ruleset(version or "")
        snapshot_candidates = list(snapshot.accepted_rules if snapshot is not None else [])
        current_candidates = service._sort_section_232_review_candidates(
            project_current_rules_by_hts_code(snapshot_candidates)
        )
        filtered_candidates = _filter_section_232_candidates_by_hts_query(current_candidates, hts_query=hts_query)
        page_candidates = _page_section_232_candidates(filtered_candidates, limit=limit, offset=offset)
        history_by_code = self._build_published_rule_history_by_code(
            version=version or "",
            active_candidates=current_candidates,
        )
        source_filename_by_id = service._build_section_232_source_filename_map(
            [
                candidate.batch_id
                for candidate in [
                    *page_candidates,
                    *[
                        history_candidate
                        for candidates_for_code in history_by_code.values()
                        for history_candidate, _history_info in candidates_for_code
                    ],
                ]
            ]
        )
        source_by_id = service._build_section_232_source_map(
            source_ids=[
                source_id
                for candidate in [
                    *page_candidates,
                    *[
                        history_candidate
                        for candidates_for_code in history_by_code.values()
                        for history_candidate, _history_info in candidates_for_code
                    ],
                ]
                for source_id in candidate.source_document_ids
            ]
        )
        batch_created_at_by_id = service._build_section_232_batch_created_at_map()
        rows = service._build_section_232_review_rows_from_candidates(
            page_candidates,
            source_filename_by_id=source_filename_by_id,
            source_by_id=source_by_id,
            batch_created_at_by_id=batch_created_at_by_id,
        )
        rows = [
            row.model_copy(
                update={
                    "history": self._build_review_history_items(
                        history_by_code.get(canonicalize_hts_code(row.legal_hts_code), []),
                        source_filename_by_id=source_filename_by_id,
                        source_by_id=source_by_id,
                        batch_created_at_by_id=batch_created_at_by_id,
                    )
                }
            )
            for row in rows
        ]
        candidate_batch_ids = {candidate.batch_id for candidate in page_candidates}
        return Section232ReviewWorkspaceResponse(
            mode="published",
            version=published.version,
            source_filenames=list(
                dict.fromkeys(
                    filename
                    for batch in service.section_232_ruleset_store.list_draft_batches()
                    if batch.batch_id in candidate_batch_ids
                    for filename in batch.source_filenames
                )
            ),
            total=len(filtered_candidates),
            limit=limit,
            offset=offset,
            rows=rows,
        )

    def _build_published_rule_history_by_code(
        self,
        *,
        version: str,
        active_candidates: Sequence[Any],
    ) -> Dict[str, List[tuple[Any, Any]]]:
        service = self.service
        active_candidate_ids_by_code: Dict[str, set[str]] = {}
        for candidate in active_candidates:
            normalized_code = canonicalize_hts_code(getattr(candidate, "hts_code", ""))
            if not normalized_code:
                continue
            active_candidate_ids_by_code.setdefault(normalized_code, set()).add(str(candidate.candidate_id))

        history_by_code: Dict[str, List[tuple[Any, Any]]] = {}
        seen_candidate_ids: set[str] = set()
        for info in service.section_232_ruleset_store.list_published_ruleset_infos():
            snapshot = service.section_232_ruleset_store.get_published_ruleset(info.version)
            for candidate in list(snapshot.accepted_rules if snapshot is not None else []):
                normalized_code = canonicalize_hts_code(getattr(candidate, "hts_code", ""))
                candidate_id = str(getattr(candidate, "candidate_id", "") or "")
                if (
                    not normalized_code
                    or not candidate_id
                    or candidate_id in seen_candidate_ids
                    or candidate_id in active_candidate_ids_by_code.get(normalized_code, set())
                ):
                    continue
                seen_candidate_ids.add(candidate_id)
                history_by_code.setdefault(normalized_code, []).append((candidate, info))
            if info.version == version:
                break

        for entries in history_by_code.values():
            entries.sort(
                key=lambda item: (
                    str(getattr(item[0], "processed_at", "") or ""),
                    str(getattr(item[1], "published_at", "") or ""),
                    str(getattr(item[0], "candidate_id", "") or ""),
                ),
                reverse=True,
            )
        return history_by_code

    def _build_review_history_items(
        self,
        entries: Sequence[tuple[Any, Any]],
        *,
        source_filename_by_id: Dict[str, str],
        source_by_id: Dict[str, Any],
        batch_created_at_by_id: Dict[str, str],
    ) -> List[Section232ReviewHistoryItem]:
        service = self.service
        history_items: List[Section232ReviewHistoryItem] = []
        for candidate, info in entries:
            source_document_ids = list(getattr(candidate, "source_document_ids", []) or [])
            source_documents = service._build_section_232_source_documents(
                source_document_ids=source_document_ids,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
            )
            uploaded_values = [
                str(getattr(source_by_id.get(source_id), "uploaded_at", "") or "").strip()
                for source_id in source_document_ids
                if source_id in source_by_id
            ]
            source_uploaded_at = max((value for value in uploaded_values if value), default=None)
            processed_at = (
                str(getattr(candidate, "processed_at", "") or "").strip()
                or batch_created_at_by_id.get(str(getattr(candidate, "batch_id", "") or ""), None)
            )
            hts_code = str(getattr(candidate, "hts_code", "") or "")
            history_items.append(
                Section232ReviewHistoryItem(
                    version=str(getattr(info, "version", "") or ""),
                    published_at=str(getattr(info, "published_at", "") or "") or None,
                    published_by=str(getattr(info, "published_by", "") or "") or None,
                    candidate_id=str(getattr(candidate, "candidate_id", "") or ""),
                    legal_hts_code=hts_code,
                    hts_code=hts_code,
                    rule_type=getattr(candidate, "rule_type", "include"),
                    coverage_effect=getattr(candidate, "coverage_effect", "include"),
                    effective_from=getattr(candidate, "effective_from", None),
                    effective_to=getattr(candidate, "effective_to", None),
                    metal_scope=str(getattr(candidate, "metal_scope", "") or "unspecified"),
                    source_document_ids=source_document_ids,
                    source_filenames=[
                        source_filename_by_id[source_id]
                        for source_id in source_document_ids
                        if source_id in source_filename_by_id
                    ],
                    source_documents=source_documents,
                    source_uploaded_at=source_uploaded_at,
                    processed_at=processed_at,
                )
            )
        return history_items

    def process_section_232_draft_batch(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
        review_page_size: int,
        include_token_usage: bool = False,
    ) -> Section232DraftBatchProcessResponse:
        from .service import interpret_section_232_batch

        service = self.service
        llm = service._get_workflow_llm()
        token_usage_recorder = TokenUsageRecorder() if include_token_usage else None
        provisional_batch_id = str(uuid.uuid4())
        ruleset_connection = service._ruleset_store_connection()
        transaction = ruleset_connection.transaction() if ruleset_connection is not None else nullcontext()
        stored_sources = [
            service.section_232_source_store.save_source(
                filename=filename,
                size_bytes=len(content),
                extracted=extract_section_232_pdf(content),
            )
            for filename, content in uploads
        ]
        try:
            interpret_kwargs: Dict[str, Any] = {
                "batch_id": provisional_batch_id,
                "extracted_sources": stored_sources,
                "llm": llm,
            }
            if token_usage_recorder is not None:
                interpret_kwargs["usage_recorder"] = token_usage_recorder
            candidates = interpret_section_232_batch(**interpret_kwargs)
            with transaction:
                batch = service.section_232_ruleset_store.create_draft_batch(
                    source_ids=[source.source_id for source in stored_sources],
                    source_filenames=[source.filename for source in stored_sources],
                )
                candidates = [
                    dataclass_replace(
                        candidate,
                        catalog_match_found=(
                            service._resolve_section_232_catalog_match(candidate.hts_code)["catalog_match_type"]
                            != "missing"
                        ),
                        processed_at=batch.created_at,
                    )
                    for candidate in candidates
                ]
                service.section_232_ruleset_store.replace_batch_candidates(batch.batch_id, candidates)
        except Exception:
            delete_sources = getattr(service.section_232_source_store, "delete_sources", None)
            if callable(delete_sources):
                try:
                    delete_sources([source.source_id for source in stored_sources])
                except Exception:
                    logger.exception("Failed to clean up Section 232 sources after draft batch processing error")
            raise
        preview_page = service.section_232_ruleset_store.list_draft_candidate_page(
            batch_id=batch.batch_id,
            limit=review_page_size,
            offset=0,
        )
        source_filename_by_id = service._build_section_232_source_filename_map([batch.batch_id])
        source_by_id = {source.source_id: source for source in stored_sources}
        return Section232DraftBatchProcessResponse(
            batch=service._build_section_232_draft_batch_summary(batch),
            ruleset_summary=self.get_section_232_ruleset_summary(),
            token_usage=(
                token_usage_recorder.build_summary()
                if token_usage_recorder is not None
                else None
            ),
            total=preview_page.total,
            limit=preview_page.limit,
            offset=preview_page.offset,
            items=service._build_section_232_draft_rule_items_from_candidates(
                preview_page.candidates,
                source_filename_by_id=source_filename_by_id,
                source_by_id=source_by_id,
                batch_created_at_by_id={batch.batch_id: batch.created_at},
            ),
        )

    def _section_232_source_ids_referenced_by_batches(self) -> set[str]:
        service = self.service
        return {
            str(source_id)
            for batch in service.section_232_ruleset_store.list_draft_batches()
            for source_id in list(getattr(batch, "source_ids", []) or [])
            if str(source_id or "").strip()
        }

    def cancel_section_232_draft_batch(self, batch_id: str) -> Section232CancelDraftBatchResponse:
        service = self.service
        normalized_batch_id = str(batch_id or "").strip()
        shared_connection = service._shared_section_232_connection()
        transaction = shared_connection.transaction() if shared_connection is not None else nullcontext()

        with transaction:
            deleted_batch = service.section_232_ruleset_store.delete_pending_draft_batch(normalized_batch_id)
            referenced_source_ids = self._section_232_source_ids_referenced_by_batches()
            deleted_source_filename_by_id = {
                str(source_id): str(source_filename or "")
                for source_id, source_filename in zip(
                    deleted_batch.source_ids,
                    deleted_batch.source_filenames,
                )
                if str(source_id or "").strip()
            }
            existing_source_filename_by_id = {
                source.source_id: source.filename
                for source in service.section_232_source_store.list_sources()
            }
            source_ids_to_delete = [
                str(source_id)
                for source_id in deleted_batch.source_ids
                if str(source_id or "").strip()
                and str(source_id) not in referenced_source_ids
                and str(source_id) in existing_source_filename_by_id
            ]
            deleted_source_filenames = [
                existing_source_filename_by_id.get(source_id)
                or deleted_source_filename_by_id.get(source_id)
                or source_id
                for source_id in source_ids_to_delete
            ]
            deleted_source_count = service.section_232_source_store.delete_sources(source_ids_to_delete)

        return Section232CancelDraftBatchResponse(
            batch_id=deleted_batch.batch_id,
            deleted_source_count=deleted_source_count,
            deleted_source_filenames=deleted_source_filenames,
            deleted_draft_rule_count=deleted_batch.deleted_rule_count,
            ruleset_summary=self.get_section_232_ruleset_summary(),
        )

    def list_section_232_draft_rules(
        self,
        *,
        batch_id: str,
        limit: int,
        offset: int,
    ) -> Section232DraftRuleListResponse:
        service = self.service
        service._get_pending_section_232_draft_batch(batch_id)
        page = service.section_232_ruleset_store.list_draft_candidate_page(
            batch_id=batch_id,
            limit=limit,
            offset=offset,
        )
        source_filename_by_id = service._build_section_232_source_filename_map([batch_id])
        items = service._build_section_232_draft_rule_items_from_candidates(
            page.candidates,
            source_filename_by_id=source_filename_by_id,
            batch_created_at_by_id={batch_id: service.section_232_ruleset_store.get_draft_batch(batch_id).created_at},
        )
        return Section232DraftRuleListResponse(
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            items=items,
        )

    def review_section_232_draft_rule(
        self,
        *,
        batch_id: str,
        candidate_id: str,
        decision: str,
    ) -> Section232DraftRuleReviewResponse:
        service = self.service
        service._get_pending_section_232_draft_batch(batch_id)
        updated = service.section_232_ruleset_store.review_candidate(
            batch_id,
            candidate_id,
            decision=decision,
        )
        return Section232DraftRuleReviewResponse(
            item=service._build_section_232_draft_rule_item(
                updated,
                source_filename_by_id=service._build_section_232_source_filename_map([batch_id]),
            )
        )

    def review_section_232_draft_rules(
        self,
        *,
        batch_id: str,
        candidate_ids: Sequence[str],
        selection_mode: str,
        excluded_candidate_ids: Sequence[str],
        decision: str,
    ) -> Section232DraftRuleBulkReviewResponse:
        service = self.service
        service._get_pending_section_232_draft_batch(batch_id)
        if selection_mode == "all":
            updated_count = service.section_232_ruleset_store.review_all_candidates(
                batch_id,
                decision=decision,
                excluded_candidate_ids=excluded_candidate_ids,
            )
            return Section232DraftRuleBulkReviewResponse(
                updated_count=updated_count,
                candidate_ids=[],
                decision=decision,
            )

        updated_candidates = service.section_232_ruleset_store.review_candidates(
            batch_id,
            candidate_ids,
            decision=decision,
        )
        return Section232DraftRuleBulkReviewResponse(
            updated_count=len(updated_candidates),
            candidate_ids=[candidate.candidate_id for candidate in updated_candidates],
            decision=decision,
        )

    def delete_section_232_draft_hts_code(
        self,
        *,
        batch_id: str,
        hts_code: str,
    ) -> Section232DraftRuleDeleteResponse:
        service = self.service
        service._get_pending_section_232_draft_batch(batch_id)
        deleted_count = service.section_232_ruleset_store.delete_draft_candidates_by_hts_code(
            batch_id,
            hts_code=hts_code,
        )
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if deleted_count <= 0 or not normalized_hts_code:
            raise KeyError(f"HTS code {hts_code} not found in draft batch {batch_id}")
        return Section232DraftRuleDeleteResponse(
            batch_id=batch_id,
            deleted_hts_code=normalized_hts_code,
            deleted_count=deleted_count,
        )

    def publish_section_232_draft_batch(
        self,
        *,
        batch_id: str,
        published_by: str,
    ) -> Section232PublishDraftBatchResponse:
        service = self.service
        shared_connection = service._shared_section_232_connection()
        transaction = shared_connection.transaction() if shared_connection is not None else nullcontext()

        with transaction:
            draft_candidates = service.section_232_ruleset_store.list_draft_candidates(batch_id=batch_id)
            pending_candidate_ids = [
                candidate.candidate_id
                for candidate in draft_candidates
                if candidate.review_decision == "pending"
            ]
            if pending_candidate_ids:
                service.section_232_ruleset_store.review_candidates(
                    batch_id,
                    pending_candidate_ids,
                    decision="rejected",
                )
                draft_candidates = service.section_232_ruleset_store.list_draft_candidates(batch_id=batch_id)

            accepted_candidates = [
                dataclass_replace(candidate, review_decision="accepted")
                for candidate in draft_candidates
                if candidate.review_decision == "accepted"
            ]
            if not accepted_candidates:
                raise ValueError(f"draft batch {batch_id} has no accepted candidates")
            try:
                merged_candidates = service._build_section_232_publish_snapshot(
                    accepted_candidates=accepted_candidates,
                )
            except ValueError as exc:
                message = str(exc)
                if "draft batch has duplicate normalized publication slots" in message:
                    raise ValueError(
                        f"draft batch {batch_id} has duplicate normalized publication slots for HTS codes "
                        f"{message.split(' for HTS codes ', 1)[1]}"
                    ) from exc
                if "publish would create overlapping active rules" in message:
                    raise ValueError(
                        f"draft batch {batch_id} would create overlapping active rules "
                        f"for normalized code/scope {message.split(' for normalized code/scope ', 1)[1]}"
                    ) from exc
                raise
            published = service.section_232_ruleset_store.publish_batch(
                batch_id,
                published_by=published_by,
                accepted_rules_snapshot=merged_candidates,
            )
        return Section232PublishDraftBatchResponse(
            published_version=published.version,
            accepted_rule_count=len(published.accepted_rules),
            ruleset_summary=self.get_section_232_ruleset_summary(),
        )

    def delete_section_232_published_hts_code(
        self,
        *,
        hts_code: str,
        published_by: str,
    ) -> Section232PublishedRuleDeleteResponse:
        service = self.service
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            raise ValueError(f"invalid hts_code: {hts_code!r}")
        active_rules = [
            dataclass_replace(candidate, review_decision="accepted")
            for candidate in service.section_232_ruleset_store.list_active_rules()
        ]
        if not active_rules:
            raise KeyError("No active published Section 232 ruleset is available")
        remaining_rules = [
            candidate
            for candidate in active_rules
            if _normalized_section_232_code(candidate) != normalized_hts_code
        ]
        removed_rule_count = len(active_rules) - len(remaining_rules)
        if removed_rule_count <= 0:
            raise KeyError(f"HTS code {normalized_hts_code} is not active in the published ruleset")

        ruleset_connection = service._ruleset_store_connection()
        transaction = ruleset_connection.transaction() if ruleset_connection is not None else nullcontext()
        with transaction:
            service.section_232_ruleset_store.create_delete_override(
                hts_code=normalized_hts_code,
                deleted_by=published_by,
            )
            published = service.section_232_ruleset_store.publish_snapshot(
                published_by=published_by,
                accepted_rules_snapshot=remaining_rules,
            )
        return Section232PublishedRuleDeleteResponse(
            deleted_hts_code=normalized_hts_code,
            removed_rule_count=removed_rule_count,
            published_version=published.version,
            ruleset_summary=self.get_section_232_ruleset_summary(),
        )

    def classify_section_232(
        self,
        payload: Section232DirectClassificationRequest,
    ) -> Section232ClassificationResponse:
        service = self.service
        selected_code = canonicalize_hts_code(payload.hts_code)
        top_level_grams = payload.top_level_grams.model_dump(mode="json")
        estimated_total_metal_grams = sum(float(value or 0.0) for value in top_level_grams.values())
        final_composition = {
            "is_metal_item": estimated_total_metal_grams > 0.0,
            "total_weight_grams": payload.total_weight_grams,
            "estimated_total_metal_grams": estimated_total_metal_grams,
            "top_level_grams": top_level_grams,
            "steel_subtype_grams": {},
            "confidence": 1.0,
            "reasoning": "Section 232 classification evaluated from direct composition inputs.",
        }
        legal_result = evaluate_section_232_ruleset(
            selected_code=selected_code,
            on_date=date.today(),
            ruleset_store=service.section_232_ruleset_store,
            top_level_grams=top_level_grams,
        )
        section_payload = build_section_232_ruleset_assessment(
            selected_code=selected_code,
            candidate_codes=[selected_code, *payload.supporting_hts_candidates],
            legal_result=legal_result,
        )
        if legal_result.get("reason") == "matched_rule" and legal_result.get("matched_hts_code"):
            matched_rule_scope = str(legal_result.get("matched_rule_scope") or "")
            if matched_rule_scope == "exact":
                scope_summary = "as an exact-match scope."
            elif matched_rule_scope == "family":
                scope_summary = "as a prefix-family scope."
            else:
                scope_summary = "under the published ruleset."
            section_payload["evidence"] = [
                *list(section_payload.get("evidence") or []),
                {
                    "source": "published_ruleset_scope",
                    "summary": (
                        f"Published legal HTS code {legal_result['matched_hts_code']} applies to "
                        f"{selected_code or 'unknown'} {scope_summary}"
                    ),
                    "matched_rule_code": legal_result["matched_hts_code"],
                    "matched_rule_scope": matched_rule_scope or None,
                },
            ]
        if legal_coverage_established(legal_result):
            section_payload, metal_weight_override = apply_section_232_weight_override(
                final_composition=final_composition,
                diagram_output={"metal_share_certainty": payload.metal_share_certainty},
                section_payload=section_payload,
            )
        else:
            metal_weight_override = build_skipped_weight_override(
                final_composition=final_composition,
                diagram_output={"metal_share_certainty": payload.metal_share_certainty},
                section_payload=section_payload,
            )
        section_232_reasoner_output = build_section_232_ruleset_reasoner_output(legal_result)
        section_232_reasoner_output["metal_weight_override"] = metal_weight_override
        return Section232ClassificationResponse(
            section_232_assessment=Section232Assessment.model_validate(section_payload),
            section_232_reasoner_output=section_232_reasoner_output,
        )

    def list_hts_catalog_sources(self) -> HTSCatalogSourceListResponse:
        service = self.service
        items = [
            HTSCatalogSourceFileSummary(
                filename=item.filename,
                source_kind=item.source_kind,
                chapter_number=item.chapter_number,
                size_bytes=item.size_bytes,
                uploaded_at=item.uploaded_at,
            )
            for item in service.hts_catalog_source_store.list_sources()
        ]
        return HTSCatalogSourceListResponse(
            total=len(items),
            summary=service._build_hts_catalog_summary(),
            items=items,
        )

    def list_section_232_sources(self) -> Section232SourceListResponse:
        service = self.service
        items = [item.to_summary() for item in service.section_232_source_store.list_sources()]
        eligible_hts_codes = service._list_active_section_232_eligible_codes()
        return Section232SourceListResponse(
            total=len(items),
            eligible_hts_code_count=len(eligible_hts_codes),
            items=items,
        )

    def list_section_232_eligible_hts_codes(self) -> Section232EligibleCodeListResponse:
        codes = self.service._list_active_section_232_eligible_codes()
        return Section232EligibleCodeListResponse(total=len(codes), codes=codes)

    def list_section_232_eligible_hts_code_details(
        self,
        *,
        query: Optional[str],
        limit: int,
        offset: int,
    ) -> Section232EligibleCodeDetailListResponse:
        service = self.service
        codes = service._list_active_section_232_eligible_codes()
        query_text = (query or "").strip().lower()
        query_digits = "".join(char for char in query_text if char.isdigit())

        try:
            resolver = service._get_hts_catalog_resolver_for_admin()
            catalog_by_code = resolver.catalog_by_code
        except Exception:
            catalog_by_code = {}

        detailed_items: List[Section232EligibleCodeDetail] = []
        for code in codes:
            row = catalog_by_code.get(code) or {}
            chapter_number = int(row.get("chapter_number")) if row.get("chapter_number") not in (None, "") else None
            description = (
                str(row.get("path_description") or "").strip()
                or str(row.get("description") or "").strip()
                or None
            )
            searchable_text = " ".join(
                part
                for part in (
                    code,
                    description or "",
                    str(row.get("searchable_text") or ""),
                    CHAPTER_TITLES.get(chapter_number or -1, ""),
                )
                if part
            ).lower()
            code_digits = "".join(char for char in code if char.isdigit())
            if query_text and query_text not in searchable_text and (not query_digits or query_digits not in code_digits):
                continue

            detailed_items.append(
                Section232EligibleCodeDetail(
                    code=code,
                    description=description,
                    chapter_number=chapter_number,
                    chapter_title=CHAPTER_TITLES.get(chapter_number) if chapter_number is not None else None,
                )
            )

        paged_items = detailed_items[offset : offset + limit]
        return Section232EligibleCodeDetailListResponse(
            total=len(detailed_items),
            limit=limit,
            offset=offset,
            query=query or None,
            items=paged_items,
        )

    def upload_section_232_sources(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
        update_mode: Section232SourceUpdateMode,
    ) -> Section232SourceUploadResponse:
        service = self.service
        prepared_uploads: List[tuple[str, bytes, Any]] = []
        for filename, content in uploads:
            prepared_uploads.append((filename, content, extract_section_232_pdf(content)))

        if update_mode == "replace":
            service.section_232_source_store.clear_sources()

        uploaded_sources: List[Section232SourceSummary] = []
        uploaded_hts_codes: List[str] = []
        for filename, content, extracted in prepared_uploads:
            source = service.section_232_source_store.save_source(
                filename=filename,
                size_bytes=len(content),
                extracted=extracted,
            )
            uploaded_sources.append(source.to_summary())
            uploaded_hts_codes.extend(extracted.hts_mentions)

        if update_mode == "replace":
            eligible_hts_codes = service.section_232_source_store.replace_eligible_hts_codes(uploaded_hts_codes)
        else:
            eligible_hts_codes = service.section_232_source_store.append_eligible_hts_codes(uploaded_hts_codes)

        return Section232SourceUploadResponse(
            update_mode=update_mode,
            source_count=len(uploaded_sources),
            uploaded_hts_code_count=len({code for code in uploaded_hts_codes if code}),
            total_eligible_hts_code_count=len(eligible_hts_codes),
            items=uploaded_sources,
        )

    def upload_section_232_source(
        self,
        *,
        filename: str,
        content: bytes,
        update_mode: Section232SourceUpdateMode,
    ) -> Section232SourceUploadResponse:
        return self.upload_section_232_sources(
            uploads=[(filename, content)],
            update_mode=update_mode,
        )

    def upload_hts_catalog_sources(
        self,
        *,
        uploads: Sequence[tuple[str, bytes]],
    ) -> HTSCatalogSourceUploadResponse:
        service = self.service
        if not uploads:
            raise ValueError("At least one CSV file is required.")

        prepared_uploads: List[tuple[str, str]] = []
        seen_filenames = set()
        for filename, content in uploads:
            normalized_text = decode_catalog_upload(filename, content)
            validate_catalog_source_text(filename, normalized_text)
            normalized_filename = normalize_hts_catalog_filename(filename, content_text=normalized_text)
            if normalized_filename in seen_filenames:
                raise ValueError(f"{normalized_filename} was uploaded more than once in the same request.")
            seen_filenames.add(normalized_filename)
            prepared_uploads.append((normalized_filename, normalized_text))

        source_texts = service.hts_catalog_source_store.get_source_texts()
        candidate_source_texts = dict(source_texts)
        candidate_source_texts.update({filename: text for filename, text in prepared_uploads})

        now = utc_now_iso()
        try:
            catalog_frame, code_map_frame, refresh_result = service._refresh_hts_catalog_source_texts(
                source_texts=candidate_source_texts
            )
            persisted_items, overwritten_count = service.hts_catalog_source_store.upsert_sources(prepared_uploads)
        except Exception as exc:
            service.hts_catalog_source_store.set_refresh_state(
                status="failed",
                last_refresh_at=now,
                last_refresh_error=str(exc),
            )
            raise

        service.hts_catalog_source_store.set_refresh_state(
            status="completed",
            last_refresh_at=now,
            last_refresh_error=None,
            catalog_row_count=int(refresh_result.get("catalog_row_count") or len(catalog_frame)),
            code_map_row_count=int(refresh_result.get("code_map_row_count") or len(code_map_frame)),
        )
        service._set_hts_catalog_resolver_for_admin(
            catalog_frame=catalog_frame,
            code_map_frame=code_map_frame,
        )
        return HTSCatalogSourceUploadResponse(
            uploaded_file_count=len(persisted_items),
            overwritten_file_count=overwritten_count,
            items=[
                HTSCatalogSourceFileSummary(
                    filename=item.filename,
                    source_kind=item.source_kind,
                    chapter_number=item.chapter_number,
                    size_bytes=item.size_bytes,
                    uploaded_at=item.uploaded_at,
                )
                for item in persisted_items
            ],
            summary=service._build_hts_catalog_summary(),
        )

    def delete_hts_catalog_source(
        self,
        *,
        filename: str,
    ) -> HTSCatalogSourceDeleteResponse:
        service = self.service
        normalized_filename = normalize_hts_catalog_filename(filename)
        source_texts = service.hts_catalog_source_store.get_source_texts()
        if normalized_filename not in source_texts:
            raise KeyError(f"managed HTS catalog source {normalized_filename} not found")

        candidate_source_texts = {
            current_filename: content_text
            for current_filename, content_text in source_texts.items()
            if current_filename != normalized_filename
        }
        remaining_chapter_count = sum(
            1 for current_filename in candidate_source_texts if CHAPTER_FILENAME_RE.fullmatch(current_filename)
        )
        if CHAPTER_FILENAME_RE.fullmatch(normalized_filename) and remaining_chapter_count == 0:
            raise ValueError("At least one HTS chapter CSV must remain in the managed catalog.")

        now = utc_now_iso()
        try:
            catalog_frame, code_map_frame, refresh_result = service._refresh_hts_catalog_source_texts(
                source_texts=candidate_source_texts
            )
            deleted_item = service.hts_catalog_source_store.delete_source(normalized_filename)
        except Exception as exc:
            service.hts_catalog_source_store.set_refresh_state(
                status="failed",
                last_refresh_at=now,
                last_refresh_error=str(exc),
            )
            raise

        if deleted_item is None:
            raise KeyError(f"managed HTS catalog source {normalized_filename} not found")

        service.hts_catalog_source_store.set_refresh_state(
            status="completed",
            last_refresh_at=now,
            last_refresh_error=None,
            catalog_row_count=int(refresh_result.get("catalog_row_count") or len(catalog_frame)),
            code_map_row_count=int(refresh_result.get("code_map_row_count") or len(code_map_frame)),
        )
        service._set_hts_catalog_resolver_for_admin(
            catalog_frame=catalog_frame,
            code_map_frame=code_map_frame,
        )
        return HTSCatalogSourceDeleteResponse(
            deleted_filename=deleted_item.filename,
            summary=service._build_hts_catalog_summary(),
        )

    def reset_section_232_data(self) -> Section232ResetResponse:
        service = self.service
        cleared_source_count = len(service.section_232_source_store.list_sources())
        clear_eligible_codes = getattr(service.section_232_source_store, "clear_eligible_hts_codes", None)
        shared_connection = service._shared_section_232_connection()
        if shared_connection is None:
            service.section_232_source_store.clear_sources()
            if callable(clear_eligible_codes):
                clear_eligible_codes()
            reset_counts = dict(service.section_232_ruleset_store.clear_all())
        else:
            with shared_connection.transaction():
                service.section_232_source_store.clear_sources()
                if callable(clear_eligible_codes):
                    clear_eligible_codes()
                reset_counts = dict(service.section_232_ruleset_store.clear_all())
        return Section232ResetResponse(
            cleared_source_count=cleared_source_count,
            cleared_draft_batch_count=int(reset_counts.get("cleared_draft_batch_count") or 0),
            cleared_draft_rule_count=int(reset_counts.get("cleared_draft_rule_count") or 0),
            cleared_delete_override_count=int(reset_counts.get("cleared_delete_override_count") or 0),
            cleared_published_ruleset_count=int(reset_counts.get("cleared_published_ruleset_count") or 0),
            cleared_published_rule_count=int(reset_counts.get("cleared_published_rule_count") or 0),
        )
