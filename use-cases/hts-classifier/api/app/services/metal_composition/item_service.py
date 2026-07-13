"""Internal item and detail manager for metal composition workflows."""

from __future__ import annotations

from difflib import SequenceMatcher
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from app.models.metal_composition import (
    MetalCompositionAppSettings,
    MetalCompositionFacetOption,
    MetalCompositionItemDetail,
    MetalCompositionItemListResponse,
    MetalCompositionItemSummary,
    MetalCompositionResponse,
)

from .serving_store import normalize_lookup_value

if TYPE_CHECKING:
    from .service import MetalCompositionService


def _optional_text(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _contains_filter(value: Optional[str], query: Optional[str]) -> bool:
    if not query:
        return True
    return query.lower() in (value or "").lower()


def _matches_facet(value: Optional[str], selected: Optional[str]) -> bool:
    if not selected:
        return True
    return (value or "").lower() == selected.lower()


def _matches_optional_boolean(value: bool, selected: Optional[bool]) -> bool:
    if selected is None:
        return True
    return value is selected


def _confidence_to_percentage(value: object) -> float:
    """Return a confidence value formatted as percent for UI presentation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numeric):
        return float("nan")
    if 0.0 <= numeric <= 1.0:
        return round(numeric * 100, 2)
    if 1.0 < numeric <= 100.0:
        return round(numeric, 2)
    return float("nan")


def _coerce_classification_payload(payload: MetalCompositionResponse | Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly copy of a classification payload for mutation."""
    if isinstance(payload, MetalCompositionResponse):
        return payload.model_dump(mode="json")
    return dict(payload)


def _apply_percentage_confidence(payload: MetalCompositionResponse | Dict[str, Any]) -> MetalCompositionResponse:
    """Convert all classification confidence values to percentages and revalidate."""
    classification = _coerce_classification_payload(payload)

    final_composition = classification.get("final_composition")
    if isinstance(final_composition, dict):
        if "confidence" in final_composition:
            converted = _confidence_to_percentage(final_composition.get("confidence"))
            if not math.isnan(converted):
                final_composition["confidence"] = converted

    hts_classification = classification.get("hts_classification")
    if isinstance(hts_classification, dict):
        if "confidence" in hts_classification:
            converted = _confidence_to_percentage(hts_classification.get("confidence"))
            if not math.isnan(converted):
                hts_classification["confidence"] = converted

        best_candidate = hts_classification.get("best_candidate")
        if isinstance(best_candidate, dict) and "confidence" in best_candidate:
            converted = _confidence_to_percentage(best_candidate.get("confidence"))
            if not math.isnan(converted):
                best_candidate["confidence"] = converted

        candidates = hts_classification.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and "confidence" in candidate:
                    converted = _confidence_to_percentage(candidate.get("confidence"))
                    if not math.isnan(converted):
                        candidate["confidence"] = converted

    section_232 = classification.get("section_232_assessment")
    if isinstance(section_232, dict) and "confidence" in section_232:
        converted = _confidence_to_percentage(section_232.get("confidence"))
        if not math.isnan(converted):
            section_232["confidence"] = converted

    return MetalCompositionResponse.model_validate(classification)


class MetalCompositionItemService:
    """Own GCC item listing and detail resolution."""

    def __init__(self, service: MetalCompositionService) -> None:
        self.service = service

    def get_app_settings(self) -> MetalCompositionAppSettings:
        return self.service.ui_state_store.get_app_settings().to_model()

    def update_app_settings(
        self,
        *,
        use_gcc_tracker_metal_composition: bool,
    ) -> MetalCompositionAppSettings:
        return self.service.ui_state_store.update_app_settings(
            use_gcc_tracker_metal_composition=use_gcc_tracker_metal_composition,
        ).to_model()

    def list_items(
        self,
        *,
        priority: Optional[str] = None,
        business_segment: Optional[str] = None,
        product_code: Optional[str] = None,
        pn_revised_standardized: Optional[str] = None,
        new_part_description: Optional[str] = None,
        part_description: Optional[str] = None,
        has_documents: Optional[bool] = None,
        is_classified: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> MetalCompositionItemListResponse:
        service = self.service
        base_items = service._list_base_items()
        filtered_items = [
            item
            for item in base_items
            if _matches_facet(item["priority"], priority)
            and _matches_facet(item["business_segment"], business_segment)
            and _contains_filter(item["product_code"], product_code)
            and _contains_filter(item["pn_revised_standardized"], pn_revised_standardized)
            and _contains_filter(item["new_part_description"], new_part_description)
            and _contains_filter(item["part_description"], part_description)
        ]

        assignment_map = (
            service.ui_state_store.get_document_assignment_map(
                [(item["item_id"], item["dataset_scope"]) for item in filtered_items]
            )
            if filtered_items
            else {}
        )
        classified_key_set = (
            service.ui_state_store.get_classification_snapshot_keys()
            if is_classified is not None
            else set()
        )
        has_documents_cache: Dict[tuple[str, str], bool] = {}

        if has_documents is not None or is_classified is not None:
            post_filter_items: List[Dict[str, Any]] = []
            for item in filtered_items:
                key = (item["item_id"], item["dataset_scope"])
                assigned_documents = service.document_store.references_from_stored(
                    assignment_map.get(key, [])
                )
                item_has_documents = (
                    service._item_has_documents(item, assigned_documents=assigned_documents)
                    if has_documents is not None
                    else False
                )
                if has_documents is not None:
                    has_documents_cache[key] = item_has_documents
                item_is_classified = key in classified_key_set if is_classified is not None else False
                if _matches_optional_boolean(item_has_documents, has_documents) and _matches_optional_boolean(
                    item_is_classified,
                    is_classified,
                ):
                    post_filter_items.append(item)
            filtered_items = post_filter_items

        filtered_items.sort(
            key=lambda item: (
                (item["product_code"] or "").lower(),
                (item["pn_revised_standardized"] or "").lower(),
                item["item_id"],
            )
        )

        facets = service._build_facets(filtered_items)
        paged_items = filtered_items[offset : offset + limit]

        keys = [(item["item_id"], item["dataset_scope"]) for item in paged_items]
        paged_assignment_map = (
            {key: assignment_map.get(key, []) for key in keys}
            if assignment_map
            else service.ui_state_store.get_document_assignment_map(keys)
        )
        snapshot_map = service.ui_state_store.get_classification_snapshot_map(keys)

        items: List[MetalCompositionItemSummary] = []
        items.extend(
            service._build_item_summaries(
                paged_items,
                assignment_map=paged_assignment_map,
                snapshot_map=snapshot_map,
                has_documents_cache=has_documents_cache,
            )
        )

        return MetalCompositionItemListResponse(
            total=len(filtered_items),
            limit=limit,
            offset=offset,
            items=items,
            facets=facets,
        )

    def search_items_by_product_code(self, product_code: str) -> List[ItemSearchMatch]:
        from .service import ItemSearchMatch

        service = self.service
        query = normalize_lookup_value(product_code)
        if not query:
            return []

        matched_items = [
            item
            for item in service._list_base_items()
            if normalize_lookup_value(item.get("product_code")) == query
        ]
        summaries = service._build_item_summaries(matched_items)
        return [
            ItemSearchMatch(item=summary, score=1.0, match_basis="product_code")
            for summary in summaries
        ]

    def search_items_by_description(self, query: str, *, limit: int = 5) -> List[ItemSearchMatch]:
        from .service import ItemSearchMatch

        service = self.service
        normalized_query = _optional_text(query)
        if not normalized_query:
            return []

        lowered_query = normalized_query.lower()
        ranked_matches: List[tuple[Dict[str, Any], float]] = []
        for item in service._list_base_items():
            new_description = (item.get("new_part_description") or "").strip()
            part_description = (item.get("part_description") or "").strip()
            searchable_parts = [part for part in (new_description, part_description) if part]
            if not searchable_parts:
                continue

            best_score = 0.0
            for candidate in searchable_parts:
                lowered_candidate = candidate.lower()
                similarity = SequenceMatcher(None, lowered_query, lowered_candidate).ratio()
                if lowered_query in lowered_candidate:
                    similarity += 0.35
                elif any(token and token in lowered_candidate for token in lowered_query.split()):
                    similarity += 0.15
                best_score = max(best_score, min(similarity, 1.0))

            if best_score >= 0.25:
                ranked_matches.append((item, best_score))

        ranked_matches.sort(
            key=lambda entry: (
                -entry[1],
                (entry[0].get("product_code") or "").lower(),
                (entry[0].get("pn_revised_standardized") or "").lower(),
                entry[0]["item_id"],
            )
        )
        top_items = [item for item, _score in ranked_matches[:limit]]
        top_summaries = service._build_item_summaries(top_items)
        summary_map = {summary.item_id: summary for summary in top_summaries}
        return [
            ItemSearchMatch(
                item=summary_map[item["item_id"]],
                score=score,
                match_basis="description",
            )
            for item, score in ranked_matches[:limit]
            if item["item_id"] in summary_map
        ]

    def get_item_detail(self, item_id: str) -> MetalCompositionItemDetail:
        service = self.service
        context = service._resolve_item_context(item_id)
        assigned_refs = service.ui_state_store.get_document_assignments(
            context.item_id,
            dataset_scope=context.dataset_scope,
        )
        assigned_documents = service.document_store.references_from_stored(assigned_refs)
        warnings: List[str] = []
        if assigned_refs and len(assigned_documents) != len(assigned_refs):
            warnings.append("Some assigned PDF files are no longer available.")

        snapshot = service.ui_state_store.get_classification_snapshot(
            context.item_id,
            dataset_scope=context.dataset_scope,
        )

        latest_classification = None
        if snapshot is not None:
            latest_classification = _apply_percentage_confidence(snapshot.payload)

        return MetalCompositionItemDetail(
            item_id=context.item_id,
            item_type=context.item_type,
            source_row_id=context.source_row_id,
            dataset_signature=service.dataset_signature if context.item_type == "gcc" else None,
            product_code=context.product_code,
            priority=context.priority,
            priority_detail=context.priority_detail,
            business_segment=context.business_segment,
            site=context.site,
            pn_revised_standardized=context.pn_revised_standardized,
            part_description=context.part_description,
            new_part_description=context.new_part_description,
            total_weight_gram=context.total_weight_gram,
            material_content_method=context.material_content_method,
            material_identified=context.material_identified,
            date_started=context.date_started,
            date_completed=context.date_completed,
            docs_status=service._detail_docs_status(
                context,
                assigned_documents=assigned_documents,
            ),
            assigned_documents=assigned_documents,
            latest_classification=latest_classification,
            last_classified_at=snapshot.last_classified_at if snapshot is not None else None,
            warnings=warnings,
        )

    def replace_item_documents(
        self,
        item_id: str,
        *,
        document_paths: Sequence[str],
    ) -> MetalCompositionItemDetail:
        service = self.service
        context = service._resolve_item_context(item_id)
        validated_paths = service.document_store.validate_paths(document_paths)
        service.ui_state_store.replace_document_assignments(
            item_id,
            dataset_scope=context.dataset_scope,
            document_refs=[
                service.document_store.to_stored_reference(reference)
                for reference in validated_paths
            ],
        )
        return self.get_item_detail(item_id)

    def upload_item_document(
        self,
        item_id: str,
        *,
        filename: str,
        content: bytes,
    ) -> MetalCompositionItemDetail:
        service = self.service
        context = service._resolve_item_context(item_id)
        reference = service.document_store.save_uploaded_pdf(
            dataset_scope=context.dataset_scope,
            item_id=context.item_id,
            filename=filename,
            content=content,
        )
        assigned_refs = service.ui_state_store.get_document_assignments(
            context.item_id,
            dataset_scope=context.dataset_scope,
        )
        merged_refs = [*assigned_refs]
        stored_reference = service.document_store.to_stored_reference(reference)
        if stored_reference not in merged_refs:
            merged_refs.append(stored_reference)
        service.ui_state_store.replace_document_assignments(
            context.item_id,
            dataset_scope=context.dataset_scope,
            document_refs=merged_refs,
        )
        return self.get_item_detail(item_id)
