"""Section 232 draft batch and published ruleset persistence."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import date
from threading import RLock
from typing import Any, Dict, List, Literal, Optional, Sequence

from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .hts_catalog import canonicalize_hts_code
from .persistence_common import (
    column_exists,
    ensure_column,
    ensure_index,
    execute_compatible_update,
    fetch_rows,
    qualified_table as _qualified_table,
    quote_identifier as _quote_identifier,
)
from .timing import utc_now_iso_after

ReviewDecision = Literal["pending", "accepted", "rejected"]
BatchStatus = Literal["pending_review", "published"]
RuleType = Literal["include", "remove", "rate_schedule"]
CoverageEffect = Literal["include", "remove"]
CandidateQuality = Literal["normal", "suspect"]

_ALLOWED_BATCH_STATUSES = {"pending_review", "published"}
_ALLOWED_RULE_TYPES = {"include", "remove", "rate_schedule"}
_ALLOWED_COVERAGE_EFFECTS = {"include", "remove"}
_ALLOWED_REVIEW_DECISIONS = {"pending", "accepted", "rejected"}
_RULE_TYPE_COVERAGE_EFFECTS = {
    "include": "include",
    "rate_schedule": "include",
    "remove": "remove",
}
_UNSPECIFIED_SCOPE_KEYS = {
    "",
    "all",
    "mixed",
    "mixed/unspecified",
    "multi",
    "multiple",
    "unknown",
    "unspecified",
}
_METAL_SCOPE_TOKEN_RE = re.compile(r"\b(?:steel|aluminum|aluminium|copper)\b", re.IGNORECASE)
_EXCLUSIONARY_SCOPE_RE = re.compile(
    r"\b(?:not|except|excluding|excluded|without|other\s+than|minus|but\s+not|save)\b",
    re.IGNORECASE,
)
_CANONICAL_METAL_SCOPE_ORDER = {
    "aluminum": 0,
    "copper": 1,
    "steel": 2,
}
_VERSION_PATTERN = re.compile(r"section232-v(\d+)$")

def _safe_json_loads(raw: Any, *, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(str(raw))
    except Exception:
        return default


def _normalize_scope_key(scope: str) -> str:
    normalized = str(scope or "").strip().lower()
    if normalized in _UNSPECIFIED_SCOPE_KEYS:
        return "unspecified"
    if _EXCLUSIONARY_SCOPE_RE.search(normalized):
        raise ValueError(f"invalid metal_scope: {scope!r}")
    tokens = []
    seen = set()
    for match in _METAL_SCOPE_TOKEN_RE.finditer(normalized):
        token = match.group(0).lower()
        if token == "aluminium":
            token = "aluminum"
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    if tokens:
        tokens.sort(key=lambda token: (_CANONICAL_METAL_SCOPE_ORDER.get(token, 99), token))
        return "+".join(tokens)
    raise ValueError(f"invalid metal_scope: {scope!r}")


def _coerce_legacy_scope(scope: object) -> str:
    raw_scope = str(scope or "unspecified")
    try:
        _normalize_scope_key(raw_scope)
    except ValueError:
        return "unspecified"
    return raw_scope


def _db_optional_date(value: Optional[str]) -> str:
    return str(value or "")


def section_232_rule_overlay_identity(candidate: "Section232DraftRuleCandidate") -> tuple[str, str]:
    return (
        canonicalize_hts_code(candidate.hts_code),
        _normalize_scope_key(candidate.metal_scope),
    )


def section_232_rule_material_group_identity(candidate: "Section232DraftRuleCandidate") -> tuple[Any, ...]:
    return (
        canonicalize_hts_code(candidate.hts_code),
        candidate.rule_type,
        candidate.coverage_effect,
        _normalize_scope_key(candidate.metal_scope),
    )


def section_232_rule_material_identity(candidate: "Section232DraftRuleCandidate") -> tuple[Any, ...]:
    return (*section_232_rule_material_group_identity(candidate), candidate.effective_from, candidate.effective_to)


def _publication_window_covers(
    covering_candidate: "Section232DraftRuleCandidate",
    covered_candidate: "Section232DraftRuleCandidate",
) -> bool:
    covering_start = date.fromisoformat(covering_candidate.effective_from) if covering_candidate.effective_from else date.min
    covering_end = date.fromisoformat(covering_candidate.effective_to) if covering_candidate.effective_to else date.max
    covered_start = date.fromisoformat(covered_candidate.effective_from) if covered_candidate.effective_from else date.min
    covered_end = date.fromisoformat(covered_candidate.effective_to) if covered_candidate.effective_to else date.max
    return covering_start <= covered_start and covering_end >= covered_end


def _remove_covers_include(
    *,
    include_rule: "Section232DraftRuleCandidate",
    remove_rule: "Section232DraftRuleCandidate",
) -> bool:
    remove_scope = _normalize_scope_key(remove_rule.metal_scope)
    include_scope = _normalize_scope_key(include_rule.metal_scope)
    return remove_scope == "unspecified" or remove_scope == include_scope


def project_active_eligible_codes(
    rules: Sequence["Section232DraftRuleCandidate"],
    *,
    on_date: date,
) -> List[str]:
    rules = project_current_rules_by_hts_code(rules)
    rules_by_code: Dict[str, Dict[str, List[Section232DraftRuleCandidate]]] = {}
    for candidate in rules:
        normalized = canonicalize_hts_code(candidate.hts_code)
        if not normalized or not _is_effective_on_date(candidate, on_date):
            continue
        bucket = rules_by_code.setdefault(normalized, {"include": [], "remove": []})
        bucket[candidate.coverage_effect].append(candidate)

    eligible_codes: List[str] = []
    for normalized_code, bucket in rules_by_code.items():
        effective_includes = bucket["include"]
        effective_removes = bucket["remove"]
        if not effective_includes:
            continue
        if any(
            not any(
                _remove_covers_include(include_rule=include_rule, remove_rule=remove_rule)
                for remove_rule in effective_removes
            )
            for include_rule in effective_includes
        ):
            eligible_codes.append(normalized_code)
    eligible_codes.sort()
    return eligible_codes


def section_232_rule_matches_hts_code(rule_code: str, hts_code: str) -> bool:
    """Return True when a published legal HTS scope applies to a more specific code."""

    normalized_rule_code = canonicalize_hts_code(rule_code)
    normalized_hts_code = canonicalize_hts_code(hts_code)
    return bool(normalized_rule_code and normalized_hts_code and normalized_hts_code.startswith(normalized_rule_code))


@dataclass(frozen=True)
class Section232DraftBatch:
    batch_id: str
    source_ids: List[str]
    source_filenames: List[str]
    created_at: str
    status: BatchStatus


@dataclass(frozen=True)
class Section232DeletedDraftBatch:
    batch_id: str
    source_ids: List[str]
    source_filenames: List[str]
    deleted_rule_count: int = 0


@dataclass(frozen=True)
class Section232DraftRuleCandidate:
    candidate_id: str
    batch_id: str
    hts_code: str
    rule_type: RuleType
    coverage_effect: CoverageEffect
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    metal_scope: str = "unspecified"
    source_document_ids: List[str] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    source_excerpt: str = ""
    interpreter_confidence: float = 0.0
    catalog_match_found: bool = False
    review_decision: ReviewDecision = "pending"
    rate_text: Optional[str] = None
    candidate_quality: CandidateQuality = "normal"
    candidate_flags: List[str] = field(default_factory=list)
    processed_at: Optional[str] = None


@dataclass(frozen=True)
class Section232PublishedRuleset:
    version: str
    published_at: str
    published_by: str
    accepted_rules: List[Section232DraftRuleCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class Section232PublishedRulesetInfo:
    version: str
    published_at: str
    published_by: str


@dataclass(frozen=True)
class Section232CodeDeleteOverride:
    hts_code: str
    deleted_at: str
    deleted_by: str


@dataclass(frozen=True)
class Section232CandidatePage:
    total: int
    limit: int
    offset: int
    candidates: List[Section232DraftRuleCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class Section232DraftBatchStats:
    total: int
    pending_count: int
    accepted_count: int
    rejected_count: int
    warning_count: int


def _validate_choice(field: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")


def _validate_batch_record(batch: Section232DraftBatch) -> None:
    _validate_choice("status", batch.status, _ALLOWED_BATCH_STATUSES)


def _validate_effective_dates(effective_from: Optional[str], effective_to: Optional[str]) -> None:
    start_date = date.min
    if effective_from not in (None, ""):
        try:
            start_date = date.fromisoformat(str(effective_from))
        except ValueError as exc:
            raise ValueError(f"invalid effective_from: {effective_from!r}") from exc
    if effective_to in (None, ""):
        return
    try:
        end_date = date.fromisoformat(str(effective_to))
    except ValueError as exc:
        raise ValueError(f"invalid effective_to: {effective_to!r}") from exc
    if end_date < start_date:
        raise ValueError(
            f"invalid effective range: effective_to {effective_to!r} is earlier than effective_from {effective_from!r}"
        )


def _validate_candidate_record(candidate: Section232DraftRuleCandidate) -> None:
    normalized_hts_code = canonicalize_hts_code(candidate.hts_code)
    if not normalized_hts_code:
        raise ValueError(f"invalid hts_code: {candidate.hts_code!r}")
    _validate_choice("rule_type", candidate.rule_type, _ALLOWED_RULE_TYPES)
    _validate_choice("coverage_effect", candidate.coverage_effect, _ALLOWED_COVERAGE_EFFECTS)
    _validate_choice("review_decision", candidate.review_decision, _ALLOWED_REVIEW_DECISIONS)
    _normalize_scope_key(candidate.metal_scope)
    if candidate.candidate_quality not in {"normal", "suspect"}:
        raise ValueError(f"invalid candidate_quality: {candidate.candidate_quality!r}")
    expected_coverage_effect = _RULE_TYPE_COVERAGE_EFFECTS[candidate.rule_type]
    if candidate.coverage_effect != expected_coverage_effect:
        raise ValueError(
            f"invalid rule combination: rule_type {candidate.rule_type!r} requires coverage_effect "
            f"{expected_coverage_effect!r}, got {candidate.coverage_effect!r}"
        )
    _validate_effective_dates(candidate.effective_from, candidate.effective_to)


def _validate_review_decision(decision: ReviewDecision) -> None:
    if decision not in _ALLOWED_REVIEW_DECISIONS:
        raise ValueError(f"invalid review_decision: {decision!r}")


def _normalize_candidate_ids(candidate_ids: Sequence[str]) -> List[str]:
    normalized_candidate_ids: List[str] = []
    seen_candidate_ids: set[str] = set()
    for candidate_id in candidate_ids:
        normalized_candidate_id = str(candidate_id or "").strip()
        if not normalized_candidate_id:
            raise ValueError("candidate_ids must contain non-empty values")
        if normalized_candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(normalized_candidate_id)
        normalized_candidate_ids.append(normalized_candidate_id)
    if not normalized_candidate_ids:
        raise ValueError("candidate_ids must contain at least one value")
    return normalized_candidate_ids


def _normalize_optional_candidate_ids(candidate_ids: Sequence[str]) -> List[str]:
    normalized_candidate_ids: List[str] = []
    seen_candidate_ids: set[str] = set()
    for candidate_id in candidate_ids:
        normalized_candidate_id = str(candidate_id or "").strip()
        if not normalized_candidate_id:
            raise ValueError("candidate_ids must contain non-empty values")
        if normalized_candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(normalized_candidate_id)
        normalized_candidate_ids.append(normalized_candidate_id)
    return normalized_candidate_ids


def _review_candidate_match_rank(candidate: Section232DraftRuleCandidate) -> int:
    return 1 if candidate.catalog_match_found else 0


def _review_candidate_sort_key(candidate: Section232DraftRuleCandidate) -> tuple[int, str, str]:
    normalized_code = canonicalize_hts_code(candidate.hts_code).replace(".", "")
    return (
        _review_candidate_match_rank(candidate),
        normalized_code,
        candidate.candidate_id,
    )


def _current_rule_sort_key(candidate: Section232DraftRuleCandidate, position: int) -> tuple[str, int, str]:
    return (
        str(candidate.processed_at or ""),
        position,
        str(candidate.candidate_id or ""),
    )


def project_current_rules_by_hts_code(
    candidates: Sequence[Section232DraftRuleCandidate],
) -> List[Section232DraftRuleCandidate]:
    selected_by_code: Dict[str, tuple[tuple[str, int, str], Section232DraftRuleCandidate]] = {}
    for position, candidate in enumerate(candidates):
        normalized_code = canonicalize_hts_code(candidate.hts_code)
        if not normalized_code:
            continue
        sort_key = _current_rule_sort_key(candidate, position)
        selected = selected_by_code.get(normalized_code)
        if selected is None or sort_key > selected[0]:
            selected_by_code[normalized_code] = (sort_key, candidate)
    return [_clone_candidate(candidate) for _sort_key, candidate in selected_by_code.values()]


def _clone_published_ruleset(snapshot: Section232PublishedRuleset) -> Section232PublishedRuleset:
    return replace(
        snapshot,
        accepted_rules=[_clone_candidate(candidate) for candidate in snapshot.accepted_rules],
    )


def _clone_delete_override(override: Section232CodeDeleteOverride) -> Section232CodeDeleteOverride:
    return replace(override)


def _clone_batch(batch: Section232DraftBatch) -> Section232DraftBatch:
    return replace(
        batch,
        source_ids=list(batch.source_ids),
        source_filenames=list(batch.source_filenames),
    )


def _clone_candidate(candidate: Section232DraftRuleCandidate) -> Section232DraftRuleCandidate:
    return replace(
        candidate,
        source_document_ids=list(candidate.source_document_ids),
        source_pages=list(candidate.source_pages),
        candidate_flags=list(candidate.candidate_flags),
    )


def _is_effective_on_date(candidate: Section232DraftRuleCandidate, on_date: date) -> bool:
    if candidate.effective_from:
        effective_from = date.fromisoformat(candidate.effective_from)
        if on_date < effective_from:
            return False
    if candidate.effective_to:
        effective_to = date.fromisoformat(candidate.effective_to)
        if on_date > effective_to:
            return False
    return True


def _version_sort_key(version: str) -> tuple[int, str]:
    match = _VERSION_PATTERN.match(str(version or ""))
    return (int(match.group(1)), str(version)) if match else (-1, str(version))


def _resolve_publish_candidates(
    *,
    batch_id: str,
    draft_candidates: Sequence[Section232DraftRuleCandidate],
    accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate] | None = None,
) -> List[Section232DraftRuleCandidate]:
    for candidate in draft_candidates:
        _validate_candidate_record(candidate)

    pending_candidates = [candidate for candidate in draft_candidates if candidate.review_decision == "pending"]
    if pending_candidates:
        raise ValueError(f"draft batch {batch_id} has pending candidates and cannot be published")

    accepted_draft_candidates = [candidate for candidate in draft_candidates if candidate.review_decision == "accepted"]
    if accepted_rules_snapshot is None:
        if not accepted_draft_candidates:
            raise ValueError(f"draft batch {batch_id} has no accepted candidates")
        publish_candidates = accepted_draft_candidates
    else:
        publish_candidates = list(accepted_rules_snapshot)
    if not publish_candidates:
        if accepted_rules_snapshot is None:
            raise ValueError(f"draft batch {batch_id} has no accepted candidates")
        return []

    prepared_publish_candidates: List[Section232DraftRuleCandidate] = []
    publish_candidate_ids: set[str] = set()
    publish_hts_codes: set[str] = set()
    for candidate in publish_candidates:
        _validate_candidate_record(candidate)
        if candidate.review_decision != "accepted":
            raise ValueError(f"published rules snapshot for batch {batch_id} must contain accepted candidates only")
        if candidate.candidate_id in publish_candidate_ids:
            raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r} in published snapshot for batch {batch_id}")
        normalized_hts_code = canonicalize_hts_code(candidate.hts_code)
        if normalized_hts_code in publish_hts_codes:
            raise ValueError(
                f"published rules snapshot for batch {batch_id} has duplicate normalized HTS code "
                f"{normalized_hts_code!r}"
            )
        publish_candidate_ids.add(candidate.candidate_id)
        publish_hts_codes.add(normalized_hts_code)
        prepared_publish_candidates.append(_clone_candidate(candidate))

    if accepted_rules_snapshot is None:
        missing_accepted_draft_candidates = []
        for candidate in accepted_draft_candidates:
            if candidate.candidate_id in publish_candidate_ids:
                continue
            if any(
                section_232_rule_material_group_identity(publish_candidate)
                == section_232_rule_material_group_identity(candidate)
                and _publication_window_covers(publish_candidate, candidate)
                for publish_candidate in prepared_publish_candidates
            ):
                continue
            missing_accepted_draft_candidates.append(candidate.candidate_id)
        if missing_accepted_draft_candidates:
            raise ValueError(
                f"published rules snapshot for batch {batch_id} is missing accepted draft candidates: "
                f"{', '.join(sorted(missing_accepted_draft_candidates))}"
            )

    return prepared_publish_candidates


class InMemorySection232RulesetStore:
    """Test-friendly in-memory store for draft batches and published rulesets."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._batches: Dict[str, Section232DraftBatch] = {}
        self._candidates: Dict[str, Dict[str, Section232DraftRuleCandidate]] = {}
        self._active_rules: List[Section232DraftRuleCandidate] = []
        self._published_rulesets: Dict[str, Section232PublishedRuleset] = {}
        self._delete_overrides: Dict[str, Section232CodeDeleteOverride] = {}
        self._version_counter = 0
        self._active_version: str | None = None
        self._last_published_at: str | None = None

    def _next_batch_created_at_locked(self) -> str:
        latest_created_at = max((batch.created_at for batch in self._batches.values()), default="")
        return utc_now_iso_after(latest_created_at or None)

    def _next_published_at_locked(self) -> str:
        return utc_now_iso_after(self._last_published_at)

    def _next_deleted_at_locked(self) -> str:
        latest_deleted_at = max((item.deleted_at for item in self._delete_overrides.values()), default="")
        return utc_now_iso_after(latest_deleted_at or None)

    def create_draft_batch(self, *, source_ids: Sequence[str], source_filenames: Sequence[str]) -> Section232DraftBatch:
        with self._lock:
            batch = Section232DraftBatch(
                batch_id=str(uuid.uuid4()),
                source_ids=list(source_ids),
                source_filenames=list(source_filenames),
                created_at=self._next_batch_created_at_locked(),
                status="pending_review",
            )
            _validate_batch_record(batch)
            self._batches[batch.batch_id] = _clone_batch(batch)
            self._candidates[batch.batch_id] = {}
        return _clone_batch(batch)

    def replace_batch_candidates(
        self,
        batch_id: str,
        candidates: Sequence[Section232DraftRuleCandidate],
    ) -> None:
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            candidate_map: Dict[str, Section232DraftRuleCandidate] = {}
            seen_candidate_ids = set()
            for candidate in candidates:
                if candidate.batch_id != batch_id:
                    candidate = replace(candidate, batch_id=batch_id)
                _validate_candidate_record(candidate)
                if candidate.candidate_id in seen_candidate_ids:
                    raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r} in batch {batch_id}")
                seen_candidate_ids.add(candidate.candidate_id)
                candidate_map[candidate.candidate_id] = _clone_candidate(candidate)
            self._candidates[batch_id] = candidate_map

    def review_candidate(self, batch_id: str, candidate_id: str, *, decision: ReviewDecision) -> Section232DraftRuleCandidate:
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            _validate_review_decision(decision)
            current = self._require_candidate_locked(batch_id, candidate_id)
            _validate_candidate_record(current)
            updated = replace(current, review_decision=decision)
            _validate_candidate_record(updated)
            self._candidates[batch_id][candidate_id] = _clone_candidate(updated)
            return _clone_candidate(updated)

    def review_candidates(
        self,
        batch_id: str,
        candidate_ids: Sequence[str],
        *,
        decision: ReviewDecision,
    ) -> List[Section232DraftRuleCandidate]:
        normalized_candidate_ids = _normalize_candidate_ids(candidate_ids)
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            _validate_review_decision(decision)
            updated_candidates: List[Section232DraftRuleCandidate] = []
            for candidate_id in normalized_candidate_ids:
                current = self._require_candidate_locked(batch_id, candidate_id)
                _validate_candidate_record(current)
                updated = replace(current, review_decision=decision)
                _validate_candidate_record(updated)
                self._candidates[batch_id][candidate_id] = _clone_candidate(updated)
                updated_candidates.append(_clone_candidate(updated))
            return updated_candidates

    def review_all_candidates(
        self,
        batch_id: str,
        *,
        decision: ReviewDecision,
        excluded_candidate_ids: Sequence[str] = (),
    ) -> int:
        normalized_excluded_candidate_ids = set(_normalize_optional_candidate_ids(excluded_candidate_ids))
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            _validate_review_decision(decision)
            updated_count = 0
            for candidate_id, current in self._candidates.get(batch_id, {}).items():
                if candidate_id in normalized_excluded_candidate_ids or current.review_decision == decision:
                    continue
                updated = replace(current, review_decision=decision)
                _validate_candidate_record(updated)
                self._candidates[batch_id][candidate_id] = _clone_candidate(updated)
                updated_count += 1
            return updated_count

    def delete_draft_candidates_by_hts_code(self, batch_id: str, *, hts_code: str) -> int:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            raise ValueError(f"invalid hts_code: {hts_code!r}")
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            candidate_map = self._candidates.get(batch_id, {})
            deleted_candidate_ids = [
                candidate_id
                for candidate_id, candidate in candidate_map.items()
                if canonicalize_hts_code(candidate.hts_code) == normalized_hts_code
            ]
            for candidate_id in deleted_candidate_ids:
                candidate_map.pop(candidate_id, None)
            return len(deleted_candidate_ids)

    def delete_pending_draft_batch(self, batch_id: str) -> Section232DeletedDraftBatch:
        with self._lock:
            batch = self._require_pending_batch_locked(batch_id)
            deleted_rule_count = len(self._candidates.get(batch_id, {}))
            self._candidates.pop(batch_id, None)
            self._batches.pop(batch_id, None)
            return Section232DeletedDraftBatch(
                batch_id=batch.batch_id,
                source_ids=list(batch.source_ids),
                source_filenames=list(batch.source_filenames),
                deleted_rule_count=deleted_rule_count,
            )

    def publish_snapshot(
        self,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate],
    ) -> Section232PublishedRuleset:
        publish_candidates = _resolve_publish_candidates(
            batch_id="snapshot",
            draft_candidates=list(accepted_rules_snapshot),
            accepted_rules_snapshot=accepted_rules_snapshot,
        )
        with self._lock:
            self._active_rules = [_clone_candidate(candidate) for candidate in publish_candidates]
            self._version_counter += 1
            version = f"section232-v{self._version_counter:04d}"
            published_at = self._next_published_at_locked()
            self._active_version = version
            self._last_published_at = published_at
            snapshot = Section232PublishedRuleset(
                version=version,
                published_at=published_at,
                published_by=published_by,
                accepted_rules=[_clone_candidate(candidate) for candidate in publish_candidates],
            )
            self._published_rulesets[version] = _clone_published_ruleset(snapshot)
            return _clone_published_ruleset(snapshot)

    def publish_batch(
        self,
        batch_id: str,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate] | None = None,
    ) -> Section232PublishedRuleset:
        with self._lock:
            batch = self._require_pending_batch_locked(batch_id)
            _validate_batch_record(batch)
            draft_candidates = list(self._candidates.get(batch_id, {}).values())
            publish_candidates = _resolve_publish_candidates(
                batch_id=batch_id,
                draft_candidates=draft_candidates,
                accepted_rules_snapshot=accepted_rules_snapshot,
            )
            self._batches[batch_id] = _clone_batch(replace(batch, status="published"))
            _validate_batch_record(self._batches[batch_id])
            snapshot = self.publish_snapshot(
                published_by=published_by,
                accepted_rules_snapshot=publish_candidates,
            )
            return snapshot

    def list_active_rules(self) -> List[Section232DraftRuleCandidate]:
        with self._lock:
            return project_current_rules_by_hts_code(self._active_rules)

    def list_draft_candidates(self, *, batch_id: str) -> List[Section232DraftRuleCandidate]:
        with self._lock:
            self._require_batch_locked(batch_id)
            return [_clone_candidate(candidate) for candidate in self._candidates.get(batch_id, {}).values()]

    def get_draft_batch_stats(self, *, batch_id: str) -> Section232DraftBatchStats:
        with self._lock:
            self._require_batch_locked(batch_id)
            candidates = list(self._candidates.get(batch_id, {}).values())
        return Section232DraftBatchStats(
            total=len(candidates),
            pending_count=sum(1 for candidate in candidates if candidate.review_decision == "pending"),
            accepted_count=sum(1 for candidate in candidates if candidate.review_decision == "accepted"),
            rejected_count=sum(1 for candidate in candidates if candidate.review_decision == "rejected"),
            warning_count=sum(
                1
                for candidate in candidates
                if (not candidate.catalog_match_found)
                or candidate.candidate_quality == "suspect"
                or bool(candidate.candidate_flags)
            ),
        )

    def list_draft_candidate_page(
        self,
        *,
        batch_id: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        with self._lock:
            self._require_batch_locked(batch_id)
            ordered_candidates = sorted(
                self._candidates.get(batch_id, {}).values(),
                key=_review_candidate_sort_key,
            )
        paged_candidates = ordered_candidates[offset : offset + limit]
        return Section232CandidatePage(
            total=len(ordered_candidates),
            limit=limit,
            offset=offset,
            candidates=[_clone_candidate(candidate) for candidate in paged_candidates],
        )

    def list_draft_batches(self, *, status: Optional[BatchStatus] = None) -> List[Section232DraftBatch]:
        with self._lock:
            batches = [
                _clone_batch(batch)
                for batch in self._batches.values()
                if status is None or batch.status == status
            ]
        batches.sort(key=lambda item: item.created_at, reverse=True)
        return batches

    def get_active_ruleset_version(self) -> str | None:
        with self._lock:
            return self._active_version

    def count_pending_batches(self) -> int:
        with self._lock:
            return sum(1 for batch in self._batches.values() if batch.status == "pending_review")

    def get_last_published_at(self) -> str | None:
        with self._lock:
            return self._last_published_at

    def get_published_ruleset(self, version: str) -> Optional[Section232PublishedRuleset]:
        with self._lock:
            snapshot = self._published_rulesets.get(version)
            if snapshot is None:
                return None
            return _clone_published_ruleset(snapshot)

    def get_published_ruleset_info(self, version: str) -> Optional[Section232PublishedRulesetInfo]:
        with self._lock:
            snapshot = self._published_rulesets.get(version)
            if snapshot is None:
                return None
            return Section232PublishedRulesetInfo(
                version=snapshot.version,
                published_at=snapshot.published_at,
                published_by=snapshot.published_by,
            )

    def list_published_ruleset_infos(self) -> List[Section232PublishedRulesetInfo]:
        with self._lock:
            infos = [
                Section232PublishedRulesetInfo(
                    version=snapshot.version,
                    published_at=snapshot.published_at,
                    published_by=snapshot.published_by,
                )
                for snapshot in self._published_rulesets.values()
            ]
        infos.sort(key=lambda item: _version_sort_key(item.version))
        return infos

    def create_delete_override(self, *, hts_code: str, deleted_by: str) -> Section232CodeDeleteOverride:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            raise ValueError(f"invalid hts_code: {hts_code!r}")
        with self._lock:
            override = Section232CodeDeleteOverride(
                hts_code=normalized_hts_code,
                deleted_at=self._next_deleted_at_locked(),
                deleted_by=str(deleted_by or "").strip(),
            )
            self._delete_overrides[normalized_hts_code] = _clone_delete_override(override)
            return _clone_delete_override(override)

    def get_delete_override(self, hts_code: str) -> Optional[Section232CodeDeleteOverride]:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            return None
        with self._lock:
            override = self._delete_overrides.get(normalized_hts_code)
            return None if override is None else _clone_delete_override(override)

    def list_delete_overrides(self) -> List[Section232CodeDeleteOverride]:
        with self._lock:
            overrides = [_clone_delete_override(item) for item in self._delete_overrides.values()]
        overrides.sort(key=lambda item: (item.deleted_at, item.hts_code), reverse=True)
        return overrides

    def list_published_rule_page(
        self,
        *,
        version: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        with self._lock:
            snapshot = self._published_rulesets.get(version)
            if snapshot is None:
                raise KeyError(f"published ruleset {version} not found")
            ordered_candidates = sorted(snapshot.accepted_rules, key=_review_candidate_sort_key)
        paged_candidates = ordered_candidates[offset : offset + limit]
        return Section232CandidatePage(
            total=len(ordered_candidates),
            limit=limit,
            offset=offset,
            candidates=[_clone_candidate(candidate) for candidate in paged_candidates],
        )

    def list_active_eligible_codes(self, *, on_date: date) -> List[str]:
        with self._lock:
            return project_active_eligible_codes(self._active_rules, on_date=on_date)

    def clear_all(self) -> Dict[str, int]:
        with self._lock:
            cleared_draft_batch_count = len(self._batches)
            cleared_draft_rule_count = sum(len(candidate_map) for candidate_map in self._candidates.values())
            cleared_published_ruleset_count = len(self._published_rulesets)
            cleared_published_rule_count = sum(
                len(snapshot.accepted_rules) for snapshot in self._published_rulesets.values()
            )
            cleared_delete_override_count = len(self._delete_overrides)
            self._batches = {}
            self._candidates = {}
            self._active_rules = []
            self._published_rulesets = {}
            self._delete_overrides = {}
            self._version_counter = 0
            self._active_version = None
            self._last_published_at = None
        return {
            "cleared_draft_batch_count": cleared_draft_batch_count,
            "cleared_draft_rule_count": cleared_draft_rule_count,
            "cleared_published_ruleset_count": cleared_published_ruleset_count,
            "cleared_published_rule_count": cleared_published_rule_count,
            "cleared_delete_override_count": cleared_delete_override_count,
        }

    def get_draft_batch(self, batch_id: str) -> Section232DraftBatch:
        with self._lock:
            return _clone_batch(self._require_batch_locked(batch_id))

    def _require_batch_locked(self, batch_id: str) -> Section232DraftBatch:
        try:
            return self._batches[batch_id]
        except KeyError as exc:
            raise KeyError(f"draft batch {batch_id} not found") from exc

    def _require_pending_batch_locked(self, batch_id: str) -> Section232DraftBatch:
        batch = self._require_batch_locked(batch_id)
        _validate_batch_record(batch)
        if batch.status != "pending_review":
            raise ValueError(f"draft batch {batch_id} is already published")
        return batch

    def _require_candidate_locked(self, batch_id: str, candidate_id: str) -> Section232DraftRuleCandidate:
        try:
            return self._candidates[batch_id][candidate_id]
        except KeyError as exc:
            raise KeyError(f"candidate {candidate_id} not found in batch {batch_id}") from exc


class PersistedSection232RulesetStore:
    """Persist Section 232 draft batches and rulesets in SAP HANA."""

    def __init__(
        self,
        settings: MetalCompositionSettings,
        *,
        connection: Optional[HANAConnection] = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or HANAConnection()
        self.schema = settings.section_232_hana_schema or settings.hana_schema or None
        self.draft_batches_table = settings.section_232_draft_batches_table
        self.draft_rules_table = settings.section_232_draft_rules_table
        self.rulesets_table = settings.section_232_rulesets_table
        self.ruleset_rules_table = settings.section_232_ruleset_rules_table
        self.delete_overrides_table = settings.section_232_delete_overrides_table
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        if not self.connection.table_exists(self.draft_batches_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.draft_batches_table, self.schema)} (
                    "BATCH_ID" NVARCHAR(36) PRIMARY KEY,
                    "SOURCE_IDS_JSON" NCLOB,
                    "SOURCE_FILENAMES_JSON" NCLOB,
                    "CREATED_AT" NVARCHAR(64) NOT NULL,
                    "STATUS" NVARCHAR(32) NOT NULL
                )
                """
            )
            ensure_index(
                self.connection,
                self.draft_batches_table,
                schema=self.schema,
                columns=("STATUS", "CREATED_AT"),
            )
        if not self.connection.table_exists(self.draft_rules_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.draft_rules_table, self.schema)} (
                    "BATCH_ID" NVARCHAR(36) NOT NULL,
                    "CANDIDATE_ID" NVARCHAR(255) NOT NULL,
                    "POSITION" INTEGER NOT NULL,
                    "HTS_CODE" NVARCHAR(32) NOT NULL,
                    "RULE_TYPE" NVARCHAR(32) NOT NULL,
                    "COVERAGE_EFFECT" NVARCHAR(32) NOT NULL,
                    "EFFECTIVE_FROM" NVARCHAR(32),
                    "EFFECTIVE_TO" NVARCHAR(32),
                    "RATE_TEXT" NCLOB,
                    "METAL_SCOPE" NVARCHAR(255) NOT NULL,
                    "SOURCE_DOCUMENT_IDS_JSON" NCLOB,
                    "SOURCE_PAGES_JSON" NCLOB,
                    "SOURCE_EXCERPT" NCLOB,
                    "INTERPRETER_CONFIDENCE" DOUBLE NOT NULL,
                    "CATALOG_MATCH_FOUND" SMALLINT NOT NULL,
                    "CANDIDATE_QUALITY" NVARCHAR(32) NOT NULL,
                    "CANDIDATE_FLAGS_JSON" NCLOB,
                    "PROCESSED_AT" NVARCHAR(64),
                    "REVIEW_DECISION" NVARCHAR(32) NOT NULL,
                    PRIMARY KEY ("BATCH_ID", "CANDIDATE_ID")
                )
                """
            )
            ensure_index(
                self.connection,
                self.draft_rules_table,
                schema=self.schema,
                columns=("BATCH_ID", "POSITION"),
            )
        if not self.connection.table_exists(self.rulesets_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.rulesets_table, self.schema)} (
                    "VERSION" NVARCHAR(32) PRIMARY KEY,
                    "PUBLISHED_AT" NVARCHAR(64) NOT NULL,
                    "PUBLISHED_BY" NVARCHAR(255) NOT NULL
                )
                """
            )
            ensure_index(
                self.connection,
                self.rulesets_table,
                schema=self.schema,
                columns=("PUBLISHED_AT",),
            )
        if not self.connection.table_exists(self.ruleset_rules_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.ruleset_rules_table, self.schema)} (
                    "VERSION" NVARCHAR(32) NOT NULL,
                    "CANDIDATE_ID" NVARCHAR(255) NOT NULL,
                    "POSITION" INTEGER NOT NULL,
                    "BATCH_ID" NVARCHAR(36) NOT NULL,
                    "HTS_CODE" NVARCHAR(32) NOT NULL,
                    "RULE_TYPE" NVARCHAR(32) NOT NULL,
                    "COVERAGE_EFFECT" NVARCHAR(32) NOT NULL,
                    "EFFECTIVE_FROM" NVARCHAR(32),
                    "EFFECTIVE_TO" NVARCHAR(32),
                    "RATE_TEXT" NCLOB,
                    "METAL_SCOPE" NVARCHAR(255) NOT NULL,
                    "SOURCE_DOCUMENT_IDS_JSON" NCLOB,
                    "SOURCE_PAGES_JSON" NCLOB,
                    "SOURCE_EXCERPT" NCLOB,
                    "INTERPRETER_CONFIDENCE" DOUBLE NOT NULL,
                    "CATALOG_MATCH_FOUND" SMALLINT NOT NULL,
                    "CANDIDATE_QUALITY" NVARCHAR(32) NOT NULL,
                    "CANDIDATE_FLAGS_JSON" NCLOB,
                    "PROCESSED_AT" NVARCHAR(64),
                    "REVIEW_DECISION" NVARCHAR(32) NOT NULL,
                    PRIMARY KEY ("VERSION", "CANDIDATE_ID")
                )
                """
            )
            ensure_index(
                self.connection,
                self.ruleset_rules_table,
                schema=self.schema,
                columns=("VERSION", "POSITION"),
            )
        if not self.connection.table_exists(self.delete_overrides_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.delete_overrides_table, self.schema)} (
                    "HTS_CODE" NVARCHAR(32) PRIMARY KEY,
                    "DELETED_AT" NVARCHAR(64) NOT NULL,
                    "DELETED_BY" NVARCHAR(255) NOT NULL
                )
                """
            )
            ensure_index(
                self.connection,
                self.delete_overrides_table,
                schema=self.schema,
                columns=("DELETED_AT",),
            )
        self._ensure_rules_table_columns(self.draft_rules_table)
        self._ensure_rules_table_columns(self.ruleset_rules_table)

    def _ensure_rules_table_columns(self, table: str) -> None:
        ensure_column(self.connection, table, "RATE_TEXT", "NCLOB", schema=self.schema)
        ensure_column(self.connection, table, "CANDIDATE_QUALITY", "NVARCHAR(32)", schema=self.schema)
        ensure_column(self.connection, table, "CANDIDATE_FLAGS_JSON", "NCLOB", schema=self.schema)
        ensure_column(self.connection, table, "PROCESSED_AT", "NVARCHAR(64)", schema=self.schema)
        table_name = _qualified_table(table, self.schema)
        execute_compatible_update(
            self.connection,
            f"""
            UPDATE {table_name}
            SET "CANDIDATE_QUALITY" = 'normal'
            WHERE "CANDIDATE_QUALITY" IS NULL OR "CANDIDATE_QUALITY" = ''
            """
        )
        execute_compatible_update(
            self.connection,
            f"""
            UPDATE {table_name}
            SET "CANDIDATE_FLAGS_JSON" = TO_NCLOB('[]')
            WHERE "CANDIDATE_FLAGS_JSON" IS NULL OR LENGTH("CANDIDATE_FLAGS_JSON") = 0
            """,
            fallback_sql=f"""
            UPDATE {table_name}
            SET "CANDIDATE_FLAGS_JSON" = '[]'
            WHERE "CANDIDATE_FLAGS_JSON" IS NULL OR LENGTH("CANDIDATE_FLAGS_JSON") = 0
            """,
        )

    def _column_exists(self, table: str, column: str) -> bool:
        return column_exists(self.connection, table, column, schema=self.schema)

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        ensure_column(self.connection, table, column, definition, schema=self.schema)

    def _execute_compatible_update(self, sql: str, *, fallback_sql: str | None = None) -> None:
        execute_compatible_update(self.connection, sql, fallback_sql=fallback_sql)

    def _ensure_index(self, table: str, *columns: str) -> None:
        ensure_index(self.connection, table, schema=self.schema, columns=columns)

    def _fetch_rows(self, sql: str, params: Sequence[object] | None = None) -> List[Dict[str, object]]:
        return fetch_rows(self.connection, sql, params)

    def _next_timestamp_locked(self, *, table: str, column: str) -> str:
        rows = self._fetch_rows(
            f'SELECT MAX({_quote_identifier(column)}) AS "MAX_VALUE" '
            f'FROM {_qualified_table(table, self.schema)}'
        )
        latest_value = str((rows[0] or {}).get("max_value") or "").strip() if rows else ""
        return utc_now_iso_after(latest_value or None)

    def create_draft_batch(self, *, source_ids: Sequence[str], source_filenames: Sequence[str]) -> Section232DraftBatch:
        with self._lock:
            batch = Section232DraftBatch(
                batch_id=str(uuid.uuid4()),
                source_ids=list(source_ids),
                source_filenames=list(source_filenames),
                created_at=self._next_timestamp_locked(table=self.draft_batches_table, column="CREATED_AT"),
                status="pending_review",
            )
            _validate_batch_record(batch)
            self.connection.execute(
                f"""
                INSERT INTO {_qualified_table(self.draft_batches_table, self.schema)} (
                    "BATCH_ID", "SOURCE_IDS_JSON", "SOURCE_FILENAMES_JSON", "CREATED_AT", "STATUS"
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    batch.batch_id,
                    json.dumps(batch.source_ids, ensure_ascii=True),
                    json.dumps(batch.source_filenames, ensure_ascii=True),
                    batch.created_at,
                    batch.status,
                ],
            )
        return _clone_batch(batch)

    def replace_batch_candidates(
        self,
        batch_id: str,
        candidates: Sequence[Section232DraftRuleCandidate],
    ) -> None:
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            seen_candidate_ids = set()
            prepared: List[Section232DraftRuleCandidate] = []
            for candidate in candidates:
                if candidate.batch_id != batch_id:
                    candidate = replace(candidate, batch_id=batch_id)
                _validate_candidate_record(candidate)
                if candidate.candidate_id in seen_candidate_ids:
                    raise ValueError(f"duplicate candidate_id {candidate.candidate_id!r} in batch {batch_id}")
                seen_candidate_ids.add(candidate.candidate_id)
                prepared.append(_clone_candidate(candidate))

            self.connection.execute(
                f'DELETE FROM {_qualified_table(self.draft_rules_table, self.schema)} WHERE "BATCH_ID" = ?',
                [batch_id],
            )
            if prepared:
                self.connection.executemany(
                    f"""
                    INSERT INTO {_qualified_table(self.draft_rules_table, self.schema)} (
                        "BATCH_ID", "CANDIDATE_ID", "POSITION", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                        "EFFECTIVE_FROM", "EFFECTIVE_TO", "RATE_TEXT", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON",
                        "SOURCE_PAGES_JSON", "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND",
                        "CANDIDATE_QUALITY", "CANDIDATE_FLAGS_JSON", "PROCESSED_AT", "REVIEW_DECISION"
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            candidate.batch_id,
                            candidate.candidate_id,
                            position,
                            candidate.hts_code,
                            candidate.rule_type,
                            candidate.coverage_effect,
                            _db_optional_date(candidate.effective_from),
                            _db_optional_date(candidate.effective_to),
                            candidate.rate_text,
                            candidate.metal_scope,
                            json.dumps(candidate.source_document_ids, ensure_ascii=True),
                            json.dumps(candidate.source_pages, ensure_ascii=True),
                            candidate.source_excerpt,
                            candidate.interpreter_confidence,
                            1 if candidate.catalog_match_found else 0,
                            candidate.candidate_quality,
                            json.dumps(candidate.candidate_flags, ensure_ascii=True),
                            candidate.processed_at,
                            candidate.review_decision,
                        )
                        for position, candidate in enumerate(prepared)
                    ],
                )

    def review_candidate(self, batch_id: str, candidate_id: str, *, decision: ReviewDecision) -> Section232DraftRuleCandidate:
        with self._lock:
            self._require_pending_batch_locked(batch_id)
            _validate_review_decision(decision)
            current = self._require_candidate_locked(batch_id, candidate_id)
            updated = replace(current, review_decision=decision)
            _validate_candidate_record(updated)
            self.connection.execute(
                f"""
                UPDATE {_qualified_table(self.draft_rules_table, self.schema)}
                SET "REVIEW_DECISION" = ?
                WHERE "BATCH_ID" = ? AND "CANDIDATE_ID" = ?
                """,
                [decision, batch_id, candidate_id],
            )
            return _clone_candidate(updated)

    def review_candidates(
        self,
        batch_id: str,
        candidate_ids: Sequence[str],
        *,
        decision: ReviewDecision,
    ) -> List[Section232DraftRuleCandidate]:
        normalized_candidate_ids = _normalize_candidate_ids(candidate_ids)
        with self._lock:
            with self.connection.transaction():
                self._require_pending_batch_locked(batch_id)
                _validate_review_decision(decision)
                current_candidates = {
                    candidate.candidate_id: candidate
                    for candidate in self.list_draft_candidates(batch_id=batch_id)
                }
                updated_candidates: List[Section232DraftRuleCandidate] = []
                for candidate_id in normalized_candidate_ids:
                    try:
                        current = current_candidates[candidate_id]
                    except KeyError as exc:
                        raise KeyError(f"candidate {candidate_id} not found in batch {batch_id}") from exc
                    updated = replace(current, review_decision=decision)
                    _validate_candidate_record(updated)
                    updated_candidates.append(updated)

                placeholders = ", ".join("?" for _ in normalized_candidate_ids)
                self.connection.execute(
                    f"""
                    UPDATE {_qualified_table(self.draft_rules_table, self.schema)}
                    SET "REVIEW_DECISION" = ?
                    WHERE "BATCH_ID" = ? AND "CANDIDATE_ID" IN ({placeholders})
                    """,
                    [decision, batch_id, *normalized_candidate_ids],
                )
                return [_clone_candidate(candidate) for candidate in updated_candidates]

    def review_all_candidates(
        self,
        batch_id: str,
        *,
        decision: ReviewDecision,
        excluded_candidate_ids: Sequence[str] = (),
    ) -> int:
        normalized_excluded_candidate_ids = _normalize_optional_candidate_ids(excluded_candidate_ids)
        with self._lock:
            with self.connection.transaction():
                self._require_pending_batch_locked(batch_id)
                _validate_review_decision(decision)
                params: List[object] = [batch_id, decision]
                exclusion_clause = ""
                if normalized_excluded_candidate_ids:
                    placeholders = ", ".join("?" for _ in normalized_excluded_candidate_ids)
                    exclusion_clause = f' AND "CANDIDATE_ID" NOT IN ({placeholders})'
                    params.extend(normalized_excluded_candidate_ids)
                rows = self._fetch_rows(
                    f"""
                    SELECT COUNT(*) AS "COUNT"
                    FROM {_qualified_table(self.draft_rules_table, self.schema)}
                    WHERE "BATCH_ID" = ? AND "REVIEW_DECISION" <> ?{exclusion_clause}
                    """,
                    params,
                )
                updated_count = int(rows[0].get("count") or 0) if rows else 0
                if updated_count <= 0:
                    return 0

                update_params: List[object] = [decision, batch_id, decision]
                if normalized_excluded_candidate_ids:
                    update_params.extend(normalized_excluded_candidate_ids)
                self.connection.execute(
                    f"""
                    UPDATE {_qualified_table(self.draft_rules_table, self.schema)}
                    SET "REVIEW_DECISION" = ?
                    WHERE "BATCH_ID" = ? AND "REVIEW_DECISION" <> ?{exclusion_clause}
                    """,
                    update_params,
                )
                return updated_count

    def delete_draft_candidates_by_hts_code(self, batch_id: str, *, hts_code: str) -> int:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            raise ValueError(f"invalid hts_code: {hts_code!r}")
        with self._lock:
            with self.connection.transaction():
                self._require_pending_batch_locked(batch_id)
                rows = self._fetch_rows(
                    f"""
                    SELECT COUNT(*) AS "COUNT"
                    FROM {_qualified_table(self.draft_rules_table, self.schema)}
                    WHERE "BATCH_ID" = ? AND "HTS_CODE" = ?
                    """,
                    [batch_id, normalized_hts_code],
                )
                deleted_count = int(rows[0].get("count") or 0) if rows else 0
                if deleted_count <= 0:
                    return 0
                self.connection.execute(
                    f"""
                    DELETE FROM {_qualified_table(self.draft_rules_table, self.schema)}
                    WHERE "BATCH_ID" = ? AND "HTS_CODE" = ?
                    """,
                    [batch_id, normalized_hts_code],
                )
                return deleted_count

    def delete_pending_draft_batch(self, batch_id: str) -> Section232DeletedDraftBatch:
        with self._lock:
            with self.connection.transaction():
                batch = self._require_pending_batch_locked(batch_id)
                rows = self._fetch_rows(
                    f"""
                    SELECT COUNT(*) AS "COUNT"
                    FROM {_qualified_table(self.draft_rules_table, self.schema)}
                    WHERE "BATCH_ID" = ?
                    """,
                    [batch_id],
                )
                deleted_rule_count = int(rows[0].get("count") or 0) if rows else 0
                self.connection.execute(
                    f'DELETE FROM {_qualified_table(self.draft_rules_table, self.schema)} WHERE "BATCH_ID" = ?',
                    [batch_id],
                )
                self.connection.execute(
                    f'DELETE FROM {_qualified_table(self.draft_batches_table, self.schema)} WHERE "BATCH_ID" = ?',
                    [batch_id],
                )
                return Section232DeletedDraftBatch(
                    batch_id=batch.batch_id,
                    source_ids=list(batch.source_ids),
                    source_filenames=list(batch.source_filenames),
                    deleted_rule_count=deleted_rule_count,
                )

    def publish_snapshot(
        self,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate],
    ) -> Section232PublishedRuleset:
        publish_candidates = _resolve_publish_candidates(
            batch_id="snapshot",
            draft_candidates=list(accepted_rules_snapshot),
            accepted_rules_snapshot=accepted_rules_snapshot,
        )
        with self._lock:
            with self.connection.transaction():
                version = self._next_version_locked()
                published_at = self._next_timestamp_locked(table=self.rulesets_table, column="PUBLISHED_AT")
                self.connection.execute(
                    f"""
                    INSERT INTO {_qualified_table(self.rulesets_table, self.schema)} (
                        "VERSION", "PUBLISHED_AT", "PUBLISHED_BY"
                    ) VALUES (?, ?, ?)
                    """,
                    [version, published_at, published_by],
                )
                self.connection.executemany(
                    f"""
                    INSERT INTO {_qualified_table(self.ruleset_rules_table, self.schema)} (
                        "VERSION", "CANDIDATE_ID", "POSITION", "BATCH_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                        "EFFECTIVE_FROM", "EFFECTIVE_TO", "RATE_TEXT", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON",
                        "SOURCE_PAGES_JSON", "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND",
                        "CANDIDATE_QUALITY", "CANDIDATE_FLAGS_JSON", "PROCESSED_AT", "REVIEW_DECISION"
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            version,
                            candidate.candidate_id,
                            position,
                            candidate.batch_id,
                            candidate.hts_code,
                            candidate.rule_type,
                            candidate.coverage_effect,
                            _db_optional_date(candidate.effective_from),
                            _db_optional_date(candidate.effective_to),
                            candidate.rate_text,
                            candidate.metal_scope,
                            json.dumps(candidate.source_document_ids, ensure_ascii=True),
                            json.dumps(candidate.source_pages, ensure_ascii=True),
                            candidate.source_excerpt,
                            candidate.interpreter_confidence,
                            1 if candidate.catalog_match_found else 0,
                            candidate.candidate_quality,
                            json.dumps(candidate.candidate_flags, ensure_ascii=True),
                            candidate.processed_at,
                            candidate.review_decision,
                        )
                        for position, candidate in enumerate(publish_candidates)
                    ],
                )
                snapshot = Section232PublishedRuleset(
                    version=version,
                    published_at=published_at,
                    published_by=published_by,
                    accepted_rules=[_clone_candidate(candidate) for candidate in publish_candidates],
                )
                return _clone_published_ruleset(snapshot)

    def publish_batch(
        self,
        batch_id: str,
        *,
        published_by: str,
        accepted_rules_snapshot: Sequence[Section232DraftRuleCandidate] | None = None,
    ) -> Section232PublishedRuleset:
        with self._lock:
            with self.connection.transaction():
                batch = self._require_pending_batch_locked(batch_id)
                draft_candidates = self.list_draft_candidates(batch_id=batch_id)
                publish_candidates = _resolve_publish_candidates(
                    batch_id=batch_id,
                    draft_candidates=draft_candidates,
                    accepted_rules_snapshot=accepted_rules_snapshot,
                )
                snapshot = self.publish_snapshot(
                    published_by=published_by,
                    accepted_rules_snapshot=publish_candidates,
                )
                self.connection.execute(
                    f"""
                    UPDATE {_qualified_table(self.draft_batches_table, self.schema)}
                    SET "STATUS" = ?
                    WHERE "BATCH_ID" = ?
                    """,
                    ["published", batch_id],
                )
                return snapshot

    def list_active_rules(self) -> List[Section232DraftRuleCandidate]:
        active_version = self.get_active_ruleset_version()
        if active_version is None:
            return []
        rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "RATE_TEXT", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "CANDIDATE_QUALITY",
                "CANDIDATE_FLAGS_JSON", "PROCESSED_AT", "REVIEW_DECISION"
            FROM {_qualified_table(self.ruleset_rules_table, self.schema)}
            WHERE "VERSION" = ?
            ORDER BY "POSITION" ASC, "CANDIDATE_ID" ASC
            """,
            [active_version],
        )
        return project_current_rules_by_hts_code([self._row_to_candidate(row) for row in rows])

    def list_draft_candidates(self, *, batch_id: str) -> List[Section232DraftRuleCandidate]:
        self._require_batch_locked(batch_id)
        rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "RATE_TEXT", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "CANDIDATE_QUALITY",
                "CANDIDATE_FLAGS_JSON", "PROCESSED_AT", "REVIEW_DECISION"
            FROM {_qualified_table(self.draft_rules_table, self.schema)}
            WHERE "BATCH_ID" = ?
            ORDER BY "POSITION" ASC, "CANDIDATE_ID" ASC
            """,
            [batch_id],
        )
        return [self._row_to_candidate(row) for row in rows]

    def get_draft_batch_stats(self, *, batch_id: str) -> Section232DraftBatchStats:
        self._require_batch_locked(batch_id)
        rows = self._fetch_rows(
            f"""
            SELECT
                COUNT(*) AS "TOTAL",
                SUM(CASE WHEN "REVIEW_DECISION" = 'pending' THEN 1 ELSE 0 END) AS "PENDING_COUNT",
                SUM(CASE WHEN "REVIEW_DECISION" = 'accepted' THEN 1 ELSE 0 END) AS "ACCEPTED_COUNT",
                SUM(CASE WHEN "REVIEW_DECISION" = 'rejected' THEN 1 ELSE 0 END) AS "REJECTED_COUNT",
                SUM(
                    CASE
                        WHEN "CATALOG_MATCH_FOUND" = 0
                            OR "CANDIDATE_QUALITY" = 'suspect'
                            OR (
                                "CANDIDATE_FLAGS_JSON" IS NOT NULL
                                AND LENGTH("CANDIDATE_FLAGS_JSON") > 2
                            )
                        THEN 1
                        ELSE 0
                    END
                ) AS "WARNING_COUNT"
            FROM {_qualified_table(self.draft_rules_table, self.schema)}
            WHERE "BATCH_ID" = ?
            """,
            [batch_id],
        )
        payload = rows[0] if rows else {}
        return Section232DraftBatchStats(
            total=int(payload.get("total") or 0),
            pending_count=int(payload.get("pending_count") or 0),
            accepted_count=int(payload.get("accepted_count") or 0),
            rejected_count=int(payload.get("rejected_count") or 0),
            warning_count=int(payload.get("warning_count") or 0),
        )

    def list_draft_candidate_page(
        self,
        *,
        batch_id: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        self._require_batch_locked(batch_id)
        total_rows = self._fetch_rows(
            f"""
            SELECT COUNT(*) AS "COUNT"
            FROM {_qualified_table(self.draft_rules_table, self.schema)}
            WHERE "BATCH_ID" = ?
            """,
            [batch_id],
        )
        total = int(total_rows[0].get("count") or 0) if total_rows else 0
        rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "REVIEW_DECISION"
            FROM {_qualified_table(self.draft_rules_table, self.schema)}
            WHERE "BATCH_ID" = ?
            ORDER BY
                CASE WHEN "CATALOG_MATCH_FOUND" = 1 THEN 1 ELSE 0 END ASC,
                REPLACE("HTS_CODE", '.', '') ASC,
                "CANDIDATE_ID" ASC
            LIMIT ? OFFSET ?
            """,
            [batch_id, limit, offset],
        )
        return Section232CandidatePage(
            total=total,
            limit=limit,
            offset=offset,
            candidates=[self._row_to_candidate(row) for row in rows],
        )

    def list_draft_batches(self, *, status: Optional[BatchStatus] = None) -> List[Section232DraftBatch]:
        sql = f"""
            SELECT "BATCH_ID", "SOURCE_IDS_JSON", "SOURCE_FILENAMES_JSON", "CREATED_AT", "STATUS"
            FROM {_qualified_table(self.draft_batches_table, self.schema)}
        """
        params: List[object] = []
        if status is not None:
            sql = f'{sql} WHERE "STATUS" = ?'
            params.append(status)
        sql = f'{sql} ORDER BY "CREATED_AT" DESC'
        return [self._row_to_batch(row) for row in self._fetch_rows(sql, params)]

    def get_active_ruleset_version(self) -> str | None:
        rows = self._fetch_rows(
            f'SELECT "VERSION" FROM {_qualified_table(self.rulesets_table, self.schema)}'
        )
        if not rows:
            return None
        return max((str(row["version"] or "") for row in rows), key=_version_sort_key)

    def count_pending_batches(self) -> int:
        rows = self._fetch_rows(
            f"""
            SELECT COUNT(*) AS "COUNT"
            FROM {_qualified_table(self.draft_batches_table, self.schema)}
            WHERE "STATUS" = ?
            """,
            ["pending_review"],
        )
        return int(rows[0].get("count") or 0) if rows else 0

    def get_last_published_at(self) -> str | None:
        active_version = self.get_active_ruleset_version()
        if active_version is None:
            return None
        rows = self._fetch_rows(
            f"""
            SELECT "PUBLISHED_AT"
            FROM {_qualified_table(self.rulesets_table, self.schema)}
            WHERE "VERSION" = ?
            """,
            [active_version],
        )
        if not rows:
            return None
        return None if rows[0].get("published_at") is None else str(rows[0]["published_at"])

    def get_published_ruleset(self, version: str) -> Optional[Section232PublishedRuleset]:
        rows = self._fetch_rows(
            f"""
            SELECT "VERSION", "PUBLISHED_AT", "PUBLISHED_BY"
            FROM {_qualified_table(self.rulesets_table, self.schema)}
            WHERE "VERSION" = ?
            """,
            [version],
        )
        if not rows:
            return None
        rule_rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "REVIEW_DECISION"
            FROM {_qualified_table(self.ruleset_rules_table, self.schema)}
            WHERE "VERSION" = ?
            ORDER BY "POSITION" ASC, "CANDIDATE_ID" ASC
            """,
            [version],
        )
        snapshot = Section232PublishedRuleset(
            version=str(rows[0]["version"]),
            published_at=str(rows[0]["published_at"]),
            published_by=str(rows[0]["published_by"]),
            accepted_rules=[self._row_to_candidate(row) for row in rule_rows],
        )
        return _clone_published_ruleset(snapshot)

    def get_published_ruleset_info(self, version: str) -> Optional[Section232PublishedRulesetInfo]:
        rows = self._fetch_rows(
            f"""
            SELECT "VERSION", "PUBLISHED_AT", "PUBLISHED_BY"
            FROM {_qualified_table(self.rulesets_table, self.schema)}
            WHERE "VERSION" = ?
            """,
            [version],
        )
        if not rows:
            return None
        return Section232PublishedRulesetInfo(
            version=str(rows[0]["version"]),
            published_at=str(rows[0]["published_at"]),
            published_by=str(rows[0]["published_by"]),
        )

    def list_published_ruleset_infos(self) -> List[Section232PublishedRulesetInfo]:
        rows = self._fetch_rows(
            f"""
            SELECT "VERSION", "PUBLISHED_AT", "PUBLISHED_BY"
            FROM {_qualified_table(self.rulesets_table, self.schema)}
            """
        )
        infos = [
            Section232PublishedRulesetInfo(
                version=str(row["version"]),
                published_at=str(row["published_at"]),
                published_by=str(row["published_by"]),
            )
            for row in rows
        ]
        infos.sort(key=lambda item: _version_sort_key(item.version))
        return infos

    def create_delete_override(self, *, hts_code: str, deleted_by: str) -> Section232CodeDeleteOverride:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            raise ValueError(f"invalid hts_code: {hts_code!r}")
        with self._lock:
            override = Section232CodeDeleteOverride(
                hts_code=normalized_hts_code,
                deleted_at=self._next_timestamp_locked(table=self.delete_overrides_table, column="DELETED_AT"),
                deleted_by=str(deleted_by or "").strip(),
            )
            with self.connection.transaction():
                self.connection.execute(
                    f'''
                    DELETE FROM {_qualified_table(self.delete_overrides_table, self.schema)}
                    WHERE "HTS_CODE" = ?
                    ''',
                    [normalized_hts_code],
                )
                self.connection.execute(
                    f'''
                    INSERT INTO {_qualified_table(self.delete_overrides_table, self.schema)} (
                        "HTS_CODE", "DELETED_AT", "DELETED_BY"
                    ) VALUES (?, ?, ?)
                    ''',
                    [override.hts_code, override.deleted_at, override.deleted_by],
                )
        return _clone_delete_override(override)

    def get_delete_override(self, hts_code: str) -> Optional[Section232CodeDeleteOverride]:
        normalized_hts_code = canonicalize_hts_code(hts_code)
        if not normalized_hts_code:
            return None
        rows = self._fetch_rows(
            f'''
            SELECT "HTS_CODE", "DELETED_AT", "DELETED_BY"
            FROM {_qualified_table(self.delete_overrides_table, self.schema)}
            WHERE "HTS_CODE" = ?
            ''',
            [normalized_hts_code],
        )
        if not rows:
            return None
        return Section232CodeDeleteOverride(
            hts_code=str(rows[0]["hts_code"]),
            deleted_at=str(rows[0]["deleted_at"]),
            deleted_by=str(rows[0]["deleted_by"]),
        )

    def list_delete_overrides(self) -> List[Section232CodeDeleteOverride]:
        rows = self._fetch_rows(
            f'''
            SELECT "HTS_CODE", "DELETED_AT", "DELETED_BY"
            FROM {_qualified_table(self.delete_overrides_table, self.schema)}
            ORDER BY "DELETED_AT" DESC, "HTS_CODE" ASC
            '''
        )
        return [
            Section232CodeDeleteOverride(
                hts_code=str(row["hts_code"]),
                deleted_at=str(row["deleted_at"]),
                deleted_by=str(row["deleted_by"]),
            )
            for row in rows
        ]

    def list_published_rule_page(
        self,
        *,
        version: str,
        limit: int,
        offset: int,
    ) -> Section232CandidatePage:
        info = self.get_published_ruleset_info(version)
        if info is None:
            raise KeyError(f"published ruleset {version} not found")
        total_rows = self._fetch_rows(
            f"""
            SELECT COUNT(*) AS "COUNT"
            FROM {_qualified_table(self.ruleset_rules_table, self.schema)}
            WHERE "VERSION" = ?
            """,
            [version],
        )
        total = int(total_rows[0].get("count") or 0) if total_rows else 0
        rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "REVIEW_DECISION"
            FROM {_qualified_table(self.ruleset_rules_table, self.schema)}
            WHERE "VERSION" = ?
            ORDER BY
                CASE WHEN "CATALOG_MATCH_FOUND" = 1 THEN 1 ELSE 0 END ASC,
                REPLACE("HTS_CODE", '.', '') ASC,
                "CANDIDATE_ID" ASC
            LIMIT ? OFFSET ?
            """,
            [version, limit, offset],
        )
        return Section232CandidatePage(
            total=total,
            limit=limit,
            offset=offset,
            candidates=[self._row_to_candidate(row) for row in rows],
        )

    def list_active_eligible_codes(self, *, on_date: date) -> List[str]:
        return project_active_eligible_codes(self.list_active_rules(), on_date=on_date)

    def clear_all(self) -> Dict[str, int]:
        with self._lock:
            with self.connection.transaction():
                cleared_draft_batch_count = self._count_rows(self.draft_batches_table)
                cleared_draft_rule_count = self._count_rows(self.draft_rules_table)
                cleared_published_ruleset_count = self._count_rows(self.rulesets_table)
                cleared_published_rule_count = self._count_rows(self.ruleset_rules_table)
                cleared_delete_override_count = self._count_rows(self.delete_overrides_table)
                self.connection.execute(f'DELETE FROM {_qualified_table(self.draft_rules_table, self.schema)}')
                self.connection.execute(f'DELETE FROM {_qualified_table(self.draft_batches_table, self.schema)}')
                self.connection.execute(f'DELETE FROM {_qualified_table(self.ruleset_rules_table, self.schema)}')
                self.connection.execute(f'DELETE FROM {_qualified_table(self.rulesets_table, self.schema)}')
                self.connection.execute(f'DELETE FROM {_qualified_table(self.delete_overrides_table, self.schema)}')
        return {
            "cleared_draft_batch_count": cleared_draft_batch_count,
            "cleared_draft_rule_count": cleared_draft_rule_count,
            "cleared_published_ruleset_count": cleared_published_ruleset_count,
            "cleared_published_rule_count": cleared_published_rule_count,
            "cleared_delete_override_count": cleared_delete_override_count,
        }

    def _next_version_locked(self) -> str:
        rows = self._fetch_rows(
            f'SELECT "VERSION" FROM {_qualified_table(self.rulesets_table, self.schema)}'
        )
        next_index = 1
        if rows:
            next_index = max(1, max(_version_sort_key(str(row["version"] or ""))[0] for row in rows) + 1)
        return f"section232-v{next_index:04d}"

    def _require_batch_locked(self, batch_id: str) -> Section232DraftBatch:
        rows = self._fetch_rows(
            f"""
            SELECT "BATCH_ID", "SOURCE_IDS_JSON", "SOURCE_FILENAMES_JSON", "CREATED_AT", "STATUS"
            FROM {_qualified_table(self.draft_batches_table, self.schema)}
            WHERE "BATCH_ID" = ?
            """,
            [batch_id],
        )
        if not rows:
            raise KeyError(f"draft batch {batch_id} not found")
        batch = self._row_to_batch(rows[0])
        _validate_batch_record(batch)
        return batch

    def get_draft_batch(self, batch_id: str) -> Section232DraftBatch:
        with self._lock:
            return _clone_batch(self._require_batch_locked(batch_id))

    def _require_pending_batch_locked(self, batch_id: str) -> Section232DraftBatch:
        batch = self._require_batch_locked(batch_id)
        if batch.status != "pending_review":
            raise ValueError(f"draft batch {batch_id} is already published")
        return batch

    def _require_candidate_locked(self, batch_id: str, candidate_id: str) -> Section232DraftRuleCandidate:
        rows = self._fetch_rows(
            f"""
            SELECT
                "BATCH_ID", "CANDIDATE_ID", "HTS_CODE", "RULE_TYPE", "COVERAGE_EFFECT",
                "EFFECTIVE_FROM", "EFFECTIVE_TO", "RATE_TEXT", "METAL_SCOPE", "SOURCE_DOCUMENT_IDS_JSON", "SOURCE_PAGES_JSON",
                "SOURCE_EXCERPT", "INTERPRETER_CONFIDENCE", "CATALOG_MATCH_FOUND", "CANDIDATE_QUALITY",
                "CANDIDATE_FLAGS_JSON", "PROCESSED_AT", "REVIEW_DECISION"
            FROM {_qualified_table(self.draft_rules_table, self.schema)}
            WHERE "BATCH_ID" = ? AND "CANDIDATE_ID" = ?
            """,
            [batch_id, candidate_id],
        )
        if not rows:
            raise KeyError(f"candidate {candidate_id} not found in batch {batch_id}")
        candidate = self._row_to_candidate(rows[0])
        _validate_candidate_record(candidate)
        return candidate

    def _count_rows(self, table: str) -> int:
        rows = self._fetch_rows(
            f'SELECT COUNT(*) AS "COUNT" FROM {_qualified_table(table, self.schema)}'
        )
        return int(rows[0].get("count") or 0) if rows else 0

    @staticmethod
    def _row_to_batch(row: Dict[str, object]) -> Section232DraftBatch:
        batch = Section232DraftBatch(
            batch_id=str(row["batch_id"]),
            source_ids=list(_safe_json_loads(row.get("source_ids_json"), default=[])),
            source_filenames=list(_safe_json_loads(row.get("source_filenames_json"), default=[])),
            created_at=str(row.get("created_at") or ""),
            status=str(row.get("status") or "pending_review"),  # type: ignore[arg-type]
        )
        _validate_batch_record(batch)
        return batch

    @staticmethod
    def _row_to_candidate(row: Dict[str, object]) -> Section232DraftRuleCandidate:
        candidate = Section232DraftRuleCandidate(
            candidate_id=str(row["candidate_id"]),
            batch_id=str(row["batch_id"]),
            hts_code=str(row["hts_code"]),
            rule_type=str(row["rule_type"]),  # type: ignore[arg-type]
            coverage_effect=str(row["coverage_effect"]),  # type: ignore[arg-type]
            effective_from=None if row.get("effective_from") in (None, "") else str(row["effective_from"]),
            effective_to=None if row.get("effective_to") in (None, "") else str(row["effective_to"]),
            rate_text=None if row.get("rate_text") in (None, "") else str(row.get("rate_text")),
            metal_scope=_coerce_legacy_scope(row.get("metal_scope")),
            source_document_ids=list(_safe_json_loads(row.get("source_document_ids_json"), default=[])),
            source_pages=[int(value) for value in _safe_json_loads(row.get("source_pages_json"), default=[])],
            source_excerpt=str(row.get("source_excerpt") or ""),
            interpreter_confidence=float(row.get("interpreter_confidence") or 0.0),
            catalog_match_found=bool(int(row.get("catalog_match_found") or 0)),
            candidate_quality=str(row.get("candidate_quality") or "normal"),  # type: ignore[arg-type]
            candidate_flags=list(_safe_json_loads(row.get("candidate_flags_json"), default=[])),
            processed_at=None if row.get("processed_at") in (None, "") else str(row.get("processed_at")),
            review_decision=str(row.get("review_decision") or "pending"),  # type: ignore[arg-type]
        )
        _validate_candidate_record(candidate)
        return candidate
