"""Unit tests for EcnRepository: stable list reads, evidence isolation, and mapper purity.

Tests are grouped into three suites following the Task 7 TDD brief:

1. List/filter/sort/pagination — SQL construction, allowlisted sort, bound parameters,
   derived analysis state, stable tie-breakers.
2. Detail/evidence/queued-job reads — exact occurrence retrieval, provenance mapping,
   Python-side cross-ECN evidence isolation.
3. Pure response mapper assertions — codec mapper purity and field contracts.

No live HANA connection is required; all assertions run against recording fakes.

Example run (list subset)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_repository.py \\
        -k 'list or filter or sort or pagination'

Example run (detail/evidence subset)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_repository.py \\
        -k 'detail or evidence or queued'

Example run (full suite)::

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_repository.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.ecn.api_models import AnalysisFilterState, AnalysisJobView, ImpactCounts, EcnListQuery
from app.ecn.models import (
    AnalysisSectionKey,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    ArtifactKind,
    JobState,
)


# ---------------------------------------------------------------------------
# Fixed UUIDs shared across suites
# ---------------------------------------------------------------------------

ECN_ROW_ID = "00000000-0000-4000-8000-000000000004"
UPLOAD_ID = "00000000-0000-4000-8000-000000000002"
JOB_ID_1 = "00000000-0000-4000-8000-000000000030"
JOB_ID_2 = "00000000-0000-4000-8000-000000000031"
ARTIFACT_ID = "00000000-0000-4000-8000-000000000001"
BOM_ARTIFACT_ID = "00000000-0000-4000-8000-000000000006"
REQUEST_ID = "00000000-0000-4000-8000-000000000007"

# Evidence item UUIDs — "own" lives in the bundle; "other" does NOT.
EV_OWN_ID = "00000000-0000-4000-8000-000000000098"
EV_OTHER_ID = "00000000-0000-4000-8000-000000000099"

# Naive UTC datetime — mirrors what the HANA TIMESTAMP driver column returns.
_NAIVE_TS = datetime(2026, 9, 14)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_engine():
    """Return a fresh RecordingEngine for each test.

    Returns:
        A ``RecordingEngine`` with zero counters and an empty unified queue.
    """
    from tests.fixtures.persistence import RecordingEngine

    return RecordingEngine()


# ---------------------------------------------------------------------------
# Local helper: minimal ECN list-query result row
# ---------------------------------------------------------------------------


def ecn_list_row(
    *,
    ordinal: int = 0,
    ecn_number: str | None = "ECN-001",
    analysis_state: str = "QUEUED",
    completeness: str | None = None,
    warning_count: int = 0,
    resolution_date: str | None = "2016-06-21T02:00:00+02:00",
    **kwargs: object,
) -> dict[str, object]:
    """Build a minimal column-keyed dict matching the ECN list SELECT result.

    All six counts default to zero.  ``UPLOADED_AT`` is a naive UTC datetime
    as the HANA driver returns it; ``ecn_list_item_from_row`` restores UTC.

    Args:
        ordinal: Zero-based source ordinal.
        ecn_number: ECN number string, or ``None`` to exercise the missing-ID path.
        analysis_state: Value for the derived ``DERIVED_ANALYSIS_STATE`` column.
        completeness: Value for ``COMPLETENESS_STATE`` (nullable).
        warning_count: Value for ``WARNING_COUNT``.
        **kwargs: Column-name overrides.

    Returns:
        A dict keyed by the exact column aliases produced by the list query.
    """
    row_uuid = f"00000000-0000-4000-8000-{0x40 + ordinal:012x}"
    defaults: dict[str, object] = {
        "ECN_ROW_ID": row_uuid,
        "ECN_NUMBER": ecn_number,
        "TITLE": "Test ECN Title",
        "DESCRIPTION_SUMMARY": "A summary",
        "SOURCE_ORDINAL": ordinal,
        "AFFECTED_ITEM_COUNT": 0,
        "RAW_ECN_MATCH_COUNT": 0,
        "EBOM_MATCH_COUNT": 0,
        "EBOM_CONTEXT_COUNT": 0,
        "MBOM_MATCH_COUNT": 0,
        "WARNING_COUNT": warning_count,
        "RESOLUTION_DATE": resolution_date,
        "UPLOADED_AT": _NAIVE_TS,
        "SOURCE_FILE_NAME": "ecn.json",
        "DERIVED_ANALYSIS_STATE": analysis_state,
        "COMPLETENESS_STATE": completeness,
        "ANALYSIS_CONTENT_JSON": None,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Local helper: evidence bundle as canonical JSON (contains only EV_OWN_ID)
# ---------------------------------------------------------------------------


def evidence_bundle_json(evidence_id: str) -> str:
    """Build canonical JSON for an EvidenceBundle with one item in executive_summary.

    The bundle contains exactly one evidence item under ``executive_summary``,
    identified by ``evidence_id``.  All other sections are empty.  This lets
    tests verify that searching for a different ID (EV_OTHER_ID) returns None,
    demonstrating Python-side cross-ECN isolation.

    Args:
        evidence_id: Canonical UUID for the single evidence item in the bundle.

    Returns:
        Canonical JSON string for the bundle, ready for the ``EVIDENCE_JSON`` column.
    """
    from app.persistence.codec import canonical_json

    item_dict = {
        "evidence_id": evidence_id,
        "section_key": AnalysisSectionKey.EXECUTIVE_SUMMARY.value,
        "kind": EvidenceKind.SOURCE_FACT.value,
        "tier": None,
        "summary": "Test evidence",
        "sources": [],
        "values": {},
        "related_ids": [],
        "quality_qualifiers": [],
    }
    bundle_dict = {
        "ecn_row_id": ECN_ROW_ID,
        "context": {
            "upload_id": UPLOAD_ID,
            "ecn_artifact_id": ARTIFACT_ID,
            "bom_artifact_id": BOM_ARTIFACT_ID,
            "rule_artifact_id": None,
            "parser_version": "ecn-source-v1",
            "deterministic_logic_version": "ecn-evidence-v1",
            "rule_evaluator_version": "ecn-rules-v1",
            "analysis_as_of_date": None,
        },
        "sections": {
            "executive_summary": [item_dict],
            "change_intent": [],
            "affected_materials": [],
            "ebom_mbom_impact": [],
            "configuration_scope": [],
            "uncertainties_and_missing_evidence": [],
        },
        "evidence_hash": "0" * 64,
    }
    return canonical_json(bundle_dict)


# ---------------------------------------------------------------------------
# Local helper: minimal EcnDetailRecord for mapper tests
# ---------------------------------------------------------------------------


def ecn_detail_record(
    *,
    source_ordinal: int = 0,
    ecn_number: str | None = "ECN-001",
    resolution_date: str | None = None,
    source_status: str | None = None,
    requestor_name: str | None = None,
    affected_products: tuple[str, ...] = (),
    include_rule_artifact: bool = False,
    include_analysis: bool = False,
) -> object:
    """Build a minimal ``EcnDetailRecord`` for pure mapper tests.

    All sub-records use deterministic UUIDs and UTC timestamps.  The
    optional rule artifact and analysis are included only when their
    respective flags are ``True``.

    Args:
        source_ordinal: Zero-based position of the ECN in its source file.
        ecn_number: Source-system ECN identifier; ``None`` exercises missing-ID path.
        include_rule_artifact: When ``True``, adds a ``SourceArtifact`` for the rule.
        include_analysis: When ``True``, adds a ``StoredAnalysisView``.

    Returns:
        A frozen ``EcnDetailRecord`` instance.
    """
    from app.ecn.analysis_models import AnalysisCompleteness
    from app.ecn.models import ArtifactKind, ParseState, SourceArtifact
    from app.ecn.api_models import AnalysisJobView, StoredAnalysisView
    from app.persistence.records import EcnDetailRecord, StoredEcnRecord, UploadRecord
    from tests.fixtures.builders import make_evidence_bundle, make_source_artifact

    created_at = datetime(2026, 9, 14, tzinfo=timezone.utc)

    stored_ecn = StoredEcnRecord(
        ecn_row_id=ECN_ROW_ID,
        upload_id=UPLOAD_ID,
        source_ordinal=source_ordinal,
        ecn_number=ecn_number,
        title="Test ECN Title",
        description_summary="A summary",
        description_text="Full description text",
        counts=ImpactCounts(
            affected_item_count=5,
            raw_ecn_match_count=3,
            ebom_match_count=2,
            ebom_context_count=1,
            mbom_match_count=1,
            warning_count=0,
        ),
        source_json={"id": ecn_number},
        change_candidates=(),
        evidence_hash="0" * 64,
        parser_version="ecn-source-v1",
        deterministic_logic_version="ecn-evidence-v1",
        rule_evaluator_version="ecn-rules-v1",
        created_at=created_at,
        source_status=source_status,
        requestor_name=requestor_name,
        affected_products=affected_products,
        resolution_date=resolution_date,
    )

    upload = UploadRecord(
        upload_id=UPLOAD_ID,
        request_id=REQUEST_ID,
        ecn_artifact_id=ARTIFACT_ID,
        bom_artifact_id=BOM_ARTIFACT_ID,
        rule_artifact_id=None,
        bom_reused=False,
        imported_ecn_count=1,
        publication_order=1,
        created_at=created_at,
    )

    ecn_artifact = make_source_artifact(
        artifact_id=ARTIFACT_ID,
        artifact_kind=ArtifactKind.ECN_JSON,
        file_name="ecn.json",
    )
    bom_artifact = make_source_artifact(
        artifact_id=BOM_ARTIFACT_ID,
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        file_name="bom.xlsx",
    )

    rule_artifact = None
    if include_rule_artifact:
        rule_artifact = make_source_artifact(
            artifact_id="00000000-0000-4000-8000-000000000009",
            artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
            file_name="rules.xlsx",
        )

    job = AnalysisJobView(
        job_id=UUID(JOB_ID_1),
        state=JobState.QUEUED,
        attempt_count=0,
        error_code=None,
        error_message=None,
    )

    analysis = None
    if include_analysis:
        from app.ecn.analysis_models import AnalysisSections, AnalysisSection, SectionStatus

        section = AnalysisSection(
            status=SectionStatus.SUFFICIENT,
            assessment_markdown="Assessment.",
            evidence_ids=["EXECUTIVE_SUMMARY-1"],
            missing_evidence=[],
        )
        sections = AnalysisSections(
            executive_summary=section,
            change_intent=section,
            affected_materials=section,
            ebom_mbom_impact=section,
            configuration_scope=section,
            uncertainties_and_missing_evidence=section,
        )
        analysis = StoredAnalysisView(
            analysis_id=UUID("00000000-0000-4000-8000-000000000050"),
            completeness=AnalysisCompleteness.COMPLETED,
            sections=sections,
            created_at=created_at,
            evidence_hash="0" * 64,
            prompt_version="v1",
            output_schema_version="v1",
            model_name="gpt-4o",
            model_deployment=None,
        )

    return EcnDetailRecord(
        ecn=stored_ecn,
        upload=upload,
        ecn_artifact=ecn_artifact,
        bom_artifact=bom_artifact,
        rule_artifact=rule_artifact,
        job=job,
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Local helper: minimal EvidenceItem
# ---------------------------------------------------------------------------


def evidence_item(
    *,
    section_key: AnalysisSectionKey = AnalysisSectionKey.CHANGE_INTENT,
    evidence_id: str = EV_OWN_ID,
) -> EvidenceItem:
    """Build a minimal ``EvidenceItem`` for pure mapper tests.

    Args:
        section_key: The analysis section this item belongs to.
        evidence_id: Canonical UUID for this evidence item.

    Returns:
        A frozen ``EvidenceItem`` instance.
    """
    return EvidenceItem(
        evidence_id=evidence_id,
        section_key=section_key,
        kind=EvidenceKind.SOURCE_FACT,
        tier=None,
        summary="A test evidence summary",
        sources=(),
        values={"key": "value"},
        related_ids=(),
        quality_qualifiers=("Q1",),
    )


# ===========================================================================
# Suite 1: List filter / sort / pagination
# ===========================================================================


def test_default_list_order_and_pagination_are_stable(recording_engine) -> None:
    """Default sort produces stable tie-breaker ORDER BY and correct pagination params."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(2)
    recording_engine.queue_mappings([ecn_list_row(ordinal=0), ecn_list_row(ordinal=1)])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert result.total_items == 2
    assert result.total_pages == 1
    assert len(result.items) == 2
    statement, params = recording_engine.executions[-1]
    assert 'U."PUBLICATION_ORDER" DESC' in statement
    assert 'E."SOURCE_ORDINAL" ASC' in statement
    assert 'E."ECN_ROW_ID" ASC' in statement
    assert params["limit"] == 25
    assert params["offset"] == 0


def test_completed_with_gaps_filter_uses_job_and_analysis(recording_engine) -> None:
    """COMPLETED_WITH_GAPS filter binds both job state and analysis completeness."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    query = EcnListQuery(analysis_state=AnalysisFilterState.COMPLETED_WITH_GAPS)
    EcnRepository(recording_engine).list_ecns(query)
    sql = "\n".join(statement for statement, _ in recording_engine.executions)
    assert 'J."STATE" = :completed_job_state' in sql
    assert 'A."COMPLETENESS_STATE" = :gap_state' in sql


def test_completed_filter_excludes_gaps(recording_engine) -> None:
    """COMPLETED filter binds job state and requires completeness is not COMPLETED_WITH_GAPS."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    query = EcnListQuery(analysis_state=AnalysisFilterState.COMPLETED)
    EcnRepository(recording_engine).list_ecns(query)
    sql = "\n".join(statement for statement, _ in recording_engine.executions)
    assert 'J."STATE" = :completed_job_state' in sql
    # Must also filter on completeness to exclude COMPLETED_WITH_GAPS rows
    assert "COMPLETENESS_STATE" in sql


def test_queued_state_filter_uses_job_state_only(recording_engine) -> None:
    """Non-gap states map directly to a single job STATE bound parameter."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    query = EcnListQuery(analysis_state=AnalysisFilterState.QUEUED)
    EcnRepository(recording_engine).list_ecns(query)
    sql = "\n".join(statement for statement, _ in recording_engine.executions)
    assert 'J."STATE" = :analysis_state_value' in sql


@pytest.mark.parametrize("group, expected", [
    ("SUMMARIES_AVAILABLE", ["complete", "gaps"]),
    ("IN_PROGRESS", ["queued", "running"]),
])
def test_group_filters_select_both_member_states_and_preserve_other_filters(group, expected) -> None:
    """Execute group predicates against real rows, excluding blocked and failed jobs."""
    from app.persistence.ecn_repository import _build_filters

    rows = [
        ("complete", "COMPLETED", "COMPLETED"),
        ("gaps", "COMPLETED", "COMPLETED_WITH_GAPS"),
        ("queued", "QUEUED", None), ("running", "RUNNING", None),
        ("blocked", "BLOCKED", None), ("failed", "FAILED", None), ("ready", "READY", None),
        ("no-summary", "COMPLETED", None),
    ]
    with sqlite3.connect(":memory:") as connection:
        connection.execute('CREATE TABLE rows (ID TEXT, STATE TEXT, COMPLETENESS_STATE TEXT, '
                           'ECN_NUMBER TEXT, TITLE TEXT, DESCRIPTION_SUMMARY TEXT, WARNING_COUNT INTEGER)')
        connection.executemany('INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?)', [
            (*row, "ECN-1", "shaft", "change", 1) for row in rows
        ])
        connection.executemany('INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?)', [
            ("other-search", rows[0][1], rows[0][2], "ECN-2", "unrelated", "change", 1),
            ("no-warnings", rows[3][1], rows[3][2], "ECN-3", "shaft", "change", 0),
        ])
        where, params = _build_filters(EcnListQuery(analysis_state=group, query="shaft", has_warnings=True))
        matches = connection.execute(
            f'SELECT E.ID FROM rows E JOIN rows J ON J.ID = E.ID JOIN rows A ON A.ID = E.ID {where}', params,
        ).fetchall()
    assert [row[0] for row in matches] == expected


@pytest.mark.parametrize("group", ["SUMMARIES_AVAILABLE", "IN_PROGRESS"])
def test_group_count_list_and_bulk_membership_share_filters(recording_engine, group) -> None:
    """Pagination counts and filtered bulk actions use identical group predicates."""
    from app.persistence.ecn_repository import EcnRepository, _build_filters

    query = EcnListQuery(analysis_state=group, query="shaft", has_warnings=True)
    where, params = _build_filters(query)
    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    recording_engine.queue_mappings([])
    repository = EcnRepository(recording_engine)
    repository.list_ecns(query)
    repository.list_matching_ecn_ids(query)
    for sql, bound in recording_engine.executions:
        assert where in sql
        assert all(bound[key] == value for key, value in params.items())


def test_sort_uploaded_at_desc_is_default(recording_engine) -> None:
    """Default sort uses U.CREATED_AT DESC as primary expression."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="uploaded_at", order="desc"))
    statement, _ = recording_engine.executions[-1]
    assert 'U."CREATED_AT" DESC' in statement


def test_sort_uploaded_at_asc(recording_engine) -> None:
    """Ascending sort on uploaded_at uses U.CREATED_AT ASC."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="uploaded_at", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert 'U."CREATED_AT" ASC' in statement


def test_sort_ecn_number_uses_nulls_last(recording_engine) -> None:
    """Nullable ECN number sort includes NULLS LAST clause."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="ecn_number", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."ECN_NUMBER" ASC NULLS LAST' in statement


def test_sort_description_uses_nulls_last(recording_engine) -> None:
    """Nullable description sort includes NULLS LAST clause."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="description", order="desc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."DESCRIPTION_SUMMARY" DESC NULLS LAST' in statement


def test_sort_affected_item_count_asc(recording_engine) -> None:
    """Numeric sort on affected_item_count uses E.AFFECTED_ITEM_COUNT."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="affected_item_count", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."AFFECTED_ITEM_COUNT" ASC' in statement
    # No NULLS LAST for non-nullable numeric columns
    assert "NULLS LAST" not in statement


def test_sort_resolution_date_uses_json_value_nulls_last(recording_engine) -> None:
    """resolution_date sort reads ResolutionDate from SOURCE_JSON and sorts NULLS LAST."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(
        EcnListQuery(sort="resolution_date", order="desc")
    )
    statement, _ = recording_engine.executions[-1]
    # Sort key and the projected column both derive from the stored raw JSON.
    assert (
        'JSON_VALUE(E."SOURCE_JSON", \'$.ResolutionDate\') DESC NULLS LAST' in statement
    )
    assert 'JSON_VALUE(E."SOURCE_JSON", \'$.ResolutionDate\') AS "RESOLUTION_DATE"' in statement


def test_list_item_parses_resolution_date(recording_engine) -> None:
    """A stored ISO ResolutionDate string is surfaced as a parsed datetime."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings(
        [ecn_list_row(resolution_date="2016-06-21T02:00:00+02:00")]
    )
    response = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert response.items[0].resolution_date == datetime.fromisoformat(
        "2016-06-21T02:00:00+02:00"
    )


def test_list_item_resolution_date_absent_is_none(recording_engine) -> None:
    """A missing ResolutionDate (NULL from JSON_VALUE) maps to None, not an error."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([ecn_list_row(resolution_date=None)])
    response = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert response.items[0].resolution_date is None


def test_sort_ebom_match_count_desc(recording_engine) -> None:
    """Sort on ebom_match_count uses E.EBOM_MATCH_COUNT DESC."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="ebom_match_count", order="desc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."EBOM_MATCH_COUNT" DESC' in statement


def test_sort_mbom_match_count_asc(recording_engine) -> None:
    """Sort on mbom_match_count uses E.MBOM_MATCH_COUNT ASC."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="mbom_match_count", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."MBOM_MATCH_COUNT" ASC' in statement


def test_sort_warning_count_desc(recording_engine) -> None:
    """Sort on warning_count uses E.WARNING_COUNT DESC."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="warning_count", order="desc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."WARNING_COUNT" DESC' in statement


def test_sort_analysis_state_uses_nulls_last(recording_engine) -> None:
    """Derived analysis_state sort uses the CASE alias and NULLS LAST."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="analysis_state", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert "DERIVED_ANALYSIS_STATE ASC NULLS LAST" in statement


def test_query_filter_is_bound_case_insensitive(recording_engine) -> None:
    """Free-text query produces a bound UPPER LIKE parameter, not interpolated text."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(query="SEARCH_TERM"))
    sql = "\n".join(stmt for stmt, _ in recording_engine.executions)
    params_all = {}
    for _, p in recording_engine.executions:
        if isinstance(p, dict):
            params_all.update(p)
    # The raw value must never appear verbatim in the SQL itself
    assert "SEARCH_TERM" not in sql
    # It must be a bound parameter
    assert "query_pattern" in params_all
    assert "SEARCH_TERM" in str(params_all["query_pattern"])
    # Both LIKE and UPPER must appear
    assert "LIKE :query_pattern" in sql
    assert "UPPER" in sql


def test_matching_ids_reuses_list_filters_without_pagination_or_sort(recording_engine) -> None:
    """Filter-wide generation resolves complete membership independently of list layout."""
    from app.persistence.ecn_repository import EcnRepository

    expected = [
        "00000000-0000-4000-8000-000000000071",
        "00000000-0000-4000-8000-000000000072",
    ]
    recording_engine.queue_mappings([{"ECN_ROW_ID": value} for value in expected])
    query = EcnListQuery(query="bearing", page=9, page_size=1, sort="warning_count")
    result = EcnRepository(recording_engine).list_matching_ecn_ids(query)

    statement, parameters = recording_engine.executions[-1]
    assert result == expected
    assert "LIKE :query_pattern" in statement
    assert "LIMIT" not in statement
    assert 'ORDER BY E."ECN_ROW_ID"' in statement
    assert 'E."WARNING_COUNT" DESC' not in statement
    assert parameters == {"query_pattern": "%BEARING%"}


def test_exact_ecn_number_filter_uses_bound_equality(recording_engine) -> None:
    """Exact ecn_number filter uses equality with a bound parameter, no case folding."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(ecn_number="ECN-999"))
    sql = "\n".join(stmt for stmt, _ in recording_engine.executions)
    assert "ECN-999" not in sql  # never interpolated
    assert 'E."ECN_NUMBER" = :ecn_number' in sql
    # Parameter must carry the exact trimmed value
    for _, params in recording_engine.executions:
        if isinstance(params, dict) and "ecn_number" in params:
            assert params["ecn_number"] == "ECN-999"


def test_has_warnings_true_filter(recording_engine) -> None:
    """has_warnings=True produces WARNING_COUNT > 0 (no bound parameter needed)."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(has_warnings=True))
    sql = "\n".join(stmt for stmt, _ in recording_engine.executions)
    assert 'E."WARNING_COUNT" > 0' in sql


def test_has_warnings_false_filter(recording_engine) -> None:
    """has_warnings=False produces WARNING_COUNT = 0."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(has_warnings=False))
    sql = "\n".join(stmt for stmt, _ in recording_engine.executions)
    assert 'E."WARNING_COUNT" = 0' in sql


def test_page_offset_calculated_correctly(recording_engine) -> None:
    """Page 3 with page_size 10 produces offset=20 and limit=10."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(30)
    recording_engine.queue_mappings([ecn_list_row(ordinal=i) for i in range(10)])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery(page=3, page_size=10))
    assert result.page == 3
    assert result.total_pages == 3
    _, params = recording_engine.executions[-1]
    assert params["limit"] == 10
    assert params["offset"] == 20


def test_zero_rows_produces_zero_total_pages(recording_engine) -> None:
    """When total_items is 0, total_pages must also be 0."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert result.total_items == 0
    assert result.total_pages == 0
    assert result.items == []


def test_display_ecn_number_falls_back_to_missing_label(recording_engine) -> None:
    """When ecn_number is None the list item shows 'Missing ECN ID' as display label."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([ecn_list_row(ordinal=0, ecn_number=None)])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert result.items[0].ecn_number is None
    assert result.items[0].display_ecn_number == "Missing ECN ID"


def test_six_counts_mapped_from_ecn_row(recording_engine) -> None:
    """All six deterministic counts are transferred verbatim from the ECN row."""
    from app.persistence.ecn_repository import EcnRepository

    row = ecn_list_row(
        ordinal=0,
        AFFECTED_ITEM_COUNT=10,
        RAW_ECN_MATCH_COUNT=8,
        EBOM_MATCH_COUNT=6,
        EBOM_CONTEXT_COUNT=4,
        MBOM_MATCH_COUNT=3,
        WARNING_COUNT=1,
    )
    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([row])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    counts = result.items[0].counts
    assert counts.affected_item_count == 10
    assert counts.raw_ecn_match_count == 8
    assert counts.ebom_match_count == 6
    assert counts.ebom_context_count == 4
    assert counts.mbom_match_count == 3
    assert counts.warning_count == 1


def test_count_and_list_where_clauses_are_identical(recording_engine) -> None:
    """Both count and list SQL use the same WHERE clause text for a given filter."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(
        EcnListQuery(analysis_state=AnalysisFilterState.QUEUED, has_warnings=True)
    )
    count_stmt, count_params = recording_engine.executions[0]
    list_stmt, list_params = recording_engine.executions[1]
    # Both statements must contain the same filter text
    assert 'J."STATE" = :analysis_state_value' in count_stmt
    assert 'J."STATE" = :analysis_state_value' in list_stmt
    assert 'E."WARNING_COUNT" > 0' in count_stmt
    assert 'E."WARNING_COUNT" > 0' in list_stmt
    # Parameter values must be equal (ignoring pagination params)
    assert count_params["analysis_state_value"] == list_params["analysis_state_value"]


def test_analysis_state_derived_correctly_from_completed_with_gaps_row(recording_engine) -> None:
    """A row with DERIVED_ANALYSIS_STATE=COMPLETED_WITH_GAPS maps to the enum value."""
    from app.persistence.ecn_repository import EcnRepository

    row = ecn_list_row(
        ordinal=0,
        analysis_state="COMPLETED_WITH_GAPS",
        completeness="COMPLETED_WITH_GAPS",
    )
    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([row])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    item = result.items[0]
    assert item.analysis_state is AnalysisFilterState.COMPLETED_WITH_GAPS
    assert item.completeness is not None


def test_failed_list_row_reports_compatible_partial_stages(recording_engine) -> None:
    """Distinguish a failed job with saved stages from one with no usable output."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([ecn_list_row(
        analysis_state="FAILED", HAS_PARTIAL_ANALYSIS=1,
    )])
    item = EcnRepository(recording_engine).list_ecns(EcnListQuery()).items[0]
    list_sql = recording_engine.executions[-1][0]
    assert item.analysis_state is AnalysisFilterState.FAILED
    assert item.has_partial_analysis is True
    assert 'S."EVIDENCE_HASH" = E."EVIDENCE_HASH"' in list_sql
    assert 'LA."ATTEMPT_NUMBER" > COALESCE(BA."ATTEMPT_NUMBER", 0)' in list_sql


def test_list_exposes_stored_overall_impact_without_evidence_payload(
    recording_engine,
) -> None:
    """Project overall impact from the bounded stored analysis document."""
    import json

    from app.persistence.ecn_repository import EcnRepository
    from tests.fixtures.builders import analysis_payload

    payload = analysis_payload()
    payload["impact_factors"] = {
        "manufacturing": {
            "severity": "LOW",
            "assessment": "Manufacturing impact is bounded.",
            "evidence_ids": ["EBOM_MBOM_IMPACT-1"],
            "missing_evidence": [],
        },
        "procurement": {
            "severity": "LOW",
            "assessment": "Procurement impact is bounded.",
            "evidence_ids": ["AFFECTED_MATERIALS-1"],
            "missing_evidence": [],
        },
        "configuration": {
            "severity": "LOW",
            "assessment": "Configuration impact is bounded.",
            "evidence_ids": ["CONFIGURATION_SCOPE-1"],
            "missing_evidence": [],
        },
        "data_readiness": {
            "severity": "UNKNOWN",
            "assessment": "Readiness is reported separately.",
            "evidence_ids": [],
            "missing_evidence": ["Deterministic readiness"],
        },
    }
    payload["overall_impact"] = "LOW"
    row = ecn_list_row(
        ordinal=0,
        analysis_state="COMPLETED",
        completeness="COMPLETED",
        ANALYSIS_CONTENT_JSON=json.dumps(payload),
    )
    recording_engine.queue_scalar(1)
    recording_engine.queue_mappings([row])
    result = EcnRepository(recording_engine).list_ecns(EcnListQuery())
    assert result.items[0].overall_impact.value == "LOW"
    sql, _ = recording_engine.executions[-1]
    assert 'A."CONTENT_JSON" AS "ANALYSIS_CONTENT_JSON"' in sql
    assert 'E."EVIDENCE_JSON"' not in sql


def test_no_filter_produces_no_where_clause(recording_engine) -> None:
    """An empty EcnListQuery generates no outer WHERE filter."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery())
    for stmt, _ in recording_engine.executions:
        # The stage-availability SELECT may have its own correlated WHERE.
        assert "WHERE" not in stmt.rsplit('FROM "ECN_IMPACT_ECN" E', 1)[1]


# ===========================================================================
# Suite 2: Detail / evidence / queued-job reads
# ===========================================================================


def test_evidence_lookup_is_scoped_to_requested_ecn(recording_engine) -> None:
    """Evidence lookup selects only the requested ECN row; lookup ID never reaches SQL."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mapping({"EVIDENCE_JSON": evidence_bundle_json(EV_OWN_ID)})
    repository = EcnRepository(recording_engine)
    # EV_OTHER_ID is not in the bundle that was queued for ECN_ROW_ID
    assert repository.get_evidence(ECN_ROW_ID, EV_OTHER_ID) is None
    statement, params = recording_engine.executions[-1]
    assert 'WHERE E."ECN_ROW_ID" = :ecn_row_id' in statement
    # The evidence_id itself must never be interpolated into SQL
    assert EV_OTHER_ID not in statement
    assert params == {"ecn_row_id": ECN_ROW_ID}


def test_same_ecn_evidence_lookup_returns_item(recording_engine) -> None:
    """Evidence lookup returns the item when the evidence_id is in the ECN bundle."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mapping({"EVIDENCE_JSON": evidence_bundle_json(EV_OWN_ID)})
    result = EcnRepository(recording_engine).get_evidence(ECN_ROW_ID, EV_OWN_ID)
    assert result is not None
    assert result.evidence_id == EV_OWN_ID
    assert result.section_key is AnalysisSectionKey.EXECUTIVE_SUMMARY


def test_evidence_lookup_returns_none_for_missing_ecn_row(recording_engine) -> None:
    """When the ECN row does not exist, get_evidence returns None without error."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mapping(None)
    result = EcnRepository(recording_engine).get_evidence(ECN_ROW_ID, EV_OWN_ID)
    assert result is None


def test_get_evidence_rejects_noncanonical_ecn_row_id() -> None:
    """A braced or bare-hex ecn_row_id raises ValueError before any DB access."""
    from tests.fixtures.persistence import RecordingEngine
    from app.persistence.ecn_repository import EcnRepository

    engine = RecordingEngine()
    braced_id = "{00000000-0000-4000-8000-000000000004}"
    with pytest.raises(ValueError, match="canonical"):
        EcnRepository(engine).get_evidence(braced_id, EV_OWN_ID)
    assert engine.connect_count == 0


def test_get_evidence_rejects_empty_evidence_id(recording_engine) -> None:
    """An empty evidence_id is rejected before a DB query is made."""
    from app.persistence.ecn_repository import EcnRepository

    result = EcnRepository(recording_engine).get_evidence(ECN_ROW_ID, "")
    assert result is None
    assert recording_engine.connect_count == 0


def test_get_evidence_rejects_oversized_evidence_id(recording_engine) -> None:
    """An evidence_id longer than 200 chars is rejected without DB access."""
    from app.persistence.ecn_repository import EcnRepository

    oversized = "x" * 201
    result = EcnRepository(recording_engine).get_evidence(ECN_ROW_ID, oversized)
    assert result is None
    assert recording_engine.connect_count == 0


def test_queued_jobs_are_deterministic_and_upload_scoped(recording_engine) -> None:
    """list_queued_job_ids returns QUEUED jobs ordered by source ordinal and job_id."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mappings([{"JOB_ID": JOB_ID_1}, {"JOB_ID": JOB_ID_2}])
    result = EcnRepository(recording_engine).list_queued_job_ids(UPLOAD_ID)
    assert result == [JOB_ID_1, JOB_ID_2]
    sql, params = recording_engine.executions[-1]
    assert 'J."STATE" = :queued_state' in sql
    assert 'ORDER BY E."SOURCE_ORDINAL" ASC, J."JOB_ID" ASC' in sql
    assert params["upload_id"] == UPLOAD_ID


def test_queued_jobs_returns_empty_when_none_queued(recording_engine) -> None:
    """list_queued_job_ids returns an empty list when no jobs match."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mappings([])
    result = EcnRepository(recording_engine).list_queued_job_ids(UPLOAD_ID)
    assert result == []


def test_get_ecn_detail_returns_none_for_unknown_row(recording_engine) -> None:
    """Return no row and keep large source/evidence payloads out of detail SQL."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mapping(None)
    result = EcnRepository(recording_engine).get_ecn_detail(ECN_ROW_ID)
    assert result is None
    statement, _params = recording_engine.executions[-1]
    assert 'E."EVIDENCE_JSON"' not in statement
    # SOURCE_JSON must never be hydrated as a full column into the detail read; it
    # is touched only by bounded JSON_VALUE scalars for source overview facts.
    assert 'E."SOURCE_JSON" AS ' not in statement
    assert 'JSON_VALUE(E."SOURCE_JSON", \'$.ResolutionDate\') AS "RESOLUTION_DATE"' in statement
    assert 'AS "SOURCE_STATUS"' in statement
    assert 'AS "REQUESTOR_NAME"' in statement
    assert 'AS "AFFECTED_PRODUCTS"' in statement
    assert '"CONTENT_BYTES"' not in statement
    assert '"NORMALIZED_CONTENT"' not in statement


def test_get_ecn_detail_rejects_noncanonical_id() -> None:
    """A non-canonical ecn_row_id raises ValueError before any DB access."""
    from tests.fixtures.persistence import RecordingEngine
    from app.persistence.ecn_repository import EcnRepository

    engine = RecordingEngine()
    with pytest.raises(ValueError):
        EcnRepository(engine).get_ecn_detail("not-a-uuid")
    assert engine.connect_count == 0


def test_malformed_evidence_json_raises_persistence_error(recording_engine) -> None:
    """get_evidence raises PersistenceStateError when EVIDENCE_JSON is malformed."""
    from app.ecn.errors import PersistenceStateError
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_mapping({"EVIDENCE_JSON": "{{not valid json}}"})
    with pytest.raises(PersistenceStateError):
        EcnRepository(recording_engine).get_evidence(ECN_ROW_ID, EV_OWN_ID)


# ===========================================================================
# Suite 3: Pure response mapper assertions (no SQL, no engine)
# ===========================================================================


def test_detail_mapper_flattens_the_exact_occurrence_and_provenance() -> None:
    """ecn_detail_response_from_record correctly maps all occurrence and provenance fields."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(source_ordinal=0, ecn_number=None)
    response = ecn_detail_response_from_record(record)
    assert str(response.ecn_row_id) == record.ecn.ecn_row_id
    assert response.display_ecn_number == "Missing ECN ID"
    assert response.ecn_number is None
    assert response.source_ordinal == 0
    assert response.source_file_name == record.ecn_artifact.file_name
    assert response.source_status is None
    assert response.requestor_name is None
    assert response.affected_products == []
    assert response.counts == record.ecn.counts
    assert response.ecn_artifact.kind is record.ecn_artifact.artifact_kind
    assert response.rule_artifact is None
    assert response.analysis is None


def test_detail_mapper_with_rule_artifact() -> None:
    """When a rule artifact is present, it is correctly mapped to ArtifactView."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(include_rule_artifact=True)
    response = ecn_detail_response_from_record(record)
    assert response.rule_artifact is not None
    assert response.rule_artifact.kind is ArtifactKind.CONDITIONAL_RULE_WORKBOOK


def test_detail_mapper_includes_source_overview_facts() -> None:
    """Expose only the source facts rendered in the source-record overview."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(
        source_status="Released",
        requestor_name="M. Ostrow",
        affected_products=("8700S", "M1V"),
    )
    response = ecn_detail_response_from_record(record)
    assert response.source_status == "Released"
    assert response.requestor_name == "M. Ostrow"
    assert response.affected_products == ["8700S", "M1V"]


def test_detail_mapper_with_analysis() -> None:
    """When an analysis is present, it is correctly forwarded to the response."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(include_analysis=True)
    response = ecn_detail_response_from_record(record)
    assert response.analysis is not None
    assert str(response.analysis.analysis_id) == "00000000-0000-4000-8000-000000000050"
    assert response.analysis.overall_impact.value == "UNKNOWN"
    assert response.analysis.impact_factors.data_readiness.severity.value == "UNKNOWN"


def test_evidence_mapper_keeps_identity_under_technical_details() -> None:
    """Keep the internal evidence ID available only in the audit disclosure."""
    from app.persistence.codec import evidence_detail_response_from_item

    item = evidence_item(section_key=AnalysisSectionKey.CHANGE_INTENT)
    response = evidence_detail_response_from_item(ECN_ROW_ID, item)
    assert str(response.ecn_row_id) == ECN_ROW_ID
    assert response.item.technical_details is not None
    assert response.item.technical_details.internal_evidence_ids == [item.evidence_id]
    assert not hasattr(response.item, "evidence_id")
    assert not hasattr(response.item, "section_key")


def test_evidence_mapper_converts_tuples_to_lists() -> None:
    """Map source and qualifier tuples into the nested public audit DTO."""
    from app.persistence.codec import evidence_detail_response_from_item

    item = evidence_item()
    response = evidence_detail_response_from_item(ECN_ROW_ID, item)
    assert isinstance(response.item.sources, list)
    assert response.item.technical_details is not None
    assert isinstance(response.item.technical_details.related_ids, list)
    assert isinstance(response.item.technical_details.quality_qualifiers, list)
    assert response.item.technical_details.quality_qualifiers == ["Q1"]


def test_evidence_mapper_rejects_noncanonical_ecn_row_id() -> None:
    """evidence_detail_response_from_item raises ValueError for a non-canonical (braced) ID."""
    from app.persistence.codec import evidence_detail_response_from_item

    # UUID() accepts braced form but canonical_uuid rejects it with a clear message.
    braced = "{00000000-0000-4000-8000-000000000004}"
    with pytest.raises(ValueError, match="canonical"):
        evidence_detail_response_from_item(braced, evidence_item())


def test_artifact_view_from_record_maps_without_bytes() -> None:
    """artifact_view_from_record exposes only safe metadata, never content bytes."""
    from app.persistence.codec import artifact_view_from_record
    from tests.fixtures.builders import make_source_artifact

    artifact = make_source_artifact(artifact_id=ARTIFACT_ID)
    view = artifact_view_from_record(artifact)
    assert str(view.artifact_id) == ARTIFACT_ID
    assert view.kind is artifact.artifact_kind
    assert view.file_name == artifact.file_name
    assert view.sha256 == artifact.sha256
    # ArtifactView must not have content_bytes or normalized_content fields
    assert not hasattr(view, "content_bytes")


def test_evidence_section_summaries_maps_all_six_groups() -> None:
    """evidence_section_summaries_from_bundle returns all six section lists."""
    from app.persistence.codec import evidence_section_summaries_from_bundle
    from tests.fixtures.builders import make_evidence_bundle, make_evidence_bundle_sections

    bundle = make_evidence_bundle(
        ecn_row_id=ECN_ROW_ID,
        sections=make_evidence_bundle_sections(
            change_intent=(evidence_item(section_key=AnalysisSectionKey.CHANGE_INTENT),)
        ),
    )
    summaries = evidence_section_summaries_from_bundle(bundle)
    assert len(summaries.change_intent) == 1
    assert summaries.change_intent[0].evidence_id == EV_OWN_ID
    # Other sections must be present and empty
    assert summaries.executive_summary == []
    assert summaries.affected_materials == []
    assert summaries.ebom_mbom_impact == []
    assert summaries.configuration_scope == []
    assert summaries.uncertainties_and_missing_evidence == []


def test_evidence_section_summaries_preserves_quality_qualifiers() -> None:
    """Evidence summaries carry over quality qualifiers exactly."""
    from app.persistence.codec import evidence_section_summaries_from_bundle
    from tests.fixtures.builders import make_evidence_bundle, make_evidence_bundle_sections

    item = evidence_item(section_key=AnalysisSectionKey.EXECUTIVE_SUMMARY)
    bundle = make_evidence_bundle(
        ecn_row_id=ECN_ROW_ID,
        sections=make_evidence_bundle_sections(executive_summary=(item,)),
    )
    summaries = evidence_section_summaries_from_bundle(bundle)
    assert summaries.executive_summary[0].quality_qualifiers == ["Q1"]


def test_detail_response_counts_match_stored_ecn(recording_engine) -> None:
    """The six counts on EcnDetailResponse are sourced directly from StoredEcnRecord."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(source_ordinal=0)
    response = ecn_detail_response_from_record(record)
    assert response.counts.affected_item_count == record.ecn.counts.affected_item_count
    assert response.counts.warning_count == record.ecn.counts.warning_count


def test_detail_mapper_uploaded_at_from_upload_record() -> None:
    """EcnDetailResponse.uploaded_at comes from the upload record's created_at."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record()
    response = ecn_detail_response_from_record(record)
    assert response.uploaded_at == record.upload.created_at


def test_detail_mapper_parses_resolution_date() -> None:
    """EcnDetailResponse.resolution_date parses the stored ISO ResolutionDate string."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record(resolution_date="2017-01-25T11:56:05+01:00")
    response = ecn_detail_response_from_record(record)
    assert response.resolution_date == datetime.fromisoformat("2017-01-25T11:56:05+01:00")


def test_detail_mapper_resolution_date_absent_is_none() -> None:
    """A record without a ResolutionDate maps resolution_date to None."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record()
    response = ecn_detail_response_from_record(record)
    assert response.resolution_date is None


def test_detail_mapper_bom_artifact_mapped_correctly() -> None:
    """BOM artifact is correctly mapped to ArtifactView."""
    from app.persistence.codec import ecn_detail_response_from_record

    record = ecn_detail_record()
    response = ecn_detail_response_from_record(record)
    assert response.bom_artifact.kind is ArtifactKind.BOM_WORKBOOK
    assert str(response.bom_artifact.artifact_id) == BOM_ARTIFACT_ID


def test_plant_filter_is_bound_case_insensitive_substring(recording_engine) -> None:
    """The plant filter is a bound UPPER LIKE over the stored plant-label JSON."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(plant=" pl10 "))
    statement, parameters = recording_engine.executions[-1]
    assert 'UPPER(E."AFFECTED_PLANTS_JSON") LIKE :plant_pattern' in statement
    assert "pl10" not in statement.lower().replace("plant_pattern", "")
    assert parameters["plant_pattern"] == "%PL10%"


def test_sort_affected_plants_uses_nulls_last(recording_engine) -> None:
    """Rows whose plants were never assessed (NULL) sort after every assessed row."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(0)
    recording_engine.queue_mappings([])
    EcnRepository(recording_engine).list_ecns(EcnListQuery(sort="affected_plants", order="asc"))
    statement, _ = recording_engine.executions[-1]
    assert 'E."AFFECTED_PLANTS_JSON" ASC NULLS LAST' in statement


def test_list_item_parses_affected_plants(recording_engine) -> None:
    """Stored labels become a list; empty stays empty; NULL means not assessed yet."""
    from app.persistence.ecn_repository import EcnRepository

    recording_engine.queue_scalar(3)
    recording_engine.queue_mappings([
        ecn_list_row(ordinal=0, AFFECTED_PLANTS_JSON='["PL10 Plant A", "DE01"]'),
        ecn_list_row(ordinal=1, AFFECTED_PLANTS_JSON="[]"),
        ecn_list_row(ordinal=2, AFFECTED_PLANTS_JSON=None),
    ])
    items = EcnRepository(recording_engine).list_ecns(EcnListQuery()).items
    assert [item.affected_plants for item in items] == [["PL10 Plant A", "DE01"], [], None]
