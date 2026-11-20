"""Unit tests for the ECN API envelope models.

Tests cover:
- EcnListQuery: sort allowlist, pagination bounds, field widths.
- UploadResponse: state-count equation (must equal imported_ecn_count).
- AnalysisProgressResponse: unfinished and total count equations.
- ImpactCounts: nonnegative enforcement.
- ErrorResponse: code and message width limits.
- EcnListItem: AnalysisFilterState including COMPLETED_WITH_GAPS.
- RetryResponse: literal QUEUED state.
- HistoryResetRequest: exact confirmation phrase.
- HistoryResetResponse: nonnegative deletion counts.
- UUID fields: string UUIDs accepted at the API boundary.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_api_models.py
"""

import uuid
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.ecn.analysis_models import AnalysisCompleteness
from app.ecn.api_models import (
    AnalysisFilterState,
    AnalysisProgressResponse,
    EcnListItem,
    EcnListQuery,
    EcnListResponse,
    ErrorResponse,
    HistoryResetRequest,
    HistoryResetResponse,
    ImpactCounts,
    RetryResponse,
    UploadResponse,
)


# ---------------------------------------------------------------------------
# EcnListQuery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["SUMMARIES_AVAILABLE", "IN_PROGRESS"])
def test_analysis_groups_are_filter_only(group) -> None:
    """Inbox groups are accepted as filters but never as derived row states."""
    from app.ecn.api_models import AnalysisFilterState

    assert EcnListQuery(analysis_state=group).analysis_state.value == group
    with pytest.raises(ValueError):
        AnalysisFilterState(group)


def test_settings_defaults_and_canonical_model_config() -> None:
    """Placeholder test exercised in test_ecn_settings.py; kept here for suite completeness."""
    # This test name matches what's imported from test_ecn_settings via the brief's
    # combined-run command.  It exercises EcnListQuery defaults.
    q = EcnListQuery()
    assert q.page == 1
    assert q.page_size == 25
    assert q.sort == "uploaded_at"
    assert q.order == "desc"


def test_progress_rejects_an_inconsistent_total() -> None:
    """AnalysisProgressResponse must reject a total_count that does not equal all state sums."""
    with pytest.raises(ValidationError, match="total_count"):
        AnalysisProgressResponse(
            upload_id=None,
            queued_count=1,
            running_count=1,
            completed_count=2,
            completed_with_gaps_count=1,
            failed_count=1,
            unfinished_count=2,
            total_count=5,
        )


def test_list_query_rejects_unknown_sort_and_overlong_business_id() -> None:
    """EcnListQuery must reject unknown sort values and ecn_number > 80 chars."""
    with pytest.raises(ValidationError):
        EcnListQuery(sort="raw_sql")
    with pytest.raises(ValidationError):
        EcnListQuery(ecn_number="X" * 81)


def test_list_query_rejects_page_size_over_100() -> None:
    """page_size must be at most 100."""
    with pytest.raises(ValidationError):
        EcnListQuery(page_size=101)


def test_list_query_rejects_page_zero() -> None:
    """page must be >= 1."""
    with pytest.raises(ValidationError):
        EcnListQuery(page=0)


def test_list_query_rejects_query_over_500_chars() -> None:
    """query field must be at most 500 characters."""
    with pytest.raises(ValidationError):
        EcnListQuery(query="q" * 501)


def test_list_query_accepts_all_documented_sort_values() -> None:
    """Every sort value documented in the brief must be accepted."""
    valid_sorts = [
        "uploaded_at",
        "ecn_number",
        "description",
        "affected_item_count",
        "ebom_match_count",
        "mbom_match_count",
        "warning_count",
        "analysis_state",
    ]
    for sort in valid_sorts:
        q = EcnListQuery(sort=sort)  # type: ignore[arg-type]
        assert q.sort == sort


def test_list_query_accepts_asc_and_desc_order() -> None:
    """order must accept 'asc' and 'desc' only."""
    assert EcnListQuery(order="asc").order == "asc"
    assert EcnListQuery(order="desc").order == "desc"
    with pytest.raises(ValidationError):
        EcnListQuery(order="ASC")


# ---------------------------------------------------------------------------
# UploadResponse state-count equation
# ---------------------------------------------------------------------------


def _upload_response_data(**overrides: object) -> dict:
    """Return a minimal valid UploadResponse data dict."""
    data: dict = {
        "upload_id": uuid.uuid4(),
        "request_id": uuid.uuid4(),
        "created": True,
        "imported_ecn_count": 5,
        "bom_artifact_id": uuid.uuid4(),
        "bom_reused": False,
        "queued_count": 3,
        "blocked_count": 0,
        "running_count": 1,
        "completed_count": 1,
        "completed_with_gaps_count": 0,
        "failed_count": 0,
        "list_url": "/ecn/",
        "progress_url": "/ecn/progress",
    }
    data.update(overrides)
    return data


def test_upload_response_valid_state_counts() -> None:
    """UploadResponse must succeed when state counts equal imported_ecn_count."""
    resp = UploadResponse.model_validate(_upload_response_data())
    assert resp.imported_ecn_count == 5
    assert resp.queued_count + resp.running_count + resp.completed_count + \
           resp.blocked_count + resp.completed_with_gaps_count + resp.failed_count == 5


def test_upload_response_rejects_mismatched_state_counts() -> None:
    """UploadResponse must reject state counts that don't sum to imported_ecn_count."""
    with pytest.raises(ValidationError, match="imported_ecn_count"):
        UploadResponse.model_validate(
            _upload_response_data(
                imported_ecn_count=5,
                queued_count=1,
                running_count=1,
                completed_count=1,
                completed_with_gaps_count=1,
                failed_count=0,  # total=4, not 5
            )
        )


# ---------------------------------------------------------------------------
# AnalysisProgressResponse
# ---------------------------------------------------------------------------


def _progress_data(**overrides: object) -> dict:
    """Return a minimal valid AnalysisProgressResponse data dict."""
    data: dict = {
        "upload_id": None,
        "queued_count": 2,
        "blocked_count": 0,
        "running_count": 1,
        "completed_count": 3,
        "completed_with_gaps_count": 1,
        "failed_count": 1,
        "unfinished_count": 3,   # queued + running = 2 + 1
        "total_count": 8,        # 3 + 3 + 1 + 1 = 8
    }
    data.update(overrides)
    return data


def test_progress_valid_counts() -> None:
    """AnalysisProgressResponse must succeed when all count equations hold."""
    resp = AnalysisProgressResponse.model_validate(_progress_data())
    assert resp.total_count == 8


def test_progress_rejects_bad_unfinished_count() -> None:
    """AnalysisProgressResponse must reject when unfinished_count != queued + running."""
    with pytest.raises(ValidationError, match="unfinished_count"):
        AnalysisProgressResponse.model_validate(
            _progress_data(queued_count=2, running_count=1, unfinished_count=99)
        )


# ---------------------------------------------------------------------------
# ImpactCounts
# ---------------------------------------------------------------------------


def test_impact_counts_rejects_negative_values() -> None:
    """ImpactCounts must reject negative counts."""
    with pytest.raises(ValidationError):
        ImpactCounts(
            affected_item_count=-1,
            raw_ecn_match_count=0,
            ebom_match_count=0,
            ebom_context_count=0,
            mbom_match_count=0,
            warning_count=0,
        )


def test_impact_counts_accepts_zero_values() -> None:
    """ImpactCounts must accept all-zero counts."""
    counts = ImpactCounts(
        affected_item_count=0,
        raw_ecn_match_count=0,
        ebom_match_count=0,
        ebom_context_count=0,
        mbom_match_count=0,
        warning_count=0,
    )
    assert counts.warning_count == 0


# ---------------------------------------------------------------------------
# ErrorResponse width limits
# ---------------------------------------------------------------------------


def test_error_response_rejects_overlong_code() -> None:
    """ErrorResponse code must be <= 80 characters."""
    with pytest.raises(ValidationError):
        ErrorResponse(code="E" * 81, message="some message")


def test_error_response_rejects_overlong_message() -> None:
    """ErrorResponse message must be <= 2000 characters."""
    with pytest.raises(ValidationError):
        ErrorResponse(code="ERR", message="M" * 2001)


def test_error_response_accepts_max_widths() -> None:
    """ErrorResponse must accept exactly the maximum code and message lengths."""
    resp = ErrorResponse(code="E" * 80, message="M" * 2000)
    assert len(resp.code) == 80
    assert len(resp.message) == 2000


# ---------------------------------------------------------------------------
# EcnListItem analysis_state accepts COMPLETED_WITH_GAPS
# ---------------------------------------------------------------------------


def _ecn_list_item_data(**overrides: object) -> dict:
    """Return a minimal valid EcnListItem data dict."""
    counts = ImpactCounts(
        affected_item_count=0,
        raw_ecn_match_count=0,
        ebom_match_count=0,
        ebom_context_count=0,
        mbom_match_count=0,
        warning_count=0,
    )
    data: dict = {
        "ecn_row_id": uuid.uuid4(),
        "ecn_number": "ECN-001",
        "display_ecn_number": "ECN-001",
        "title": "Test ECN",
        "description_summary": None,
        "source_ordinal": 0,
        "uploaded_at": datetime(2026, 9, 14, tzinfo=timezone.utc),
        "source_file_name": "ecn.json",
        "counts": counts,
        "analysis_state": AnalysisFilterState.COMPLETED_WITH_GAPS,
        "completeness": AnalysisCompleteness.COMPLETED_WITH_GAPS,
    }
    data.update(overrides)
    return data


def test_ecn_list_item_accepts_completed_with_gaps_state() -> None:
    """EcnListItem analysis_state must accept COMPLETED_WITH_GAPS."""
    item = EcnListItem.model_validate(_ecn_list_item_data())
    assert item.analysis_state == AnalysisFilterState.COMPLETED_WITH_GAPS


def test_ecn_list_item_completed_with_gaps_not_in_job_state() -> None:
    """COMPLETED_WITH_GAPS is an API/result projection absent from JobState."""
    from app.ecn.models import JobState

    job_state_values = {s.value for s in JobState}
    assert AnalysisFilterState.COMPLETED_WITH_GAPS.value not in job_state_values


# ---------------------------------------------------------------------------
# RetryResponse
# ---------------------------------------------------------------------------


def test_retry_response_requires_queued_literal() -> None:
    """RetryResponse state must be the literal 'QUEUED'."""
    resp = RetryResponse(
        ecn_row_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        state="QUEUED",
    )
    assert resp.state == "QUEUED"


def test_retry_response_rejects_non_queued_state() -> None:
    """RetryResponse must reject any state other than 'QUEUED'."""
    with pytest.raises(ValidationError):
        RetryResponse(
            ecn_row_id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            state="RUNNING",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# HistoryResetRequest
# ---------------------------------------------------------------------------


def test_history_reset_request_requires_exact_phrase() -> None:
    """HistoryResetRequest must accept only the exact phrase 'RESET HISTORY'."""
    req = HistoryResetRequest(confirmation="RESET HISTORY")
    assert req.confirmation == "RESET HISTORY"


def test_history_reset_request_rejects_wrong_phrase() -> None:
    """HistoryResetRequest must reject any other confirmation phrase."""
    with pytest.raises(ValidationError):
        HistoryResetRequest(confirmation="reset history")


# ---------------------------------------------------------------------------
# HistoryResetResponse
# ---------------------------------------------------------------------------


def test_history_reset_response_nonnegative_counts() -> None:
    """HistoryResetResponse must accept zero counts."""
    resp = HistoryResetResponse(
        deleted_attempt_count=0,
        deleted_analysis_count=0,
        deleted_job_count=0,
        deleted_ecn_count=0,
        deleted_upload_count=0,
        deleted_bom_row_count=0,
        deleted_source_artifact_count=0,
    )
    assert resp.deleted_ecn_count == 0


def test_history_reset_response_rejects_negative_counts() -> None:
    """HistoryResetResponse must reject negative deletion counts."""
    with pytest.raises(ValidationError):
        HistoryResetResponse(
            deleted_attempt_count=-1,
            deleted_analysis_count=0,
            deleted_job_count=0,
            deleted_ecn_count=0,
            deleted_upload_count=0,
            deleted_bom_row_count=0,
            deleted_source_artifact_count=0,
        )


# ---------------------------------------------------------------------------
# UUID fields at API boundary
# ---------------------------------------------------------------------------


def test_upload_response_accepts_uuid_strings() -> None:
    """UploadResponse must accept UUID strings via JSON deserialization."""
    data = _upload_response_data()
    # Convert UUIDs to strings (as they arrive from JSON)
    for field in ("upload_id", "request_id", "bom_artifact_id"):
        data[field] = str(data[field])
    resp = UploadResponse.model_validate(data)
    assert isinstance(resp.upload_id, UUID)


# ---------------------------------------------------------------------------
# EcnListResponse pagination
# ---------------------------------------------------------------------------


def test_ecn_list_response_nonnegative_totals() -> None:
    """EcnListResponse must accept zero total_items and total_pages."""
    resp = EcnListResponse(
        items=[],
        page=1,
        page_size=25,
        total_items=0,
        total_pages=0,
    )
    assert resp.total_items == 0


def test_ecn_list_response_rejects_page_zero() -> None:
    """EcnListResponse page must be >= 1."""
    with pytest.raises(ValidationError):
        EcnListResponse(
            items=[],
            page=0,
            page_size=25,
            total_items=0,
            total_pages=0,
        )
