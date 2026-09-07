"""Public API tests for profile-authoritative responses and benchmark scores."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.models.benchmarking import (
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
    BenchmarkScopeScore,
)
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


class _ProfileRouteRepository(InMemoryAiReviewRepository):
    """Provide one real-scoring framework and active imported peer cohort."""

    def __init__(self) -> None:
        """Initialize framework, active version metadata, and peer score rows."""

        super().__init__()
        self.framework_questions = [
            AssessmentQuestion(
                question_id="Q.STR.03.01",
                dimension="Strategy",
                section="Objectives",
                topic_title="Definition of Objectives",
                question="Are objectives monitored?",
                answer_items=[
                    AnswerItem(
                        answer_item_id="Q.STR.03.01-L1-001",
                        question_id="Q.STR.03.01",
                        level=1,
                        item_index=1,
                        text="Objectives are set.",
                    ),
                    AnswerItem(
                        answer_item_id="Q.STR.03.01-L2-001",
                        question_id="Q.STR.03.01",
                        level=2,
                        item_index=1,
                        text="Objectives are monitored.",
                    ),
                    AnswerItem(
                        answer_item_id="Q.STR.03.01-L3-001",
                        question_id="Q.STR.03.01",
                        level=3,
                        item_index=1,
                        text="Objectives drive corrective actions.",
                    ),
                ],
            )
        ]
        self.active_benchmark_import = BenchmarkImportInfo(
            import_id="import-active",
            source_filename="benchmark.xlsx",
            source_sha256="c" * 64,
            scoring_version="assessment-v1",
            status="active",
            is_active=True,
        )
        self.benchmark_peer_submissions = [
            self._peer("company-current", "self", 100),
            self._peer("peer-1", "q1", 40),
            self._peer("peer-2", "q2", 50),
            self._peer("peer-3", "q3", 60),
        ]

    @staticmethod
    def _peer(company_id: str, questionnaire_id: str, score: float) -> BenchmarkPeerSubmission:
        """Return one exact-cohort released submission at all score scopes."""

        return BenchmarkPeerSubmission(
            source_company_id=company_id,
            questionnaire_id=questionnaire_id,
            customer_class="class_1",
            nace1="Energy",
            submission_date=date(2026, 1, 1),
            release_status="REL",
            scores=[
                BenchmarkScopeScore(scope_type="overall", calculated_score=score),
                BenchmarkScopeScore(
                    scope_type="dimension",
                    dimension="Strategy",
                    calculated_score=score,
                ),
                BenchmarkScopeScore(
                    scope_type="topic",
                    dimension="Strategy",
                    question_id="Q.STR.03.01",
                    calculated_score=score,
                ),
            ],
        )


@pytest.fixture
def profile_repository() -> _ProfileRouteRepository:
    """Return a mutable in-memory repository for one API test."""

    return _ProfileRouteRepository()


@pytest.fixture(autouse=True)
def override_profile_repository(
    profile_repository: _ProfileRouteRepository,
) -> Iterator[None]:
    """Install and later remove the route repository dependency override."""

    app.dependency_overrides[get_ai_review_repository] = lambda: profile_repository
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_ai_review_repository, None)


def _put_profile(api_client: TestClient) -> object:
    """Persist the canonical class-1 Energy profile through the API."""

    return api_client.put(
        "/api/assessment/profile",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "display_name": "  assessment  ",
            "source_company_id": " company-current ",
            "customer_class": "class_1",
            "nace1": " Energy ",
        },
    )


def test_profile_put_get_normalizes_text_and_rejects_unknown_class(
    api_client: TestClient,
) -> None:
    """Verify profile API persistence, trimming, lookup, and strict class validation."""

    created = _put_profile(api_client)
    loaded = api_client.get(
        "/api/assessment/profile?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )
    rejected = api_client.put(
        "/api/assessment/profile",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-2",
            "display_name": "Invalid",
            "customer_class": "unknown",
            "nace1": "Energy",
        },
    )

    assert created.status_code == 200
    assert created.json() == {
        "assessment_id": "assessment-1",
        "display_name": "assessment",
        "source_company_id": "company-current",
        "customer_class": "class_1",
        "nace1": "Energy",
    }
    assert loaded.status_code == 200
    assert loaded.json() == created.json()
    assert rejected.status_code == 400
    assert "Unsupported customer class" in rejected.json()["detail"]


def test_profile_class_is_authoritative_for_response_save_and_get(
    api_client: TestClient,
) -> None:
    """Verify omitted class uses the profile and explicit conflicts return 409."""

    assert _put_profile(api_client).status_code == 200
    saved = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L2-001"]},
        },
    )
    loaded = api_client.get(
        "/api/assessment/responses?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )
    conflict = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_5",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L1-001"]},
        },
    )

    assert saved.status_code == 200
    assert saved.json()["customer_class"] == "class_1"
    assert loaded.status_code == 200
    assert loaded.json()["customer_class"] == "class_1"
    assert conflict.status_code == 409
    assert "persisted assessment profile" in conflict.json()["detail"]


def test_profile_class_change_removes_now_unavailable_saved_answers(
    api_client: TestClient,
) -> None:
    """Verify lowering profile scope cannot resurrect higher-level selections."""

    broad_profile = api_client.put(
        "/api/assessment/profile",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "display_name": "assessment",
            "customer_class": "class_5",
            "nace1": "Energy",
        },
    )
    saved = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L3-001"]},
        },
    )
    narrowed_profile = _put_profile(api_client)
    loaded = api_client.get(
        "/api/assessment/responses?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )

    assert broad_profile.status_code == 200
    assert saved.json()["answers"] == {
        "Q.STR.03.01": ["Q.STR.03.01-L3-001"]
    }
    assert narrowed_profile.status_code == 200
    assert loaded.json()["customer_class"] == "class_1"
    assert loaded.json()["answers"].get("Q.STR.03.01", []) == []


def test_score_uses_profile_exact_cohort_excludes_company_and_rejects_legacy_override(
    api_client: TestClient,
) -> None:
    """Verify score route uses profile context and never a legacy query override."""

    assert _put_profile(api_client).status_code == 200
    api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L1-001"]},
        },
    )
    scored = api_client.get(
        "/api/assessment/score?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )
    class_conflict = api_client.get(
        "/api/assessment/score?assessment_id=assessment-1&customer_class=class_2",
        headers={"X-API-Key": "test-api-key"},
    )
    sector_conflict = api_client.get(
        "/api/assessment/score?assessment_id=assessment-1&sector=Manufacturing",
        headers={"X-API-Key": "test-api-key"},
    )

    assert scored.status_code == 200
    payload = scored.json()
    assert payload["customer_class"] == "class_1"
    assert payload["sector"] == "Energy"
    assert payload["benchmark_context"]["peer_sample_size"] == 3
    assert payload["benchmark"]["peer_average"] == 50.0
    assert payload["benchmark"]["best_peer"] == 60.0
    assert payload["benchmark_score"] == 50.0
    assert payload["same_sector_benchmark_score"] is None
    assert payload["same_size_benchmark_score"] is None
    assert class_conflict.status_code == 409
    assert sector_conflict.status_code == 409


def test_benchmark_options_expose_active_exact_cohorts_without_peer_identities(
    api_client: TestClient,
) -> None:
    """Verify selector options are active-version cohort labels, never companies."""

    response = api_client.get(
        "/api/assessment/benchmark-options",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "reason": None,
        "import_id": "import-active",
        "customer_classes": ["class_1"],
        "nace1_sectors": ["Energy"],
        "cohorts": [{"customer_class": "class_1", "nace1": "Energy"}],
    }
    serialized = str(response.json()).lower()
    assert "source_company_id" not in serialized
    assert "questionnaire_id" not in serialized
