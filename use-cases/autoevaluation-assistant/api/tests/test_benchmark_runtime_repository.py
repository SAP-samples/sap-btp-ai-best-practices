"""Tests for HANA/memory runtime reads of profiles and active benchmark data."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from app.models.benchmarking import (
    AssessmentProfile,
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
    BenchmarkScopeScore,
)
from app.services.ai_review_repository.hana_scoring import HanaScoringMixin
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


class _MappingResult:
    """Provide SQLAlchemy-compatible mapping access for fixed fake rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store rows returned by one fake query."""

        self.rows = rows

    def mappings(self) -> "_MappingResult":
        """Return this result for chained mapping access."""

        return self

    def all(self) -> list[dict[str, Any]]:
        """Return every configured row."""

        return self.rows

    def first(self) -> dict[str, Any] | None:
        """Return the first configured mapping or ``None``."""

        return self.rows[0] if self.rows else None


class _RuntimeSession:
    """Record runtime SQL and return fixture rows by queried table."""

    def __init__(self) -> None:
        """Initialize SQL capture and deterministic profile/import/score rows."""

        self.statements: list[tuple[str, Any]] = []

    def execute(
        self,
        statement: object,
        parameters: Any = None,
    ) -> _MappingResult:
        """Return rows for the profile, active import, or joined score query."""

        sql = str(statement).lower()
        self.statements.append((sql, parameters))
        if sql.startswith("upsert assessment_profiles"):
            return _MappingResult([])
        if "from assessment_profiles" in sql:
            return _MappingResult(
                [
                    {
                        "assessment_id": "assessment-1",
                        "display_name": "assessment",
                        "source_company_id": "company-current",
                        "customer_class": "class_1",
                        "nace1": "Energy",
                    }
                ]
            )
        if "from assessment_benchmark_imports" in sql:
            return _MappingResult(
                [
                    {
                        "import_id": "import-active",
                        "source_filename": "benchmark.xlsx",
                        "source_sha256": "b" * 64,
                        "scoring_version": "assessment-v1",
                        "row_count": 10,
                        "company_count": 4,
                        "questionnaire_count": 4,
                        "question_count": 1,
                        "accepted_count": 10,
                        "rejected_count": 0,
                        "status": "active",
                        "is_active": 1,
                        "created_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
                        "activated_at": datetime(2026, 7, 2, tzinfo=timezone.utc),
                    }
                ]
            )
        if "from assessment_benchmark_submissions" in sql:
            return _MappingResult(
                [
                    {
                        "source_company_id": "peer-1",
                        "questionnaire_id": "questionnaire-1",
                        "customer_class": "class_1",
                        "nace1": "Energy",
                        "submission_date": date(2026, 1, 1),
                        "extraction_date": date(2026, 1, 2),
                        "release_status": "REL",
                        "scope_type": "overall",
                        "dimension": None,
                        "question_id": None,
                        "calculated_score": 50,
                    },
                    {
                        "source_company_id": "peer-1",
                        "questionnaire_id": "questionnaire-1",
                        "customer_class": "class_1",
                        "nace1": "Energy",
                        "submission_date": date(2026, 1, 1),
                        "extraction_date": date(2026, 1, 2),
                        "release_status": "REL",
                        "scope_type": "topic",
                        "dimension": "Strategy",
                        "question_id": "Q.STR.03.01",
                        "calculated_score": 60,
                    },
                ]
            )
        return _MappingResult([])


class _HanaRuntimeRepository(HanaScoringMixin):
    """Expose runtime scoring mixin operations over the fake session."""

    def __init__(self) -> None:
        """Attach the deterministic fake SQL session."""

        self.session = _RuntimeSession()


def _profile() -> AssessmentProfile:
    """Return one exact profile fixture for repository operations."""

    return AssessmentProfile(
        assessment_id="assessment-1",
        display_name="assessment",
        source_company_id="company-current",
        customer_class="class_1",
        nace1="Energy",
    )


def _active_import() -> BenchmarkImportInfo:
    """Return safe active import metadata without workbook or identities."""

    return BenchmarkImportInfo(
        import_id="import-active",
        source_filename="benchmark.xlsx",
        source_sha256="b" * 64,
        scoring_version="assessment-v1",
        status="active",
        is_active=True,
    )


def _peer() -> BenchmarkPeerSubmission:
    """Return one score-bearing active-import peer submission."""

    return BenchmarkPeerSubmission(
        source_company_id="peer-1",
        questionnaire_id="questionnaire-1",
        customer_class="class_1",
        nace1="Energy",
        release_status="REL",
        scores=[BenchmarkScopeScore(scope_type="overall", calculated_score=50)],
    )


def test_memory_runtime_repository_exposes_profile_active_version_and_exact_cohort() -> None:
    """Verify memory runtime reads mirror durable profile/import interfaces."""

    repository = InMemoryAiReviewRepository()
    repository.upsert_assessment_profile(_profile())
    repository.active_benchmark_import = _active_import()
    repository.benchmark_peer_submissions = [
        _peer(),
        _peer().model_copy(update={"customer_class": "class_2"}),
        _peer().model_copy(update={"nace1": "Manufacturing"}),
    ]

    assert repository.get_assessment_profile("assessment-1") == _profile()
    assert repository.get_active_benchmark_import() == _active_import()
    assert repository.list_benchmark_peer_submissions(
        "import-active",
        "class_1",
        "Energy",
    ) == [_peer()]


def test_hana_runtime_repository_upserts_profile_and_reads_active_joined_scores() -> None:
    """Verify HANA reads use versioned tables, exact cohort binds, and no fake table."""

    repository = _HanaRuntimeRepository()

    assert repository.upsert_assessment_profile(_profile()) == _profile()
    assert repository.get_active_benchmark_import().import_id == "import-active"
    submissions = repository.list_benchmark_peer_submissions(
        "import-active",
        "class_1",
        "Energy",
    )

    assert len(submissions) == 1
    assert submissions[0].source_company_id == "peer-1"
    assert [score.scope_type for score in submissions[0].scores] == [
        "overall",
        "topic",
    ]
    joined_sql, joined_parameters = next(
        (sql, parameters)
        for sql, parameters in repository.session.statements
        if "from assessment_benchmark_submissions" in sql
    )
    assert "assessment_benchmark_companies" in joined_sql
    assert "assessment_benchmark_scores" in joined_sql
    assert "customer_class = :customer_class" in joined_sql
    assert "nace1 = :nace1" in joined_sql
    assert joined_parameters == {
        "import_id": "import-active",
        "customer_class": "class_1",
        "nace1": "Energy",
    }
    assert all(
        "assessment_score_benchmarks" not in sql
        for sql, _parameters in repository.session.statements
    )


def test_hana_response_save_aligns_all_existing_rows_to_profile_class() -> None:
    """Verify a save cannot leave older answer rows tagged with another class."""

    repository = _HanaRuntimeRepository()

    repository.save_assessment_responses(
        assessment_id="assessment-1",
        customer_class="class_1",
        answers={},
        source="manual",
    )

    assert any(
        sql.startswith("update assessment_user_answers set customer_class")
        and parameters == {
            "assessment_id": "assessment-1",
            "customer_class": "class_1",
        }
        for sql, parameters in repository.session.statements
    )
