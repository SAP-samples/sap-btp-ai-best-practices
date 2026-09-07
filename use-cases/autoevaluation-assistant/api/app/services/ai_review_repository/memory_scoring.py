"""In-memory persistence for assessment responses and score benchmarks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.models.assessment import AssessmentDimension, AssessmentQuestion
from app.models.benchmarking import (
    AssessmentProfile,
    BenchmarkCohortOption,
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
)
from app.models.scoring import ScoreBenchmarkRow
from app.services.score_benchmarks import (
    SECTOR_BENCHMARK_CLASS,
    generate_mock_score_benchmarks,
)


def _benchmark_scope_key(row: ScoreBenchmarkRow) -> tuple[str, str | None, str | None, str | None]:
    """Return the logical identity for an in-memory benchmark row.

    Inputs:
        row: Benchmark row to identify.

    Outputs:
        tuple[str, str | None, str | None, str | None]: Class, sector,
        dimension, and question scope.
    """
    return (row.customer_class, row.sector, row.dimension, row.question_id)


class MemoryScoringMixin:
    """Store assessment responses, apply events, and benchmarks in memory."""

    assessment_user_answers: dict[str, dict[str, list[str]]]
    assessment_profiles: dict[str, AssessmentProfile]
    active_benchmark_import: BenchmarkImportInfo | None
    benchmark_import_history: list[BenchmarkImportInfo]
    benchmark_peer_submissions: list[BenchmarkPeerSubmission]
    applied_suggestions: list[dict[str, Any]]
    score_benchmarks: list[ScoreBenchmarkRow]

    def upsert_assessment_profile(
        self,
        profile: AssessmentProfile,
    ) -> AssessmentProfile:
        """Insert or replace one assessment profile in memory.

        Inputs:
            profile: Validated profile carrying authoritative class/NACE context.

        Outputs:
            AssessmentProfile: Deep-copied persisted profile.
        """

        self.assessment_profiles[profile.assessment_id] = deepcopy(profile)
        return deepcopy(profile)

    def get_assessment_profile(
        self,
        assessment_id: str,
    ) -> AssessmentProfile | None:
        """Return an independent copy of one persisted assessment profile.

        Inputs:
            assessment_id: Assessment identity whose profile should be loaded.

        Outputs:
            AssessmentProfile | None: Stored profile, or ``None`` when absent.
        """

        profile = self.assessment_profiles.get(assessment_id)
        return None if profile is None else deepcopy(profile)

    def get_active_benchmark_import(self) -> BenchmarkImportInfo | None:
        """Return safe metadata for the active in-memory benchmark version.

        Inputs:
            None.

        Outputs:
            BenchmarkImportInfo | None: Independent active metadata, or ``None``.
        """

        return deepcopy(self.active_benchmark_import)

    def list_benchmark_peer_submissions(
        self,
        import_id: str,
        customer_class: str,
        nace1: str,
    ) -> list[BenchmarkPeerSubmission]:
        """Return active-version submissions in one exact class/NACE cohort.

        Inputs:
            import_id: Captured active benchmark version identity.
            customer_class: Exact normalized assessment class.
            nace1: Exact level-one NACE label.

        Outputs:
            list[BenchmarkPeerSubmission]: Deep-copied exact-cohort submissions.
        """

        if (
            self.active_benchmark_import is None
            or self.active_benchmark_import.import_id != import_id
        ):
            return []
        return [
            deepcopy(submission)
            for submission in self.benchmark_peer_submissions
            if submission.customer_class == customer_class
            and submission.nace1 == nace1
        ]

    def list_benchmark_cohort_options(
        self,
        import_id: str,
    ) -> list[BenchmarkCohortOption]:
        """Return identity-free exact cohort pairs for the active memory version.

        Inputs:
            import_id: Captured active import identity.

        Outputs:
            list[BenchmarkCohortOption]: Sorted unique class/NACE pairs.
        """

        if (
            self.active_benchmark_import is None
            or self.active_benchmark_import.import_id != import_id
        ):
            return []
        pairs = {
            (submission.customer_class, submission.nace1)
            for submission in self.benchmark_peer_submissions
            if submission.nace1
        }
        return [
            BenchmarkCohortOption(customer_class=customer_class, nace1=nace1)
            for customer_class, nace1 in sorted(pairs)
        ]

    def list_recent_benchmark_imports(
        self,
        limit: int,
    ) -> list[BenchmarkImportInfo]:
        """Return a bounded independent copy of recent memory import metadata.

        Inputs:
            limit: Requested positive maximum, clamped to the public limit of 50.

        Outputs:
            list[BenchmarkImportInfo]: At most ``limit`` safe version summaries.
        """

        bounded_limit = max(1, min(int(limit), 50))
        history = self.benchmark_import_history
        if not history and self.active_benchmark_import is not None:
            history = [self.active_benchmark_import]
        return deepcopy(history[:bounded_limit])

    def _score_benchmark_scope_exists(
        self,
        customer_class: str,
        sector: str | None,
    ) -> bool:
        """Return whether benchmark rows exist for one class and exact sector.

        Inputs:
            customer_class: Normalized customer class to inspect.
            sector: Exact sector scope, or ``None`` for fallback rows.

        Outputs:
            bool: True when at least one row exists for the requested scope.
        """
        return any(
            row.customer_class == customer_class and row.sector == sector
            for row in self.score_benchmarks
        )

    def _score_benchmark_base_exists(
        self,
        customer_class: str,
        sector: str | None,
    ) -> bool:
        """Return whether the aggregate benchmark row exists for a scope.

        Inputs:
            customer_class: Normalized customer class to inspect.
            sector: Exact sector scope, or ``None`` for the class fallback.

        Outputs:
            bool: True when the class/sector aggregate benchmark exists.
        """
        return any(
            row.customer_class == customer_class
            and row.sector == sector
            and row.dimension is None
            and row.question_id is None
            for row in self.score_benchmarks
        )

    def save_assessment_responses(
        self,
        assessment_id: str,
        customer_class: str,
        answers: dict[str, list[str]],
        source: str,
    ) -> dict[str, list[str]]:
        """Replace submitted question answers for one assessment.

        Inputs:
            assessment_id: Assessment instance being updated.
            customer_class: Normalized customer class used by the caller.
            answers: Selected answer item IDs keyed by submitted question ID.
            source: Source label for parity with the durable repository.

        Outputs:
            dict[str, list[str]]: Persisted answer IDs for submitted questions.
        """
        _ = customer_class
        _ = source
        stored = self.assessment_user_answers.setdefault(assessment_id, {})
        for question_id, answer_ids in answers.items():
            stored[question_id] = list(answer_ids)
        return {question_id: list(stored[question_id]) for question_id in answers}

    def _upsert_score_benchmark(self, benchmark: ScoreBenchmarkRow) -> None:
        """Replace any benchmark row with the same logical in-memory scope.

        Inputs:
            benchmark: Benchmark row to add or replace.

        Outputs:
            None. The in-memory benchmark list has at most one matching row.
        """
        benchmark_key = _benchmark_scope_key(benchmark)
        self.score_benchmarks = [
            row for row in self.score_benchmarks if _benchmark_scope_key(row) != benchmark_key
        ]
        self.score_benchmarks.append(benchmark)

    def get_assessment_responses(self, assessment_id: str) -> dict[str, list[str]]:
        """Return selected answer IDs keyed by question ID.

        Inputs:
            assessment_id: Assessment instance whose selections are requested.

        Outputs:
            dict[str, list[str]]: Deep-copied answer IDs keyed by question ID.
        """
        return deepcopy(self.assessment_user_answers.get(assessment_id, {}))

    def record_ai_applied_suggestion(
        self,
        task_id: str | None,
        question_id: str,
        answer_item_ids: list[str],
    ) -> None:
        """Record one AI apply event for audit parity with HANA.

        Inputs:
            task_id: Optional AI review task that produced the suggestion.
            question_id: Assessment question receiving applied answer IDs.
            answer_item_ids: Persisted answer item IDs after validation.

        Outputs:
            None. The event is appended to the in-memory audit list.
        """
        self.applied_suggestions.append(
            {
                "task_id": task_id,
                "question_id": question_id,
                "answer_item_ids": list(answer_item_ids),
            }
        )

    def ensure_mock_score_benchmarks(
        self,
        customer_class: str,
        sector: str | None,
        dimensions: list[AssessmentDimension],
        questions: list[AssessmentQuestion],
    ) -> None:
        """Seed deterministic benchmark rows for the requested scope.

        Inputs:
            customer_class: Normalized customer class that needs benchmarks.
            sector: Optional sector requested by the scoring route.
            dimensions: Framework dimensions available for scoring.
            questions: Framework questions available for scoring.

        Outputs:
            None. Mock benchmark rows are replaced once per logical scope.
        """
        for benchmark in generate_mock_score_benchmarks(
            customer_class=customer_class,
            sector=sector,
            dimensions=dimensions,
            questions=questions,
        ):
            self._upsert_score_benchmark(benchmark)

    def list_score_benchmarks(
        self,
        customer_class: str,
        sector: str | None,
    ) -> list[ScoreBenchmarkRow]:
        """Return benchmark rows matching same-size and same-sector peers.

        Inputs:
            customer_class: Normalized customer class for same-size peers.
            sector: Optional sector for same-sector peers.

        Outputs:
            list[ScoreBenchmarkRow]: Deep-copied benchmark rows for scoring.
        """
        return [
            deepcopy(row)
            for row in self.score_benchmarks
            if (row.customer_class == customer_class and row.sector is None)
            or (row.customer_class == SECTOR_BENCHMARK_CLASS and row.sector == sector)
        ]
