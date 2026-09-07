"""HANA persistence for assessment responses and score benchmarks."""

from __future__ import annotations

import json
from collections import OrderedDict
from hashlib import sha1
from uuid import uuid4

from sqlalchemy import text

from app.models.assessment import AssessmentDimension, AssessmentQuestion
from app.models.benchmarking import (
    AssessmentProfile,
    BenchmarkCohortOption,
    BenchmarkImportInfo,
    BenchmarkPeerSubmission,
    BenchmarkScopeScore,
)
from app.models.scoring import ScoreBenchmarkRow
from app.services.score_benchmarks import (
    SECTOR_BENCHMARK_CLASS,
    generate_mock_score_benchmarks,
)


def _stable_benchmark_id(
    customer_class: str,
    sector: str | None,
    dimension: str | None,
    question_id: str | None,
) -> str:
    """Return a deterministic benchmark primary key for one logical scope.

    Inputs:
        customer_class: Normalized customer class.
        sector: Optional sector scope.
        dimension: Optional dimension benchmark scope.
        question_id: Optional question benchmark scope.

    Outputs:
        str: Stable HANA-safe benchmark ID within the table column length.
    """
    scope = json.dumps(
        [customer_class, sector, dimension, question_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"benchmark-{sha1(scope.encode('utf-8')).hexdigest()}"


class HanaScoringMixin:
    """Persist assessment user responses, apply audit, and benchmarks in HANA."""

    def upsert_assessment_profile(
        self,
        profile: AssessmentProfile,
    ) -> AssessmentProfile:
        """Insert or update authoritative assessment class/NACE profile context.

        Inputs:
            profile: Validated assessment profile to persist.

        Outputs:
            AssessmentProfile: Profile read back from HANA after the upsert.
        """

        self.session.execute(
            text(
                "upsert assessment_profiles ("
                "assessment_id, display_name, source_company_id, customer_class, "
                "nace1, updated_at"
                ") values ("
                ":assessment_id, :display_name, :source_company_id, :customer_class, "
                ":nace1, current_utctimestamp"
                ") with primary key"
            ),
            profile.model_dump(),
        )
        persisted = self.get_assessment_profile(profile.assessment_id)
        if persisted is None:  # pragma: no cover - defensive database boundary
            raise RuntimeError("Assessment profile upsert did not persist a row")
        return persisted

    def get_assessment_profile(
        self,
        assessment_id: str,
    ) -> AssessmentProfile | None:
        """Load one authoritative assessment profile from HANA.

        Inputs:
            assessment_id: Assessment identity whose context should be loaded.

        Outputs:
            AssessmentProfile | None: Persisted profile, or ``None`` when absent.
        """

        row = self.session.execute(
            text(
                "select assessment_id, display_name, source_company_id, "
                "customer_class, nace1 from assessment_profiles "
                "where assessment_id = :assessment_id"
            ),
            {"assessment_id": assessment_id},
        ).mappings().first()
        return None if row is None else AssessmentProfile.model_validate(row)

    def get_active_benchmark_import(self) -> BenchmarkImportInfo | None:
        """Return safe metadata for the currently active imported dataset.

        Inputs:
            None.

        Outputs:
            BenchmarkImportInfo | None: Most recently activated active version,
            excluding the workbook BLOB and every peer identity.
        """

        row = self.session.execute(
            text(
                "select top 1 import_id, source_filename, source_sha256, "
                "scoring_version, row_count, company_count, questionnaire_count, "
                "question_count, accepted_count, rejected_count, status, is_active, "
                "created_at, activated_at "
                "from assessment_benchmark_imports where is_active = 1 "
                "order by activated_at desc, created_at desc"
            )
        ).mappings().first()
        return None if row is None else BenchmarkImportInfo.model_validate(row)

    def list_benchmark_peer_submissions(
        self,
        import_id: str,
        customer_class: str,
        nace1: str,
    ) -> list[BenchmarkPeerSubmission]:
        """Load score-bearing submissions from one exact active-version cohort.

        Inputs:
            import_id: Active version identity captured once for the request.
            customer_class: Exact normalized assessment class filter.
            nace1: Exact persisted NACE-1 filter.

        Outputs:
            list[BenchmarkPeerSubmission]: Submission metadata grouped with all
            overall/dimension/topic calculated scores for pure latest selection.
        """

        rows = self.session.execute(
            text(
                "select s.source_company_id, s.questionnaire_id, "
                "c.customer_class, c.nace1, s.submission_date, s.extraction_date, "
                "s.release_status, sc.scope_type, sc.dimension, sc.question_id, "
                "sc.calculated_score "
                "from assessment_benchmark_submissions s "
                "join assessment_benchmark_companies c "
                "on c.import_id = s.import_id "
                "and c.source_company_id = s.source_company_id "
                "join assessment_benchmark_scores sc "
                "on sc.import_id = s.import_id "
                "and sc.questionnaire_id = s.questionnaire_id "
                "where s.import_id = :import_id "
                "and c.customer_class = :customer_class "
                "and c.nace1 = :nace1 "
                "order by s.source_company_id, s.questionnaire_id, "
                "sc.scope_type, sc.dimension, sc.question_id"
            ),
            {
                "import_id": import_id,
                "customer_class": customer_class,
                "nace1": nace1,
            },
        ).mappings().all()

        grouped: OrderedDict[tuple[str, str], dict[str, object]] = OrderedDict()
        for row in rows:
            key = (str(row["source_company_id"]), str(row["questionnaire_id"]))
            record = grouped.setdefault(
                key,
                {
                    "source_company_id": key[0],
                    "questionnaire_id": key[1],
                    "customer_class": str(row["customer_class"]),
                    "nace1": row["nace1"],
                    "submission_date": row["submission_date"],
                    "extraction_date": row["extraction_date"],
                    "release_status": row["release_status"],
                    "scores": [],
                },
            )
            scores = record["scores"]
            if isinstance(scores, list):
                scores.append(
                    BenchmarkScopeScore(
                        scope_type=str(row["scope_type"]),
                        dimension=row["dimension"],
                        question_id=row["question_id"],
                        calculated_score=float(row["calculated_score"]),
                    )
                )
        return [BenchmarkPeerSubmission.model_validate(record) for record in grouped.values()]

    def list_benchmark_cohort_options(
        self,
        import_id: str,
    ) -> list[BenchmarkCohortOption]:
        """Return sorted identity-free exact cohort pairs for one active version.

        Inputs:
            import_id: Captured active benchmark import identity.

        Outputs:
            list[BenchmarkCohortOption]: Distinct class/NACE pairs without peers.
        """

        rows = self.session.execute(
            text(
                "select distinct customer_class, nace1 "
                "from assessment_benchmark_companies "
                "where import_id = :import_id "
                "and nace1 is not null and trim(nace1) <> '' "
                "order by customer_class, nace1"
            ),
            {"import_id": import_id},
        ).mappings().all()
        return [BenchmarkCohortOption.model_validate(row) for row in rows]

    def list_recent_benchmark_imports(
        self,
        limit: int,
    ) -> list[BenchmarkImportInfo]:
        """Return bounded safe recent import summaries without workbook BLOBs.

        Inputs:
            limit: Requested positive maximum, clamped to 50 before SQL creation.

        Outputs:
            list[BenchmarkImportInfo]: Most recent version metadata only.
        """

        bounded_limit = max(1, min(int(limit), 50))
        rows = self.session.execute(
            text(
                f"select top {bounded_limit} import_id, source_filename, "
                "source_sha256, scoring_version, row_count, company_count, "
                "questionnaire_count, question_count, accepted_count, "
                "rejected_count, status, is_active, created_at, activated_at "
                "from assessment_benchmark_imports "
                "order by created_at desc, import_id desc"
            )
        ).mappings().all()
        return [BenchmarkImportInfo.model_validate(row) for row in rows]

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
        sector_filter = "sector is null" if sector is None else "sector = :sector"
        row = self.session.execute(
            text(
                "select top 1 benchmark_id "
                "from assessment_score_benchmarks "
                "where customer_class = :customer_class "
                f"and {sector_filter}"
            ),
            {"customer_class": customer_class, "sector": sector},
        ).first()
        return row is not None

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
        sector_filter = "sector is null" if sector is None else "sector = :sector"
        row = self.session.execute(
            text(
                "select top 1 benchmark_id "
                "from assessment_score_benchmarks "
                "where customer_class = :customer_class "
                f"and {sector_filter} "
                "and dimension is null "
                "and question_id is null"
            ),
            {"customer_class": customer_class, "sector": sector},
        ).first()
        return row is not None

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
            customer_class: Normalized customer class stored with selections.
            answers: Selected answer item IDs keyed by submitted question ID.
            source: Source label such as ``manual`` or ``ai_apply``.

        Outputs:
            dict[str, list[str]]: Persisted answer IDs for submitted questions.
        """
        # The persisted profile class is authoritative for the whole assessment,
        # including rows written before a profile class change.
        self.session.execute(
            text(
                "update assessment_user_answers set customer_class = :customer_class, "
                "updated_at = current_utctimestamp "
                "where assessment_id = :assessment_id"
            ),
            {
                "assessment_id": assessment_id,
                "customer_class": customer_class,
            },
        )
        for question_id, answer_ids in answers.items():
            self.session.execute(
                text(
                    "delete from assessment_user_answers "
                    "where assessment_id = :assessment_id "
                    "and question_id = :question_id"
                ),
                {"assessment_id": assessment_id, "question_id": question_id},
            )
            for answer_id in answer_ids:
                self.session.execute(
                    text(
                        "insert into assessment_user_answers "
                        "(assessment_id, question_id, answer_item_id, customer_class, source) "
                        "values (:assessment_id, :question_id, :answer_item_id, "
                        ":customer_class, :source)"
                    ),
                    {
                        "assessment_id": assessment_id,
                        "question_id": question_id,
                        "answer_item_id": answer_id,
                        "customer_class": customer_class,
                        "source": source,
                    },
                )
        return {question_id: list(answer_ids) for question_id, answer_ids in answers.items()}

    def get_assessment_responses(self, assessment_id: str) -> dict[str, list[str]]:
        """Return selected answer IDs keyed by question ID.

        Inputs:
            assessment_id: Assessment instance whose selections are requested.

        Outputs:
            dict[str, list[str]]: Stored answer IDs ordered by question and item.
        """
        rows = self.session.execute(
            text(
                "select question_id, answer_item_id "
                "from assessment_user_answers "
                "where assessment_id = :assessment_id "
                "order by question_id, answer_item_id"
            ),
            {"assessment_id": assessment_id},
        ).mappings().all()
        answers: dict[str, list[str]] = {}
        for row in rows:
            answers.setdefault(str(row["question_id"]), []).append(str(row["answer_item_id"]))
        return answers

    def record_ai_applied_suggestion(
        self,
        task_id: str | None,
        question_id: str,
        answer_item_ids: list[str],
    ) -> None:
        """Record one AI apply event in the existing audit table.

        Inputs:
            task_id: Optional AI review task that produced the suggestion.
            question_id: Assessment question receiving applied answer IDs.
            answer_item_ids: Persisted answer item IDs after validation.

        Outputs:
            None. Rows are inserted only when a task ID is available.
        """
        if not task_id:
            return
        self.session.execute(
            text(
                "insert into ai_applied_suggestions "
                "(applied_id, task_id, question_id, applied_answer_item_ids_json) "
                "values (:applied_id, :task_id, :question_id, :answer_ids)"
            ),
            {
                "applied_id": f"applied-{uuid4()}",
                "task_id": task_id,
                "question_id": question_id,
                "answer_ids": json.dumps(answer_item_ids, ensure_ascii=False),
            },
        )

    def ensure_mock_score_benchmarks(
        self,
        customer_class: str,
        sector: str | None,
        dimensions: list[AssessmentDimension],
        questions: list[AssessmentQuestion],
    ) -> None:
        """Seed deterministic mock benchmark rows in HANA for the requested scope.

        Inputs:
            customer_class: Normalized customer class that needs benchmarks.
            sector: Optional sector requested by the scoring route.
            dimensions: Framework dimensions available for scoring.
            questions: Framework questions available for scoring.

        Outputs:
            None. Mock benchmark rows are upserted once per logical scope.
        """
        rows = [
            {
                "benchmark_id": _stable_benchmark_id(
                    benchmark.customer_class,
                    benchmark.sector,
                    benchmark.dimension,
                    benchmark.question_id,
                ),
                "customer_class": benchmark.customer_class,
                "sector": benchmark.sector,
                "dimension": benchmark.dimension,
                "question_id": benchmark.question_id,
                "benchmark_score": benchmark.benchmark_score,
                "sample_size": benchmark.sample_size,
            }
            for benchmark in generate_mock_score_benchmarks(
                customer_class=customer_class,
                sector=sector,
                dimensions=dimensions,
                questions=questions,
            )
        ]
        self.session.execute(
            text(
                "upsert assessment_score_benchmarks "
                "(benchmark_id, customer_class, sector, dimension, question_id, "
                "benchmark_score, sample_size) "
                "values (:benchmark_id, :customer_class, :sector, :dimension, "
                ":question_id, :benchmark_score, :sample_size) "
                "with primary key"
            ),
            rows,
        )

    def list_score_benchmarks(
        self,
        customer_class: str,
        sector: str | None,
    ) -> list[ScoreBenchmarkRow]:
        """Return benchmark rows matching class and exact or fallback sector.

        Inputs:
            customer_class: Normalized customer class filter.
            sector: Optional sector filter.

        Outputs:
            list[ScoreBenchmarkRow]: Benchmark rows converted to scoring models.
        """
        rows = self.session.execute(
            text(
                "select customer_class, sector, dimension, question_id, "
                "benchmark_score, sample_size "
                "from assessment_score_benchmarks "
                "where (customer_class = :customer_class and sector is null) "
                "or (customer_class = :sector_benchmark_class and sector = :sector) "
                "order by sample_size desc"
            ),
            {
                "customer_class": customer_class,
                "sector": sector,
                "sector_benchmark_class": SECTOR_BENCHMARK_CLASS,
            },
        ).mappings().all()
        return [
            ScoreBenchmarkRow(
                customer_class=str(row["customer_class"]),
                sector=row["sector"],
                dimension=row["dimension"],
                question_id=row["question_id"],
                benchmark_score=float(row["benchmark_score"]),
                sample_size=int(row["sample_size"] or 0),
            )
            for row in rows
        ]
