"""Tests for assessment framework API route behavior."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.models.scoring import ScoreBenchmarkRow
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.hana_scoring import HanaScoringMixin
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository

SECTOR_BENCHMARK_CLASS = "__sector__"


class _FakeAssessmentRepository:
    """Provide assessment framework data for route tests.

    Inputs:
        None. Static dimension and question fixtures are returned.

    Outputs:
        Fake repository exposing the lookup methods used by assessment routes.
    """

    def __init__(self) -> None:
        """Initialize mutable response and benchmark stores for route tests."""
        self.saved_answers: dict[str, dict[str, list[str]]] = {}
        self.applied: list[dict[str, object]] = []
        self.benchmarks = [
            ScoreBenchmarkRow(
                customer_class="class_1",
                sector=None,
                benchmark_score=40.0,
            ),
            ScoreBenchmarkRow(
                customer_class="class_1",
                sector=None,
                dimension="Strategy",
                benchmark_score=45.0,
            ),
            ScoreBenchmarkRow(
                customer_class="class_1",
                sector=None,
                question_id="Q.STR.03.01",
                benchmark_score=50.0,
            ),
            ScoreBenchmarkRow(
                customer_class=SECTOR_BENCHMARK_CLASS,
                sector="Energy",
                benchmark_score=55.0,
            ),
            ScoreBenchmarkRow(
                customer_class=SECTOR_BENCHMARK_CLASS,
                sector="Energy",
                dimension="Strategy",
                benchmark_score=60.0,
            ),
            ScoreBenchmarkRow(
                customer_class=SECTOR_BENCHMARK_CLASS,
                sector="Energy",
                question_id="Q.STR.03.01",
                benchmark_score=65.0,
            ),
        ]

    def list_dimensions(self, language: str = "en") -> list[AssessmentDimension]:
        """Return fake framework dimension summaries.

        Inputs:
            language: Requested response language.

        Outputs:
            list[AssessmentDimension]: One Strategy dimension fixture.
        """
        display_name = "Strategia" if language == "it" else "Strategy"
        return [
            AssessmentDimension(
                dimension="Strategy",
                display_name=display_name,
                question_count=2,
                answered_count=0,
            )
        ]

    def list_questions(
        self,
        dimension: str,
        language: str = "en",
    ) -> list[AssessmentQuestion]:
        """Return fake questions for the requested dimension.

        Inputs:
            dimension: Framework dimension path parameter.
            language: Requested response language.

        Outputs:
            list[AssessmentQuestion]: Strategy questions with answer items.
        """
        if dimension != "Strategy":
            return []
        if language == "it":
            section = "Pianificazione Strategica"
            topic_title = "Pianificazione Strategica"
            question_text = "La tua organizzazione è dotata di una strategia AI?"
            answer_text = "La strategia AI è documentata."
        else:
            section = "Strategy and governance"
            topic_title = "Strategic Planning"
            question_text = "Does the organization define an AI strategy?"
            answer_text = "AI strategy is documented."
        return [
            AssessmentQuestion(
                question_id="Q.STR.01.01",
                dimension="Strategy",
                section=section,
                topic_title=topic_title,
                question=question_text,
                explanation="Review strategy evidence.",
                answer_items=[
                    AnswerItem(
                        answer_item_id="Q.STR.01.01-L1-001",
                        question_id="Q.STR.01.01",
                        level=1,
                        item_index=1,
                        text=answer_text,
                    )
                ],
            ),
            AssessmentQuestion(
                question_id="Q.STR.03.01",
                dimension="Strategy",
                section="Objectives",
                topic_title=(
                    "Definizione degli Obiettivi"
                    if language == "it"
                    else "Definition of Objectives"
                ),
                question="Are objectives monitored?",
                explanation="Review objective evidence.",
                answer_items=[
                    AnswerItem(
                        answer_item_id="Q.STR.03.01-L1-001",
                        question_id="Q.STR.03.01",
                        level=1,
                        item_index=1,
                        text="Objectives are defined.",
                    ),
                    AnswerItem(
                        answer_item_id="Q.STR.03.01-L2-001",
                        question_id="Q.STR.03.01",
                        level=2,
                        item_index=1,
                        text="Objectives are monitored.",
                    ),
                ],
            ),
        ]

    def list_all_questions(self, language: str = "en") -> list[AssessmentQuestion]:
        """Return all fake questions across dimensions."""
        _ = language
        return self.list_questions("Strategy", language="en")

    def save_assessment_responses(
        self,
        assessment_id: str,
        customer_class: str,
        answers: dict[str, list[str]],
        source: str,
    ) -> dict[str, list[str]]:
        """Persist filtered answer IDs in memory for route tests."""
        _ = customer_class
        _ = source
        stored = self.saved_answers.setdefault(assessment_id, {})
        for question_id, answer_ids in answers.items():
            stored[question_id] = list(answer_ids)
        return {question_id: list(stored[question_id]) for question_id in answers}

    def get_assessment_responses(self, assessment_id: str) -> dict[str, list[str]]:
        """Return persisted answer IDs for one fake assessment."""
        return {
            question_id: list(answer_ids)
            for question_id, answer_ids in self.saved_answers.get(assessment_id, {}).items()
        }

    def record_ai_applied_suggestion(
        self,
        task_id: str | None,
        question_id: str,
        answer_item_ids: list[str],
    ) -> None:
        """Record one apply event for assertions."""
        self.applied.append(
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
        """Keep pre-seeded benchmark rows for route tests."""
        _ = customer_class
        _ = sector
        _ = dimensions
        _ = questions

    def list_score_benchmarks(
        self,
        customer_class: str,
        sector: str | None,
    ) -> list[ScoreBenchmarkRow]:
        """Return benchmark rows matching same-size or same-sector peers."""
        return [
            row
            for row in self.benchmarks
            if (row.customer_class == customer_class and row.sector is None)
            or (row.customer_class == SECTOR_BENCHMARK_CLASS and row.sector == sector)
        ]


class _RecordingHanaScoringRepository(HanaScoringMixin):
    """Record HANA scoring SQL calls without connecting to HANA.

    Inputs:
        None. The object exposes ``session`` for the HANA mixin.

    Outputs:
        Test double that captures executed SQL statements and parameters.
    """

    def __init__(self) -> None:
        """Initialize the repository with a self-recording fake session."""
        self.session = self
        self.executed: list[tuple[str, list[dict[str, object]]]] = []

    def execute(self, statement: object, parameters: object | None = None) -> None:
        """Record SQL text and batch parameters from the HANA mixin.

        Inputs:
            statement: SQLAlchemy text statement.
            parameters: Optional statement parameters.

        Outputs:
            None. Calls are appended to ``executed`` for assertions.
        """
        batch_parameters = parameters if isinstance(parameters, list) else [parameters or {}]
        self.executed.append((str(statement), batch_parameters))


@pytest.fixture
def assessment_repository() -> _FakeAssessmentRepository:
    """Create a mutable fake repository for assessment route tests."""
    return _FakeAssessmentRepository()


@pytest.fixture(autouse=True)
def override_assessment_repository(
    assessment_repository: _FakeAssessmentRepository,
) -> Iterator[None]:
    """Override the assessment route repository dependency with a fake.

    Inputs:
        assessment_repository: Mutable fake repository shared by one test.

    Outputs:
        Iterator[None]: Yields while FastAPI dependency overrides are active.
    """

    def _repository_override() -> _FakeAssessmentRepository:
        """Return the fake repository for FastAPI dependency injection.

        Inputs:
            None.

        Outputs:
            _FakeAssessmentRepository: Fake assessment lookup repository.
        """
        return assessment_repository

    app.dependency_overrides[get_ai_review_repository] = _repository_override
    yield
    app.dependency_overrides.pop(get_ai_review_repository, None)


def test_list_assessment_dimensions_returns_repository_dimensions(
    api_client: TestClient,
) -> None:
    """Verify dimensions route serializes repository dimension summaries.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the fake Strategy dimension is returned.
    """
    response = api_client.get(
        "/api/assessment/dimensions",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "dimension": "Strategy",
            "display_name": "Strategy",
            "question_count": 2,
            "answered_count": 0,
        }
    ]


def test_list_assessment_dimension_questions_returns_answer_items(
    api_client: TestClient,
) -> None:
    """Verify question route serializes questions with nested answer items.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the fake Strategy question and answer item.
    """
    response = api_client.get(
        "/api/assessment/dimensions/Strategy/questions",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0] == {
        "question_id": "Q.STR.01.01",
        "dimension": "Strategy",
            "section": "Strategy and governance",
            "topic_title": "Strategic Planning",
        "question": "Does the organization define an AI strategy?",
        "explanation": "Review strategy evidence.",
        "answer_items": [
            {
                "answer_item_id": "Q.STR.01.01-L1-001",
                "question_id": "Q.STR.01.01",
                "level": 1,
                "item_index": 1,
                "text": "AI strategy is documented.",
            }
        ],
    }


def test_list_assessment_dimensions_returns_localized_display_name(
    api_client: TestClient,
) -> None:
    """Verify dimensions route passes the requested language to the repository.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm Italian display text is returned while the
        canonical dimension key remains stable.
    """
    response = api_client.get(
        "/api/assessment/dimensions?language=it",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json()[0]["dimension"] == "Strategy"
    assert response.json()[0]["display_name"] == "Strategia"


def test_list_assessment_dimension_questions_returns_localized_text(
    api_client: TestClient,
) -> None:
    """Verify question route passes the requested language to the repository.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm Italian question and answer text is returned.
    """
    response = api_client.get(
        "/api/assessment/dimensions/Strategy/questions?language=it",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()[0]
    assert payload["section"] == "Pianificazione Strategica"
    assert payload["question"] == "La tua organizzazione è dotata di una strategia AI?"
    assert payload["answer_items"][0]["text"] == "La strategia AI è documentata."


def test_list_assessment_questions_rejects_unsupported_language(
    api_client: TestClient,
) -> None:
    """Verify unsupported framework languages fail before repository lookup.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm only supported language codes are accepted.
    """
    response = api_client.get(
        "/api/assessment/dimensions/Strategy/questions?language=fr",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 400
    assert "language" in response.json()["detail"]


def test_get_customer_class_scope_returns_editable_config(
    api_client: TestClient,
) -> None:
    """Verify the assessment API exposes the backend class-scope config.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm UI clients can fetch labels and max levels.
    """
    response = api_client.get(
        "/api/assessment/customer-class-scope",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["default_customer_class"] == "class_5"
    assert payload["classes"]["class_1"]["labels"]["en"] == "Micro - Class 1"
    assert payload["question_max_levels"]["Q.STR.03.01"]["class_1"] == 2


def test_save_assessment_responses_replaces_submitted_question_answers(
    api_client: TestClient,
) -> None:
    """Verify response save route removes stale answer IDs for a question."""
    first_response = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_5",
            "answers": {
                "Q.STR.03.01": [
                    "Q.STR.03.01-L1-001",
                    "Q.STR.03.01-L2-001",
                ]
            },
        },
    )
    response = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_5",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L2-001"]},
        },
    )

    assert first_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["answers"] == {"Q.STR.03.01": ["Q.STR.03.01-L2-001"]}


def test_save_assessment_responses_deduplicates_answer_ids(
    api_client: TestClient,
) -> None:
    """Verify duplicate submitted answer IDs are persisted once in order."""
    response = api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_5",
            "answers": {
                "Q.STR.03.01": [
                    "Q.STR.03.01-L1-001",
                    "Q.STR.03.01-L1-001",
                    "Q.STR.03.01-L2-001",
                ]
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"] == {
        "Q.STR.03.01": [
            "Q.STR.03.01-L1-001",
            "Q.STR.03.01-L2-001",
        ]
    }


def test_memory_score_benchmark_seeding_adds_sector_and_size_peers() -> None:
    """Verify mock seeding creates separate sector and class peer rows."""
    repository = InMemoryAiReviewRepository()
    fake_repository = _FakeAssessmentRepository()
    dimensions = fake_repository.list_dimensions()
    questions = fake_repository.list_all_questions()

    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector=None,
        dimensions=dimensions,
        questions=questions,
    )
    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
    )

    benchmarks = repository.list_score_benchmarks("class_1", "Energy")
    same_size = [
        row
        for row in benchmarks
        if row.customer_class == "class_1" and row.sector is None
    ]
    same_sector = [
        row
        for row in benchmarks
        if row.customer_class == SECTOR_BENCHMARK_CLASS and row.sector == "Energy"
    ]

    assert any(row.dimension is None and row.question_id is None for row in same_size)
    assert any(row.dimension is None and row.question_id is None for row in same_sector)
    assert any(row.question_id == "Q.STR.03.01" for row in same_size)
    assert any(row.question_id == "Q.STR.03.01" for row in same_sector)
    assert {row.benchmark_score for row in same_size} != {row.benchmark_score for row in same_sector}


def test_memory_score_benchmark_seeding_is_logically_idempotent() -> None:
    """Verify repeated benchmark seeding keeps one row per logical scope."""
    repository = InMemoryAiReviewRepository()
    fake_repository = _FakeAssessmentRepository()
    dimensions = fake_repository.list_dimensions()
    questions = fake_repository.list_all_questions()

    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
    )
    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
    )

    logical_scopes = [
        (row.customer_class, row.sector, row.dimension, row.question_id)
        for row in repository.score_benchmarks
    ]
    assert len(logical_scopes) == len(set(logical_scopes))


def test_hana_score_benchmark_seeding_uses_stable_upserts() -> None:
    """Verify HANA mock benchmark seeding is deterministic and idempotent."""
    repository = _RecordingHanaScoringRepository()
    fake_repository = _FakeAssessmentRepository()
    dimensions = fake_repository.list_dimensions()
    questions = fake_repository.list_all_questions()

    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
    )
    first_call_ids = [row["benchmark_id"] for row in repository.executed[0][1]]
    repository.ensure_mock_score_benchmarks(
        customer_class="class_1",
        sector="Energy",
        dimensions=dimensions,
        questions=questions,
    )
    second_call_ids = [row["benchmark_id"] for row in repository.executed[1][1]]

    assert "upsert assessment_score_benchmarks" in repository.executed[0][0].lower()
    assert "with primary key" in repository.executed[0][0].lower()
    assert first_call_ids == second_call_ids
    assert len(first_call_ids) == len(set(first_call_ids))
    assert any(row["customer_class"] == SECTOR_BENCHMARK_CLASS for row in repository.executed[0][1])
    assert any(row["customer_class"] == "class_1" and row["sector"] is None for row in repository.executed[0][1])


def test_get_assessment_responses_returns_empty_object_for_new_assessment(
    api_client: TestClient,
) -> None:
    """Verify unknown assessment responses are an empty persisted state."""
    response = api_client.get(
        "/api/assessment/responses?assessment_id=assessment-1",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json()["answers"] == {}


def test_apply_ai_marks_replaces_question_answers_and_records_apply(
    api_client: TestClient,
    assessment_repository: _FakeAssessmentRepository,
) -> None:
    """Verify AI apply persists answer IDs and stores an apply audit event."""
    response = api_client.post(
        "/api/assessment/responses/apply-ai",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_5",
            "question_id": "Q.STR.01.01",
            "task_id": "task-1",
            "answer_item_ids": ["Q.STR.01.01-L1-001"],
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"] == {"Q.STR.01.01": ["Q.STR.01.01-L1-001"]}
    assert assessment_repository.applied == [
        {
            "task_id": "task-1",
            "question_id": "Q.STR.01.01",
            "answer_item_ids": ["Q.STR.01.01-L1-001"],
        }
    ]


def test_get_assessment_score_uses_persisted_answers(api_client: TestClient) -> None:
    """Verify score route calculates from repository-persisted selections."""
    api_client.put(
        "/api/assessment/responses",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": "assessment-1",
            "customer_class": "class_1",
            "answers": {"Q.STR.03.01": ["Q.STR.03.01-L2-001"]},
        },
    )

    response = api_client.get(
        "/api/assessment/score?assessment_id=assessment-1&customer_class=class_1&sector=Energy",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["final_score"] == 50.0
    assert payload["benchmark_context"]["reason"] == "no_active_dataset"
    assert payload["benchmark_score"] is None
    assert payload["same_sector_benchmark_score"] is None
    assert payload["same_size_benchmark_score"] is None
    scorable_question = next(
        question
        for question in payload["questions"]
        if question["question_id"] == "Q.STR.03.01"
    )
    assert scorable_question["question_text"] == "Are objectives monitored?"
    assert scorable_question["selected_answer_item_ids"] == ["Q.STR.03.01-L2-001"]
    assert scorable_question["benchmark_score"] is None
    assert scorable_question["same_sector_benchmark_score"] is None
    assert scorable_question["same_size_benchmark_score"] is None
