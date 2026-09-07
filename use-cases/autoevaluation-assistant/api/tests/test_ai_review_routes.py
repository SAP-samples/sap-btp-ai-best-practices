"""Tests for corpus-backed AI review API route behavior."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


@pytest.fixture
def ai_review_repository() -> InMemoryAiReviewRepository:
    """Create an isolated repository for AI review route tests.

    Inputs:
        None. The fixture constructs a fresh in-memory repository.

    Outputs:
        InMemoryAiReviewRepository: Repository used by dependency overrides.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        AssessmentQuestion(
            question_id="Q.STR.01.01",
            dimension="Strategy",
            section="Governance",
            question="Is strategy documented?",
            explanation="Review strategy evidence.",
            answer_items=[
                AnswerItem(
                    answer_item_id="Q.STR.01.01-L1-001",
                    question_id="Q.STR.01.01",
                    level=1,
                    item_index=1,
                    text="Strategy exists.",
                )
            ],
        ),
        AssessmentQuestion(
            question_id="Q.STR.03.01",
            dimension="Strategy",
            section="Objectives",
            question="Are objectives monitored?",
            explanation="Review objective evidence.",
            answer_items=[
                AnswerItem(
                    answer_item_id=f"Q.STR.03.01-L{level}-001",
                    question_id="Q.STR.03.01",
                    level=level,
                    item_index=1,
                    text=f"Level {level} objective evidence.",
                )
                for level in range(1, 4)
            ],
        ),
    ]
    return repository


@pytest.fixture(autouse=True)
def override_ai_review_repository(
    ai_review_repository: InMemoryAiReviewRepository,
) -> Iterator[None]:
    """Override the production HANA repository dependency for route tests.

    Inputs:
        ai_review_repository: Fresh in-memory repository fixture.

    Outputs:
        Iterator[None]: Yields while FastAPI dependency overrides are active.
    """

    def _repository_override() -> InMemoryAiReviewRepository:
        """Return the test repository for FastAPI dependency injection.

        Inputs:
            None. The repository is closed over from the pytest fixture.

        Outputs:
            InMemoryAiReviewRepository: Shared repository for one test.
        """
        return ai_review_repository

    app.dependency_overrides[get_ai_review_repository] = _repository_override
    yield
    app.dependency_overrides.pop(get_ai_review_repository, None)


def _create_corpus_review_job(
    api_client: TestClient,
    *,
    assessment_id: str = "assessment-1",
    dimension: str = "Strategy",
    question_ids: list[str] | None = None,
    current_answers: dict[str, list[str]] | None = None,
    language: str = "en",
    customer_class: str = "class_5",
):
    """Submit a JSON corpus-backed question review request.

    Inputs:
        api_client: FastAPI test client.
        assessment_id: Assessment scope for the review job.
        dimension: Framework dimension containing the selected questions.
        question_ids: Question IDs to analyze against the document corpus.
        current_answers: Current answer selections keyed by question ID.
        language: Output language for generated review text.
        customer_class: Customer class used to scope available questions.

    Outputs:
        Response: FastAPI test response for the create-job request.
    """
    return api_client.post(
        "/api/ai-review/jobs",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": assessment_id,
            "dimension": dimension,
            "question_ids": question_ids or ["Q.STR.01.01"],
            "current_answers": current_answers or {"Q.STR.01.01": []},
            "language": language,
            "customer_class": customer_class,
        },
    )


def test_create_ai_review_job_creates_corpus_task(
    api_client: TestClient,
) -> None:
    """Verify JSON analysis requests create one corpus-backed question task.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the route returns a pending job with one task.
    """
    response = _create_corpus_review_job(
        api_client,
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert response.json()["task_count"] == 1
    assert response.json()["language"] == "en"


def test_create_ai_review_job_stores_current_answers_without_attachments(
    api_client: TestClient,
    ai_review_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify question tasks use corpus documents instead of manual attachments.

    Inputs:
        api_client: FastAPI test client.
        ai_review_repository: In-memory repository used for route verification.

    Outputs:
        None. Assertions confirm the task has selected answers but no uploaded
        task-level evidence files.
    """
    response = _create_corpus_review_job(
        api_client,
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
    )

    assert response.status_code == 200

    task = ai_review_repository.list_pending_tasks(limit=1)[0]
    attachments = ai_review_repository.get_task_attachments(task["task_id"])

    assert task["current_selected_answer_item_ids"] == ["Q.STR.01.01-L1-001"]
    assert attachments == []


def test_create_ai_review_job_accepts_and_polls_italian_language(
    api_client: TestClient,
) -> None:
    """Verify AI review jobs persist the submitted generation language.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm create and polling responses expose the same
        per-job language selected by the browser at submission time.
    """
    create_response = _create_corpus_review_job(api_client, language="it")

    job_id = create_response.json()["job_id"]
    status_response = api_client.get(
        f"/api/ai-review/jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert create_response.status_code == 200
    assert create_response.json()["language"] == "it"
    assert status_response.status_code == 200
    assert status_response.json()["language"] == "it"


def test_get_ai_review_job_status_returns_progress_fields(
    api_client: TestClient,
    ai_review_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify polling exposes active corpus analysis progress for the UI.

    Inputs:
        api_client: FastAPI test client.
        ai_review_repository: In-memory repository used to simulate a worker
            updating one leased question task.

    Outputs:
        None. Assertions confirm question-level jobs return RAG progress fields.
    """
    create_response = _create_corpus_review_job(api_client)
    job_id = create_response.json()["job_id"]
    task = ai_review_repository.lease_next_task("worker-manual")
    assert task is not None
    ai_review_repository.update_question_progress(
        task_id=task["task_id"],
        worker_id="worker-manual",
        status="retrieving_evidence",
        progress_message="RAG call 1/2 retrieved 4 chunks.",
        rag_call_count=1,
        query_count=3,
        retrieved_chunk_count=4,
    )

    response = api_client.get(
        f"/api/ai-review/jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "in_progress"
    assert payload["tasks"][0]["status"] == "retrieving_evidence"
    assert payload["tasks"][0]["progress_message"] == (
        "RAG call 1/2 retrieved 4 chunks."
    )
    assert payload["tasks"][0]["rag_call_count"] == 1
    assert payload["tasks"][0]["query_count"] == 3
    assert payload["tasks"][0]["retrieved_chunk_count"] == 4


def test_create_ai_review_job_rejects_unsupported_language(
    api_client: TestClient,
) -> None:
    """Verify AI review jobs only accept supported generation languages.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm unsupported language codes fail clearly.
    """
    response = _create_corpus_review_job(api_client, language="fr")

    assert response.status_code == 400
    assert "language" in response.json()["detail"]


def test_create_ai_review_job_rejects_unknown_question_ids(
    api_client: TestClient,
) -> None:
    """Verify analysis requires at least one known question in the dimension.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm invalid question IDs return a clear bad request.
    """
    response = _create_corpus_review_job(
        api_client,
        question_ids=["Q.DOES.NOT.EXIST"],
        current_answers={},
    )

    assert response.status_code == 400
    assert "valid question ID" in response.json()["detail"]


def test_create_ai_review_job_rejects_question_unavailable_to_customer_class(
    api_client: TestClient,
) -> None:
    """Verify selected question analysis respects customer-class scope."""
    response = _create_corpus_review_job(api_client, customer_class="class_1")

    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_create_ai_review_job_filters_mixed_unavailable_selected_questions(
    api_client: TestClient,
    ai_review_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify mixed selected-question analysis keeps available questions only."""
    response = _create_corpus_review_job(
        api_client,
        question_ids=["Q.STR.01.01", "Q.STR.03.01"],
        current_answers={
            "Q.STR.01.01": ["Q.STR.01.01-L1-001"],
            "Q.STR.03.01": ["Q.STR.03.01-L2-001"],
        },
        customer_class="class_1",
    )

    assert response.status_code == 200
    assert response.json()["task_count"] == 1
    assert len(ai_review_repository.tasks) == 1
    task = next(iter(ai_review_repository.tasks.values()))
    assert task["question_id"] == "Q.STR.03.01"
    assert task["max_allowed_level"] == 2


def test_create_ai_review_job_rejects_malformed_json_body(
    api_client: TestClient,
) -> None:
    """Verify malformed JSON requests fail before any review job is created.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm invalid JSON returns FastAPI validation errors.
    """
    response = api_client.post(
        "/api/ai-review/jobs",
        headers={"X-API-Key": "test-api-key", "Content-Type": "application/json"},
        content="{not-json",
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


def test_get_ai_review_job_status_returns_pending_tasks(
    api_client: TestClient,
) -> None:
    """Verify job polling returns pending question tasks before worker results.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the polling route exposes a pending task with
        no AI result before a worker completes it.
    """
    create_response = _create_corpus_review_job(api_client)
    job_id = create_response.json()["job_id"]

    response = api_client.get(
        f"/api/ai-review/jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["task_count"] == 1
    assert payload["completed_count"] == 0
    assert payload["tasks"][0]["question_id"] == "Q.STR.01.01"
    assert payload["tasks"][0]["result"] is None


def test_get_ai_review_job_status_returns_completed_results(
    api_client: TestClient,
    ai_review_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify job polling returns persisted level-grouped AI review results.

    Inputs:
        api_client: FastAPI test client.
        ai_review_repository: In-memory repository used by route dependency
            overrides.

    Outputs:
        None. Assertions confirm polling returns a completed worker result.
    """
    create_response = _create_corpus_review_job(api_client)
    job_id = create_response.json()["job_id"]
    task = ai_review_repository.lease_next_task(worker_id="worker-1")
    assert task is not None
    ai_review_repository.save_question_result(
        task_id=task["task_id"],
        worker_id="worker-1",
        result=QuestionReviewResult(
            question_id="Q.STR.01.01",
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=["Q.STR.01.01-L1-001"],
        ),
    )

    response = api_client.get(
        f"/api/ai-review/jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["completed_count"] == 1
    assert payload["tasks"][0]["result"]["overall_status"] == "supported"


def test_delete_ai_review_jobs_clears_dimension_review_state(
    api_client: TestClient,
) -> None:
    """Verify reset route deletes persisted review data for a dimension.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the reset endpoint reports deleted work and the
        former job can no longer be polled.
    """
    create_response = _create_corpus_review_job(api_client)
    job_id = create_response.json()["job_id"]

    delete_response = api_client.delete(
        "/api/ai-review/jobs",
        headers={"X-API-Key": "test-api-key"},
        params={"assessment_id": "assessment-1", "dimension": "Strategy"},
    )
    status_response = api_client.get(
        f"/api/ai-review/jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_job_count"] == 1
    assert delete_response.json()["deleted_task_count"] == 1
    assert status_response.status_code == 404


def test_get_ai_review_job_status_returns_not_found(
    api_client: TestClient,
) -> None:
    """Verify polling an unknown job returns HTTP 404.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm missing jobs are not reported as server errors.
    """
    response = api_client.get(
        "/api/ai-review/jobs/missing-job",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 404
