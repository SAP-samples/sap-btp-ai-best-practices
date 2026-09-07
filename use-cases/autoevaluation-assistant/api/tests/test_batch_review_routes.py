"""Tests for all-question corpus AI review API routes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.routers.ai_review import get_ai_review_repository
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


@pytest.fixture
def batch_repository() -> InMemoryAiReviewRepository:
    """Create a repository with two framework questions for route tests.

    Inputs:
        None. The fixture builds an in-memory repository.

    Outputs:
        InMemoryAiReviewRepository: Repository used by FastAPI overrides.
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
def override_batch_repository(
    batch_repository: InMemoryAiReviewRepository,
) -> Iterator[None]:
    """Override the production repository for all-question route tests.

    Inputs:
        batch_repository: Fresh in-memory repository fixture.

    Outputs:
        Iterator[None]: Yields while dependency overrides are active.
    """

    def _repository_override() -> InMemoryAiReviewRepository:
        """Return the in-memory repository for FastAPI dependency injection.

        Inputs:
            None. The repository is closed over from the fixture.

        Outputs:
            InMemoryAiReviewRepository: Test repository.
        """
        return batch_repository

    app.dependency_overrides[get_ai_review_repository] = _repository_override
    yield
    app.dependency_overrides.pop(get_ai_review_repository, None)


def _create_all_questions_job(
    api_client: TestClient,
    *,
    assessment_id: str = "assessment-1",
    current_answers: dict[str, list[str]] | None = None,
    language: str = "en",
    customer_class: str = "class_5",
):
    """Submit a JSON all-question corpus review request.

    Inputs:
        api_client: FastAPI test client.
        assessment_id: Assessment scope for the review job.
        current_answers: Current answer selections keyed by question ID.
        language: Output language for generated review text.
        customer_class: Customer class used to scope available questions.

    Outputs:
        Response: FastAPI test response for the create-job request.
    """
    return api_client.post(
        "/api/ai-review/batch-jobs",
        headers={"X-API-Key": "test-api-key"},
        json={
            "assessment_id": assessment_id,
            "current_answers": current_answers or {},
            "language": language,
            "customer_class": customer_class,
        },
    )


def test_create_batch_job_creates_tasks_for_all_framework_questions(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify all-question analysis creates one task per framework question.

    Inputs:
        api_client: FastAPI test client.
        batch_repository: Repository used to inspect persisted task state.

    Outputs:
        None. Assertions confirm the response and task table use the unified
        corpus-backed review process.
    """
    response = _create_all_questions_job(
        api_client,
        language="it",
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "it"
    assert payload["status"] == "pending"
    assert payload["task_count"] == 2
    assert len(batch_repository.tasks) == 2
    assert batch_repository.batch_jobs == {}


def test_create_batch_job_scopes_tasks_and_answers_to_customer_class(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify all-question analysis skips unavailable scoped questions."""
    response = _create_all_questions_job(
        api_client,
        customer_class="class_1",
        current_answers={
            "Q.STR.01.01": ["Q.STR.01.01-L1-001"],
            "Q.STR.03.01": ["Q.STR.03.01-L2-001", "Q.STR.03.01-L3-001"],
        },
    )

    assert response.status_code == 200
    assert response.json()["task_count"] == 1
    task = next(iter(batch_repository.tasks.values()))
    assert task["question_id"] == "Q.STR.03.01"
    assert task["max_allowed_level"] == 2
    assert task["current_selected_answer_item_ids"] == ["Q.STR.03.01-L2-001"]


def test_get_batch_job_status_returns_question_tasks(api_client: TestClient) -> None:
    """Verify all-question polling returns pending tasks across dimensions.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the polling payload reuses task/result fields
        expected by the assessment UI.
    """
    create_response = _create_all_questions_job(api_client)
    job_id = create_response.json()["job_id"]

    response = api_client.get(
        f"/api/ai-review/batch-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_count"] == 2
    assert payload["batch_phase"] is None
    assert sorted(task["question_id"] for task in payload["tasks"]) == [
        "Q.STR.01.01",
        "Q.STR.03.01",
    ]


def test_create_batch_job_rejects_missing_framework_questions(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify all-question analysis requires imported framework questions.

    Inputs:
        api_client: FastAPI test client.
        batch_repository: In-memory repository whose framework is cleared.

    Outputs:
        None. Assertions confirm a clear HTTP 400 response.
    """
    batch_repository.framework_questions = []

    response = _create_all_questions_job(api_client)

    assert response.status_code == 400
    assert "framework questions" in response.json()["detail"]


def test_get_batch_job_status_returns_progress_fields(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify all-question polling exposes active RAG progress for the UI.

    Inputs:
        api_client: FastAPI test client.
        batch_repository: In-memory repository used to simulate a worker.

    Outputs:
        None. Assertions confirm unified task progress is returned by the
        backwards-compatible batch polling endpoint.
    """
    create_response = _create_all_questions_job(api_client)
    job_id = create_response.json()["job_id"]
    task = batch_repository.lease_next_task("worker-q")
    assert task is not None
    batch_repository.update_question_progress(
        task_id=task["task_id"],
        worker_id="worker-q",
        status="retrieving_evidence",
        progress_message="RAG call 1/2 retrieved 4 chunks.",
        rag_call_count=1,
        query_count=3,
        retrieved_chunk_count=4,
    )

    response = api_client.get(
        f"/api/ai-review/batch-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "in_progress"
    assert payload["indexed_chunk_count"] == 0
    assert any(task["progress_message"] for task in payload["tasks"])


def test_get_batch_job_status_returns_completed_results(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify all-question polling returns completed question results.

    Inputs:
        api_client: FastAPI test client.
        batch_repository: In-memory repository used to save a worker result.

    Outputs:
        None. Assertions confirm completed tasks are surfaced through the
        batch-job compatibility endpoint.
    """
    create_response = _create_all_questions_job(api_client)
    job_id = create_response.json()["job_id"]
    first_task = batch_repository.lease_next_task("worker-q")
    assert first_task is not None
    batch_repository.save_question_result(
        task_id=first_task["task_id"],
        worker_id="worker-q",
        result=QuestionReviewResult(
            question_id=first_task["question_id"],
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        ),
    )

    response = api_client.get(
        f"/api/ai-review/batch-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "pending"
    assert payload["completed_count"] == 1
    assert any(task["result"] for task in payload["tasks"])


def test_delete_ai_review_jobs_clears_all_question_state(
    api_client: TestClient,
    batch_repository: InMemoryAiReviewRepository,
) -> None:
    """Verify Reset Changes removes all-question corpus review state.

    Inputs:
        api_client: FastAPI test client.
        batch_repository: In-memory repository used to inspect persisted state.

    Outputs:
        None. Assertions confirm deleting with one selected dimension removes
        the global all-question job and makes it unpollable.
    """
    create_response = _create_all_questions_job(api_client)
    job_id = create_response.json()["job_id"]

    delete_response = api_client.delete(
        "/api/ai-review/jobs",
        headers={"X-API-Key": "test-api-key"},
        params={"assessment_id": "assessment-1", "dimension": "Strategy"},
    )
    status_response = api_client.get(
        f"/api/ai-review/batch-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )

    payload = delete_response.json()
    assert delete_response.status_code == 200
    assert payload["dimension"] == "all"
    assert payload["deleted_job_count"] == 1
    assert payload["deleted_task_count"] == 2
    assert status_response.status_code == 404
    assert batch_repository.jobs == {}
    assert batch_repository.tasks == {}
