"""API routes for creating AI document review jobs."""

from collections.abc import Generator
from functools import lru_cache
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from app.db import create_hana_engine, is_transient_database_error
from app.models.ai_review import (
    CorpusBatchReviewJobRequest,
    CorpusReviewJobRequest,
    ReviewJobResponse,
    ReviewJobStatusResponse,
    ReviewResetResponse,
)
from app.security import get_api_key
from app.services.ai_review_repository.hana import HanaAiReviewRepository

router = APIRouter(dependencies=[Depends(get_api_key)])

@lru_cache(maxsize=1)
def get_hana_session_factory() -> sessionmaker[Session]:
    """Return a cached SQLAlchemy session factory for HANA-backed routes.

    Inputs:
        None. HANA connection settings are read by ``create_hana_engine`` from
        process environment variables.

    Outputs:
        sessionmaker[Session]: Session factory bound to the cached HANA engine.
    """
    return sessionmaker(bind=create_hana_engine())


def get_ai_review_repository() -> Generator[HanaAiReviewRepository, None, None]:
    """Yield the default HANA AI review repository for API requests.

    Inputs:
        None. The dependency creates one SQLAlchemy session from the cached HANA
        session factory.

    Outputs:
        Generator[HanaAiReviewRepository, None, None]: Repository yielded to
        route handlers. The session is committed after successful route
        execution, rolled back on errors, and always closed.
    """
    session = get_hana_session_factory()()
    try:
        yield HanaAiReviewRepository(session)
        session.commit()
    except DBAPIError as exc:
        session.rollback()
        if is_transient_database_error(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The HANA database is temporarily unavailable. Retry shortly."
                ),
                headers={"Retry-After": "5"},
            ) from exc
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

@router.post("/jobs", response_model=ReviewJobResponse)
async def create_ai_review_job(
    request: CorpusReviewJobRequest,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> ReviewJobResponse:
    """Create a corpus-backed question-level AI review job.

    Inputs:
        request: JSON body containing assessment, dimension, question IDs,
            current answers, and language.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        ReviewJobResponse: Created job ID, initial status, and task count.
    """
    try:
        return repository.create_corpus_question_job(
            assessment_id=request.assessment_id,
            dimension=request.dimension,
            question_ids=request.question_ids,
            current_answers=request.current_answers,
            language=request.language,
            customer_class=request.customer_class,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/batch-jobs", response_model=ReviewJobResponse)
async def create_batch_ai_review_job(
    request: CorpusBatchReviewJobRequest,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> ReviewJobResponse:
    """Create an all-question AI review job from the document corpus.

    Inputs:
        request: JSON body containing assessment, current answers, and language.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        ReviewJobResponse: Created job ID, initial status, and task count.
    """
    try:
        return repository.create_corpus_all_questions_job(
            assessment_id=request.assessment_id,
            current_answers=request.current_answers,
            language=request.language,
            customer_class=request.customer_class,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/batch-jobs/{job_id}", response_model=ReviewJobStatusResponse)
async def get_batch_ai_review_job_status(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> ReviewJobStatusResponse:
    """Return current progress and completed question results for a batch job.

    Inputs:
        job_id: Batch AI review job identifier returned by
            ``POST /batch-jobs``.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        ReviewJobStatusResponse: Derived job status and per-question results.

    Raises:
        HTTPException: Raised with HTTP 404 when the batch job does not exist.
    """
    try:
        return repository.get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get("/jobs/{job_id}", response_model=ReviewJobStatusResponse)
async def get_ai_review_job_status(
    job_id: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> ReviewJobStatusResponse:
    """Return current worker progress and completed question results for a job.

    Inputs:
        job_id: AI review job identifier returned by ``POST /jobs``.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        ReviewJobStatusResponse: Derived job status, task counts, and completed
        level-grouped AI review results.

    Raises:
        HTTPException: Raised with HTTP 404 when the job does not exist.
    """
    try:
        return repository.get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete("/jobs", response_model=ReviewResetResponse)
async def clear_ai_review_dimension_state(
    assessment_id: Annotated[str, Query()],
    dimension: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> ReviewResetResponse:
    """Clear all stored AI review process state for one assessment.

    Inputs:
        assessment_id: Assessment instance whose persisted legacy and batch
            review state should be cleared.
        dimension: Currently selected framework dimension. The parameter is
            accepted for backward compatibility with existing clients; reset is
            intentionally assessment-wide.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        ReviewResetResponse: Counts of deleted persisted jobs and tasks.
    """
    _ = dimension
    return repository.clear_assessment_review_state(assessment_id=assessment_id)
