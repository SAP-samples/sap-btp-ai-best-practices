"""API routes for assessment framework dimensions and questions."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.assessment import AssessmentDimension, AssessmentQuestion
from app.models.benchmarking import (
    AssessmentProfile,
    BenchmarkOptionsResponse,
)
from app.models.language import DEFAULT_LANGUAGE, normalize_language
from app.models.scoring import (
    ApplyAiMarksRequest,
    AssessmentResponsesRequest,
    AssessmentResponsesResponse,
    AssessmentScoreResponse,
)
from app.routers.ai_review import get_ai_review_repository
from app.security import get_api_key
from app.services.assessment_scoring import calculate_assessment_score
from app.services.customer_class_scope import (
    filter_current_answer_ids,
    load_customer_class_scope,
    max_allowed_level,
    normalize_customer_class,
    require_customer_class,
)

router = APIRouter(dependencies=[Depends(get_api_key)])


def _repository_profile(repository: Any, assessment_id: str) -> AssessmentProfile | None:
    """Load a profile when the repository implements the current interface.

    Inputs:
        repository: Assessment repository supplied by dependency injection.
        assessment_id: Assessment identity whose profile is requested.

    Outputs:
        AssessmentProfile | None: Persisted profile, or ``None`` for legacy test
        repositories/assessments without profile context.
    """

    loader = getattr(repository, "get_assessment_profile", None)
    return loader(assessment_id) if callable(loader) else None


def _effective_customer_class(
    profile: AssessmentProfile | None,
    requested_class: str | None,
) -> str:
    """Resolve class context with a persisted profile as the authority.

    Inputs:
        profile: Optional persisted assessment profile.
        requested_class: Optional legacy request/query class alias.

    Outputs:
        str: Exact profile class, or legacy normalized/default class when no
        profile exists.

    Raises:
        HTTPException: HTTP 409 when an explicit valid class conflicts with the
        persisted profile.
        ValueError: When an explicit profile-bound class is unsupported.
    """

    if profile is None:
        return normalize_customer_class(requested_class)
    if requested_class is not None:
        normalized_requested = require_customer_class(requested_class)
        if normalized_requested != profile.customer_class:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Requested customer_class conflicts with the persisted "
                    "assessment profile."
                ),
            )
    return profile.customer_class


def _effective_nace1(
    profile: AssessmentProfile | None,
    requested_sector: str | None,
) -> str | None:
    """Resolve legacy sector display input without overriding profile NACE-1.

    Inputs:
        profile: Optional persisted authoritative profile.
        requested_sector: Optional legacy score-query sector alias.

    Outputs:
        str | None: Exact persisted NACE-1, or trimmed legacy display value when
        no profile exists. Legacy-only values never enable benchmark cohorts.

    Raises:
        HTTPException: HTTP 409 when an explicit sector differs from profile
        NACE-1.
    """

    normalized_sector = requested_sector.strip() if requested_sector else None
    if profile is None:
        return normalized_sector
    if normalized_sector is not None and normalized_sector != profile.nace1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Requested sector conflicts with the persisted assessment profile.",
        )
    return profile.nace1


def _normalized_profile(profile: AssessmentProfile) -> AssessmentProfile:
    """Trim and strictly validate an incoming profile before persistence.

    Inputs:
        profile: Request body parsed into the shared profile model.

    Outputs:
        AssessmentProfile: Clean non-empty text and exact configured class.

    Raises:
        ValueError: If required text is blank or the class is unsupported.
    """

    assessment_id = profile.assessment_id.strip()
    display_name = profile.display_name.strip()
    nace1 = profile.nace1.strip()
    if not assessment_id:
        raise ValueError("assessment_id must not be blank")
    if not display_name:
        raise ValueError("display_name must not be blank")
    if not nace1:
        raise ValueError("nace1 must not be blank")
    source_company_id = (
        profile.source_company_id.strip() if profile.source_company_id else None
    )
    return AssessmentProfile(
        assessment_id=assessment_id,
        display_name=display_name,
        source_company_id=source_company_id or None,
        customer_class=require_customer_class(profile.customer_class),
        nace1=nace1,
    )


@router.get("/customer-class-scope")
def get_customer_class_scope() -> dict[str, Any]:
    """Return editable customer-class scope rules for the Assessment UI.

    Inputs:
        None. The route reads the cached backend JSON config.

    Outputs:
        dict[str, Any]: JSON-compatible class labels and max-level rules.
    """
    return load_customer_class_scope()


@router.get("/profile", response_model=AssessmentProfile)
def get_assessment_profile(
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentProfile:
    """Return persisted company/cohort context for one assessment.

    Inputs:
        assessment_id: Assessment identity whose profile should be returned.
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        AssessmentProfile: Persisted display/company/class/NACE context.

    Raises:
        HTTPException: HTTP 404 when the assessment has no profile.
    """

    profile = _repository_profile(repository, assessment_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment profile not found.",
        )
    return profile


@router.get("/benchmark-options", response_model=BenchmarkOptionsResponse)
def get_benchmark_options(
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> BenchmarkOptionsResponse:
    """Return identity-free class/NACE options from the active imported dataset.

    Inputs:
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        BenchmarkOptionsResponse: Useful unavailable state or sorted exact
        cohort pairs plus their independent selector label lists.
    """

    active_import = repository.get_active_benchmark_import()
    if active_import is None:
        return BenchmarkOptionsResponse(
            available=False,
            reason="no_active_dataset",
        )
    cohorts = repository.list_benchmark_cohort_options(active_import.import_id)
    if not cohorts:
        return BenchmarkOptionsResponse(
            available=False,
            reason="no_cohort_options",
            import_id=active_import.import_id,
        )
    return BenchmarkOptionsResponse(
        available=True,
        reason=None,
        import_id=active_import.import_id,
        customer_classes=sorted({item.customer_class for item in cohorts}),
        nace1_sectors=sorted({item.nace1 for item in cohorts}),
        cohorts=cohorts,
    )


@router.put("/profile", response_model=AssessmentProfile)
def put_assessment_profile(
    request: AssessmentProfile,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentProfile:
    """Validate and persist authoritative assessment benchmark context.

    Inputs:
        request: Assessment identity, display/company identity, exact class, and
        exact NACE-1 label.
        repository: HANA-backed repository supplied by dependency injection.

    Outputs:
        AssessmentProfile: Normalized persisted profile. Existing answers are
        revalidated against its class in the same request transaction.
    """

    try:
        normalized_profile = _normalized_profile(request)
        existing_answers = repository.get_assessment_responses(
            normalized_profile.assessment_id
        )
        validated_answers = _validated_answers(
            repository,
            existing_answers,
            normalized_profile.customer_class,
        )
        persisted_profile = repository.upsert_assessment_profile(normalized_profile)
        if existing_answers:
            # A narrower profile must remove selections that are no longer in
            # scope; otherwise response reads could expose stale higher levels.
            repository.save_assessment_responses(
                assessment_id=normalized_profile.assessment_id,
                customer_class=normalized_profile.customer_class,
                answers=validated_answers,
                source="profile_revalidation",
            )
        return persisted_profile
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


def _validated_answers(
    repository: Any,
    answers: dict[str, list[str]],
    customer_class: str,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, list[str]]:
    """Validate and filter response answer IDs against the framework.

    Inputs:
        repository: Assessment repository exposing framework question lookups.
        answers: Submitted selected answer IDs keyed by question ID.
        customer_class: Normalized customer class used for max-level filtering.
        language: Framework language for ``list_all_questions``.

    Outputs:
        dict[str, list[str]]: Answer IDs that belong to known questions and are
        allowed for the requested customer class.

    Raises:
        ValueError: Raised when an answer payload references an unknown
        assessment question ID.
    """
    questions = {
        question.question_id: question
        for question in repository.list_all_questions(language=language)
    }
    validated: dict[str, list[str]] = {}
    for question_id, answer_ids in answers.items():
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"Unknown assessment question ID: {question_id}")
        max_level = max_allowed_level(question_id, customer_class)
        filtered_answer_ids = filter_current_answer_ids(
            question,
            answer_ids,
            max_level,
        )
        seen_answer_ids: set[str] = set()
        validated[question_id] = []
        for answer_id in filtered_answer_ids:
            if answer_id in seen_answer_ids:
                continue
            seen_answer_ids.add(answer_id)
            validated[question_id].append(answer_id)
    return validated


@router.get("/dimensions", response_model=list[AssessmentDimension])
def list_dimensions(
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    language: Annotated[str, Query()] = DEFAULT_LANGUAGE,
) -> list[AssessmentDimension]:
    """Return assessment framework dimensions available for review.

    Inputs:
        repository: AI review repository that exposes framework lookup methods.
        language: Requested response language, either ``en`` or ``it``.

    Outputs:
        list[AssessmentDimension]: Dimension summaries ordered by repository
        implementation.
    """
    try:
        normalized_language = normalize_language(language)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return repository.list_dimensions(language=normalized_language)


@router.get("/dimensions/{dimension}/questions", response_model=list[AssessmentQuestion])
def list_questions(
    dimension: str,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    language: Annotated[str, Query()] = DEFAULT_LANGUAGE,
) -> list[AssessmentQuestion]:
    """Return assessment questions and answer items for one dimension.

    Inputs:
        dimension: Framework dimension path parameter.
        repository: AI review repository that exposes framework lookup methods.
        language: Requested response language, either ``en`` or ``it``.

    Outputs:
        list[AssessmentQuestion]: Questions with nested answer items for the
        requested dimension.
    """
    try:
        normalized_language = normalize_language(language)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return repository.list_questions(dimension, language=normalized_language)


@router.put("/responses", response_model=AssessmentResponsesResponse)
def save_assessment_responses(
    request: AssessmentResponsesRequest,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentResponsesResponse:
    """Persist user questionnaire answer selections in HANA.

    Inputs:
        request: Assessment ID, customer class, source, and selected answers.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        AssessmentResponsesResponse: Persisted answers for submitted questions.
    """
    try:
        profile = _repository_profile(repository, request.assessment_id)
        customer_class = _effective_customer_class(profile, request.customer_class)
        answers = _validated_answers(repository, request.answers, customer_class)
        persisted = repository.save_assessment_responses(
            assessment_id=request.assessment_id,
            customer_class=customer_class,
            answers=answers,
            source=request.source,
        )
        return AssessmentResponsesResponse(
            assessment_id=request.assessment_id,
            customer_class=customer_class,
            answers=persisted,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/responses", response_model=AssessmentResponsesResponse)
def get_assessment_responses(
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentResponsesResponse:
    """Return persisted questionnaire answer selections for one assessment.

    Inputs:
        assessment_id: Assessment instance whose answers should be returned.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        AssessmentResponsesResponse: Stored answers keyed by question ID.
    """
    profile = _repository_profile(repository, assessment_id)
    return AssessmentResponsesResponse(
        assessment_id=assessment_id,
        customer_class=profile.customer_class if profile is not None else None,
        answers=repository.get_assessment_responses(assessment_id),
    )


@router.post("/responses/apply-ai", response_model=AssessmentResponsesResponse)
def apply_ai_marks(
    request: ApplyAiMarksRequest,
    repository: Annotated[Any, Depends(get_ai_review_repository)],
) -> AssessmentResponsesResponse:
    """Apply AI verified answer IDs to the persisted user form state.

    Inputs:
        request: Assessment ID, customer class, target question, optional task
            ID, and AI-selected answer item IDs.
        repository: AI review repository provided by FastAPI dependency
            injection.

    Outputs:
        AssessmentResponsesResponse: Persisted answer IDs for the question.
    """
    try:
        profile = _repository_profile(repository, request.assessment_id)
        customer_class = _effective_customer_class(profile, request.customer_class)
        answers = _validated_answers(
            repository,
            {request.question_id: request.answer_item_ids},
            customer_class,
        )
        persisted = repository.save_assessment_responses(
            assessment_id=request.assessment_id,
            customer_class=customer_class,
            answers=answers,
            source="ai_apply",
        )
        repository.record_ai_applied_suggestion(
            task_id=request.task_id,
            question_id=request.question_id,
            answer_item_ids=persisted.get(request.question_id, []),
        )
        return AssessmentResponsesResponse(
            assessment_id=request.assessment_id,
            customer_class=customer_class,
            answers=persisted,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/score", response_model=AssessmentScoreResponse)
def get_assessment_score(
    assessment_id: Annotated[str, Query()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    customer_class: Annotated[str | None, Query()] = None,
    sector: Annotated[str | None, Query()] = None,
    language: Annotated[str, Query()] = DEFAULT_LANGUAGE,
) -> AssessmentScoreResponse:
    """Calculate the report-ready assessment score from persisted answers.

    Inputs:
        assessment_id: Assessment instance to score.
        repository: AI review repository provided by FastAPI dependency
            injection.
        customer_class: Customer class used for answer applicability.
        sector: Optional operating sector used for benchmark lookup.
        language: Framework language used for question and dimension labels.

    Outputs:
        AssessmentScoreResponse: Report-ready score payload for future PDF
        generation and UI consumption.
    """
    try:
        normalized_language = normalize_language(language)
        profile = _repository_profile(repository, assessment_id)
        normalized_class = _effective_customer_class(profile, customer_class)
        effective_nace1 = _effective_nace1(profile, sector)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    dimensions = repository.list_dimensions(language=normalized_language)
    questions = repository.list_all_questions(language=normalized_language)
    active_import_loader = getattr(repository, "get_active_benchmark_import", None)
    active_import = active_import_loader() if callable(active_import_loader) else None
    peer_submissions = []
    if active_import is not None and profile is not None:
        peer_submissions = repository.list_benchmark_peer_submissions(
            active_import.import_id,
            normalized_class,
            profile.nace1,
        )
    return calculate_assessment_score(
        assessment_id=assessment_id,
        customer_class=normalized_class,
        sector=effective_nace1,
        dimensions=dimensions,
        questions=questions,
        selected_answers=repository.get_assessment_responses(assessment_id),
        benchmarks=[],
        benchmark_import=active_import,
        peer_submissions=peer_submissions,
        nace1=profile.nace1 if profile is not None else None,
        source_company_id=(
            profile.source_company_id if profile is not None else None
        ),
        language=normalized_language,
    )
