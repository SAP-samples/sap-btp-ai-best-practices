"""API routes for importing assessment framework and Joule knowledge resources into HANA."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import DBAPIError

from app.models.imports import (
    AssessmentImportResponse,
    JouleKnowledgeImportResponse,
)
from app.models.benchmarking import (
    BenchmarkImportHistoryResponse,
    PublicBenchmarkValidationSummary,
)
from app.routers.ai_review import get_ai_review_repository
from app.security import get_api_key
from app.services.benchmark_import import (
    BenchmarkValidationError,
    import_benchmark_workbook,
)
from app.services.benchmark_import.public_summary import (
    public_benchmark_validation_summary,
)
from scripts import (
    import_assessment_framework,
    import_joule_knowledge_resources as joule_import_script,
)
from app.services.framework_importer import ITALIAN_DIMENSION_FILES
from app.services.joule_knowledge_importer import (
    DEFAULT_EMBEDDING_MODEL,
    GenAiHubEmbeddingClient,
    build_question_embedding_rows,
    load_joule_knowledge_seed,
)

router = APIRouter(dependencies=[Depends(get_api_key)])


def _expected_italian_filenames() -> set[str]:
    """Return the canonical Italian translation CSV filenames used by the importer.

    Inputs:
        None.

    Outputs:
        set[str]: The exact expected Italian filenames by dimension.
    """

    return {file_name for file_name, _display_name in ITALIAN_DIMENSION_FILES.values()}


def _assert_extension(upload: UploadFile, suffix: str, field_name: str) -> None:
    """Validate multipart upload extension before routing to parser or loader.

    Inputs:
        upload: Multipart upload from FastAPI.
        suffix: Required file extension, e.g. ``.xlsx``.
        field_name: Input field name for error messages.

    Outputs:
        None.

    Raises:
        HTTPException: Raised with HTTP 400 when filename is missing or extension
        does not match.
    """

    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must include a filename.",
        )
    if Path(upload.filename).suffix.lower() != suffix.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} must have extension {suffix!r}.",
        )


async def _save_uploads_to_temp(
    temp_dir: Path,
    uploads: list[UploadFile],
) -> list[Path]:
    """Persist uploads to a temporary directory and return file paths.

    Inputs:
        temp_dir: Writable temporary directory root.
        uploads: Uploads from multipart/form-data request body.

    Outputs:
        list[Path]: Concrete filesystem paths in the same order as uploads.

    Raises:
        HTTPException: Raised with HTTP 400 for missing filename and file read
        errors.
    """

    saved_paths: list[Path] = []
    for upload in uploads:
        if not upload.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All uploaded files must include a filename.",
            )
        file_name = Path(upload.filename).name
        if not file_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All uploaded files must include a non-empty filename.",
            )

        target = temp_dir / file_name
        file_content = await upload.read()
        target.write_bytes(file_content)
        saved_paths.append(target)
    return saved_paths


@router.post(
    "/assessment-benchmarks",
    response_model=PublicBenchmarkValidationSummary,
)
async def import_assessment_benchmarks(
    workbook: Annotated[UploadFile, File()],
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    write: Annotated[bool, Form()] = False,
) -> PublicBenchmarkValidationSummary:
    """Validate or explicitly persist one versioned assessment benchmark XLSX.

    Inputs:
        workbook: Multipart XLSX bytes; parsing never depends on a temp filepath.
        repository: HANA-backed repository providing canonical framework rows and
        the current transaction session.
        write: Explicit opt-in to persist/activate the validated version.

    Outputs:
        PublicBenchmarkValidationSummary: Identity-free structured dry-run,
        active, or no-op result suitable for external clients.

    Raises:
        HTTPException: HTTP 400 for malformed/unsafe/invalid workbooks and safe
        HTTP 500 for non-database write failures. Database-driver errors remain
        visible to the shared repository dependency for retryable 503 handling.
    """

    _assert_extension(workbook, ".xlsx", "workbook")
    filename = Path(workbook.filename or "benchmark.xlsx").name
    try:
        content = await workbook.read()
        framework_questions = repository.list_all_questions(language="en")
        return public_benchmark_validation_summary(
            import_benchmark_workbook(
                content,
                filename,
                framework_questions,
                write=write,
                session=getattr(repository, "session", None) if write else None,
            )
        )
    except DBAPIError:
        # Preserve the application-wide HANA resilience boundary. It maps
        # transient connection loss to HTTP 503 + Retry-After and leaves SQL or
        # schema programming failures visible instead of mislabeling the XLSX.
        raise
    except BenchmarkValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Assessment benchmark workbook validation failed.",
                "validation": public_benchmark_validation_summary(
                    exc.summary
                ).model_dump(mode="json"),
            },
        ) from exc
    except (ValueError, KeyError) as exc:
        if write:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Assessment benchmark import failed while writing to HANA.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment benchmark workbook is invalid.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive write/read boundary
        if write:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Assessment benchmark import failed while writing to HANA.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment benchmark workbook could not be validated.",
        ) from exc


@router.get(
    "/assessment-benchmarks",
    response_model=BenchmarkImportHistoryResponse,
)
def list_assessment_benchmark_imports(
    repository: Annotated[Any, Depends(get_ai_review_repository)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> BenchmarkImportHistoryResponse:
    """Return the active and bounded recent import summaries without sensitive data.

    Inputs:
        repository: HANA-backed repository exposing safe benchmark metadata.
        limit: Positive recent-version maximum capped at 50 by validation.

    Outputs:
        BenchmarkImportHistoryResponse: Active/recent metadata without workbook
        BLOBs, source company IDs, or questionnaire IDs.
    """

    active = repository.get_active_benchmark_import()
    recent = repository.list_recent_benchmark_imports(limit)
    return BenchmarkImportHistoryResponse(
        available=active is not None,
        reason=None if active is not None else "no_active_dataset",
        active=active,
        recent=recent,
    )


def _ensure_expected_italian_filenames(upload_paths: list[Path]) -> None:
    """Validate optional Italian uploads match exactly the expected filenames.

    Inputs:
        upload_paths: Paths saved for ``italian_csvs`` inputs.

    Outputs:
        None.

    Raises:
        HTTPException: Raised with HTTP 400 when files are missing or unexpected.
    """

    expected = _expected_italian_filenames()
    received = {path.name for path in upload_paths}

    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Italian CSV uploads must exactly match the expected files.",
                "expected": sorted(expected),
                "missing": missing,
                "extra": extra,
            },
        )


@router.post("/assessment-framework", response_model=AssessmentImportResponse)
async def import_assessment_framework_resources(
    workbook: Annotated[UploadFile, File()],
    explanations: Annotated[UploadFile, File()],
    write: Annotated[bool, Form()] = False,
    italian_csvs: Annotated[list[UploadFile] | None, File()] = None,
) -> AssessmentImportResponse:
    """Import assessment framework seed files from multipart uploads.

    Inputs:
        workbook: XLSX framework export with question rows.
        explanations: CSV explanations mapped to question IDs.
        write: Whether to persist into HANA.
        italian_csvs: Optional repeated Italian translation CSV uploads.

    Outputs:
        AssessmentImportResponse: Parsed counts and import write status.
    """

    _assert_extension(workbook, ".xlsx", "workbook")
    _assert_extension(explanations, ".csv", "explanations")
    if italian_csvs is None:
        italian_csvs = []
    for upload in italian_csvs:
        _assert_extension(upload, ".csv", f"italian_csvs ({upload.filename})")

    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        saved_workbook, saved_explanations = await _save_uploads_to_temp(
            temp_dir,
            [workbook, explanations],
        )
        try:
            seed = import_assessment_framework.load_framework_seed(
                workbook_path=saved_workbook,
                explanations_path=saved_explanations,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid assessment workbook content: {exc}",
            ) from exc

        translated_seed = None
        if italian_csvs:
            italian_dir = temp_dir / "italian-csvs"
            italian_dir.mkdir()
            italian_paths = await _save_uploads_to_temp(italian_dir, italian_csvs)
            _ensure_expected_italian_filenames(italian_paths)

            try:
                translated_seed = import_assessment_framework.load_italian_framework_translations(
                    italian_dir,
                    seed.questions,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Italian CSV content: {exc}",
                ) from exc

        write_completed = False
        if write:
            try:
                import_assessment_framework.write_framework_to_hana(
                    seed=seed,
                    workbook_path=saved_workbook,
                    explanations_path=saved_explanations,
                    translations=translated_seed,
                )
                write_completed = True
            except Exception as exc:  # pragma: no cover - write-path guardrail
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Assessment import failed while writing to HANA: {exc}",
                ) from exc

        return AssessmentImportResponse(
            dimensions=len(import_assessment_framework.ordered_dimensions(seed)),
            questions=len(seed.questions),
            answer_items=sum(len(question.answer_items) for question in seed.questions),
            explanations=seed.explanation_count,
            italian_questions=(
                len(translated_seed.question_translations)
                if translated_seed is not None
                else None
            ),
            italian_answer_items=(
                len(translated_seed.answer_item_translations)
                if translated_seed is not None
                else None
            ),
            write_completed=write_completed,
        )


@router.post("/joule-knowledge", response_model=JouleKnowledgeImportResponse)
async def import_joule_knowledge_resources_route(
    glossary_workbook: Annotated[UploadFile, File()],
    explanations_workbook: Annotated[UploadFile, File()],
    write: Annotated[bool, Form()] = False,
    embedding_model: Annotated[str, Form()] = DEFAULT_EMBEDDING_MODEL,
    batch_size: Annotated[int, Form()] = 32,
) -> JouleKnowledgeImportResponse:
    """Import Joule glossary and explanation workbooks from multipart uploads.

    Inputs:
        glossary_workbook: XLSX glossary workbook.
        explanations_workbook: XLSX question explanation workbook.
        write: Whether to persist into HANA.
        embedding_model: Embedding model used to build semantic vectors.
        batch_size: Optional batch size for embedding calls.

    Outputs:
        JouleKnowledgeImportResponse: Parsed counts and import write status.
    """

    _assert_extension(glossary_workbook, ".xlsx", "glossary_workbook")
    _assert_extension(explanations_workbook, ".xlsx", "explanations_workbook")
    if batch_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="batch_size must be greater than 0.",
        )

    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        saved_glossary, saved_explanations = await _save_uploads_to_temp(
            temp_dir,
            [glossary_workbook, explanations_workbook],
        )

        try:
            seed = load_joule_knowledge_seed(
                glossary_workbook=saved_glossary,
                explanations_workbook=saved_explanations,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Joule workbook content: {exc}",
            ) from exc

        write_completed = False
        if write:
            try:
                embedding_client = GenAiHubEmbeddingClient(model_name=embedding_model)
                question_embeddings = build_question_embedding_rows(
                    seed=seed,
                    embedding_client=embedding_client,
                    batch_size=batch_size,
                    show_progress=False,
                )
                joule_import_script.write_joule_knowledge_to_hana(
                    glossary_workbook=saved_glossary,
                    explanations_workbook=saved_explanations,
                    embedding_model=embedding_model,
                    batch_size=batch_size,
                    seed=seed,
                    question_embeddings=question_embeddings,
                    show_progress=False,
                )
                write_completed = True
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Joule import configuration: {exc}",
                ) from exc
            except Exception as exc:  # pragma: no cover - write-path guardrail
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Joule import failed while writing to HANA: {exc}",
                ) from exc

        return JouleKnowledgeImportResponse(
            glossary_terms=len(seed.glossary_terms),
            question_explanations=len(seed.question_explanations),
            dimensions=len(seed.dimensions),
            embedding_model=embedding_model,
            batch_size=batch_size,
            write_completed=write_completed,
        )
