"""Orchestrate pure validation, canonical mapping, scoring, and import mode."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models.assessment import AssessmentQuestion

from .constants import SCORING_VERSION
from .entities import (
    cohort_counts,
    company_from_row,
    dimension_aliases,
    submission_from_row,
)
from .models import (
    BenchmarkCompany,
    BenchmarkDataset,
    BenchmarkResponse,
    BenchmarkSubmission,
    BenchmarkValidationError,
    BenchmarkValidationSummary,
)
from .scoring import calculate_benchmark_scores
from .security import WorkbookSafetyError, validate_xlsx_container
from .validation import RowValueError, ValidationCollector, WarningAccumulator
from .values import boolean, float_or_none, identifier, level, normalized_text, text_or_none
from .workbook import file_failure, read_source_rows


def _required_identities(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return normalized questionnaire, company, question, and answer IDs.

    Inputs:
        row: Source row mapping keyed by exact workbook headers.

    Outputs:
        tuple[str, str, str, str]: Normalized source identities in stable order.
    """

    return (
        identifier(row.get("ID Questionario")),
        identifier(row.get("ID Impresa")),
        identifier(row.get("ID Domanda")),
        identifier(row.get("ID Risposta")),
    )


def _missing_identity_labels(identities: tuple[str, str, str, str]) -> list[str]:
    """Return source column labels whose normalized identities are blank.

    Inputs:
        identities: Questionnaire, company, question, and answer IDs.

    Outputs:
        list[str]: Exact source labels for blank required identities.
    """

    labels = ("ID Questionario", "ID Impresa", "ID Domanda", "ID Risposta")
    return [label for label, value in zip(labels, identities, strict=True) if not value]


def _summary_fields(
    *,
    import_id: str,
    filename: str,
    source_sha256: str,
    source_row_count: int,
    companies: list[BenchmarkCompany],
    submissions: list[BenchmarkSubmission],
    responses: list[BenchmarkResponse],
    rejected_count: int,
) -> dict[str, Any]:
    """Build shared validation summary fields from normalized parser state.

    Inputs:
        import_id: New benchmark import version identity.
        filename: Original source filename.
        source_sha256: Hex-encoded workbook content hash.
        source_row_count: Number of non-empty source rows.
        companies: Accepted normalized unique companies.
        submissions: Accepted normalized unique questionnaires.
        responses: Accepted canonically mapped response rows.
        rejected_count: Count of unique rejected worksheet rows.

    Outputs:
        dict[str, Any]: Fields shared by rejected and successful summaries.
    """

    return {
        "import_id": import_id,
        "source_filename": filename,
        "source_sha256": source_sha256,
        "scoring_version": SCORING_VERSION,
        "row_count": source_row_count,
        "company_count": len(companies),
        "questionnaire_count": len(submissions),
        "question_count": len({response.question_id for response in responses}),
        "class_counts": cohort_counts(companies, "customer_class"),
        "nace1_counts": cohort_counts(companies, "nace1"),
        "nace2_counts": cohort_counts(companies, "nace2"),
        "nace3_counts": cohort_counts(companies, "nace3"),
        "accepted_count": source_row_count - rejected_count,
        "rejected_count": rejected_count,
    }


def _supplied_score_reconciles(calculated: float, supplied: float) -> bool:
    """Compare calculated and source dimension scores by source precision.

    Inputs:
        calculated: Four-decimal application-compatible dimension score.
        supplied: Finite audit score read from ``Dimensione Score``.

    Outputs:
        bool: Integer source scores use the source workbook's historical
        truncation rule; decimal source scores must match at four decimals.
    """

    if supplied.is_integer():
        return math.floor(calculated + 1e-9) == int(supplied)
    return round(calculated, 4) == round(supplied, 4)


def parse_benchmark_workbook(
    content: bytes,
    filename: str,
    framework_questions: list[AssessmentQuestion],
) -> BenchmarkDataset:
    """Parse workbook bytes into a normalized, scored benchmark dataset.

    Inputs:
        content: Raw source XLSX bytes.
        filename: Original upload or CLI filename; must end in ``.xlsx``.
        framework_questions: Canonical questions and answer items used for
            dimension derivation and within-level answer mapping.

    Outputs:
        BenchmarkDataset: Validated entities, pure calculated scores, original
        workbook bytes, and a JSON-safe validation summary.

    Raises:
        BenchmarkValidationError: If file, headers, or any row is blocking.
    """

    source_sha256 = hashlib.sha256(content).hexdigest()
    if Path(filename).suffix.casefold() != ".xlsx":
        raise file_failure(
            filename,
            source_sha256,
            "invalid_extension",
            "Assessment benchmark source must use the .xlsx extension",
        )
    try:
        validate_xlsx_container(content)
    except WorkbookSafetyError as exc:
        raise file_failure(filename, source_sha256, exc.code, str(exc)) from exc

    headers, source_rows = read_source_rows(content, filename, source_sha256)
    if not source_rows:
        raise file_failure(
            filename,
            source_sha256,
            "empty_workbook",
            "Assessment benchmark source must contain at least one response row",
        )
    import_id = uuid4().hex
    questions_by_id = {
        question.question_id: question for question in framework_questions
    }
    canonical_by_level = {
        (question.question_id, maturity_level): sorted(
            [
                item
                for item in question.answer_items
                if item.level == maturity_level
            ],
            key=lambda item: item.item_index,
        )
        for question in framework_questions
        for maturity_level in range(1, 6)
    }

    collector = ValidationCollector()
    warnings = WarningAccumulator()
    companies_by_id: dict[str, BenchmarkCompany] = {}
    submissions_by_id: dict[str, BenchmarkSubmission] = {}
    metadata_by_questionnaire: dict[
        str, tuple[BenchmarkCompany, BenchmarkSubmission]
    ] = {}
    responses: list[BenchmarkResponse] = []
    responses_by_questionnaire: dict[str, list[BenchmarkResponse]] = defaultdict(list)
    seen_response_rows: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    source_answer_levels: dict[tuple[str, str], int] = {}
    source_answer_mapping: dict[tuple[str, int, str], str] = {}
    source_order_by_level: dict[tuple[str, int], list[str]] = defaultdict(list)
    supplied_scores: dict[tuple[str, str], float | None] = {}

    for row_number, row in source_rows:
        questionnaire_id, company_id, question_id, source_answer_id = (
            _required_identities(row)
        )
        missing = _missing_identity_labels(
            (questionnaire_id, company_id, question_id, source_answer_id)
        )
        if missing:
            collector.add(
                row_number,
                "missing_identifier",
                f"Required identifiers are blank: {', '.join(missing)}",
            )
            continue

        identity = (questionnaire_id, question_id, source_answer_id)
        fingerprint = tuple(row.get(header) for header in headers)
        previous_fingerprint = seen_response_rows.get(identity)
        if previous_fingerprint is not None:
            code = (
                "duplicate_row"
                if previous_fingerprint == fingerprint
                else "conflicting_row"
            )
            collector.add(
                row_number,
                code,
                f"Response identity {identity} appears more than once",
            )
            continue
        seen_response_rows[identity] = fingerprint

        try:
            company = company_from_row(row, import_id, company_id, warnings)
            submission = submission_from_row(
                row,
                import_id,
                questionnaire_id,
                company_id,
            )
            parsed_level = level(row.get("Livello"))
            selected = boolean(row.get("Valore Risposta"), "Valore Risposta")
            optional = boolean(row.get("Opzionale"), "Opzionale")
            managed = boolean(row.get("Domanda Gestita"), "Domanda Gestita")
            supplied_score = float_or_none(
                row.get("Dimensione Score"),
                "Dimensione Score",
            )
        except RowValueError as exc:
            collector.add(row_number, exc.code, str(exc))
            continue

        metadata = (company, submission)
        previous_metadata = metadata_by_questionnaire.get(questionnaire_id)
        if previous_metadata is not None and previous_metadata != metadata:
            collector.add(
                row_number,
                "mixed_questionnaire_metadata",
                f"Questionnaire {questionnaire_id} contains mixed company or submission metadata",
            )
            continue
        existing_company = companies_by_id.get(company_id)
        if existing_company is not None and existing_company != company:
            collector.add(
                row_number,
                "mixed_company_metadata",
                f"Company {company_id} contains conflicting profile metadata",
            )
            continue
        metadata_by_questionnaire.setdefault(questionnaire_id, metadata)

        question = questions_by_id.get(question_id)
        if question is None:
            collector.add(
                row_number,
                "unknown_question",
                f"Workbook references unknown framework question {question_id}",
            )
            continue

        answer_level_identity = (question_id, source_answer_id)
        prior_level = source_answer_levels.get(answer_level_identity)
        if prior_level is not None and prior_level != parsed_level:
            collector.add(
                row_number,
                "conflicting_answer_catalog",
                f"Source answer {source_answer_id} changes maturity level",
            )
            continue
        source_answer_levels[answer_level_identity] = parsed_level

        mapping_key = (question_id, parsed_level, source_answer_id)
        canonical_answer_id = source_answer_mapping.get(mapping_key)
        if canonical_answer_id is None:
            order_key = (question_id, parsed_level)
            source_order_by_level[order_key].append(source_answer_id)
            ordinal = len(source_order_by_level[order_key])
            candidates = canonical_by_level.get(order_key, [])
            if ordinal > len(candidates):
                collector.add(
                    row_number,
                    "unmappable_answer",
                    f"Source answer {source_answer_id} has no canonical within-level mapping",
                )
                continue
            canonical_answer_id = candidates[ordinal - 1].answer_item_id
            source_answer_mapping[mapping_key] = canonical_answer_id

        response = BenchmarkResponse(
            import_id=import_id,
            questionnaire_id=questionnaire_id,
            question_id=question_id,
            canonical_answer_id=canonical_answer_id,
            source_answer_id=source_answer_id,
            source_answer_text=str(row.get("Testo risposta") or "").strip(),
            source_answer_value=text_or_none(row.get("Valore Risposta")),
            level=parsed_level,
            selected=bool(selected),
            optional=bool(optional),
            managed=bool(managed),
            validation_status=text_or_none(row.get("Stato Validazione")),
        )
        canonical_item = next(
            item
            for item in canonical_by_level[(question_id, parsed_level)]
            if item.answer_item_id == canonical_answer_id
        )
        if normalized_text(response.source_answer_text) != normalized_text(
            canonical_item.text
        ):
            warnings.add(
                "answer_text_mismatch",
                "Source answer text differs from its canonical mapped answer",
                f"row {row_number}: {source_answer_id} -> {canonical_answer_id}",
                token=(
                    f"{question_id}:{canonical_answer_id}:"
                    f"{normalized_text(response.source_answer_text)}"
                ),
            )
        if normalized_text(row.get("Dimensione")) not in dimension_aliases(question):
            warnings.add(
                "dimension_mismatch",
                "Source dimensions differ from the framework; canonical dimensions were derived by question",
                f"row {row_number}: {row.get('Dimensione')!r} -> {question.dimension}",
            )

        supplied_key = (questionnaire_id, question.dimension)
        if (
            supplied_key in supplied_scores
            and supplied_scores[supplied_key] != supplied_score
        ):
            collector.add(
                row_number,
                "mixed_dimension_score",
                f"Questionnaire {questionnaire_id} has mixed supplied scores for {question.dimension}",
            )
            continue
        supplied_scores.setdefault(supplied_key, supplied_score)
        companies_by_id.setdefault(company_id, company)
        submissions_by_id.setdefault(questionnaire_id, submission)
        responses.append(response)
        responses_by_questionnaire[questionnaire_id].append(response)

    companies = list(companies_by_id.values())
    submissions = list(submissions_by_id.values())
    common_summary = _summary_fields(
        import_id=import_id,
        filename=filename,
        source_sha256=source_sha256,
        source_row_count=len(source_rows),
        companies=companies,
        submissions=submissions,
        responses=responses,
        rejected_count=len(collector.rejected_rows),
    )
    if collector.errors:
        raise BenchmarkValidationError(
            BenchmarkValidationSummary(
                **common_summary,
                warnings=warnings.build(),
                sampled_errors=collector.errors,
                success=False,
                status="rejected",
            )
        )

    scores = []
    company_class_by_questionnaire = {
        submission.questionnaire_id: companies_by_id[
            submission.source_company_id
        ].customer_class
        for submission in submissions
    }
    for submission in submissions:
        questionnaire_id = submission.questionnaire_id
        score_rows = calculate_benchmark_scores(
            import_id=import_id,
            questionnaire_id=questionnaire_id,
            customer_class=company_class_by_questionnaire[questionnaire_id],
            questions=questions_by_id,
            responses=responses_by_questionnaire[questionnaire_id],
            supplied_dimension_scores={
                dimension: score
                for (source_questionnaire, dimension), score in supplied_scores.items()
                if source_questionnaire == questionnaire_id
            },
        )
        scores.extend(score_rows)
        for score in score_rows:
            if score.scope_type != "dimension" or score.supplied_score is None:
                continue
            if not _supplied_score_reconciles(
                score.calculated_score,
                score.supplied_score,
            ):
                warnings.add(
                    "score_reconciliation",
                    "Calculated dimension score does not reconcile with the supplied truncated score",
                    f"questionnaire {questionnaire_id}, {score.dimension}: "
                    f"calculated {score.calculated_score}, supplied {score.supplied_score}",
                    token=f"{questionnaire_id}:{score.dimension}",
                )

    summary = BenchmarkValidationSummary(
        **common_summary,
        warnings=warnings.build(),
        sampled_errors=[],
        success=True,
        status="validated",
        write_completed=False,
    )
    return BenchmarkDataset(
        workbook_bytes=content,
        companies=companies,
        submissions=submissions,
        responses=responses,
        scores=scores,
        summary=summary,
    )


def import_benchmark_workbook(
    content: bytes,
    filename: str,
    framework_questions: list[AssessmentQuestion],
    *,
    write: bool = False,
    session: Any | None = None,
    batch_size: int = 250,
) -> BenchmarkValidationSummary:
    """Validate a workbook and optionally persist one active HANA version.

    Inputs:
        content: Raw source XLSX bytes.
        filename: Original upload or CLI filename.
        framework_questions: Canonical questions used for validation and scores.
        write: Explicit opt-in for HANA persistence; defaults to dry-run.
        session: SQLAlchemy-compatible HANA session required when ``write``.
        batch_size: Positive maximum DML rows per HANA executemany call.

    Outputs:
        BenchmarkValidationSummary: JSON-safe dry-run, imported, or no-op result.
    """

    dataset = parse_benchmark_workbook(content, filename, framework_questions)
    if not write:
        return dataset.summary
    if session is None:
        raise ValueError("A HANA session is required when write=True")
    from .repository import write_benchmark_dataset

    result = write_benchmark_dataset(session, dataset, batch_size=batch_size)
    return dataset.summary.model_copy(
        update={
            "import_id": result.import_id,
            "status": result.status,
            "write_completed": not result.no_op,
            "no_op": result.no_op,
        }
    )
