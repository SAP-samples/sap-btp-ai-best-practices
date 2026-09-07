"""Behavior tests for pure assessment benchmark workbook validation and parsing."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.benchmark_import import (
    BenchmarkValidationError,
    import_benchmark_workbook,
    parse_benchmark_workbook,
)
from app.services.framework_importer import (
    load_framework_seed,
    load_italian_framework_translations,
)


REQUIRED_COLUMNS = [
    "Data Estrazione",
    "Data Sottomissione",
    "ID Impresa",
    "Fatturato",
    "Nr Dipendenti",
    "Settore Operativo (NACE) 1",
    "Settore Operativo (NACE) 2",
    "Settore Operativo (NACE) 3",
    "Dimensione Azienda",
    "Classe",
    "Forma Giuridica",
    "Presenza Geografica",
    "Quotata",
    "Committente Contratti Pubblici",
    "Adesione Codice di  Autodisciplina",
    "ID Questionario",
    "Stato Questionario",
    "Assessment Score",
    "Dimensione",
    "Dimensione Score",
    "ID Domanda",
    "Sezione",
    "Domanda",
    "Domanda Gestita",
    "Stato Validazione",
    "Livello",
    "ID Risposta",
    "Testo risposta",
    "Valore Risposta",
    "Opzionale",
]
"""Exact source workbook columns in their required order."""


def _question() -> AssessmentQuestion:
    """Return a compact class-3 framework question for parser tests.

    Inputs:
        None.

    Outputs:
        AssessmentQuestion: One question with two level-one canonical answers.
    """

    return AssessmentQuestion(
        question_id="Q.STR.03.01",
        dimension="Strategy",
        section="Objectives",
        question="Are objectives defined?",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.03.01-L1-001",
                question_id="Q.STR.03.01",
                level=1,
                item_index=1,
                text="First objective practice",
            ),
            AnswerItem(
                answer_item_id="Q.STR.03.01-L1-002",
                question_id="Q.STR.03.01",
                level=1,
                item_index=2,
                text="Second objective practice",
            ),
        ],
    )


def _row(**updates: Any) -> dict[str, Any]:
    """Return one valid source row with caller-provided field updates.

    Inputs:
        updates: Source column values that override the valid defaults.

    Outputs:
        dict[str, Any]: Mapping ready to write in exact workbook column order.
    """

    row: dict[str, Any] = {
        "Data Estrazione": "2026-07-06",
        "Data Sottomissione": "2026-04-18",
        "ID Impresa": "SYN-COMPANY-001",
        "Fatturato": 2_500_000,
        "Nr Dipendenti": 80,
        "Settore Operativo (NACE) 1": "Synthetic professional activities",
        "Settore Operativo (NACE) 2": "Synthetic consulting activities",
        "Settore Operativo (NACE) 3": "Synthetic technical consulting",
        "Dimensione Azienda": "Media Impresa sintetica",
        "Classe": "3",
        "Forma Giuridica": "Synthetic limited company",
        "Presenza Geografica": "Nazionale",
        "Quotata": "No",
        "Committente Contratti Pubblici": "No",
        "Adesione Codice di  Autodisciplina": "No",
        "ID Questionario": "SYN-QUESTIONNAIRE-001",
        "Stato Questionario": "REL",
        "Assessment Score": "Visione",
        "Dimensione": "Strategia",
        "Dimensione Score": 12,
        "ID Domanda": "Q.STR.03.01",
        "Sezione": "Objectives",
        "Domanda": "Are objectives defined?",
        "Domanda Gestita": True,
        "Stato Validazione": "ATT",
        "Livello": 1,
        "ID Risposta": "R.STR.03.01.01",
        "Testo risposta": "First objective practice",
        "Valore Risposta": "Si",
        "Opzionale": "No",
    }
    row.update(updates)
    return row


def _workbook_bytes(
    rows: list[dict[str, Any]],
    *,
    headers: list[str] | None = None,
    sheet_name: str = "Estrazione",
) -> bytes:
    """Build in-memory XLSX bytes from source-shaped rows.

    Inputs:
        rows: Row mappings to append after the header.
        headers: Optional exact header order; defaults to required columns.
        sheet_name: Worksheet name to create.

    Outputs:
        bytes: Valid XLSX package containing the requested worksheet.
    """

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    ordered_headers = headers or REQUIRED_COLUMNS
    sheet.append(ordered_headers)
    for row in rows:
        sheet.append([row.get(header) for header in ordered_headers])
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def _two_answer_rows() -> list[dict[str, Any]]:
    """Return two source rows that map by within-level order.

    Inputs:
        None.

    Outputs:
        list[dict[str, Any]]: Selected first answer and unselected second answer.
    """

    return [
        _row(),
        _row(
            **{
                "ID Risposta": "R.STR.03.01.02",
                "Testo risposta": "Second objective practice",
                "Valore Risposta": "No",
            }
        ),
    ]


def _error_codes(error: BenchmarkValidationError) -> set[str]:
    """Return blocking validation codes from an importer exception.

    Inputs:
        error: Validation exception raised by the pure parser.

    Outputs:
        set[str]: Codes attached to sampled row or workbook errors.
    """

    return {item.code for item in error.summary.sampled_errors}


def _warning_codes(dataset: Any) -> set[str]:
    """Return aggregate warning codes from a parsed benchmark dataset.

    Inputs:
        dataset: Successful normalized parser output.

    Outputs:
        set[str]: Warning codes present in its validation summary.
    """

    return {item.code for item in dataset.summary.warnings}


def _italian_framework(repo_root: Path) -> list[AssessmentQuestion]:
    """Load canonical questions with Italian source-comparison text.

    Inputs:
        repo_root: Repository root containing framework seed files.

    Outputs:
        list[AssessmentQuestion]: Canonical IDs/dimensions with Italian labels.
    """

    seed = load_framework_seed(
        workbook_path=repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        explanations_path=repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    translations = load_italian_framework_translations(
        italian_dir=repo_root / "data" / "sanitized" / "IT",
        questions=seed.questions,
    )
    localized: list[AssessmentQuestion] = []
    for question in seed.questions:
        question_translation = translations.question_translations[question.question_id]
        localized.append(
            question.model_copy(
                update={
                    "section": question_translation.section,
                    "question": question_translation.question,
                    "answer_items": [
                        item.model_copy(
                            update={
                                "text": translations.answer_item_translations[
                                    item.answer_item_id
                                ]
                            }
                        )
                        for item in question.answer_items
                    ],
                }
            )
        )
    return localized


def test_parser_maps_answers_and_calculates_application_compatible_scores() -> None:
    """Verify canonical mapping and the application question score formula.

    Inputs:
        None. The test builds a two-answer source workbook in memory.

    Outputs:
        None. Assertions cover canonical identity and all persisted score scopes.
    """

    dataset = parse_benchmark_workbook(
        _workbook_bytes(_two_answer_rows()),
        "benchmark.xlsx",
        [_question()],
    )

    assert dataset.summary.success is True
    assert dataset.summary.status == "validated"
    assert dataset.summary.accepted_count == 2
    assert dataset.summary.rejected_count == 0
    assert [response.canonical_answer_id for response in dataset.responses] == [
        "Q.STR.03.01-L1-001",
        "Q.STR.03.01-L1-002",
    ]
    assert [response.selected for response in dataset.responses] == [True, False]
    assert {(score.scope_type, score.calculated_score) for score in dataset.scores} == {
        ("topic", 12.5),
        ("dimension", 12.5),
        ("overall", 12.5),
    }
    dimension_score = next(
        score for score in dataset.scores if score.scope_type == "dimension"
    )
    assert dimension_score.supplied_score == 12.0
    assert json.loads(json.dumps(dataset.summary.model_dump(mode="json")))["success"]


def test_scores_include_missing_applicable_questions_and_exclude_non_applicable() -> None:
    """Verify benchmark averages use the same applicability set as live scoring.

    Inputs:
        None. One applicable framework question is absent while one class-3
        unavailable question is present in the source workbook.

    Outputs:
        None. The missing applicable question scores zero and the unavailable
        source question is excluded from topic, dimension, and overall scores.
    """

    missing_applicable = AssessmentQuestion(
        question_id="Q.STR.02.01",
        dimension="Strategy",
        section="Objectives",
        question="Is strategy execution monitored?",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.02.01-L1-001",
                question_id="Q.STR.02.01",
                level=1,
                item_index=1,
                text="Strategy execution is monitored",
            )
        ],
    )
    present_non_applicable = AssessmentQuestion(
        question_id="Q.CAM.01.01",
        dimension="Combined Assurance & Management Oversight",
        section="Assurance",
        question="Is combined assurance established?",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.CAM.01.01-L1-001",
                question_id="Q.CAM.01.01",
                level=1,
                item_index=1,
                text="Combined assurance is established",
            )
        ],
    )
    rows = _two_answer_rows()
    rows.append(
        _row(
            **{
                "ID Domanda": "Q.CAM.01.01",
                "ID Risposta": "R.CAM.01.01.01",
                "Dimensione": "Combined Assurance & Management Oversight",
                "Testo risposta": "Combined assurance is established",
            }
        )
    )

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows),
        "benchmark.xlsx",
        [_question(), missing_applicable, present_non_applicable],
    )

    topic_scores = {
        score.question_id: score.calculated_score
        for score in dataset.scores
        if score.scope_type == "topic"
    }
    assert topic_scores == {
        "Q.STR.03.01": 12.5,
        "Q.STR.02.01": 0.0,
    }
    assert {
        score.dimension: score.calculated_score
        for score in dataset.scores
        if score.scope_type == "dimension"
    } == {"Strategy": 6.25}
    assert next(
        score.calculated_score
        for score in dataset.scores
        if score.scope_type == "overall"
    ) == 6.25


def test_decimal_supplied_score_uses_four_decimal_reconciliation() -> None:
    """Verify non-integer supplied scores reconcile at four-decimal precision.

    Inputs:
        None. The calculated score is 12.5 while the supplied audit score is 12.9.

    Outputs:
        None. Equal integer buckets do not hide the decimal discrepancy.
    """

    rows = _two_answer_rows()
    for row in rows:
        row["Dimensione Score"] = 12.9

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows),
        "benchmark.xlsx",
        [_question()],
    )

    assert "score_reconciliation" in _warning_codes(dataset)


def test_parser_accepts_the_single_optional_company_name_column() -> None:
    """Verify an optional future company name is accepted and persisted.

    Inputs:
        None. A ``Nome Impresa`` column is inserted after the company ID.

    Outputs:
        None. The normalized company retains the supplied display name.
    """

    headers = REQUIRED_COLUMNS.copy()
    headers.insert(headers.index("ID Impresa") + 1, "Nome Impresa")
    rows = _two_answer_rows()
    for row in rows:
        row["Nome Impresa"] = "Synthetic Example S.p.A."

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows, headers=headers),
        "benchmark.xlsx",
        [_question()],
    )

    assert dataset.companies[0].company_name == "Synthetic Example S.p.A."


def test_parser_aggregates_non_blocking_quality_warnings() -> None:
    """Verify bad source labels/text/profile scores warn without blocking import.

    Inputs:
        None. Source values intentionally contain known quality defects.

    Outputs:
        None. Assertions verify warning aggregation and null normalization.
    """

    rows = _two_answer_rows()
    rows[0]["Dimensione"] = "Incorrect dimension"
    rows[0]["Testo risposta"] = "First objective practcie"
    rows[0]["Forma Giuridica"] = 0
    rows[1]["Forma Giuridica"] = 0
    rows[0]["Dimensione Score"] = 99
    rows[1]["Dimensione Score"] = 99

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows),
        "benchmark.xlsx",
        [_question()],
    )

    assert {
        "dimension_mismatch",
        "answer_text_mismatch",
        "profile_placeholder_normalized",
        "score_reconciliation",
    } <= _warning_codes(dataset)
    assert dataset.companies[0].legal_form is None


@pytest.mark.parametrize(
    ("column", "attribute", "value"),
    [
        ("Quotata", "is_listed", False),
        ("Quotata", "is_listed", 0),
        (
            "Committente Contratti Pubblici",
            "is_public_contracting_client",
            False,
        ),
        ("Committente Contratti Pubblici", "is_public_contracting_client", 0),
        (
            "Adesione Codice di  Autodisciplina",
            "uses_self_governance_code",
            False,
        ),
        ("Adesione Codice di  Autodisciplina", "uses_self_governance_code", 0),
    ],
)
def test_boolean_profile_false_values_are_not_placeholder_nulls(
    column: str,
    attribute: str,
    value: bool | int,
) -> None:
    """Verify boolean ``False`` and numeric zero persist as explicit false.

    Inputs:
        column: Exact boolean profile column under test.
        attribute: Normalized company model attribute for that column.
        value: Workbook boolean ``False`` or numeric zero representation.

    Outputs:
        None. The profile flag is false and no placeholder warning is emitted.
    """

    rows = _two_answer_rows()
    for row in rows:
        row[column] = value

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows),
        "benchmark.xlsx",
        [_question()],
    )

    assert getattr(dataset.companies[0], attribute) is False
    assert "profile_placeholder_normalized" not in _warning_codes(dataset)


def test_boolean_profile_text_placeholders_remain_null_with_warnings() -> None:
    """Verify blank-like text flags still normalize to null with audit warnings.

    Inputs:
        None. All optional boolean profile fields use the text placeholder ``null``.

    Outputs:
        None. Flags remain unknown rather than false and warnings stay aggregated.
    """

    rows = _two_answer_rows()
    for row in rows:
        row["Quotata"] = "null"
        row["Committente Contratti Pubblici"] = "null"
        row["Adesione Codice di  Autodisciplina"] = "null"

    dataset = parse_benchmark_workbook(
        _workbook_bytes(rows),
        "benchmark.xlsx",
        [_question()],
    )

    company = dataset.companies[0]
    assert company.is_listed is None
    assert company.is_public_contracting_client is None
    assert company.uses_self_governance_code is None
    warning = next(
        item
        for item in dataset.summary.warnings
        if item.code == "profile_placeholder_normalized"
    )
    assert warning.count == 3


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Fatturato", "NaN"),
        ("Fatturato", "Infinity"),
        ("Fatturato", "-Infinity"),
        ("Nr Dipendenti", "NaN"),
        ("Nr Dipendenti", "Infinity"),
        ("Nr Dipendenti", "-Infinity"),
    ],
)
def test_non_finite_profile_numbers_are_structured_json_safe_errors(
    column: str,
    value: str,
) -> None:
    """Verify non-finite revenue/employees never escape structured validation.

    Inputs:
        column: Numeric company profile field being corrupted.
        value: Decimal spelling for NaN or positive/negative infinity.

    Outputs:
        None. A sampled row error is JSON-safe and no overflow exception leaks.
    """

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(
            _workbook_bytes([_row(**{column: value})]),
            "benchmark.xlsx",
            [_question()],
        )

    assert "invalid_profile_value" in _error_codes(caught.value)
    json.dumps(caught.value.summary.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("filename", ["benchmark.xls", "benchmark.csv", "benchmark"])
def test_parser_rejects_non_xlsx_extensions(filename: str) -> None:
    """Verify only the modern XLSX source extension is accepted.

    Inputs:
        filename: Invalid source filename supplied by the parameterization.

    Outputs:
        None. The parser raises a structured ``invalid_extension`` error.
    """

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(
            _workbook_bytes(_two_answer_rows()),
            filename,
            [_question()],
        )

    assert "invalid_extension" in _error_codes(caught.value)


def test_parser_rejects_malformed_and_oversized_xlsx() -> None:
    """Verify invalid ZIP bytes and over-limit source sizes fail before parsing.

    Inputs:
        None. The test supplies malformed and 26 MiB payloads.

    Outputs:
        None. Structured errors identify malformed and oversized sources.
    """

    with pytest.raises(BenchmarkValidationError) as malformed:
        parse_benchmark_workbook(b"not a zip", "benchmark.xlsx", [_question()])
    with pytest.raises(BenchmarkValidationError) as oversized:
        parse_benchmark_workbook(
            b"x" * (26 * 1024 * 1024),
            "benchmark.xlsx",
            [_question()],
        )

    assert "malformed_xlsx" in _error_codes(malformed.value)
    assert "source_too_large" in _error_codes(oversized.value)


def test_parser_rejects_unsafe_zip_compression_ratio() -> None:
    """Verify highly compressed ZIP content is rejected as an XLSX bomb risk.

    Inputs:
        None. A small archive expands to repeated zero bytes.

    Outputs:
        None. The parser raises a structured ``unsafe_zip`` error.
    """

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"0" * (2 * 1024 * 1024))

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(buffer.getvalue(), "benchmark.xlsx", [_question()])

    assert "unsafe_zip" in _error_codes(caught.value)


def test_parser_rejects_missing_sheet_and_non_exact_headers() -> None:
    """Verify the required sheet and exact ordered headers are enforced.

    Inputs:
        None. Workbooks omit the sheet, omit a column, add a column, or reorder.

    Outputs:
        None. Structured errors distinguish sheet and header failures.
    """

    invalid_headers = [
        REQUIRED_COLUMNS[:-1],
        REQUIRED_COLUMNS + ["Unexpected"],
        [REQUIRED_COLUMNS[1], REQUIRED_COLUMNS[0], *REQUIRED_COLUMNS[2:]],
        [f" {REQUIRED_COLUMNS[0]}", *REQUIRED_COLUMNS[1:]],
        [*REQUIRED_COLUMNS[:-1], f"{REQUIRED_COLUMNS[-1]} "],
    ]
    with pytest.raises(BenchmarkValidationError) as missing_sheet:
        parse_benchmark_workbook(
            _workbook_bytes(_two_answer_rows(), sheet_name="Wrong"),
            "benchmark.xlsx",
            [_question()],
        )
    assert "missing_sheet" in _error_codes(missing_sheet.value)

    for headers in invalid_headers:
        with pytest.raises(BenchmarkValidationError) as invalid:
            parse_benchmark_workbook(
                _workbook_bytes(_two_answer_rows(), headers=headers),
                "benchmark.xlsx",
                [_question()],
            )
        assert "invalid_headers" in _error_codes(invalid.value)


def test_header_only_workbook_is_rejected_before_write() -> None:
    """Verify an empty benchmark cannot replace the active HANA version.

    Inputs:
        None. A structurally valid workbook contains only the exact header row.

    Outputs:
        None. Validation fails before the supplied database session is touched.
    """

    class _ForbiddenSession:
        """Fail if an empty workbook reaches any persistence operation."""

        def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            """Reject unexpected SQL execution for a header-only workbook.

            Inputs:
                _args: Unexpected positional database arguments.
                _kwargs: Unexpected keyword database arguments.

            Outputs:
                Never returns; the assertion marks a validation regression.
            """

            raise AssertionError("empty workbook reached persistence")

    with pytest.raises(BenchmarkValidationError) as caught:
        import_benchmark_workbook(
            _workbook_bytes([]),
            "benchmark.xlsx",
            [_question()],
            write=True,
            session=_ForbiddenSession(),
        )

    assert "empty_workbook" in _error_codes(caught.value)


def test_parser_rejects_duplicate_and_conflicting_response_rows() -> None:
    """Verify repeated response identities are never silently accepted.

    Inputs:
        None. One workbook repeats a row; another changes its selected value.

    Outputs:
        None. Duplicate and conflicting rows receive distinct error codes.
    """

    identical_rows = _two_answer_rows()
    identical_rows.append(identical_rows[0].copy())
    conflicting_rows = _two_answer_rows()
    conflicting_rows.append(
        _row(**{"Valore Risposta": "No"})
    )

    with pytest.raises(BenchmarkValidationError) as duplicate:
        parse_benchmark_workbook(
            _workbook_bytes(identical_rows),
            "benchmark.xlsx",
            [_question()],
        )
    with pytest.raises(BenchmarkValidationError) as conflict:
        parse_benchmark_workbook(
            _workbook_bytes(conflicting_rows),
            "benchmark.xlsx",
            [_question()],
        )

    assert "duplicate_row" in _error_codes(duplicate.value)
    assert "conflicting_row" in _error_codes(conflict.value)


def test_parser_rejects_mixed_questionnaire_metadata() -> None:
    """Verify one questionnaire cannot change company/submission metadata midstream.

    Inputs:
        None. The second answer row changes revenue for the same questionnaire.

    Outputs:
        None. A structured mixed-metadata row error blocks the import.
    """

    rows = _two_answer_rows()
    rows[1]["Fatturato"] = 9_999_999

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(
            _workbook_bytes(rows),
            "benchmark.xlsx",
            [_question()],
        )

    assert "mixed_questionnaire_metadata" in _error_codes(caught.value)


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        ({"Livello": 6}, "invalid_level"),
        ({"Livello": 1.5}, "invalid_level"),
        ({"Classe": "9"}, "invalid_class"),
        ({"Classe": "3.5"}, "invalid_class"),
        ({"ID Domanda": "Q.UNKNOWN.01"}, "unknown_question"),
        ({"Valore Risposta": "maybe"}, "invalid_boolean"),
    ],
)
def test_parser_rejects_invalid_semantic_rows(
    updates: dict[str, Any],
    expected_code: str,
) -> None:
    """Verify invalid levels, classes, questions, and booleans block import.

    Inputs:
        updates: Invalid source values applied to one otherwise valid row.
        expected_code: Structured validation code expected from the parser.

    Outputs:
        None. Assertions confirm the sampled error identifies the defect.
    """

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(
            _workbook_bytes([_row(**updates)]),
            "benchmark.xlsx",
            [_question()],
        )

    assert expected_code in _error_codes(caught.value)


def test_parser_rejects_answers_without_a_within_level_canonical_mapping() -> None:
    """Verify excess source answers cannot overflow the canonical level catalog.

    Inputs:
        None. Three source answers compete for two canonical level-one items.

    Outputs:
        None. The third row is rejected as ``unmappable_answer``.
    """

    rows = _two_answer_rows()
    rows.append(
        _row(
            **{
                "ID Risposta": "R.STR.03.01.03",
                "Testo risposta": "Unexpected third practice",
                "Valore Risposta": "No",
            }
        )
    )

    with pytest.raises(BenchmarkValidationError) as caught:
        parse_benchmark_workbook(
            _workbook_bytes(rows),
            "benchmark.xlsx",
            [_question()],
        )

    assert "unmappable_answer" in _error_codes(caught.value)


def test_real_workbook_dry_run_has_expected_counts_and_quality_warnings(
    repo_root: Path,
) -> None:
    """Verify the supplied local workbook fully validates with known warnings.

    Inputs:
        repo_root: Repository root used to locate ignored source data and seeds.

    Outputs:
        None. Assertions cover the exact source counts and warning categories.
    """

    workbook_path = (
        repo_root / "data" / "report data" / "DB_extraction_clustering_NACE.xlsx"
    )
    if not workbook_path.exists():
        pytest.skip("ignored real benchmark workbook is unavailable in this checkout")

    dataset = parse_benchmark_workbook(
        workbook_path.read_bytes(),
        workbook_path.name,
        _italian_framework(repo_root),
    )

    assert dataset.summary.source_sha256 == (
        "f73473c68acf222327687de3be9f37f05d5430cdf55b6911c7459e7da69a7b1a"
    )
    assert dataset.summary.row_count == 3306
    assert dataset.summary.company_count == 6
    assert dataset.summary.questionnaire_count == 6
    assert dataset.summary.question_count == 38
    assert dataset.summary.accepted_count == 3306
    assert dataset.summary.rejected_count == 0
    assert dataset.summary.class_counts == {"class_3": 6}
    assert sum(dataset.summary.nace1_counts.values()) == 6
    assert len(dataset.scores) == 6 * (38 + 7 + 1)
    assert {
        "dimension_mismatch",
        "answer_text_mismatch",
        "profile_placeholder_normalized",
        "score_reconciliation",
    } <= _warning_codes(dataset)


def test_import_service_is_dry_run_by_default() -> None:
    """Verify validation does not access persistence unless write is explicit.

    Inputs:
        None. A sentinel session would fail if the dry-run path touched it.

    Outputs:
        None. The returned JSON-safe summary reports validation only.
    """

    class _ForbiddenSession:
        """Fail if a dry-run importer attempts any database operation."""

        def execute(self, *_args: Any, **_kwargs: Any) -> Any:
            """Reject an unexpected database call.

            Inputs:
                _args: Unexpected positional database arguments.
                _kwargs: Unexpected keyword database arguments.

            Outputs:
                Never returns; the assertion marks a dry-run regression.
            """

            raise AssertionError("dry-run import touched the database")

    summary = import_benchmark_workbook(
        _workbook_bytes(_two_answer_rows()),
        "benchmark.xlsx",
        [_question()],
        session=_ForbiddenSession(),
    )

    assert summary.status == "validated"
    assert summary.write_completed is False
