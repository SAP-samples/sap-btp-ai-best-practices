"""Build identity-free benchmark validation summaries for HTTP responses."""

from __future__ import annotations

import re

from app.models.benchmarking import (
    PublicBenchmarkRowError,
    PublicBenchmarkValidationSummary,
    PublicBenchmarkWarning,
)

from .models import BenchmarkValidationSummary

_ROW_NUMBER_PATTERN = re.compile(r"\brow\s+(\d+)\b", re.IGNORECASE)
"""Recognize worksheet row references without retaining surrounding samples."""

_PUBLIC_WARNING_MESSAGES = {
    "answer_text_mismatch": "Some answer text differs from the canonical framework.",
    "dimension_mismatch": "Some source dimensions differ from the canonical framework.",
    "profile_placeholder_normalized": "Some blank profile placeholders were normalized.",
    "score_reconciliation": "Some calculated scores differ from supplied audit scores.",
}
"""Stable warning copy that cannot interpolate workbook source values."""

_PUBLIC_ERROR_MESSAGES = {
    "conflicting_answer_catalog": "An answer has conflicting maturity metadata.",
    "conflicting_row": "A response identity appears more than once with conflicting data.",
    "duplicate_row": "A response identity appears more than once.",
    "empty_workbook": "The workbook must contain at least one response row.",
    "invalid_boolean": "A field contains an unsupported yes/no value.",
    "invalid_class": "A company class value is unsupported.",
    "invalid_date": "A date field is invalid.",
    "invalid_extension": "The benchmark source must use the XLSX extension.",
    "invalid_headers": "The workbook headers do not match the required contract.",
    "invalid_level": "A maturity level is invalid.",
    "invalid_profile_value": "A company profile field is invalid.",
    "invalid_score": "A supplied audit score is invalid.",
    "malformed_xlsx": "The workbook could not be opened as a valid XLSX file.",
    "missing_identifier": "One or more required identifiers are blank.",
    "missing_sheet": "The workbook is missing its required worksheet.",
    "mixed_company_metadata": "A company has conflicting profile metadata.",
    "mixed_dimension_score": "A questionnaire has conflicting dimension scores.",
    "mixed_questionnaire_metadata": "A questionnaire has conflicting metadata.",
    "source_too_large": "The workbook exceeds the maximum accepted size.",
    "unmappable_answer": "An answer cannot be mapped to the canonical framework.",
    "unknown_question": "The workbook references an unknown framework question.",
    "unsafe_zip": "The workbook container does not meet upload safety rules.",
}
"""Stable validation copy that never embeds source identifiers or values."""


def _public_warning_message(code: str) -> str:
    """Return generic client-safe text for one warning code.

    Inputs:
        code: Stable internal benchmark warning category.

    Outputs:
        str: Public message without workbook-derived values.
    """

    return _PUBLIC_WARNING_MESSAGES.get(code, "Workbook quality warning detected.")


def _public_error_message(code: str) -> str:
    """Return generic client-safe text for one validation error code.

    Inputs:
        code: Stable internal benchmark validation category.

    Outputs:
        str: Public message without workbook-derived values.
    """

    return _PUBLIC_ERROR_MESSAGES.get(code, "Workbook row failed validation.")


def _warning_row_numbers(samples: list[str]) -> list[int]:
    """Extract ordered unique worksheet rows from internal warning samples.

    Inputs:
        samples: Trusted-parser samples that may also contain source identities.

    Outputs:
        list[int]: Only positive row numbers; all surrounding sample text is lost.
    """

    return list(
        dict.fromkeys(
            int(match.group(1))
            for sample in samples
            for match in _ROW_NUMBER_PATTERN.finditer(sample)
            if int(match.group(1)) > 0
        )
    )


def public_benchmark_validation_summary(
    summary: BenchmarkValidationSummary,
) -> PublicBenchmarkValidationSummary:
    """Convert an internal audit summary into an identity-free API response.

    Inputs:
        summary: Full trusted parser/CLI summary, including raw bounded samples.

    Outputs:
        PublicBenchmarkValidationSummary: Safe aggregate fields, generic messages,
        and row locations without company, questionnaire, or answer identities.
    """

    fields = summary.model_dump(
        exclude={"warnings", "sampled_errors"},
        mode="python",
    )
    return PublicBenchmarkValidationSummary(
        **fields,
        warnings=[
            PublicBenchmarkWarning(
                code=warning.code,
                message=_public_warning_message(warning.code),
                count=warning.count,
                row_numbers=_warning_row_numbers(warning.samples),
            )
            for warning in summary.warnings
        ],
        sampled_errors=[
            PublicBenchmarkRowError(
                row_number=error.row_number,
                code=error.code,
                message=_public_error_message(error.code),
            )
            for error in summary.sampled_errors
        ],
    )
