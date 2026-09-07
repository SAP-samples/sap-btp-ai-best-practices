"""Company, submission, dimension, and cohort normalization helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.models.assessment import AssessmentQuestion
from app.services.framework_importer import ITALIAN_DIMENSION_FILES

from .constants import OPTIONAL_COMPANY_NAME_COLUMN
from .models import BenchmarkCompany, BenchmarkSubmission
from .validation import WarningAccumulator
from .values import (
    boolean,
    customer_class,
    date_or_none,
    decimal_or_none,
    integer_or_none,
    normalized_text,
    profile_value,
    text_or_none,
)


def company_from_row(
    row: dict[str, Any],
    import_id: str,
    company_id: str,
    warnings: WarningAccumulator,
) -> BenchmarkCompany:
    """Normalize one source row's complete company profile.

    Inputs:
        row: Source row mapping keyed by exact workbook headers.
        import_id: New benchmark import version identity.
        company_id: Normalized source company identity.
        warnings: Aggregate warning collector for profile placeholders.

    Outputs:
        BenchmarkCompany: Immutable normalized company profile.
    """

    def profile(
        column: str,
        *,
        numeric_zero_is_placeholder: bool = True,
    ) -> Any | None:
        """Normalize one profile field and aggregate placeholder warnings.

        Inputs:
            column: Exact source profile column name.
            numeric_zero_is_placeholder: Whether zero represents missing data
                instead of a valid explicit boolean false.

        Outputs:
            Any | None: Meaningful original value or normalized ``None``.
        """

        return profile_value(
            row.get(column),
            column,
            company_id,
            warnings,
            numeric_zero_is_placeholder=numeric_zero_is_placeholder,
        )

    return BenchmarkCompany(
        import_id=import_id,
        source_company_id=company_id,
        company_name=text_or_none(row.get(OPTIONAL_COMPANY_NAME_COLUMN)),
        customer_class=customer_class(row.get("Classe")),
        revenue=decimal_or_none(profile("Fatturato"), "Fatturato"),
        employees=integer_or_none(profile("Nr Dipendenti"), "Nr Dipendenti"),
        nace1=text_or_none(profile("Settore Operativo (NACE) 1")),
        nace2=text_or_none(profile("Settore Operativo (NACE) 2")),
        nace3=text_or_none(profile("Settore Operativo (NACE) 3")),
        company_size=text_or_none(profile("Dimensione Azienda")),
        legal_form=text_or_none(profile("Forma Giuridica")),
        geographic_presence=text_or_none(profile("Presenza Geografica")),
        is_listed=boolean(
            profile("Quotata", numeric_zero_is_placeholder=False),
            "Quotata",
            allow_none=True,
        ),
        is_public_contracting_client=boolean(
            profile(
                "Committente Contratti Pubblici",
                numeric_zero_is_placeholder=False,
            ),
            "Committente Contratti Pubblici",
            allow_none=True,
        ),
        uses_self_governance_code=boolean(
            profile(
                "Adesione Codice di  Autodisciplina",
                numeric_zero_is_placeholder=False,
            ),
            "Adesione Codice di Autodisciplina",
            allow_none=True,
        ),
    )


def submission_from_row(
    row: dict[str, Any],
    import_id: str,
    questionnaire_id: str,
    company_id: str,
) -> BenchmarkSubmission:
    """Normalize one source row's questionnaire-level metadata.

    Inputs:
        row: Source row mapping keyed by exact workbook headers.
        import_id: New benchmark import version identity.
        questionnaire_id: Normalized source questionnaire identity.
        company_id: Normalized source company identity.

    Outputs:
        BenchmarkSubmission: Immutable normalized submission metadata.
    """

    return BenchmarkSubmission(
        import_id=import_id,
        questionnaire_id=questionnaire_id,
        source_company_id=company_id,
        submission_date=date_or_none(row.get("Data Sottomissione"), "Data Sottomissione"),
        extraction_date=date_or_none(row.get("Data Estrazione"), "Data Estrazione"),
        release_status=text_or_none(row.get("Stato Questionario")),
        raw_assessment_category=text_or_none(row.get("Assessment Score")),
    )


def dimension_aliases(question: AssessmentQuestion) -> set[str]:
    """Return normalized canonical and Italian labels for a question dimension.

    Inputs:
        question: Canonical framework question.

    Outputs:
        set[str]: Normalized labels accepted as equivalent source dimensions.
    """

    aliases = {question.dimension}
    translated = ITALIAN_DIMENSION_FILES.get(question.dimension)
    if translated is not None:
        aliases.add(translated[1])
    return {normalized_text(alias) for alias in aliases}


def cohort_counts(
    companies: list[BenchmarkCompany],
    attribute: str,
) -> dict[str, int]:
    """Count non-null company cohort labels for one profile attribute.

    Inputs:
        companies: Normalized unique benchmark companies.
        attribute: ``customer_class`` or NACE attribute name to count.

    Outputs:
        dict[str, int]: Stable sorted label-to-company counts.
    """

    counts = Counter(
        value
        for company in companies
        if (value := getattr(company, attribute)) is not None
    )
    return dict(sorted(counts.items()))
