"""Repository-focused tests for persisted assessment benchmark profiles."""

from __future__ import annotations

from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


def _profile(
    *,
    customer_class: str = "class_2",
    nace1: str = "Energy",
) -> object:
    """Return one validated assessment profile fixture.

    Inputs:
        customer_class: Exact configured class for the assessment.
        nace1: Exact level-one NACE cohort label.

    Outputs:
        AssessmentProfile: Profile suitable for repository persistence.
    """

    from app.models.benchmarking import AssessmentProfile

    return AssessmentProfile(
        assessment_id="assessment-1",
        display_name="assessment",
        source_company_id="company-42",
        customer_class=customer_class,
        nace1=nace1,
    )


def test_memory_profile_upsert_and_get_return_independent_copies() -> None:
    """Verify memory persistence replaces profiles and protects stored state."""

    repository = InMemoryAiReviewRepository()
    saved = repository.upsert_assessment_profile(_profile())

    assert saved == _profile()
    saved.display_name = "Caller mutation"
    assert repository.get_assessment_profile("assessment-1") == _profile()

    updated = repository.upsert_assessment_profile(
        _profile(customer_class="class_1", nace1="Manufacturing")
    )
    assert updated.customer_class == "class_1"
    assert updated.nace1 == "Manufacturing"
    assert repository.get_assessment_profile("missing") is None


def test_strict_customer_class_validation_rejects_unknown_profile_class() -> None:
    """Verify profile classes never silently default to the broadest class."""

    import pytest

    from app.services.customer_class_scope import require_customer_class

    assert require_customer_class("class_3") == "class_3"
    with pytest.raises(ValueError, match="Unsupported customer class"):
        require_customer_class("not-a-class")
