"""Tests for Assessment knowledge workbook ingestion."""

from pathlib import Path

import pytest

from app.services.joule_knowledge import normalize_question_id
from app.services.joule_knowledge_importer import load_joule_knowledge_seed, source_metadata


def test_load_joule_knowledge_seed_parses_real_workbooks(repo_root: Path) -> None:
    """Verify the two PoC workbooks normalize into HANA-ready resources.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions validate workbook counts, multilingual glossary rows,
        question explanations, and dimension descriptions.
    """

    data_dir = repo_root / "data" / "sanitized"

    seed = load_joule_knowledge_seed(
        glossary_workbook=data_dir / "assessment_glossary.xlsx",
        explanations_workbook=(
            data_dir / "assessment_explanations.xlsx"
        ),
    )

    assert len(seed.glossary_terms) == 549
    assert len(seed.question_explanations) == 50
    assert len(seed.dimensions) == 7
    assert seed.question_explanations[-1].question_id == "Q.FDR.12.01"
    assert seed.dimensions[0].name == "Strategy"
    assert any(
        term.language == "it"
        and term.term == "AI"
        and "Tecnologia" in term.definition
        for term in seed.glossary_terms
    )


def test_load_joule_knowledge_seed_preserves_duplicate_glossary_terms(
    repo_root: Path,
) -> None:
    """Verify duplicate source glossary rows remain distinct searchable terms.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions confirm duplicated English terms from the workbook are
        not collapsed during ingestion.
    """

    data_dir = repo_root / "data" / "sanitized"

    seed = load_joule_knowledge_seed(
        glossary_workbook=data_dir / "assessment_glossary.xlsx",
        explanations_workbook=(
            data_dir / "assessment_explanations.xlsx"
        ),
    )
    procurement_terms = [
        term
        for term in seed.glossary_terms
        if term.language == "en" and term.normalized_term == "procurement"
    ]

    assert len(procurement_terms) == 2
    assert {term.source_row for term in procurement_terms} == {11, 182}


def test_source_metadata_uses_neutral_basenames(repo_root: Path) -> None:
    """Verify import tracking never persists a local filesystem path.

    Inputs:
        repo_root: Repository root containing sanitized Joule resources.

    Outputs:
        None. Metadata contains neutral basenames with no parent directory.
    """

    data_dir = repo_root / "data" / "sanitized"
    seed = load_joule_knowledge_seed(
        data_dir / "assessment_glossary.xlsx",
        data_dir / "assessment_explanations.xlsx",
    )

    assert [row["source_path"] for row in source_metadata(seed)] == [
        "assessment_glossary.xlsx",
        "assessment_explanations.xlsx",
    ]


def test_normalize_question_id_repairs_known_workbook_typo() -> None:
    """Verify the importer fixes only the documented FDR question typo.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm typo repair and strict validation behavior.
    """

    assert normalize_question_id("Q,FDR.12.01") == "Q.FDR.12.01"
    assert normalize_question_id(" Q.STR.01.01 ") == "Q.STR.01.01"
    with pytest.raises(ValueError, match="Malformed question ID"):
        normalize_question_id("Q-STR-01-01")
