"""Tests for importing assessment framework seed files."""

from pathlib import Path

import pytest

from scripts import import_assessment_framework
from app.services.framework_importer import (
    load_framework_seed,
    load_italian_framework_translations,
)


def test_load_framework_seed_counts_questions(repo_root: Path) -> None:
    """Verify the framework workbook normalizes into 50 questions.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        None. Assertions validate the normalized framework counts.
    """

    seed = load_framework_seed(
        workbook_path=repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        explanations_path=repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )

    assert len(seed.questions) == 50
    assert {question.dimension for question in seed.questions} >= {"Strategy"}
    assert seed.explanation_count == 50
    assert seed.questions[0].dimension == "Strategy"


def test_ordered_dimensions_preserves_workbook_tab_order(repo_root: Path) -> None:
    """Verify HANA import display order follows the framework export order.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions confirm Strategy remains the first dimension, matching
        the real assessment UI instead of alphabetical sorting.
    """

    seed = load_framework_seed(
        workbook_path=repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        explanations_path=repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )

    assert import_assessment_framework.ordered_dimensions(seed)[0] == "Strategy"


def test_strategy_question_has_level_grouped_answer_ids(repo_root: Path) -> None:
    """Verify generated answer item IDs are stable and level-aware.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        None. Assertions validate the normalized strategy question payload.
    """

    seed = load_framework_seed(
        workbook_path=repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        explanations_path=repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    question = next(item for item in seed.questions if item.question_id == "Q.STR.01.01")

    assert question.explanation
    assert any(
        answer.answer_item_id.startswith("Q.STR.01.01-L1-")
        for answer in question.answer_items
    )
    assert any(answer.level == 5 for answer in question.answer_items)
    assert all(not hasattr(answer, "default_selected") for answer in question.answer_items)
    assert all(not hasattr(answer, "optional") for answer in question.answer_items)


def test_load_italian_framework_translations_match_english_seed(
    repo_root: Path,
) -> None:
    """Verify Italian CSVs provide a complete translation layer for the seed.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions confirm every English question and answer item has a
        matching Italian text row keyed by the canonical identifiers.
    """

    seed = load_framework_seed(
        workbook_path=repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        explanations_path=repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )

    translations = load_italian_framework_translations(
        italian_dir=repo_root / "data" / "sanitized" / "IT",
        questions=seed.questions,
    )

    assert translations.language == "it"
    assert len(translations.dimension_translations) == 7
    assert len(translations.question_translations) == 50
    assert len(translations.answer_item_translations) == 701
    assert translations.dimension_translations["Strategy"] == "Strategia"
    assert (
        translations.question_translations["Q.STR.01.01"].question
        == "La tua organizzazione è dotata di un processo per la pianificazione strategica?"
    )
    assert (
        translations.answer_item_translations["Q.STR.01.01-L1-001"]
        == "La pianificazione segue un processo non strutturato e non formalizzato "
        "a livello procedurale con riferimento al piano strategico pluriennale"
    )


def test_import_cli_defaults_to_no_write(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify omitting --write does not call the HANA write path.

    Inputs:
        monkeypatch: Pytest helper used to replace CLI arguments and HANA writer.
        repo_root: Absolute path to the repository root.
        capsys: Pytest output capture fixture.

    Outputs:
        None. Assertions confirm counts are printed and no HANA write occurs.
    """

    write_calls: list[object] = []
    monkeypatch.setattr(
        import_assessment_framework,
        "write_framework_to_hana",
        lambda *args: write_calls.append(args),
    )
    monkeypatch.setattr(
        import_assessment_framework.sys,
        "argv",
        [
            "import_assessment_framework.py",
            "--workbook",
            str(repo_root / "data" / "sanitized" / "assessment_framework.xlsx"),
            "--explanations",
            str(repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"),
        ],
    )

    import_assessment_framework.main()

    output = capsys.readouterr().out
    assert "Questions: 50" in output
    assert "Dry run complete; no HANA writes performed." in output
    assert write_calls == []


def test_import_cli_write_flag_calls_hana_writer(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    """Verify --write explicitly calls the HANA replacement writer.

    Inputs:
        monkeypatch: Pytest helper used to replace CLI arguments and HANA writer.
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions confirm the fake writer receives the loaded seed and
        source paths without requiring a live HANA connection.
    """

    write_calls: list[tuple[object, Path, Path]] = []

    def fake_write_framework_to_hana(
        seed: object,
        workbook_path: Path,
        explanations_path: Path,
    ) -> None:
        """Record importer write calls for CLI safety assertions.

        Inputs:
            seed: Framework seed loaded by the CLI.
            workbook_path: Workbook path parsed from command arguments.
            explanations_path: Explanations path parsed from command arguments.

        Outputs:
            None. The call is appended to ``write_calls``.
        """

        write_calls.append((seed, workbook_path, explanations_path))

    monkeypatch.setattr(
        import_assessment_framework,
        "write_framework_to_hana",
        fake_write_framework_to_hana,
    )
    monkeypatch.setattr(
        import_assessment_framework.sys,
        "argv",
        [
            "import_assessment_framework.py",
            "--workbook",
            str(repo_root / "data" / "sanitized" / "assessment_framework.xlsx"),
            "--explanations",
            str(repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"),
            "--write",
        ],
    )

    import_assessment_framework.main()

    assert len(write_calls) == 1
    assert write_calls[0][1] == repo_root / "data" / "sanitized" / "assessment_framework.xlsx"
    assert write_calls[0][2] == repo_root / "data" / "sanitized" / "assessment_question_explanations.csv"
