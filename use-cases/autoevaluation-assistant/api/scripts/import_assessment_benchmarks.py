"""Validate or import an assessment benchmark workbook.

Examples:
    cd api
    python scripts/import_assessment_benchmarks.py --workbook path/to/benchmarks.xlsx
    python scripts/import_assessment_benchmarks.py --workbook path/to/benchmarks.xlsx --write
    python scripts/import_assessment_benchmarks.py --workbook path/to/benchmarks.xlsx \
        --write --replace-history --backup-confirmed
    python scripts/import_assessment_benchmarks.py --workbook path/to/benchmarks.xlsx --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.db import create_hana_engine
from app.models.assessment import AssessmentQuestion
from app.services.benchmark_import import BenchmarkValidationError, parse_benchmark_workbook
from app.services.benchmark_import.models import (
    BenchmarkDataset,
    BenchmarkValidationSummary,
)
from app.services.benchmark_import.repository import write_benchmark_dataset
from app.services.framework_importer import (
    load_framework_seed,
    load_italian_framework_translations,
)


REPOSITORY_ROOT = API_ROOT.parent
"""Repository root used to resolve default local framework seed paths."""


def load_localized_framework_questions(
    workbook_path: Path,
    explanations_path: Path,
    italian_dir: Path,
) -> list[AssessmentQuestion]:
    """Load canonical framework IDs/dimensions with Italian comparison text.

    Inputs:
        workbook_path: Canonical ``assessment_framework.xlsx`` framework seed path.
        explanations_path: Canonical question explanations CSV path.
        italian_dir: Directory containing Italian framework translation CSVs.

    Outputs:
        list[AssessmentQuestion]: Canonical questions localized to source text.
    """

    seed = load_framework_seed(workbook_path, explanations_path)
    translations = load_italian_framework_translations(italian_dir, seed.questions)
    questions: list[AssessmentQuestion] = []
    for question in seed.questions:
        question_translation = translations.question_translations[question.question_id]
        questions.append(
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
    return questions


def _write_dataset(
    dataset: BenchmarkDataset,
    *,
    batch_size: int,
    replace_history: bool,
) -> BenchmarkValidationSummary:
    """Persist a validated dataset through existing HANA engine conventions.

    Inputs:
        dataset: Fully validated normalized workbook payload.
        batch_size: Positive maximum child rows per executemany call.
        replace_history: Whether inactive benchmark versions must be purged.

    Outputs:
        BenchmarkValidationSummary: Active or idempotent no-op write summary.
    """

    engine = create_hana_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        if replace_history:
            result = write_benchmark_dataset(
                session,
                dataset,
                batch_size=batch_size,
                replace_history=True,
            )
        else:
            result = write_benchmark_dataset(session, dataset, batch_size=batch_size)
        return dataset.summary.model_copy(
            update={
                "import_id": result.import_id,
                "status": result.status,
                "write_completed": not result.no_op,
                "no_op": result.no_op,
            }
        )
    finally:
        session.close()
        engine.dispose()


def run_import(arguments: argparse.Namespace) -> BenchmarkValidationSummary:
    """Execute CLI framework loading, pure validation, and optional HANA write.

    Inputs:
        arguments: Parsed CLI namespace containing paths and mode flags.

    Outputs:
        BenchmarkValidationSummary: Dry-run, active, or no-op result.
    """

    if not arguments.workbook.is_file():
        raise ValueError(f"Benchmark workbook does not exist: {arguments.workbook}")
    if arguments.replace_history and not arguments.write:
        raise ValueError("--replace-history is write-only and requires --write")
    if arguments.replace_history and not arguments.backup_confirmed:
        raise ValueError(
            "--replace-history requires --backup-confirmed after a successful backup"
        )

    # Stage-based progress keeps the CLI informative while the decomposed
    # parser/repository remain side-effect-free and reusable by future routes.
    stage_count = 3 if arguments.write else 2
    with tqdm(
        total=stage_count,
        desc="Assessment benchmark import",
        unit="stage",
        disable=True if arguments.json else None,
    ) as progress:
        questions = load_localized_framework_questions(
            arguments.framework_workbook,
            arguments.explanations,
            arguments.italian_dir,
        )
        progress.update(1)
        dataset = parse_benchmark_workbook(
            arguments.workbook.read_bytes(),
            arguments.workbook.name,
            questions,
        )
        progress.update(1)
        if not arguments.write:
            return dataset.summary
        summary = _write_dataset(
            dataset,
            batch_size=arguments.batch_size,
            replace_history=arguments.replace_history,
        )
        progress.update(1)
        return summary


def _print_text_summary(summary: BenchmarkValidationSummary) -> None:
    """Print a concise human-readable validation or import summary.

    Inputs:
        summary: JSON-safe benchmark result to render for a terminal user.

    Outputs:
        None. Counts, SHA, warnings, and mode outcome are printed to stdout.
    """

    print(f"Status: {summary.status}")
    print(f"SHA-256: {summary.source_sha256}")
    print(
        f"Rows: {summary.accepted_count:,} accepted / "
        f"{summary.rejected_count:,} rejected"
    )
    print(f"Companies: {summary.company_count:,}")
    print(f"Questionnaires: {summary.questionnaire_count:,}")
    print(f"Questions: {summary.question_count:,}")
    if summary.warnings:
        print("Warnings:")
        for warning in summary.warnings:
            print(f"  - {warning.code}: {warning.count:,} ({warning.message})")
    if summary.status == "validated":
        print("Dry run complete; no HANA writes performed.")
    elif summary.no_op:
        print(f"Active workbook already matches import {summary.import_id}; no rows changed.")
    elif summary.write_completed:
        print(f"Activated benchmark import {summary.import_id}.")


def _print_summary(summary: BenchmarkValidationSummary, *, as_json: bool) -> None:
    """Render a benchmark result as JSON or terminal-oriented text.

    Inputs:
        summary: JSON-safe dry-run or write result.
        as_json: Whether to emit machine-readable JSON.

    Outputs:
        None. The chosen representation is printed to stdout.
    """

    if as_json:
        print(
            json.dumps(
                summary.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    _print_text_summary(summary)


def build_parser() -> argparse.ArgumentParser:
    """Build the dry-run-first assessment benchmark import argument parser.

    Inputs:
        None.

    Outputs:
        argparse.ArgumentParser: Parser for workbook, framework, mode, and output.
    """

    parser = argparse.ArgumentParser(
        description="Validate or atomically import versioned assessment benchmarks"
    )
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument(
        "--framework-workbook",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "sanitized" / "assessment_framework.xlsx",
    )
    parser.add_argument(
        "--explanations",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    parser.add_argument(
        "--italian-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "sanitized" / "IT",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist and activate after successful validation (default: dry run)",
    )
    parser.add_argument(
        "--replace-history",
        action="store_true",
        help="After activation, delete all inactive benchmark versions and BLOBs",
    )
    parser.add_argument(
        "--backup-confirmed",
        action="store_true",
        help="Confirm that an operator-verified reference-data backup exists",
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main() -> None:
    """Run the benchmark importer CLI in dry-run or explicit write mode.

    Inputs:
        Command-line arguments parsed from ``sys.argv``.

    Outputs:
        None. Validation or import results are written to standard output.
    """

    arguments = build_parser().parse_args()
    try:
        summary = run_import(arguments)
    except BenchmarkValidationError as exc:
        _print_summary(exc.summary, as_json=arguments.json)
        raise SystemExit(2) from exc
    _print_summary(summary, as_json=arguments.json)
    if arguments.replace_history and not arguments.json:
        print("Inactive benchmark history removed; the active import was preserved.")


if __name__ == "__main__":
    main()
