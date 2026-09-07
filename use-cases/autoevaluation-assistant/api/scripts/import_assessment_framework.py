"""Import assessment framework seed files.

Example commands:
    cd api
    python scripts/import_assessment_framework.py \
      --workbook ../data/sanitized/assessment_framework.xlsx \
      --explanations ../data/sanitized/assessment_question_explanations.csv \
      --italian-dir ../data/sanitized/IT \
      --dry-run
    python scripts/import_assessment_framework.py \
      --workbook ../data/sanitized/assessment_framework.xlsx \
      --explanations ../data/sanitized/assessment_question_explanations.csv \
      --italian-dir ../data/sanitized/IT \
      --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.db import create_hana_engine
from app.services.ai_review_repository.hana import HanaAiReviewRepository
from app.services.framework_importer import (
    FrameworkSeed,
    FrameworkTranslationSeed,
    build_topic_translation_rows,
    load_framework_seed,
    load_italian_framework_translations,
)


def ordered_dimensions(seed: FrameworkSeed) -> list[str]:
    """Return dimensions in their first-seen workbook order.

    Inputs:
        seed: Normalized framework seed loaded from the framework export.

    Outputs:
        list[str]: Distinct dimension names ordered by first question
        occurrence, matching the real assessment tab order.
    """

    dimensions: list[str] = []
    seen_dimensions: set[str] = set()
    for question in seed.questions:
        if question.dimension in seen_dimensions:
            continue
        dimensions.append(question.dimension)
        seen_dimensions.add(question.dimension)
    return dimensions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the framework importer.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Import assessment framework files.")
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--explanations", required=True, type=Path)
    parser.add_argument(
        "--italian-dir",
        type=Path,
        default=None,
        help="Directory containing Italian framework CSV translations.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser.parse_args()


def write_framework_to_hana(
    seed: FrameworkSeed,
    workbook_path: Path,
    explanations_path: Path,
    translations: FrameworkTranslationSeed | None = None,
) -> None:
    """Replace framework reference data in HANA with an imported seed.

    Inputs:
        seed: Normalized framework questions, answer items, and explanation
            counts loaded from the source files.
        workbook_path: Source workbook path used for import metadata.
        explanations_path: Source explanations CSV path used for import metadata.
        translations: Optional Italian translation seed loaded from CSV files.

    Outputs:
        None. The function creates schema if needed, deletes prior framework
        reference rows, inserts the imported rows, and commits the transaction.
    """

    engine = create_hana_engine()
    Session = sessionmaker(bind=engine)
    answer_count = sum(len(question.answer_items) for question in seed.questions)

    with Session() as session:
        repository = HanaAiReviewRepository(session)
        repository.create_schema()

        session.execute(text("delete from assessment_answer_item_translations"))
        session.execute(text("delete from assessment_answer_items"))
        session.execute(text("delete from assessment_question_topic_translations"))
        session.execute(text("delete from assessment_question_translations"))
        session.execute(text("delete from assessment_question_explanations"))
        session.execute(text("delete from assessment_questions"))
        session.execute(text("delete from assessment_dimension_translations"))
        session.execute(text("delete from assessment_dimensions"))
        session.execute(text("delete from assessment_framework_imports"))

        dimensions = ordered_dimensions(seed)
        for display_order, dimension in enumerate(dimensions, start=1):
            question_count = sum(
                1 for question in seed.questions if question.dimension == dimension
            )
            session.execute(
                text(
                    "insert into assessment_dimensions "
                    "(dimension, display_order, question_count) "
                    "values (:dimension, :display_order, :question_count)"
                ),
                {
                    "dimension": dimension,
                    "display_order": display_order,
                    "question_count": question_count,
                },
            )
            if translations is not None:
                session.execute(
                    text(
                        "insert into assessment_dimension_translations "
                        "(dimension, language, display_name) "
                        "values (:dimension, :language, :display_name)"
                    ),
                    {
                        "dimension": dimension,
                        "language": translations.language,
                        "display_name": translations.dimension_translations[dimension],
                    },
                )

        for display_order, question in enumerate(seed.questions, start=1):
            session.execute(
                text(
                    "insert into assessment_questions "
                    "(question_id, dimension, section, question_text, display_order) "
                    "values (:question_id, :dimension, :section, "
                    ":question_text, :display_order)"
                ),
                {
                    "question_id": question.question_id,
                    "dimension": question.dimension,
                    "section": question.section,
                    "question_text": question.question,
                    "display_order": display_order,
                },
            )
            if translations is not None:
                question_translation = translations.question_translations[
                    question.question_id
                ]
                session.execute(
                    text(
                        "insert into assessment_question_translations "
                        "(question_id, language, section, question_text) "
                        "values (:question_id, :language, :section, :question_text)"
                    ),
                    {
                        "question_id": question.question_id,
                        "language": translations.language,
                        "section": question_translation.section,
                        "question_text": question_translation.question,
                    },
                )

            if question.explanation:
                session.execute(
                    text(
                        "insert into assessment_question_explanations "
                        "(question_id, dimension, question_text, explanation) "
                        "values (:question_id, :dimension, :question_text, "
                        ":explanation)"
                    ),
                    {
                        "question_id": question.question_id,
                        "dimension": question.dimension,
                        "question_text": question.question,
                        "explanation": question.explanation,
                    },
                )

            for item in question.answer_items:
                session.execute(
                    text(
                        "insert into assessment_answer_items "
                        "(answer_item_id, question_id, level, item_index, "
                        "answer_text) "
                        "values (:answer_item_id, :question_id, :level, "
                        ":item_index, :answer_text)"
                    ),
                    {
                        "answer_item_id": item.answer_item_id,
                        "question_id": item.question_id,
                        "level": item.level,
                        "item_index": item.item_index,
                        "answer_text": item.text,
                    },
                )
                if translations is not None:
                    session.execute(
                        text(
                            "insert into assessment_answer_item_translations "
                            "(answer_item_id, language, answer_text) "
                            "values (:answer_item_id, :language, :answer_text)"
                        ),
                        {
                            "answer_item_id": item.answer_item_id,
                            "language": translations.language,
                            "answer_text": translations.answer_item_translations[
                                item.answer_item_id
                            ],
                    },
                )

        for topic_row in build_topic_translation_rows(seed.questions):
            session.execute(
                text(
                    "insert into assessment_question_topic_translations "
                    "(question_id, language, topic_title) "
                    "values (:question_id, :language, :topic_title)"
                ),
                topic_row,
            )

        session.execute(
            text(
                "insert into assessment_framework_imports "
                "(import_id, source_workbook, source_explanations, "
                "dimension_count, question_count, answer_item_count, "
                "explanation_count, status) "
                "values (:import_id, :source_workbook, :source_explanations, "
                ":dimension_count, :question_count, :answer_item_count, "
                ":explanation_count, :status)"
            ),
            {
                "import_id": f"import-{uuid4()}",
                "source_workbook": workbook_path.name,
                "source_explanations": explanations_path.name,
                "dimension_count": len(dimensions),
                "question_count": len(seed.questions),
                "answer_item_count": answer_count,
                "explanation_count": seed.explanation_count,
                "status": "completed",
            },
        )
        session.commit()


def main() -> None:
    """Run the framework import CLI.

    Returns:
        None. The command prints import counts and exits.
    """

    args = parse_args()
    seed = load_framework_seed(args.workbook, args.explanations)
    translations = (
        load_italian_framework_translations(args.italian_dir, seed.questions)
        if args.italian_dir is not None
        else None
    )
    answer_count = sum(len(question.answer_items) for question in seed.questions)

    print(f"Questions: {len(seed.questions)}")
    print(f"Answer items: {answer_count}")
    print(f"Explanations: {seed.explanation_count}")
    if translations is not None:
        print(f"Italian questions: {len(translations.question_translations)}")
        print(f"Italian answer items: {len(translations.answer_item_translations)}")

    if not args.write:
        print("Dry run complete; no HANA writes performed.")
        return

    if translations is None:
        write_framework_to_hana(seed, args.workbook, args.explanations)
    else:
        write_framework_to_hana(seed, args.workbook, args.explanations, translations)
    print("HANA import complete.")


if __name__ == "__main__":
    main()
