"""Import Assessment knowledge workbooks into HANA.

Example commands:
    cd api
    python scripts/import_joule_knowledge_resources.py \
      --glossary-workbook "../data/sanitized/assessment_glossary.xlsx" \
      --explanations-workbook "../data/sanitized/assessment_explanations.xlsx" \
      --dry-run
    python scripts/import_joule_knowledge_resources.py \
      --glossary-workbook "../data/sanitized/assessment_glossary.xlsx" \
      --explanations-workbook "../data/sanitized/assessment_explanations.xlsx" \
      --embedding-model text-embedding-3-small \
      --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy.orm import sessionmaker

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.db import create_hana_engine
from app.services.joule_knowledge_importer import (
    DEFAULT_EMBEDDING_MODEL,
    GenAiHubEmbeddingClient,
    build_question_embedding_rows,
    load_joule_knowledge_seed,
    JouleKnowledgeSeed,
)
from app.services.joule_knowledge_repository import HanaJouleKnowledgeRepository


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Joule knowledge importer.

    Inputs:
        None. Arguments are read from ``sys.argv``.

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Import Assessment glossary and explanation workbooks into HANA."
    )
    parser.add_argument("--glossary-workbook", required=True, type=Path)
    parser.add_argument("--explanations-workbook", required=True, type=Path)
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="SAP Gen AI Hub embedding model used for semantic question retrieval.",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Embedding request batch size.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser.parse_args()


def write_joule_knowledge_to_hana(
    glossary_workbook: Path,
    explanations_workbook: Path,
    embedding_model: str,
    batch_size: int,
    seed: JouleKnowledgeSeed | None = None,
    question_embeddings: dict[str, tuple[list[float], list[float]]] | None = None,
    show_progress: bool = True,
) -> None:
    """Load workbooks, generate embeddings, and replace HANA knowledge rows.

    Inputs:
        glossary_workbook: Path to the bilingual glossary workbook.
        explanations_workbook: Path to the question and dimension explanation
            workbook.
        embedding_model: SAP Gen AI Hub embedding model name.
        batch_size: Number of texts embedded per request.
        seed: Optional preloaded Joule knowledge seed.
        question_embeddings: Optional precomputed embeddings keyed by question ID.
        show_progress: Whether to show an import progress bar.

    Outputs:
        None. Data is written to HANA in one SQLAlchemy session.
    """

    if seed is None:
        seed = load_joule_knowledge_seed(glossary_workbook, explanations_workbook)
    if question_embeddings is None:
        embedding_client = GenAiHubEmbeddingClient(model_name=embedding_model)
        question_embeddings = build_question_embedding_rows(
            seed,
            embedding_client=embedding_client,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    engine = create_hana_engine()
    Session = sessionmaker(bind=engine)
    with Session() as session:
        repository = HanaJouleKnowledgeRepository(session)
        repository.replace_knowledge_seed(
            seed,
            question_embeddings=question_embeddings,
            embedding_model=embedding_model,
        )
        session.commit()


def main() -> None:
    """Run the Joule knowledge import CLI.

    Inputs:
        None. Command arguments specify source files and dry-run/write mode.

    Outputs:
        None. The command prints import counts and optionally writes to HANA.
    """

    args = parse_args()
    seed = load_joule_knowledge_seed(args.glossary_workbook, args.explanations_workbook)

    print("Joule knowledge resources loaded:")
    print(f"  Glossary terms: {len(seed.glossary_terms)}")
    print(f"  Question explanations: {len(seed.question_explanations)}")
    print(f"  Dimensions: {len(seed.dimensions)}")
    print(f"  Embedding model: {args.embedding_model}")

    if not args.write:
        print("Dry run complete; no HANA writes performed.")
        return

    write_joule_knowledge_to_hana(
        glossary_workbook=args.glossary_workbook,
        explanations_workbook=args.explanations_workbook,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print("Joule knowledge resources written to HANA.")


if __name__ == "__main__":
    main()
