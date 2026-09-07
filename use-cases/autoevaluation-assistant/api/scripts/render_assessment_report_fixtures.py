"""Render deterministic assessment PDF fixtures for page-level visual QA.

Example commands:
    cd api
    ../.venv/bin/python scripts/render_assessment_report_fixtures.py \
        --customer-class class_5 \
        --output-dir ../tmp/pdfs/assessment-report
    pdftoppm -png -r 120 \
        ../tmp/pdfs/assessment-report/assessment-report-class_5-en.pdf \
        ../tmp/pdfs/assessment-report/rendered/class_5-en
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfReader

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.assessment_report_pdf import render_assessment_report_pdf
from app.services.customer_class_scope import (
    customer_class_options,
    load_customer_class_scope,
)
from tests.report_fixtures import make_class_source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deterministic fixture-render command line.

    Inputs:
        argv: Optional argument sequence; process arguments are used when omitted.

    Outputs:
        argparse.Namespace: Validated customer class and output directory.
    """

    parser = argparse.ArgumentParser(
        description="Render deterministic English and Italian assessment fixtures."
    )
    configured_classes = tuple(
        option["value"] for option in customer_class_options("en")
    )
    parser.add_argument(
        "--customer-class",
        choices=configured_classes,
        default=load_customer_class_scope()["default_customer_class"],
        help="Configured customer-class scope used by both fixtures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../tmp/pdfs/assessment-report"),
        help="Directory receiving the deterministic PDF fixtures.",
    )
    return parser.parse_args(argv)


def render_fixture(
    *,
    repo_root: Path,
    output_path: Path,
    customer_class: str,
    language: str,
) -> int:
    """Render one representative fixture and return its page count.

    Inputs:
        repo_root: Repository root containing canonical framework source files.
        output_path: Destination PDF path.
        customer_class: Configured class controlling question and answer scope.
        language: Report language, ``en`` or ``it``.

    Outputs:
        int: Number of pages in the written PDF.
    """

    source = make_class_source(
        repo_root,
        customer_class=customer_class,
        language=language,
        include_peers=True,
        long_labels=True,
    )
    output_path.write_bytes(render_assessment_report_pdf(source))
    return len(PdfReader(output_path).pages)


def main(argv: Sequence[str] | None = None) -> None:
    """Render all visual-QA fixtures and print paths with page counts.

    Inputs:
        argv: Optional CLI arguments for tests or direct invocation.

    Outputs:
        None. Two PDFs are written and one audit line is printed per file.
    """

    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    fixtures = (
        (f"assessment-report-{args.customer_class}-en.pdf", "en"),
        (f"assessment-report-{args.customer_class}-it.pdf", "it"),
    )
    for file_name, language in fixtures:
        output_path = output_dir / file_name
        page_count = render_fixture(
            repo_root=repo_root,
            output_path=output_path,
            customer_class=args.customer_class,
            language=language,
        )
        print(f"{output_path} | pages={page_count}")


if __name__ == "__main__":
    main()
