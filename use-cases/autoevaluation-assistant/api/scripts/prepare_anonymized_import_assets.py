"""Build neutral, metadata-clean reference files without changing source files.

Examples:
    cd api
    python scripts/prepare_anonymized_import_assets.py \
      --framework-workbook /path/to/framework.xlsx \
      --framework-explanations /path/to/question_explanations.csv \
      --italian-dir /path/to/italian_csvs \
      --glossary-workbook /path/to/glossary.xlsx \
      --joule-explanations-workbook /path/to/explanations.xlsx \
      --benchmark-template /path/to/benchmark_template.xlsx \
      --output-dir ../data/sanitized
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from tqdm import tqdm

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.framework_importer import ITALIAN_DIMENSION_FILES
from scripts.generate_synthetic_benchmark_workbook import generate_workbook


NEUTRAL_CREATOR = "Evaluation Assessment Assistant"
"""Office metadata author used for regenerated import workbooks."""


def copy_workbook_sheets(source: Path, destination: Path, sheets: tuple[str, ...]) -> None:
    """Rebuild selected workbook sheets with neutral Office metadata.

    Inputs:
        source: Existing XLSX containing the required worksheets.
        destination: XLSX path to create or replace.
        sheets: Exact worksheet names whose cell values must be retained.

    Outputs:
        None. A clean workbook containing only the requested sheets is written.
    """

    source_workbook = load_workbook(source, read_only=True, data_only=True)
    destination_workbook = Workbook(write_only=True)
    try:
        for sheet_name in sheets:
            if sheet_name not in source_workbook.sheetnames:
                raise ValueError(f"Missing required sheet {sheet_name!r} in {source}")
            destination_sheet = destination_workbook.create_sheet(sheet_name)
            for row in source_workbook[sheet_name].iter_rows(values_only=True):
                destination_sheet.append(list(row))
        destination_workbook.properties.creator = NEUTRAL_CREATOR
        destination_workbook.properties.lastModifiedBy = NEUTRAL_CREATOR
        destination_workbook.properties.title = "Sanitized assessment import resource"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_workbook.save(destination)
    finally:
        source_workbook.close()
        destination_workbook.close()


def copy_italian_translations(source_dir: Path, destination_dir: Path) -> None:
    """Copy seven translation CSVs to customer-prefix-independent filenames.

    Inputs:
        source_dir: Directory containing one source CSV per framework dimension.
        destination_dir: Directory that receives neutral CSV basenames.

    Outputs:
        None. Exact CSV bytes are copied because CSV files have no Office metadata.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_name, _display_name in ITALIAN_DIMENSION_FILES.values():
        matches = list(source_dir.glob(f"*{file_name}"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one translation CSV ending in {file_name!r}; "
                f"found {len(matches)} in {source_dir}"
            )
        shutil.copyfile(matches[0], destination_dir / file_name)


def prepare_assets(arguments: argparse.Namespace) -> None:
    """Create every neutral framework, Joule, and benchmark import asset.

    Inputs:
        arguments: Parsed source paths and neutral output directory.

    Outputs:
        None. Six resource groups are regenerated beneath ``output_dir``.
    """

    output_dir = arguments.output_dir
    with tqdm(total=6, desc="Preparing anonymized import assets", unit="asset") as progress:
        copy_workbook_sheets(
            arguments.framework_workbook,
            output_dir / "assessment_framework.xlsx",
            ("Ecomarket",),
        )
        progress.update(1)
        shutil.copyfile(
            arguments.framework_explanations,
            output_dir / "assessment_question_explanations.csv",
        )
        progress.update(1)
        copy_italian_translations(arguments.italian_dir, output_dir / "IT")
        progress.update(1)
        copy_workbook_sheets(
            arguments.glossary_workbook,
            output_dir / "assessment_glossary.xlsx",
            ("Glossario ENG", "Glossario ITA"),
        )
        progress.update(1)
        copy_workbook_sheets(
            arguments.joule_explanations_workbook,
            output_dir / "assessment_explanations.xlsx",
            ("Recap-eng", "Dimensioni-ing"),
        )
        progress.update(1)
        generate_workbook(
            arguments.benchmark_template,
            output_dir / "assessment_benchmark.xlsx",
            companies=24,
        )
        progress.update(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit-source anonymized asset preparation parser.

    Inputs:
        None.

    Outputs:
        argparse.ArgumentParser: Parser for source resources and output location.
    """

    parser = argparse.ArgumentParser(
        description="Create metadata-clean, neutral reference import assets"
    )
    parser.add_argument("--framework-workbook", type=Path, required=True)
    parser.add_argument("--framework-explanations", type=Path, required=True)
    parser.add_argument("--italian-dir", type=Path, required=True)
    parser.add_argument("--glossary-workbook", type=Path, required=True)
    parser.add_argument("--joule-explanations-workbook", type=Path, required=True)
    parser.add_argument("--benchmark-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    """Parse CLI arguments and prepare neutral import assets.

    Inputs:
        Command-line arguments from ``sys.argv``.

    Outputs:
        None. Generated file paths are printed after successful completion.
    """

    arguments = build_parser().parse_args()
    prepare_assets(arguments)
    print(f"Sanitized import assets written to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
