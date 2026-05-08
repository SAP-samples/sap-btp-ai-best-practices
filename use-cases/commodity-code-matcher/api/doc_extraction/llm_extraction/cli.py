"""
Command line interface for the LLM PDF extraction module.

This script provides a lightweight wrapper around :func:`extract` so the module
can be executed directly from the shell or through ``python -m``.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load from the single .env file at api/.env
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from doc_extraction.llm_extraction.extractor import extract, to_table
else:
    from .extractor import extract, to_table


def _build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Extract invoice data using an LLM")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF document")
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write the raw JSON extraction result",
    )
    parser.add_argument(
        "--table",
        type=Path,
        help="Optional path to write the tabular output as CSV",
    )
    parser.add_argument(
        "--no-dataframe",
        action="store_true",
        help="Skip building a Pandas DataFrame (falls back to list of dicts)",
    )
    return parser


def output_table(table: Any, output_path: Path) -> None:
    """
    Persist the table to disk.

    DataFrames are written as CSV, while list-of-dict payloads are serialised as
    JSON so that the output format stays consumable without extra dependencies.
    """

    if hasattr(table, "to_csv"):
        table.to_csv(output_path, index=False)  # type: ignore[call-arg]
    else:
        output_path.write_text(json.dumps(table, indent=2, ensure_ascii=False))


def _table_as_records(table: Any) -> Any:
    """Return a JSON-serialisable representation of the table."""

    if table is None:
        return None
    if hasattr(table, "to_dict"):
        try:
            return table.to_dict(orient="records")  # type: ignore[call-arg]
        except TypeError:
            return table.to_dict()
    if isinstance(table, (list, tuple)):
        return list(table)
    return table


def _json_ready_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return a deep copy of result with DataFrame/list payloads made JSON safe.
    """

    serialisable = dict(result)
    if "table" in serialisable:
        serialisable["table"] = _table_as_records(serialisable["table"])
    return serialisable


def main(args: argparse.Namespace | None = None) -> None:
    """Execute the CLI flow."""

    parser = _build_parser()
    parsed = args or parser.parse_args()

    result = extract(parsed.pdf_path, return_dataframe=not parsed.no_dataframe)
    table_obj = result.get("table")

    json_payload = _json_ready_result(result)

    if parsed.json:
        parsed.json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(json_payload, indent=2, ensure_ascii=False))

    if parsed.table:
        table = table_obj or to_table(result)
        output_table(table, parsed.table)


if __name__ == "__main__":
    main()

