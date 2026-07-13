#!/usr/bin/env python3
"""
Examples:
  python api/scripts/generate_anonymized_seed_artifacts.py
  python api/scripts/generate_anonymized_seed_artifacts.py --output-dir api/demo_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "api"

for candidate in (str(REPO_ROOT), str(API_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from app.nbo.anonymized_seed_data import synthetic_runtime_datasets  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse the destination for generated public anonymized artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate public anonymized seed workbooks and source-document placeholders."
    )
    parser.add_argument(
        "--output-dir",
        default=str(API_ROOT / "demo_data"),
        help="Root output directory for public demo data artifacts.",
    )
    return parser.parse_args()


def _source_documents_from_catalogs() -> set[str]:
    """Return unique source-document filenames referenced by active catalogs."""
    catalog_dir = API_ROOT / "app" / "nbo" / "catalogs"
    sources: set[str] = set()

    def walk(value) -> None:
        """Collect source-document lists from one JSON-compatible value."""
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_documents", "evidence_references"}:
                    sources.update(str(source) for source in item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in sorted(catalog_dir.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return sources


def _placeholder_path(source_document_dir: Path, source_document: str) -> Path:
    """Map a catalog source-document filename to the public demo-data path."""
    return source_document_dir / source_document


def _pdf_escape(text: str) -> str:
    """Escape text for a simple PDF text object."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_minimal_pdf(path: Path, title: str) -> None:
    """Write a small valid PDF placeholder with generic public-demo text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        title,
        "Customer Offer Advisor public demo placeholder.",
        "This file contains no proprietary source material.",
    ]
    text_commands = [
        "BT",
        "/F1 14 Tf",
        "72 740 Td",
    ]
    for index, line in enumerate(lines):
        if index:
            text_commands.append("0 -22 Td")
        text_commands.append(f"({_pdf_escape(line)}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


def _write_minimal_docx(path: Path, title: str) -> None:
    """Write a small valid DOCX placeholder with generic public-demo text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = [
        title,
        "Customer Offer Advisor public demo placeholder.",
        "This file contains no proprietary source material.",
    ]
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(
            f"<w:p><w:r><w:t>{xml_escape(paragraph)}</w:t></w:r></w:p>"
            for paragraph in paragraphs
        )
        + "</w:body></w:document>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr("word/document.xml", document_xml)


def _write_seed_workbooks(data_seed_dir: Path) -> tuple[Path, Path]:
    """Write anonymized seed workbooks that match the HANA loader contract."""
    datasets = synthetic_runtime_datasets()
    data_seed_dir.mkdir(parents=True, exist_ok=True)
    customer_path = data_seed_dir / "customer_seed.xlsx"
    program_path = data_seed_dir / "program_seed.xlsx"

    with pd.ExcelWriter(customer_path) as writer:
        datasets["residential"].to_excel(writer, sheet_name="Residential", index=False)
        datasets["res_segment"].to_excel(
            writer,
            sheet_name="Residential_Residential_Segment",
            index=False,
        )
        datasets["commercial"].to_excel(writer, sheet_name="Commercial", index=False)
        datasets["comm_segment"].to_excel(
            writer,
            sheet_name="Commercial_Residential_Segment",
            index=False,
        )
        datasets["active_offering"].to_excel(
            writer,
            sheet_name="Active business offering",
            index=False,
        )

    with pd.ExcelWriter(program_path) as writer:
        datasets["program_contract"].to_excel(
            writer,
            sheet_name="Program Contract",
            index=False,
            startrow=1,
        )
        datasets["program_samples"].to_excel(
            writer,
            sheet_name="Sample Accounts",
            index=False,
        )
    return customer_path, program_path


def _write_source_placeholders(source_document_dir: Path) -> list[Path]:
    """Write public source-document placeholders for active catalog references."""
    written: list[Path] = []
    for source_document in sorted(_source_documents_from_catalogs()):
        path = _placeholder_path(source_document_dir, source_document)
        title = Path(source_document).stem.replace("_", " ").replace("-", " ").title()
        if path.suffix == ".docx":
            _write_minimal_docx(path, title)
        else:
            _write_minimal_pdf(path, title)
        written.append(path)
    return written


def main() -> int:
    """Generate public anonymized seed workbooks and source placeholders."""
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_seed_dir = output_dir / "data_seed"
    source_document_dir = output_dir / "source_documents"

    seed_workbooks = _write_seed_workbooks(data_seed_dir)
    source_files = _write_source_placeholders(source_document_dir)

    print("Generated public anonymized seed workbooks:")
    for path in seed_workbooks:
        print(f"  {path}")
    print("Generated public anonymized source placeholders:")
    for path in source_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
