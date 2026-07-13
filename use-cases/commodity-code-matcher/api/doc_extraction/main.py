from __future__ import annotations

"""
Commodity Code Extraction and Matching

Extract commodity line items from PDF quotations and match them to commodity codes
using LLM-based extraction and embedding-based similarity search with optional
LLM verification.

USAGE:

  1. Process a single PDF:
     python -m doc_extraction.main /path/to/pdf.pdf --llm-verify

  2. Batch process all PDFs in a directory:
     python -m doc_extraction.main /path/to/pdf/folder

  3. Custom output location:
     python -m doc_extraction.main /path/to/pdf.pdf --output /path/to/output.xlsx

KEY OPTIONS:

  --llm-verify              Run LLM verification on embedding results (slower, more accurate) (default: False)
  --llm-model               LLM model for verification (default: None, uses LLM_MODEL_NAME env var or 'gpt-4.1')
  --llm-min-confidence      Minimum LLM confidence threshold (default: 0.6)
  --top-k                   Number of top commodity codes to return (default: 5)
  --merge-headers           Include document headers in line-item matching (default: False)
  --output                  Custom output Excel file path (default: None, auto-generated in outputs/)
  --community-codes         Path to commodity codes catalog Excel file (legacy local mode only)
  --unspsc-context          Path to UNSPSC commodity mapping file (legacy local mode only)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Sequence
from dotenv import load_dotenv

# Load from the single .env file at api/.env
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)

import pandas as pd

from doc_extraction.embedding.matcher import run_community_code_matching


def _collect_pdfs(source: Path) -> List[Path]:
    source = source.expanduser()
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {source}")
        return [source.resolve()]
    if source.is_dir():
        pdfs = sorted(p.resolve() for p in source.rglob("*.pdf") if p.is_file())
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under {source}")
        return pdfs
    raise FileNotFoundError(f"Input path not found: {source}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract line items from PDFs and run commodity-code embeddings.",
    )
    parser.add_argument(
        "source",
        help="Path to a PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--community-codes",
        default="doc_extraction/embedding/Copy of Commodity codes list Jan 2021.xlsx",
        help="Path to the commodity/commodity code catalog Excel file.",
    )
    parser.add_argument(
        "--unspsc-context",
        default="doc_extraction/embedding/UNSPSC_COMM CODE_BUYER MAPPING - tree structure - updated.xlsx",
        help="Path to the UNSPSC → reference-code mapping workbook used for embedding context.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination Excel path for the final merged table (defaults to outputs/<pdf_or_folder>.xlsx).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top codes to store for each line item.")
    parser.add_argument(
        "--product-sheet-name",
        default="Product Categories - Structure",
        help="Exact sheet name inside the community code Excel file, if known.",
    )
    parser.add_argument(
        "--product-sheet-hint",
        default="Product category Structure",
        help="Fallback substring used to match the sheet if --product-sheet-name is missing.",
    )
    parser.add_argument(
        "--code-column",
        default=None,
        help="Explicit column name in the catalog that stores the commodity code.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model identifier (defaults to EMBEDDING_MODEL env var).",
    )
    parser.add_argument(
        "--merge-headers",
        action="store_true",
        help="Merge extracted headers into the line-item table before embeddings.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Attempt to display a preview via caas_jupyter_tools (if available).",
    )
    parser.add_argument(
        "--llm-verify",
        action="store_true",
        help="Run the final LLM verification/selection step after the embedding-based TOP-K search.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM identifier for verification (defaults to LLM_MODEL_NAME env var or 'gpt-4.1').",
    )
    parser.add_argument(
        "--llm-min-confidence",
        type=float,
        default=0.6,
        help="Minimum LLM confidence required before a row is considered reliable.",
    )
    return parser.parse_args(argv)


def _retry_with_multipage_images(pdf_path: Path):
    """
    Retry extraction using multi-page image processing when text extraction fails.

    This fallback is triggered when text+first-page extraction returns no line items.
    Useful for PDFs where:
    - Line items are on pages other than page 1
    - Content is in non-tabular format (lists, cost breakdowns)
    - Visual context is needed to identify line item boundaries

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ExtractionResult with lineItems from image-based extraction
    """
    from doc_extraction.llm_extraction.extractor import extract as llm_extract

    result = llm_extract(pdf_path)
    return result


def _extract_with_llm(
    pdf_paths: Sequence[Path],
    retry_multipage: bool = True,
    add_placeholders: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the LLM extraction flow for each PDF and consolidate the outputs into DataFrames.

    Implements three-tier extraction strategy:
    1. Standard extraction (text + first page image)
    2. Multi-page image fallback when Tier 1 returns empty lineItems
    3. Placeholder columns for true header-only documents

    Args:
        pdf_paths: Sequence of PDF paths to process
        retry_multipage: Enable Tier 2 fallback with multi-page image processing
        add_placeholders: Enable Tier 3 placeholder columns for consistency

    Returns two DataFrames so downstream embedding logic can stay unchanged.
    """

    from doc_extraction.llm_extraction.extractor import extract as llm_extract
    import warnings

    header_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []

    for pdf_path in pdf_paths:
        print(f"→ Processing (LLM): {pdf_path}")

        # Tier 1: Standard extraction (text + first page image)
        result = llm_extract(pdf_path)
        header = result.get("header") or {}
        doc_type = result.get("docType")
        line_items = result.get("lineItems") or []

        # Tier 2: If no line items found and retry enabled, try multi-page images
        if not line_items and retry_multipage:
            warnings.warn(
                f"⚠️  No line items extracted from '{pdf_path.name}' using {doc_type} extraction. "
                f"Attempting retry with enhanced multi-page image processing..."
            )

            retry_result = _retry_with_multipage_images(pdf_path)
            retry_items = retry_result.get("lineItems") or []

            if retry_items:
                # Success! Use retry results
                print(f"   ✓ Retry successful: Found {len(retry_items)} line items using multi-page image extraction")
                line_items = retry_items
                # Update header if retry provided better data
                if retry_result.get("header"):
                    header = retry_result.get("header")
                # Update doc_type if changed
                if retry_result.get("docType"):
                    doc_type = retry_result.get("docType")
            else:
                # Tier 3: Both attempts failed - assume header-only document
                warnings.warn(
                    f"   ✗ Retry failed: No line items found even with multi-page image extraction. "
                    f"'{pdf_path.name}' appears to be a header-only document or has unrecognizable structure. "
                    f"Proceeding with header data only."
                )

        # Collect a per-file header record so --merge-headers can join correctly.
        header_row: dict[str, object] = {"file": pdf_path.name, "doc_type": doc_type, **header}
        header_rows.append(header_row)

        prefixed_header = {f"header_{key}": value for key, value in header.items()}

        if not line_items:
            # Tier 3: Add row with placeholder columns for consistency
            if add_placeholders:
                row = {
                    "file": pdf_path.name,
                    "doc_type": doc_type,
                    "line_index": 1,
                    **prefixed_header,
                    # Add expected line item columns as None for consistent structure
                    "description": None,
                    "netAmount": None,
                    "quantity": None,
                    "unitPrice": None,
                    "materialNumber": None,
                    "itemNumber": None,
                    "usageSummary": None,
                }
            else:
                # Legacy behavior: minimal row
                row = {"file": pdf_path.name, "doc_type": doc_type, **prefixed_header}

            line_rows.append(row)
            continue

        # Normal processing for extracted line items
        for index, item in enumerate(line_items, start=1):
            row = {
                "file": pdf_path.name,
                "doc_type": doc_type,
                "line_index": index,
                **prefixed_header,
                **item,
            }
            line_rows.append(row)

    headers_df = pd.DataFrame(header_rows)
    line_items_df = pd.DataFrame(line_rows)

    # Add extraction summary diagnostics
    print(f"\n📊 Extraction Summary:")
    print(f"   • Total PDFs processed: {len(pdf_paths)}")

    # Count PDFs with successful line item extraction
    if "description" in line_items_df.columns:
        has_items = line_items_df["description"].notna()
        if "file" in line_items_df.columns:
            success_count = line_items_df.loc[has_items, "file"].nunique()
            failed_count = len(pdf_paths) - success_count

            print(f"   • PDFs with line items: {success_count}")
            if failed_count > 0:
                failed_files = line_items_df.loc[~has_items, "file"].unique().tolist()
                print(f"   • PDFs without line items: {failed_count}")
                print(f"     Files: {failed_files}")

    return headers_df, line_items_df


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    source_path = Path(args.source).expanduser()
    pdf_paths = _collect_pdfs(source_path)
    print(f"Found {len(pdf_paths)} PDF(s).")

    headers_df, line_items_df = _extract_with_llm(pdf_paths)

    if line_items_df.empty:
        raise RuntimeError("No line items were extracted from the provided PDFs.")

    # Prepare headers for merging with line-items during embedding step, if requested by --merge-headers
    headers_arg: pd.DataFrame | None = headers_df if args.merge_headers else None

    if args.output:
        output_path = Path(args.output)
    else:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        if len(pdf_paths) == 1:
            base_name = pdf_paths[0].stem
        else:
            folder_name = source_path.name or source_path.resolve().name or "documents"
            base_name = folder_name or "documents"
        output_path = outputs_dir / f"{base_name}.xlsx"

    output_path, _ = run_community_code_matching(
        line_items=line_items_df,
        headers=headers_arg,
        community_codes_path=args.community_codes,
        unspsc_context_path=args.unspsc_context,
        output_path=output_path,
        product_structure_sheet_name=args.product_sheet_name,
        product_structure_sheet_hint=args.product_sheet_hint,
        code_column_hint=args.code_column,
        embedding_model=args.embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        llm_verify=args.llm_verify,
        llm_model=args.llm_model,
        llm_min_confidence=args.llm_min_confidence,
        top_k_codes=args.top_k,
        show_preview=args.show_preview,
    )

    print(f"\nComplete. Final table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
