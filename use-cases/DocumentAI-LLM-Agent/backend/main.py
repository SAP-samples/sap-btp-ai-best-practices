"""
main.py
-------
Entry point for the SAP Document AI + Gen AI Hub project.

Usage:
    python main.py                  # Interactive menu
    python main.py --schemas        # Fetch schemas only
    python main.py --templates      # Fetch templates only
    python main.py --invoice        # Process invoice (SAP DocAI)
    python main.py --genai          # Full GenAI pipeline
    python main.py --evaluate       # Evaluate extraction quality
"""

import argparse
import json
import logging
import sys

from modules.auth.get_token import AuthenticationError
from modules.invoice.process_invoice import (
    InvoiceProcessingError,
    JobFailedError,
    PollingTimeoutError,
    process_invoice,
)
from modules.schemas.get_schema import DocumentAIError as SchemaError
from modules.schemas.get_schema import get_schemas
from modules.templates.get_templates import DocumentAIError as TemplateError
from modules.templates.get_templates import get_templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_schemas(data: dict) -> None:
    schemas_list = (
        data.get("schemas") or data.get("value")
        or (data if isinstance(data, list) else None)
    )
    if not schemas_list:
        print("\n  No schemas found.\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    total = len(schemas_list)
    print(f"\n{'='*60}")
    print(f"  SAP Document AI - Schemas ({total} found)")
    print(f"{'='*60}\n")
    for idx, s in enumerate(schemas_list, 1):
        print(f"  [{idx:02d}] {s.get('name') or 'N/A'}  |  {s.get('id') or 'N/A'}")
        print(f"       Type: {s.get('documentType') or 'N/A'}  |  Status: {s.get('state') or s.get('status') or 'N/A'}")
        print()
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def display_templates(data: dict) -> None:
    tlist = (
        data.get("results") or data.get("templates") or data.get("value")
        or (data if isinstance(data, list) else None)
    )
    if not tlist:
        print("\n  No templates found.\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    total = len(tlist)
    print(f"\n{'='*60}")
    print(f"  SAP Document AI - Templates ({total} found)")
    print(f"{'='*60}\n")
    for idx, t in enumerate(tlist, 1):
        print(f"  [{idx:02d}] {t.get('name') or 'N/A'}  |  {t.get('id') or 'N/A'}")
        print(f"       Type: {t.get('documentType') or 'N/A'}  |  Status: {t.get('state') or t.get('status') or 'N/A'}")
        desc = t.get("description") or ""
        if desc:
            print(f"       Desc: {desc}")
        print()
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Line items helper
# ---------------------------------------------------------------------------

def format_line_items(line_items: list) -> str:
    """
    Format lineItems for console display.

    lineItems is an ARRAY OF ARRAYS:
        [
            [ {name, value, confidence, ...}, ... ],  ← Line Item #1
            [ {name, value, confidence, ...}, ... ],  ← Line Item #2
        ]

    Each inner array is one invoice row containing field objects.
    """
    if not line_items or not isinstance(line_items, list):
        return ""

    lines = [f"\n{'='*60}", "  LINE ITEMS", f"{'='*60}"]

    for idx, item_group in enumerate(line_items, start=1):
        lines.append(f"\n  [{idx}]")

        # item_group can be a list of field dicts (array-of-arrays format)
        # or a single dict (flat format) — handle both
        if isinstance(item_group, list):
            fields = item_group
        elif isinstance(item_group, dict):
            # Flat format: treat the dict itself as a single field set
            fields = [item_group]
        else:
            lines.append(f"    {item_group}")
            continue

        for field in fields:
            if not isinstance(field, dict):
                continue
            name  = field.get("name") or "N/A"
            value = field.get("value") if field.get("value") is not None else field.get("rawValue", "N/A")
            conf  = field.get("confidence")
            cs    = f"  ({conf:.0%})" if conf is not None else ""
            lines.append(f"    {name}: {value}{cs}")

    lines.append("")
    return "\n".join(lines)


def display_invoice_result(result: dict, output_path) -> None:
    print(f"\n{'='*60}")
    print("  SAP Document AI - Result")
    print(f"{'='*60}\n")
    print(f"  Job ID : {result.get('id') or 'N/A'}")
    print(f"  Status : {result.get('status') or 'N/A'}")
    print(f"  Saved  : {output_path}\n")

    extraction = result.get("extraction") or result.get("document") or {}
    if extraction:
        header_fields = extraction.get("headerFields") or []
        line_items    = extraction.get("lineItems") or []

        if header_fields:
            print(f"  Extracted fields ({len(header_fields)}):\n")
            for f in header_fields:
                name  = f.get("name") or "N/A"
                value = f.get("value") if f.get("value") is not None else f.get("rawValue", "N/A")
                conf  = f.get("confidence")
                cs    = f"  ({conf:.0%})" if conf is not None else ""
                print(f"    - {name}: {value}{cs}")
            print()

        if line_items:
            logger.info("Detected %d line items", len(line_items))
            print(format_line_items(line_items))

    print(f"{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def display_genai_result(pipeline_result: dict) -> None:
    """Display the result of the GenAI pipeline, handling both routing outcomes."""
    route = pipeline_result.get("route", "genai")

    # Always print the summary (works for both template and genai routes)
    summary = pipeline_result.get("summary", "")
    print(summary)

    if route == "template":
        # Template flow: show routing decision details + extracted fields
        routing = pipeline_result.get("routing_decision") or {}
        match = routing.get("template_match") or {}
        supplier = routing.get("supplier_detection") or {}

        print(f"\n{'='*60}")
        print("  ROUTING: SAP Template Processing")
        print(f"{'='*60}\n")
        print(f"  Supplier        : {supplier.get('supplier_name', 'N/A')}")
        print(f"  Matched Template: {match.get('template_name', 'N/A')}")
        print(f"  Template ID     : {match.get('template_id', 'N/A')}")
        print(f"  Confidence      : {match.get('confidence_pct', 0):.1f}%")
        print(f"\n  GenAI flow      : BYPASSED")
        print(f"  Output saved to : {pipeline_result.get('template_output_path')}")

        # Show extracted fields from template result
        template_result = pipeline_result.get("template_result") or {}
        extraction = template_result.get("extraction") or template_result.get("document") or {}
        header_fields = extraction.get("headerFields") or []
        line_items    = extraction.get("lineItems") or []

        if header_fields:
            print(f"\n{'='*60}")
            print(f"  EXTRACTED FIELDS ({len(header_fields)})")
            print(f"{'='*60}\n")
            for f in header_fields:
                name  = f.get("name") or "N/A"
                value = f.get("value") if f.get("value") is not None else f.get("rawValue", "N/A")
                conf  = f.get("confidence")
                cs    = f"  ({conf:.0%})" if conf is not None else ""
                print(f"  - {name}: {value}{cs}")

        if line_items:
            logger.info("Detected %d line items in template result", len(line_items))
            print(format_line_items(line_items))

        return

    # GenAI fallback flow: show LLM extracted fields
    llm_p1 = pipeline_result.get("llm_prompting") or {}
    llm_p2 = pipeline_result.get("llm_structured") or {}

    from modules.genai.compare_results import INVOICE_FIELDS, _get_confidence

    print(f"\n{'='*60}")
    print("  LLM Technique 1 — Extracted Fields")
    print(f"{'='*60}\n")
    for field in INVOICE_FIELDS:
        val = llm_p1.get(field)
        if val is not None:
            conf = _get_confidence(llm_p1, field)
            cs = f"  ({conf:.0%})" if conf is not None else ""
            print(f"  - {field}: {val}{cs}")

    items_p1 = llm_p1.get("lineItems") or []
    if items_p1:
        print(f"\n  Line Items ({len(items_p1)}):")
        for i, item in enumerate(items_p1, 1):
            print(f"    [{i}] desc={item.get('description')}  qty={item.get('quantity')}  unit={item.get('unitPrice')}  net={item.get('netAmount')}")

    print(f"\n{'='*60}")
    print("  LLM Technique 2 — Extracted Fields")
    print(f"{'='*60}\n")
    for field in INVOICE_FIELDS:
        val = llm_p2.get(field)
        if val is not None:
            conf = _get_confidence(llm_p2, field)
            cs = f"  ({conf:.0%})" if conf is not None else ""
            print(f"  - {field}: {val}{cs}")

    items_p2 = llm_p2.get("lineItems") or []
    if items_p2:
        print(f"\n  Line Items ({len(items_p2)}):")
        for i, item in enumerate(items_p2, 1):
            print(f"    [{i}] desc={item.get('description')}  qty={item.get('quantity')}  unit={item.get('unitPrice')}  net={item.get('netAmount')}")

    print(f"\n  Output saved to: {pipeline_result.get('output_dir')}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_schemas() -> None:
    logger.info("-- Module: Get Schemas --")
    try:
        display_schemas(get_schemas())
    except (FileNotFoundError, ValueError, AuthenticationError, SchemaError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def run_templates() -> None:
    logger.info("-- Module: Get Templates --")
    try:
        display_templates(get_templates())
    except (FileNotFoundError, ValueError, AuthenticationError, TemplateError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def run_invoice() -> None:
    logger.info("-- Module: Process Invoice --")
    try:
        result, output_path = process_invoice()
        display_invoice_result(result, output_path)
    except (FileNotFoundError, ValueError, AuthenticationError,
            JobFailedError, PollingTimeoutError, InvoiceProcessingError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


def run_genai() -> None:
    logger.info("-- Module: Process Invoice with GenAI Multimodal (Intelligent Routing) --")
    try:
        from modules.genai.process_with_genai import GenAIPipelineError, run_genai_pipeline
        result = run_genai_pipeline()
        display_genai_result(result)

        route = result.get("route", "genai")
        if route == "template":
            logger.info("Pipeline completed via SAP Template route.")
        else:
            logger.info("Pipeline completed via GenAI fallback route.")

    except ImportError as exc:
        logger.error("GenAI dependencies not installed: %s", exc)
        logger.error("Run: pip install -r requirements.txt")
        sys.exit(1)
    except (FileNotFoundError, ValueError, AuthenticationError,
            JobFailedError, PollingTimeoutError, InvoiceProcessingError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("GenAI pipeline error: %s", exc)
        sys.exit(1)


def run_evaluation() -> None:
    logger.info("-- Module: Evaluate Extraction Quality --")
    try:
        from modules.evaluation.evaluator import EvaluationError, run_evaluation as _run
        result = _run()
        print(result["summary"])
        print(f"\n  Files saved to: output/evaluation/")
    except ImportError as exc:
        logger.error("Dependencies not installed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

MENU_OPTIONS: dict = {
    "1": ("Get Schemas",                                        run_schemas),
    "2": ("Get Templates",                                      run_templates),
    "3": ("Process Invoice (SAP DocAI)",                        run_invoice),
    "4": ("Process Invoice with GenAI + Intelligent Routing",   run_genai),
    "5": ("Evaluate Extraction Quality",                        run_evaluation),
}


def show_menu() -> None:
    print(f"\n{'='*60}")
    print("  SAP Document AI + Gen AI Hub — Main Menu")
    print(f"{'='*60}\n")
    for key, (label, _) in MENU_OPTIONS.items():
        print(f"  [{key}] {label}")
    print("  [0] Exit\n")

    while True:
        try:
            choice = input("  Select an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting...")
            sys.exit(0)

        if choice == "0":
            print("\n  Goodbye.\n")
            sys.exit(0)

        if choice in MENU_OPTIONS:
            _, runner = MENU_OPTIONS[choice]
            logger.info("=" * 50)
            runner()
            logger.info("=" * 50)
            break

        print(f"  Invalid option. Enter 0-{len(MENU_OPTIONS)}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAP Document AI + Gen AI Hub — Invoice Extraction Pipeline"
    )
    parser.add_argument("--schemas",   action="store_true", help="Fetch schemas")
    parser.add_argument("--templates", action="store_true", help="Fetch templates")
    parser.add_argument("--invoice",   action="store_true", help="Process invoice (SAP DocAI)")
    parser.add_argument("--genai",     action="store_true", help="Full GenAI pipeline")
    parser.add_argument("--evaluate",  action="store_true", help="Evaluate extraction quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.schemas or args.templates or args.invoice or args.genai or args.evaluate:
        logger.info("SAP Document AI + Gen AI Hub")
        logger.info("=" * 50)
        if args.schemas:   run_schemas()
        if args.templates: run_templates()
        if args.invoice:   run_invoice()
        if args.genai:     run_genai()
        if args.evaluate:  run_evaluation()
        logger.info("=" * 50)
    else:
        show_menu()


if __name__ == "__main__":
    main()