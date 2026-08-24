"""
process_with_genai.py
---------------------
Orchestrator for the complete pipeline with intelligent routing:

  STEP 1: SAP Document AI (generic) — extract supplier name
  STEP 2: Routing Engine — detect supplier, match template
  STEP 3a: IF template found → SAP template-based processing (bypass GenAI)
  STEP 3b: IF no template   → LLM Multimodal T1 + T2 + Comparison (GenAI flow)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.genai.compare_results import compare
from modules.genai.multimodal_prompting import extract_multimodal_prompting
from modules.genai.multimodal_structured import extract_multimodal_structured
from modules.invoice.process_invoice import InvoiceProcessor
from modules.routing.routing_engine import route_invoice
from modules.routing.template_processor import TemplateInvoiceProcessor

logger = logging.getLogger(__name__)

OUTPUT_GENAI_DIR: Path = Path(__file__).parent.parent.parent / "output" / "genai"


class GenAIPipelineError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def _save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Saved: %s", path)


def _build_summary(
    pdf_name: str,
    sap_result: dict,
    llm_p1: dict,
    llm_p2: dict,
    comparison: dict,
    output_dir: Path,
) -> str:
    """Build the final comparison summary as plain text."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = comparison["summary"]
    lines = [
        "=" * 70,
        "  SAP DOCUMENT AI + GEN AI HUB — COMPARISON SUMMARY",
        "=" * 70,
        f"  File       : {pdf_name}",
        f"  Processed  : {ts}",
        f"  Output dir : {output_dir}",
        "",
        "─" * 70,
        "  SAP Document AI",
        "─" * 70,
        f"  Fields found     : {s['sap_fields_found']}",
        f"  Confidence avg   : {s['sap_confidence_avg']:.0%}",
        "",
        "─" * 70,
        "  LLM Multimodal — Technique 1 (Free Prompting)",
        "─" * 70,
        f"  Fields found     : {s['llm_prompting_fields_found']}",
        f"  Confidence avg   : {s['llm_prompting_confidence_avg']:.0%}",
        "",
        "─" * 70,
        "  LLM Multimodal — Technique 2 (Structured JSON)",
        "─" * 70,
        f"  Fields found     : {s['llm_structured_fields_found']}",
        f"  Confidence avg   : {s['llm_structured_confidence_avg']:.0%}",
        "",
        "─" * 70,
        "  COMPARISON",
        "─" * 70,
        f"  Total unique fields : {s['total_unique_fields']}",
        f"  SAP + LLM agreements: {s['agreements']}",
        f"  Conflicts           : {s['conflicts']}",
        f"  SAP only            : {s['only_in_sap']}",
        f"  LLM only            : {s['only_in_llm']}",
        "",
    ]

    if comparison.get("conflicts"):
        lines.append("  DIFFERENCES DETECTED:")
        for c in comparison["conflicts"]:
            lines.append(f"    Field: {c['field']}")
            lines.append(f"      SAP           : {c['sap']}")
            lines.append(f"      LLM Prompting : {c['llm_prompting']}")
            lines.append(f"      LLM Structured: {c['llm_structured']}")
        lines.append("")

    if comparison.get("only_in_llm"):
        lines.append("  FIELDS FOUND ONLY BY LLM (not in SAP):")
        for f in comparison["only_in_llm"]:
            v1 = llm_p1.get(f)
            v2 = llm_p2.get(f)
            lines.append(f"    {f}: {v1 or v2}")
        lines.append("")

    if comparison.get("only_in_sap"):
        lines.append("  FIELDS FOUND ONLY BY SAP (not in LLM):")
        for f in comparison["only_in_sap"]:
            lines.append(f"    {f}: {comparison['sap_normalized'].get(f)}")
        lines.append("")

    lines += [
        "=" * 70,
        "  GENERATED FILES",
        "=" * 70,
        f"  {output_dir}/sap_result.json",
        f"  {output_dir}/llm_multimodal_prompting.json",
        f"  {output_dir}/llm_multimodal_structured.json",
        f"  {output_dir}/comparison.json",
        f"  {output_dir}/final_summary.txt",
        "=" * 70,
    ]

    return "\n".join(lines)


def _format_line_items_text(line_items: list) -> list[str]:
    """
    Format lineItems (array of arrays) into text lines for summaries.

    SAP Document AI returns lineItems as an array of arrays:
        [ [{name, value, ...}, ...], [{name, value, ...}, ...] ]
    Each inner array is one invoice row.
    """
    if not line_items or not isinstance(line_items, list):
        return []

    lines = [
        "",
        "─" * 70,
        f"  LINE ITEMS ({len(line_items)} rows)",
        "─" * 70,
    ]

    for idx, item_group in enumerate(line_items, start=1):
        lines.append(f"\n  [{idx}]")

        if isinstance(item_group, list):
            fields = item_group
        elif isinstance(item_group, dict):
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

    return lines


def _build_template_summary(
    pdf_name: str,
    routing_decision: dict,
    result: dict,
    output_path: Path,
) -> str:
    """Build a summary string for the template-based processing flow."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match = routing_decision.get("template_match") or {}
    supplier = routing_decision.get("supplier_detection") or {}

    extraction = result.get("extraction") or result.get("document") or {}
    header_fields = extraction.get("headerFields") or []
    line_items    = extraction.get("lineItems") or []

    if line_items:
        logger.info("Detected %d line items in template result", len(line_items))

    lines = [
        "=" * 70,
        "  SAP DOCUMENT AI — TEMPLATE-BASED PROCESSING SUMMARY",
        "=" * 70,
        f"  File        : {pdf_name}",
        f"  Processed   : {ts}",
        f"  Output file : {output_path}",
        "",
        "─" * 70,
        "  ROUTING DECISION",
        "─" * 70,
        f"  Supplier        : {supplier.get('supplier_name', 'N/A')}",
        f"  Matched Template: {match.get('template_name', 'N/A')}",
        f"  Template ID     : {match.get('template_id', 'N/A')}",
        f"  Confidence      : {match.get('confidence_pct', 0):.1f}%",
        f"  Route           : SAP Template Processing (GenAI bypassed)",
        "",
        "─" * 70,
        f"  EXTRACTED FIELDS ({len(header_fields)})",
        "─" * 70,
    ]

    if header_fields:
        for field in header_fields:
            name  = field.get("name") or "N/A"
            value = field.get("value") if field.get("value") is not None else field.get("rawValue", "N/A")
            conf  = field.get("confidence")
            cs    = f"  ({conf:.0%})" if conf is not None else ""
            lines.append(f"  - {name}: {value}{cs}")
    else:
        lines.append("  (no header fields extracted)")

    # Render line items (array of arrays)
    if line_items:
        lines.extend(_format_line_items_text(line_items))

    lines += [
        "",
        "=" * 70,
        "  GenAI flow: BYPASSED (template match found)",
        "=" * 70,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_genai_pipeline(
    schema_name: str = "SAP_invoice_schema",
    client_id: str = "default",
    document_type: str = "invoice",
) -> dict[str, Any]:
    """
    Execute the complete pipeline with intelligent routing:

      STEP 1: SAP Document AI (generic) — initial extraction
      STEP 2: Routing Engine
                → Supplier detection
                → Template matching
      STEP 3a: Template found → SAP template-based reprocessing (GenAI bypassed)
      STEP 3b: No template   → LLM T1 + LLM T2 + Comparison (full GenAI flow)

    Returns:
        Dictionary with all results and routing metadata.
    """
    OUTPUT_GENAI_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: SAP Document AI (generic) ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/3: SAP Document AI — Initial extraction")
    logger.info("=" * 60)

    processor = InvoiceProcessor()
    pdf_path = processor.select_document()

    print(f"\n  Processing with SAP Document AI: {pdf_path.name}")
    job_id = processor.submit_document(pdf_path, schema_name, client_id, document_type)
    print(f"  Job ID: {job_id}")
    print("  Waiting for SAP result...\n")

    sap_result = processor.poll_until_done(job_id)
    processor.save_result(job_id, sap_result)
    _save_json(sap_result, OUTPUT_GENAI_DIR / "sap_result.json")

    logger.info("SAP Document AI initial extraction completed.")

    # ── STEP 2: Routing Engine ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/3: Routing Engine — Supplier detection & template matching")
    logger.info("=" * 60)

    routing_decision = route_invoice(sap_result, client_id=client_id)
    route = routing_decision.get("route", "genai")

    # ── STEP 3a: Template flow ───────────────────────────────────────────
    if route == "template":
        logger.info("=" * 60)
        logger.info("STEP 3/3: SAP Template Processing (GenAI bypassed)")
        logger.info("=" * 60)

        template_match = routing_decision["template_match"]
        template_id = template_match["template_id"]
        template_name = template_match.get("template_name", "N/A")
        supplier_name = routing_decision.get("supplier_detection", {}).get("supplier_name", "N/A")

        print(f"\n{'='*52}")
        print("  SUPPLIER DETECTED")
        print(f"{'='*52}")
        print(f"\n  Supplier: {supplier_name}")
        print(f"\n{'='*52}")
        print("  MATCHING TEMPLATE FOUND")
        print(f"{'='*52}")
        print(f"\n  Template   : {template_name}")
        print(f"  Template ID: {template_id}")
        print(f"\n  Reprocessing invoice using SAP specialized template...")

        logger.info("Reprocessing invoice using schema + template...")
        logger.info("  Supplier : %s", supplier_name)
        logger.info("  Template : %s", template_name)
        logger.info("  Schema   : %s", schema_name)

        template_processor = TemplateInvoiceProcessor()
        template_result, template_output_path = template_processor.process(
            pdf_path,
            template_id=template_id,
            schema_name=schema_name,
            client_id=client_id,
            document_type=document_type,
        )

        summary_text = _build_template_summary(
            pdf_path.name, routing_decision, template_result, template_output_path
        )
        _save_text(summary_text, OUTPUT_GENAI_DIR / "final_summary.txt")

        return {
            "route": "template",
            "pdf_path": pdf_path,
            "sap_result": sap_result,
            "routing_decision": routing_decision,
            "template_result": template_result,
            "template_output_path": template_output_path,
            "llm_prompting": None,
            "llm_structured": None,
            "comparison": None,
            "summary": summary_text,
            "output_dir": OUTPUT_GENAI_DIR,
        }

    # ── STEP 3b: GenAI fallback flow ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3/3: GenAI Multimodal Pipeline (fallback)")
    logger.info("=" * 60)

    # LLM Technique 1: Free Prompting
    logger.info("STEP 3a/3: LLM Multimodal — Technique 1 (Free Prompting)")
    print("\n  Sending PDF to LLM (Technique 1: Free Prompting)...")

    llm_p1 = extract_multimodal_prompting(pdf_path)
    _save_json(llm_p1, OUTPUT_GENAI_DIR / "llm_multimodal_prompting.json")

    # LLM Technique 2: Structured JSON
    logger.info("STEP 3b/3: LLM Multimodal — Technique 2 (Structured JSON)")
    print("\n  Sending PDF to LLM (Technique 2: Structured JSON)...")

    llm_p2 = extract_multimodal_structured(pdf_path)
    _save_json(llm_p2, OUTPUT_GENAI_DIR / "llm_multimodal_structured.json")

    # Comparison
    logger.info("STEP 3c/3: Results comparison")

    comparison = compare(sap_result, llm_p1, llm_p2)
    _save_json(comparison, OUTPUT_GENAI_DIR / "comparison.json")

    summary_text = _build_summary(
        pdf_path.name, sap_result, llm_p1, llm_p2, comparison, OUTPUT_GENAI_DIR
    )
    _save_text(summary_text, OUTPUT_GENAI_DIR / "final_summary.txt")

    return {
        "route": "genai",
        "pdf_path": pdf_path,
        "sap_result": sap_result,
        "routing_decision": routing_decision,
        "template_result": None,
        "template_output_path": None,
        "llm_prompting": llm_p1,
        "llm_structured": llm_p2,
        "comparison": comparison,
        "summary": summary_text,
        "output_dir": OUTPUT_GENAI_DIR,
    }