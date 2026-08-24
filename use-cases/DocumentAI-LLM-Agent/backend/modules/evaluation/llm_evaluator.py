"""
llm_evaluator.py
----------------
Evaluacion inteligente usando LLM reasoning.

El LLM analiza los tres resultados de extraccion y genera:
- evaluacion de calidad semantica
- deteccion de inconsistencias
- razonamiento sobre campos faltantes
- recomendacion del mejor metodo
"""

import json
import logging
from typing import Any

from modules.genai.llm_client import LLMClientError, ask_llm

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are an expert document extraction quality evaluator.

You have three extraction results from the same invoice document:

=== SAP Document AI ===
{sap_summary}

=== LLM Multimodal Prompting ===
{llm_p1_summary}

=== LLM Multimodal Structured ===
{llm_p2_summary}

=== CONFLICTS DETECTED ===
{conflicts}

=== FIELDS MISSING IN SAP ===
{missing_in_sap}

Analyze ALL three extraction results and return ONLY a valid JSON object:

{{
  "quality_assessment": {{
    "sap": {{
      "quality": "excellent|good|fair|poor",
      "reasoning": "<brief explanation>",
      "strengths": ["<strength1>", "<strength2>"],
      "weaknesses": ["<weakness1>"]
    }},
    "llm_prompting": {{
      "quality": "excellent|good|fair|poor",
      "reasoning": "<brief explanation>",
      "strengths": ["<strength1>"],
      "weaknesses": ["<weakness1>"]
    }},
    "llm_structured": {{
      "quality": "excellent|good|fair|poor",
      "reasoning": "<brief explanation>",
      "strengths": ["<strength1>"],
      "weaknesses": ["<weakness1>"]
    }}
  }},
  "conflict_analysis": [
    {{
      "field": "<field_name>",
      "assessment": "<which value is likely correct and why>",
      "recommended_value": "<value>",
      "confidence": 0.0
    }}
  ],
  "missing_field_analysis": [
    {{
      "field": "<field_name>",
      "likely_present_in_document": true,
      "reason_sap_missed": "<explanation>",
      "recommended_value": "<value from LLM or null>"
    }}
  ],
  "recommendation": {{
    "best_method": "SAP Document AI|LLM Prompting|LLM Structured",
    "reason": "<detailed explanation>",
    "use_case_notes": "<when to use each method>"
  }},
  "executive_summary": "<2-3 sentence professional summary of extraction quality>"
}}

Be precise, professional, and base your analysis on the actual data provided.
Return ONLY the JSON, no markdown, no explanations."""


def _build_summary(data: dict, method_name: str) -> str:
    """Construye un resumen compacto de los campos extraidos."""
    from modules.evaluation.field_analyzer import INVOICE_FIELDS, _get_conf

    lines = [f"Method: {method_name}"]
    found = []
    for f in INVOICE_FIELDS:
        val = data.get(f)
        if val is not None:
            conf = _get_conf(data, f)
            found.append(f"  {f}: {val} (conf: {conf:.0%})")

    lines.append(f"Fields found: {len(found)}/{len(INVOICE_FIELDS)}")
    lines.extend(found[:15])  # Limitar para no exceder contexto
    if len(found) > 15:
        lines.append(f"  ... and {len(found) - 15} more fields")

    line_items = data.get("lineItems") or []
    lines.append(f"Line items: {len(line_items)}")
    for i, item in enumerate(line_items[:3], 1):
        lines.append(f"  [{i}] {item.get('description', 'N/A')} | qty={item.get('quantity')} | unit={item.get('unitPrice')}")

    return "\n".join(lines)


def evaluate_with_llm(
    analysis: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """
    Usa el LLM para evaluar inteligentemente los resultados de extraccion.

    Args:
        analysis: Resultado de field_analyzer.analyze_fields()
        scores: Resultado de score_calculator.calculate_scores()

    Returns:
        Evaluacion completa del LLM.
    """
    normalized = analysis.get("normalized") or {}
    sap_data  = normalized.get("sap") or {}
    p1_data   = normalized.get("llm_prompting") or {}
    p2_data   = normalized.get("llm_structured") or {}

    conflicts     = analysis.get("conflicts") or []
    missing_in_sap = analysis.get("missing_in_sap") or []

    # Construir resumen compacto para el prompt
    sap_summary = _build_summary(sap_data, "SAP Document AI")
    p1_summary  = _build_summary(p1_data, "LLM Multimodal Prompting")
    p2_summary  = _build_summary(p2_data, "LLM Multimodal Structured")

    conflicts_str = json.dumps(conflicts[:10], indent=2) if conflicts else "None detected"
    missing_str   = json.dumps(missing_in_sap[:10], indent=2) if missing_in_sap else "None"

    prompt = EVALUATION_PROMPT.format(
        sap_summary=sap_summary,
        llm_p1_summary=p1_summary,
        llm_p2_summary=p2_summary,
        conflicts=conflicts_str,
        missing_in_sap=missing_str,
    )

    logger.info("Invoking LLM for intelligent evaluation...")

    try:
        raw = ask_llm(prompt)
        result = _parse_json(raw)
        logger.info("LLM evaluation completed.")
        return result
    except (LLMClientError, ValueError) as exc:
        logger.warning("LLM evaluation failed: %s. Using basic evaluation.", exc)
        return _fallback_evaluation(scores, analysis)


def _parse_json(raw: str) -> dict[str, Any]:
    """Parsea JSON de la respuesta del LLM."""
    import re
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    for pattern in [r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```", r"(\{[\s\S]+\})"]:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in LLM response: {raw[:300]}")


def _fallback_evaluation(scores: dict, analysis: dict) -> dict[str, Any]:
    """Evaluacion basica cuando el LLM no esta disponible."""
    best = scores.get("best_method") or {}
    return {
        "quality_assessment": {
            k: {"quality": "good", "reasoning": "Based on field count and confidence scores",
                "strengths": [], "weaknesses": []}
            for k in ["sap", "llm_prompting", "llm_structured"]
        },
        "conflict_analysis": [],
        "missing_field_analysis": [],
        "recommendation": {
            "best_method": best.get("name", "Unknown"),
            "reason": best.get("reason", "Highest overall score"),
            "use_case_notes": "Use SAP for structured invoices, LLM for complex layouts",
        },
        "executive_summary": (
            f"Extraction evaluation completed. Best method: {best.get('name', 'N/A')} "
            f"with score {best.get('overall_score', 0)}."
        ),
    }