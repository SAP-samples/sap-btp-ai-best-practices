"""Handover pre-fill extraction from ideation conversations and text sources.

Extracts structured fields from an ideation conversation's final output
(or arbitrary text) and maps them to the handover form payload shape.
Used by the prefill endpoint and the add-context flow.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .agent import AGENT_LLM_MODEL
from .common import make_llm

logger = logging.getLogger(__name__)

# ── Field mapping schema ────────────────────────────────────────────────────

_EXTRACTION_SCHEMA = """{
  "usecase_title": "string or null — short title for the use case",
  "industry": "string or null — industry or domain",
  "problem_statement": "string or null — the business problem being solved",
  "process_name": "string or null — name of the process being automated/augmented",
  "process_freq": "string or null — how often the process runs (trigger/frequency)",
  "process_volume": "string or null — transaction volume (e.g., '200 invoices/week', '50-100 PDFs/day')",
  "process_team": "string or null — who is involved (user roles/teams)",
  "process_people": "string or null — number of people involved (e.g., '5 procurement clerks')",
  "data_input": "string or null — what data goes in (summary of data sources)",
  "data_output": "string or null — what the AI solution produces as output",
  "data_usage": "string or null — how the output is used downstream",
  "tobe_narrative": "string or null — description of the to-be AI-augmented process",
  "in_scope": "string or null — what is explicitly in scope for the AI solution",
  "out_scope": "string or null — what is explicitly out of scope",
  "done_definition": "string or null — definition of done or success criteria",
  "priority": "string or null — one of: high, medium, low (infer from evaluation score if present)",
  "data_sources": [{"name": "source name", "system": "system/platform", "data_type": "type of data", "owner": "team or person responsible", "accessibility": "API, Export, Manual, etc.", "location": "On-prem, Cloud, etc."}],
  "business_rules": [{"description": "rule description", "condition": "when it applies", "action": "what happens", "source": "where this rule comes from (SOP, policy doc, etc.)"}],
  "kpis": [{"metric": "KPI name", "current": "current value or baseline", "target": "target value"}],
  "thresholds": [{"field": "field or variable name", "range": "threshold value or range", "action_below": "what happens below threshold", "action_above": "what happens above threshold"}],
  "approvals": [{"scenario": "scenario type", "approver": "role or person who approves", "escalation": "escalation path", "sla": "time limit"}],
  "edge_cases": [{"case": "edge case name", "description": "when/how it occurs", "handling": "how to handle it"}],
  "integrations": [{"system_a": "first system", "system_b": "second system", "method": "integration method (API, RFC, File, etc.)", "frequency": "real-time, batch, etc."}],
  "business_impact": {
    "volume": {"current": "current transaction volume", "target": "target with AI"},
    "time_spent": {"current": "current manual time", "target": "target with AI"},
    "error_rate": {"current": "current error/exception rate", "target": "target with AI"},
    "cost_per_error": {"current": "current cost per error", "target": "target with AI"},
    "annual_cost": {"current": "current annual cost/effort", "target": "target with AI"},
    "downstream": {"current": "current downstream consequences", "target": "target with AI"}
  },
  "ai_capabilities": ["list of applicable capability IDs from: doc_understanding, classification, anomaly_detection, nlu, predictive, recommendation, conversational, agentic, rag, computer_vision, generative_text"]
}"""

_EXTRACTION_PROMPT = """You are a structured data extractor. Given the ideation conversation output below, extract fields into the JSON schema provided.

RULES:
- Only extract information explicitly stated or clearly implied in the source text.
- Use null for fields where no information is available. Do NOT invent or hallucinate values.
- For array fields (data_sources, business_rules, kpis, thresholds, approvals, edge_cases, integrations): return an empty array [] if no relevant info found.
- Keep values concise but complete. Preserve technical terms and proper nouns exactly.
- For tobe_narrative: combine "WHAT IT DOES" and "PROCESSING STEPS" into a cohesive description.
- For process_freq: extract from "TRIGGER" section if available.
- For data_input: summarize from "DATA SOURCES & SYSTEMS" section.
- For process_team: extract from "USER ROLES" section.
- For process_volume: look for transaction counts, document volumes, frequency numbers in Q&A (e.g., "200 per week", "50-100 PDFs/day").
- For process_people: look for team size or headcount in Q&A (e.g., "team of 5", "3 clerks").
- For data_output: describe what the AI solution produces — derive from PROCESSING STEPS or solution description.
- For data_usage: describe how outputs are used downstream — who consumes them, what decisions they inform.
- For in_scope / out_scope: extract only if explicitly discussed in the conversation. Do not infer.
- For done_definition: extract from success criteria, KPI targets, or acceptance conditions mentioned.
- For priority: if EVALUATION SCORE is present, map it: 750+ → "high", 450-749 → "medium", below 450 → "low". Otherwise null.
- For business_impact: extract any quantitative current/target metrics. Use null for sub-objects where no data exists (e.g., if only volume is mentioned, other dimensions should have null values).
- For ai_capabilities: map the solution description to capability IDs. Examples: PDF/document parsing → doc_understanding, categorization/routing → classification, outlier detection → anomaly_detection, text understanding → nlu, forecasting → predictive, suggestions/matching → recommendation, chatbot/Q&A → conversational, multi-step orchestration → agentic, knowledge base search → rag, image analysis → computer_vision, text generation/summarization → generative_text.
- For thresholds: extract confidence levels, amount limits, or routing criteria with above/below actions.
- For approvals: extract escalation scenarios with approver roles and SLAs if discussed.
- For edge_cases: extract exception scenarios and their handling if discussed.
- For integrations: derive system-to-system connections from DATA SOURCES & SYSTEMS section.
- When CONVERSATION Q&A is present, extract detailed information from the user's answers. Prefer specific user-stated values over generic summaries.
- Look for data_sources, business_rules, kpis, thresholds, and edge_cases in the conversation Q&A — users often mention these in answers but they may not appear in the final summary.
- When RESEARCH CONTEXT is present, use it to enrich and validate extractions but do not invent details not supported by the conversation.
- When EVALUATION FEEDBACK is present, use the gap questions as hints for fields that may have been discussed but not yet captured.

OUTPUT SCHEMA:
{schema}

SOURCE TEXT:
{source_text}

Respond with ONLY the JSON object matching the schema above. No markdown, no explanation."""

_TEXT_EXTRACTION_PROMPT = """You are a structured data extractor. Given the text below (which may be a meeting transcript, document, or notes), extract any relevant information into the JSON schema provided.

RULES:
- Only extract information explicitly stated or clearly implied in the source text.
- Use null for fields where no information is available. Do NOT invent or hallucinate values.
- For array fields (data_sources, business_rules, kpis): return an empty array [] if no relevant info found.
- Keep values concise but complete.
- This text may only contain partial information — extract whatever is available.

OUTPUT SCHEMA:
{schema}

SOURCE TEXT:
{source_text}

Respond with ONLY the JSON object matching the schema above. No markdown, no explanation."""


# ── Extraction functions ────────────────────────────────────────────────────


async def extract_from_ideation(
    conversation_id: str,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load ideation final output, extract into handover form fields.

    Returns a partial payload dict with extracted values.
    Returns empty dict on failure (never raises).
    """
    try:
        from .hana_store import get_store

        store = await get_store()
        messages = await store.get_messages(conversation_id)

        # Find the final delivery message (last message with metadata.ui.kind == "final")
        final_msg = None
        for msg in reversed(messages):
            metadata = msg.get("metadata") or {}
            ui = metadata.get("ui") or {}
            if ui.get("kind") == "final":
                final_msg = msg
                break

        if not final_msg:
            logger.warning(
                "No final message found for conversation %s", conversation_id
            )
            return {}

        # Build source text from structured fields
        ui = final_msg["metadata"]["ui"]
        source_parts = []

        if ui.get("title"):
            source_parts.append(f"TITLE: {ui['title']}")
        if ui.get("industry"):
            source_parts.append(f"INDUSTRY: {ui['industry']}")
        if ui.get("problem"):
            source_parts.append(f"PROBLEM: {ui['problem']}")
        if ui.get("solution"):
            source_parts.append(f"SOLUTION:\n{ui['solution']}")

        # Also include the rendered display text for additional context
        content = final_msg.get("content", "")
        if content and len(content) > len("\n".join(source_parts)):
            source_parts.append(f"\nFULL RENDERED OUTPUT:\n{content}")

        # Include conversation Q&A for richer extraction context
        qa_text = _build_conversation_qa(messages)
        if qa_text:
            source_parts.append(f"\nCONVERSATION Q&A:\n{qa_text}")

        # Include research context (Perplexity + LinkedIn) if available
        try:
            all_research = await store.get_all_research(conversation_id)
            research_parts: List[str] = []

            px_company = all_research.get("perplexity_company", {})
            company_md = (px_company.get("payload") or {}).get("company_markdown", "")
            if company_md:
                research_parts.append(f"Company Research:\n{company_md[:4000]}")

            px_problem = all_research.get("perplexity_problem", {})
            problem_md = (px_problem.get("payload") or {}).get("problem_markdown", "")
            if problem_md:
                research_parts.append(f"Problem Domain Research:\n{problem_md[:3000]}")

            li_data = all_research.get("linkedin", {})
            li_payload = li_data.get("payload") or {}
            if li_payload.get("status") == "ok":
                li_lines: List[str] = []
                for tech in (li_payload.get("top_technologies") or [])[:5]:
                    li_lines.append(f"- Tech: {tech.get('name', '')} ({tech.get('insight', '')})")
                insights = li_payload.get("ideation_insights") or {}
                for b in (insights.get("bottlenecks") or []):
                    li_lines.append(f"- Bottleneck: {b}")
                for o in (insights.get("automation_opportunities") or []):
                    li_lines.append(f"- Opportunity: {o}")
                if li_lines:
                    research_parts.append("LinkedIn Analysis:\n" + "\n".join(li_lines))

            if research_parts:
                source_parts.append("\nRESEARCH CONTEXT:\n" + "\n\n".join(research_parts))
        except Exception as exc:
            logger.debug("Research context not available for %s: %s", conversation_id, exc)

        # Include evaluation feedback if available
        try:
            from app.services.hana_submissions import get_submission_store

            sub_store = await get_submission_store()
            subs = await sub_store.get_all_by_context_any_user(conversation_id)
            if subs:
                best = max(subs, key=lambda s: s.get("score", 0))
                eval_parts: List[str] = [f"Evaluation Score: {best.get('score', 0)}/1000"]
                analysis = best.get("evaluator_detailed_analysis") or ""
                if analysis:
                    eval_parts.append(f"Analysis: {analysis[:2000]}")
                suggestions = best.get("evaluator_suggestions") or []
                if suggestions:
                    eval_parts.append("Gap Questions: " + " | ".join(str(s) for s in suggestions))
                source_parts.append("\nEVALUATION FEEDBACK:\n" + "\n".join(eval_parts))
        except Exception as exc:
            logger.debug("Evaluation context not available for %s: %s", conversation_id, exc)

        source_text = "\n\n".join(source_parts)

        if not source_text.strip():
            logger.warning(
                "Empty source text from final message for conversation %s",
                conversation_id,
            )
            return {}

        extracted = await _run_extraction(source_text, _EXTRACTION_PROMPT, llm_usage_meta)

        # Use the conversation's stored title (e.g. "Volvo Car AB: Automated material grouping")
        # instead of the LLM-extracted usecase_title which may differ
        conv = await store.get_conversation_by_id(conversation_id)
        conv_title = conv.get("title") if conv else None
        if conv_title:
            extracted["usecase_title"] = conv_title

        return extracted

    except Exception as exc:
        logger.error(
            "Failed to extract from ideation conversation %s: %s",
            conversation_id,
            exc,
            exc_info=True,
        )
        return {}


async def extract_from_text(
    text: str,
    source_type: str = "manual",
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract handover fields from arbitrary text (transcript, notes, etc.).

    Returns a partial payload dict. Returns empty dict on failure (never raises).
    """
    if not text or not text.strip():
        return {}

    try:
        return await _run_extraction(text, _TEXT_EXTRACTION_PROMPT, llm_usage_meta)
    except Exception as exc:
        logger.error(
            "Failed to extract from %s text: %s", source_type, exc, exc_info=True
        )
        return {}


# ── Internal helpers ────────────────────────────────────────────────────────


async def _run_extraction(
    source_text: str,
    prompt_template: str,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the LLM extraction with observability logging."""
    from app.observability.llm_usage_logging import emit_llm_usage_for_response

    # Truncate very long source texts to avoid token limits
    max_chars = 40_000  # ~10k tokens for context (expanded to accommodate research + eval)
    if len(source_text) > max_chars:
        source_text = source_text[:max_chars] + "\n\n[... truncated]"

    prompt = prompt_template.format(
        schema=_EXTRACTION_SCHEMA,
        source_text=source_text,
    )

    llm = make_llm(
        provider="openai",
        model_name=AGENT_LLM_MODEL,
        temperature=0.1,
        max_tokens=4096,
        json_mode=True,
    )

    start = time.perf_counter()
    response = await llm.ainvoke(prompt)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Log usage
    emit_llm_usage_for_response(
        response,
        meta=llm_usage_meta,
        provider="sap-ai-core",
        model=AGENT_LLM_MODEL,
        step_name="handover_prefill",
        feature="handover",
        latency_ms=elapsed_ms,
    )

    # Parse result
    raw_text = response.content if hasattr(response, "content") else str(response)
    extracted = _parse_json_response(raw_text)

    if not extracted:
        logger.warning("LLM returned unparseable response for prefill extraction")
        return {}

    # Clean up: remove null values and empty arrays
    return _clean_extracted(extracted)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code fences."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


_VALID_AI_CAPABILITIES = frozenset({
    "doc_understanding", "classification", "anomaly_detection", "nlu",
    "predictive", "recommendation", "conversational", "agentic",
    "rag", "computer_vision", "generative_text",
})


def _clean_extracted(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove null values, empty arrays, and empty nested objects from extracted data."""
    cleaned = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        # Special handling: business_impact — strip empty sub-dimensions
        if key == "business_impact" and isinstance(value, dict):
            cleaned_impact = {}
            for dim, pair in value.items():
                if not isinstance(pair, dict):
                    continue
                cur = (pair.get("current") or "").strip() if isinstance(pair.get("current"), str) else pair.get("current")
                tgt = (pair.get("target") or "").strip() if isinstance(pair.get("target"), str) else pair.get("target")
                if cur or tgt:
                    cleaned_impact[dim] = {"current": cur or None, "target": tgt or None}
            if cleaned_impact:
                cleaned[key] = cleaned_impact
            continue

        # Special handling: ai_capabilities — validate against known IDs
        if key == "ai_capabilities" and isinstance(value, list):
            valid = [cap for cap in value if isinstance(cap, str) and cap in _VALID_AI_CAPABILITIES]
            if valid:
                cleaned[key] = valid
            continue

        # Empty dicts (e.g., tables returning {} instead of [])
        if isinstance(value, dict) and len(value) == 0:
            continue

        cleaned[key] = value
    return cleaned


def _build_conversation_qa(messages: List[Dict[str, Any]]) -> str:
    """Build a Q&A transcript from conversation messages.

    Extracts agent clarification questions and user answers,
    skipping thinking messages, intro messages, and the final delivery.
    Returns an empty string if no Q&A content is found.
    """
    qa_lines: List[str] = []
    skip_kinds = {"final", "thinking", "error", "confirmed", "refinement"}

    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        metadata = msg.get("metadata") or {}
        ui = metadata.get("ui") or {}
        kind = ui.get("kind", "")

        if role in ("agent", "assistant"):
            if kind in skip_kinds:
                continue
            if content.startswith("[thinking:"):
                continue
            # Clarification questions — extract the focused question text
            if kind == "clarification":
                question = ui.get("question") or content
                qa_lines.append(f"Q: {question}")
            elif len(content) > 100:
                # Substantive agent messages (e.g. intro context)
                qa_lines.append(f"CONTEXT: {content}")
        elif role == "user":
            qa_lines.append(f"A: {content}")

    return "\n".join(qa_lines)


def merge_extracted_fields(*field_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple extracted field dicts. Later values override earlier ones.

    For array fields, later non-empty arrays replace earlier ones (no dedup).
    For scalar fields, later non-empty values replace earlier ones.
    For dict fields (business_impact), deep-merge sub-keys.
    For ai_capabilities (list of strings), union-merge deduplicated.
    """
    merged: Dict[str, Any] = {}
    for fields in field_dicts:
        if not fields:
            continue
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, dict) and len(value) == 0:
                continue

            # Deep-merge dict fields (e.g., business_impact)
            if key == "business_impact" and isinstance(value, dict) and isinstance(merged.get(key), dict):
                existing = merged[key]
                for dim, pair in value.items():
                    if not isinstance(pair, dict):
                        continue
                    if dim not in existing:
                        existing[dim] = pair
                    else:
                        # Merge sub-keys: later non-empty values win
                        for sub in ("current", "target"):
                            new_val = pair.get(sub)
                            if new_val and (isinstance(new_val, str) and new_val.strip()):
                                existing[dim][sub] = new_val
                continue

            # Union-merge ai_capabilities
            if key == "ai_capabilities" and isinstance(value, list) and isinstance(merged.get(key), list):
                seen = set(merged[key])
                for cap in value:
                    if cap not in seen:
                        merged[key].append(cap)
                        seen.add(cap)
                continue

            merged[key] = value
    return merged
