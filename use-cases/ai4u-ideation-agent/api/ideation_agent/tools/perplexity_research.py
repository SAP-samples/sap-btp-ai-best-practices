"""Perplexity Sonar Pro grounding for external market research.

Two independent research functions via SAP AI Core Gen AI Hub orchestration:
  1. Company research — broad company intelligence (fires on turn 1)
  2. Problem research — domain-specific research (fires when agent deems problem clear)

Ported from ideation_deep_agent_extra/tools/perplexity_grounding.py,
simplified for the ideation agent architecture.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List

from app.observability.llm_usage_logging import (
    LLMUsageMeta,
    emit_llm_usage_event,
    emit_llm_usage_for_response,
)

logger = logging.getLogger(__name__)

DEFAULT_SONAR_PRO_MODEL = "sonar-pro"
DEFAULT_MAX_TOKENS = 1800
DEFAULT_TEMPERATURE = 0.2


def perplexity_enabled() -> bool:
    """Returns True when Perplexity research is explicitly enabled via env var."""
    return os.getenv("IDEATION_PERPLEXITY_SKILL_ENABLED", "").strip().lower() in ("true", "1", "yes")

_CITATION_RE = re.compile(r"\[\d+\]")
_URL_RE = re.compile(r"https?://[^\s<>\"'`\[\]]+")


def _read_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _read_env_int(name: str, default: int) -> int:
    raw = _read_env(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sonar_pro_model() -> str:
    return _read_env("IDEATION_PERPLEXITY_SONAR_PRO_MODEL", DEFAULT_SONAR_PRO_MODEL) or DEFAULT_SONAR_PRO_MODEL


def _max_tokens() -> int:
    return _read_env_int("IDEATION_PERPLEXITY_MAX_TOKENS", DEFAULT_MAX_TOKENS)


def _normalize_text(value: Any) -> str:
    return str(value or "").replace("\r", "\n").strip()


def _normalize_inline_text(value: Any) -> str:
    return " ".join(_normalize_text(value).split()).strip()


def _clean_markdown_text(text: Any) -> str:
    value = _normalize_text(text)
    if not value:
        return ""
    value = _CITATION_RE.sub("", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" +([.,;:])", r"\1", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    # Ensure space before markdown links not at start of line
    value = re.sub(r"([^\s\n])\[([^\]]+)\]\(http", r"\1 [\2](http", value)
    # Ensure space between consecutive markdown links
    value = re.sub(r"\)\[([^\]]+)\]\(http", r") [\1](http", value)
    return value.strip()


def _extract_urls(text: str) -> List[Dict[str, str]]:
    details: List[Dict[str, str]] = []
    seen: set = set()
    for raw_url in _URL_RE.findall(text):
        url = raw_url.rstrip(".,;:)")
        if ")(" in url:
            url = url.split(")(")[0]
        if url and url not in seen:
            seen.add(url)
            details.append({"url": url, "title": "", "ref_id": ""})
    return details


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _strip_markdown_links(text: str) -> str:
    """Convert [title](url) → title — keeps descriptive text, drops URLs."""
    return _MD_LINK_RE.sub(r"\1", text)


def _truncate(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    compact = value[:max_chars].rsplit(" ", 1)[0].strip()
    return compact or value[:max_chars].strip()


def _orchestration_imports() -> Dict[str, Any]:
    from gen_ai_hub.orchestration_v2.models.config import ModuleConfig, OrchestrationConfig
    from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
    from gen_ai_hub.orchestration_v2.models.message import SystemMessage, UserMessage
    from gen_ai_hub.orchestration_v2.models.template import (
        PromptTemplatingModuleConfig,
        Template,
    )
    from gen_ai_hub.orchestration_v2.service import OrchestrationService

    return {
        "ModuleConfig": ModuleConfig,
        "OrchestrationConfig": OrchestrationConfig,
        "LLMModelDetails": LLMModelDetails,
        "SystemMessage": SystemMessage,
        "UserMessage": UserMessage,
        "PromptTemplatingModuleConfig": PromptTemplatingModuleConfig,
        "Template": Template,
        "OrchestrationService": OrchestrationService,
    }


def _run_orchestration_prompt(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    context_id: str = "",
    step_name: str = "ideation_agent.perplexity_research",
) -> Any:
    imports = _orchestration_imports()
    Template = imports["Template"]
    SystemMessage = imports["SystemMessage"]
    UserMessage = imports["UserMessage"]
    LLMModelDetails = imports["LLMModelDetails"]
    PromptTemplatingModuleConfig = imports["PromptTemplatingModuleConfig"]
    ModuleConfig = imports["ModuleConfig"]
    OrchestrationConfig = imports["OrchestrationConfig"]
    OrchestrationService = imports["OrchestrationService"]

    template = Template(
        template=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]
    )
    llm = LLMModelDetails(
        name=model_name,
        params={
            "temperature": DEFAULT_TEMPERATURE,
            "max_completion_tokens": int(max_tokens),
        },
    )
    prompt_template = PromptTemplatingModuleConfig(prompt=template, model=llm)
    config = OrchestrationConfig(modules=ModuleConfig(prompt_templating=prompt_template))
    service = OrchestrationService(config=config)
    meta = LLMUsageMeta(
        route="/background/ideation-agent/perplexity",
        method="BACKGROUND",
        correlation_id=context_id or None,
    )
    started = time.perf_counter()
    try:
        result = service.run()
    except Exception:
        emit_llm_usage_event(
            route=meta.route,
            method=meta.method,
            provider="sap-ai-core",
            model=model_name,
            llm_endpoint="orchestration",
            input_tokens=0,
            output_tokens=0,
            outcome="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            correlation_id=meta.correlation_id,
            step_name=step_name,
            feature="perplexity",
        )
        raise
    emit_llm_usage_for_response(
        result,
        meta=meta,
        provider="sap-ai-core",
        model=model_name,
        llm_endpoint="orchestration",
        latency_ms=int((time.perf_counter() - started) * 1000),
        step_name=step_name,
        feature="perplexity",
    )
    return result


def _extract_message_content(result: Any) -> str:
    final_result = getattr(result, "final_result", None)
    choices = getattr(final_result, "choices", None)
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        return _normalize_text(getattr(message, "content", ""))
    return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _company_research_prompt(company_name: str) -> str:
    return (
        f"Research {company_name} and provide a concise company intelligence summary.\n\n"
        "Cover:\n"
        "1. Company overview, operating footprint, and business model.\n"
        "2. Current strategic priorities, transformation themes, or major pressures.\n"
        "3. Digital, data, AI, or automation signals.\n"
        "4. Relevant external risks or constraints.\n\n"
        "Formatting:\n"
        "- Markdown headers, short paragraphs, bullet points. No tables.\n"
        "- Only externally grounded facts.\n"
        "- IMPORTANT: For every factual claim, include the source as a markdown link [title](url). "
        "Do not use citation numbers like [1]. Always use the full URL in markdown link format."
    )


def _problem_research_prompt(company_name: str, user_message: str) -> str:
    return (
        f"Research how {company_name} and similar companies in its industry "
        f"handle the following challenge:\n\n"
        f"{user_message}\n\n"
        "Cover:\n"
        "1. How this company or peers currently address this problem area.\n"
        "2. Industry best practices, leading solutions, or technology approaches.\n"
        "3. Common pain points, failure modes, and success factors.\n"
        "4. Relevant AI/automation opportunities specific to this domain.\n\n"
        "Formatting:\n"
        "- Markdown headers, short paragraphs, bullet points. No tables.\n"
        "- Only externally grounded facts.\n"
        "- IMPORTANT: For every factual claim, include the source as a markdown link [title](url). "
        "Do not use citation numbers like [1]. Always use the full URL in markdown link format."
    )


# ---------------------------------------------------------------------------
# Research functions (synchronous — called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _run_company_research(company_name: str, *, context_id: str = "") -> Dict[str, Any]:
    model = _sonar_pro_model()
    logger.info("[Perplexity] starting company research model=%s company=%s", model, _truncate(company_name, 60))

    result = _run_orchestration_prompt(
        model_name=model,
        system_prompt="You are an expert business researcher and analyst.",
        user_prompt=_company_research_prompt(company_name),
        max_tokens=_max_tokens(),
        context_id=context_id,
        step_name="ideation_agent.perplexity_company",
    )
    company_markdown = _clean_markdown_text(_extract_message_content(result))
    source_details = _extract_urls(company_markdown)

    logger.info("[Perplexity] company research complete len=%s sources=%s", len(company_markdown), len(source_details))

    clean_company = _strip_markdown_links(company_markdown)

    return {
        "status": "ok",
        "company_markdown": clean_company,
        "problem_markdown": "",
        "summary_markdown": clean_company,
        "sources": [s["url"] for s in source_details],
        "source_details": source_details,
        "summary_model": model,
    }


def _run_problem_research(company_name: str, user_message: str, *, context_id: str = "") -> Dict[str, Any]:
    model = _sonar_pro_model()
    logger.info("[Perplexity] starting problem research model=%s company=%s", model, _truncate(company_name, 60))

    result = _run_orchestration_prompt(
        model_name=model,
        system_prompt="You are an expert business researcher and analyst specializing in enterprise technology solutions.",
        user_prompt=_problem_research_prompt(company_name, user_message),
        max_tokens=_max_tokens(),
        context_id=context_id,
        step_name="ideation_agent.perplexity_problem",
    )
    problem_markdown = _clean_markdown_text(_extract_message_content(result))
    source_details = _extract_urls(problem_markdown)

    logger.info("[Perplexity] problem research complete len=%s sources=%s", len(problem_markdown), len(source_details))

    clean_problem = _strip_markdown_links(problem_markdown)

    return {
        "status": "ok",
        "company_markdown": "",
        "problem_markdown": clean_problem,
        "summary_markdown": "",
        "sources": [s["url"] for s in source_details],
        "source_details": source_details,
        "summary_model": model,
    }


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def run_perplexity_research(company_name: str, *, context_id: str = "") -> Dict[str, Any]:
    """Run company intelligence research. Fires on turn 1."""
    normalized_company = _normalize_inline_text(company_name)
    if not normalized_company:
        return {"status": "error", "message": "company_name is required"}

    try:
        return await asyncio.to_thread(_run_company_research, normalized_company, context_id=context_id)
    except Exception as exc:
        logger.exception("[Perplexity] company research failed company=%s", normalized_company)
        return {"status": "error", "message": str(exc)}


async def run_perplexity_problem_research(
    company_name: str,
    user_message: str,
    *,
    context_id: str = "",
) -> Dict[str, Any]:
    """Run problem/domain research. Fires when agent signals problem_clear=true or at turn 3."""
    normalized_company = _normalize_inline_text(company_name)
    normalized_message = _normalize_inline_text(user_message)
    if not normalized_company:
        return {"status": "error", "message": "company_name is required"}
    if not normalized_message:
        return {"status": "error", "message": "user_message is required"}

    try:
        return await asyncio.to_thread(
            _run_problem_research,
            normalized_company,
            normalized_message,
            context_id=context_id,
        )
    except Exception as exc:
        logger.exception("[Perplexity] problem research failed company=%s", normalized_company)
        return {"status": "error", "message": str(exc)}
