from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from app.observability.llm_usage_logging import (
    emit_llm_usage_event,
    emit_llm_usage_for_response,
    meta_from_mapping,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from .common import make_llm
    from .cse_background import get_result as get_cse_result, start_enrichment
    from .linkedin_background import (
        get_result as get_linkedin_result,
        start_enrichment as start_linkedin_enrichment,
    )
    from .perplexity_background import (
        get_result as get_perplexity_result,
        get_problem_result as get_perplexity_problem_result,
        start_enrichment as start_perplexity_enrichment,
        start_problem_enrichment as start_perplexity_problem_enrichment,
    )
    from .response_contract import (
        DEFAULT_CLARIFICATION_ERROR,
        DEFAULT_FINAL_ERROR,
        MAX_CLARIFICATION_QUESTIONS,
        MIN_CLARIFICATION_QUESTIONS,
        extract_question_candidate,
        extract_structured_candidate,
        finalize_normalized_turn,
        looks_like_clarification_candidate,
        make_error_turn,
        strict_normalize_turn,
        to_ui_payload,
    )
    from .skill_catalog import scan_skill_summaries
    from .tools import get_all_tools
    from .tools.company_search import cse_enabled
    from .tools.perplexity_research import perplexity_enabled
    from .tools.linkedin_research import linkedin_enabled
    from .system_prompt import get_system_prompt
    from .state import AgentState
except ImportError:  # pragma: no cover
    from common import make_llm
    from cse_background import get_result as get_cse_result, start_enrichment
    from linkedin_background import (
        get_result as get_linkedin_result,
        start_enrichment as start_linkedin_enrichment,
    )
    from perplexity_background import (
        get_result as get_perplexity_result,
        get_problem_result as get_perplexity_problem_result,
        start_enrichment as start_perplexity_enrichment,
        start_problem_enrichment as start_perplexity_problem_enrichment,
    )
    from response_contract import (
        DEFAULT_CLARIFICATION_ERROR,
        DEFAULT_FINAL_ERROR,
        MAX_CLARIFICATION_QUESTIONS,
        extract_question_candidate,
        extract_structured_candidate,
        finalize_normalized_turn,
        looks_like_clarification_candidate,
        make_error_turn,
        strict_normalize_turn,
        to_ui_payload,
    )
    from skill_catalog import scan_skill_summaries
    from tools import get_all_tools
    from tools.company_search import cse_enabled
    from tools.perplexity_research import perplexity_enabled
    from tools.linkedin_research import linkedin_enabled
    from system_prompt import get_system_prompt
    from state import AgentState


def _extract_tool_calls(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Extracts tool calls from LangGraph messages.

    Args:
        messages: Sequence of LangGraph messages to extract tool calls from.

    Returns:
        List of tool call dictionaries with 'name' and 'args' keys.
    """
    calls: List[Dict[str, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args")
            else:
                name = getattr(call, "name", None)
                args = getattr(call, "args", None)
            if name:
                calls.append({"name": name, "args": args})
    return calls


def _extract_final_text(result: Any) -> str:
    """Extracts the final text response from the graph result.

    Looks for the last AIMessage without tool calls (the final answer).
    Falls back to any message content if no clean final answer is found.

    Args:
        result: The result dictionary from graph.ainvoke().

    Returns:
        The final text response as a string.
    """
    if isinstance(result, dict) and "messages" in result:
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                return message.content or ""
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content:
                return content
    return str(result)


_GRAPH: Optional[Any] = None
_GRAPH_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)

AGENT_LLM_MODEL = "gpt-5.4"
AGENT_INTERVIEW_TEMPERATURE = float(os.getenv("AGENT_INTERVIEW_TEMPERATURE", "0.5"))
AGENT_FINAL_TEMPERATURE = float(os.getenv("AGENT_FINAL_TEMPERATURE", "0.2"))
_CSE_PROMPT_MAX_CHARS = 6000


def _format_cse_for_prompt(enrichment: Dict[str, Any]) -> Optional[str]:
    """Format CSE enrichment data as a concise prompt block (~1500 tokens max)."""
    if not enrichment or enrichment.get("status") != "ok":
        return None

    company = enrichment.get("company") or {}
    company_name = company.get("name", "Unknown")
    ticker = company.get("ticker", "")
    exchange = company.get("exchange", "")
    report_link = enrichment.get("report_link", "")
    analysis_md = enrichment.get("analysis_markdown", "")
    stories_md = enrichment.get("stories_markdown", "")

    parts: List[str] = []
    id_line = f"Company: {company_name}"
    if ticker:
        id_line += f" ({ticker}"
        id_line += f" on {exchange})" if exchange else ")"
    parts.append(id_line)

    if report_link:
        parts.append(f"EDGAR Source: {report_link}")

    if analysis_md:
        truncated = analysis_md[:3200]
        if len(analysis_md) > 3200:
            truncated = truncated.rsplit("\n", 1)[0] + "\n..."
        parts.append(f"\n--- Company Analysis ---\n{truncated}")

    if stories_md:
        truncated = stories_md[:2000]
        if len(stories_md) > 2000:
            truncated = truncated.rsplit("\n", 1)[0] + "\n..."
        parts.append(f"\n--- Related Customer Stories ---\n{truncated}")

    parts.append(
        "\nInstructions:\n"
        "- Use this data to ask more targeted clarification questions\n"
        "- Reference company-specific challenges in your final recommendation\n"
        "- Include relevant customer stories in the solution paragraph when applicable\n"
        "- Do NOT ask the user to identify or confirm their company — already resolved"
    )

    full_text = "\n".join(parts)
    if len(full_text) > _CSE_PROMPT_MAX_CHARS:
        full_text = full_text[:_CSE_PROMPT_MAX_CHARS].rsplit("\n", 1)[0] + "\n..."
    return full_text


_PERPLEXITY_PROMPT_MAX_CHARS = 5000


def _format_perplexity_for_prompt(
    enrichment: Optional[Dict[str, Any]],
    problem_enrichment: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Format Perplexity research data as a concise prompt block."""
    company_md = ""
    problem_md = ""

    if enrichment and enrichment.get("status") == "ok":
        company_md = enrichment.get("company_markdown", "") or enrichment.get("summary_markdown", "")
        problem_md = enrichment.get("problem_markdown", "")

    if not problem_md and problem_enrichment and problem_enrichment.get("status") == "ok":
        problem_md = problem_enrichment.get("problem_markdown", "")

    if not company_md and not problem_md:
        return None

    parts: List[str] = []

    if company_md:
        t = company_md[:2200]
        if len(company_md) > 2200:
            t = t.rsplit("\n", 1)[0] + "\n..."
        parts.append(f"--- Company Intelligence ---\n{t}")

    if problem_md:
        t = problem_md[:2200]
        if len(problem_md) > 2200:
            t = t.rsplit("\n", 1)[0] + "\n..."
        parts.append(f"\n--- Domain & Problem Research ---\n{t}")

    parts.append(
        "\nInstructions:\n"
        "- Use company intelligence to understand the customer's context and constraints\n"
        "- Use domain research to ask sharper questions about their specific challenge\n"
        "- Reference industry best practices in your recommendation\n"
        "- Cite specific market signals when they strengthen the solution rationale"
    )

    full_text = "\n".join(parts)
    if len(full_text) > _PERPLEXITY_PROMPT_MAX_CHARS:
        full_text = full_text[:_PERPLEXITY_PROMPT_MAX_CHARS].rsplit("\n", 1)[0] + "\n..."
    return full_text


_LINKEDIN_PROMPT_MAX_CHARS = 2000


def _format_linkedin_for_prompt(enrichment: Optional[Dict[str, Any]]) -> Optional[str]:
    """Format LinkedIn analysis as a concise prompt block."""
    if not enrichment or enrichment.get("status") != "ok":
        return None

    parts: List[str] = []
    rows = enrichment.get("rows_analyzed", 0)
    window = enrichment.get("time_window", "")
    parts.append(f"LinkedIn hiring analysis: {rows} positions analyzed ({window}).")

    techs = enrichment.get("top_technologies", [])
    if techs:
        tech_str = ", ".join(f"{t['name']} ({t['count']})" for t in techs[:5])
        parts.append(f"Top technologies: {tech_str}")

    ai = enrichment.get("ai_signals", {})
    if ai.get("ai_signal_found"):
        parts.append(f"AI maturity: {ai.get('ai_maturity_level', 'unknown')}. {ai.get('ai_maturity_reasoning', '')}")

    roles = enrichment.get("top_business_vacancies", [])
    if roles:
        role_str = ", ".join(f"{r['role_family']} ({r['count']})" for r in roles[:4])
        parts.append(f"Active business hiring: {role_str}")

    insights = enrichment.get("ideation_insights", {})
    if insights.get("bottlenecks"):
        parts.append(f"Process bottlenecks: {'; '.join(insights['bottlenecks'][:3])}")
    if insights.get("automation_opportunities"):
        parts.append(f"Automation opportunities: {'; '.join(insights['automation_opportunities'][:3])}")

    parts.append(
        "\nInstructions:\n"
        "- Use hiring signals to understand where the company is investing and struggling\n"
        "- Technology mentions indicate integration points for AI solutions\n"
        "- Bottleneck signals suggest areas ripe for automation"
    )

    full_text = "\n".join(parts)
    if len(full_text) > _LINKEDIN_PROMPT_MAX_CHARS:
        full_text = full_text[:_LINKEDIN_PROMPT_MAX_CHARS].rsplit("\n", 1)[0] + "\n..."
    return full_text


def _repair_instruction(
    remaining_questions: int,
    *,
    force_clarification: bool = False,
    question_hint: str = "",
    refinement_mode: bool = False,
) -> str:
    if refinement_mode:
        return (
            "Your previous answer did not match the required JSON output contract.\n"
            "You are in REFINEMENT MODE. Rewrite the same intent as one valid JSON object only.\n"
            "Do not use markdown fences. Do not add explanatory prose.\n"
            "Valid shapes in refinement mode:\n"
            '- {"kind": "refinement", "message": "<your conversational reply>"}\n'
            '- {"kind": "final", "app_type": "...", "industry": "...", "title": "...", "problem": "...", "solution": "..."}\n'
            '- {"kind": "confirmed", "message": "<brief closing>"}\n'
            '- {"kind": "clarification", "message": "...", "question": "...", "suggested_answers": ["...", "...", "..."], "problem_clear": true}\n'
            'Default to kind="refinement" unless the user explicitly requested an update or confirmed.'
        )

    must_return_final = remaining_questions <= 0
    question_instruction = (
        f"Clarification question to preserve: {question_hint}\n"
        if question_hint
        else ""
    )
    if force_clarification and not must_return_final:
        next_step_instruction = 'you must return kind="clarification".'
    else:
        next_step_instruction = (
            'you must return kind="final".'
            if must_return_final
            else 'you may return kind="clarification" or kind="final" as appropriate.'
        )
    return (
        "Your previous answer did not match the required JSON output contract.\n"
        "Rewrite the same intent as one valid JSON object only.\n"
        "Do not use markdown fences.\n"
        "Do not add explanatory prose.\n"
        "If the turn is a clarification, include exactly 3 suggested_answers.\n"
        "Each suggested answer must be short, clickable, and directly answer the question.\n"
        "If you cannot produce a valid clarification, return a valid final object instead.\n"
        "For final turns, required fields are: kind, app_type, industry, title, problem, solution.\n"
        "app_type must be either \"Agent\" or \"Application\".\n"
        "solution must be a build specification (what to build, data sources, workflow, outputs).\n"
        f"{question_instruction}"
        f"If clarification questions remaining is {remaining_questions}, "
        f"{next_step_instruction}"
    )


async def _invoke_repair(
    llm: Any,
    *,
    sys_msg: SystemMessage,
    state_messages: Sequence[Any],
    invalid_text: str,
    repair_prompt: str,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = meta_from_mapping(llm_usage_meta, default_route="/api/ideation-agent/a2a")
    started = time.perf_counter()
    try:
        repaired_response = await llm.ainvoke(
            [sys_msg] + list(state_messages) + [AIMessage(content=invalid_text), HumanMessage(content=repair_prompt)]
        )
    except Exception:
        emit_llm_usage_event(
            route=meta.route,
            method=meta.method,
            user_id=meta.user_id,
            client_host=meta.client_host,
            provider="sap-ai-core",
            model=AGENT_LLM_MODEL,
            llm_endpoint="chat.completions",
            input_tokens=0,
            output_tokens=0,
            outcome="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            correlation_id=meta.correlation_id,
            step_name="ideation_agent.repair",
            feature="ideation_agent",
        )
        raise
    emit_llm_usage_for_response(
        repaired_response,
        meta=meta,
        provider="sap-ai-core",
        model=AGENT_LLM_MODEL,
        llm_endpoint="chat.completions",
        latency_ms=int((time.perf_counter() - started) * 1000),
        step_name="ideation_agent.repair",
        feature="ideation_agent",
    )
    return (
        repaired_response.content
        if isinstance(repaired_response.content, str)
        else str(repaired_response.content or "")
    )


async def _coerce_structured_turn(
    *,
    llm: Any,
    raw_text: str,
    state_messages: Sequence[Any],
    sys_msg: SystemMessage,
    context_id: str,
    clarification_count: int,
    remaining_questions: int,
    refinement_mode: bool = False,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    initial_candidate = extract_structured_candidate(raw_text)
    raw_kind_candidate = initial_candidate.get("kind", "")
    strict_turn = strict_normalize_turn(
        raw_text,
        clarification_count=clarification_count,
        refinement_mode=refinement_mode,
    )
    if strict_turn is not None:
        # Enforce minimum questions: if the model tries to finalize too early,
        # re-prompt it to ask another question (unless forced by remaining=0).
        if (
            strict_turn.get("kind") == "final"
            and not refinement_mode
            and clarification_count < MIN_CLARIFICATION_QUESTIONS
            and remaining_questions > 0
        ):
            logger.info(
                "Premature final blocked: clarification_count=%s < MIN=%s. Re-prompting for more questions. context_id=%s",
                clarification_count,
                MIN_CLARIFICATION_QUESTIONS,
                context_id,
            )
            depth_prompt = (
                "You tried to finalize but have only asked {count} questions (minimum {min} required). "
                "You still have {remaining} questions remaining. Ask another clarification question — "
                "ideally one that challenges an assumption, exposes a hidden constraint, or suggests "
                "a creative angle the user hasn't considered. Return valid JSON with kind=\"clarification\"."
            ).format(
                count=clarification_count,
                min=MIN_CLARIFICATION_QUESTIONS,
                remaining=remaining_questions,
            )
            depth_text = await _invoke_repair(
                llm,
                sys_msg=sys_msg,
                state_messages=state_messages,
                invalid_text=raw_text,
                repair_prompt=depth_prompt,
                llm_usage_meta=llm_usage_meta,
            )
            depth_turn = strict_normalize_turn(
                depth_text,
                clarification_count=clarification_count,
                refinement_mode=refinement_mode,
            )
            if depth_turn is not None and depth_turn.get("kind") == "clarification":
                depth_turn["normalization_source"] = "depth_enforcement"
                return finalize_normalized_turn(
                    depth_turn,
                    clarification_count=clarification_count,
                )
            # If depth enforcement failed to produce a valid clarification,
            # fall through to accept the original final (better than erroring)

        return finalize_normalized_turn(
            strict_turn,
            clarification_count=clarification_count,
        )

    logger.warning(
        "Structured turn validation failed; attempting repair. context_id=%s clarification_count=%s remaining_questions=%s attempt_type=%s",
        context_id,
        clarification_count,
        remaining_questions,
        "generic_repair",
    )
    repaired_text = await _invoke_repair(
        llm,
        sys_msg=sys_msg,
        state_messages=state_messages,
        invalid_text=str(raw_text or ""),
        repair_prompt=_repair_instruction(remaining_questions, refinement_mode=refinement_mode),
        llm_usage_meta=llm_usage_meta,
    )
    repaired_turn = strict_normalize_turn(
        repaired_text,
        clarification_count=clarification_count,
        refinement_mode=refinement_mode,
    )
    if repaired_turn is not None:
        repaired_turn["normalization_source"] = "repair"
        logger.info(
            "Structured turn repair succeeded. context_id=%s clarification_count=%s remaining_questions=%s",
            context_id,
            clarification_count,
            remaining_questions,
        )
        return finalize_normalized_turn(
            repaired_turn,
            clarification_count=clarification_count,
        )

    clarification_source = ""
    if remaining_questions > 0:
        repaired_candidate = extract_structured_candidate(repaired_text)
        raw_kind_candidate = (
            repaired_candidate.get("kind")
            or initial_candidate.get("kind")
            or ""
        )
        if looks_like_clarification_candidate(repaired_text):
            clarification_source = repaired_text
        elif looks_like_clarification_candidate(raw_text):
            clarification_source = raw_text

    if clarification_source:
        structured_candidate = extract_structured_candidate(clarification_source)
        question_hint = (
            structured_candidate.get("question", "").strip()
            or extract_question_candidate(clarification_source)
            or ""
        )
        logger.warning(
            "Structured turn clarification repair attempt. context_id=%s clarification_count=%s remaining_questions=%s attempt_type=%s raw_kind_candidate=%s",
            context_id,
            clarification_count,
            remaining_questions,
            "forced_clarification_repair",
            raw_kind_candidate or "unknown",
        )
        clarification_text = await _invoke_repair(
            llm,
            sys_msg=sys_msg,
            state_messages=state_messages,
            invalid_text=str(clarification_source or ""),
            repair_prompt=_repair_instruction(
                remaining_questions,
                force_clarification=True,
                question_hint=question_hint,
            ),
            llm_usage_meta=llm_usage_meta,
        )
        clarification_turn = strict_normalize_turn(
            clarification_text,
            clarification_count=clarification_count,
            refinement_mode=refinement_mode,
        )
        if clarification_turn is not None and clarification_turn.get("kind") == "clarification":
            clarification_turn["normalization_source"] = "repair"
            logger.info(
                "Forced clarification repair succeeded. context_id=%s clarification_count=%s remaining_questions=%s",
                context_id,
                clarification_count,
                remaining_questions,
            )
            return finalize_normalized_turn(
                clarification_turn,
                clarification_count=clarification_count,
            )

    logger.error(
        "Structured turn generation failed after repair attempts. context_id=%s clarification_count=%s remaining_questions=%s raw_kind_candidate=%s",
        context_id,
        clarification_count,
        remaining_questions,
        raw_kind_candidate or "unknown",
    )
    logger.error(
        "Structured turn full raw output. context_id=%s\n%s",
        context_id,
        str(raw_text or ""),
    )
    logger.error(
        "Structured turn full repaired output. context_id=%s\n%s",
        context_id,
        str(repaired_text or ""),
    )
    error_message = (
        DEFAULT_FINAL_ERROR
        if remaining_questions <= 0 or raw_kind_candidate == "final"
        else DEFAULT_CLARIFICATION_ERROR
    )
    return make_error_turn(error_message, clarification_count=clarification_count)


# ═══════════════════════════════════════════════════════════════════════════════
# Thinking hooks — extensible system for pre-question thinking steps
# ═══════════════════════════════════════════════════════════════════════════════


async def _run_thinking_hooks(
    *,
    state: Dict[str, Any],
    clarification_count: int,
    reimagine_context: Optional[str],
    mini_reimagine_result: Optional[Dict[str, Any]],
    context_id: str,
    cse_state_updates: Dict[str, Any],
    provocateur_interventions: int,
) -> Optional[Dict[str, Any]]:
    """Run thinking hooks before question generation.

    Each hook checks a condition and, if met, runs a thinking function.
    If a hook fires, returns a state update dict (the agent returns early
    with just the thinking card). The frontend then auto-sends "__continue__"
    to get the informed question.

    Returns None if no hook fires (normal question generation proceeds).
    To add a new thinking step: add a new hook entry below.
    """
    # ── Hook 1: Mini-reimagine (direction check before Q3) ──
    if clarification_count == 2 and mini_reimagine_result is None:
        try:
            from .provocateur import run_mini_reimagine
            mini_reimagine_result = await run_mini_reimagine(state["messages"])
            logger.info(
                "[MiniReimagine] direction=%s context=%s hint=%s",
                mini_reimagine_result.get("direction"), context_id,
                mini_reimagine_result.get("hint", "")[:60],
            )
            if mini_reimagine_result.get("direction") == "pivot":
                hint = mini_reimagine_result.get("hint", "")
                if hint:
                    return {
                        "messages": [AIMessage(content=f"[thinking:direction]{hint}")],
                        "last_turn_kind": "thinking",
                        "ui_payload": {"kind": "thinking", "level": "direction"},
                        "reimagine_context": reimagine_context,
                        "provocateur_interventions": provocateur_interventions,
                        "mini_reimagine_result": mini_reimagine_result,
                        "clarification_count": clarification_count,
                        **cse_state_updates,
                    }
        except Exception as exc:
            logger.warning("[MiniReimagine] failed context=%s: %s", context_id, exc)

    # ── Hook 2: Full reimagine (deep reframe before Q5) ──
    if clarification_count == 4 and reimagine_context is None:
        try:
            from .provocateur import run_reimagine
            reimagine_result = await run_reimagine(state["messages"])
            reimagine_context = reimagine_result.get("reframe", "")
            if reimagine_context:
                logger.info(
                    "[Reimagine] Fired context=%s reframe_len=%d",
                    context_id, len(reimagine_context),
                )
                return {
                    "messages": [AIMessage(content=(
                        f"[thinking:reframe]I've been thinking about your problem "
                        f"differently. {reimagine_context}"
                    ))],
                    "last_turn_kind": "thinking",
                    "ui_payload": {"kind": "thinking", "level": "reframe"},
                    "reimagine_context": reimagine_context,
                    "provocateur_interventions": provocateur_interventions,
                    "mini_reimagine_result": mini_reimagine_result,
                    "clarification_count": clarification_count,
                    **cse_state_updates,
                }
        except Exception as exc:
            logger.warning("[Reimagine] failed context=%s: %s", context_id, exc)

    # ── Add future hooks here ──
    # Example: Hook 3 — pre-finalization reflection
    # if clarification_count == 7 and some_condition:
    #     ...

    return None  # No hook fired — proceed to normal question generation


async def _get_graph():
    """Gets or creates the compiled LangGraph agent.

    The graph is compiled once and cached as a singleton. Conversation state
    is managed externally via HANA — no local checkpointer is used.

    Returns:
        The compiled LangGraph graph.
    """
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    async with _GRAPH_LOCK:
        if _GRAPH is not None:
            return _GRAPH

        llm_interview = make_llm(provider="openai", model_name=AGENT_LLM_MODEL, temperature=AGENT_INTERVIEW_TEMPERATURE, json_mode=True)
        llm_final = make_llm(provider="openai", model_name=AGENT_LLM_MODEL, temperature=AGENT_FINAL_TEMPERATURE, json_mode=True)
        tools = await get_all_tools()
        llm_interview_with_tools = llm_interview.bind_tools(tools)
        llm_final_with_tools = llm_final.bind_tools(tools)

        async def skill(state: AgentState):
            summaries = scan_skill_summaries()
            company_name = str(state.get("company_name", "") or "").strip()
            context_id = str(state.get("context_id", "") or "")
            current_cse_status = state.get("cse_status")
            current_perplexity_status = state.get("perplexity_status")

            updates: Dict[str, Any] = {"skill_summaries": summaries}

            if current_cse_status is None and cse_enabled() and company_name:
                await start_enrichment(context_id, company_name)
                updates["cse_status"] = "pending"
            elif current_cse_status is None:
                updates["cse_status"] = "disabled"

            if current_perplexity_status is None and perplexity_enabled() and company_name:
                await start_perplexity_enrichment(context_id, company_name)
                updates["perplexity_status"] = "pending"
            elif current_perplexity_status is None:
                updates["perplexity_status"] = "disabled"

            current_linkedin_status = state.get("linkedin_status")
            if current_linkedin_status is None and linkedin_enabled() and company_name:
                await start_linkedin_enrichment(context_id, company_name)
                updates["linkedin_status"] = "pending"
            elif current_linkedin_status is None:
                updates["linkedin_status"] = "disabled"

            return updates

        async def assistant(state: AgentState):
            clarification_count = int(state.get("clarification_count", 0) or 0)
            remaining_questions = max(
                0,
                MAX_CLARIFICATION_QUESTIONS - clarification_count,
            )

            # Detect refinement mode
            refinement_mode = bool(state.get("refinement_mode"))
            if not refinement_mode and state.get("last_turn_kind") == "final":
                refinement_mode = True
            refinement_clarification_count = int(state.get("refinement_clarification_count", 0) or 0)
            refinement_clarifications_remaining = max(0, 3 - refinement_clarification_count)

            # Poll CSE background enrichment
            context_id = str(state.get("context_id", "") or "")
            cse_enrichment = state.get("cse_enrichment")
            cse_state_updates: Dict[str, Any] = {}

            if state.get("cse_status") == "pending" and not cse_enrichment:
                result = get_cse_result(context_id)
                if result is not None:
                    cse_enrichment = result
                    cse_state_updates["cse_enrichment"] = result
                    cse_state_updates["cse_status"] = (
                        "complete" if result.get("status") == "ok" else "failed"
                    )

            # Poll Perplexity background enrichment
            perplexity_enrichment = state.get("perplexity_enrichment")
            if state.get("perplexity_status") == "pending" and not perplexity_enrichment:
                px_result = get_perplexity_result(context_id)
                if px_result is not None:
                    perplexity_enrichment = px_result
                    cse_state_updates["perplexity_enrichment"] = px_result
                    cse_state_updates["perplexity_status"] = (
                        "complete" if px_result.get("status") == "ok" else "failed"
                    )

            # Poll Perplexity problem research
            perplexity_problem_enrichment = state.get("perplexity_problem_enrichment")
            if state.get("perplexity_problem_status") == "pending" and not perplexity_problem_enrichment:
                px_problem = get_perplexity_problem_result(context_id)
                if px_problem is not None:
                    perplexity_problem_enrichment = px_problem
                    cse_state_updates["perplexity_problem_enrichment"] = px_problem
                    cse_state_updates["perplexity_problem_status"] = (
                        "complete" if px_problem.get("status") == "ok" else "failed"
                    )

            # Poll LinkedIn background enrichment
            linkedin_enrichment = state.get("linkedin_enrichment")
            if state.get("linkedin_status") == "pending" and not linkedin_enrichment:
                li_result = get_linkedin_result(context_id)
                if li_result is not None:
                    linkedin_enrichment = li_result
                    cse_state_updates["linkedin_enrichment"] = li_result
                    cse_state_updates["linkedin_status"] = (
                        "complete" if li_result.get("status") == "ok" else "failed"
                    )

            cse_context_str = _format_cse_for_prompt(cse_enrichment) if cse_enrichment else None
            perplexity_context_str = _format_perplexity_for_prompt(
                perplexity_enrichment,
                problem_enrichment=perplexity_problem_enrichment,
            )
            linkedin_context_str = _format_linkedin_for_prompt(linkedin_enrichment)

            # ── Pre-turn thinking hooks ──────────────────────────────────────
            # Generic system: any thinking step can fire before a question.
            # If a hook fires, the agent returns JUST the thinking card.
            # The frontend auto-sends "__continue__" to get the informed question.
            conversation_variant = state.get("conversation_variant")
            reimagine_context = state.get("reimagine_context")
            mini_reimagine_result = state.get("mini_reimagine_result")
            provocateur_interventions = int(state.get("provocateur_interventions", 0) or 0)

            if conversation_variant and not refinement_mode:
                # Check if this is a "__continue__" call (thinking already fired, now generate question)
                last_msg = state["messages"][-1] if state["messages"] else None
                is_continue = (
                    last_msg
                    and getattr(last_msg, "type", "") == "ai"
                    and str(getattr(last_msg, "content", "")).startswith("[thinking:")
                )

                if not is_continue:
                    # Run thinking hooks — return early if one fires
                    thinking_result = await _run_thinking_hooks(
                        state=state,
                        clarification_count=clarification_count,
                        reimagine_context=reimagine_context,
                        mini_reimagine_result=mini_reimagine_result,
                        context_id=context_id,
                        cse_state_updates=cse_state_updates,
                        provocateur_interventions=provocateur_interventions,
                    )
                    if thinking_result is not None:
                        return thinking_result

            # ── Build system prompt (now WITH thinking context available) ──
            system_prompt = get_system_prompt(
                state.get("skill_summaries", []),
                clarification_count=clarification_count,
                remaining_questions=remaining_questions,
                cse_context=cse_context_str,
                market_context=perplexity_context_str,
                linkedin_context=linkedin_context_str,
                refinement_mode=refinement_mode,
                refinement_clarifications_remaining=refinement_clarifications_remaining,
            )

            # Inject thinking context into system prompt so the question is informed
            if not refinement_mode:
                if reimagine_context:
                    if remaining_questions <= 0:
                        system_prompt += (
                            "\n\n[STRATEGIC REFRAME — INTEGRATE INTO CORE SOLUTION]\n"
                            "A process architect identified an alternative framing:\n"
                            f"{reimagine_context}\n\n"
                            "Integration rules:\n"
                            "1. If the user embraced or validated this reframe → make it PRIMARY "
                            "in WHAT IT DOES. The user's original direction becomes secondary.\n"
                            "2. If the user acknowledged but wants to start elsewhere → ENABLE "
                            "this direction (e.g., phase 1 does X, architecture supports Y).\n"
                            "3. If the user rejected it → respect their choice, brief 'Consider also' only.\n"
                            "[/STRATEGIC REFRAME]"
                        )
                    else:
                        system_prompt += (
                            "\n\n[STRATEGIC REFRAME — ACTIVE CONTEXT]\n"
                            "A process architect proposed this alternative framing:\n"
                            f"{reimagine_context}\n\n"
                            "Your next question should TEST this hypothesis — not by asking "
                            "'should we go direction A or B?' but by probing for evidence that "
                            "confirms or denies whether the reframe is correct. Ask about:\n"
                            "- What happens downstream when the current process fails\n"
                            "- Who owns the problem the reframe identifies\n"
                            "- Whether anyone has tried to address the root cause before\n"
                            "- What the consequences are of NOT solving the deeper issue\n\n"
                            "The user's answer will tell you whether the reframe is valid. "
                            "Use that signal when producing the final solution.\n"
                            "[/STRATEGIC REFRAME]"
                        )
                elif mini_reimagine_result and mini_reimagine_result.get("direction") == "pivot":
                    system_prompt += (
                        "\n\n[DIRECTION SIGNAL]\n"
                        "An early analysis flagged a potential pivot:\n"
                        f"{mini_reimagine_result.get('hint', '')}\n"
                        f"Assumption at risk: {mini_reimagine_result.get('assumption_at_risk', '')}\n\n"
                        "Let this inform your questions — probe whether this alternative "
                        "framing has merit before moving to implementation details.\n"
                        "[/DIRECTION SIGNAL]"
                    )

            sys_msg = SystemMessage(content=system_prompt)
            llm_usage_meta = state.get("llm_usage_meta")
            usage_meta = meta_from_mapping(llm_usage_meta, default_route="/api/ideation-agent/a2a")

            # Select temperature-appropriate LLM based on conversation phase
            should_use_final_llm = refinement_mode or remaining_questions <= 0
            active_llm = llm_final_with_tools if should_use_final_llm else llm_interview_with_tools
            active_llm_bare = llm_final if should_use_final_llm else llm_interview
            logger.debug(
                "Using %s temperature (phase=%s, remaining_q=%d)",
                "final" if should_use_final_llm else "interview",
                "refinement" if refinement_mode else "interview",
                remaining_questions,
            )

            started = time.perf_counter()
            try:
                response = await active_llm.ainvoke([sys_msg] + state["messages"])
            except Exception:
                emit_llm_usage_event(
                    route=usage_meta.route,
                    method=usage_meta.method,
                    user_id=usage_meta.user_id,
                    client_host=usage_meta.client_host,
                    provider="sap-ai-core",
                    model=AGENT_LLM_MODEL,
                    llm_endpoint="chat.completions",
                    input_tokens=0,
                    output_tokens=0,
                    outcome="error",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    correlation_id=usage_meta.correlation_id,
                    step_name="ideation_agent.assistant",
                    feature="ideation_agent",
                )
                raise
            emit_llm_usage_for_response(
                response,
                meta=usage_meta,
                provider="sap-ai-core",
                model=AGENT_LLM_MODEL,
                llm_endpoint="chat.completions",
                latency_ms=int((time.perf_counter() - started) * 1000),
                step_name="ideation_agent.assistant",
                feature="ideation_agent",
            )
            if getattr(response, "tool_calls", None):
                return {"messages": [response], **cse_state_updates}

            raw_text = response.content if isinstance(response.content, str) else str(response.content or "")
            normalized_turn = await _coerce_structured_turn(
                llm=active_llm_bare,
                raw_text=raw_text,
                state_messages=state["messages"],
                sys_msg=sys_msg,
                context_id=context_id,
                clarification_count=clarification_count,
                remaining_questions=remaining_questions,
                refinement_mode=refinement_mode,
                llm_usage_meta=llm_usage_meta,
            )
            response.content = normalized_turn.get("display_text", "")

            # ── Provocateur intercept (post-response, evaluates the informed question) ──
            if conversation_variant and not refinement_mode:
                turn_kind_check = str(normalized_turn.get("kind", "")).strip()

                if (
                    conversation_variant == "provocateur"
                    and turn_kind_check == "clarification"
                    and clarification_count >= 3  # provocateur_start_at=3
                ):
                    try:
                        from .provocateur import intercept_question
                        intercepted = await intercept_question(
                            state["messages"],
                            normalized_turn,
                            clarification_count,
                            reimagine_context=reimagine_context,
                            mini_reimagine_result=mini_reimagine_result,
                            remaining_budget=remaining_questions,
                        )
                        if intercepted.get("action") in ("replace", "inject"):
                            normalized_turn["question"] = intercepted["question"]
                            normalized_turn["suggested_answers"] = intercepted["suggested_answers"]
                            from .response_contract import finalize_normalized_turn
                            normalized_turn = finalize_normalized_turn(
                                normalized_turn,
                                clarification_count=clarification_count,
                            )
                            response.content = normalized_turn.get("display_text", "")
                            provocateur_interventions += 1
                            logger.info(
                                "[Provocateur] %s Q%d context=%s reason=%s",
                                intercepted["action"], clarification_count + 1, context_id,
                                intercepted.get("reasoning", "")[:80],
                            )
                    except Exception as exc:
                        logger.warning("[Provocateur] intercept_question failed context=%s: %s", context_id, exc)

            # Adaptive problem research trigger (skip in refinement mode)
            company_name = str(state.get("company_name", "") or "").strip()
            if not refinement_mode:
                should_fire_problem = (
                    perplexity_enabled()
                    and state.get("perplexity_problem_status") is None
                    and company_name
                    and (normalized_turn.get("problem_clear", False) or clarification_count >= 2)
                )
                if should_fire_problem:
                    user_messages = [
                        str(getattr(m, "content", "") or "")
                        for m in state["messages"]
                        if getattr(m, "type", "") == "human"
                    ]
                    accumulated = "\n".join(user_messages)
                    await start_perplexity_problem_enrichment(context_id, company_name, accumulated)
                    cse_state_updates["perplexity_problem_status"] = "pending"

            # State transitions
            turn_kind = str(normalized_turn.get("kind", "")).strip()
            state_updates: Dict[str, Any] = {
                "messages": [response],
                "last_turn_kind": turn_kind or "error",
                "ui_payload": to_ui_payload(normalized_turn),
                **cse_state_updates,
            }

            # Persist provocateur/reimagine state
            if conversation_variant:
                state_updates["reimagine_context"] = reimagine_context
                state_updates["provocateur_interventions"] = provocateur_interventions
                state_updates["mini_reimagine_result"] = mini_reimagine_result

            if turn_kind == "final":
                state_updates["clarification_count"] = int(normalized_turn.get("clarification_count", clarification_count) or 0)
                state_updates["refinement_mode"] = True
                state_updates["refinement_clarification_count"] = 0
            elif turn_kind == "refinement":
                state_updates["clarification_count"] = clarification_count
            elif turn_kind == "clarification" and refinement_mode:
                state_updates["refinement_clarification_count"] = refinement_clarification_count + 1
                state_updates["clarification_count"] = clarification_count
            else:
                state_updates["clarification_count"] = int(normalized_turn.get("clarification_count", clarification_count) or 0)

            return state_updates

        graph = StateGraph(AgentState)
        graph.add_node("skill", skill)
        graph.add_node("assistant", assistant)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_edge(START, "skill")
        graph.add_edge("skill", "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")

        _GRAPH = graph.compile()
        return _GRAPH


def _rebuild_langchain_messages(stored_messages: List[Dict[str, Any]], file_blocks_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> List[Any]:
    """Convert stored message dicts (from HANA) to LangChain message objects.

    Supports multimodal messages: if a user message has file_content_blocks
    in its metadata, the HumanMessage is constructed with mixed content
    (text + image_url blocks) for native multimodal input.

    Also supports the new Object Store flow: if metadata has "files" (list of
    file_id references), the pre-resolved content blocks are looked up from
    file_blocks_cache.
    """
    messages = []
    for msg in stored_messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("agent", "assistant"):
            # Strip [thinking:] prefix so LLM sees clean context on reload
            if content.startswith("[thinking:"):
                parts = content.split("\n\n", 1)
                content = parts[1] if len(parts) > 1 else content
            messages.append(AIMessage(content=content))
        elif role == "user":
            # Check for multimodal content (file_content_blocks in metadata)
            metadata = msg.get("metadata") or {}
            file_blocks = metadata.get("file_content_blocks")
            # New flow: resolve file_id references from cache
            files_meta = metadata.get("files")
            if file_blocks and isinstance(file_blocks, list):
                # Legacy: inline base64 content blocks
                multimodal_content = [{"type": "text", "text": content}]
                multimodal_content.extend(file_blocks)
                messages.append(HumanMessage(content=multimodal_content))
            elif files_meta and isinstance(files_meta, list) and file_blocks_cache:
                # New Object Store flow: look up pre-resolved blocks
                multimodal_content = [{"type": "text", "text": content}]
                for f in files_meta:
                    fid = f.get("file_id", "")
                    if fid in file_blocks_cache:
                        multimodal_content.extend(file_blocks_cache[fid])
                if len(multimodal_content) > 1:
                    messages.append(HumanMessage(content=multimodal_content))
                else:
                    messages.append(HumanMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
    return messages


async def _build_file_blocks_cache(
    stored_messages: List[Dict[str, Any]], context_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Pre-resolve Object Store file references into LLM content blocks.

    Scans stored messages for the new "files" metadata format and downloads
    processed outputs from Object Store, returning a cache keyed by file_id.

    Handles:
    - Images: base64 image_url blocks
    - PDF: re-analyzed via Responses API for native understanding (text + vision)
    - Office/Text/Email: re-analyzed via Responses API (text extraction)
    """
    import base64
    cache: Dict[str, List[Dict[str, Any]]] = {}

    # Collect all file_ids that need resolving
    file_refs = []
    for msg in stored_messages:
        metadata = msg.get("metadata") or {}
        files_meta = metadata.get("files")
        if files_meta and isinstance(files_meta, list):
            for f in files_meta:
                fid = f.get("file_id", "")
                if fid and fid not in cache:
                    file_refs.append(f)

    if not file_refs:
        return cache

    try:
        from app.services.object_store import get_object_store
        obj_store = await get_object_store()

        # Extension mapping for document types (used to locate the stored file)
        _doc_ext_map = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/msword": "doc",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.ms-excel": "xls",
            "application/vnd.oasis.opendocument.text": "odt",
            "application/vnd.oasis.opendocument.presentation": "odp",
            "application/vnd.oasis.opendocument.spreadsheet": "ods",
            "application/rtf": "rtf",
            "text/csv": "csv",
            "text/tab-separated-values": "tsv",
            "message/rfc822": "eml",
            "text/plain": "txt",
            "text/markdown": "md",
            "application/json": "json",
            "text/xml": "xml",
            "text/yaml": "yaml",
            "text/x-python": "py",
            "text/javascript": "js",
            "text/html": "html",
            "text/css": "css",
            "application/sql": "sql",
        }

        for f in file_refs:
            fid = f["file_id"]
            fctx = f.get("context_id") or context_id
            num_outputs = f.get("num_outputs", 1)
            original_type = f.get("original_type", "")
            blocks = []

            if original_type and not original_type.startswith("image/"):
                # Documents (PDF, Office, text, email, etc.): analyze via Responses API
                doc_ext = _doc_ext_map.get(original_type, "bin")
                output_key = f"{fctx}/{fid}/0.{doc_ext}"
                doc_data = await obj_store.download(output_key)
                if doc_data:
                    from app.routers.ideation_agent import _analyze_document_via_responses_api
                    doc_text = await _analyze_document_via_responses_api(
                        doc_data, f.get("filename", "document"), original_type
                    )
                    if doc_text:
                        blocks.append({"type": "text", "text": doc_text})
            else:
                # Images: existing image_url flow
                for idx in range(num_outputs):
                    output_key = f"{fctx}/{fid}/{idx}.png"
                    img_data = await obj_store.download(output_key)
                    if img_data:
                        b64 = base64.b64encode(img_data).decode("ascii")
                        blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        })
            if blocks:
                cache[fid] = blocks
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Failed to build file blocks cache: %s", exc)

    return cache


async def _ensure_research_loaded(context_id: str) -> None:
    """Load research from HANA into memory cache if not already present."""
    from . import cse_background, perplexity_background, linkedin_background
    await cse_background.load_from_hana(context_id)
    await perplexity_background.load_from_hana(context_id)
    await linkedin_background.load_from_hana(context_id)


async def run_agent(
    user_text: str,
    context_id: str,
    company_name: str = "",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
    prior_state: Optional[Dict[str, Any]] = None,
    llm_usage_meta: Optional[Dict[str, Any]] = None,
    variant: str = "standard",
    file_content_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Runs the agent with a user message in a specific conversation context.

    All conversation state is persisted in HANA. On each turn the full message
    history is reconstructed from stored messages and passed to the graph.

    Args:
        user_text: The user's message text.
        context_id: The conversation context identifier (maps to thread_id).
        company_name: Optional company name for CSE background enrichment.
        prior_messages: Stored messages from HANA for multi-turn resume.
        prior_state: Stored agent state from HANA (clarification_count, research statuses, etc.).
        llm_usage_meta: Optional LLM usage metadata for observability.
        variant: Conversation variant — "standard" | "reimagine" | "provocateur".
        file_content_blocks: Optional list of image content blocks for multimodal input.

    Returns:
        Dictionary with 'text' (final response) and 'tool_calls' (list of tools used).
    """
    graph = await _get_graph()
    config = {"configurable": {"thread_id": context_id}}

    await _ensure_research_loaded(context_id)

    # Build the current user message (text-only or multimodal)
    if file_content_blocks:
        current_user_content = [{"type": "text", "text": user_text}] + file_content_blocks
    else:
        current_user_content = user_text

    if prior_messages:
        # Build file_blocks_cache for messages that have Object Store file references
        file_blocks_cache = await _build_file_blocks_cache(prior_messages, context_id)
        lc_messages = _rebuild_langchain_messages(prior_messages, file_blocks_cache=file_blocks_cache)
        lc_messages.append(HumanMessage(content=current_user_content))
        initial_state: Dict[str, Any] = {
            "messages": lc_messages,
            "context_id": context_id,
            "company_name": company_name,
            "llm_usage_meta": llm_usage_meta,
            "conversation_variant": variant if variant in ("reimagine", "provocateur") else None,
        }
        if prior_state:
            for key in ("clarification_count", "last_turn_kind", "refinement_mode",
                        "refinement_clarification_count"):
                if key in prior_state:
                    initial_state[key] = prior_state[key]
            for key in ("cse_status", "perplexity_status", "perplexity_problem_status",
                        "linkedin_status"):
                if prior_state.get(key):
                    initial_state[key] = prior_state[key]
            # Restore provocateur/reimagine state
            for key in ("reimagine_context", "provocateur_interventions", "mini_reimagine_result"):
                if key in prior_state:
                    initial_state[key] = prior_state[key]
            # Restore conversation_variant from state if not explicitly set on this turn
            if not initial_state.get("conversation_variant") and prior_state.get("conversation_variant"):
                initial_state["conversation_variant"] = prior_state["conversation_variant"]
    else:
        initial_state = {
            "messages": [HumanMessage(content=current_user_content)],
            "context_id": context_id,
            "company_name": company_name,
            "llm_usage_meta": llm_usage_meta,
            "conversation_variant": variant if variant in ("reimagine", "provocateur") else None,
        }

    result = await graph.ainvoke(initial_state, config=config)

    result_messages = result.get("messages", []) if isinstance(result, dict) else []

    # Extract provocateur/reimagine state for persistence across turns
    provocateur_state = None
    if isinstance(result, dict) and result.get("conversation_variant"):
        provocateur_state = {
            "reimagine_context": result.get("reimagine_context"),
            "provocateur_interventions": result.get("provocateur_interventions"),
            "mini_reimagine_result": result.get("mini_reimagine_result"),
            "conversation_variant": result.get("conversation_variant"),
        }

    return {
        "text": _extract_final_text(result),
        "tool_calls": _extract_tool_calls(result_messages),
        "ui": result.get("ui_payload", {}) if isinstance(result, dict) else {},
        "provocateur_state": provocateur_state,
    }
