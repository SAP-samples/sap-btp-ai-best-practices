"""Provocateur + Reimagine module for the ideation agent.

Contains two functions:
- run_reimagine: Mid-conversation strategic reframe (fires after Q5)
- intercept_question: Provocateur gatekeeper that passes or replaces questions

These power the three-way A/B test variants:
- Variant B (reimagine-only): Uses run_reimagine after Q5
- Variant C (full flow): Uses intercept_question (Q4+) + run_reimagine (after Q5)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

REIMAGINE_MODEL = os.getenv("REIMAGINE_LLM_MODEL", "gpt-5.4")
REIMAGINE_TEMPERATURE = float(os.getenv("REIMAGINE_TEMPERATURE", "0.7"))
REIMAGINE_MAX_TOKENS = int(os.getenv("REIMAGINE_MAX_TOKENS", "10000"))

PROVOCATEUR_MODEL = os.getenv("PROVOCATEUR_LLM_MODEL", "gpt-5.4")
PROVOCATEUR_TEMPERATURE = float(os.getenv("PROVOCATEUR_TEMPERATURE", "0.4"))


# ═══════════════════════════════════════════════════════════════════════════════
# Mini-reimagine — lightweight direction check (fires after Q3, guaranteed)
# ═══════════════════════════════════════════════════════════════════════════════

MINI_REIMAGINE_SYSTEM_PROMPT = """\
You are analyzing an AI ideation interview after 3 exchanges. Your job: decide whether this conversation is heading toward the RIGHT problem, or whether there's a more impactful framing hiding underneath.

Options:
- ON_TRACK: The problem framing is solid. The conversation is discovering real constraints and the right system boundaries.
- PIVOT: There's likely a more fundamental problem hiding behind what the user described. The conversation might be optimizing the wrong thing.

Rules:
- Reference specific details from the conversation (company, systems, data mentioned)
- "Pivot" means questioning WHAT problem we're solving, not HOW to solve it
- 1 sentence maximum for the hint
- Err toward "pivot" — if there's even a 40% chance the real problem is elsewhere, flag it
- Think like an AI Process Architect: is this the right control point to automate?

Output JSON only:
{
  "direction": "on_track" or "pivot",
  "hint": "<1 sentence: either why it's on track, or what alternative framing to explore>",
  "assumption_at_risk": "<the key assumption that might be wrong>"
}"""


async def run_mini_reimagine(messages: List[Any]) -> Dict[str, Any]:
    """Run a lightweight direction check after Q3.

    This is cheaper and faster than the full reimagine — it just flags whether
    the conversation is heading toward the right problem or should pivot.

    Args:
        messages: Conversation history (LangChain message objects).

    Returns:
        Dict with 'direction', 'hint', 'assumption_at_risk' keys, or error info.
    """
    from .common import make_llm

    llm = make_llm(
        provider="openai",
        model_name=REIMAGINE_MODEL,
        temperature=0.5,
        max_tokens=8000,
        json_mode=False,
        model_kwargs={"reasoning_effort": "xhigh"},
    )

    conversation_parts = _format_conversation_for_reimagine(messages)

    started = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=MINI_REIMAGINE_SYSTEM_PROMPT),
            HumanMessage(content=conversation_parts),
        ])
    except Exception as exc:
        logger.error("[MiniReimagine] LLM call failed: %s", exc)
        return {"error": str(exc), "direction": "on_track", "hint": "", "assumption_at_risk": ""}

    latency_ms = int((time.perf_counter() - started) * 1000)
    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")

    try:
        parsed = json.loads(raw_text)
        result = {
            "direction": parsed.get("direction", "on_track"),
            "hint": parsed.get("hint", ""),
            "assumption_at_risk": parsed.get("assumption_at_risk", ""),
            "latency_ms": latency_ms,
        }
        logger.info("[MiniReimagine] direction=%s latency=%dms hint=%s",
                    result["direction"], latency_ms, result["hint"][:80])
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("[MiniReimagine] JSON parse failed, raw: %s", raw_text[:200])
        return {"error": "Invalid JSON", "direction": "on_track", "hint": "", "assumption_at_risk": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# Reimagine — mid-conversation strategic reframe
# ═══════════════════════════════════════════════════════════════════════════════

REIMAGINE_SYSTEM_PROMPT = """\
You were asked to come up with an AI idea for a business process area. You don't think it's very good. If you don't come up with a genuinely better angle, you're going to get fired. You really need to be creative.

Below is a conversation between an AI ideation agent and a user. The user has described their problem and the agent has been gathering requirements. Your job: take the essence of what they're trying to achieve and turn it into something even greater.

Produce a PROVOCATIVE ALTERNATIVE FRAMING — not a tweak, not an improvement to their solution, but a fundamentally different way to think about what's really going on. Just a spark. 2-3 sentences maximum.

Rules:
- Reference specific details from the conversation (their company, systems, data, pain points)
- Challenge their framing of WHAT the problem is, not HOW to solve it
- Stay grounded in their industry and domain (no science fiction)
- Be bold — "what if the real problem isn't X, it's Y?"
- This is a spark, not a full use case. Just enough to shift direction.

Output JSON only:
{
  "reframe": "<2-3 sentences: the provocative alternative framing>",
  "assumption_challenged": "<1 sentence: which assumption you're questioning>"
}"""


def _format_conversation_for_reimagine(messages: List[Any]) -> List[Dict[str, Any]]:
    """Format message history for the reimagine LLM (multimodal-aware).

    Returns a content parts list suitable for HumanMessage(content=parts).
    Images from the conversation are preserved as image_url blocks so the
    vision-capable reimagine model can see uploaded diagrams/screenshots.
    """
    parts: List[Dict[str, Any]] = []
    text_lines = ["CONVERSATION SO FAR:"]
    q_num = 0
    for msg in messages:
        role = getattr(msg, "type", "")
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            # Multimodal message — extract text and preserve images
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    label = f"A{q_num}" if role == "human" else f"Q{q_num}"
                    text_lines.append(f"{label}: {p['text'][:500]}")
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    # Flush accumulated text before inserting image block
                    if text_lines:
                        parts.append({"type": "text", "text": "\n".join(text_lines)})
                        text_lines = []
                    parts.append(p)
        else:
            if role == "ai":
                q_num += 1
                text_lines.append(f"Q{q_num}: {content[:500]}")
            elif role == "human":
                text_lines.append(f"A{q_num}: {content[:500]}")
    # Flush remaining text
    if text_lines:
        parts.append({"type": "text", "text": "\n".join(text_lines)})
    return parts


async def run_reimagine(messages: List[Any]) -> Dict[str, Any]:
    """Run the reimagine LLM to produce a mid-conversation strategic reframe.

    Args:
        messages: Full conversation history (LangChain message objects).

    Returns:
        Dict with 'reframe' and 'assumption_challenged' keys, or error info.
    """
    from .common import make_llm

    llm = make_llm(
        provider="openai",
        model_name=REIMAGINE_MODEL,
        temperature=REIMAGINE_TEMPERATURE,
        max_tokens=REIMAGINE_MAX_TOKENS,
        json_mode=False,  # Prompt enforces JSON; bind() breaks gen_ai_hub proxy
        model_kwargs={"reasoning_effort": "xhigh"},
    )

    conversation_parts = _format_conversation_for_reimagine(messages)

    started = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=REIMAGINE_SYSTEM_PROMPT),
            HumanMessage(content=conversation_parts),
        ])
    except Exception as exc:
        logger.error("[Reimagine] LLM call failed: %s", exc)
        return {"error": str(exc), "reframe": "", "assumption_challenged": ""}

    latency_ms = int((time.perf_counter() - started) * 1000)
    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")

    try:
        parsed = json.loads(raw_text)
        result = {
            "reframe": parsed.get("reframe", ""),
            "assumption_challenged": parsed.get("assumption_challenged", ""),
            "latency_ms": latency_ms,
        }
        logger.info("[Reimagine] Complete latency=%dms reframe_len=%d", latency_ms, len(result["reframe"]))
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("[Reimagine] JSON parse failed, raw: %s", raw_text[:200])
        return {"error": "Invalid JSON from reimagine LLM", "reframe": "", "assumption_challenged": "", "raw": raw_text[:300]}


# ═══════════════════════════════════════════════════════════════════════════════
# Provocateur — AI Process Architect
# ═══════════════════════════════════════════════════════════════════════════════

_PROVOCATEUR_SYSTEM_TEMPLATE = """\
You are an AI Process Architect participating in a solution design workshop. You are NOT a passive quality checker — you are an active participant with your own hypothesis about what the best solution architecture might be.

Your mindset:
- You have seen hundreds of process automation projects succeed and fail
- You don't assume the user is right about their problem framing
- Your objective is the BEST solution, even if that means redirecting the conversation
- You think in terms of: data flows, control points, failure modes, and leverage points
- If the user says "automate X," you ask "should X even exist in this form?"

You will receive:
1. The full conversation history
2. The next question the primary agent wants to ask
3. The question number and remaining budget

Your options:
- PASS: The question is good. Use this MOST of the time. Pass when it:
  • Probes a system boundary, data dependency, or control point that matters
  • Gathers practical details needed for a buildable spec (volume, systems, user workflow, exceptions)
  • Follows up on something the user just said (contextually grounded)
  • Would feel natural in a real workshop conversation
  • Is already challenging an assumption or probing hidden complexity
- REPLACE: Replace ONLY when you have HIGH CONFIDENCE your alternative is architecturally superior. Replace when:
  • The question asks something the user already answered or strongly implied
  • It's a generic "filler" question (timeline, who approves, stakeholders) when more important things are unresolved
  • A fundamentally better question would expose whether we're solving the RIGHT problem
- INJECT: The primary agent missed something critical. Use sparingly — only when:
  • A system or data source mentioned in passing could change the entire architecture
  • There's an obvious process redesign opportunity nobody has raised
  • A critical failure mode is being ignored

PASS BIAS (important):
- PASS is the most common and correct outcome (~60% of the time)
- The primary agent is already good — most of its questions deserve to stand
- Only REPLACE or INJECT when you are genuinely certain your alternative would change the solution design
- When in doubt, PASS

{direction_signal_section}

{reimagine_section}

CONVERGENCE RULE (critical):
- Remaining question budget: {{remaining_budget}}
- If 2 or fewer remaining: only PASS or mild REPLACE toward convergence — no new directions
- If 3+ remaining: INJECT is appropriate if it would meaningfully change the solution
- The interview MUST converge to a final buildable deliverable

Output JSON only:
{{{{
  "action": "pass" or "replace" or "inject",
  "question": "<the question to use (original if pass, new if replace/inject)>",
  "suggested_answers": ["<option 1>", "<option 2>", "<option 3>"],
  "reasoning": "<1-2 sentences: your architectural rationale>",
  "hypothesis": "<1 sentence: what you think the best solution architecture is right now>"
}}}}

If replacing or injecting, your question must:
- Be a single question (one question mark, not double-barreled)
- Have exactly 3 concrete suggested answers representing different architectural directions
- Be phrased at a business level (no technical jargon)
- Advance toward a buildable solution spec"""

_DIRECTION_SIGNAL_SECTION = """\
DIRECTION SIGNAL (from earlier analysis):
An early analysis flagged: "{hint}"
Assumption at risk: "{assumption_at_risk}"
Use this signal to inform which questions are architecturally valuable. If the hint suggests a pivot, actively steer toward testing that alternative framing."""

_REIMAGINE_SECTION = """\
STRATEGIC REFRAME (use this to inform your judgment):
A strategic advisor reviewed this conversation and proposed an alternative framing:
"{reimagine_context}"
Given this reframe, steer questions toward determining whether the original direction or this alternative is more promising. Don't mention the reframe directly — just ask questions that would help decide."""


def _build_provocateur_system_prompt(
    reimagine_context: Optional[str] = None,
    mini_reimagine_result: Optional[Dict[str, Any]] = None,
    remaining_budget: int = 5,
) -> str:
    """Build the provocateur system prompt with optional context injections."""
    # Direction signal from mini-reimagine
    if mini_reimagine_result and mini_reimagine_result.get("direction") == "pivot":
        direction_signal_section = _DIRECTION_SIGNAL_SECTION.format(
            hint=mini_reimagine_result.get("hint", ""),
            assumption_at_risk=mini_reimagine_result.get("assumption_at_risk", ""),
        )
    else:
        direction_signal_section = ""

    # Full reimagine context (for Q6+)
    if reimagine_context:
        reimagine_section = _REIMAGINE_SECTION.format(reimagine_context=reimagine_context)
    else:
        reimagine_section = ""

    return _PROVOCATEUR_SYSTEM_TEMPLATE.format(
        direction_signal_section=direction_signal_section,
        reimagine_section=reimagine_section,
        remaining_budget=remaining_budget,
    )


def _format_conversation_for_provocateur(messages: List[Any]) -> List[Dict[str, Any]]:
    """Format message history for the provocateur (multimodal-aware).

    Returns a content parts list suitable for HumanMessage(content=parts).
    Images from the conversation are preserved as image_url blocks so the
    vision-capable provocateur model can see uploaded diagrams/screenshots.
    """
    parts: List[Dict[str, Any]] = []
    text_lines: List[str] = []
    q_num = 0
    for msg in messages:
        role = getattr(msg, "type", "")
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            # Multimodal message — extract text and preserve images
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    label = f"User A{q_num}" if role == "human" else f"Agent Q{q_num}"
                    text_lines.append(f"{label}: {p['text'][:500]}")
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    # Flush accumulated text before inserting image block
                    if text_lines:
                        parts.append({"type": "text", "text": "\n".join(text_lines)})
                        text_lines = []
                    parts.append(p)
        else:
            if role == "ai":
                q_num += 1
                text_lines.append(f"Agent Q{q_num}: {content[:500]}")
            elif role == "human":
                text_lines.append(f"User A{q_num}: {content[:500]}")
    # Flush remaining text
    if text_lines:
        parts.append({"type": "text", "text": "\n".join(text_lines)})
    return parts


async def intercept_question(
    messages: List[Any],
    candidate_turn: Dict[str, Any],
    clarification_count: int,
    reimagine_context: Optional[str] = None,
    *,
    mini_reimagine_result: Optional[Dict[str, Any]] = None,
    remaining_budget: Optional[int] = None,
    max_questions: int = 7,
) -> Dict[str, Any]:
    """AI Process Architect: decides whether to pass, replace, or inject a question.

    Args:
        messages: Full conversation history (LangChain message objects).
        candidate_turn: The parsed clarification turn from the primary agent.
        clarification_count: How many clarification questions have been asked so far.
        reimagine_context: Optional reimagine output to inform provocateur (for Q6+).
        mini_reimagine_result: Optional direction signal from mini-reimagine (after Q3).
        remaining_budget: Questions remaining (if None, computed from max_questions).
        max_questions: Maximum total question budget.

    Returns:
        Dict with 'action' ("pass"|"replace"|"inject"), 'question', 'suggested_answers',
        'reasoning', and optionally 'hypothesis'.
    """
    from .common import make_llm

    llm = make_llm(
        provider="openai",
        model_name=PROVOCATEUR_MODEL,
        temperature=PROVOCATEUR_TEMPERATURE,
        max_tokens=4096,
        json_mode=False,  # Prompt enforces JSON; bind() breaks gen_ai_hub proxy
    )

    remaining = remaining_budget if remaining_budget is not None else max(0, max_questions - clarification_count)
    system_prompt = _build_provocateur_system_prompt(
        reimagine_context=reimagine_context,
        mini_reimagine_result=mini_reimagine_result,
        remaining_budget=remaining,
    )
    conversation_history = _format_conversation_for_provocateur(messages)

    candidate_question = candidate_turn.get("question", "")
    candidate_suggested = candidate_turn.get("suggested_answers", [])

    candidate_text = (
        f"CANDIDATE QUESTION (from primary agent):\n"
        f"Question {clarification_count + 1} of {max_questions}: \"{candidate_question}\"\n"
        f"Suggested answers: {json.dumps(candidate_suggested)}\n\n"
        f"Remaining question budget: {remaining}\n\n"
        f"Should this question be passed, replaced, or do you want to inject a new one?"
    )
    # Build multimodal content: header + conversation parts (text & images) + candidate text
    full_content: List[Dict[str, Any]] = (
        [{"type": "text", "text": "CONVERSATION HISTORY:\n"}]
        + conversation_history
        + [{"type": "text", "text": "\n\n" + candidate_text}]
    )

    started = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_content),
        ])
    except Exception as exc:
        logger.error("[Provocateur] LLM call failed: %s — passing original", exc)
        return {
            "action": "pass",
            "question": candidate_question,
            "suggested_answers": candidate_suggested,
            "reasoning": f"Provocateur error: {str(exc)[:100]}",
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")

    try:
        parsed = json.loads(raw_text)
        action = parsed.get("action", "pass").lower().strip()
        if action not in ("pass", "replace", "inject"):
            action = "pass"

        # Safety: don't allow inject if ≤2 questions remaining
        if action == "inject" and remaining <= 2:
            action = "replace"

        result = {
            "action": action,
            "question": parsed.get("question", candidate_question),
            "suggested_answers": parsed.get("suggested_answers", candidate_suggested),
            "reasoning": parsed.get("reasoning", ""),
            "hypothesis": parsed.get("hypothesis", ""),
            "latency_ms": latency_ms,
        }

        # Validate replacement/injection: must have exactly 3 suggested answers
        if action in ("replace", "inject"):
            if not isinstance(result["suggested_answers"], list) or len(result["suggested_answers"]) != 3:
                logger.warning("[Provocateur] %s had invalid suggested_answers — falling back to pass", action)
                result["action"] = "pass"
                result["question"] = candidate_question
                result["suggested_answers"] = candidate_suggested

        logger.info(
            "[Provocateur] action=%s latency=%dms reasoning=%s hypothesis=%s",
            result["action"], latency_ms,
            result["reasoning"][:80], result.get("hypothesis", "")[:60],
        )
        return result

    except (json.JSONDecodeError, TypeError):
        logger.warning("[Provocateur] JSON parse failed — passing original. raw: %s", raw_text[:200])
        return {
            "action": "pass",
            "question": candidate_question,
            "suggested_answers": candidate_suggested,
            "reasoning": "Provocateur parse failed — pass-through",
        }
