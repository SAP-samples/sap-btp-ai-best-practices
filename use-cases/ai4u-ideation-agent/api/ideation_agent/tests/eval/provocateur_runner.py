"""Conversation runners for provocateur + reimagine variants.

Provides two runner functions:
- run_reimagine_conversation (Variant B): Primary agent + reimagine at Q5
- run_provocateur_conversation (Variant C): Provocateur (Q4+) + reimagine (Q5+)

Both reuse the existing runner infrastructure (make_agent_llm, simulate_user_answer,
parse_json, ConversationResult) and extend it with provocateur/reimagine logic.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ideation_agent.common import make_llm
from ideation_agent.provocateur import intercept_question, run_reimagine, run_mini_reimagine
from ideation_agent.system_prompt import get_system_prompt
from ideation_agent.skill_catalog import scan_skill_summaries

from .config import RunnerConfig
from .runner import (
    ConversationResult,
    make_agent_llm,
    make_eval_llm,
    parse_json,
    simulate_user_answer,
    _extract_text_content,
    _strip_two_tier_block,
)


@dataclass
class ProvocateurConversationResult(ConversationResult):
    """Extended result with provocateur/reimagine metadata."""

    reimagine_output: Optional[Dict[str, Any]] = None
    provocateur_interventions: int = 0
    provocateur_passes: int = 0
    variant: str = ""  # "reimagine" or "provocateur"


async def run_reimagine_conversation(
    llm,
    eval_llm,
    scenario: Dict[str, str],
    config: RunnerConfig,
    *,
    skill_summaries: Optional[List] = None,
    system_prompt_fn: Optional[Callable[..., str]] = None,
) -> ProvocateurConversationResult:
    """Run Variant B: Primary agent solo, reimagine fires after Q5.

    Flow:
        Q1-Q5: Primary agent solo (standard flow)
        --- After Q5: reimagine fires ---
            Full conversation → reimagine LLM → spark stored
            Insert spark as visible AI message in conversation
        Q6-Q7: Primary agent continues with reimagine context in system prompt
            Can finalize immediately if reimagine didn't open a new direction
        Final delivery: primary agent with reimagine context
    """
    if skill_summaries is None:
        skill_summaries = scan_skill_summaries()

    turns: List[Dict[str, Any]] = []
    messages = [HumanMessage(content=scenario["message"])]
    clarification_count = 0
    reimagine_context = None
    reimagine_output = None
    t_start = time.time()

    prompt_builder = system_prompt_fn or get_system_prompt

    for turn_idx in range(config.max_turns + 1):
        remaining = max(0, config.max_questions - clarification_count)

        # Build system prompt (inject reimagine context only on the final turn)
        system_prompt = prompt_builder(
            skill_summaries,
            clarification_count=clarification_count,
            remaining_questions=remaining,
        )

        if not config.two_tier_strategy:
            system_prompt = _strip_two_tier_block(system_prompt)

        # If reimagine context exists and this is the final delivery turn,
        # inject it into the system prompt
        if reimagine_context and remaining == 0:
            system_prompt += (
                "\n\n[REIMAGINE CONTEXT — A strategic advisor analyzed this conversation "
                "and proposed an alternative framing. Incorporate this perspective into "
                "your \"Consider also\" section if relevant:]\n"
                f"{reimagine_context}\n"
                "[/REIMAGINE CONTEXT]"
            )

        t0 = time.time()
        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
        elapsed = time.time() - t0

        raw_text = _extract_text_content(response.content)
        parsed = parse_json(raw_text)

        # Repair if needed
        if config.repair_enabled:
            if parsed.get("kind") not in ("clarification", "final") and "error" not in parsed:
                retry_msg = "Your response must be valid JSON with kind='clarification' or kind='final'. Try again."
                t0 = time.time()
                response = await llm.ainvoke(
                    [SystemMessage(content=system_prompt)] + messages
                    + [AIMessage(content=raw_text), HumanMessage(content=retry_msg)]
                )
                elapsed = time.time() - t0
                raw_text = _extract_text_content(response.content)
                parsed = parse_json(raw_text)

        parsed["_elapsed_s"] = round(elapsed, 1)
        parsed["_turn"] = turn_idx + 1

        # MIN enforcement: block premature finals (matches production behavior)
        if (
            parsed.get("kind") == "final"
            and clarification_count < config.min_questions
        ):
            depth_msg = (
                f"You must ask at least {config.min_questions} questions before finalizing. "
                f"You've only asked {clarification_count}. Ask another clarification question "
                "— ideally one that challenges an assumption or probes a hidden constraint."
            )
            t0 = time.time()
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt)] + messages
                + [AIMessage(content=raw_text), HumanMessage(content=depth_msg)]
            )
            elapsed = time.time() - t0
            raw_text_retry = _extract_text_content(response.content)
            parsed_retry = parse_json(raw_text_retry)
            if parsed_retry.get("kind") == "clarification":
                parsed = parsed_retry
                parsed["_elapsed_s"] = round(elapsed, 1)
                parsed["_turn"] = turn_idx + 1
                parsed["_depth_enforced"] = True
                raw_text = raw_text_retry
            # else: accept original final as fallback

        # If final → done
        if parsed.get("kind") == "final":
            turns.append(parsed)
            break

        # If not a clarification either, append and break
        if parsed.get("kind") != "clarification":
            turns.append(parsed)
            break

        turns.append(parsed)
        clarification_count += 1
        messages.append(AIMessage(content=raw_text))

        # Fire reimagine after the configured question count
        if clarification_count == config.reimagine_after and reimagine_context is None:
            reimagine_result = await run_reimagine(messages)
            reimagine_output = reimagine_result
            reimagine_context = reimagine_result.get("reframe", "")

            if reimagine_context:
                # Insert as visible message
                insight_msg = (
                    f"💡 I've been thinking about your problem differently. "
                    f"{reimagine_context} "
                    f"Let me explore this angle with a couple more questions."
                )
                messages.append(AIMessage(content=insight_msg))
                turns.append({
                    "kind": "insight",
                    "message": insight_msg,
                    "_reimagine": reimagine_result,
                    "_elapsed_s": reimagine_result.get("latency_ms", 0) / 1000,
                    "_turn": turn_idx + 1,
                })

        # Simulate user answer
        question = parsed.get("question", "")
        suggested = parsed.get("suggested_answers", [])
        answer = await simulate_user_answer(
            eval_llm, question, suggested, scenario["message"],
            seed=hash(scenario["name"] + str(clarification_count)),
        )
        turns[-1]["_user_answer"] = answer
        messages.append(HumanMessage(content=answer))

    return ProvocateurConversationResult(
        turns=turns,
        elapsed_s=round(time.time() - t_start, 1),
        config=config,
        scenario_name=scenario["name"],
        reimagine_output=reimagine_output,
        provocateur_interventions=0,
        provocateur_passes=0,
        variant="reimagine",
    )


async def run_provocateur_conversation(
    llm,
    eval_llm,
    scenario: Dict[str, str],
    config: RunnerConfig,
    *,
    skill_summaries: Optional[List] = None,
    system_prompt_fn: Optional[Callable[..., str]] = None,
    thinking_first: bool = False,
) -> ProvocateurConversationResult:
    """Run Variant C/D/E: Provocateur active (Q4+) + reimagine.

    Args:
        thinking_first: If True (variant E), thinking fires BEFORE the question
            is generated so the question is informed by the thinking.
            If False (variant D), thinking fires AFTER and is bolted on visually.
        Final delivery: primary agent with reimagine context in system prompt
    """
    if skill_summaries is None:
        skill_summaries = scan_skill_summaries()

    turns: List[Dict[str, Any]] = []
    messages = [HumanMessage(content=scenario["message"])]
    clarification_count = 0
    reimagine_context = None
    reimagine_output = None
    mini_reimagine_result = None
    provocateur_interventions = 0
    provocateur_passes = 0
    t_start = time.time()

    prompt_builder = system_prompt_fn or get_system_prompt

    for turn_idx in range(config.max_turns + 1):
        remaining = max(0, config.max_questions - clarification_count)

        # ── Pre-turn thinking (variant E): fire BEFORE generating the question ──
        if thinking_first:
            if clarification_count == 2 and mini_reimagine_result is None:
                mini_reimagine_result = await run_mini_reimagine(messages)
                if mini_reimagine_result.get("direction") == "pivot":
                    turns.append({
                        "kind": "direction_signal",
                        "_mini_reimagine": mini_reimagine_result,
                        "_turn": turn_idx + 1,
                    })

            if clarification_count == config.reimagine_after - 1 and reimagine_context is None:
                reimagine_result = await run_reimagine(messages)
                reimagine_output = reimagine_result
                reimagine_context = reimagine_result.get("reframe", "")
                if reimagine_context:
                    turns.append({
                        "kind": "insight",
                        "message": f"[thinking:reframe]{reimagine_context}",
                        "_reimagine": reimagine_result,
                        "_elapsed_s": reimagine_result.get("latency_ms", 0) / 1000,
                        "_turn": turn_idx + 1,
                    })

        # Build system prompt (now WITH thinking context available)
        system_prompt = prompt_builder(
            skill_summaries,
            clarification_count=clarification_count,
            remaining_questions=remaining,
        )

        if not config.two_tier_strategy:
            system_prompt = _strip_two_tier_block(system_prompt)

        # Inject thinking context for primary agent (mirrors production agent.py)
        if reimagine_context:
            if remaining == 0:
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

        t0 = time.time()
        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
        elapsed = time.time() - t0

        raw_text = _extract_text_content(response.content)
        parsed = parse_json(raw_text)

        # Repair if needed
        if config.repair_enabled:
            if parsed.get("kind") not in ("clarification", "final") and "error" not in parsed:
                retry_msg = "Your response must be valid JSON with kind='clarification' or kind='final'. Try again."
                t0 = time.time()
                response = await llm.ainvoke(
                    [SystemMessage(content=system_prompt)] + messages
                    + [AIMessage(content=raw_text), HumanMessage(content=retry_msg)]
                )
                elapsed = time.time() - t0
                raw_text = _extract_text_content(response.content)
                parsed = parse_json(raw_text)

        parsed["_elapsed_s"] = round(elapsed, 1)
        parsed["_turn"] = turn_idx + 1

        # MIN enforcement: block premature finals (matches production behavior)
        if (
            parsed.get("kind") == "final"
            and clarification_count < config.min_questions
        ):
            depth_msg = (
                f"You must ask at least {config.min_questions} questions before finalizing. "
                f"You've only asked {clarification_count}. Ask another clarification question "
                "— ideally one that challenges an assumption or probes a hidden constraint."
            )
            t0 = time.time()
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt)] + messages
                + [AIMessage(content=raw_text), HumanMessage(content=depth_msg)]
            )
            elapsed = time.time() - t0
            raw_text_retry = _extract_text_content(response.content)
            parsed_retry = parse_json(raw_text_retry)
            if parsed_retry.get("kind") == "clarification":
                parsed = parsed_retry
                parsed["_elapsed_s"] = round(elapsed, 1)
                parsed["_turn"] = turn_idx + 1
                parsed["_depth_enforced"] = True
                raw_text = raw_text_retry
            # else: accept original final as fallback

        # If final → done
        if parsed.get("kind") == "final":
            turns.append(parsed)
            break

        # If not a clarification either, append and break
        if parsed.get("kind") != "clarification":
            turns.append(parsed)
            break

        # Provocateur intercept (active after provocateur_start_at questions)
        if clarification_count >= config.provocateur_start_at:
            intercepted = await intercept_question(
                messages, parsed, clarification_count,
                reimagine_context=reimagine_context,
                mini_reimagine_result=mini_reimagine_result,
                remaining_budget=max(0, config.max_questions - clarification_count),
                max_questions=config.max_questions,
            )
            if intercepted["action"] in ("replace", "inject"):
                parsed["question"] = intercepted["question"]
                parsed["suggested_answers"] = intercepted["suggested_answers"]
                provocateur_interventions += 1
            else:
                provocateur_passes += 1
            parsed["_provocateur_action"] = intercepted["action"]
            parsed["_provocateur_reasoning"] = intercepted["reasoning"]
            if intercepted.get("hypothesis"):
                parsed["_provocateur_hypothesis"] = intercepted["hypothesis"]

        turns.append(parsed)
        clarification_count += 1
        messages.append(AIMessage(content=json.dumps(parsed)))

        # Simulate user answer
        question = parsed.get("question", "")
        suggested = parsed.get("suggested_answers", [])
        answer = await simulate_user_answer(
            eval_llm, question, suggested, scenario["message"],
            seed=hash(scenario["name"] + str(clarification_count)),
        )
        turns[-1]["_user_answer"] = answer
        messages.append(HumanMessage(content=answer))

        # ── Post-turn thinking (variant D): fire AFTER question + answer ──
        if not thinking_first:
            if clarification_count == 3 and mini_reimagine_result is None:
                mini_reimagine_result = await run_mini_reimagine(messages)
                if mini_reimagine_result.get("direction") == "pivot":
                    turns.append({
                        "kind": "direction_signal",
                        "_mini_reimagine": mini_reimagine_result,
                        "_turn": turn_idx + 1,
                    })

            if clarification_count == config.reimagine_after and reimagine_context is None:
                reimagine_result = await run_reimagine(messages)
                reimagine_output = reimagine_result
                reimagine_context = reimagine_result.get("reframe", "")
                if reimagine_context:
                    turns.append({
                        "kind": "insight",
                        "message": f"[thinking:reframe]{reimagine_context}",
                        "_reimagine": reimagine_result,
                        "_elapsed_s": reimagine_result.get("latency_ms", 0) / 1000,
                        "_turn": turn_idx + 1,
                    })

    return ProvocateurConversationResult(
        turns=turns,
        elapsed_s=round(time.time() - t_start, 1),
        config=config,
        scenario_name=scenario["name"],
        reimagine_output=reimagine_output,
        provocateur_interventions=provocateur_interventions,
        provocateur_passes=provocateur_passes,
        variant="provocateur" if not thinking_first else "provocateur_thinking_first",
    )
