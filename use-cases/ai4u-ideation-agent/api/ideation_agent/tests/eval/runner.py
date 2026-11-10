"""Parameterized conversation runner for evaluation tests.

Runs multi-turn conversations directly with the LLM (no LangGraph graph),
simulates user answers via an eval LLM. Parameterized by RunnerConfig to
support ablation testing of individual technologies.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ideation_agent.common import make_llm
from ideation_agent.system_prompt import get_system_prompt
from ideation_agent.skill_catalog import scan_skill_summaries

from .config import RunnerConfig


_USER_SIM_SYSTEM = (
    "You are roleplaying as a business user who submitted a scenario to an AI ideation agent. "
    "The agent is now asking you a clarification question. Respond naturally as if you were "
    "that user — concise (2-3 sentences), realistic, and based on the scenario context below.\n\n"
    "You may draw on one of the suggested replies as inspiration, but rephrase and add "
    "specific details from the scenario context. Do NOT just copy a suggested reply verbatim.\n\n"
    "SCENARIO CONTEXT:\n{scenario}\n\n"
    "Respond ONLY with your answer — no preamble, no quotes."
)

_USER_SIM_SYSTEM_ANCHORED = (
    "You are roleplaying as a business user who submitted a scenario to an AI ideation agent. "
    "The agent is asking you a clarification question.\n\n"
    "Your preferred answer direction is: \"{anchor}\"\n"
    "Flesh this out with 2-3 sentences of realistic detail from the scenario context below, "
    "but keep the same directional choice. Do NOT contradict the anchor direction.\n\n"
    "SCENARIO CONTEXT:\n{scenario}\n\n"
    "Respond ONLY with your answer — no preamble, no quotes."
)

_TWO_TIER_MARKER = "Question strategy (two tiers):"


@dataclass
class ConversationResult:
    """Result of a single conversation run."""

    turns: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    config: Optional[RunnerConfig] = None
    scenario_name: str = ""

    def final_turn(self) -> Optional[Dict[str, Any]]:
        for t in reversed(self.turns):
            if t.get("kind") == "final":
                return t
        return None

    @property
    def num_clarifications(self) -> int:
        return sum(1 for t in self.turns if t.get("kind") == "clarification")


def parse_json(text: str) -> Dict[str, Any]:
    """Parse agent JSON leniently — strip fences, handle duplicates."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if "\n{" in cleaned:
        cleaned = cleaned.split("\n{")[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text, start)
            return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return {"error": "parse_failed", "raw": text[:300]}


def _extract_text_content(content) -> str:
    """Extract text from response content (handles str, list of parts, objects)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts)
    if hasattr(content, "text"):
        return content.text
    return str(content or "")


def _strip_two_tier_block(prompt: str) -> str:
    """Remove the two-tier question strategy block from system prompt."""
    lines = prompt.splitlines()
    result = []
    skip = False
    for line in lines:
        if _TWO_TIER_MARKER in line:
            skip = True
            continue
        if skip:
            if line.strip() == "" or (not line.startswith("-") and not line.startswith(" ")):
                skip = False
                if line.strip():
                    result.append(line)
            continue
        result.append(line)
    return "\n".join(result)


def make_agent_llm(config: RunnerConfig):
    """Create the agent LLM from a RunnerConfig."""
    return make_llm(
        provider=config.model_provider,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=8192,
        json_mode=config.json_mode,
    )


def make_eval_llm(provider: str = "google", model: str = "gemini-2.5-flash"):
    """Create the evaluation LLM (for user simulation and judging)."""
    import os
    provider = os.getenv("AB_EVAL_PROVIDER", provider).lower()
    model = os.getenv("AB_EVAL_MODEL", model)
    return make_llm(provider=provider, model_name=model, temperature=0.3, max_tokens=4096)


async def simulate_user_answer(
    eval_llm, question: str, suggested: list, scenario_context: str,
    *, seed: Optional[int] = None,
) -> str:
    """Simulate a user answer using the eval LLM.

    Args:
        seed: If provided, deterministically pick a suggested answer as the
            direction anchor. This ensures consistent user responses across
            A/B variants for the same scenario + turn.
    """
    if seed is not None and suggested:
        # Deterministic mode: pick a suggested answer as anchor
        idx = seed % len(suggested)
        anchor = suggested[idx]
        system = _USER_SIM_SYSTEM_ANCHORED.format(scenario=scenario_context, anchor=anchor)
    else:
        system = _USER_SIM_SYSTEM.format(scenario=scenario_context)

    suggestions_text = "\n".join(f"- {r}" for r in suggested) if suggested else "- (none provided)"
    user_prompt = (
        f"Agent's question: {question}\n\n"
        f"Suggested replies:\n{suggestions_text}\n\n"
        f"Your answer:"
    )
    response = await eval_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])
    return _extract_text_content(response.content).strip()


async def run_conversation(
    llm,
    eval_llm,
    scenario: Dict[str, str],
    config: RunnerConfig,
    *,
    cse_context: Optional[str] = None,
    perplexity_context: Optional[str] = None,
    perplexity_problem_context: Optional[str] = None,
    linkedin_context: Optional[str] = None,
    skill_summaries: Optional[List[str]] = None,
    system_prompt_fn: Optional[Callable[..., str]] = None,
) -> ConversationResult:
    """Run a multi-turn conversation with the given configuration.

    Args:
        llm: The agent LLM instance.
        eval_llm: The evaluation LLM for simulating user answers.
        scenario: Scenario dict with 'name', 'company', 'message'.
        config: RunnerConfig controlling all evaluation knobs.
        cse_context: Pre-fetched CSE context string (None if CSE disabled or unavailable).
        perplexity_context: Pre-fetched Perplexity context string (None if disabled or unavailable).
        perplexity_problem_context: Pre-fetched problem research context (None if disabled).
        linkedin_context: Pre-fetched LinkedIn context string (None if disabled or unavailable).
        skill_summaries: Pre-loaded skill summaries (loaded automatically if None).
        system_prompt_fn: Optional callable to override the default get_system_prompt.
            Must accept the same signature as get_system_prompt (positional skill_summaries,
            then keyword args for clarification_count, remaining_questions, cse_context, etc.).
    """
    if skill_summaries is None:
        skill_summaries = scan_skill_summaries()

    effective_cse = cse_context if config.cse_enabled else None
    effective_perplexity = perplexity_context if config.perplexity_enabled else None
    if effective_perplexity and config.perplexity_problem_research and perplexity_problem_context:
        effective_perplexity = effective_perplexity + "\n" + perplexity_problem_context
    elif not config.perplexity_problem_research:
        pass  # keep company-only context
    effective_linkedin = linkedin_context if config.linkedin_enabled else None

    turns: List[Dict[str, Any]] = []
    messages = [HumanMessage(content=scenario["message"])]
    clarification_count = 0
    t_start = time.time()

    for turn_num in range(config.max_turns + 1):
        remaining = max(0, config.max_questions - clarification_count)
        prompt_builder = system_prompt_fn or get_system_prompt
        system_prompt = prompt_builder(
            skill_summaries,
            clarification_count=clarification_count,
            remaining_questions=remaining,
            cse_context=effective_cse,
            market_context=effective_perplexity,
            linkedin_context=effective_linkedin,
        )

        if not config.two_tier_strategy:
            system_prompt = _strip_two_tier_block(system_prompt)

        t0 = time.time()
        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
        elapsed = time.time() - t0

        raw_text = _extract_text_content(response.content)
        parsed = parse_json(raw_text)

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
        parsed["_turn"] = turn_num + 1

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
                parsed["_turn"] = turn_num + 1
                parsed["_depth_enforced"] = True
                raw_text = raw_text_retry
            # else: accept original final as fallback (better than erroring)

        turns.append(parsed)

        messages.append(AIMessage(content=raw_text))

        if parsed.get("kind") != "clarification":
            break

        clarification_count += 1
        question = parsed.get("question", "")
        suggested = parsed.get("suggested_answers", [])

        answer = await simulate_user_answer(
            eval_llm, question, suggested, scenario["message"],
            seed=hash(scenario["name"] + str(clarification_count)),
        )
        turns[-1]["_user_answer"] = answer
        messages.append(HumanMessage(content=answer))

    return ConversationResult(
        turns=turns,
        elapsed_s=round(time.time() - t_start, 1),
        config=config,
        scenario_name=scenario["name"],
    )
