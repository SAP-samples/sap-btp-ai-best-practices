"""Head-to-head preference judge for quality evaluation.

Forces a binary winner between two conversations — eliminates ceiling effects
from absolute scoring. Also provides per-dimension scoring for granular insight.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .runner import parse_json, _extract_text_content


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence gate — auto-loss if variant doesn't produce kind="final"
# ═══════════════════════════════════════════════════════════════════════════════


def apply_convergence_gate(
    turns_a: List[Dict[str, Any]],
    turns_b: List[Dict[str, Any]],
) -> Optional[str]:
    """Check if either conversation failed to converge (no kind='final').

    Returns:
        "A" if only A converged (A wins by default),
        "B" if only B converged (B wins by default),
        "BOTH_FAILED" if neither converged,
        None if both converged (proceed to normal judging).
    """
    a_has_final = any(t.get("kind") == "final" for t in turns_a)
    b_has_final = any(t.get("kind") == "final" for t in turns_b)

    if a_has_final and b_has_final:
        return None  # Both converged — proceed to normal judging
    if a_has_final and not b_has_final:
        return "A"
    if b_has_final and not a_has_final:
        return "B"
    return "BOTH_FAILED"


_JUDGE_PREFERENCE_SYSTEM = """You are comparing two AI agent conversations (A and B) that help a business user define an AI use case. Both conversations address the same scenario.

Your job: decide which conversation produced a MORE USEFUL result for someone who needs to build the solution.

Consider:
- Did the agent ask targeted, non-obvious questions that revealed important details?
- Is the final recommendation specific enough to start building from (data sources, workflow, outputs)?
- Did the conversation feel efficient — no wasted questions, no redundant asks?
- Does the output reference the actual company, systems, and constraints discussed?

You MUST pick a winner. Ties are NOT allowed — even if both are good, one is always slightly better.

Return JSON only (no markdown fences):
{"winner": "A" or "B", "confidence": "high" or "medium" or "low", "reasoning": "2-3 sentences explaining WHY the winner is better, referencing specific differences"}"""


_JUDGE_DIMENSIONS_SYSTEM = """You are evaluating two AI agent conversations (A and B) that help a business user define an AI use case. Both address the same scenario.

Score EACH conversation on these 6 dimensions (1-10 scale):

1. **question_depth** — Did the agent's questions go deep enough to uncover non-obvious requirements, hidden constraints, and real-world complexities? Did it ask about data quality, exceptions, failure modes, and system realities? (1=surface-level/obvious questions only, 10=expert probing that revealed things the user hadn't articulated)

2. **creative_challenge** — Did the agent add its own ideas, challenge assumptions, or reframe the problem? Or did it just collect and summarize user answers? (1=pure summarizer that only echoes back what the user said, 10=true ideation partner that shaped the solution with its own creative insight and challenged at least one assumption)

3. **solution_specificity** — Is the final recommendation concrete enough to start building? Does it name data sources, workflows, integration points, and outputs? (1=vague/generic, 10=implementation-ready blueprint)

4. **solution_actionability** — Could a developer or product manager take this output and begin work? Are next steps, constraints, and success criteria clear? (1=needs major clarification, 10=ready to hand off to engineering)

5. **assumption_challenge_quality** — Did the conversation expose a hidden assumption, challenge the user's framing of the problem, or surface an angle the user hadn't considered? (1=no challenge, accepted everything at face value, 10=fundamentally reframed the problem and the solution is richer because of it)

6. **reframe_influence** — Did the conversation produce a solution that goes BEYOND what the user originally described? Does the final recommendation address a deeper or different problem than the user's opening message suggested, or does it just implement exactly what the user asked for in structured form? (1=pure implementation of stated request with no additional framing, 10=fundamentally reframed the problem and produced a solution the user wouldn't have arrived at alone)

Return JSON only (no markdown fences):
{"scores_a": {"question_depth": N, "creative_challenge": N, "solution_specificity": N, "solution_actionability": N, "assumption_challenge_quality": N, "reframe_influence": N}, "scores_b": {"question_depth": N, "creative_challenge": N, "solution_specificity": N, "solution_actionability": N, "assumption_challenge_quality": N, "reframe_influence": N}, "notes": "1-2 sentences on the most notable difference between A and B"}"""


def _format_turns_for_judge(turns: List[Dict[str, Any]]) -> str:
    """Format conversation turns for judge evaluation."""
    lines = []
    for t in turns:
        kind = t.get("kind", "?")
        elapsed = t.get("_elapsed_s", "?")
        if kind == "clarification":
            lines.append(f"  Agent [{elapsed}s]: Q: {t.get('question', '?')}")
            lines.append(f"    Suggestions: {t.get('suggested_answers', [])}")
            if "_user_answer" in t:
                lines.append(f"  User: {t['_user_answer']}")
        elif kind == "final":
            lines.append(f"  Agent [{elapsed}s] FINAL:")
            lines.append(f"    Title: {t.get('title', '?')}")
            lines.append(f"    App Type: {t.get('app_type', '?')}")
            lines.append(f"    Problem: {t.get('problem', '?')}")
            lines.append(f"    Solution: {t.get('solution', '?')}")
        else:
            lines.append(f"  Agent [{elapsed}s]: (kind={kind}) {str(t.get('raw', t.get('message', '?')))[:200]}")
    return "\n".join(lines)


async def judge_preference(
    eval_llm,
    scenario: Dict[str, str],
    turns_a: List[Dict[str, Any]],
    turns_b: List[Dict[str, Any]],
    *,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Head-to-head preference judge: which conversation is better?

    Forces a winner (no ties). Returns winner, confidence, and reasoning.
    """
    conv_a = _format_turns_for_judge(turns_a)
    conv_b = _format_turns_for_judge(turns_b)

    prompt = (
        f"## Scenario\n"
        f"Company: {scenario.get('company', 'Unknown')}\n"
        f"User message: {scenario['message']}\n\n"
        f"## Conversation A\n{conv_a}\n\n"
        f"## Conversation B\n{conv_b}\n\n"
        f"Which conversation is better? Pick a winner."
    )

    for attempt in range(max_retries):
        response = await eval_llm.ainvoke([
            SystemMessage(content=_JUDGE_PREFERENCE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = _extract_text_content(response.content).strip()
        parsed = parse_json(raw)

        winner = parsed.get("winner", "").upper()
        if winner in ("A", "B"):
            return {
                "winner": winner,
                "confidence": parsed.get("confidence", "medium"),
                "reasoning": parsed.get("reasoning", ""),
            }

    return {"winner": "A", "confidence": "low", "reasoning": "judge parse failed", "error": True}


async def judge_dimensions(
    eval_llm,
    scenario: Dict[str, str],
    turns_a: List[Dict[str, Any]],
    turns_b: List[Dict[str, Any]],
    *,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Per-dimension scoring: scores both conversations on 6 dimensions (1-10).

    Returns scores_a, scores_b dicts with keys:
        question_depth, creative_challenge, solution_specificity,
        solution_actionability, assumption_challenge_quality, reframe_influence
    """
    conv_a = _format_turns_for_judge(turns_a)
    conv_b = _format_turns_for_judge(turns_b)

    prompt = (
        f"## Scenario\n"
        f"Company: {scenario.get('company', 'Unknown')}\n"
        f"User message: {scenario['message']}\n\n"
        f"## Conversation A\n{conv_a}\n\n"
        f"## Conversation B\n{conv_b}\n\n"
        f"Score each conversation on the 6 dimensions."
    )

    for attempt in range(max_retries):
        response = await eval_llm.ainvoke([
            SystemMessage(content=_JUDGE_DIMENSIONS_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = _extract_text_content(response.content).strip()
        parsed = parse_json(raw)

        if "scores_a" in parsed and "scores_b" in parsed:
            return parsed

    return {
        "scores_a": {"question_depth": 0, "creative_challenge": 0, "solution_specificity": 0, "solution_actionability": 0, "assumption_challenge_quality": 0, "reframe_influence": 0},
        "scores_b": {"question_depth": 0, "creative_challenge": 0, "solution_specificity": 0, "solution_actionability": 0, "assumption_challenge_quality": 0, "reframe_influence": 0},
        "notes": "judge parse failed",
        "error": True,
    }
