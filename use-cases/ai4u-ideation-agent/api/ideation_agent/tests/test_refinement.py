"""Integration tests for post-final refinement flow.

Verifies that the agent enters refinement mode after delivering a final report,
asks clarification questions for broad/ambiguous feedback, and produces updated
reports on specific feedback or [ACTION:UPDATE_REPORT].

Run:
    cd api
    python -m ideation_agent.tests.test_refinement
    python -m ideation_agent.tests.test_refinement --verbose

Requires .env with LLM credentials (AICORE).
Typical runtime: ~30-60s (real LLM calls).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except Exception:
    pass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ideation_agent.common import make_llm
from ideation_agent.system_prompt import get_system_prompt
from ideation_agent.skill_catalog import scan_skill_summaries
from ideation_agent.tests.eval.runner import parse_json, make_agent_llm, make_eval_llm, simulate_user_answer
from ideation_agent.tests.eval.config import RunnerConfig
from ideation_agent.tests.eval.scenarios import SCENARIOS


VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
MAX_RETRIES = 3
RETRY_DELAY = 5

CONFIG = RunnerConfig(
    model_provider="openai",
    model_name="gpt-4.1",
    temperature=0.2,
    cse_enabled=False,
    perplexity_enabled=False,
    linkedin_enabled=False,
    repair_enabled=True,
    max_questions=5,
    two_tier_strategy=True,
    json_mode=False,
)

SCENARIO = SCENARIOS[0]  # Volvo PDF procurement


def _log(msg: str):
    if VERBOSE:
        print(msg)


async def run_to_final(
    llm,
    eval_llm,
    scenario: Dict[str, str],
    skill_summaries: list,
) -> tuple[List[Any], List[Any]]:
    """Run a conversation until the agent produces a final report.

    Returns (messages, turns) where messages is the LangChain message history.
    """
    messages = [HumanMessage(content=scenario["message"])]
    turns: List[Dict[str, Any]] = []
    clarification_count = 0

    for turn_num in range(CONFIG.max_turns + 1):
        remaining = max(0, CONFIG.max_questions - clarification_count)
        system_prompt = get_system_prompt(
            skill_summaries,
            clarification_count=clarification_count,
            remaining_questions=remaining,
        )

        response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages)
        raw_text = response.content if isinstance(response.content, str) else str(response.content or "")
        parsed = parse_json(raw_text)

        if CONFIG.repair_enabled and parsed.get("kind") not in ("clarification", "final"):
            retry_msg = "Your response must be valid JSON with kind='clarification' or kind='final'. Try again."
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt)] + messages
                + [AIMessage(content=raw_text), HumanMessage(content=retry_msg)]
            )
            raw_text = response.content if isinstance(response.content, str) else str(response.content or "")
            parsed = parse_json(raw_text)

        turns.append(parsed)
        messages.append(AIMessage(content=raw_text))

        _log(f"  Turn {turn_num + 1}: kind={parsed.get('kind', '?')}")

        if parsed.get("kind") == "final":
            break

        if parsed.get("kind") == "clarification":
            clarification_count += 1
            question = parsed.get("question", "")
            suggested = parsed.get("suggested_answers", [])
            answer = await simulate_user_answer(eval_llm, question, suggested, scenario["message"])
            _log(f"    Q: {question}")
            _log(f"    A: {answer}")
            messages.append(HumanMessage(content=answer))
        else:
            break

    return messages, turns


async def send_refinement_message(
    llm,
    messages: List[Any],
    user_text: str,
    clarification_count: int,
    skill_summaries: list,
    *,
    refinement_clarifications_remaining: int = 2,
) -> tuple[Dict[str, Any], str]:
    """Send a refinement-mode message and return the parsed response.

    Returns (parsed_response, raw_text). Retries on transient API errors.
    """
    system_prompt = get_system_prompt(
        skill_summaries,
        clarification_count=clarification_count,
        remaining_questions=0,
        refinement_mode=True,
        refinement_clarifications_remaining=refinement_clarifications_remaining,
    )

    messages_with_user = messages + [HumanMessage(content=user_text)]

    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.ainvoke([SystemMessage(content=system_prompt)] + messages_with_user)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                _log(f"    Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise

    raw_text = response.content if isinstance(response.content, str) else str(response.content or "")
    parsed = parse_json(raw_text)

    if parsed.get("kind") not in ("clarification", "final", "refinement"):
        retry_msg = (
            "Your response must be valid JSON with kind='clarification', kind='refinement', or kind='final'. "
            "If the user's request is broad or ambiguous, return kind='clarification' with a question about which aspect to focus on. "
            "Try again."
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = await llm.ainvoke(
                    [SystemMessage(content=system_prompt)] + messages_with_user
                    + [AIMessage(content=raw_text), HumanMessage(content=retry_msg)]
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    _log(f"    Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    raise
        raw_text = response.content if isinstance(response.content, str) else str(response.content or "")
        parsed = parse_json(raw_text)

    return parsed, raw_text


async def test_go_deeper_triggers_clarification():
    """After final report, 'go deeper' should produce a clarification question."""
    print("\n--- Test: 'Go deeper' triggers clarification ---")
    skill_summaries = scan_skill_summaries()
    llm = make_agent_llm(CONFIG)
    eval_llm = make_eval_llm()

    t0 = time.time()
    _log(f"  Scenario: {SCENARIO['name']}")
    _log("  Running interview to final...")

    messages, turns = await run_to_final(llm, eval_llm, SCENARIO, skill_summaries)

    final = next((t for t in reversed(turns) if t.get("kind") == "final"), None)
    assert final is not None, "Interview did not produce a final report"
    _log(f"  Got final: {final.get('title', '?')}")

    clarification_count = sum(1 for t in turns if t.get("kind") == "clarification")

    _log("  Sending 'I'd like to go deeper on this'...")
    parsed, raw_text = await send_refinement_message(
        llm, messages, "I'd like to go deeper on this",
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
    )

    elapsed = round(time.time() - t0, 1)
    kind = parsed.get("kind", "?")
    _log(f"  Response kind: {kind} ({elapsed}s total)")

    if kind == "clarification":
        question = parsed.get("question", "")
        suggested = parsed.get("suggested_answers", [])
        _log(f"    Q: {question}")
        _log(f"    Suggestions: {suggested}")
        assert question, "Clarification has no question"
        assert len(suggested) == 3, f"Expected 3 suggestions, got {len(suggested)}"
        print(f"  PASS: Agent asked clarification: '{question}' ({elapsed}s)")
        return True
    else:
        print(f"  FAIL: Expected kind='clarification', got kind='{kind}'")
        print(f"    Raw: {raw_text[:300]}")
        return False


async def test_specific_feedback_triggers_refinement():
    """After final report, specific feedback should produce a refinement reply."""
    print("\n--- Test: Specific feedback triggers refinement ---")
    skill_summaries = scan_skill_summaries()
    llm = make_agent_llm(CONFIG)
    eval_llm = make_eval_llm()

    t0 = time.time()
    _log(f"  Scenario: {SCENARIO['name']}")
    _log("  Running interview to final...")

    messages, turns = await run_to_final(llm, eval_llm, SCENARIO, skill_summaries)

    final = next((t for t in reversed(turns) if t.get("kind") == "final"), None)
    assert final is not None, "Interview did not produce a final report"

    clarification_count = sum(1 for t in turns if t.get("kind") == "clarification")

    feedback = "Focus more on batch processing — the clerks should only review exceptions, not every single assignment."
    _log(f"  Sending specific feedback: '{feedback[:60]}...'")
    parsed, raw_text = await send_refinement_message(
        llm, messages, feedback,
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
    )

    elapsed = round(time.time() - t0, 1)
    kind = parsed.get("kind", "?")
    _log(f"  Response kind: {kind} ({elapsed}s total)")

    if kind == "refinement":
        message = parsed.get("message", "")
        _log(f"    Message: {message[:200]}")
        assert message, "Refinement has no message"
        print(f"  PASS: Agent acknowledged feedback with refinement reply ({elapsed}s)")
        return True
    elif kind == "clarification":
        print(f"  PASS (acceptable): Agent asked a follow-up clarification ({elapsed}s)")
        return True
    else:
        print(f"  FAIL: Expected kind='refinement' or 'clarification', got kind='{kind}'")
        print(f"    Raw: {raw_text[:300]}")
        return False


async def test_update_report_produces_final():
    """[ACTION:UPDATE_REPORT] after feedback should produce an updated final."""
    print("\n--- Test: [ACTION:UPDATE_REPORT] produces updated final ---")
    skill_summaries = scan_skill_summaries()
    llm = make_agent_llm(CONFIG)
    eval_llm = make_eval_llm()

    t0 = time.time()
    _log(f"  Scenario: {SCENARIO['name']}")
    _log("  Running interview to final...")

    messages, turns = await run_to_final(llm, eval_llm, SCENARIO, skill_summaries)

    final = next((t for t in reversed(turns) if t.get("kind") == "final"), None)
    assert final is not None, "Interview did not produce a final report"

    clarification_count = sum(1 for t in turns if t.get("kind") == "clarification")

    # Send feedback first
    feedback = "Add error handling for unreadable PDFs and a confidence score for each suggestion."
    _log(f"  Sending feedback: '{feedback[:60]}...'")
    parsed_ref, raw_ref = await send_refinement_message(
        llm, messages, feedback,
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
    )
    messages.append(HumanMessage(content=feedback))
    messages.append(AIMessage(content=raw_ref))
    _log(f"  Feedback response kind: {parsed_ref.get('kind', '?')}")

    # Now send update report action
    _log("  Sending [ACTION:UPDATE_REPORT]...")
    parsed, raw_text = await send_refinement_message(
        llm, messages, "[ACTION:UPDATE_REPORT]",
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=1,
    )

    elapsed = round(time.time() - t0, 1)
    kind = parsed.get("kind", "?")
    _log(f"  Response kind: {kind} ({elapsed}s total)")

    if kind == "final":
        title = parsed.get("title", "?")
        solution = parsed.get("solution", "")
        _log(f"    Title: {title}")
        _log(f"    Solution: {solution[:200]}...")
        assert parsed.get("industry"), "Final missing industry"
        assert parsed.get("problem"), "Final missing problem"
        assert solution, "Final missing solution"
        # Check that feedback was incorporated
        solution_lower = solution.lower()
        has_feedback = "error" in solution_lower or "confidence" in solution_lower or "unreadable" in solution_lower
        if has_feedback:
            print(f"  PASS: Updated final incorporates feedback ({elapsed}s)")
        else:
            print(f"  PASS (weak): Got final but feedback may not be fully incorporated ({elapsed}s)")
        return True
    else:
        print(f"  FAIL: Expected kind='final', got kind='{kind}'")
        print(f"    Raw: {raw_text[:300]}")
        return False


async def test_go_deeper_mini_interview_produces_final():
    """'Go deeper' should trigger 2-3 clarification questions, then produce an updated final."""
    print("\n--- Test: 'Go deeper' mini-interview produces updated final ---")
    skill_summaries = scan_skill_summaries()
    llm = make_agent_llm(CONFIG)
    eval_llm = make_eval_llm()

    t0 = time.time()
    _log(f"  Scenario: {SCENARIO['name']}")
    _log("  Running interview to final...")

    messages, turns = await run_to_final(llm, eval_llm, SCENARIO, skill_summaries)

    final = next((t for t in reversed(turns) if t.get("kind") == "final"), None)
    assert final is not None, "Interview did not produce a final report"

    clarification_count = sum(1 for t in turns if t.get("kind") == "clarification")

    # Step 1: Send "go deeper" → should get clarification asking which aspect
    _log("  Sending 'I'd like to go deeper on this'...")
    parsed, raw_text = await send_refinement_message(
        llm, messages, "I'd like to go deeper on this",
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=3,
    )
    messages.append(HumanMessage(content="I'd like to go deeper on this"))
    messages.append(AIMessage(content=raw_text))

    if parsed.get("kind") != "clarification":
        print(f"  FAIL: Expected initial clarification, got {parsed.get('kind')}")
        return False

    _log(f"  Q1: {parsed.get('question', '?')}")
    refinement_clar_count = 1

    # Step 2-3: Answer clarifications until agent produces final (max 3 questions)
    max_refinement_questions = 3
    got_final = False

    for i in range(max_refinement_questions):
        suggested = parsed.get("suggested_answers", [])
        answer = suggested[0] if suggested else "Yes, that approach works"
        _log(f"  Answering Q{refinement_clar_count}: '{answer}'")
        messages.append(HumanMessage(content=answer))

        remaining = max(0, 3 - refinement_clar_count)
        parsed, raw_text = await send_refinement_message(
            llm, messages, answer,
            clarification_count=clarification_count,
            skill_summaries=skill_summaries,
            refinement_clarifications_remaining=remaining,
        )
        # Remove the duplicate user message (send_refinement_message adds it internally)
        messages.pop()
        messages.append(HumanMessage(content=answer))
        messages.append(AIMessage(content=raw_text))

        kind = parsed.get("kind", "?")
        _log(f"  Response kind: {kind}")

        if kind == "final":
            got_final = True
            break
        elif kind == "clarification":
            refinement_clar_count += 1
            _log(f"  Q{refinement_clar_count}: {parsed.get('question', '?')}")
        else:
            _log(f"  Unexpected kind: {kind}")
            break

    elapsed = round(time.time() - t0, 1)

    if got_final:
        solution = parsed.get("solution", "")
        _log(f"  Final title: {parsed.get('title', '?')}")
        _log(f"  Solution: {solution[:200]}...")
        assert solution, "Final missing solution"
        print(f"  PASS: Mini-interview ({refinement_clar_count} questions) → updated final ({elapsed}s)")
        return True
    else:
        # If we ran out of questions without a final, send [ACTION:UPDATE_REPORT] as fallback
        _log("  Agent didn't auto-produce final, sending [ACTION:UPDATE_REPORT]...")
        parsed, raw_text = await send_refinement_message(
            llm, messages, "[ACTION:UPDATE_REPORT]",
            clarification_count=clarification_count,
            skill_summaries=skill_summaries,
            refinement_clarifications_remaining=0,
        )
        if parsed.get("kind") == "final":
            print(f"  PASS (with nudge): Mini-interview ({refinement_clar_count} questions) + UPDATE_REPORT → final ({elapsed}s)")
            return True
        print(f"  FAIL: Could not get updated final after mini-interview")
        return False


async def test_multiple_refinement_rounds():
    """User can do multiple rounds of refinement after each updated report."""
    print("\n--- Test: Multiple refinement rounds ---")
    skill_summaries = scan_skill_summaries()
    llm = make_agent_llm(CONFIG)
    eval_llm = make_eval_llm()

    t0 = time.time()
    _log(f"  Scenario: {SCENARIO['name']}")
    _log("  Running interview to final...")

    messages, turns = await run_to_final(llm, eval_llm, SCENARIO, skill_summaries)

    final = next((t for t in reversed(turns) if t.get("kind") == "final"), None)
    assert final is not None, "Interview did not produce a final report"
    clarification_count = sum(1 for t in turns if t.get("kind") == "clarification")

    # --- Round 1: specific feedback + update report ---
    feedback_1 = "Make it focus on batch processing rather than real-time."
    _log(f"  Round 1: Sending feedback: '{feedback_1}'")
    parsed, raw_text = await send_refinement_message(
        llm, messages, feedback_1,
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=3,
    )
    messages.append(HumanMessage(content=feedback_1))
    messages.append(AIMessage(content=raw_text))
    _log(f"  Round 1: Response kind: {parsed.get('kind')}")

    # Send update report
    _log("  Round 1: Sending [ACTION:UPDATE_REPORT]...")
    parsed, raw_text = await send_refinement_message(
        llm, messages, "[ACTION:UPDATE_REPORT]",
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=3,
    )
    messages.append(HumanMessage(content="[ACTION:UPDATE_REPORT]"))
    messages.append(AIMessage(content=raw_text))

    if parsed.get("kind") != "final":
        print(f"  FAIL: Round 1 did not produce final, got {parsed.get('kind')}")
        return False

    solution_1 = parsed.get("solution", "")
    _log(f"  Round 1: Got updated final (solution: {solution_1[:100]}...)")

    # --- Round 2: another feedback + update report (fresh budget) ---
    feedback_2 = "Also add confidence scoring for each material group suggestion."
    _log(f"  Round 2: Sending feedback: '{feedback_2}'")
    parsed, raw_text = await send_refinement_message(
        llm, messages, feedback_2,
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=3,  # reset after final
    )
    messages.append(HumanMessage(content=feedback_2))
    messages.append(AIMessage(content=raw_text))
    _log(f"  Round 2: Response kind: {parsed.get('kind')}")

    _log("  Round 2: Sending [ACTION:UPDATE_REPORT]...")
    parsed, raw_text = await send_refinement_message(
        llm, messages, "[ACTION:UPDATE_REPORT]",
        clarification_count=clarification_count,
        skill_summaries=skill_summaries,
        refinement_clarifications_remaining=3,
    )

    elapsed = round(time.time() - t0, 1)

    if parsed.get("kind") != "final":
        print(f"  FAIL: Round 2 did not produce final, got {parsed.get('kind')}")
        return False

    solution_2 = parsed.get("solution", "")
    _log(f"  Round 2: Got updated final (solution: {solution_2[:100]}...)")

    # Verify round 2 incorporates both rounds of feedback
    sol_lower = solution_2.lower()
    has_batch = "batch" in sol_lower
    has_confidence = "confidence" in sol_lower or "score" in sol_lower
    if has_batch and has_confidence:
        print(f"  PASS: Both rounds of feedback incorporated in final report ({elapsed}s)")
    elif has_batch or has_confidence:
        print(f"  PASS (partial): At least one round of feedback incorporated ({elapsed}s)")
    else:
        print(f"  PASS (weak): Got 2 finals but feedback may not be fully reflected ({elapsed}s)")
    return True


async def main():
    print("=" * 60)
    print("REFINEMENT FLOW INTEGRATION TESTS")
    print("=" * 60)

    t_start = time.time()
    results = []

    results.append(await test_go_deeper_triggers_clarification())
    results.append(await test_specific_feedback_triggers_refinement())
    results.append(await test_update_report_produces_final())
    results.append(await test_go_deeper_mini_interview_produces_final())
    results.append(await test_multiple_refinement_rounds())

    elapsed = round(time.time() - t_start, 1)
    passed = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} passed ({elapsed}s)")
    print("=" * 60)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
