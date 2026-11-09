"""Four-way A/B test: Old Baseline vs New Baseline vs Reimagine vs Provocateur+Reimagine.

Compares four conversation variants:
- A (old baseline): Production prompt (creative partner, 4-7 questions)
- B (new baseline): Process Architect persona, no provocateur/reimagine
- C (reimagine-only): Process Architect + reimagine at Q5
- D (provocateur + reimagine): Full pipeline (mini-reimagine Q3 + provocateur Q4+ + reimagine Q5)

What we learn:
- B vs A → How much does the Process Architect persona alone improve?
- C vs B → Does the reimagine step change the final output?
- D vs B → Does the provocateur add value beyond the persona?
- D vs A → Full improvement vs current production

Run:
    cd api
    python -m ideation_agent.tests.test_provocateur_4way              # 1 run
    python -m ideation_agent.tests.test_provocateur_4way --runs 3     # statistical
    python -m ideation_agent.tests.test_provocateur_4way -v           # verbose
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
except Exception:
    pass

from ideation_agent.skill_catalog import scan_skill_summaries

from ideation_agent.tests.eval.config import RunnerConfig
from ideation_agent.tests.eval.scenarios import ALL_SCENARIOS
from ideation_agent.tests.eval.runner import (
    make_agent_llm,
    make_eval_llm,
    run_conversation,
)
from ideation_agent.tests.eval.provocateur_runner import (
    run_reimagine_conversation,
    run_provocateur_conversation,
)
from ideation_agent.tests.eval.judge import (
    apply_convergence_gate,
    judge_preference,
    judge_dimensions,
)
from ideation_agent.tests.eval.prompt_versions import get_system_prompt_baseline
from ideation_agent.tests.eval_output import save_batch_result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLD_BASELINE_CONFIG = RunnerConfig(
    model_provider="openai",
    model_name="gpt-5.4",
    temperature=0.5,
    cse_enabled=False,
    perplexity_enabled=False,
    perplexity_problem_research=False,
    linkedin_enabled=False,
    repair_enabled=True,
    max_questions=7,
    min_questions=3,
    two_tier_strategy=True,
    json_mode=False,
)

NEW_BASELINE_CONFIG = RunnerConfig(
    model_provider="openai",
    model_name="gpt-5.4",
    temperature=0.5,
    cse_enabled=False,
    perplexity_enabled=False,
    perplexity_problem_research=False,
    linkedin_enabled=False,
    repair_enabled=True,
    max_questions=9,
    min_questions=5,
    two_tier_strategy=True,
    json_mode=False,
)

REIMAGINE_CONFIG = RunnerConfig(
    model_provider="openai",
    model_name="gpt-5.4",
    temperature=0.5,
    cse_enabled=False,
    perplexity_enabled=False,
    perplexity_problem_research=False,
    linkedin_enabled=False,
    repair_enabled=True,
    max_questions=9,
    min_questions=5,
    two_tier_strategy=True,
    json_mode=False,
    provocateur_enabled=False,
    reimagine_enabled=True,
    reimagine_after=5,
)

PROVOCATEUR_CONFIG = RunnerConfig(
    model_provider="openai",
    model_name="gpt-5.4",
    temperature=0.5,
    cse_enabled=False,
    perplexity_enabled=False,
    perplexity_problem_research=False,
    linkedin_enabled=False,
    repair_enabled=True,
    max_questions=9,
    min_questions=5,
    two_tier_strategy=True,
    json_mode=False,
    provocateur_enabled=True,
    reimagine_enabled=True,
    provocateur_start_at=3,
    reimagine_after=5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_report_lines: List[str] = []
_CONCURRENCY = asyncio.Semaphore(2)  # 2 scenarios at a time (5 variants have many LLM calls)


def _log(line: str = ""):
    print(line)
    _report_lines.append(line)


def _format_conversation(turns: List[Dict[str, Any]]) -> str:
    lines = []
    for t in turns:
        kind = t.get("kind", "?")
        if kind == "clarification":
            prov = ""
            if "_provocateur_action" in t:
                prov = f" [provocateur:{t['_provocateur_action']}]"
            lines.append(f"    Q: {t.get('question', '?')}{prov}")
            lines.append(f"       Suggestions: {t.get('suggested_answers', [])}")
            if "_user_answer" in t:
                lines.append(f"    A: {t['_user_answer']}")
        elif kind == "insight":
            lines.append(f"    REIMAGINE: {t.get('message', '?')[:200]}")
        elif kind == "direction_signal":
            mini = t.get("_mini_reimagine", {})
            lines.append(f"    DIRECTION: {mini.get('direction', '?')} — {mini.get('hint', '')[:100]}")
        elif kind == "final":
            lines.append(f"    FINAL:")
            lines.append(f"      Title: {t.get('title', '?')}")
            lines.append(f"      Problem: {t.get('problem', '?')[:200]}")
            solution = t.get("solution", "?")
            for sol_line in solution.split("\n")[:8]:
                lines.append(f"      {sol_line}")
        else:
            lines.append(f"    ({kind}): {str(t.get('raw', t.get('message', '?')))[:200]}")
    return "\n".join(lines)


async def _judge_pair(eval_llm, scenario, turns_x, turns_y):
    """Judge a pair, returning (winner_label, judgment_dict). X=first, Y=second."""
    gate = apply_convergence_gate(turns_x, turns_y)
    if gate == "A":
        return "X", {"winner": "A", "confidence": "high", "reasoning": "X converged, Y did not"}
    elif gate == "B":
        return "Y", {"winner": "B", "confidence": "high", "reasoning": "Y converged, X did not"}
    elif gate == "BOTH_FAILED":
        return "both_failed", {"winner": "NONE", "confidence": "high", "reasoning": "Neither converged"}

    swapped = random.random() < 0.5
    first, second = (turns_y, turns_x) if swapped else (turns_x, turns_y)
    judgment = await judge_preference(eval_llm, scenario, first, second)
    if swapped:
        winner = "X" if judgment["winner"] == "B" else "Y"
    else:
        winner = "X" if judgment["winner"] == "A" else "Y"
    return winner, judgment


# ---------------------------------------------------------------------------
# Core 4-way comparison
# ---------------------------------------------------------------------------

async def _run_four_way(scenario, *, eval_llm, skill_summaries, max_retries=5):
    for attempt in range(max_retries):
        try:
            llm_a = make_agent_llm(OLD_BASELINE_CONFIG)
            llm_b = make_agent_llm(NEW_BASELINE_CONFIG)
            llm_c = make_agent_llm(REIMAGINE_CONFIG)
            llm_d = make_agent_llm(PROVOCATEUR_CONFIG)
            llm_e = make_agent_llm(PROVOCATEUR_CONFIG)

            result_a, result_b, result_c, result_d, result_e = await asyncio.gather(
                run_conversation(llm_a, eval_llm, scenario, OLD_BASELINE_CONFIG,
                                 skill_summaries=skill_summaries,
                                 system_prompt_fn=get_system_prompt_baseline),
                run_conversation(llm_b, eval_llm, scenario, NEW_BASELINE_CONFIG,
                                 skill_summaries=skill_summaries),
                run_reimagine_conversation(llm_c, eval_llm, scenario, REIMAGINE_CONFIG,
                                           skill_summaries=skill_summaries),
                run_provocateur_conversation(llm_d, eval_llm, scenario, PROVOCATEUR_CONFIG,
                                             skill_summaries=skill_summaries,
                                             thinking_first=False),
                run_provocateur_conversation(llm_e, eval_llm, scenario, PROVOCATEUR_CONFIG,
                                             skill_summaries=skill_summaries,
                                             thinking_first=True),
            )

            # Judge pairs
            w_ba, j_ba = await _judge_pair(eval_llm, scenario, result_b.turns, result_a.turns)
            w_cb, j_cb = await _judge_pair(eval_llm, scenario, result_c.turns, result_b.turns)
            w_db, j_db = await _judge_pair(eval_llm, scenario, result_d.turns, result_b.turns)
            w_da, j_da = await _judge_pair(eval_llm, scenario, result_d.turns, result_a.turns)
            w_ed, j_ed = await _judge_pair(eval_llm, scenario, result_e.turns, result_d.turns)
            w_ea, j_ea = await _judge_pair(eval_llm, scenario, result_e.turns, result_a.turns)

            # Dimension scores: E vs A (most informative for the new approach)
            gate_ea = apply_convergence_gate(result_e.turns, result_a.turns)
            dimensions = None
            if gate_ea is None:
                swapped = random.random() < 0.5
                first, second = (result_a.turns, result_e.turns) if swapped else (result_e.turns, result_a.turns)
                dimensions = await judge_dimensions(eval_llm, scenario, first, second)
                if swapped:
                    scores_old = dimensions.get("scores_a", {})
                    scores_prov = dimensions.get("scores_b", {})
                else:
                    scores_prov = dimensions.get("scores_a", {})
                    scores_old = dimensions.get("scores_b", {})
            else:
                scores_old = {}
                scores_prov = {}

            # Convergence
            a_conv = any(t.get("kind") == "final" for t in result_a.turns)
            b_conv = any(t.get("kind") == "final" for t in result_b.turns)
            c_conv = any(t.get("kind") == "final" for t in result_c.turns)
            d_conv = any(t.get("kind") == "final" for t in result_d.turns)
            e_conv = any(t.get("kind") == "final" for t in result_e.turns)

            # Provocateur stats (from E)
            prov_interventions = getattr(result_e, "provocateur_interventions", 0)
            prov_passes = getattr(result_e, "provocateur_passes", 0)

            return {
                "scenario": scenario["name"],
                "winner_b_vs_a": "new_baseline" if w_ba == "X" else "old_baseline" if w_ba == "Y" else w_ba,
                "winner_c_vs_b": "reimagine" if w_cb == "X" else "new_baseline" if w_cb == "Y" else w_cb,
                "winner_d_vs_b": "provocateur" if w_db == "X" else "new_baseline" if w_db == "Y" else w_db,
                "winner_d_vs_a": "provocateur" if w_da == "X" else "old_baseline" if w_da == "Y" else w_da,
                "winner_e_vs_d": "thinking_first" if w_ed == "X" else "thinking_after" if w_ed == "Y" else w_ed,
                "winner_e_vs_a": "thinking_first" if w_ea == "X" else "old_baseline" if w_ea == "Y" else w_ea,
                "reasoning_d_vs_a": j_da.get("reasoning", ""),
                "reasoning_e_vs_d": j_ed.get("reasoning", ""),
                "reasoning_e_vs_a": j_ea.get("reasoning", ""),
                "confidence_e_vs_a": j_ea.get("confidence", ""),
                "converged": {"old": a_conv, "new": b_conv, "reimagine": c_conv, "provocateur": d_conv, "thinking_first": e_conv},
                "questions": {
                    "old": result_a.num_clarifications,
                    "new": result_b.num_clarifications,
                    "reimagine": result_c.num_clarifications,
                    "provocateur": result_d.num_clarifications,
                    "thinking_first": result_e.num_clarifications,
                },
                "provocateur_interventions": prov_interventions,
                "provocateur_passes": prov_passes,
                "scores_old": scores_old,
                "scores_thinking_first": scores_prov,
                "dimension_notes": dimensions.get("notes", "") if dimensions else "",
                "reimagine_e_output": getattr(result_e, "reimagine_output", None),
                "conversations": {
                    "old": result_a.turns,
                    "new": result_b.turns,
                    "reimagine": result_c.turns,
                    "provocateur": result_d.turns,
                    "thinking_first": result_e.turns,
                },
            }
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "quota" in err_str
            if attempt < max_retries - 1:
                backoff = 15 * (attempt + 1) if is_rate_limit else 3
                print(f"   ⏳ Retry ({type(e).__name__}: {str(e)[:60]}), attempt {attempt + 2}...")
                await asyncio.sleep(backoff)
                continue
            raise


async def _guarded_four_way(*args, **kwargs):
    async with _CONCURRENCY:
        return await _run_four_way(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def _parse_args():
    runs = 1
    verbose = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--runs="):
            runs = int(arg.split("=", 1)[1])
        elif arg == "--runs" and i + 1 < len(args):
            i += 1
            runs = int(args[i])
        elif arg in ("--verbose", "-v"):
            verbose = True
        i += 1
    return runs, verbose


async def main():
    num_runs, verbose = _parse_args()
    wall_t0 = time.time()

    eval_llm = make_eval_llm()
    skill_summaries = scan_skill_summaries()

    _log(f"{'═' * 70}")
    _log(f" PROVOCATEUR 4-WAY TEST — {time.strftime('%Y-%m-%d')} | runs={num_runs}")
    _log(f"{'═' * 70}")
    _log(f" A: Old Baseline (creative partner, max 7 questions)")
    _log(f" B: New Baseline (Process Architect persona, max 9 questions)")
    _log(f" C: Reimagine only (Process Architect + reimagine at Q5)")
    _log(f" D: Provocateur + Reimagine (full pipeline)")
    _log(f" No enrichment (CSE/Perplexity/LinkedIn disabled)\n")

    tasks = []
    for run_idx in range(num_runs):
        for scenario in ALL_SCENARIOS:
            tasks.append((run_idx, scenario, asyncio.ensure_future(
                _guarded_four_way(scenario, eval_llm=eval_llm, skill_summaries=skill_summaries)
            )))

    wins = {
        "b_vs_a": {"new_baseline": 0, "old_baseline": 0, "both_failed": 0},
        "c_vs_b": {"reimagine": 0, "new_baseline": 0, "both_failed": 0},
        "d_vs_b": {"provocateur": 0, "new_baseline": 0, "both_failed": 0},
        "d_vs_a": {"provocateur": 0, "old_baseline": 0, "both_failed": 0},
        "e_vs_d": {"thinking_first": 0, "thinking_after": 0, "both_failed": 0},
        "e_vs_a": {"thinking_first": 0, "old_baseline": 0, "both_failed": 0},
    }
    details: List[Dict[str, Any]] = []

    for run_idx, scenario, fut in tasks:
        try:
            result = await fut
        except Exception as e:
            _log(f"   ⚠ {scenario['name'][:45]:45s} error: {str(e)[:60]}")
            continue

        details.append(result)
        wins["b_vs_a"][result["winner_b_vs_a"]] = wins["b_vs_a"].get(result["winner_b_vs_a"], 0) + 1
        wins["c_vs_b"][result["winner_c_vs_b"]] = wins["c_vs_b"].get(result["winner_c_vs_b"], 0) + 1
        wins["d_vs_b"][result["winner_d_vs_b"]] = wins["d_vs_b"].get(result["winner_d_vs_b"], 0) + 1
        wins["d_vs_a"][result["winner_d_vs_a"]] = wins["d_vs_a"].get(result["winner_d_vs_a"], 0) + 1
        wins["e_vs_d"][result["winner_e_vs_d"]] = wins["e_vs_d"].get(result["winner_e_vs_d"], 0) + 1
        wins["e_vs_a"][result["winner_e_vs_a"]] = wins["e_vs_a"].get(result["winner_e_vs_a"], 0) + 1

    n = max(len(details), 1)

    # ─── B vs A ───────────────────────────────────────────────────────────
    total = wins["b_vs_a"]["new_baseline"] + wins["b_vs_a"]["old_baseline"]
    rate = wins["b_vs_a"]["new_baseline"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" B vs A — Does the Process Architect persona alone improve things?")
    _log(f" {'─' * 66}")
    for d in details:
        icon = "✓" if d["winner_b_vs_a"] == "new_baseline" else "✗"
        _log(f"   {icon} {d['scenario'][:45]:45s} [{d.get('confidence_d_vs_a','')}]")
    _log(f"\n   New Baseline wins: {wins['b_vs_a']['new_baseline']}/{total} ({rate:.0%})")

    # ─── D vs A (most important) ──────────────────────────────────────────
    total = wins["d_vs_a"]["provocateur"] + wins["d_vs_a"]["old_baseline"]
    rate = wins["d_vs_a"]["provocateur"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" D vs A — Does the full pipeline beat current production?")
    _log(f" {'─' * 66}")
    for d in details:
        icon = "✓" if d["winner_d_vs_a"] == "provocateur" else "✗"
        _log(f"   {icon} {d['scenario'][:45]:45s} [{d.get('confidence_d_vs_a','')}] {d.get('reasoning_d_vs_a','')[:55]}")
    _log(f"\n   Provocateur wins: {wins['d_vs_a']['provocateur']}/{total} ({rate:.0%})")

    # ─── D vs B ───────────────────────────────────────────────────────────
    total = wins["d_vs_b"]["provocateur"] + wins["d_vs_b"]["new_baseline"]
    rate = wins["d_vs_b"]["provocateur"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" D vs B — Does provocateur add value beyond the persona?")
    _log(f" {'─' * 66}")
    _log(f"   Provocateur wins: {wins['d_vs_b']['provocateur']}/{total} ({rate:.0%})")

    # ─── C vs B ───────────────────────────────────────────────────────────
    total = wins["c_vs_b"]["reimagine"] + wins["c_vs_b"]["new_baseline"]
    rate = wins["c_vs_b"]["reimagine"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" C vs B — Does reimagine alone change the output?")
    _log(f" {'─' * 66}")
    _log(f"   Reimagine wins: {wins['c_vs_b']['reimagine']}/{total} ({rate:.0%})")

    # ─── E vs D (key comparison: thinking-first vs thinking-after) ────────
    total = wins["e_vs_d"]["thinking_first"] + wins["e_vs_d"]["thinking_after"]
    rate = wins["e_vs_d"]["thinking_first"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" E vs D — Does thinking-first beat thinking-after?")
    _log(f" {'─' * 66}")
    for d in details:
        icon = "✓" if d.get("winner_e_vs_d") == "thinking_first" else "✗"
        _log(f"   {icon} {d['scenario'][:45]:45s} {d.get('reasoning_e_vs_d','')[:55]}")
    _log(f"\n   Thinking-first wins: {wins['e_vs_d']['thinking_first']}/{total} ({rate:.0%})")

    # ─── E vs A ───────────────────────────────────────────────────────────
    total = wins["e_vs_a"]["thinking_first"] + wins["e_vs_a"]["old_baseline"]
    rate = wins["e_vs_a"]["thinking_first"] / total if total > 0 else 0
    _log(f"\n{'═' * 70}")
    _log(f" E vs A — Does thinking-first beat old production?")
    _log(f" {'─' * 66}")
    for d in details:
        icon = "✓" if d.get("winner_e_vs_a") == "thinking_first" else "✗"
        _log(f"   {icon} {d['scenario'][:45]:45s} [{d.get('confidence_e_vs_a','')}] {d.get('reasoning_e_vs_a','')[:55]}")
    _log(f"\n   Thinking-first wins: {wins['e_vs_a']['thinking_first']}/{total} ({rate:.0%})")

    # ─── Convergence ──────────────────────────────────────────────────────
    conv = {"old": 0, "new": 0, "reimagine": 0, "provocateur": 0, "thinking_first": 0}
    for d in details:
        for k in conv:
            if d["converged"].get(k):
                conv[k] += 1
    _log(f"\n{'═' * 70}")
    _log(f" Convergence rates")
    _log(f" {'─' * 66}")
    _log(f"   Old: {conv['old']}/{n}  New: {conv['new']}/{n}  Reimagine: {conv['reimagine']}/{n}  Provocateur: {conv['provocateur']}/{n}")

    # ─── Provocateur stats ────────────────────────────────────────────────
    total_interv = sum(d.get("provocateur_interventions", 0) for d in details)
    total_passes = sum(d.get("provocateur_passes", 0) for d in details)
    total_dec = total_interv + total_passes
    _log(f"\n{'═' * 70}")
    _log(f" Provocateur stats")
    _log(f" {'─' * 66}")
    _log(f"   Decisions: {total_dec} | Replaced: {total_interv} ({total_interv/total_dec:.0%} if total_dec else 0) | Passed: {total_passes}")

    # ─── Dimension scores D vs A ──────────────────────────────────────────
    dims = ["question_depth", "creative_challenge", "solution_specificity",
            "solution_actionability", "assumption_challenge_quality", "reframe_influence"]
    dim_labels = ["Question Depth", "Creative Challenge", "Solution Specificity",
                  "Solution Actionability", "Assumption Challenge", "Reframe Influence"]
    old_totals = {d: 0.0 for d in dims}
    prov_totals = {d: 0.0 for d in dims}
    dim_count = 0

    for d in details:
        if d.get("scores_old") and d.get("scores_thinking_first"):
            dim_count += 1
            for dim in dims:
                old_totals[dim] += d["scores_old"].get(dim, 0)
                prov_totals[dim] += d["scores_thinking_first"].get(dim, 0)

    _log(f"\n{'═' * 70}")
    _log(f" Dimension Scores — E (Thinking-First) vs A (Old Baseline)")
    _log(f" {'─' * 66}")
    if dim_count > 0:
        _log(f"   {'Dimension':<25s} {'Old':>8s} {'ThinkFirst':>12s} {'Δ':>8s}")
        _log(f"   {'─' * 55}")
        for dim, label in zip(dims, dim_labels):
            old_avg = old_totals[dim] / dim_count
            prov_avg = prov_totals[dim] / dim_count
            delta = prov_avg - old_avg
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            _log(f"   {label:<25s} {old_avg:>6.1f}/10 {prov_avg:>10.1f}/10  {arrow}{delta:>+5.1f}")
    _log("")

    # ─── Avg questions ────────────────────────────────────────────────────
    for variant in ("old", "new", "reimagine", "provocateur", "thinking_first"):
        avg = sum(d["questions"].get(variant, 0) for d in details) / n
        _log(f"   Avg questions {variant}: {avg:.1f}")
    _log("")

    wall_elapsed = round(time.time() - wall_t0, 1)
    _log(f" Total time: {wall_elapsed}s")
    _log(f"{'═' * 70}")

    # Verbose output
    if verbose:
        for d in details:
            _log(f"\n{'─' * 70}")
            _log(f" Scenario: {d['scenario']}")
            for variant in ("old", "new", "reimagine", "provocateur", "thinking_first"):
                conv_status = d["converged"].get(variant, False)
                q_count = d["questions"].get(variant, 0)
                _log(f"\n   {variant.upper()} ({q_count} questions, converged={conv_status}):")
                _log(_format_conversation(d["conversations"][variant]))
            if d.get("reimagine_e_output"):
                _log(f"\n   Reimagine output (E): {d['reimagine_e_output']}")

    # Save
    save_data = {
        "eval_type": "provocateur_4way",
        "num_runs": num_runs,
        "wins": wins,
        "details": details,
        "total_elapsed_s": wall_elapsed,
        "report": "\n".join(_report_lines),
    }
    result_path = save_batch_result("provocateur_4way", save_data, runs_total=num_runs)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
