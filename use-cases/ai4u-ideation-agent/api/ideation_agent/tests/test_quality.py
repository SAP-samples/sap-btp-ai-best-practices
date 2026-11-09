"""Quality scorecard: prove our system decisions are the best.

Head-to-head preference judging — forces a winner, no ceiling effect.
Shows stakeholders a clear win rate.

Run:
    cd api
    python -m ideation_agent.tests.test_quality              # full scorecard (1 run)
    python -m ideation_agent.tests.test_quality --runs 3     # statistical
    python -m ideation_agent.tests.test_quality --all        # include model comparison
    python -m ideation_agent.tests.test_quality --only cse   # single technology

Sections (--only):
    stack              Full Stack vs Naive baseline
    models             GPT-4.1 vs alternative models
    cse                CSE enrichment value
    perplexity         Perplexity market research value
    perplexity_problem Perplexity problem-specific research value
    linkedin           LinkedIn hiring intelligence value

Only data-enrichment technologies are tested in per-technology isolation.
Conversation-structure decisions (two-tier, question count) cannot be reliably
A/B tested with a user simulator — their combined value is captured in the
Stack vs Naive comparison instead.

Requires .env with CSE_URL, CSE_KEY, and LLM credentials (AICORE).
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except Exception:
    pass

from ideation_agent.common import make_llm
from ideation_agent.skill_catalog import scan_skill_summaries
from ideation_agent.tools.company_search import run_cse_enrichment
from ideation_agent.tools.perplexity_research import run_perplexity_research, run_perplexity_problem_research
from ideation_agent.tools.linkedin_research import run_linkedin_research
from ideation_agent.agent import _format_cse_for_prompt, _format_perplexity_for_prompt, _format_linkedin_for_prompt

from ideation_agent.tests.eval.config import RunnerConfig
from ideation_agent.tests.eval.scenarios import ALL_SCENARIOS
from ideation_agent.tests.eval.runner import (
    make_agent_llm,
    make_eval_llm,
    run_conversation,
)
from ideation_agent.tests.eval.judge import judge_preference
from ideation_agent.tests.eval_output import save_batch_result


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

FULL_STACK = RunnerConfig(
    model_provider="openai",
    model_name="gpt-4.1",
    temperature=0.2,
    cse_enabled=True,
    repair_enabled=True,
    max_questions=5,
    two_tier_strategy=True,
    json_mode=False,  # bind() breaks ainvoke with gen_ai_hub proxy; prompt enforces JSON
)

NAIVE = RunnerConfig(
    model_provider="openai",
    model_name="gpt-4.1",
    temperature=0.2,
    cse_enabled=False,
    perplexity_enabled=False,
    linkedin_enabled=False,
    repair_enabled=False,
    max_questions=3,
    two_tier_strategy=False,
    json_mode=False,  # bind() breaks ainvoke with gen_ai_hub proxy; prompt enforces JSON
)

MODEL_ALTERNATIVES = [
    RunnerConfig(model_provider="openai", model_name="gpt-5.4"),
    RunnerConfig(model_provider="google", model_name="gemini-2.5-flash"),
]

TECHNOLOGIES = [
    ("cse", "CSE enrichment", RunnerConfig(cse_enabled=False)),
    ("perplexity", "Perplexity market research", RunnerConfig(perplexity_enabled=False)),
    ("perplexity_problem", "Perplexity problem research", RunnerConfig(perplexity_problem_research=False)),
    ("linkedin", "LinkedIn hiring intelligence", RunnerConfig(linkedin_enabled=False)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_report_lines: List[str] = []


def _log(line: str = ""):
    """Print and accumulate for the saved report."""
    print(line)
    _report_lines.append(line)


async def _fetch_cse_context(company: str) -> Optional[str]:
    if not company:
        return None
    try:
        enrichment = await run_cse_enrichment(company)
        if enrichment.get("status") == "ok":
            return _format_cse_for_prompt(enrichment)
    except Exception:
        pass
    return None


async def _fetch_perplexity_context(company: str) -> Optional[str]:
    if not company:
        return None
    try:
        enrichment = await run_perplexity_research(company)
        if enrichment.get("status") == "ok":
            return _format_perplexity_for_prompt(enrichment)
    except Exception:
        pass
    return None


async def _fetch_perplexity_problem_context(company: str, message: str) -> Optional[str]:
    if not company or not message:
        return None
    try:
        enrichment = await run_perplexity_problem_research(company, message)
        if enrichment.get("status") == "ok":
            return _format_perplexity_for_prompt(None, problem_enrichment=enrichment)
    except Exception:
        pass
    return None


_LINKEDIN_RATE_LIMIT_THRESHOLD = 20


async def _fetch_linkedin_context(company: str) -> tuple[Optional[str], int]:
    """Returns (formatted_context, rows_analyzed). rows=0 means no data or error."""
    if not company:
        return None, 0
    try:
        enrichment = await run_linkedin_research(company)
        rows = enrichment.get("rows_analyzed", 0)
        if enrichment.get("status") == "ok":
            return _format_linkedin_for_prompt(enrichment), rows
        return None, rows
    except Exception:
        return None, 0


@dataclass
class ComparisonResult:
    label: str
    scenario: str
    winner: str  # "A" or "B"
    confidence: str
    reasoning: str
    a_config: str = ""
    b_config: str = ""


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

async def run_comparison(
    scenario: Dict[str, str],
    config_a: RunnerConfig,
    config_b: RunnerConfig,
    *,
    eval_llm,
    cse_context: Optional[str],
    perplexity_context: Optional[str] = None,
    perplexity_problem_context: Optional[str] = None,
    linkedin_context: Optional[str] = None,
    skill_summaries: list,
    label: str = "",
    max_retries: int = 5,
    prompt_fn_a=None,
    prompt_fn_b=None,
) -> ComparisonResult:
    """Run two configs head-to-head on one scenario.

    Randomizes presentation order to the judge to eliminate positional bias.
    Retries on transient errors (content filter, rate limits).
    """
    for attempt in range(max_retries):
        try:
            llm_a = make_agent_llm(config_a)
            llm_b = make_agent_llm(config_b)

            result_a, result_b = await asyncio.gather(
                run_conversation(
                    llm_a, eval_llm, scenario, config_a,
                    cse_context=cse_context, perplexity_context=perplexity_context,
                    perplexity_problem_context=perplexity_problem_context,
                    linkedin_context=linkedin_context,
                    skill_summaries=skill_summaries,
                    system_prompt_fn=prompt_fn_a,
                ),
                run_conversation(
                    llm_b, eval_llm, scenario, config_b,
                    cse_context=cse_context, perplexity_context=perplexity_context,
                    perplexity_problem_context=perplexity_problem_context,
                    linkedin_context=linkedin_context,
                    skill_summaries=skill_summaries,
                    system_prompt_fn=prompt_fn_b,
                ),
            )

            # Randomize position to eliminate judge's positional bias
            swapped = random.random() < 0.5
            if swapped:
                turns_first, turns_second = result_b.turns, result_a.turns
            else:
                turns_first, turns_second = result_a.turns, result_b.turns

            judgment = await judge_preference(
                eval_llm, scenario, turns_first, turns_second,
            )

            # Map winner back to original A/B
            raw_winner = judgment["winner"]
            if swapped:
                actual_winner = "B" if raw_winner == "A" else "A"
            else:
                actual_winner = raw_winner

            return ComparisonResult(
                label=label,
                scenario=scenario["name"],
                winner=actual_winner,
                confidence=judgment.get("confidence", "medium"),
                reasoning=judgment.get("reasoning", ""),
                a_config=config_a.model_name if config_a.model_name != config_b.model_name else "full_stack",
                b_config=config_b.model_name if config_a.model_name != config_b.model_name else "naive",
            )
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
            if attempt < max_retries - 1:
                backoff = 15 * (attempt + 1) if is_rate_limit else 2
                if is_rate_limit:
                    print(f"   ⏳ Rate limit hit, backing off {backoff}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(backoff)
                continue
            raise


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------

_CONCURRENCY = asyncio.Semaphore(15)


async def _guarded_comparison(*args, **kwargs) -> ComparisonResult:
    """Run a comparison under the concurrency semaphore."""
    async with _CONCURRENCY:
        return await run_comparison(*args, **kwargs)


async def _run_stack_vs_naive(
    eval_llm, num_runs: int, cse_contexts: Dict, perplexity_contexts: Dict,
    perplexity_problem_contexts: Dict, linkedin_contexts: Dict, skill_summaries: list
) -> tuple:
    """Full Stack vs Naive baseline comparison. Returns (data_dict, report_lines)."""
    lines: List[str] = []
    lines.append(f"\n Full Stack vs Naive Baseline")
    lines.append(f" {'─' * 40}")

    tasks = []
    for run_idx in range(num_runs):
        for scenario in ALL_SCENARIOS:
            cse_context = cse_contexts.get(scenario.get("company", ""))
            perplexity_context = perplexity_contexts.get(scenario.get("company", ""))
            problem_key = f"{scenario.get('company', '')}::{scenario.get('name', '')}"
            perplexity_problem_context = perplexity_problem_contexts.get(problem_key)
            linkedin_context = linkedin_contexts.get(scenario.get("company", ""))
            tasks.append((run_idx, scenario, asyncio.ensure_future(
                _guarded_comparison(
                    scenario, FULL_STACK, NAIVE,
                    eval_llm=eval_llm,
                    cse_context=cse_context,
                    perplexity_context=perplexity_context,
                    perplexity_problem_context=perplexity_problem_context,
                    linkedin_context=linkedin_context,
                    skill_summaries=skill_summaries,
                    label="stack_vs_naive",
                )
            )))

    stack_wins = 0
    naive_wins = 0
    details: List[Dict] = []

    for run_idx, scenario, fut in tasks:
        try:
            result = await fut
        except Exception as e:
            lines.append(f"   ⚠ {scenario['name'][:45]:45s} error: {str(e)[:60]}")
            continue

        details.append({
            "scenario": result.scenario, "winner": result.winner,
            "confidence": result.confidence, "reasoning": result.reasoning,
        })

        if result.winner == "A":
            stack_wins += 1
        else:
            naive_wins += 1

        icon = "✓" if result.winner == "A" else "✗"
        conf = f"[{result.confidence}]"
        lines.append(f"   {icon} {scenario['name'][:45]:45s} {conf:8s} {result.reasoning[:60]}")

    total = stack_wins + naive_wins
    rate = stack_wins / total if total > 0 else 0
    lines.append(f"\n   Our system wins: {stack_wins}/{total} ({rate:.0%})")
    lines.append(f"   Naive wins:      {naive_wins}/{total} ({1 - rate:.0%})")

    data = {
        "stack_wins": stack_wins, "naive_wins": naive_wins,
        "total": total, "win_rate": round(rate, 3), "details": details,
    }
    return data, lines


async def _run_model_comparison(
    eval_llm, num_runs: int, cse_contexts: Dict, perplexity_contexts: Dict,
    perplexity_problem_contexts: Dict, linkedin_contexts: Dict, skill_summaries: list
) -> tuple:
    """GPT-4.1 vs alternative models. Returns (data_dict, report_lines)."""
    lines: List[str] = []
    lines.append(f"\n Model Choice (GPT-4.1 vs alternatives)")
    lines.append(f" {'─' * 40}")

    model_results: Dict[str, Dict[str, int]] = {}

    for alt_config in MODEL_ALTERNATIVES:
        alt_label = alt_config.model_name
        model_results[alt_label] = {"gpt41_wins": 0, "alt_wins": 0}

        tasks = []
        for _ in range(num_runs):
            for scenario in ALL_SCENARIOS:
                cse_context = cse_contexts.get(scenario.get("company", ""))
                perplexity_context = perplexity_contexts.get(scenario.get("company", ""))
                problem_key = f"{scenario.get('company', '')}::{scenario.get('name', '')}"
                perplexity_problem_context = perplexity_problem_contexts.get(problem_key)
                linkedin_context = linkedin_contexts.get(scenario.get("company", ""))
                tasks.append((scenario, asyncio.ensure_future(
                    _guarded_comparison(
                        scenario, FULL_STACK, alt_config,
                        eval_llm=eval_llm,
                        cse_context=cse_context,
                        perplexity_context=perplexity_context,
                        perplexity_problem_context=perplexity_problem_context,
                        linkedin_context=linkedin_context,
                        skill_summaries=skill_summaries,
                        label=f"gpt41_vs_{alt_label}",
                    )
                )))

        for scenario, fut in tasks:
            try:
                result = await fut
            except Exception as e:
                lines.append(f"   ⚠ {scenario['name'][:30]} — {alt_label} error: {str(e)[:60]}")
                continue

            if result.winner == "A":
                model_results[alt_label]["gpt41_wins"] += 1
            else:
                model_results[alt_label]["alt_wins"] += 1

        wins = model_results[alt_label]["gpt41_wins"]
        total = wins + model_results[alt_label]["alt_wins"]
        if total > 0:
            lines.append(f"   GPT-4.1 beats {alt_label}: {wins}/{total} ({wins/total:.0%})")
        else:
            lines.append(f"   GPT-4.1 vs {alt_label}: no successful comparisons")

    return model_results, lines


async def _run_technology(
    tech_key: str, tech_name: str, disabled_config: RunnerConfig,
    eval_llm, num_runs: int, cse_contexts: Dict, perplexity_contexts: Dict,
    perplexity_problem_contexts: Dict, linkedin_contexts: Dict, skill_summaries: list,
) -> Dict[str, int]:
    """Test a single technology's value against relevant scenarios only."""
    relevant_scenarios = [s for s in ALL_SCENARIOS if tech_key in s.get("stresses", [])]
    enabled_wins = 0
    disabled_wins = 0

    tasks = []
    for _ in range(num_runs):
        for scenario in relevant_scenarios:
            cse_context = cse_contexts.get(scenario.get("company", ""))
            perplexity_context = perplexity_contexts.get(scenario.get("company", ""))
            problem_key = f"{scenario.get('company', '')}::{scenario.get('name', '')}"
            perplexity_problem_context = perplexity_problem_contexts.get(problem_key)
            linkedin_context = linkedin_contexts.get(scenario.get("company", ""))
            tasks.append((scenario, asyncio.ensure_future(
                _guarded_comparison(
                    scenario, FULL_STACK, disabled_config,
                    eval_llm=eval_llm,
                    cse_context=cse_context,
                    perplexity_context=perplexity_context,
                    perplexity_problem_context=perplexity_problem_context,
                    linkedin_context=linkedin_context,
                    skill_summaries=skill_summaries,
                    label=f"tech_{tech_key}",
                )
            )))

    for scenario, fut in tasks:
        try:
            result = await fut
        except Exception as e:
            print(f"   ⚠ {scenario['name'][:30]} — {tech_name} error: {str(e)[:60]}")
            continue

        if result.winner == "A":
            enabled_wins += 1
        else:
            disabled_wins += 1

    return {"enabled_wins": enabled_wins, "disabled_wins": disabled_wins}


async def _run_per_technology(
    eval_llm, num_runs: int, cse_contexts: Dict, perplexity_contexts: Dict,
    perplexity_problem_contexts: Dict, linkedin_contexts: Dict, skill_summaries: list,
    *,
    only_tech: Optional[str] = None,
) -> tuple:
    """Per-technology value testing. Returns (data_dict, report_lines)."""
    lines: List[str] = []
    lines.append(f"\n Per-Technology Value")
    lines.append(f" {'─' * 40}")

    tech_results: Dict[str, Dict[str, int]] = {}

    techs_to_run = TECHNOLOGIES
    if only_tech:
        techs_to_run = [(k, n, c) for k, n, c in TECHNOLOGIES if k == only_tech]

    for tech_key, tech_name, disabled_config in techs_to_run:
        counts = await _run_technology(
            tech_key, tech_name, disabled_config,
            eval_llm, num_runs, cse_contexts, perplexity_contexts,
            perplexity_problem_contexts, linkedin_contexts, skill_summaries,
        )
        tech_results[tech_name] = counts

        wins = counts["enabled_wins"]
        total = wins + counts["disabled_wins"]
        rate = wins / total if total > 0 else 0
        icon = "✓" if rate > 0.5 else "~" if rate == 0.5 else "✗"
        lines.append(f"   {icon} {tech_name:25s} wins {wins}/{total} ({rate:.0%})")

    return tech_results, lines


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

VALID_SECTIONS = ["stack", "models"] + [key for key, _, _ in TECHNOLOGIES]


def _parse_args():
    runs = 1
    full = False
    only = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--runs="):
            runs = int(arg.split("=", 1)[1])
        elif arg == "--runs" and i + 1 < len(args):
            i += 1
            runs = int(args[i])
        elif arg == "--all":
            full = True
        elif arg.startswith("--only="):
            only = arg.split("=", 1)[1]
        elif arg == "--only" and i + 1 < len(args):
            i += 1
            only = args[i]
        i += 1

    if only and only not in VALID_SECTIONS:
        print(f"Error: --only must be one of: {', '.join(VALID_SECTIONS)}")
        sys.exit(1)

    return runs, full, only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    num_runs, include_models, only = _parse_args()
    wall_t0 = time.time()

    eval_llm = make_eval_llm()
    skill_summaries = scan_skill_summaries()

    # Determine which enrichments are needed for the selected section
    needs_cse = only in (None, "stack", "cse", "models")
    needs_perplexity = only in (None, "stack", "perplexity", "models")
    needs_perplexity_problem = only in (None, "stack", "perplexity_problem", "models")
    needs_linkedin = only in (None, "stack", "linkedin")  # skip for --only models (too slow)

    # Pre-fetch enrichment data in parallel (only what's needed)
    unique_companies = list({s.get("company", "") for s in ALL_SCENARIOS} - {""})
    unique_problems: list[tuple[str, str, str]] = []
    if needs_perplexity_problem:
        seen_keys: set[str] = set()
        for s in ALL_SCENARIOS:
            company, message = s.get("company", ""), s.get("message", "")
            key = f"{company}::{s.get('name', '')}"
            if company and message and key not in seen_keys:
                seen_keys.add(key)
                unique_problems.append((company, message, key))

    # Launch all fetches concurrently
    cse_futs = {c: asyncio.ensure_future(_fetch_cse_context(c))
                for c in unique_companies} if needs_cse else {}
    perplexity_futs = {c: asyncio.ensure_future(_fetch_perplexity_context(c))
                       for c in unique_companies} if needs_perplexity else {}
    problem_futs = {key: asyncio.ensure_future(_fetch_perplexity_problem_context(company, message))
                    for company, message, key in unique_problems} if needs_perplexity_problem else {}
    linkedin_futs = {c: asyncio.ensure_future(_fetch_linkedin_context(c))
                     for c in unique_companies} if needs_linkedin else {}

    # Await all together
    all_futs = [*cse_futs.values(), *perplexity_futs.values(),
                *problem_futs.values(), *linkedin_futs.values()]
    if all_futs:
        await asyncio.gather(*all_futs)

    cse_contexts: Dict[str, Optional[str]] = {c: fut.result() for c, fut in cse_futs.items()}
    perplexity_contexts: Dict[str, Optional[str]] = {c: fut.result() for c, fut in perplexity_futs.items()}
    perplexity_problem_contexts: Dict[str, Optional[str]] = {k: fut.result() for k, fut in problem_futs.items()}
    linkedin_contexts: Dict[str, Optional[str]] = {}
    linkedin_stats: Dict[str, int] = {}
    for c, fut in linkedin_futs.items():
        context, rows = fut.result()
        linkedin_contexts[c] = context
        linkedin_stats[c] = rows

    total_t0 = time.time()

    _report_lines.clear()
    _log(f"{'═' * 60}")
    _log(f" QUALITY REPORT — {time.strftime('%Y-%m-%d')} | runs={num_runs}" +
         (f" | only={only}" if only else ""))
    _log(f"{'═' * 60}")

    # ─── LinkedIn data health ────────────────────────────────────────
    linkedin_is_testing = only in (None, "stack", "linkedin")
    if linkedin_stats and linkedin_is_testing:
        rate_limited = {c: r for c, r in linkedin_stats.items() if r < _LINKEDIN_RATE_LIMIT_THRESHOLD}
        _log(f"\n LinkedIn Data Health")
        _log(f" {'─' * 40}")
        for company, rows in sorted(linkedin_stats.items()):
            flag = " ⚠ RATE-LIMITED?" if rows < _LINKEDIN_RATE_LIMIT_THRESHOLD else ""
            _log(f"   {company:20s} {rows:>4} positions{flag}")
        if rate_limited:
            _log(f"\n   ⚠ WARNING: {len(rate_limited)} company(ies) returned <{_LINKEDIN_RATE_LIMIT_THRESHOLD} positions.")
            _log(f"     LinkedIn results may be unreliable due to rate limiting.")
            _log(f"     Wait and retry, or interpret LinkedIn win rates with caution.")
        _log("")

    # Decide which sections to run
    run_stack = only in (None, "stack")
    run_models = only == "models" or (only is None and include_models)
    run_all_tech = only is None
    run_single_tech = only in [key for key, _, _ in TECHNOLOGIES]

    # ─── Run sections in parallel ────────────────────────────────────
    stack_fut = asyncio.ensure_future(
        _run_stack_vs_naive(eval_llm, num_runs, cse_contexts, perplexity_contexts,
                            perplexity_problem_contexts, linkedin_contexts, skill_summaries)
    ) if run_stack else None

    model_fut = asyncio.ensure_future(
        _run_model_comparison(eval_llm, num_runs, cse_contexts, perplexity_contexts,
                              perplexity_problem_contexts, linkedin_contexts, skill_summaries)
    ) if run_models else None

    tech_fut = asyncio.ensure_future(
        _run_per_technology(
            eval_llm, num_runs, cse_contexts, perplexity_contexts,
            perplexity_problem_contexts, linkedin_contexts, skill_summaries,
            only_tech=only if run_single_tech else None,
        )
    ) if (run_all_tech or run_single_tech) else None

    stack_data, stack_lines = (await stack_fut) if stack_fut else ({}, [])
    model_data, model_lines = (await model_fut) if model_fut else ({}, [])
    tech_data, tech_lines = (await tech_fut) if tech_fut else ({}, [])

    # ─── Assemble report in correct order ────────────────────────────
    for line in stack_lines + model_lines + tech_lines:
        _log(line)

    # ─── Final verdict ────────────────────────────────────────────────
    total_elapsed = round(time.time() - total_t0, 1)

    _log(f"\n{'═' * 60}")
    if stack_data:
        stack_rate = stack_data["win_rate"]
        if stack_rate >= 0.7:
            _log(f" ✓ PASS — System outperforms naive baseline ({stack_rate:.0%} win rate)")
        elif stack_rate >= 0.5:
            _log(f" ~ MARGINAL — System wins {stack_rate:.0%} (target: >70%)")
        else:
            _log(f" ✗ FAIL — Naive baseline wins more often ({1 - stack_rate:.0%})")
    wall_elapsed = round(time.time() - wall_t0, 1)
    _log(f" Total time: {wall_elapsed}s (eval: {total_elapsed}s)")
    _log(f"{'═' * 60}")

    # ─── Save results ─────────────────────────────────────────────────
    save_data = {
        "eval_type": "quality_scorecard",
        "num_runs": num_runs,
        "section": only or "full",
        "stack_vs_naive": stack_data,
        "model_comparison": model_data,
        "per_technology": tech_data,
        "linkedin_positions": linkedin_stats,
        "total_elapsed_s": total_elapsed,
        "report": "\n".join(_report_lines),
    }

    result_path = save_batch_result("quality", save_data, runs_total=num_runs)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
