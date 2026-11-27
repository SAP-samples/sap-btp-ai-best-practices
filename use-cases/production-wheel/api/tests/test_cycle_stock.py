"""Cycle stock strategy in the optimizer: runner-multiple versions, coefficients and choice.

Uses the worked example of docs/managed-complexity.md: one subgroup with base
coverage 10 days and 0.5 batches per week; multiples 1/2/2/4.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_cycle_stock.py -q
"""

from __future__ import annotations

import pytest

from production_wheel import cycle_stock
from production_wheel.candidates import BlockProjection, Candidate, CandidateMember, CandidatePool
from production_wheel.cycle_stock import RUNNER_MULTIPLE, SINGLE, multiple_version
from production_wheel.optimization import BlockSolverSession, precompute_coefficients, solve_candidate_pools
from production_wheel.optimization.solver import strategy_candidates
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig

BLOCK = ("P", "S")
# FINI -> (demand litres, full-batch coverage days); classes high, high, medium, low.
EXAMPLE = {"P1": (4000, 12), "P2": (1600, 30), "P3": (690, 70), "P4": (160, 300)}
BATCH = 10 * sum(demand for demand, _ in EXAMPLE.values()) / 250  # base coverage 10 days


def members(example=EXAMPLE) -> tuple[CandidateMember, ...]:
    """Build the worked-example FINIs with their source full-batch coverage."""

    return tuple(
        CandidateMember(fini_id=fini, plant=BLOCK[0], sefi=BLOCK[1], eligible_lines=frozenset({"L1"}),
                        demand_litres=demand, pallet_litres=1, package_volume=1, fixed_pv="PV1",
                        reference_coverage_source_days=full)
        for fini, (demand, full) in example.items()
    )


def group(example=EXAMPLE, identity="all") -> Candidate:
    """Build the single (pool) candidate grouping every FINI of the example."""

    ids = tuple(sorted(example))
    return Candidate(block_key=BLOCK, member_ids=ids, pv_id="PV1", nominal_lot_litres=BATCH / 0.9,
                     effective_batch_litres=BATCH, common_lines=("L1",),
                     group_demand_litres=sum(example[f][0] for f in ids), matrix_exception_pairs=(),
                     equal_pck=False, source_tiers=("exact",), membership_hash=f"membership-{identity}",
                     candidate_hash=f"candidate-{identity}")


def config(mode: str, **policy) -> RunConfig:
    """Return a trade-off configuration with the given cycle stock mode."""

    return RunConfig(coverage_mode=CoverageMode.PARETO, matrix_mode=MatrixMode.OFF,
                     cycle_stock_policy={"mode": mode, **policy})


def pool(candidates, run_config) -> CandidatePool:
    """Wrap candidates in a complete pool built under ``run_config``."""

    return CandidatePool(
        block_key=BLOCK, method="exact", completeness="complete", structural_fingerprint="fixture",
        projection=BlockProjection(BLOCK, 4, 7, len(candidates), 1, len(candidates), "exact"),
        candidates=tuple(candidates), mandatory_member_sets=0, mandatory_pv_configurations=0,
        size_traces=(), pool_hash="pool", config_fingerprint=run_config.structural_ruleset_fingerprint())


def test_module_self_check_passes() -> None:
    """The worked example inside the module (also run by ``python -m``) holds."""

    cycle_stock._self_check()


def test_worked_example_coefficients() -> None:
    """Single keeps today's numbers; runner multiples give J_CH 0.625 and coverage 14.29."""

    single, multiple = precompute_coefficients((group(), multiple_version(group())), members(),
                                               config(RUNNER_MULTIPLE))
    assert single.coverage_days == pytest.approx(10)
    assert single.j_ch == pytest.approx(1.5)
    assert single.max_runner_multiple == 1
    assert multiple.coverage_days == pytest.approx(92200 / 6450)
    assert round(multiple.coverage_days, 2) == 14.29
    assert multiple.j_ch == pytest.approx(0.625)
    assert multiple.max_runner_multiple == 4


def test_versions_have_distinct_identity_and_same_membership() -> None:
    """Both versions survive hash de-duplication but describe the same group."""

    single = group()
    variant = multiple_version(single)
    assert variant.candidate_hash != single.candidate_hash
    assert variant.membership_hash == single.membership_hash
    assert (variant.member_ids, variant.pv_id, variant.cycle_stock_strategy) == (
        single.member_ids, single.pv_id, RUNNER_MULTIPLE)


def test_strategy_candidates_per_mode() -> None:
    """single: unchanged; runner_multiple: replaced; optimizer_choice: both versions."""

    singleton = group({"P1": EXAMPLE["P1"]}, "one")
    pair = (singleton, group())
    unchanged, _ = strategy_candidates(pair, members(), config(SINGLE))
    assert unchanged == pair
    replaced, coefficients = strategy_candidates(pair, members(), config(RUNNER_MULTIPLE))
    assert [c.cycle_stock_strategy for c in replaced] == [SINGLE, RUNNER_MULTIPLE]
    assert [c.stable_rank for c in coefficients] == [1, 2]
    both, _ = strategy_candidates(pair, members(), config("optimizer_choice"))
    assert [c.cycle_stock_strategy for c in both] == [SINGLE, SINGLE, RUNNER_MULTIPLE]


def test_mixed_groups_only_and_no_op_multiples() -> None:
    """Same-class groups are not offered multiples; a group whose multiples are all 1 stays single."""

    # Two high runners (ratio 1.0 and 2.0): same class.
    similar = {"P1": (4000, 12), "P2": (1600, 24)}
    offered, _ = strategy_candidates((group(similar),), members(similar), config("optimizer_choice"))
    assert [c.cycle_stock_strategy for c in offered] == [SINGLE]
    offered, _ = strategy_candidates((group(similar),), members(similar),
                                     config("optimizer_choice", mixed_groups_only=False))
    assert [c.cycle_stock_strategy for c in offered] == [SINGLE, RUNNER_MULTIPLE]
    # Identical full-batch coverage: every multiple is 1.
    equal = {"P1": (4000, 12), "P2": (1600, 12)}
    kept, coefficients = strategy_candidates((group(equal),), members(equal), config(RUNNER_MULTIPLE))
    assert kept == (group(equal),)
    assert coefficients[0].j_ch == pytest.approx(precompute_coefficients(
        (group(equal),), members(equal), config(SINGLE))[0].j_ch)
    offered, _ = strategy_candidates((group(equal),), members(equal),
                                     config("optimizer_choice", mixed_groups_only=False))
    assert offered == (group(equal),)


@pytest.mark.parametrize("epsilon,expected", [(0.7, RUNNER_MULTIPLE), (2.0, SINGLE)])
def test_optimizer_choice_picks_the_version_on_the_trade_off(epsilon, expected) -> None:
    """A tight J_CH budget needs multiples; a loose one keeps the lower single coverage.

    Both versions cover the same FINIs, so the exact cover selects exactly one.
    """

    run_config = config("optimizer_choice")
    # Same batch for every candidate; all singletons: coverage 40 days, J_CH 0.
    singletons = tuple(group({fini: EXAMPLE[fini]}, fini) for fini in EXAMPLE)
    pools = (pool((group(), *singletons), run_config),)
    result = solve_candidate_pools(pools, members(), run_config, j_ch_epsilon=epsilon)
    assert [(item.candidate.member_ids, item.candidate.cycle_stock_strategy) for item in result.selected] == [
        (tuple(sorted(EXAMPLE)), expected)]
    session = BlockSolverSession(pools[0], members(), run_config)
    again = session.solve(CoverageMode.PARETO, j_ch_epsilon=epsilon)
    assert [item.candidate.candidate_hash for item in again.selected] == [
        item.candidate.candidate_hash for item in result.selected]


def test_validation_and_reports_agree_with_the_multiple_version() -> None:
    """Independent validation recomputes the strategy figures; reports carry them per FINI."""

    from production_wheel.candidates import ProductionVersion
    from production_wheel.reporting import build_solution_groups_rows
    from production_wheel.solution_validation import validate_solution

    run_config = config("optimizer_choice")
    singletons = tuple(group({fini: EXAMPLE[fini]}, fini) for fini in EXAMPLE)
    pools = (pool((group(), *singletons), run_config),)
    result = solve_candidate_pools(pools, members(), run_config, j_ch_epsilon=0.7)
    catalog = (ProductionVersion(*BLOCK, "PV1", BATCH / 0.9),)
    validation = validate_solution(result, members(), pools, run_config, catalog)
    assert [issue.rule_id for issue in validation.issues if issue.severity == "error"] == []
    (evidence,) = validation.groups
    assert evidence.cycle_stock_strategy == RUNNER_MULTIPLE
    assert evidence.coverage_days == pytest.approx(92200 / 6450)
    assert evidence.j_ch_contribution == pytest.approx(0.625)
    assert evidence.base_group_coverage_days == pytest.approx(10)
    assert dict(evidence.fini_runner_multiples) == {"P1": 1, "P2": 2, "P3": 2, "P4": 4}
    assert dict(evidence.fini_strategy_coverage_days) == pytest.approx({"P1": 10, "P2": 20, "P3": 20, "P4": 40})
    (row,) = build_solution_groups_rows(validation.groups, result, run_config)
    assert (row["cycle_stock_strategy"], round(row["strategy_coverage_days"], 2)) == (RUNNER_MULTIPLE, 14.29)

    # The same multiple version is not in the library of a single-mode run.
    single_config = config(SINGLE)
    tampered = validate_solution(result, members(), (pool((group(), *singletons), single_config),),
                                 single_config, catalog)
    assert "SELECTED_CANDIDATE_IN_LIBRARY" in {issue.rule_id for issue in tampered.issues}


def test_business_rule_facts_follow_the_strategy() -> None:
    """group.coverage_days / group.j_ch use the multiples; member.runner_multiple is exposed."""

    from production_wheel.business_rules import facts, runner_ratios

    run_config = config(RUNNER_MULTIPLE)
    ratios = runner_ratios(members(), run_config)
    single = facts(group(), members(), run_config, ratios)
    multiple = facts(multiple_version(group()), members(), run_config, ratios)
    assert (single["group"]["cycle_stock_strategy"], single["group"]["j_ch"]) == (SINGLE, pytest.approx(1.5))
    assert [m["runner_multiple"] for m in single["members"]] == [1, 1, 1, 1]
    assert multiple["group"]["cycle_stock_strategy"] == RUNNER_MULTIPLE
    assert multiple["group"]["j_ch"] == pytest.approx(0.625)
    assert multiple["group"]["coverage_days"] == pytest.approx(92200 / 6450)
    assert [m["runner_multiple"] for m in multiple["members"]] == [1, 2, 2, 4]


def test_group_rule_on_coverage_removes_the_multiple_version() -> None:
    """A hard rule capping group coverage at 12 days keeps only the single version."""

    from production_wheel.rule_models import GroupRule

    rule = GroupRule.model_validate({
        "kind": "group_rule", "constraint_id": "cap", "approval_status": "approved",
        "when": {"op": "literal", "value": True},
        "assertion": {"op": "lte", "args": [{"op": "field", "field": "group.coverage_days"},
                                             {"op": "literal", "value": 12}]}})
    run_config = config("optimizer_choice").model_copy(update={"business_rules": (rule,)})
    offered, _ = strategy_candidates((group(),), members(), run_config)
    assert [c.cycle_stock_strategy for c in offered] == [SINGLE]


def test_rule_grammar_rejects_unknown_strategy_labels() -> None:
    """Comparing group.cycle_stock_strategy with an unknown label fails at validation."""

    from production_wheel.rule_models import Expression

    field = {"op": "field", "field": "group.cycle_stock_strategy"}
    Expression.model_validate({"op": "eq", "args": [field, {"op": "literal", "value": RUNNER_MULTIPLE}]})
    with pytest.raises(ValueError, match="cycle_stock_strategy is one of"):
        Expression.model_validate({"op": "eq", "args": [field, {"op": "literal", "value": "adjusted"}]})
