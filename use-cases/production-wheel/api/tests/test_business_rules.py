"""Direct solver acceptance cases for conditional rules and plant-wide bounds."""

from pathlib import Path

import pytest

from production_wheel.candidates import CandidateMember, ProductionVersion
from production_wheel.constraints import compile_request
from production_wheel.scenarios import CanonicalInputs, run_greenfield_frontier
from production_wheel.schemas import SolveRequest
from production_wheel.solution_validation import validate_solution


def expression(op, *args, **values):
    """Build a small expression in the public business-rule vocabulary."""
    return {"op": op, "args": list(args), **values}


def fixture_inputs(blocks=("S",)):
    """Create two compatible FINIs per block with independently known coverage."""
    members = tuple(CandidateMember.from_record({
        "fini_id": f"{block}{i}", "plant": "P", "sefi": block,
        "eligible_lines": "4|6", "demand_litres": demand,
        "pallet_litres": 10, "package_volume": 2, "fixed_pv": "PV1",
        "lot_size_considered_litres": 60, "source_horizon_days": 250,
        "xyz_x_dc_count": int(i == 0), "xyz_y_dc_count": 0,
        "xyz_z_dc_count": int(i == 1), "pck_code": "CAN",
    }) for block in blocks for i, demand in enumerate((1000, 100)))
    return CanonicalInputs(Path("."), (), members,
        tuple(ProductionVersion("P", block, "PV1", 100) for block in blocks))


def solve_rules(rules, *, blocks=("S",), **settings):
    """Compile and directly solve a tiny exact request without UI or persistence."""
    inputs = fixture_inputs(blocks)
    request = SolveRequest.model_validate({"config": {
        "coverage_mode": "PARETO", "matrix_mode": "OFF", **settings,
    }, "constraints": rules})
    compiled = compile_request(request, inputs)
    result = run_greenfield_frontier(compiled.inputs, compiled.config,
        point_count=3, block_option_count=3, per_block_total_seconds=10,
        block_worker_count=1, pool_transform=compiled.filter_pools)
    return inputs, compiled, result


def test_no_singletons_remains_feasible_and_independently_valid():
    """Removing all singletons still permits a complete two-FINI group."""
    rule = {"kind": "group_rule", "constraint_id": "no-singletons",
        "approval_status": "approved", "assertion": expression("gte",
            expression("field", field="group.size"), expression("literal", value=2))}
    inputs, compiled, result = solve_rules([rule])
    assert result.points and not result.block_failures
    for point in result.points:
        assert [len(s.candidate.member_ids) for s in point.selected] == [2]
        assert validate_solution(point.solve_result, inputs.members, result.pools,
            compiled.config, inputs.production_versions).is_valid


def test_plant_total_is_enforced_across_two_blocks():
    """A plant total of three groups forces one pair plus two singletons."""
    rule = {"kind": "selection_bound", "constraint_id": "three-groups",
        "approval_status": "approved", "scope": {"plant": "P"},
        "measure": expression("literal", value=1), "lower": 3, "upper": 3}
    _, _, result = solve_rules([rule], blocks=("S", "T"))
    assert result.points
    assert all(len(p.selected) == 3 for p in result.points)


def test_assigned_line_is_a_real_candidate_choice():
    """The optimizer can choose line 4 to admit a pair forbidden on line 6."""
    rule = {"kind": "group_rule", "constraint_id": "line-six-singletons",
        "approval_status": "approved", "when": expression("eq",
            expression("field", field="group.selected_line"), expression("literal", value="6")),
        "assertion": expression("lte", expression("field", field="group.size"),
            expression("literal", value=1))}
    _, _, result = solve_rules([rule], assign_filling_lines=True)
    assert result.points
    for point in result.points:
        for selected in point.selected:
            assert selected.candidate.selected_line in ("4", "6")
            assert selected.candidate.selected_line != "6" or len(selected.candidate.member_ids) == 1


def test_missing_optional_data_is_not_treated_as_zero():
    """Unknown DC evidence must stop an XYZ-dependent rule."""
    rule = {"kind": "group_rule", "constraint_id": "missing",
        "approval_status": "approved", "assertion": expression("all",
            expression("gt", expression("field", field="member.primary_dc_count"),
                expression("literal", value=0)))}
    with pytest.raises(ValueError, match="primary_dc_count"):
        solve_rules([rule])


def test_joint_frontier_report_accepts_unconstrained_anchors():
    """Persistable joint results label anchors without fabricating epsilon bounds."""
    from production_wheel.result_bundle import build_result_bundle
    inputs, _, result = solve_rules([{'kind': 'selection_bound', 'constraint_id': 'total',
        'approval_status': 'approved', 'measure': expression('literal', value=1), 'upper': 2}])
    bundle = build_result_bundle(result, inputs)
    assert bundle.metadata['integrity_valid']
    assert 'anchor' in bundle.artifacts['solution_report.md']


def test_three_tier_runner_classes_and_fingerprint():
    """Runner classes follow the profile thresholds; altered evidence invalidates pools.

    Fixture coverage (reference basis): S0 = 60 / (1000 / 250) = 15 days,
    S1 = 60 / (100 / 250) = 150 days, so S1's ratio to the SEFI's fastest is 10.
    """
    from dataclasses import replace
    from production_wheel.candidates import generate_candidate_pools
    from production_wheel.business_rules import facts, runner_ratios
    from production_wheel.runners import RunnerThresholds
    from production_wheel.schemas import RunConfig
    inputs = fixture_inputs()
    config = RunConfig(matrix_mode='OFF', business_rules=({'kind': 'group_rule',
        'constraint_id': 'positive-size', 'approval_status': 'approved',
        'assertion': expression('gt', expression('field', field='group.size'), expression('literal', value=0))},))
    pools = generate_candidate_pools(inputs.members, inputs.production_versions, config)
    pair = next(c for c in pools[0].candidates if c.member_ids == ('S0', 'S1'))

    def classes(members, cfg):
        return [r['runner'] for r in facts(pair, members, cfg, runner_ratios(members, cfg))['members']]

    # Defaults 10/80 days, 3/10 ratio: S0 is medium by days but the SEFI's fastest (ratio 1), so high.
    assert classes(inputs.members, config) == ['high', 'low']
    assert facts(pair, inputs.members, config, runner_ratios(inputs.members, config))['members'][1]['sefi_ratio'] == 10
    wide = config.model_copy(update={'runner_thresholds': RunnerThresholds(low_min_days=200, low_min_ratio=20)})
    assert classes(inputs.members, wide) == ['high', 'medium']
    # The source "RoC 1 full SEFI batch" wins over lot size / demand (same number SAP uses).
    sourced = (inputs.members[0], replace(inputs.members[1], reference_coverage_source_days=5))
    assert classes(sourced, config) == ['high', 'high']
    # Candidate-PV basis: 90 L effective batch gives 22.5 and 225 days; ratios unchanged.
    assert classes(inputs.members, config.model_copy(update={'runner_basis': 'candidate_pv'})) == ['high', 'low']
    changed = (replace(inputs.members[0], lot_size_considered_litres=60.000004), *inputs.members[1:])
    other = generate_candidate_pools(changed, inputs.production_versions, config)
    assert pools[0].structural_fingerprint != other[0].structural_fingerprint


def test_explicit_line_priority_is_preserved_across_frontier():
    """Line priority chooses the high/low pair and preserves its assigned demand."""
    rule = {'kind': 'group_rule', 'constraint_id': 'line-high', 'approval_status': 'approved',
        'when': expression('eq', expression('field', field='group.selected_line'), expression('literal', value='6')),
        'assertion': expression('any', expression('eq', expression('field', field='member.runner'), expression('literal', value='high')))}
    _, _, result = solve_rules([rule], preferred_line='6')
    assert result.points
    assert all(sum(s.candidate.group_demand_litres for s in point.selected if s.candidate.selected_line == '6') == 1100 for point in result.points)


def disperser_share_rule(threshold):
    """Require at least half of all SEFI batches on PV lots of at least `threshold` litres."""
    high = expression("if", expression("gte", expression("field", field="group.pv_lot_litres"),
        expression("literal", value=threshold)), expression("literal", value=0.5),
        expression("literal", value=-0.5))
    return {"kind": "selection_bound", "constraint_id": "high-disperser-share",
        "approval_status": "approved", "scope": {"plant": "P"},
        "measure": expression("mul", expression("field", field="group.batch_count"), high),
        "lower": 0}


def test_batch_share_rule_uses_pv_lot_and_batch_count():
    """All batches on lot-100 PVs satisfy a 100 L threshold; a 101 L threshold is infeasible."""
    _, _, feasible = solve_rules([disperser_share_rule(100)], blocks=("S", "T"))
    assert feasible.points
    _, _, infeasible = solve_rules([disperser_share_rule(101)], blocks=("S", "T"))
    assert not infeasible.points


def test_new_group_facts_and_if_operator():
    """Batch facts follow the effective batch convention and `if` only evaluates one branch."""
    from types import SimpleNamespace
    from production_wheel.business_rules import evaluate, facts
    from production_wheel.rule_models import Expression
    inputs = fixture_inputs()
    _, compiled, _ = solve_rules([])
    candidate = SimpleNamespace(nominal_lot_litres=100, selected_line=None, pv_id="PV1")
    from production_wheel.business_rules import runner_ratios
    data = facts(candidate, inputs.members, compiled.config, runner_ratios(inputs.members, compiled.config))
    batch = 100 * compiled.config.canonical_factor
    assert data["group"]["pv_lot_litres"] == 100
    assert data["group"]["batch_litres"] == pytest.approx(batch)
    assert data["group"]["batch_count"] == pytest.approx(1100 / batch)
    lazy = Expression.model_validate(expression("if", expression("literal", value=True),
        expression("literal", value=1), expression("field", field="group.selected_line")))
    assert evaluate(lazy, data) == 1
    with pytest.raises(ValueError):
        Expression.model_validate(expression("if", expression("literal", value=True),
            expression("literal", value=1)))
    with pytest.raises(ValueError, match="true or false"):
        evaluate(Expression.model_validate(expression("if", expression("literal", value=1),
            expression("literal", value=1), expression("literal", value=0))), data)


def test_optimizer_and_sap_settings_agree_on_runner_classes():
    """Both stages classify a wheel's FINIs identically from the same cut-offs.

    SAP settings compute the SEFI ratio over the selected point's wheel members,
    the optimizer over the admitted block. Every point covers every admitted FINI
    exactly once, so both populations are the same set.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    from production_wheel.business_rules import facts, runner_ratios
    from production_wheel.runners import CycleStockPolicy, RunnerThresholds
    from production_wheel.sap_settings.roc import runner_adjustments

    inputs, compiled, result = solve_rules([])
    admitted = {m.fini_id for m in inputs.members}
    for point in result.points:
        covered = [fid for s in point.selected for fid in s.candidate.member_ids]
        assert sorted(covered) == sorted(admitted)

    members = tuple(replace(m, fini_id=f"F{i}", reference_coverage_source_days=days)
                    for i, (m, days) in enumerate(zip(inputs.members * 3, (8, 25, 40, 90, 300, 12))))
    thresholds = RunnerThresholds(high_max_days=10, low_min_days=80, high_max_ratio=3, low_min_ratio=10)
    config = compiled.config.model_copy(update={'runner_thresholds': thresholds})
    candidate = SimpleNamespace(nominal_lot_litres=100, selected_line=None, pv_id="PV1")
    optimizer = [r['runner'] for r in facts(candidate, members, config, runner_ratios(members, config))['members']]
    wheel = [{"sefi": m.sefi, "plant": m.plant, "material": m.fini_id, "proposed_subgroup_code": "G1",
              "reference_coverage_source_days": m.reference_coverage_source_days, "base_group_coverage_days": 8}
             for m in members]
    sap = [row["runner_class"] for row in runner_adjustments(wheel, CycleStockPolicy(), thresholds)]
    assert optimizer == ['high', 'medium', 'medium', 'low', 'low', 'high']  # 25 days is 3.1x the fastest
    assert sap == [label.capitalize() for label in optimizer]


def tank_inputs():
    """One block with a small-tank PV (100 L) and a large-tank PV (400 L) to choose from."""
    members = fixture_inputs().members
    return CanonicalInputs(Path("."), (), members,
        (ProductionVersion("P", "S", "PV1", 100), ProductionVersion("P", "S", "PV2", 400)))


def tank_capacity_rule(lo, hi, upper, rule_id="small-tank"):
    """Weekly-average lots per day on PVs with lot in (lo, hi] litres stay at most upper."""
    in_tank = expression("and", expression("gt", expression("field", field="group.pv_lot_litres"),
                                           expression("literal", value=lo)),
                         expression("lte", expression("field", field="group.pv_lot_litres"),
                                    expression("literal", value=hi)))
    return {"kind": "selection_bound", "constraint_id": rule_id, "approval_status": "approved",
            "scope": {"plant": "P"}, "upper": upper,
            "measure": expression("if", in_tank, expression("field", field="group.batches_per_day"),
                                  expression("literal", value=0))}


def solve_tanks(rules):
    """Solve the two-PV block in OPTIMIZED mode with the given capacity rules."""
    inputs = tank_inputs()
    request = SolveRequest.model_validate({"config": {"coverage_mode": "PARETO", "matrix_mode": "OFF",
                                                      "pv_mode": "OPTIMIZED"}, "constraints": rules})
    compiled = compile_request(request, inputs)
    result = run_greenfield_frontier(compiled.inputs, compiled.config, point_count=3, block_option_count=3,
                                     per_block_total_seconds=10, block_worker_count=1,
                                     pool_transform=compiled.filter_pools)
    return compiled, result


def small_tank_lots_per_day(point, config):
    """Sum of 1 / R over the selected groups of a point that use the 100 L PV."""
    from production_wheel.business_rules import facts, runner_ratios
    members = tank_inputs().members
    ratios = runner_ratios(members, config)
    total = 0.0
    for selected in point.selected:
        candidate = selected.candidate
        if candidate.nominal_lot_litres <= 100:
            group = [m for m in members if m.fini_id in candidate.member_ids]
            total += facts(candidate, group, config, ratios)["group"]["batches_per_day"]
    return total


def test_batch_frequency_facts_follow_the_whole_day_wheel_roc():
    """batches_per_day = 1 / R and batches_per_week = days per week / R for both strategies."""
    from types import SimpleNamespace
    from production_wheel.business_rules import facts, runner_ratios
    inputs = fixture_inputs()
    _, compiled, _ = solve_rules([])
    ratios = runner_ratios(inputs.members, compiled.config)
    for strategy in ("single", "runner_multiple"):
        candidate = SimpleNamespace(nominal_lot_litres=100, selected_line=None, pv_id="PV1",
                                    cycle_stock_strategy=strategy)
        group = facts(candidate, inputs.members, compiled.config, ratios)["group"]
        # 250 demand days x 90 L effective batch / 1100 L = 20.45 -> R = 20.
        assert group["wheel_roc_days"] == 20
        assert group["batches_per_day"] == pytest.approx(1 / 20)
        assert group["batches_per_week"] == pytest.approx(5 / 20)


def test_tank_capacity_bound_moves_batches_to_the_larger_tank():
    """Uncapped, the small tank runs more than 0.01 lots/day; a 0.01 cap is respected; 0.005 on all tanks is infeasible."""
    compiled, free = solve_tanks([])
    assert free.points
    assert max(small_tank_lots_per_day(p, compiled.config) for p in free.points) > 0.01
    compiled, capped = solve_tanks([tank_capacity_rule(0, 100, 0.01)])
    assert capped.points and not capped.block_failures
    assert all(small_tank_lots_per_day(p, compiled.config) <= 0.01 + 1e-9 for p in capped.points)
    _, impossible = solve_tanks([tank_capacity_rule(0, 1000, 0.005, "all-tanks")])
    assert not impossible.points
