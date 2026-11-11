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


def test_reference_boundary_has_no_second_factor_and_changes_fingerprint():
    """15 days is high, immediately above is low; altered evidence invalidates pools."""
    from dataclasses import replace
    from production_wheel.candidates import generate_candidate_pools
    from production_wheel.business_rules import facts
    from production_wheel.schemas import RunConfig
    inputs = fixture_inputs()
    config = RunConfig(matrix_mode='OFF', business_rules=({'kind': 'group_rule',
        'constraint_id': 'positive-size', 'approval_status': 'approved',
        'assertion': expression('gt', expression('field', field='group.size'), expression('literal', value=0))},))
    pools = generate_candidate_pools(inputs.members, inputs.production_versions, config)
    singleton = next(c for c in pools[0].candidates if c.member_ids == ('S0',))
    assert facts(singleton, inputs.members[:1], config)['members'][0]['runner'] == 'high'
    changed = (replace(inputs.members[0], lot_size_considered_litres=60.000004), *inputs.members[1:])
    assert facts(singleton, changed[:1], config)['members'][0]['runner'] == 'low'
    assert facts(singleton, inputs.members[:1], config.model_copy(update={'runner_basis': 'candidate_pv'}))['members'][0]['runner'] == 'low'
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
