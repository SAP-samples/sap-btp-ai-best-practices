"""Fail-closed profile matrix resolution and independent rule audit checks."""

import pytest
from production_wheel.matrix import configured_matrix, matrix_status_lookup
from production_wheel.rule_models import MatrixPair
from production_wheel.schemas import MatrixMode, RunConfig


def _config(pairs):
    """Return a run configuration with explicit source matrix pairs."""
    return RunConfig(matrix_pairs=tuple(MatrixPair(volume_a=a, volume_b=b, status=s) for a,b,s in pairs))


def test_profile_matrix_normalizes_and_expands_symmetric_pairs():
    """A triangular input becomes a complete ordered matrix with unchanged N/AVOID."""
    from types import SimpleNamespace
    config = _config([('1.0','1','Y'),('2','2','Y'),('1','2','N')])
    lookup = matrix_status_lookup(configured_matrix(config, [SimpleNamespace(package_volume=v) for v in (1,2)]))
    assert len(lookup) == 4
    assert lookup[(1,2)] == lookup[(2,1)] == 'N'


@pytest.mark.parametrize('pairs', [
    [('1','1','Y'),('2','2','Y')],
    [('1','1','Y'),('2','2','Y'),('1','2','Y'),('2','1','N')],
])
def test_profile_matrix_rejects_missing_and_conflicting_pairs(pairs):
    """Incomplete or contradictory profile data cannot quietly use built-in pairs."""
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        configured_matrix(_config(pairs), [SimpleNamespace(package_volume=v) for v in (1,2)])


def _solved_pair():
    """Create a valid joined pair with a legacy OFF configuration for mutation checks."""
    from production_wheel.candidates import CandidateMember, ProductionVersion, generate_candidate_pools
    from production_wheel.optimization import solve_candidate_pools
    members = tuple(CandidateMember(fini_id=f'F{index}', plant='P1', sefi='S1',
        eligible_lines=frozenset({'L1'}), demand_litres=1000, pallet_litres=100,
        package_volume=volume, fixed_pv='PV1', pck_code='PCK') for index,volume in enumerate((1,2)))
    versions = (ProductionVersion('P1','S1','PV1',1000),)
    config = RunConfig(matrix_mode='OFF')
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    assert len(result.selected) == 1
    return members, versions, pools, result


@pytest.mark.parametrize('mode', ['HARD', 'FLEXIBLE', 'OFF'])
def test_independent_audit_rejects_n_even_when_solver_claims_success(mode):
    """Tampered profile prohibitions are rediscovered from members in every mode."""
    from production_wheel.solution_validation import validate_solution
    members, versions, pools, result = _solved_pair()
    config = _config([('1','1','Y'),('2','2','Y'),('1','2','N')]).model_copy(update={'matrix_mode': MatrixMode(mode)})
    validation = validate_solution(result, members, pools, config, versions)
    assert not validation.is_valid
    assert 'PROFILE_MATRIX_PROHIBITION' in {issue.rule_id for issue in validation.issues}


def test_independent_audit_rechecks_group_and_selection_rules_and_line():
    """Reject altered group claims, aggregate bounds and ineligible selected lines."""
    from dataclasses import replace
    from production_wheel.rule_models import GroupRule, SelectionBound, Expression
    from production_wheel.solution_validation import validate_solution
    members, versions, pools, result = _solved_pair()
    result = replace(result, selected=tuple(replace(item, candidate=replace(item.candidate, selected_line='L9')) for item in result.selected))
    config = RunConfig(matrix_mode='OFF', assign_filling_lines=True, business_rules=(
        GroupRule(constraint_id='singleton', approval_status='approved', assertion=Expression(op='lte', args=(Expression(op='field',field='group.size'), Expression(op='literal',value=1)))),
        SelectionBound(constraint_id='min_groups', approval_status='approved', measure=Expression(op='literal',value=1), lower=2),
    ))
    validation = validate_solution(result, members, pools, config, versions)
    assert {'SELECTED_LINE_ELIGIBLE','singleton','min_groups'} <= {issue.rule_id for issue in validation.issues}
    assert validation.groups[0].selected_line == 'L9'


@pytest.mark.parametrize('mode', ['FLEXIBLE', 'OFF'])
def test_profile_avoid_is_allowed_and_reported(mode):
    """Valid profile AVOID pairs retain selected-line and matrix provenance in outputs."""
    from production_wheel.candidates import generate_candidate_pools
    from production_wheel.optimization import solve_candidate_pools
    from production_wheel.reporting import build_matrix_exception_audit_rows, build_solution_groups_rows
    from production_wheel.schemas import MatrixMode
    from production_wheel.solution_validation import validate_solution
    members, versions, _, _ = _solved_pair()
    config = _config([('1','1','Y'),('2','2','Y'),('1','2','AVOID')]).model_copy(update={'matrix_mode':MatrixMode(mode), 'assign_filling_lines': True})
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)
    assert validation.is_valid, validation.issues
    audit = build_matrix_exception_audit_rows(validation.groups, config, members)
    assert len(audit) == 1 and audit[0]['status'] == 'AVOID'
    assert audit[0]['matrix_version'] == 'PLANT_PROFILE_MATRIX_V1'
    assert build_solution_groups_rows(validation.groups, result, config)[0]['selected_line'] == 'L1'


def test_baseline_profile_prohibition_is_infeasible_with_off_mode():
    """Historical comparison cannot label a forbidden pair feasible just because OFF."""
    from dataclasses import replace
    from production_wheel.reporting import build_baseline_summary
    members, versions, _, _ = _solved_pair()
    members = tuple(replace(member, baseline_group='B1') for member in members)
    config = _config([('1','1','Y'),('2','2','Y'),('1','2','N')]).model_copy(update={'matrix_mode': MatrixMode.OFF})
    summary = build_baseline_summary(members, versions, config)
    assert summary['feasible_under_active_matrix'] == 0


def test_constraint_audit_includes_frozen_business_rules():
    """Business rules carry their interpreted source and approval evidence to reports."""
    from production_wheel.reporting import build_constraint_audit_rows
    from production_wheel.rule_models import GroupRule, Expression
    rule = GroupRule(constraint_id='size', approval_status='approved', source_text='Only singletons', assertion=Expression(op='literal',value=True))
    config = RunConfig(business_rules=(rule,))
    audit = build_constraint_audit_rows(config)
    evidence = next(row for row in audit if row['constraint_id'] == 'size')
    assert evidence['source_text'] == 'Only singletons'
    assert evidence['approval_status'] == 'approved'


def test_independent_audit_rejects_avoid_in_hard_mode():
    """A stale solver result cannot bypass a profile HARD preference restriction."""
    from production_wheel.solution_validation import validate_solution
    members, versions, pools, result = _solved_pair()
    config = _config([('1','1','Y'),('2','2','Y'),('1','2','AVOID')]).model_copy(update={'matrix_mode': MatrixMode.HARD})
    validation = validate_solution(result, members, pools, config, versions)
    assert 'HARD_MATRIX_COMPATIBILITY' in {issue.rule_id for issue in validation.issues}


@pytest.mark.parametrize('maximum,expected_valid', [(8, True), (3, False)])
def test_independent_audit_uses_scoped_group_cap(maximum, expected_valid):
    """Honor profile block caps above seven and reject stale groups above lower caps."""
    from dataclasses import replace
    from production_wheel.candidates import generate_candidate_pools
    from production_wheel.optimization import solve_candidate_pools
    from production_wheel.schemas import GroupSizeOverride
    from production_wheel.solution_validation import validate_solution
    source, versions, _, _ = _solved_pair()
    members = tuple(replace(source[0], fini_id=str(index)) for index in range(8))
    config = RunConfig(matrix_mode='OFF', group_size_overrides=(
        GroupSizeOverride(plant='P1', sefi='S1', maximum=8),))
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    assert [len(item.candidate.member_ids) for item in result.selected] == [8]
    audit_config = config.model_copy(update={'group_size_overrides': (
        GroupSizeOverride(plant='P1', sefi='S1', maximum=maximum),)})
    validation = validate_solution(result, members, pools, audit_config, versions)
    assert validation.is_valid is expected_valid
    assert ('GROUP_SIZE_CAP' in {issue.rule_id for issue in validation.issues}) is not expected_valid
