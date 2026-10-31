"""Regressions for the reported singleton-only profile frontier."""

from dataclasses import replace

from test_business_rules import expression, solve_rules
from production_wheel.pareto import _nondominated_points


def test_local_profile_rules_reuse_block_search_and_find_grouped_options():
    """A group-local cap keeps the existing greedy start and independent block search."""
    rule = {'kind': 'group_rule', 'constraint_id': 'cap', 'approval_status': 'approved',
            'assertion': expression('lte', expression('field', field='group.size'),
                                    expression('literal', value=2))}
    _, _, result = solve_rules([rule], blocks=('S', 'T'), assign_filling_lines=True)
    assert result.block_execution_mode != 'joint_candidate_master'
    assert any(len(point.selected) == 2 for point in result.points)
    assert result.block_session_audits


def test_same_kpi_alternative_assignments_are_one_frontier_tradeoff():
    """Different line assignments with equal axes must not create duplicate choices."""
    _, _, result = solve_rules([])
    point = result.points[0]
    alternate = replace(point, portfolio_hash='different-lines-same-kpis')
    assert len(_nondominated_points([point, alternate])) == 1


def test_identical_joint_anchors_do_not_repeat_zero_epsilon_solves():
    """A policy forcing singletons has one tradeoff and no interior epsilon interval."""
    rule = {'kind': 'selection_bound', 'constraint_id': 'two-groups', 'approval_status': 'approved',
            'measure': expression('literal', value=1), 'lower': 2, 'upper': 2}
    _, _, result = solve_rules([rule])
    assert len(result.points) == 1
    assert not result.global_solve_audits
