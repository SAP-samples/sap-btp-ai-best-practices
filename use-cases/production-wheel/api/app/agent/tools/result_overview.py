"""Compact persisted frontier evidence and deterministic baseline comparisons."""

from app.workspace.run_views import result_metadata_summary, run_status

POINT_FIELDS = (
    'point_index', 'demand_weighted_mean_coverage_days', 'j_ch', 'group_count',
    'singleton_group_count', 'modeled_fini_count', 'modeled_demand_litres',
    'proof_scope', 'validation_status', 'validation_error_count',
    'candidate_pool_completeness', 'operations_anchor_match', 'coverage_anchor_match',
    'baseline_demand_weighted_mean_coverage_days', 'baseline_j_ch',
    'baseline_group_count', 'baseline_singleton_group_count',
    'baseline_constraint_scope', 'acceptance_status',
)


def result_overview(result: dict) -> dict:
    """Return small point records and exact comparison counts from persisted results.

    Input is WorkspaceService.results output. Detailed audits remain queryable
    through query_optimizer_data; repeated per-block solver evidence is omitted.
    A baseline comparison is emitted only when every point has identical stored
    baseline metrics. Lower coverage and lower Changeover are both preferred.
    """
    points = [{k: p[k] for k in POINT_FIELDS if k in p} for p in result['points']]
    metadata = result_metadata_summary(result['metadata'])
    run = {key: value for key, value in run_status(result['run']).items()
           if key != 'configuration' and value is not None}
    overview = {'run': run, 'points': points, 'metadata': metadata}
    baseline_keys = ('baseline_demand_weighted_mean_coverage_days', 'baseline_j_ch')
    if not points or any(p.get(k) is None for p in points for k in baseline_keys):
        return overview
    baseline = tuple(points[0][k] for k in baseline_keys)
    if any(tuple(p[k] for k in baseline_keys) != baseline for p in points):
        return overview
    coverage, changeover = baseline
    below_coverage, below_changeover, dominates, dominated = [], [], [], []
    for p in points:
        index, c, j = p['point_index'], p['demand_weighted_mean_coverage_days'], p['j_ch']
        if c < coverage:
            below_coverage.append(index)
        if j < changeover:
            below_changeover.append(index)
        if c <= coverage and j <= changeover and (c < coverage or j < changeover):
            dominates.append(index)
        if c >= coverage and j >= changeover and (c > coverage or j > changeover):
            dominated.append(index)
    overview['baseline_comparison'] = {
        'source': 'baseline metrics persisted with this run, using its snapshot and config',
        'population_basis': 'Baseline KPIs are recomputed from modeled FINIs carrying historical groups. The extraction legacy-assignment count also includes unmodeled evidence and is not the KPI population.',
        'objective_direction': 'Both coverage days and j_ch are minimized; lower on both is dominance.',
        'coverage_days': coverage, 'j_ch': changeover,
        'point_count': len(points),
        'below_baseline_coverage': {'count': len(below_coverage), 'point_indices': below_coverage},
        'below_baseline_j_ch': {'count': len(below_changeover), 'point_indices': below_changeover},
        'dominates_baseline': {'count': len(dominates), 'point_indices': dominates},
        'dominated_by_baseline': {'count': len(dominated), 'point_indices': dominated},
        'tradeoff_count': len(points) - len(dominates) - len(dominated)
            - sum(p['demand_weighted_mean_coverage_days'] == coverage and p['j_ch'] == changeover for p in points),
    }
    return overview
