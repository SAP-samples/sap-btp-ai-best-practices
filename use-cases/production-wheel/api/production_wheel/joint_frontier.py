"""Sample constrained Pareto points using the existing joint candidate master."""

from dataclasses import replace
import time

from production_wheel.optimization import solve_candidate_pools
from production_wheel.schemas import CoverageMode


def build_joint_frontier(pools, members, config, *, point_count=17,
                         total_seconds=None, stage_seconds=None, exponent=2.0, progress=None):
    """Solve coupled rules without assuming independent block optima or singletons.

    Inputs are generated libraries and canonical members. Outputs retain each
    selected block partition for the normal result bundle. Proof applies to the
    complete/restricted joint library, never to a fabricated block decomposition.
    """
    from production_wheel.pareto import (GreenfieldFrontierResult, ParetoPoint,
        GreenfieldBlockFailure, GreenfieldGlobalSolveAudit, _block_option,
        _portfolio_hash, _selected_axes, _nondominated_points, _low_epsilon_budgets)

    started = time.monotonic()
    deadline = started + (total_seconds or config.solver_limits.suite_time_limit_seconds)
    points, options, audits = [], {}, []
    preferred_minimum = None

    def solve(mode, epsilon=None):
        """Run one bounded joint solve, preserving all frozen rule settings."""
        remaining = max(deadline - time.monotonic(), 0.001)
        # The existing master solves at most six objective tiers. Bound each
        # stage within the remaining request budget; preprocessing is additional.
        limit = min(stage_seconds or config.solver_limits.time_limit_seconds, remaining / 6)
        runtime = config.model_copy(update={"coverage_mode": mode,
            "solver_limits": config.solver_limits.model_copy(update={"time_limit_seconds": limit})})
        return solve_candidate_pools(pools, members, runtime, j_ch_epsilon=epsilon,
            preferred_line_minimum=preferred_minimum)

    def retain(solved, source, epsilon=None):
        """Record selected definitions, actual KPI axes and solver proof evidence."""
        if not solved.has_incumbent:
            return None
        proof = ("runtime-limited" if solved.status != "optimal" else
            "exact" if all(p.completeness == "complete" for p in pools) else "restricted-library")
        selected_options = []
        for pool in pools:
            selected = tuple(s for s in solved.selected if s.candidate.block_key == pool.block_key)
            option = _block_option(pool, replace(solved, selected=selected), source)
            option = replace(option, proof_scope=proof)
            options[(option.block_key, option.partition_hash)] = option
            selected_options.append(option)
        numerator, demand, j_ch = _selected_axes(solved.selected)
        identity = _portfolio_hash(selected_options)
        classified = replace(solved, result_class=f"greenfield-{proof}",
            solver_method="pyomo_appsi_highs_joint_candidate_master")
        point = ParetoPoint(point_index=len(points) + 1, epsilon_j_ch=epsilon,
            status=solved.status, result_class=classified.result_class,
            selected=solved.selected, maximum_coverage_days=max(s.coefficients.coverage_days for s in solved.selected),
            j_ch=j_ch, portfolio_hash=identity, source_scenarios=(source,),
            termination_condition=solved.termination_condition, relative_gap=solved.relative_gap,
            solver_method=classified.solver_method, frontier_label="greenfield_frontier_over_joint_candidate_library",
            demand_weighted_mean_coverage_days=numerator / demand, proof_scope=proof,
            block_option_count=len(selected_options), solve_result=classified)
        points.append(point)
        if progress:
            progress("joint_frontier", len(points), point_count)
        return point

    coverage_solve = solve(CoverageMode.GREENFIELD_COVERAGE)
    coverage = retain(coverage_solve, "joint_coverage_anchor")
    if coverage and config.preferred_line:
        preferred_minimum = sum(s.candidate.group_demand_litres for s in coverage.selected
            if s.candidate.selected_line == config.preferred_line)
    operations = retain(solve(CoverageMode.GREENFIELD_OPERATIONS), "joint_operations_anchor")
    if coverage and operations and coverage.j_ch > operations.j_ch + 1e-7:
        for index, epsilon in enumerate(_low_epsilon_budgets(operations.j_ch, coverage.j_ch, point_count, exponent)[1:-1], 1):
            if time.monotonic() >= deadline:
                break
            point = retain(solve(CoverageMode.PARETO, epsilon), f"joint_epsilon_{index}", epsilon)
            if point:
                audits.append(GreenfieldGlobalSolveAudit(index, index / (point_count - 1), exponent,
                    epsilon, point.j_ch, point.demand_weighted_mean_coverage_days,
                    point.portfolio_hash, point.status, point.termination_condition,
                    point.solve_result.primary_relative_gap, point.relative_gap, point.proof_scope))
    retained = _nondominated_points(points)
    hashes = {p.portfolio_hash for p in retained}
    audits = tuple(replace(a, retained_nondominated=a.portfolio_hash in hashes) for a in audits)
    failures = () if retained else (GreenfieldBlockFailure(("*", "*"),
        sum(len(p.candidates) for p in pools), coverage_solve.status,
        f"Joint rule master has no incumbent: {coverage_solve.termination_condition}"),)
    return GreenfieldFrontierResult(config, tuple(pools), tuple(members), tuple(options.values()), retained,
        coverage.portfolio_hash if coverage else "", operations.portfolio_hash if operations else "",
        time.monotonic() - started, per_block_stage_seconds=stage_seconds,
        per_block_total_seconds=total_seconds, block_failures=failures,
        block_execution_mode="joint_candidate_master", global_epsilon_exponent=exponent,
        global_solve_audits=audits)
