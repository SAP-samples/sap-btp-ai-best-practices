"""Persistent baseline-free block solving and deterministic incumbent construction."""

from __future__ import annotations

import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, replace
from heapq import nsmallest
from typing import Iterable, Sequence

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition

from production_wheel.candidates import Candidate, CandidateMember, CandidatePool
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig

from .solver import (
    _LOCK_TOLERANCE,
    BlockSessionMetrics,
    CandidateCoefficients,
    HeuristicEvidence,
    ObjectiveLevel,
    SelectedCandidate,
    SolveResult,
    _candidate_member_keys,
    _configure_solver,
    _eligible_candidates,
    _infeasible_result,
    _linear,
    _member_key,
    _result_class,
    _singleton_warm_start_indexes,
    _solve_once,
    precompute_coefficients,
)

@dataclass(frozen=True, slots=True)
class _HeuristicStart:
    """Internal selected indexes plus public baseline-free construction evidence."""

    indexes: frozenset[int]
    evidence: HeuristicEvidence


def _peak_rss_bytes() -> int:
    """Return the current process peak resident-set size in bytes."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _selected_index_axes(
    indexes: Iterable[int], coefficients: Sequence[CandidateCoefficients]
) -> tuple[float, float, float]:
    """Return coverage numerator, represented demand, and ``J_CH`` for indexes."""

    selected = tuple(indexes)
    return (
        sum(
            coefficients[index].coverage_days
            * coefficients[index].group_demand_litres
            for index in selected
        ),
        sum(coefficients[index].group_demand_litres for index in selected),
        sum(coefficients[index].j_ch for index in selected),
    )


def _indexes_are_exact_cover(
    indexes: Iterable[int],
    candidates: Sequence[Candidate],
    members: Sequence[CandidateMember],
) -> bool:
    """Return whether candidate indexes cover every block FINI exactly once."""

    covered = [
        key
        for index in indexes
        for key in _candidate_member_keys(candidates[index])
    ]
    expected = [_member_key(member) for member in members]
    return len(covered) == len(set(covered)) and sorted(covered) == sorted(expected)


def _greenfield_heuristic_start(
    candidates: Sequence[Candidate],
    coefficients: Sequence[CandidateCoefficients],
    members: Sequence[CandidateMember],
    mode: CoverageMode,
    j_ch_epsilon: float | None,
) -> _HeuristicStart:
    """Construct a deterministic singleton or greedy-merge greenfield start.

    The heuristic reads only the active candidate library and coefficient
    vectors. It starts from the canonical singleton exact cover and applies
    disjoint or whole-group replacement candidates that improve coverage while
    respecting the visible ``J_CH`` cap.
    """

    started = time.perf_counter()
    singleton_indexes = _singleton_warm_start_indexes(candidates, members)
    if not singleton_indexes:
        raise ValueError("greenfield heuristic requires a singleton exact cover")
    selected = set(singleton_indexes)
    owner = {
        candidates[index].member_ids[0]: index for index in singleton_indexes
    }
    iterations = 0
    moves = 0
    if mode is not CoverageMode.GREENFIELD_OPERATIONS:
        singleton_numerators = {
            candidates[index].member_ids[0]: (
                coefficients[index].coverage_days
                * coefficients[index].group_demand_litres
            )
            for index in singleton_indexes
        }

        def initial_key(index: int) -> tuple[float, float, str]:
            """Rank a merge by baseline-free coverage gain and deterministic ties."""

            item = coefficients[index]
            numerator = item.coverage_days * item.group_demand_litres
            delta_coverage = numerator - sum(
                singleton_numerators[fini_id]
                for fini_id in candidates[index].member_ids
            )
            if j_ch_epsilon is None:
                return delta_coverage, item.j_ch, candidates[index].candidate_hash
            efficiency = delta_coverage / max(item.j_ch, 1e-12)
            return efficiency, delta_coverage, candidates[index].candidate_hash

        eligible = (
            index
            for index, candidate in enumerate(candidates)
            if len(candidate.member_ids) > 1
            and (
                j_ch_epsilon is None
                or coefficients[index].j_ch
                <= j_ch_epsilon + _LOCK_TOLERANCE * max(1.0, j_ch_epsilon)
            )
        )
        # ponytail: inspect only the strongest 10k merge candidates; raise this
        # ceiling if benchmarked incumbent quality, rather than runtime, is limiting.
        ranked = nsmallest(10_000, eligible, key=initial_key)
        current_coverage, _, current_j_ch = _selected_index_axes(
            selected, coefficients
        )
        for index in ranked:
            iterations += 1
            candidate = candidates[index]
            replaced = {owner[fini_id] for fini_id in candidate.member_ids}
            replaced_members = {
                fini_id
                for selected_index in replaced
                for fini_id in candidates[selected_index].member_ids
            }
            if replaced_members != set(candidate.member_ids):
                continue
            old_coverage, _, old_j_ch = _selected_index_axes(replaced, coefficients)
            item = coefficients[index]
            new_coverage = item.coverage_days * item.group_demand_litres
            next_coverage = current_coverage - old_coverage + new_coverage
            next_j_ch = current_j_ch - old_j_ch + item.j_ch
            if next_coverage >= current_coverage - _LOCK_TOLERANCE:
                continue
            if j_ch_epsilon is not None and next_j_ch > (
                j_ch_epsilon
                + _LOCK_TOLERANCE * max(1.0, abs(j_ch_epsilon))
            ):
                continue
            selected.difference_update(replaced)
            selected.add(index)
            for fini_id in candidate.member_ids:
                owner[fini_id] = index
            current_coverage = next_coverage
            current_j_ch = next_j_ch
            moves += 1
            if moves >= len(members) - 1:
                break
    coverage, demand, j_ch = _selected_index_axes(selected, coefficients)
    exact_cover = _indexes_are_exact_cover(selected, candidates, members)
    within_epsilon = j_ch_epsilon is None or j_ch <= (
        j_ch_epsilon + _LOCK_TOLERANCE * max(1.0, abs(j_ch_epsilon))
    )
    if not exact_cover or not within_epsilon:
        raise ValueError("greenfield heuristic produced an invalid start")
    return _HeuristicStart(
        frozenset(selected),
        HeuristicEvidence(
            kind=(
                "greenfield_singleton_j_ch"
                if mode is CoverageMode.GREENFIELD_OPERATIONS
                else "greenfield_greedy_merge"
            ),
            iterations=iterations,
            moves=moves,
            construction_seconds=time.perf_counter() - started,
            coverage_numerator=coverage,
            demand_weighted_mean_coverage_days=coverage / demand,
            j_ch=j_ch,
            exact_cover_valid=exact_cover,
            within_epsilon=within_epsilon,
        ),
    )


class BlockSolverSession:
    """Reuse one APPSI HiGHS exact-cover model across greenfield frontier solves.

    Args:
        pool: One deterministic plant/SEFI candidate library.
        members: Canonical FINIs for exactly that block.
        config: Greenfield structural and solver settings.

    The class transfers the structural model once. Objectives, the visible
    epsilon row, and temporary lexicographic locks are updated explicitly.
    HiGHS receives prior or heuristic variable values as MIP starts, but no
    branch-and-bound tree reuse is claimed.
    """

    def __init__(
        self,
        pool: CandidatePool,
        members: Iterable[CandidateMember],
        config: RunConfig,
    ) -> None:
        """Build and transfer one baseline-independent block model."""

        if not config.is_greenfield:
            raise ValueError("persistent block sessions require a greenfield mode")
        ordered_members = tuple(sorted(members, key=_member_key))
        if not ordered_members or {
            member.block_key for member in ordered_members
        } != {pool.block_key}:
            raise ValueError("persistent block session members must match one pool")
        if pool.config_fingerprint != config.structural_ruleset_fingerprint():
            raise ValueError("candidate pool uses different structural settings")
        self.pool = pool
        self.members = ordered_members
        self.config = config
        tick = time.perf_counter()
        self.candidates = _eligible_candidates((pool,), config)
        self._candidate_order_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        self.coefficients = precompute_coefficients(
            self.candidates, ordered_members, config
        )
        self._coefficient_preparation_seconds = time.perf_counter() - tick
        member_keys = tuple(_member_key(member) for member in ordered_members)
        incidence: dict[tuple[str, str, str], list[int]] = {
            key: [] for key in member_keys
        }
        for index, candidate in enumerate(self.candidates):
            for key in _candidate_member_keys(candidate):
                incidence[key].append(index)
        if any(not indexes for indexes in incidence.values()):
            raise ValueError("persistent block session has an uncovered FINI")
        tick = time.perf_counter()
        model = pyo.ConcreteModel(name="persistent_greenfield_block")
        model.C = pyo.RangeSet(0, len(self.candidates) - 1)
        model.x = pyo.Var(model.C, domain=pyo.Binary)
        singleton_start = _singleton_warm_start_indexes(
            self.candidates, ordered_members
        )
        if not singleton_start:
            raise ValueError("persistent block session requires singleton candidates")
        for index in model.C:
            model.x[index].value = float(index in singleton_start)
        model.cover = pyo.ConstraintList()
        for key in member_keys:
            model.cover.add(
                pyo.quicksum(model.x[index] for index in incidence[key]) == 1
            )
        coverage_values = [
            item.coverage_days * item.group_demand_litres
            for item in self.coefficients
        ]
        model.coverage_expression = pyo.Expression(
            expr=_linear(coverage_values, model)
        )
        model.j_ch_expression = pyo.Expression(
            expr=_linear([item.j_ch for item in self.coefficients], model)
        )
        model.matrix_exception_expression = pyo.Expression(
            expr=_linear(
                [item.matrix_exceptions for item in self.coefficients], model
            )
        )
        model.relaxed_group_expression = pyo.Expression(
            expr=_linear([item.relaxed_group for item in self.coefficients], model)
        )
        model.size_excess_expression = pyo.Expression(
            expr=_linear([item.size_excess for item in self.coefficients], model)
        )
        model.epsilon_limit = pyo.Param(mutable=True, initialize=0.0)
        model.pareto_epsilon = pyo.Constraint(
            expr=model.j_ch_expression <= model.epsilon_limit
        )
        model.pareto_epsilon.deactivate()
        model.lexicographic_locks = pyo.ConstraintList()
        model.objective = pyo.Objective(
            expr=model.coverage_expression, sense=pyo.minimize
        )
        self.model = model
        self._model_construction_seconds = time.perf_counter() - tick
        self.solver = _configure_solver(config)
        tick = time.perf_counter()
        self.solver.set_instance(model)
        self._solver_transfer_seconds = time.perf_counter() - tick
        for name in self.solver.update_config:
            setattr(self.solver.update_config, name, False)
        self._epsilon_loaded = False
        self._solver_update_seconds = 0.0
        self._solver_optimize_seconds = 0.0
        self._solver_call_count = 0
        self._candidate_index = {
            candidate.candidate_hash: index
            for index, candidate in enumerate(self.candidates)
        }
        self._total_demand = sum(member.demand_litres for member in ordered_members)

    @property
    def metrics(self) -> BlockSessionMetrics:
        """Return a cumulative immutable instrumentation snapshot."""

        return BlockSessionMetrics(
            candidate_order_seconds=self._candidate_order_seconds,
            coefficient_preparation_seconds=self._coefficient_preparation_seconds,
            model_construction_seconds=self._model_construction_seconds,
            solver_transfer_seconds=self._solver_transfer_seconds,
            solver_update_seconds=self._solver_update_seconds,
            solver_optimize_seconds=self._solver_optimize_seconds,
            model_build_count=1,
            solver_call_count=self._solver_call_count,
            peak_rss_bytes=_peak_rss_bytes(),
        )

    def _set_epsilon(self, epsilon: float | None) -> None:
        """Activate, update, or remove the one visible ``J_CH`` epsilon row."""

        tick = time.perf_counter()
        if epsilon is None:
            if self._epsilon_loaded:
                self.solver.remove_constraints([self.model.pareto_epsilon])
                self.model.pareto_epsilon.deactivate()
                self._epsilon_loaded = False
        else:
            tolerance = _LOCK_TOLERANCE * max(1.0, abs(epsilon))
            self.model.epsilon_limit.set_value(epsilon + tolerance)
            if not self._epsilon_loaded:
                self.model.pareto_epsilon.activate()
                self.solver.add_constraints([self.model.pareto_epsilon])
                self._epsilon_loaded = True
            else:
                self.solver.update_params()
        self._solver_update_seconds += time.perf_counter() - tick

    def _add_lock(self, expression: object, optimum: float) -> None:
        """Add one temporary scale-aware lexicographic lock to Pyomo and HiGHS."""

        tick = time.perf_counter()
        tolerance = _LOCK_TOLERANCE * max(1.0, abs(optimum))
        constraint = self.model.lexicographic_locks.add(
            expression <= optimum + tolerance
        )
        self.solver.add_constraints([constraint])
        self._solver_update_seconds += time.perf_counter() - tick

    def _clear_locks(self) -> None:
        """Remove every point-local lock so no constraint leaks to another point."""

        constraints = list(self.model.lexicographic_locks.values())
        if not constraints:
            return
        tick = time.perf_counter()
        self.solver.remove_constraints(constraints)
        self.model.lexicographic_locks.clear()
        self._solver_update_seconds += time.perf_counter() - tick

    def _solve_expression(self, expression: object) -> _SolveSnapshot:
        """Update the objective explicitly and solve the persistent instance."""

        tick = time.perf_counter()
        self.model.objective.set_value(expression)
        self.solver.set_objective(self.model.objective)
        self._solver_update_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        snapshot = _solve_once(self.model, self.solver)
        self._solver_optimize_seconds += time.perf_counter() - tick
        self._solver_call_count += 1
        return snapshot

    def _previous_start(
        self,
        candidate_hashes: Iterable[str],
        epsilon: float | None,
    ) -> frozenset[int]:
        """Return a valid prior incumbent index set for a nearby solve."""

        hashes = tuple(candidate_hashes)
        if not hashes or any(value not in self._candidate_index for value in hashes):
            return frozenset()
        indexes = frozenset(self._candidate_index[value] for value in hashes)
        if not _indexes_are_exact_cover(indexes, self.candidates, self.members):
            return frozenset()
        _, _, j_ch = _selected_index_axes(indexes, self.coefficients)
        if epsilon is not None and j_ch > (
            epsilon + _LOCK_TOLERANCE * max(1.0, abs(epsilon))
        ):
            return frozenset()
        return indexes

    def solve(
        self,
        mode: CoverageMode,
        *,
        j_ch_epsilon: float | None = None,
        warm_start_candidate_hashes: Iterable[str] = (),
    ) -> SolveResult:
        """Solve one anchor or epsilon point without rebuilding block structure.

        Args:
            mode: ``GREENFIELD_COVERAGE``, ``GREENFIELD_OPERATIONS``, or ``PARETO``.
            j_ch_epsilon: Required only for ``PARETO``.
            warm_start_candidate_hashes: Optional previous feasible partition.

        Returns:
            Selected exact-cover groups plus proof, heuristic, and runtime evidence.
        """

        if mode not in {
            CoverageMode.GREENFIELD_COVERAGE,
            CoverageMode.GREENFIELD_OPERATIONS,
            CoverageMode.PARETO,
        }:
            raise ValueError("unsupported persistent greenfield objective")
        if mode is CoverageMode.PARETO:
            if (
                j_ch_epsilon is None
                or not math.isfinite(j_ch_epsilon)
                or j_ch_epsilon < 0
            ):
                raise ValueError("PARETO mode requires a finite non-negative J_CH epsilon")
        elif j_ch_epsilon is not None:
            raise ValueError("J_CH epsilon is valid only in PARETO mode")
        self._set_epsilon(j_ch_epsilon)
        heuristic_mode = (
            CoverageMode.GREENFIELD_OPERATIONS
            if mode is CoverageMode.GREENFIELD_OPERATIONS
            else CoverageMode.GREENFIELD_COVERAGE
        )
        heuristic = _greenfield_heuristic_start(
            self.candidates,
            self.coefficients,
            self.members,
            heuristic_mode,
            j_ch_epsilon,
        )
        previous = self._previous_start(
            warm_start_candidate_hashes, j_ch_epsilon
        )
        start = heuristic.indexes
        warm_start_kind = heuristic.evidence.kind
        if previous:
            previous_axes = _selected_index_axes(previous, self.coefficients)
            heuristic_axes = _selected_index_axes(start, self.coefficients)
            previous_key = (
                (previous_axes[2], previous_axes[0])
                if mode is CoverageMode.GREENFIELD_OPERATIONS
                else (previous_axes[0], previous_axes[2])
            )
            heuristic_key = (
                (heuristic_axes[2], heuristic_axes[0])
                if mode is CoverageMode.GREENFIELD_OPERATIONS
                else (heuristic_axes[0], heuristic_axes[2])
            )
            if previous_key <= heuristic_key:
                start = previous
                warm_start_kind = "previous_incumbent"
        for index in self.model.C:
            self.model.x[index].value = float(index in start)
        levels: list[ObjectiveLevel] = []
        last: _SolveSnapshot | None = None
        stopped = False

        def solve_and_lock(
            name: str, expression: object, divisor: float = 1.0
        ) -> bool:
            """Solve, report, and temporarily lock one exact tier."""

            nonlocal last
            last = self._solve_expression(expression)
            if not last.feasible:
                return False
            optimum = float(last.incumbent)
            levels.append(
                ObjectiveLevel(
                    name=name,
                    value=optimum / divisor,
                    termination_condition=last.tier_termination,
                    best_bound=(
                        None if last.bound is None else last.bound / divisor
                    ),
                    relative_gap=last.gap,
                    wallclock_seconds=last.wallclock_seconds,
                )
            )
            if not last.optimal:
                return False
            self._add_lock(expression, optimum)
            return True

        try:
            if (
                self.config.matrix_mode is MatrixMode.DIAGNOSTIC
                and any(item.matrix_exceptions for item in self.coefficients)
            ):
                stopped = not solve_and_lock(
                    "matrix_exception_pairs",
                    self.model.matrix_exception_expression,
                )
            business = (
                (
                    "j_ch",
                    self.model.j_ch_expression,
                    1.0,
                ),
                (
                    "demand_weighted_mean_coverage_days",
                    self.model.coverage_expression,
                    self._total_demand,
                ),
            ) if mode is CoverageMode.GREENFIELD_OPERATIONS else (
                (
                    "demand_weighted_mean_coverage_days",
                    self.model.coverage_expression,
                    self._total_demand,
                ),
                (
                    "j_ch",
                    self.model.j_ch_expression,
                    1.0,
                ),
            )
            for name, expression, divisor in business:
                if stopped or not solve_and_lock(name, expression, divisor):
                    stopped = True
                    break
            post_business = (
                (
                    "relaxed_group_count",
                    self.model.relaxed_group_expression,
                    any(item.relaxed_group for item in self.coefficients),
                ),
                (
                    "group_size_total_excess",
                    self.model.size_excess_expression,
                    any(item.size_excess for item in self.coefficients),
                ),
            )
            if not stopped:
                for name, expression, required in post_business:
                    if not required:
                        levels.append(ObjectiveLevel(name, 0.0, "not_required"))
                    elif not solve_and_lock(name, expression):
                        stopped = True
                        break
            if (
                self.config.matrix_mode is MatrixMode.DIAGNOSTIC
                and not any(item.matrix_exceptions for item in self.coefficients)
            ):
                levels.append(
                    ObjectiveLevel("matrix_exception_pairs", 0.0, "not_required")
                )
            if last is None or not last.feasible:
                result = _infeasible_result(
                    completeness=self.pool.completeness,
                    candidate_count=len(self.candidates),
                    status=(
                        "infeasible"
                        if last is not None
                        and last.termination
                        in {
                            TerminationCondition.infeasible,
                            TerminationCondition.infeasibleOrUnbounded,
                        }
                        else "no_incumbent"
                    ),
                    termination=(
                        "unknown" if last is None else last.termination.name
                    ),
                    warm_start_kind=warm_start_kind,
                    warm_start_group_count=len(start),
                    epsilon_j_ch=j_ch_epsilon,
                )
                return replace(
                    result,
                    heuristic_evidence=heuristic.evidence,
                    session_metrics=self.metrics,
                )
            selected_indexes = tuple(
                index
                for index, value in sorted(last.primals.items())
                if value > 0.5
            )
            selected = tuple(
                SelectedCandidate(self.candidates[index], self.coefficients[index])
                for index in selected_indexes
            )
            levels.append(
                ObjectiveLevel(
                    "stable_candidate_rank_diagnostic",
                    float(
                        sum(
                            self.coefficients[index].stable_rank
                            for index in selected_indexes
                        )
                    ),
                    "deterministic_selection",
                )
            )
            final_coverage, _, final_j_ch = _selected_index_axes(
                selected_indexes, self.coefficients
            )
            solver_improved = (
                final_j_ch < heuristic.evidence.j_ch - _LOCK_TOLERANCE
                if mode is CoverageMode.GREENFIELD_OPERATIONS
                else final_coverage
                < heuristic.evidence.coverage_numerator - _LOCK_TOLERANCE
            )
            heuristic_evidence = HeuristicEvidence(
                **{
                    **asdict(heuristic.evidence),
                    "solver_improved": solver_improved,
                }
            )
            optimal = last.optimal and not stopped
            primary = levels[0] if levels else None
            return SolveResult(
                status="optimal" if optimal else "feasible_limit",
                termination_condition=last.tier_termination,
                has_incumbent=True,
                result_class=_result_class(
                    self.pool.completeness,
                    optimal,
                    last.gap,
                ),
                selected=selected,
                objective_levels=tuple(levels),
                primary_objective=primary.value if primary else None,
                primary_best_bound=primary.best_bound if primary else None,
                primary_relative_gap=primary.relative_gap if primary else None,
                incumbent_objective=last.incumbent,
                best_bound=last.bound,
                relative_gap=0.0 if optimal else last.gap,
                candidate_count=len(self.candidates),
                pool_completeness=self.pool.completeness,
                solver_method="pyomo_appsi_highs_persistent_block",
                warm_start_kind=warm_start_kind,
                warm_start_group_count=len(start),
                epsilon_j_ch=j_ch_epsilon,
                heuristic_evidence=heuristic_evidence,
                session_metrics=self.metrics,
            )
        finally:
            self._clear_locks()

