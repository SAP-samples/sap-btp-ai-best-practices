"""Tests for the independent HiGHS result checks and the pinned solver version.

Run from api/:
    ../.venv/bin/python -m pytest tests/test_solver_integrity.py -q
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import pyomo.environ as pyo
import pytest
from pyomo.common.collections import ComponentMap
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from app.workspace.run_failures import classify_failure
from production_wheel.optimization.solver_integrity import (
    SolverResultInconsistentError,
    installed_highspy_version,
    verify_solver_result,
)

# Synthetic exact-cover instance (six products, eleven groups, singleton MIP
# start). highspy 1.16.0 returns the start with objective 54 although its real
# objective is 168; the true optimum is 59 (the six-product group).
_GROUPS = [
    (0,), (1,), (2,), (3,), (4,), (5,),
    (0, 1, 4, 5), (0, 1, 2, 3, 5), (0, 1, 2, 3, 4, 5), (0, 1, 3, 5), (1, 5),
]
_COSTS = [57, 26, 29, 23, 2, 31, 83, 68, 59, 68, 4]
_MEMBERS = 6


def _exact_cover_model() -> pyo.ConcreteModel:
    """Build the synthetic exact-cover model with a singleton MIP start."""

    model = pyo.ConcreteModel()
    model.C = pyo.RangeSet(0, len(_GROUPS) - 1)
    model.x = pyo.Var(model.C, domain=pyo.Binary)
    model.cover = pyo.ConstraintList()
    for member in range(_MEMBERS):
        model.cover.add(
            sum(model.x[i] for i, group in enumerate(_GROUPS) if member in group) == 1
        )
    model.objective = pyo.Objective(
        expr=sum(cost * model.x[i] for i, cost in enumerate(_COSTS))
    )
    for i, group in enumerate(_GROUPS):
        model.x[i].value = float(len(group) == 1)
    return model


def _brute_force_optimum() -> int:
    """Return the exact optimum of the synthetic instance by enumeration."""

    best = None
    for picks in itertools.product((0, 1), repeat=len(_GROUPS)):
        counts = [0] * _MEMBERS
        for pick, group in zip(picks, _GROUPS):
            for member in group:
                counts[member] += pick
        if all(count == 1 for count in counts):
            cost = sum(pick * c for pick, c in zip(picks, _COSTS))
            best = cost if best is None else min(best, cost)
    return best


def _fake_results(model, termination, objective=None, bound=None, values=None):
    """Return an object shaped like an APPSI result for the given claims."""

    primals = ComponentMap()
    for index, value in (values or {}).items():
        primals[model.x[index]] = value
    return SimpleNamespace(
        termination_condition=termination,
        best_feasible_objective=objective,
        best_objective_bound=bound,
        solution_loader=SimpleNamespace(get_primals=lambda: primals),
    )


def _optimal_values() -> dict[int, float]:
    """Return the optimal selection (the single six-product group) as values."""

    chosen = _GROUPS.index((0, 1, 2, 3, 4, 5))
    return {i: float(i == chosen) for i in range(len(_GROUPS))}


def test_real_highs_solve_with_mip_start_is_consistent_and_optimal():
    """The pinned HiGHS returns the true optimum for the instance 1.16.0 breaks."""

    model = _exact_cover_model()
    solver = Highs()
    solver.config.load_solution = False
    solver.config.warmstart = True
    solver.highs_options = {
        "random_seed": 0,
        "threads": 1,
        "parallel": "off",
        "presolve_reduction_limit": -1,
    }
    results = solver.solve(model)
    verify_solver_result(model, results)
    assert results.termination_condition is TerminationCondition.optimal
    assert results.best_feasible_objective == pytest.approx(_brute_force_optimum())


def test_consistent_incumbent_passes_and_restores_start_values():
    """A correct incumbent passes and the model keeps its start values."""

    model = _exact_cover_model()
    start = [model.x[i].value for i in model.C]
    results = _fake_results(
        model, TerminationCondition.optimal, 59.0, 59.0, _optimal_values()
    )
    verify_solver_result(model, results)
    assert [model.x[i].value for i in model.C] == start


def test_false_objective_is_rejected():
    """The 1.16.0 defect: the start is returned with a smaller false objective."""

    model = _exact_cover_model()
    singleton = {i: float(len(group) == 1) for i, group in enumerate(_GROUPS)}
    results = _fake_results(model, TerminationCondition.optimal, 54.0, 54.0, singleton)
    with pytest.raises(SolverResultInconsistentError, match="objective 54 .* is 168"):
        verify_solver_result(model, results)


def test_incumbent_violating_a_constraint_is_rejected():
    """An incumbent that covers a product twice is rejected."""

    model = _exact_cover_model()
    values = {i: 0.0 for i in model.C}
    values[_GROUPS.index((0, 1, 2, 3, 4, 5))] = 1.0
    values[_GROUPS.index((0,))] = 1.0
    results = _fake_results(model, TerminationCondition.optimal, 116.0, 116.0, values)
    with pytest.raises(SolverResultInconsistentError, match="violates the model"):
        verify_solver_result(model, results)


def test_bound_worse_than_incumbent_is_rejected():
    """A minimization bound above the incumbent is a contradiction."""

    model = _exact_cover_model()
    results = _fake_results(
        model, TerminationCondition.optimal, 59.0, 80.0, _optimal_values()
    )
    with pytest.raises(SolverResultInconsistentError, match="bound"):
        verify_solver_result(model, results)


def test_infeasible_claim_with_feasible_start_is_rejected():
    """A false 'infeasible' is caught when the MIP start satisfies the model."""

    model = _exact_cover_model()
    results = _fake_results(model, TerminationCondition.infeasible)
    with pytest.raises(SolverResultInconsistentError, match="infeasible"):
        verify_solver_result(model, results)


def test_infeasible_claim_without_usable_start_is_kept():
    """Without a complete feasible start the solver's infeasibility stands."""

    model = _exact_cover_model()
    for i in model.C:
        model.x[i].value = None
    verify_solver_result(model, _fake_results(model, TerminationCondition.infeasible))
    for i in model.C:
        model.x[i].value = 1.0
    verify_solver_result(model, _fake_results(model, TerminationCondition.infeasible))


def test_time_limit_without_incumbent_is_not_flagged():
    """Only explicit infeasibility claims are checked against the start."""

    model = _exact_cover_model()
    verify_solver_result(
        model, _fake_results(model, TerminationCondition.maxTimeLimit)
    )


def test_failure_classification_names_the_solver_defect():
    """Run failure reports use the dedicated solver_inconsistent type."""

    failure_type, details = classify_failure(SolverResultInconsistentError("x"))
    assert failure_type == "solver_inconsistent"
    assert details["highspy_version"] == installed_highspy_version()


def test_installed_highspy_matches_the_pinned_requirement():
    """The environment runs the exact highspy version pinned for production."""

    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
    pinned = re.search(r"^highspy==(\S+)$", requirements, re.MULTILINE)
    assert pinned, "api/requirements.txt must pin highspy with =="
    assert installed_highspy_version() == pinned.group(1)
