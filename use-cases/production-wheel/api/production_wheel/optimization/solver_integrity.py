"""Independent checks of what HiGHS reports after each MIP solve.

The optimizer trusts the solver for two claims: "this incumbent has objective
value V" and "this model has no feasible solution". A defective solver release
can break either claim (highspy 1.16.0 reported a MIP start as optimal with a
false objective, and Pyomo then reported the trivially feasible block model as
infeasible). These checks re-evaluate the claims in Pyomo, so a solver defect
stops the affected solve with a clear message instead of being mistaken for a
property of the plant rules or the data. Example::

    results = solver.solve(model)
    verify_solver_result(model, results)  # raises SolverResultInconsistentError

Two checks run:

- Incumbent check (an incumbent was reported): the incumbent, with integer
  variables rounded, must satisfy every active constraint and integrality, its
  recomputed objective must match the reported objective, and the reported
  bound must not be worse than the incumbent. Tolerances are loose so only a
  clear contradiction fails.
- Infeasibility check (termination ``infeasible`` or ``infeasibleOrUnbounded``):
  when the variable values left in the model before the solve (the MIP start)
  form a feasible point under a strict tolerance, the infeasibility claim is
  contradicted. Models without a complete start keep the solver's answer.

Variable values in the model are restored after every check, so warm starts
of later solves are unchanged.
"""

from __future__ import annotations

import math
from importlib import metadata
from typing import Any

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition

# Incumbent checks: fail only on a clear contradiction, not on solver round-off.
_INCUMBENT_RELATIVE_TOLERANCE = 1e-5
_INCUMBENT_ABSOLUTE_TOLERANCE = 1e-6
_INTEGRALITY_TOLERANCE = 1e-5
# Start checks: a start is called feasible only when it clearly is. This is far
# tighter than the 1e-7 lexicographic lock tolerance, so a start that sits on a
# lock boundary is never used to overrule a legitimate infeasibility proof.
_START_TOLERANCE = 1e-9
_INFEASIBLE_TERMINATIONS = {
    TerminationCondition.infeasible,
    TerminationCondition.infeasibleOrUnbounded,
}


class SolverResultInconsistentError(RuntimeError):
    """Raised when HiGHS reports a result that an independent check contradicts.

    The message names the contradiction and the installed highspy version. It
    states that the cause is the solver library, not the plant rules or data.
    """


def installed_highspy_version() -> str:
    """Return the installed highspy distribution version, or ``"unknown"``."""

    try:
        return metadata.version("highspy")
    except metadata.PackageNotFoundError:
        return "unknown"


def verify_solver_result(model: pyo.ConcreteModel, results: Any) -> None:
    """Check one APPSI HiGHS result against the model it was solved for.

    Args:
        model: The Pyomo model passed to ``solver.solve``. Its variable values
            must still be the values set before the solve (the MIP start, or
            ``None``); results must not have been loaded yet.
        results: The APPSI ``Results`` object returned by ``solver.solve``.

    Returns:
        ``None`` when the result is consistent or cannot be checked.

    Raises:
        SolverResultInconsistentError: When the incumbent violates the model or
            its objective/bound do not match, or when the solver claims
            infeasibility although the MIP start is feasible.
    """

    incumbent = _finite(results.best_feasible_objective)
    if incumbent is None:
        if results.termination_condition in _INFEASIBLE_TERMINATIONS:
            problem = _start_violation(model)
            if problem is None:
                _raise(
                    f"it reported the model as {results.termination_condition.name}, "
                    "although the starting solution passed to it satisfies every "
                    "constraint"
                )
        return
    primals = results.solution_loader.get_primals()
    problem = _incumbent_problem(
        model, primals, incumbent, _finite(results.best_objective_bound)
    )
    if problem is not None:
        _raise(problem)


def _raise(detail: str) -> None:
    """Raise the planner-readable inconsistency error for one contradiction."""

    raise SolverResultInconsistentError(
        f"The HiGHS solver (highspy {installed_highspy_version()}) returned an "
        f"inconsistent result: {detail}. This is a solver library defect, not a "
        "property of the plant rules or the data; install the pinned highspy "
        "version from api/requirements.txt and rerun."
    )


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _variables(model: pyo.ConcreteModel) -> list[Any]:
    """Return every variable data object of the model in a stable order."""

    return list(model.component_data_objects(pyo.Var, descend_into=True))


def _objective(model: pyo.ConcreteModel) -> Any | None:
    """Return the single active objective, or ``None`` when there is none."""

    return next(
        iter(model.component_data_objects(pyo.Objective, active=True, descend_into=True)),
        None,
    )


def _constraint_violation(
    model: pyo.ConcreteModel, relative: float, absolute: float
) -> str | None:
    """Return the first active constraint violated beyond tolerance, or ``None``.

    Args:
        model: Model whose variables currently hold the point to check.
        relative: Tolerance relative to the larger of the bound and body value.
        absolute: Absolute tolerance added to the relative part.

    Returns:
        A short description of the violation, or ``None`` when all hold.
    """

    for constraint in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        body = pyo.value(constraint.body, exception=False)
        if body is None:
            return f"constraint {constraint.name} has unassigned variables"
        for bound, below in (
            (pyo.value(constraint.lower, exception=False), True),
            (pyo.value(constraint.upper, exception=False), False),
        ):
            if bound is None:
                continue
            tolerance = absolute + relative * max(1.0, abs(bound), abs(body))
            violation = bound - body if below else body - bound
            if violation > tolerance:
                side = "lower" if below else "upper"
                return (
                    f"constraint {constraint.name} has value {body:.6g}, "
                    f"outside its {side} bound {bound:.6g}"
                )
    return None


def _start_violation(model: pyo.ConcreteModel) -> str | None:
    """Return why the model's current values are not a feasible start, or ``None``.

    A start is feasible when every variable has a value, integer variables are
    integral and within their bounds, and every active constraint holds, all
    under the strict start tolerance. Values are not modified.
    """

    for variable in _variables(model):
        value = variable.value
        if value is None:
            return f"variable {variable.name} has no start value"
        if variable.is_integer() and abs(value - round(value)) > _START_TOLERANCE:
            return f"variable {variable.name} is not integral"
        lower, upper = variable.lb, variable.ub
        if lower is not None and value < lower - _START_TOLERANCE:
            return f"variable {variable.name} is below its lower bound"
        if upper is not None and value > upper + _START_TOLERANCE:
            return f"variable {variable.name} is above its upper bound"
    return _constraint_violation(model, _START_TOLERANCE, _START_TOLERANCE)


def _incumbent_problem(
    model: pyo.ConcreteModel,
    primals: Any,
    reported_objective: float,
    reported_bound: float | None,
) -> str | None:
    """Return the first contradiction in a reported incumbent, or ``None``.

    The incumbent is placed in the model temporarily (integer variables rounded,
    as downstream code reads them), checked, and the previous values restored.

    Args:
        model: The solved model.
        primals: Mapping of variable data object to the solver's value.
        reported_objective: The solver's incumbent objective value.
        reported_bound: The solver's best objective bound, when finite.

    Returns:
        A short description of the contradiction, or ``None``.
    """

    variables = _variables(model)
    previous = [variable.value for variable in variables]
    try:
        for variable in variables:
            if variable not in primals:
                continue
            value = float(primals[variable])
            if variable.is_integer():
                if abs(value - round(value)) > _INTEGRALITY_TOLERANCE:
                    return f"variable {variable.name} = {value:.6g} is not integral"
                value = float(round(value))
            variable.set_value(value, skip_validation=True)
        problem = _constraint_violation(
            model, _INCUMBENT_RELATIVE_TOLERANCE, _INCUMBENT_ABSOLUTE_TOLERANCE
        )
        if problem is not None:
            return f"its solution violates the model: {problem}"
        objective = _objective(model)
        if objective is None:
            return None
        recomputed = pyo.value(objective.expr, exception=False)
        if recomputed is None:
            return "its solution leaves objective variables unassigned"
        scale = max(1.0, abs(recomputed), abs(reported_objective))
        tolerance = _INCUMBENT_ABSOLUTE_TOLERANCE + _INCUMBENT_RELATIVE_TOLERANCE * scale
        if abs(recomputed - reported_objective) > tolerance:
            return (
                f"it reported objective {reported_objective:.6g} for a solution "
                f"whose objective is {recomputed:.6g}"
            )
        if reported_bound is not None:
            minimizing = objective.sense == pyo.minimize
            worse = (
                reported_bound - reported_objective
                if minimizing
                else reported_objective - reported_bound
            )
            if worse > tolerance:
                return (
                    f"its objective bound {reported_bound:.6g} is worse than its "
                    f"own solution {reported_objective:.6g}"
                )
        return None
    finally:
        for variable, value in zip(variables, previous):
            variable.set_value(value, skip_validation=True)
