"""Interpret finite business expressions over canonical candidate facts; never eval code."""

from __future__ import annotations

import math
import operator

from production_wheel.rule_models import Expression, GroupRule, SelectionBound
from production_wheel.runners import block_minimum, classify, sefi_ratio


def in_scope(block, scope):
    """Return whether a plant/SEFI identity matches the rule's explicit scope."""
    return (scope.plant is None or scope.plant == block[0]) and (scope.sefi is None or scope.sefi == block[1])


def runner_coverage(member, config, batch_litres=None):
    """Return the full-batch coverage days that classify one FINI as a runner.

    Args:
        member: Candidate FINI.
        config: Run configuration (``runner_basis`` and ``demand_days``).
        batch_litres: Effective batch of the proposed PV; required for the
            ``candidate_pv`` basis.

    Returns:
        ``reference`` basis: the source "RoC 1 full SEFI batch" when present
        (the same number SAP settings classify with), else source lot size
        considered / daily demand. ``candidate_pv`` basis: the proposed batch /
        daily demand. ``None`` when the inputs are missing.
    """
    daily = member.demand_litres / config.demand_days
    if config.runner_basis == "candidate_pv":
        return batch_litres / daily
    if member.reference_coverage_source_days is not None and member.reference_coverage_source_days > 0:
        return member.reference_coverage_source_days
    lot = member.lot_size_considered_litres
    return lot / daily if lot is not None and lot > 0 else None


def runner_ratios(members, config):
    """Return each FINI's ratio to the fastest runner of its plant/SEFI block.

    The ratio does not depend on the proposed PV: under ``candidate_pv`` every
    FINI of a block uses the same batch, which cancels out, so a unit batch is used.

    Args:
        members: Every FINI of the population (all blocks), not one candidate.
        config: Run configuration.

    Returns:
        ``{(plant, sefi, fini_id): ratio or None}``.
    """
    coverage = {(*m.block_key, m.fini_id): runner_coverage(m, config, 1.0) for m in members}
    minimum = {}
    for (plant, sefi, _), value in coverage.items():
        minimum.setdefault((plant, sefi), []).append(value)
    minimum = {key: block_minimum(values) for key, values in minimum.items()}
    return {key: sefi_ratio(value, minimum[key[:2]]) for key, value in coverage.items()}


def facts(candidate, members, config, ratios):
    """Reconstruct rule inputs from FINI primitives and the proposed PV/line.

    Args:
        candidate: Proposed group (members, PV, line).
        members: The candidate's FINIs.
        config: Run configuration.
        ratios: ``runner_ratios`` of the whole population, so the relative
            runner view sees the block's fastest FINI, not only this group.

    Returns:
        ``{"group": {...}, "members": [...]}`` facts used by rule expressions.
        ``group.coverage_days`` and ``group.j_ch`` come from
        ``cycle_stock.strategy_figures`` (whole-day wheel RoC, the run's
        coverage basis) and follow the candidate's cycle stock strategy
        (runner multiples raise coverage and lower ``J_CH``);
        ``member.runner_multiple`` is 1 for every FINI of a single-RoC group.
        ``group.wheel_roc_days`` is the whole-day wheel RoC ``R``;
        ``group.batches_per_week`` is days per week / ``R`` and
        ``group.batches_per_day`` is ``1 / R`` (lots per production day). The
        batch is made every cycle under both strategies (a runner-multiple FINI
        is added to one batch out of k), so these do not depend on the multiples.
        All three are fixed per candidate, so sums of them stay linear rows.
    """
    demand = sum(m.demand_litres for m in members)
    batch = candidate.nominal_lot_litres * config.canonical_factor
    records = []
    for m in members:
        reference = (m.lot_size_considered_litres / (m.demand_litres / config.demand_days)
            if m.lot_size_considered_litres is not None and m.lot_size_considered_litres > 0 else None)
        ratio = ratios.get((*m.block_key, m.fini_id))
        record = {"fini_id": m.fini_id, "volume": m.package_volume, "pck": m.pck_code or None,
            "reference_coverage_days": reference, "sefi_ratio": ratio,
            "runner": classify(runner_coverage(m, config, batch), ratio, config.runner_thresholds),
            "demand_litres": m.demand_litres}
        record.update({name: getattr(m, name) for name in ("xyz_x_dc_count", "xyz_y_dc_count", "xyz_z_dc_count",
            "primary_dc_count", "secondary_dc_count", "sales_scenario", "sales_network_scenario")})
        records.append(record)
    common = sorted(set.intersection(*(set(m.eligible_lines) for m in members)))
    pcks = {m.pck_code for m in members}
    # Imported here: cycle_stock builds on this module's runner_coverage.
    from production_wheel.cycle_stock import strategy_figures
    strategy = getattr(candidate, "cycle_stock_strategy", "single")
    # Same rule as the solver coefficients (production_wheel.wheel_roc): whole-day
    # wheel RoC for J_CH and runner multiples, pallets only in the inventory bases.
    figures = strategy_figures(strategy, members, candidate.nominal_lot_litres, batch, config, ratios,
                               classify_members=False)
    coverage, j_ch = figures.coverage_days, figures.j_ch
    roc = figures.wheel.roc_days
    multiples = {item.fini_id: item.multiple for item in figures.members}
    for record in records:
        record["runner_multiple"] = multiples[record["fini_id"]]
    return {"group": {"size": len(members), "common_lines": common,
        "selected_line": candidate.selected_line, "pv_id": candidate.pv_id,
        # SEFI batch size of the selected PV, its effective (usable) batch and the
        # fractional number of SEFI batches needed to cover the horizon demand.
        "pv_lot_litres": candidate.nominal_lot_litres, "batch_litres": batch,
        "batch_count": demand / batch,
        "demand_litres": demand, "coverage_days": coverage, "j_ch": j_ch,
        # Wheel RoC and batch frequency (production_wheel.wheel_roc), e.g. for
        # plant capacity rules: sum of batches_per_day per tank size <= tanks x lots.
        "wheel_roc_days": roc, "batches_per_week": figures.frequency_per_week,
        "batches_per_day": 1.0 / roc if roc else None,
        "cycle_stock_strategy": strategy,
        "equal_pck": len(pcks) == 1 and None not in pcks and "" not in pcks}, "members": records}


def number(value):
    """Require a finite scalar coefficient instead of accepting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"business rule requires a finite number, got {value!r}")
    return value


def boolean(value):
    """Require actual boolean predicate results; never coerce missing values."""
    if not isinstance(value, bool):
        raise ValueError("business predicate must return true or false")
    return value


def evaluate(expr: Expression, data, member=None):
    """Evaluate a validated expression with explicit missing-data and type errors."""
    op = expr.op
    if op == "literal":
        return expr.value
    if op == "field":
        owner, key = expr.field.split(".", 1)
        if owner == "member" and member is None:
            raise ValueError(f"{expr.field} requires a member aggregate")
        value = (member if owner == "member" else data["group"])[key]
        if value is None:
            raise ValueError(f"missing required business rule field: {expr.field}")
        return value
    if op in {"count", "sum", "min", "max", "distinct", "any", "all"}:
        if member is not None:
            raise ValueError("nested member aggregates are not supported")
        values = [evaluate(expr.args[0], data, row) for row in data["members"]]
        if op in {"count", "any", "all"}:
            values = [boolean(value) for value in values]
            return sum(values) if op == "count" else any(values) if op == "any" else all(values)
        if op == "distinct":
            return len(set(values))
        values = [number(value) for value in values]
        return {"sum": sum, "min": min, "max": max}[op](values)
    if op == "if":
        # Only the chosen branch is evaluated, so the other branch may reference absent data.
        chosen = expr.args[1] if boolean(evaluate(expr.args[0], data, member)) else expr.args[2]
        return evaluate(chosen, data, member)
    values = [evaluate(arg, data, member) for arg in expr.args]
    if op in {"and", "or", "not"}:
        values = [boolean(value) for value in values]
        return all(values) if op == "and" else any(values) if op == "or" else not values[0]
    left, right = values
    if op in {"add", "sub", "mul", "div", "gt", "gte", "lt", "lte"}:
        left, right = number(left), number(right)
    functions = {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt,
        "gte": operator.ge, "lt": operator.lt, "lte": operator.le,
        "add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv,
        "contains": operator.contains, "in": lambda a, b: a in b}
    try:
        result = functions[op](left, right)
    except (TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid business expression {op}: {exc}") from exc
    return number(result) if op in {"add", "sub", "mul", "div"} else result


def group_allowed(candidate, members, config, ratios):
    """Apply every in-scope hard group rule to one complete candidate.

    Args:
        candidate: Complete candidate group.
        members: The candidate's FINIs.
        config: Run configuration with business rules.
        ratios: ``runner_ratios`` of the candidate's whole block (or population).

    Returns:
        ``True`` when every applicable rule holds.
    """
    rules = [r for r in config.business_rules if isinstance(r, GroupRule) and in_scope(candidate.block_key, r.scope)]
    if not rules:
        return True
    data = facts(candidate, members, config, ratios)
    for rule in rules:
        try:
            if boolean(evaluate(rule.when, data)) and not boolean(evaluate(rule.assertion, data)):
                return False
        except ValueError as exc:
            raise ValueError(f"rule {rule.constraint_id}: {exc}") from exc
    return True


def linear_rows(candidates, members, config):
    """Compile scoped selection measures into numeric rows for the trusted master."""
    rules = [r for r in config.business_rules if isinstance(r, SelectionBound)]
    by_key = {(*m.block_key, m.fini_id): m for m in members}
    ratios = runner_ratios(members, config)
    result = []
    for rule in rules:
        coefficients = {}
        for c in candidates:
            if in_scope(c.block_key, rule.scope):
                data = facts(c, [by_key[(*c.block_key, fid)] for fid in c.member_ids], config, ratios)
                coefficients[c.candidate_hash] = number(evaluate(rule.measure, data))
        result.append({"rule_id": rule.constraint_id, "coefficients": coefficients,
            "lower": rule.lower, "upper": rule.upper})
    return result


def selected_rule_failures(selected, members, config):
    """Recheck selected group predicates and aggregate totals from original FINI data."""
    by_key = {(*m.block_key, m.fini_id): m for m in members}
    ratios = runner_ratios(members, config)
    totals = {r.constraint_id: 0.0 for r in config.business_rules if isinstance(r, SelectionBound)}
    failures = []
    for c in selected:
        data = facts(c, [by_key[(*c.block_key, fid)] for fid in c.member_ids], config, ratios)
        if config.assign_filling_lines and c.selected_line not in data["group"]["common_lines"]:
            failures.append(("SELECTED_LINE_ELIGIBLE", c.selected_line, data["group"]["common_lines"]))
        for rule in config.business_rules:
            if not in_scope(c.block_key, rule.scope):
                continue
            try:
                if isinstance(rule, GroupRule):
                    if boolean(evaluate(rule.when, data)) and not boolean(evaluate(rule.assertion, data)):
                        failures.append((rule.constraint_id, c.member_ids, "group assertion"))
                else:
                    totals[rule.constraint_id] += number(evaluate(rule.measure, data))
            except ValueError as exc:
                failures.append((rule.constraint_id, str(exc), "complete valid rule inputs"))
    for rule in config.business_rules:
        if isinstance(rule, SelectionBound):
            total = totals[rule.constraint_id]
            if (rule.lower is not None and total < rule.lower - 1e-7) or (rule.upper is not None and total > rule.upper + 1e-7):
                failures.append((rule.constraint_id, total, (rule.lower, rule.upper)))
    return failures
