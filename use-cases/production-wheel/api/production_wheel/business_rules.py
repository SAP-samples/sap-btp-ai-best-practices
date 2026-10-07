"""Interpret finite business expressions over canonical candidate facts; never eval code."""

from __future__ import annotations

import math
import operator

from production_wheel.rule_models import Expression, GroupRule, SelectionBound
from production_wheel.metrics import group_coverage, proportional_allocations, pallet_allocations


def in_scope(block, scope):
    """Return whether a plant/SEFI identity matches the rule's explicit scope."""
    return (scope.plant is None or scope.plant == block[0]) and (scope.sefi is None or scope.sefi == block[1])


def facts(candidate, members, config):
    """Reconstruct rule inputs from FINI primitives and the proposed PV/line."""
    demand = sum(m.demand_litres for m in members)
    batch = candidate.nominal_lot_litres * config.canonical_factor
    records = []
    for m in members:
        reference = (m.lot_size_considered_litres / (m.demand_litres / config.demand_days)
            if m.lot_size_considered_litres is not None and m.lot_size_considered_litres > 0 else None)
        classification_coverage = batch * config.demand_days / m.demand_litres if config.runner_basis == "candidate_pv" else reference
        record = {"fini_id": m.fini_id, "volume": m.package_volume, "pck": m.pck_code or None,
            "reference_coverage_days": reference,
            "runner": None if classification_coverage is None else "high" if classification_coverage <= config.high_runner_threshold_days else "low",
            "demand_litres": m.demand_litres}
        record.update({name: getattr(m, name) for name in ("xyz_x_dc_count", "xyz_y_dc_count", "xyz_z_dc_count",
            "primary_dc_count", "secondary_dc_count", "sales_scenario", "sales_network_scenario")})
        records.append(record)
    common = sorted(set.intersection(*(set(m.eligible_lines) for m in members)))
    pcks = {m.pck_code for m in members}
    demands = {m.fini_id: m.demand_litres for m in members}
    allocations = pallet_allocations(proportional_allocations(demands, batch),
        {m.fini_id: m.pallet_litres for m in members}, config.pallet_formula)
    coverage = float(group_coverage(config.coverage_basis, demands, batch, allocations, config.demand_days))
    return {"group": {"size": len(members), "common_lines": common,
        "selected_line": candidate.selected_line, "pv_id": candidate.pv_id,
        "demand_litres": demand, "coverage_days": coverage,
        "j_ch": demand / (config.productive_weeks * batch) * (len(members) - 1),
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


def group_allowed(candidate, members, config):
    """Apply every in-scope hard group rule to one complete candidate."""
    rules = [r for r in config.business_rules if isinstance(r, GroupRule) and in_scope(candidate.block_key, r.scope)]
    if not rules:
        return True
    data = facts(candidate, members, config)
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
    result = []
    for rule in rules:
        coefficients = {}
        for c in candidates:
            if in_scope(c.block_key, rule.scope):
                data = facts(c, [by_key[(*c.block_key, fid)] for fid in c.member_ids], config)
                coefficients[c.candidate_hash] = number(evaluate(rule.measure, data))
        result.append({"rule_id": rule.constraint_id, "coefficients": coefficients,
            "lower": rule.lower, "upper": rule.upper})
    return result


def selected_rule_failures(selected, members, config):
    """Recheck selected group predicates and aggregate totals from original FINI data."""
    by_key = {(*m.block_key, m.fini_id): m for m in members}
    totals = {r.constraint_id: 0.0 for r in config.business_rules if isinstance(r, SelectionBound)}
    failures = []
    for c in selected:
        data = facts(c, [by_key[(*c.block_key, fid)] for fid in c.member_ids], config)
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
