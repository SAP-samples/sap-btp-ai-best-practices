"""Finite read-only analytical queries over registered workspace table views."""

from __future__ import annotations

import json
import math
import statistics
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.workspace.models import INPUT_VIEWS, RESULT_VIEWS, QuerySpec
from app.workspace.analytics_evidence import explain_assignment as explain_assignment
from app.workspace.analytics_comparison import compare_solutions as compare_solutions

CATALOG = json.loads(
    Path(__file__).with_name("columns.json").read_text(encoding="utf-8")
)
NUMERIC_TYPES = ("DOUBLE", "INTEGER", "BIGINT", "DECIMAL", "REAL")


def available_fields(view: str, rows: Iterable[Mapping[str, Any]] = ()) -> list[str]:
    """Return the finite catalog plus observed fields for one registered view."""
    if view not in INPUT_VIEWS | RESULT_VIEWS:
        raise ValueError(f"unknown analytical view: {view}")
    fields = set(CATALOG.get(view, {}))
    for row in rows:
        fields.update(row)
    return sorted(fields)


def _number(value: Any) -> float | None:
    """Parse finite numerical evidence; blanks and missing values are null."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _value(value: Any, numeric: bool) -> Any:
    """Normalize typed numeric columns while preserving identifier strings."""
    if value is None or value == "":
        return None
    return _number(value) if numeric else value


def _matches(row: Mapping[str, Any], predicate: Any, numeric: set[str]) -> bool:
    """Evaluate one validated finite predicate against a typed source value."""
    left = _value(row.get(predicate.field), predicate.field in numeric)
    right = _value(predicate.value, predicate.field in numeric)
    if predicate.op == "is_null":
        return (left is None) if predicate.value is not False else (left is not None)
    if predicate.op == "in":
        if not isinstance(predicate.value, list):
            raise ValueError("in filter requires a list")
        return left in [
            _value(item, predicate.field in numeric) for item in predicate.value
        ]
    if predicate.op == "eq":
        return left == right
    if predicate.op == "ne":
        return left != right
    if left is None or right is None:
        return False
    if predicate.op == "contains":
        return str(right).casefold() in str(left).casefold()
    if predicate.op in ("gt", "gte", "lt", "lte"):
        if type(left) is not type(right) and not (
            isinstance(left, (int, float)) and isinstance(right, (int, float))
        ):
            raise ValueError(f"incompatible filter value for {predicate.field}")
        return {
            "gt": lambda: left > right,
            "gte": lambda: left >= right,
            "lt": lambda: left < right,
            "lte": lambda: left <= right,
        }[predicate.op]()
    raise ValueError("unknown predicate")


def _metric_name(metric: Any) -> str:
    """Build a stable aggregate output name unless an explicit alias is given."""
    return metric.alias or (
        "count"
        if metric.operation == "count" and metric.field == "*"
        else f"{metric.operation}_{metric.field}"
    )


def _aggregate(rows: list[dict[str, Any]], metric: Any) -> int | float | None:
    """Calculate a metric over the complete filtered group, excluding nulls."""
    if metric.operation == "count":
        return (
            len(rows)
            if metric.field == "*"
            else sum(row.get(metric.field) not in (None, "") for row in rows)
        )
    values = [_number(row.get(metric.field)) for row in rows]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    if metric.operation == "weighted_mean":
        if not metric.weight_field:
            raise ValueError("weighted_mean requires weight_field")
        pairs = [
            (value, _number(row.get(metric.weight_field)))
            for row, value in zip(rows, values)
        ]
        pairs = [
            (value, weight)
            for value, weight in pairs
            if value is not None and weight is not None
        ]
        if any(weight < 0 for _, weight in pairs):
            raise ValueError("weighted_mean requires nonnegative weights")
        denominator = sum(weight for _, weight in pairs)
        return (
            sum(value * weight for value, weight in pairs) / denominator
            if denominator > 0
            else None
        )
    if metric.operation == "p90":
        return sorted(numbers)[math.ceil(0.9 * len(numbers)) - 1]
    operations = {
        "sum": sum,
        "mean": statistics.mean,
        "median": statistics.median,
        "min": min,
        "max": max,
    }
    return operations[metric.operation](numbers)


def _unit(field: str) -> str | None:
    """Describe units only when their field semantics are explicit."""
    if field.endswith("days"):
        return "days"
    if "litres" in field:
        return "litres"
    if field == "frequency_per_week":
        return "runs/week"
    if field in ("j_ch", "j_ch_contribution"):
        return "recurring within-group FINI changes/week (PoC proxy)"
    if field.endswith("count") or field == "group_size":
        return "count"
    return None


def _sort_rows(rows: list[dict[str, Any]], terms: list[Any]) -> list[dict[str, Any]]:
    """Sort deterministically by requested terms with nulls last in both directions."""

    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        """Compare typed rows and break ties by canonical full-row representation."""
        for term in terms:
            a, b = left.get(term.field), right.get(term.field)
            if a is None and b is None:
                continue
            if a is None:
                return 1
            if b is None:
                return -1
            if type(a) is not type(b) and not (
                isinstance(a, (int, float)) and isinstance(b, (int, float))
            ):
                a, b = str(a), str(b)
            try:
                outcome = (a > b) - (a < b)
            except TypeError:
                a, b = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
                outcome = (a > b) - (a < b)
            if outcome:
                return outcome if term.direction == "asc" else -outcome
        a, b = (
            json.dumps(left, sort_keys=True, default=str),
            json.dumps(right, sort_keys=True, default=str),
        )
        return (a > b) - (a < b)

    return sorted(rows, key=cmp_to_key(compare))


def execute_query(repo: Any, spec: QuerySpec) -> dict[str, Any]:
    """Execute a bounded, typed query with full-scope aggregates before pagination.

    Args:
        repo: Repository exposing rows(owner_id, registered_view).
        spec: Validated semantic query; SQL and executable expressions are absent.
    Returns:
        Page of rows with total, source scope, column units and calculation evidence.
    """
    spec = QuerySpec.model_validate(spec.model_dump())
    owner_id = spec.run_id or spec.dataset_id
    pushed_filters = (
        {"point_index": spec.point_index} if spec.point_index is not None else {}
    )
    source = repo.rows(owner_id, spec.view, filters=pushed_filters)
    fields = set(available_fields(spec.view, source))
    numeric = {
        name
        for name, kind in CATALOG.get(spec.view, {}).items()
        if kind.startswith(NUMERIC_TYPES)
    }
    numeric.update(
        name
        for row in source
        for name, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    numeric.add("point_index")
    requested = set(spec.fields + spec.group_by + [item.field for item in spec.filters])
    for metric in spec.metrics:
        if metric.field != "*":
            requested.add(metric.field)
        elif metric.operation != "count":
            raise ValueError("only count accepts field '*'")
        if metric.operation == "weighted_mean" and not metric.weight_field:
            raise ValueError("weighted_mean requires weight_field")
        if metric.weight_field:
            requested.add(metric.weight_field)
    unknown = requested - fields
    if unknown:
        raise ValueError(f"unknown fields for {spec.view}: {sorted(unknown)}")
    if spec.metrics and spec.fields and set(spec.fields) - set(spec.group_by):
        raise ValueError("aggregate query fields must be group_by fields")
    for predicate in spec.filters:
        if predicate.op == "in" and not isinstance(predicate.value, list):
            raise ValueError("in filter requires a list")
        values = predicate.value if predicate.op == "in" else [predicate.value]
        if predicate.field in numeric and predicate.op != "is_null":
            if any(
                value not in (None, "") and _number(value) is None for value in values
            ):
                raise ValueError(
                    f"numeric filter requires numeric values: {predicate.field}"
                )
        if predicate.op == "is_null" and predicate.value not in (None, True, False):
            raise ValueError("is_null filter requires a boolean or null")
    for metric in spec.metrics:
        if metric.operation != "count" and metric.field not in numeric:
            raise ValueError(f"aggregate requires a numeric field: {metric.field}")
        if metric.weight_field and metric.weight_field not in numeric:
            raise ValueError(f"weight requires a numeric field: {metric.weight_field}")
    names = [_metric_name(metric) for metric in spec.metrics]
    if len(set(names)) != len(names) or set(names) & set(spec.group_by):
        raise ValueError(
            "aggregate aliases must be unique and distinct from group fields"
        )
    normalized = [
        {key: _value(value, key in numeric) for key, value in row.items()}
        for row in source
    ]
    if spec.point_index is not None:
        if spec.view in INPUT_VIEWS:
            raise ValueError("point_index applies only to point-scoped result views")
        if "point_index" not in fields:
            raise ValueError("selected result view is not point scoped")
        normalized = [
            row for row in normalized if row.get("point_index") == spec.point_index
        ]
    filtered = [
        row
        for row in normalized
        if all(_matches(row, predicate, numeric) for predicate in spec.filters)
    ]
    if spec.metrics or spec.group_by:
        groups: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for row in filtered:
            group = {field: row.get(field) for field in spec.group_by}
            key = json.dumps(group, sort_keys=True, default=str)
            groups.setdefault(key, (group, []))[1].append(row)
        if not spec.group_by and not groups:
            groups["{}"] = ({}, [])
        output = [
            {
                **group,
                **{
                    _metric_name(metric): _aggregate(rows, metric)
                    for metric in spec.metrics
                },
            }
            for group, rows in groups.values()
        ]
        columns = list(spec.group_by) + names
        sort_fields = set(columns)
    else:
        output = filtered
        columns = spec.fields or sorted(fields)
        sort_fields = fields
    if {term.field for term in spec.sort} - sort_fields:
        raise ValueError("sort contains unknown output fields")
    ordered = _sort_rows(output, spec.sort)
    total = len(ordered)
    rows = [
        {key: row.get(key) for key in columns}
        for row in ordered[spec.offset : spec.offset + spec.limit]
    ]
    units = {field: _unit(field) for field in columns}
    for metric in spec.metrics:
        units[_metric_name(metric)] = (
            "count" if metric.operation == "count" else _unit(metric.field)
        )
    return {
        "rows": rows,
        "total": total,
        "columns": columns,
        "scope": {
            "dataset_id": spec.dataset_id,
            "run_id": spec.run_id,
            "point_index": spec.point_index,
            "source_rows": len(source),
            "filtered_rows": len(filtered),
            "storage_filters": pushed_filters,
            "aggregate_before_pagination": bool(spec.metrics),
            "view": spec.view,
        },
        "units": units,
        "truncated": spec.offset + len(rows) < total,
        "offset": spec.offset,
        "limit": spec.limit,
        "evidence": {
            "source": "persisted_registered_view",
            "query": spec.model_dump(),
            "null_policy": "exclude nulls; empty aggregates are null except count",
            "p90_method": "nearest_rank",
            "business_acceptance_assessed": False,
        },
    }
