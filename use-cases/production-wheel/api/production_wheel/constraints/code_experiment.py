"""Strict result contract for generated constraint CF-task experiments.

Generated code never runs in this application. See code_experiment_cf.py for the
trusted durable broker/controller and the separately deployed constraint runner.
"""
from __future__ import annotations
import math
from typing import Any

MAX_INPUT_BYTES = 1_000_000
MAX_OUTPUT_BYTES = 1_000_000
MAX_ROWS = 256
MAX_COEFFICIENTS = 20_000

class ExperimentError(ValueError):
    """Report invalid experiments or execution failures without accepting partial output."""


def _finite(value: Any) -> bool:
    """Return whether a JSON number is finite and within a conservative solver range."""
    return type(value) in (int, float) and abs(value) <= 1e12 and math.isfinite(value)


def _candidate_index(candidates: Any) -> dict[str, dict]:
    """Validate feature records and index unique IDs with required plant and sefi scope."""
    if not isinstance(candidates, list) or len(candidates) > MAX_COEFFICIENTS:
        raise ExperimentError("candidates must be a bounded list")
    index = {}
    for record in candidates:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key] or len(record[key]) > 256
            for key in ("candidate_hash", "plant", "sefi")
        ):
            raise ExperimentError("candidate_hash, plant and sefi must be nonempty strings")
        identifier = record["candidate_hash"]
        if identifier in index:
            raise ExperimentError("duplicate candidate_hash")
        index[identifier] = record
    return index


def validate_result(result: Any, candidates: list[dict]) -> dict:
    """Validate admission IDs and sparse rows against trusted candidate scopes.

    Return the JSON result unchanged only after strict structure, bounds, uniqueness,
    coefficient-count and scope checks. This validates structure, not rule semantics.
    """
    index = _candidate_index(candidates)
    if not isinstance(result, dict) or set(result) != {"allowed_candidate_ids", "rows"}:
        raise ExperimentError("result must contain only allowed_candidate_ids and rows")
    allowed, rows = result["allowed_candidate_ids"], result["rows"]
    if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
        raise ExperimentError("allowed_candidate_ids must be a list of strings")
    if len(allowed) != len(set(allowed)) or not set(allowed) <= index.keys():
        raise ExperimentError("duplicate or unknown allowed candidate IDs")
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise ExperimentError("rows must be a bounded list")
    identifiers, coefficient_count = set(), 0
    for row in rows:
        if not isinstance(row, dict):
            raise ExperimentError("row must be an object")
        required = {"rule_id", "scope", "plant", "coefficients"}
        if not required <= row.keys() or row.keys() - required - {"sefi", "lower", "upper"}:
            raise ExperimentError("missing or unknown row fields")
        rule_id = row["rule_id"]
        if not isinstance(rule_id, str) or not rule_id or len(rule_id) > 128 or rule_id in identifiers:
            raise ExperimentError("rule_id must be unique and nonempty")
        identifiers.add(rule_id)
        if row["scope"] not in ("plant", "block"):
            raise ExperimentError("scope must be plant or block")
        if not isinstance(row["plant"], str) or not row["plant"]:
            raise ExperimentError("plant must be nonempty")
        if row["scope"] == "block":
            if not isinstance(row.get("sefi"), str) or not row["sefi"]:
                raise ExperimentError("block scope requires sefi")
        elif "sefi" in row:
            raise ExperimentError("plant scope must not contain sefi")
        if not any(record["plant"] == row["plant"] and (
            row["scope"] == "plant" or record["sefi"] == row["sefi"]
        ) for record in index.values()):
            raise ExperimentError("unknown row scope")
        bounds = [key for key in ("lower", "upper") if key in row]
        if not bounds or any(not _finite(row[key]) for key in bounds):
            raise ExperimentError("row needs finite numeric lower and/or upper bounds")
        if row.get("lower", -math.inf) > row.get("upper", math.inf):
            raise ExperimentError("lower exceeds upper")
        coefficients = row["coefficients"]
        if not isinstance(coefficients, dict) or not coefficients:
            raise ExperimentError("coefficients must be a nonempty sparse object")
        coefficient_count += len(coefficients)
        if coefficient_count > MAX_COEFFICIENTS:
            raise ExperimentError("too many coefficients")
        for identifier, value in coefficients.items():
            if identifier not in index or not _finite(value):
                raise ExperimentError("unknown coefficient ID or invalid coefficient")
            record = index[identifier]
            if record["plant"] != row["plant"] or (
                row["scope"] == "block" and record["sefi"] != row["sefi"]
            ):
                raise ExperimentError("coefficient candidate outside row scope")
    return result
