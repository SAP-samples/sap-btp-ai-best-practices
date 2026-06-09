"""Load and validate versioned program, tariff, and taxonomy catalogs."""

from __future__ import annotations

import copy
import json
from functools import lru_cache

from app.nbo.config import (
    COL_RATE_PLAN,
    COMMERCIAL_TAXONOMY_FILE,
    PROGRAM_CATALOG_FILE,
    PROGRAM_RULE_MATRIX_FILE,
    TARIFF_CATALOG_FILE,
)
from app.nbo.data_loader import DataStore
from app.nbo.fact_registry import known_fact_ids


class CatalogValidationError(ValueError):
    """Raised when a catalog entry is invalid."""


PROGRAM_CONTRACT_TERM_OVERRIDES: dict[str, list[str]] = {
    "income_qualified_discount": ["economy discount"],
    "free_shade_trees": ["trees for change", "healthy forest initiatives"],
    "renewable_energy_credit_purchase_program": ["rec charge", "rec select"],
    "battery_partner": ["residential battery incentive"],
    "demand_management_system_rebate": ["demand assurance"],
    "tax_deductible_solar_programs": [
        "community solar",
        "community solar choice",
        "solar choice select",
        "solar for nonprofits",
        "solar for schools",
    ],
    "tax_deductible_donation_programs": ["solar for nonprofits", "solar for schools"],
    "non_tax_deductible_donation_programs": ["earthwise energy", "environmental programs"],
    "tax_deductible_environmental_programs": ["healthy forest initiatives", "trees for change"],
    "non_tax_deductible_environmental_programs": ["environmental programs", "earthwise energy"],
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_program_catalog_raw() -> list[dict]:
    return _load_json(PROGRAM_CATALOG_FILE)


@lru_cache(maxsize=1)
def load_program_rule_matrix() -> dict[str, dict]:
    raw = _load_json(PROGRAM_RULE_MATRIX_FILE)
    return {entry["program_id"]: entry for entry in raw}


def _default_program_contract_terms(entry: dict) -> list[str]:
    terms = [entry.get("display_name", "").strip()]
    explicit_terms = PROGRAM_CONTRACT_TERM_OVERRIDES.get(entry["program_id"], [])
    return [term for term in [*explicit_terms, *terms] if term]


def _enrich_program_aliases(entries: list[dict], ds: DataStore | None) -> list[dict]:
    if ds is None:
        return entries

    enriched: list[dict] = []
    for entry in entries:
        updated = copy.deepcopy(entry)
        runtime_aliases = ds.resolve_program_code_aliases(
            _default_program_contract_terms(updated)
        )
        updated["program_code_aliases"] = sorted(
            {
                *updated.get("program_code_aliases", []),
                *runtime_aliases,
            }
        )
        enriched.append(updated)
    return enriched


def load_program_catalog(ds: DataStore | None = None) -> list[dict]:
    return _enrich_program_aliases(copy.deepcopy(_load_program_catalog_raw()), ds)


@lru_cache(maxsize=1)
def load_tariff_catalog() -> list[dict]:
    return _load_json(TARIFF_CATALOG_FILE)


@lru_cache(maxsize=1)
def load_commercial_taxonomy() -> list[dict]:
    return _load_json(COMMERCIAL_TAXONOMY_FILE)


def _validate_predicate(predicate: dict, label: str) -> None:
    known = known_fact_ids()
    fact_id = predicate.get("fact_id")
    if fact_id not in known:
        raise CatalogValidationError(f"{label}: unknown fact id {fact_id!r}")
    if predicate.get("op") not in {"eq", "lt", "lte", "gt", "gte", "in", "contains"}:
        raise CatalogValidationError(f"{label}: unsupported operator {predicate.get('op')!r}")


def _validate_logic(node: dict | list | None, label: str) -> None:
    if not node:
        return
    if isinstance(node, list):
        for index, child in enumerate(node):
            _validate_logic(child, f"{label}[{index}]")
        return
    if "fact_id" in node:
        _validate_predicate(node, label)
        return

    group_keys = [key for key in ("all_of", "any_of", "none_of") if key in node]
    if len(group_keys) != 1:
        raise CatalogValidationError(
            f"{label}: logic node must contain exactly one of all_of/any_of/none_of"
        )
    group = group_keys[0]
    children = node.get(group)
    if not isinstance(children, list) or not children:
        raise CatalogValidationError(f"{label}: {group} must be a non-empty list")
    for index, child in enumerate(children):
        _validate_logic(child, f"{label}.{group}[{index}]")


def _logic_fact_ids(node: dict | list | None) -> set[str]:
    if not node:
        return set()
    if isinstance(node, list):
        return {fact_id for child in node for fact_id in _logic_fact_ids(child)}
    if "fact_id" in node:
        return {node["fact_id"]}
    fact_ids: set[str] = set()
    for key in ("all_of", "any_of", "none_of"):
        for child in node.get(key, []):
            fact_ids.update(_logic_fact_ids(child))
    return fact_ids


def validate_catalogs(ds: DataStore | None = None) -> None:
    """Validate all source-controlled catalogs against the fact registry and workbook."""
    program_catalog = load_program_catalog(ds)
    rule_matrix = load_program_rule_matrix()
    tariff_catalog = load_tariff_catalog()
    taxonomy_catalog = load_commercial_taxonomy()
    known = known_fact_ids()

    program_ids: set[str] = set()
    for entry in program_catalog:
        program_id = entry.get("program_id")
        if not program_id:
            raise CatalogValidationError("program_catalog: program_id is required")
        if program_id in program_ids:
            raise CatalogValidationError(f"program_catalog: duplicate program_id {program_id}")
        program_ids.add(program_id)

        for field in (
            "display_name",
            "customer_type",
            "offer_family",
            "priority_group",
            "program_rank",
            "eligibility_logic",
            "routing_logic",
            "exclusion_logic",
            "required_facts",
            "question_templates",
            "program_code_aliases",
            "evidence_references",
            "manual_curation_required",
        ):
            if field not in entry:
                raise CatalogValidationError(f"{program_id}: missing required field {field}")

        _validate_logic(entry["eligibility_logic"], f"{program_id}.eligibility_logic")
        _validate_logic(entry.get("routing_logic"), f"{program_id}.routing_logic")
        _validate_logic(entry.get("exclusion_logic"), f"{program_id}.exclusion_logic")

        referenced = (
            _logic_fact_ids(entry["eligibility_logic"])
            | _logic_fact_ids(entry.get("routing_logic"))
            | _logic_fact_ids(entry.get("exclusion_logic"))
        )
        unknown_required = set(entry["required_facts"]) - known
        if unknown_required:
            raise CatalogValidationError(
                f"{program_id}: unknown required fact ids {sorted(unknown_required)}"
            )
        if not set(entry["required_facts"]).issubset(referenced):
            raise CatalogValidationError(
                f"{program_id}: required_facts must be referenced in the logic tree"
            )
        for template in entry["question_templates"]:
            expected_fact = template.get("expected_fact")
            if expected_fact not in known:
                raise CatalogValidationError(
                    f"{program_id}: question template references unknown fact {expected_fact}"
                )

        matrix_entry = rule_matrix.get(program_id)
        if matrix_entry is None:
            raise CatalogValidationError(f"program_rule_matrix: missing entry for {program_id}")
        for field in ("source_documents", "rule_summary", "required_facts", "rule_basis"):
            if field not in matrix_entry:
                raise CatalogValidationError(
                    f"program_rule_matrix[{program_id}]: missing required field {field}"
                )
        if sorted(matrix_entry["required_facts"]) != sorted(entry["required_facts"]):
            raise CatalogValidationError(
                f"program_rule_matrix[{program_id}]: required_facts must match program catalog"
            )

    extra_rule_entries = set(rule_matrix) - program_ids
    if extra_rule_entries:
        raise CatalogValidationError(
            f"program_rule_matrix: entries without program catalog row {sorted(extra_rule_entries)}"
        )

    rate_plan_ids: set[str] = set()
    for entry in tariff_catalog:
        rate_plan = entry.get("rate_plan")
        if not rate_plan:
            raise CatalogValidationError("tariff_catalog: rate_plan is required")
        if rate_plan in rate_plan_ids:
            raise CatalogValidationError(f"tariff_catalog: duplicate rate_plan {rate_plan}")
        rate_plan_ids.add(rate_plan)
        for field in (
            "display_name",
            "customer_type",
            "rate_kind",
            "simulation_supported",
            "new_enrollment_allowed",
            "fixed_charges",
            "energy_rates",
            "demand_rates",
            "seasonality",
            "eligibility_logic",
            "evidence_references",
        ):
            if field not in entry:
                raise CatalogValidationError(f"{rate_plan}: missing tariff field {field}")
        _validate_logic(entry["eligibility_logic"], f"{rate_plan}.eligibility_logic")

    if not taxonomy_catalog or taxonomy_catalog[-1].get("taxonomy") != "GENERAL_BUSINESS":
        raise CatalogValidationError(
            "commercial_taxonomy: GENERAL_BUSINESS fallback must be the last entry"
        )

    if ds is not None:
        known_rate_plans = set(
            ds.active_offering[COL_RATE_PLAN].dropna().astype(str).str.strip().unique()
        )
        unknown_rate_plans = rate_plan_ids - known_rate_plans
        if unknown_rate_plans:
            raise CatalogValidationError(
                f"tariff_catalog: unknown workbook rate plans {sorted(unknown_rate_plans)}"
            )
