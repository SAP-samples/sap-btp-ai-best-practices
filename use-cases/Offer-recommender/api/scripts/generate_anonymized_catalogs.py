#!/usr/bin/env python3
"""
Examples:
  python api/scripts/generate_anonymized_catalogs.py
  python api/scripts/generate_anonymized_catalogs.py --dry-run
  python api/scripts/generate_anonymized_catalogs.py \
    --customer-workbook api/demo_data/data_seed/customer_seed.xlsx \
    --program-workbook api/demo_data/data_seed/program_seed.xlsx \
    --output-dir api/app/nbo/catalogs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "api"

for candidate in (str(REPO_ROOT), str(API_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from app.nbo.config import (  # noqa: E402
    COL_BIZ_OFFERING_NAME,
    COL_BILLING_ACCOUNT,
    COL_COMM_SEGMENT_NAME,
    COL_CUSTOMER_TYPE,
    COL_INDUSTRY,
    COL_NAIC_CODE,
    COL_PC_PROGRAM,
    COL_PC_SCREEN_NAME,
    COL_PC_SERVICE_OPTION,
    COL_RATE_PLAN,
    COL_STATUS,
)
from app.nbo.fact_registry import known_fact_ids  # noqa: E402
from app.nbo.hana_loader import load_seed_datasets  # noqa: E402


DEFAULT_DEMO_DATA_DIR = API_ROOT / "demo_data"
DEFAULT_CUSTOMER_WORKBOOK = DEFAULT_DEMO_DATA_DIR / "data_seed" / "customer_seed.xlsx"
DEFAULT_PROGRAM_WORKBOOK = DEFAULT_DEMO_DATA_DIR / "data_seed" / "program_seed.xlsx"
DEFAULT_OUTPUT_DIR = API_ROOT / "app" / "nbo" / "catalogs"
DEFAULT_SOURCE_DOCUMENT_DIR = DEFAULT_DEMO_DATA_DIR / "source_documents"

FORBIDDEN_PUBLIC_TERMS = (
    "S" + "RP",
    "s" + "rp",
    "Next best " + "Offer",
    "M-" + "Power",
    "I" + "QD",
    "data/" + "Programs",
    "data/" + "Rate Plans",
)

CATALOG_FILENAMES = {
    "program_catalog": "program_catalog.json",
    "program_rule_matrix": "program_rule_matrix.json",
    "tariff_catalog": "tariff_catalog.json",
    "commercial_taxonomy": "commercial_taxonomy.json",
}

PUBLIC_SOURCE_DOCUMENTS = {
    "customer-assistance-guide.pdf",
    "battery-partner-guide.pdf",
    "rate-plan-guide.pdf",
    "offer-guidance-notes.docx",
}

SEASONALITY = {
    "winter_months": [11, 12, 1, 2, 3, 4],
    "summer_months": [5, 6, 9, 10],
    "summer_peak_months": [7, 8],
}
FIXED_CHARGES = {"tier1": 20.0, "tier2": 30.0, "tier3": 40.0}

RATE_TEMPLATES: dict[str, dict[str, Any]] = {
    "E21": {
        "display_name": "Time of Use 3-6",
        "rate_kind": "tou",
        "simulation_supported": True,
        "new_enrollment_allowed": False,
        "energy_rates": {
            "winter": {"off_peak": 0.0994, "on_peak": 0.1508},
            "summer": {"off_peak": 0.1030, "on_peak": 0.3096},
            "summer_peak": {"off_peak": 0.0996, "on_peak": 0.3661},
        },
    },
    "E23": {
        "display_name": "Basic Residential",
        "rate_kind": "flat",
        "simulation_supported": True,
        "new_enrollment_allowed": True,
        "energy_rates": {
            "winter": {"all_kwh": 0.1097},
            "summer": {"all_kwh": 0.1204},
            "summer_peak": {"all_kwh": 0.1398},
        },
    },
    "E24": {
        "display_name": "Prepay",
        "rate_kind": "flat",
        "simulation_supported": False,
        "new_enrollment_allowed": False,
        "energy_rates": {
            "winter": {"all_kwh": 0.1097},
            "summer": {"all_kwh": 0.1204},
            "summer_peak": {"all_kwh": 0.1398},
        },
        "eligibility_logic": {
            "all_of": [{"fact_id": "is_mpower_enrolled", "op": "eq", "value": True}]
        },
    },
    "E26": {
        "display_name": "Time of Use",
        "rate_kind": "tou",
        "simulation_supported": True,
        "new_enrollment_allowed": False,
        "energy_rates": {
            "winter": {"off_peak": 0.0963, "on_peak": 0.1294},
            "summer": {"off_peak": 0.0995, "on_peak": 0.1885},
            "summer_peak": {"off_peak": 0.1069, "on_peak": 0.2604},
        },
    },
    "E16": {
        "display_name": "Demand Saver 5-10 P.M.",
        "rate_kind": "demand",
        "simulation_supported": True,
        "new_enrollment_allowed": True,
        "energy_rates": {
            "winter": {"super_off_peak": 0.0438, "off_peak": 0.0994, "on_peak": 0.1119},
            "summer": {"super_off_peak": 0.0393, "off_peak": 0.0995, "on_peak": 0.1257},
            "summer_peak": {"super_off_peak": 0.0622, "off_peak": 0.0996, "on_peak": 0.1654},
        },
        "demand_rates": {
            "winter": {"avg_on_peak_daily_kw": 9.61},
            "summer": {"avg_on_peak_daily_kw": 13.56},
            "summer_peak": {"avg_on_peak_daily_kw": 17.78},
        },
    },
    "E28": {
        "display_name": "Conserve 6-9 P.M.",
        "rate_kind": "tou_with_super_off_peak",
        "simulation_supported": True,
        "new_enrollment_allowed": True,
        "energy_rates": {
            "winter": {"super_off_peak": 0.0432, "off_peak": 0.1355, "on_peak": 0.1508},
            "summer": {"super_off_peak": 0.0395, "off_peak": 0.1506, "on_peak": 0.1885},
            "summer_peak": {"super_off_peak": 0.0661, "off_peak": 0.1276, "on_peak": 0.4020},
        },
    },
    "E29": {
        "display_name": "EV Time of Use",
        "rate_kind": "tou_with_super_off_peak",
        "simulation_supported": False,
        "new_enrollment_allowed": False,
        "energy_rates": {
            "winter": {"super_off_peak": 0.0792, "off_peak": 0.0963, "on_peak": 0.1097},
            "summer": {"super_off_peak": 0.0793, "off_peak": 0.0964, "on_peak": 0.2195},
            "summer_peak": {"super_off_peak": 0.0794, "off_peak": 0.0965, "on_peak": 0.2604},
        },
        "eligibility_logic": {
            "all_of": [{"fact_id": "ev_ownership", "op": "eq", "value": True}]
        },
    },
}


def _parse_args() -> argparse.Namespace:
    """Parse catalog generation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate public Customer Offer Advisor catalog JSON from demo seed workbooks."
    )
    parser.add_argument(
        "--customer-workbook",
        type=Path,
        default=DEFAULT_CUSTOMER_WORKBOOK,
        help="Customer/rate-plan workbook used to scope generated catalogs.",
    )
    parser.add_argument(
        "--program-workbook",
        "--program-codes-workbook",
        dest="program_workbook",
        type=Path,
        default=DEFAULT_PROGRAM_WORKBOOK,
        help="Program contract workbook used to derive public program aliases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where catalog JSON files should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate payloads without writing catalog files.",
    )
    return parser.parse_args()


def _clean_text(value: object) -> str:
    """Return a stripped string while treating NaN-like values as blank."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_key(value: str) -> str:
    """Normalize one workbook label for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _walk_values(value):
    """Yield scalar values from nested JSON-compatible data."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _logic_fact_ids(node: dict | list | None) -> set[str]:
    """Return all fact IDs referenced by a catalog logic tree."""
    if not node:
        return set()
    if isinstance(node, list):
        return {fact_id for child in node for fact_id in _logic_fact_ids(child)}
    if "fact_id" in node:
        return {str(node["fact_id"])}
    fact_ids: set[str] = set()
    for key in ("all_of", "any_of", "none_of"):
        for child in node.get(key, []):
            fact_ids.update(_logic_fact_ids(child))
    return fact_ids


def _program_contract_metadata(datasets: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Derive public program display names and aliases from Program Contract rows."""
    metadata = {
        "income_qualified_discount": {
            "display_name": "Household Assistance Discount",
            "aliases": [],
        },
        "battery_partner": {
            "display_name": "Battery Partner",
            "aliases": [],
        },
    }
    program_contract = datasets["program_contract"]
    for _, row in program_contract.iterrows():
        program_name = _clean_text(row.get(COL_PC_PROGRAM))
        screen_name = _clean_text(row.get(COL_PC_SCREEN_NAME))
        service_option = _clean_text(row.get(COL_PC_SERVICE_OPTION))
        searchable = _normalize_key(f"{program_name} {screen_name}")

        if "household assistance discount" in searchable:
            metadata["income_qualified_discount"]["display_name"] = program_name or metadata[
                "income_qualified_discount"
            ]["display_name"]
            if service_option:
                metadata["income_qualified_discount"]["aliases"].append(service_option)
        elif "battery" in searchable:
            metadata["battery_partner"]["display_name"] = screen_name or program_name or metadata[
                "battery_partner"
            ]["display_name"]
            if service_option:
                metadata["battery_partner"]["aliases"].append(service_option)

    for entry in metadata.values():
        entry["aliases"] = sorted(set(entry["aliases"]))
    return metadata


def _has_prepay_demo_scenario(datasets: dict[str, pd.DataFrame]) -> bool:
    """Return true when demo data includes a disconnected prepay account scenario."""
    active_offering = datasets["active_offering"]
    prepay_rate_plans = set(
        active_offering.loc[
            active_offering[COL_BIZ_OFFERING_NAME].str.contains("prepay", case=False, na=False),
            COL_RATE_PLAN,
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )
    if not prepay_rate_plans:
        return False

    residential = datasets["residential"]
    disconnected = residential[COL_STATUS].astype(str).str.upper().eq("DISCONNECTED")
    on_prepay = residential[COL_RATE_PLAN].astype(str).str.strip().isin(prepay_rate_plans)
    return bool((disconnected & on_prepay).any())


def _program_entries(datasets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Build the public program catalog entries supported by demo workbooks."""
    metadata = _program_contract_metadata(datasets)
    entries: list[dict[str, Any]] = []

    if _has_prepay_demo_scenario(datasets):
        entries.append(
            {
                "program_id": "prepay_advance",
                "display_name": "Prepay Advance",
                "customer_type": "RESIDENTIAL",
                "offer_family": "payment_assistance",
                "priority_group": 1,
                "program_rank": 10,
                "eligibility_logic": {
                    "all_of": [
                        {"fact_id": "current_status", "op": "eq", "value": "DISCONNECTED"},
                        {"fact_id": "is_mpower_enrolled", "op": "eq", "value": True},
                        {
                            "fact_id": "prepay_advance_offers_this_month",
                            "op": "lt",
                            "value": 2,
                        },
                    ]
                },
                "routing_logic": {},
                "exclusion_logic": {},
                "required_facts": [
                    "current_status",
                    "is_mpower_enrolled",
                    "prepay_advance_offers_this_month",
                ],
                "question_templates": [],
                "program_code_aliases": [],
                "evidence_references": [
                    "offer-guidance-notes.docx",
                    "rate-plan-guide.pdf",
                ],
                "manual_curation_required": False,
                "rule_basis": "demo_workbook_backed",
            }
        )

    entries.extend(
        [
            {
                "program_id": "income_qualified_discount",
                "display_name": metadata["income_qualified_discount"]["display_name"],
                "customer_type": "RESIDENTIAL",
                "offer_family": "payment_assistance",
                "priority_group": 3,
                "program_rank": 30,
                "eligibility_logic": {
                    "all_of": [
                        {
                            "fact_id": "residential_price_plan_customer",
                            "op": "eq",
                            "value": True,
                        },
                        {
                            "any_of": [
                                {
                                    "fact_id": "household_income_qualified",
                                    "op": "eq",
                                    "value": True,
                                },
                                {
                                    "fact_id": "income_assistance_auto_qualifier",
                                    "op": "eq",
                                    "value": True,
                                },
                            ]
                        },
                        {
                            "fact_id": "customer_of_record_on_site",
                            "op": "eq",
                            "value": True,
                        },
                    ]
                },
                "routing_logic": {},
                "exclusion_logic": {
                    "any_of": [{"fact_id": "account_name_type", "op": "eq", "value": "COMPANY"}]
                },
                "required_facts": [
                    "residential_price_plan_customer",
                    "household_income_qualified",
                    "income_assistance_auto_qualifier",
                    "customer_of_record_on_site",
                    "account_name_type",
                ],
                "question_templates": [],
                "program_code_aliases": metadata["income_qualified_discount"]["aliases"],
                "evidence_references": ["customer-assistance-guide.pdf"],
                "manual_curation_required": False,
                "rule_basis": "demo_workbook_backed",
            },
            {
                "program_id": "battery_partner",
                "display_name": metadata["battery_partner"]["display_name"],
                "customer_type": "RESIDENTIAL",
                "offer_family": "battery_incentive",
                "priority_group": 6,
                "program_rank": 70,
                "eligibility_logic": {
                    "all_of": [
                        {"fact_id": "customer_wants_followup", "op": "eq", "value": True},
                        {"fact_id": "battery_ownership", "op": "eq", "value": True},
                        {
                            "fact_id": "battery_partner_brand_supported",
                            "op": "eq",
                            "value": True,
                        },
                    ]
                },
                "routing_logic": {
                    "all_of": [
                        {"fact_id": "customer_wants_followup", "op": "eq", "value": True},
                        {"fact_id": "payments_on_time", "op": "eq", "value": True},
                        {
                            "fact_id": "bill_increase_or_high_usage",
                            "op": "eq",
                            "value": True,
                        },
                    ]
                },
                "exclusion_logic": {},
                "required_facts": [
                    "customer_wants_followup",
                    "battery_ownership",
                    "battery_partner_brand_supported",
                    "payments_on_time",
                    "bill_increase_or_high_usage",
                ],
                "question_templates": [],
                "program_code_aliases": metadata["battery_partner"]["aliases"],
                "evidence_references": ["battery-partner-guide.pdf"],
                "manual_curation_required": False,
                "rule_basis": "demo_workbook_backed",
            },
        ]
    )
    return entries


def _program_rule_matrix(program_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a rule matrix whose program IDs and required facts mirror the catalog."""
    summaries = {
        "prepay_advance": (
            "Disconnected residential prepay accounts can receive a Prepay Advance "
            "when fewer than two offers have already been made this month."
        ),
        "income_qualified_discount": (
            "Household Assistance Discount requires a residential price-plan customer, "
            "the customer of record on site, and either income qualification or an "
            "income-assistance auto qualifier; company-name accounts are excluded."
        ),
        "battery_partner": (
            "Battery Partner requires customer follow-up intent, battery ownership, "
            "a supported partner brand, on-time payments, and a bill or usage signal."
        ),
    }
    return [
        {
            "program_id": entry["program_id"],
            "source_documents": entry["evidence_references"],
            "rule_summary": summaries[entry["program_id"]],
            "required_facts": entry["required_facts"],
            "rule_basis": entry["rule_basis"],
        }
        for entry in program_catalog
    ]


def _rate_plan_display_names(datasets: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return display names from the active-offering workbook sheet."""
    display_names: dict[str, str] = {}
    for _, row in datasets["active_offering"].iterrows():
        rate_plan = _clean_text(row.get(COL_RATE_PLAN))
        offering_name = _clean_text(row.get(COL_BIZ_OFFERING_NAME))
        if not rate_plan:
            continue
        display_names[rate_plan] = re.sub(rf"^{re.escape(rate_plan)}[- ]*", "", offering_name).strip()
    return display_names


def _tariff_catalog(datasets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Build a filtered tariff catalog from active-offering rate plans and demo templates."""
    active_rate_plans = set(
        datasets["active_offering"][COL_RATE_PLAN].dropna().astype(str).str.strip()
    )
    customer_type_by_rate = {
        _clean_text(row.get(COL_RATE_PLAN)): _clean_text(row.get(COL_CUSTOMER_TYPE)) or "RESIDENTIAL"
        for _, row in datasets["active_offering"].iterrows()
    }
    display_names = _rate_plan_display_names(datasets)
    entries: list[dict[str, Any]] = []
    for rate_plan, template in RATE_TEMPLATES.items():
        if rate_plan not in active_rate_plans:
            continue
        entries.append(
            {
                "rate_plan": rate_plan,
                "display_name": display_names.get(rate_plan) or template["display_name"],
                "customer_type": customer_type_by_rate.get(rate_plan, "RESIDENTIAL"),
                "rate_kind": template["rate_kind"],
                "simulation_supported": template["simulation_supported"],
                "new_enrollment_allowed": template["new_enrollment_allowed"],
                "fixed_charges": FIXED_CHARGES,
                "energy_rates": template["energy_rates"],
                "demand_rates": template.get("demand_rates", {}),
                "seasonality": SEASONALITY,
                "eligibility_logic": template.get("eligibility_logic", {}),
                "evidence_references": ["rate-plan-guide.pdf"],
            }
        )
    return entries


def _commercial_taxonomy(datasets: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Build a compact commercial taxonomy from commercial demo seed rows."""
    commercial = datasets["commercial"]
    segment_names = sorted(
        {
            _clean_text(value)
            for value in datasets["comm_segment"].get(COL_COMM_SEGMENT_NAME, pd.Series(dtype=object))
            if _clean_text(value)
        }
    )
    entries: list[dict[str, Any]] = []
    for _, row in commercial[[COL_NAIC_CODE, COL_INDUSTRY]].drop_duplicates().iterrows():
        naic = _clean_text(row.get(COL_NAIC_CODE))
        industry = _clean_text(row.get(COL_INDUSTRY))
        if not naic and not industry:
            continue
        taxonomy = "UTILITY_SERVICES" if naic.startswith("221") else "DEMO_COMMERCIAL"
        keywords = sorted({token for token in [industry.casefold(), *[s.casefold() for s in segment_names]] if token})
        prefixes = [naic[:3]] if naic else []
        entries.append(
            {
                "taxonomy": taxonomy,
                "naic_prefixes": prefixes,
                "industry_keywords": keywords,
            }
        )

    entries.append(
        {
            "taxonomy": "GENERAL_BUSINESS",
            "naic_prefixes": [],
            "industry_keywords": [],
        }
    )
    return entries


def generate_catalog_payloads(
    customer_workbook: Path = DEFAULT_CUSTOMER_WORKBOOK,
    program_workbook: Path = DEFAULT_PROGRAM_WORKBOOK,
) -> dict[str, list[dict[str, Any]]]:
    """Create all catalog payloads from explicit public demo workbooks.

    Inputs:
        customer_workbook: Workbook containing customer, segment, commercial,
            and active-offering sheets.
        program_workbook: Workbook containing program contract and sample
            account sheets.

    Output:
        Mapping of catalog base names to JSON-serializable catalog rows.
    """
    datasets = load_seed_datasets(
        customer_workbook=Path(customer_workbook),
        program_codes_workbook=Path(program_workbook),
    )
    program_catalog = _program_entries(datasets)
    payloads = {
        "program_catalog": program_catalog,
        "program_rule_matrix": _program_rule_matrix(program_catalog),
        "tariff_catalog": _tariff_catalog(datasets),
        "commercial_taxonomy": _commercial_taxonomy(datasets),
    }
    validate_public_catalog_payloads(payloads)
    return payloads


def public_source_document_path(source_document: str) -> Path:
    """Return the public demo placeholder path for a source-document filename."""
    return DEFAULT_SOURCE_DOCUMENT_DIR / source_document


def _validate_logic(node: dict | list | None, label: str) -> None:
    """Validate one generated logic node against supported operators and facts."""
    if not node:
        return
    if isinstance(node, list):
        for index, child in enumerate(node):
            _validate_logic(child, f"{label}[{index}]")
        return
    if "fact_id" in node:
        if node["fact_id"] not in known_fact_ids():
            raise ValueError(f"{label}: unknown fact {node['fact_id']!r}")
        if node.get("op") not in {"eq", "lt", "lte", "gt", "gte", "in", "contains"}:
            raise ValueError(f"{label}: unsupported operator {node.get('op')!r}")
        return
    keys = [key for key in ("all_of", "any_of", "none_of") if key in node]
    if len(keys) != 1:
        raise ValueError(f"{label}: logic node must contain exactly one group")
    for index, child in enumerate(node[keys[0]]):
        _validate_logic(child, f"{label}.{keys[0]}[{index}]")


def _source_documents(value) -> set[str]:
    """Collect source-document and evidence filenames from generated payloads."""
    sources: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_documents", "evidence_references"}:
                sources.update(str(source) for source in item)
            else:
                sources.update(_source_documents(item))
    elif isinstance(value, list):
        for item in value:
            sources.update(_source_documents(item))
    return sources


def validate_public_catalog_payloads(payloads: dict[str, list[dict[str, Any]]]) -> None:
    """Validate generated catalog payloads for public safety and runtime shape."""
    joined_values = "\n".join(
        str(value)
        for payload in payloads.values()
        for value in _walk_values(payload)
    )
    for term in FORBIDDEN_PUBLIC_TERMS:
        if term in joined_values:
            raise ValueError(f"Generated public catalogs contain forbidden term {term!r}")

    program_ids = {entry["program_id"] for entry in payloads.get("program_catalog", [])}
    matrix_ids = {entry["program_id"] for entry in payloads.get("program_rule_matrix", [])}
    if program_ids != matrix_ids:
        raise ValueError("Program catalog and rule matrix program IDs must match exactly")

    known = known_fact_ids()
    for entry in payloads.get("program_catalog", []):
        program_id = entry["program_id"]
        _validate_logic(entry.get("eligibility_logic"), f"{program_id}.eligibility_logic")
        _validate_logic(entry.get("routing_logic"), f"{program_id}.routing_logic")
        _validate_logic(entry.get("exclusion_logic"), f"{program_id}.exclusion_logic")
        referenced = (
            _logic_fact_ids(entry.get("eligibility_logic"))
            | _logic_fact_ids(entry.get("routing_logic"))
            | _logic_fact_ids(entry.get("exclusion_logic"))
        )
        required = set(entry.get("required_facts", []))
        unknown_required = required - known
        if unknown_required:
            raise ValueError(f"{program_id}: unknown required facts {sorted(unknown_required)}")
        if not required.issubset(referenced):
            raise ValueError(f"{program_id}: required facts must be referenced by logic")

    for entry in payloads.get("program_rule_matrix", []):
        program_entry = next(
            item for item in payloads["program_catalog"] if item["program_id"] == entry["program_id"]
        )
        if sorted(entry["required_facts"]) != sorted(program_entry["required_facts"]):
            raise ValueError(f"{entry['program_id']}: rule matrix required facts do not match")

    for entry in payloads.get("tariff_catalog", []):
        _validate_logic(entry.get("eligibility_logic"), f"{entry['rate_plan']}.eligibility_logic")

    for source in _source_documents(payloads):
        if source != Path(source).name or "/" in source or "\\" in source:
            raise ValueError(f"Source document must be a final filename: {source}")
        if source not in PUBLIC_SOURCE_DOCUMENTS:
            raise ValueError(f"Unexpected public source document: {source}")


def _write_catalog(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write one generated catalog with deterministic formatting."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def generate_catalogs(
    customer_workbook: Path = DEFAULT_CUSTOMER_WORKBOOK,
    program_workbook: Path = DEFAULT_PROGRAM_WORKBOOK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """Generate, validate, and optionally write the public catalog JSON files.

    Inputs:
        customer_workbook: Public demo customer workbook.
        program_workbook: Public demo program workbook.
        output_dir: Destination directory for catalog JSON files.
        dry_run: When true, validate without writing files.

    Output:
        Small summary with generated row counts and destination information.
    """
    payloads = generate_catalog_payloads(
        customer_workbook=Path(customer_workbook),
        program_workbook=Path(program_workbook),
    )
    output_path = Path(output_dir).expanduser().resolve()
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        for catalog_name, filename in CATALOG_FILENAMES.items():
            _write_catalog(output_path / filename, payloads[catalog_name])

    return {
        "program_count": len(payloads["program_catalog"]),
        "rule_count": len(payloads["program_rule_matrix"]),
        "tariff_count": len(payloads["tariff_catalog"]),
        "taxonomy_count": len(payloads["commercial_taxonomy"]),
        "output_dir": str(output_path),
        "dry_run": int(dry_run),
    }


def main() -> int:
    """CLI entrypoint for public catalog generation."""
    args = _parse_args()
    summary = generate_catalogs(
        customer_workbook=args.customer_workbook,
        program_workbook=args.program_workbook,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Generated"
    print(f"{action} public Customer Offer Advisor catalogs:")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  programs: {summary['program_count']}")
    print(f"  rules: {summary['rule_count']}")
    print(f"  tariffs: {summary['tariff_count']}")
    print(f"  taxonomy rows: {summary['taxonomy_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
