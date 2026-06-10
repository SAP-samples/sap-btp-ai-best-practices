#!/usr/bin/env python3
"""
Generate synthetic extraction data for multi-week optimizer and RPT-1 evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "synthetic"
SHEET_NAME = "SAPUI5 Export"


@dataclass(frozen=True)
class GeneratorConfig:
    start_date: str
    weeks: int = 8
    invoices_per_week: int = 40
    history_weeks: int = 4
    history_invoices_per_week: int | None = None
    companies: int = 2
    customers_per_company: int = 10
    groups_per_company: int = 2
    seed: int = 42
    base_exposure_ratio: float = 0.08
    facility_utilization_target: float = 0.75
    facility_limit_fraction_of_total: float | None = None
    customer_limit_fraction_of_facility: float = 0.20
    group_limit_fraction_of_facility: float = 0.45
    lifetime_min_days: int = 14
    lifetime_max_days: int = 112
    due_min_days_after_offer: int = 7
    due_max_days_after_offer: int = 140
    enable_reconciliation_date: bool = False
    output_root: Path = DEFAULT_OUTPUT_ROOT
    scenario_name: str | None = None


def _scenario_name(config: GeneratorConfig) -> str:
    if config.scenario_name:
        return config.scenario_name
    return (
        f"synthetic_{config.start_date}"
        f"_w{config.weeks}"
        f"_n{config.invoices_per_week}"
        f"_seed{config.seed}"
    )


def _week_dates(start_date: str, weeks: int) -> List[pd.Timestamp]:
    start = pd.to_datetime(start_date).to_period("W-MON").start_time
    return [start + pd.Timedelta(weeks=i) for i in range(max(0, int(weeks)))]


def _build_entities(config: GeneratorConfig) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    companies = [f"C{i + 1:03d}" for i in range(config.companies)]
    customers_by_company: Dict[str, List[str]] = {}
    customer_to_company: Dict[str, str] = {}
    customer_to_group: Dict[str, str] = {}

    next_customer_id = 1
    for company in companies:
        groups = [
            f"G_{company}_{j + 1:02d}"
            for j in range(max(1, config.groups_per_company))
        ]
        company_customers: List[str] = []
        for local_idx in range(max(1, config.customers_per_company)):
            customer = f"CU{next_customer_id:04d}"
            next_customer_id += 1
            company_customers.append(customer)
            customer_to_company[customer] = company
            customer_to_group[customer] = groups[local_idx % len(groups)]
        customers_by_company[company] = company_customers

    return companies, customers_by_company, customer_to_company, customer_to_group


def _sample_amount(rng: random.Random) -> float:
    base = rng.triangular(500.0, 60000.0, 12000.0)
    return round(base, 2)


def _create_row(
    *,
    invoice_ref: str,
    company: str,
    customer: str,
    offer_file_date: pd.Timestamp,
    rng: random.Random,
    enable_reconciliation_date: bool,
    row_type: str,
    week_index: int,
    lifetime_min_days: int,
    lifetime_max_days: int,
    due_min_days_after_offer: int,
    due_max_days_after_offer: int,
) -> Dict[str, Any]:
    issuance_offset_days = rng.randint(7, 45)
    days_to_due_after_offer = rng.randint(due_min_days_after_offer, due_max_days_after_offer)
    summary_offset_days = rng.randint(1, 2)
    lifetime_days = rng.randint(lifetime_min_days, lifetime_max_days)

    issuance_date = offer_file_date - pd.Timedelta(days=issuance_offset_days)
    due_date = offer_file_date + pd.Timedelta(days=days_to_due_after_offer)
    summary_file_date = offer_file_date + pd.Timedelta(days=summary_offset_days)
    if enable_reconciliation_date:
        reconciliation_file_date = summary_file_date + pd.Timedelta(days=lifetime_days)
        paid_on_date = reconciliation_file_date + pd.Timedelta(days=rng.randint(0, 2))
    else:
        reconciliation_file_date = pd.NaT
        paid_on_date = pd.NaT

    purchase_price = _sample_amount(rng)
    invoice_amount = round(purchase_price * rng.uniform(1.0, 1.25), 2)

    return {
        "Status": "Accepted",
        "Company Code": company,
        "Customer": customer,
        "Amount": invoice_amount,
        "Due Date": due_date,
        "Document Number": invoice_ref,
        "Invoice Reference": invoice_ref,
        "Reference Key": f"RK-{invoice_ref}",
        "Document Indicator": "I",
        "Document Type": "DR",
        "Issuance date": issuance_date,
        "Offer File Date (UTC)": offer_file_date,
        "Summary File Date (UTC)": summary_file_date,
        "Fiscal Year": int(offer_file_date.year),
        "Pledging Indicator": "Y",
        "Document Item": 1,
        "Reconciliation File Date (UTC)": reconciliation_file_date,
        "All in Rate": round(rng.uniform(3.0, 6.0), 6),
        "Amount (Funding CCY)": invoice_amount,
        "Base Rate": round(rng.uniform(1.0, 3.0), 6),
        "Currency": "EUR",
        "Funding Currency": "EUR",
        "Dispatch Date": issuance_date + pd.Timedelta(days=rng.randint(0, 2)),
        "Interests": round(rng.uniform(0.0, 150.0), 2),
        "Margin": round(rng.uniform(0.5, 3.5), 6),
        "Net Value": purchase_price,
        "Outstanding Amount": invoice_amount,
        "Paid On (Europe, Madrid)": paid_on_date,
        "Payment Block": "",
        "Purchase Date": summary_file_date,
        "Purchase Price": purchase_price,
        "Rate": round(rng.uniform(0.95, 1.05), 6),
        "Reason": pd.NA,
        "Repurchase Date": pd.NaT,
        "Repurchase": 0.0,
        "PROGRAMA": "SYNTHETIC PROGRAM",
        "Synthetic Row Type": row_type,
        "Synthetic Week Index": int(week_index),
    }


def _build_rows(config: GeneratorConfig) -> pd.DataFrame:
    rng = random.Random(config.seed)
    (
        companies,
        customers_by_company,
        _customer_to_company,
        _customer_to_group,
    ) = _build_entities(config)
    cohort_weeks = _week_dates(config.start_date, config.weeks)

    rows: List[Dict[str, Any]] = []
    invoice_seq = 1

    for history_idx in range(max(0, int(config.history_weeks))):
        hist_week = cohort_weeks[0] - pd.Timedelta(weeks=(max(0, int(config.history_weeks)) - history_idx))
        if config.history_invoices_per_week is not None:
            history_count = max(1, int(config.history_invoices_per_week))
        else:
            history_count = max(1, int(config.invoices_per_week) // 3)
        for local_idx in range(history_count):
            company = companies[local_idx % len(companies)]
            customers = customers_by_company[company]
            customer = customers[rng.randrange(len(customers))]
            invoice_ref = f"HIST-{invoice_seq:06d}"
            invoice_seq += 1
            rows.append(
                _create_row(
                    invoice_ref=invoice_ref,
                    company=company,
                    customer=customer,
                    offer_file_date=hist_week,
                    rng=rng,
                    enable_reconciliation_date=True,
                    row_type="history",
                    week_index=-(max(0, int(config.history_weeks)) - history_idx),
                    lifetime_min_days=int(config.lifetime_min_days),
                    lifetime_max_days=int(config.lifetime_max_days),
                    due_min_days_after_offer=int(config.due_min_days_after_offer),
                    due_max_days_after_offer=int(config.due_max_days_after_offer),
                )
            )

    for week_idx, offer_week in enumerate(cohort_weeks):
        for local_idx in range(max(0, int(config.invoices_per_week))):
            company = companies[local_idx % len(companies)]
            customers = customers_by_company[company]
            customer = customers[rng.randrange(len(customers))]
            invoice_ref = f"INV-{week_idx + 1:02d}-{invoice_seq:06d}"
            invoice_seq += 1
            rows.append(
                _create_row(
                    invoice_ref=invoice_ref,
                    company=company,
                    customer=customer,
                    offer_file_date=offer_week,
                    rng=rng,
                    enable_reconciliation_date=config.enable_reconciliation_date,
                    row_type="candidate",
                    week_index=week_idx,
                    lifetime_min_days=int(config.lifetime_min_days),
                    lifetime_max_days=int(config.lifetime_max_days),
                    due_min_days_after_offer=int(config.due_min_days_after_offer),
                    due_max_days_after_offer=int(config.due_max_days_after_offer),
                )
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Offer File Date (UTC)", "Synthetic Row Type", "Invoice Reference"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def _build_limits_payload(
    data_df: pd.DataFrame,
    config: GeneratorConfig,
    customer_to_group: Dict[str, str],
) -> Dict[str, Any]:
    candidate_df = data_df[data_df["Synthetic Row Type"] == "candidate"].copy()
    company_totals = candidate_df.groupby("Company Code", dropna=False)["Purchase Price"].sum()

    if config.facility_limit_fraction_of_total is not None:
        fraction = max(0.01, float(config.facility_limit_fraction_of_total))
        facility_limits = {
            str(company): round(float(total) * fraction, 2)
            for company, total in company_totals.items()
        }
    else:
        utilization = max(0.05, min(0.99, float(config.facility_utilization_target)))
        facility_limits = {
            str(company): round(float(total) / utilization, 2)
            for company, total in company_totals.items()
        }

    customer_limits: Dict[str, float] = {}
    for customer, company in (
        candidate_df[["Customer", "Company Code"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        facility = facility_limits.get(str(company), 0.0)
        customer_limits[str(customer)] = round(
            float(facility) * float(config.customer_limit_fraction_of_facility), 2
        )

    active_customer_to_group = {
        customer: group
        for customer, group in customer_to_group.items()
        if customer in customer_limits
    }

    group_limits: Dict[str, float] = {}
    for customer, group in active_customer_to_group.items():
        company = candidate_df.loc[candidate_df["Customer"] == customer, "Company Code"]
        if company.empty:
            continue
        facility = facility_limits.get(str(company.iloc[0]), 0.0)
        group_limits[group] = round(
            max(group_limits.get(group, 0.0), facility * float(config.group_limit_fraction_of_facility)),
            2,
        )

    base_ratio = max(0.0, min(0.95, float(config.base_exposure_ratio)))
    base_customer = {
        customer: round(limit * base_ratio, 2)
        for customer, limit in customer_limits.items()
    }
    base_group: Dict[str, float] = {}
    for customer, group in active_customer_to_group.items():
        if customer not in base_customer:
            continue
        base_group[group] = round(base_group.get(group, 0.0) + base_customer[customer], 2)
    for group, limit in group_limits.items():
        if group in base_group:
            base_group[group] = min(base_group[group], round(limit * 0.9, 2))

    base_facility: Dict[str, float] = {}
    for customer, company in (
        candidate_df[["Customer", "Company Code"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        if customer not in base_customer:
            continue
        base_facility[str(company)] = round(
            base_facility.get(str(company), 0.0) + base_customer[customer],
            2,
        )
    for company, limit in facility_limits.items():
        if company in base_facility:
            base_facility[company] = min(base_facility[company], round(limit * 0.9, 2))

    return {
        "facility_limits_by_company_code": facility_limits,
        "customer_limits": customer_limits,
        "group_limits": group_limits,
        "customer_to_group": active_customer_to_group,
        "base_exposure": {
            "facility": base_facility,
            "customer": base_customer,
            "group": base_group,
        },
        "defaults": {
            "customer_limit_fraction_of_facility": float(config.customer_limit_fraction_of_facility),
            "group_limit_fraction_of_facility": float(config.group_limit_fraction_of_facility),
        },
        "synthetic_generation": {
            "enabled": False,
            "alpha": 0.85,
            "beta": float(config.customer_limit_fraction_of_facility),
            "gamma": float(config.group_limit_fraction_of_facility),
        },
    }


def _build_manifest(
    data_df: pd.DataFrame,
    config: GeneratorConfig,
    limits_payload: Dict[str, Any],
    scenario_dir: Path,
) -> Dict[str, Any]:
    candidate_df = data_df[data_df["Synthetic Row Type"] == "candidate"].copy()
    candidate_df["offer_file_date_iso"] = pd.to_datetime(
        candidate_df["Offer File Date (UTC)"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    weekly_stats = (
        candidate_df.groupby("offer_file_date_iso")
        .agg(invoice_count=("Invoice Reference", "size"), total_amount=("Purchase Price", "sum"))
        .reset_index()
    )
    weekly_rows: List[Dict[str, Any]] = []
    for _, row in weekly_stats.iterrows():
        weekly_rows.append(
            {
                "offer_file_date": str(row["offer_file_date_iso"]),
                "invoice_count": int(row["invoice_count"]),
                "total_amount": round(float(row["total_amount"]), 2),
            }
        )

    return {
        "scenario_name": _scenario_name(config),
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "generator_config": {
            "start_date": config.start_date,
            "weeks": int(config.weeks),
            "invoices_per_week": int(config.invoices_per_week),
            "history_weeks": int(config.history_weeks),
            "history_invoices_per_week": (
                None if config.history_invoices_per_week is None else int(config.history_invoices_per_week)
            ),
            "companies": int(config.companies),
            "customers_per_company": int(config.customers_per_company),
            "groups_per_company": int(config.groups_per_company),
            "seed": int(config.seed),
            "facility_limit_fraction_of_total": (
                None
                if config.facility_limit_fraction_of_total is None
                else float(config.facility_limit_fraction_of_total)
            ),
            "lifetime_min_days": int(config.lifetime_min_days),
            "lifetime_max_days": int(config.lifetime_max_days),
            "due_min_days_after_offer": int(config.due_min_days_after_offer),
            "due_max_days_after_offer": int(config.due_max_days_after_offer),
            "enable_reconciliation_date": bool(config.enable_reconciliation_date),
        },
        "paths": {
            "scenario_dir": str(scenario_dir),
            "synthetic_extraction": str(scenario_dir / "synthetic_extraction.xlsx"),
            "limits_yaml": str(scenario_dir / "limits.yaml"),
            "manifest_json": str(scenario_dir / "scenario_manifest.json"),
            "readme": str(scenario_dir / "README.md"),
        },
        "rows": {
            "total": int(len(data_df)),
            "candidate": int((data_df["Synthetic Row Type"] == "candidate").sum()),
            "history": int((data_df["Synthetic Row Type"] == "history").sum()),
        },
        "weekly_candidate_stats": weekly_rows,
        "limits_summary": {
            "facility_count": len(limits_payload["facility_limits_by_company_code"]),
            "customer_count": len(limits_payload["customer_limits"]),
            "group_count": len(limits_payload["group_limits"]),
        },
    }


def _write_readme(
    scenario_dir: Path,
    config: GeneratorConfig,
    manifest: Dict[str, Any],
) -> None:
    readme_path = scenario_dir / "README.md"
    readme = (
        "# Synthetic Optimizer Scenario\n\n"
        f"- Scenario: `{_scenario_name(config)}`\n"
        f"- Start date: `{config.start_date}`\n"
        f"- Weeks: `{config.weeks}`\n"
        f"- Invoices/week: `{config.invoices_per_week}`\n"
        f"- Reconciliation date generated: `{config.enable_reconciliation_date}`\n\n"
        "## Files\n\n"
        "- `synthetic_extraction.xlsx` (sheet `SAPUI5 Export`)\n"
        "- `limits.yaml`\n"
        "- `scenario_manifest.json`\n\n"
        "## Cohort semantics\n\n"
        "- `Offer File Date (UTC)` is the weekly cohort anchor.\n"
        "- `Synthetic Row Type = candidate` are the rows intended for optimization runs.\n"
        "- `Synthetic Row Type = history` are pre-start rows to emulate carry-over.\n\n"
        "## RPT-1 reliability\n\n"
        "- Use `api/scripts/evaluate_rpt1_on_synthetic.py`.\n"
        "- Default RPT-1 context input is `data/2026/EXTRACTION BTP.xlsx`.\n"
        "- Synthetic reliability evaluation requires reconciliation dates on candidate rows.\n\n"
        "## Generated summary\n\n"
        f"- Candidate rows: `{manifest['rows']['candidate']}`\n"
        f"- History rows: `{manifest['rows']['history']}`\n"
    )
    readme_path.write_text(readme, encoding="utf-8")


def generate_synthetic_package(config: GeneratorConfig) -> Dict[str, Any]:
    if config.weeks <= 0:
        raise ValueError("--weeks must be > 0")
    if config.invoices_per_week <= 0:
        raise ValueError("--invoices-per-week must be > 0")
    if config.history_weeks < 0:
        raise ValueError("--history-weeks must be >= 0")
    if config.history_invoices_per_week is not None and config.history_invoices_per_week <= 0:
        raise ValueError("--history-invoices-per-week must be > 0")
    if config.companies <= 0:
        raise ValueError("--companies must be > 0")
    if config.customers_per_company <= 0:
        raise ValueError("--customers-per-company must be > 0")
    if config.lifetime_min_days <= 0:
        raise ValueError("--lifetime-min-days must be > 0")
    if config.lifetime_max_days < config.lifetime_min_days:
        raise ValueError("--lifetime-max-days must be >= --lifetime-min-days")
    if config.due_min_days_after_offer <= 0:
        raise ValueError("--due-min-days-after-offer must be > 0")
    if config.due_max_days_after_offer < config.due_min_days_after_offer:
        raise ValueError("--due-max-days-after-offer must be >= --due-min-days-after-offer")
    if (
        config.facility_limit_fraction_of_total is not None
        and config.facility_limit_fraction_of_total <= 0
    ):
        raise ValueError("--facility-limit-fraction-of-total must be > 0")

    scenario_dir = Path(config.output_root) / _scenario_name(config)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    data_df = _build_rows(config)
    _, _, _, customer_to_group = _build_entities(config)
    limits_payload = _build_limits_payload(data_df, config, customer_to_group)
    manifest = _build_manifest(data_df, config, limits_payload, scenario_dir)

    extraction_path = scenario_dir / "synthetic_extraction.xlsx"
    limits_path = scenario_dir / "limits.yaml"
    manifest_path = scenario_dir / "scenario_manifest.json"

    with pd.ExcelWriter(extraction_path, engine="openpyxl") as writer:
        data_df.to_excel(writer, sheet_name=SHEET_NAME, index=False)

    limits_path.write_text(
        json.dumps(limits_payload, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(scenario_dir, config, manifest)

    return {
        "scenario_dir": str(scenario_dir),
        "synthetic_extraction": str(extraction_path),
        "limits_yaml": str(limits_path),
        "manifest_json": str(manifest_path),
        "readme": str(scenario_dir / "README.md"),
        "rows_total": int(len(data_df)),
        "rows_candidate": int((data_df["Synthetic Row Type"] == "candidate").sum()),
        "rows_history": int((data_df["Synthetic Row Type"] == "history").sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic extraction data for optimizer testing.")
    parser.add_argument("--start-date", required=True, help="Simulation start date (YYYY-MM-DD).")
    parser.add_argument("--weeks", type=int, default=8, help="Number of weekly cohorts.")
    parser.add_argument("--invoices-per-week", type=int, default=40, help="Incoming invoices per week.")
    parser.add_argument("--history-weeks", type=int, default=4, help="Pre-start historical weeks.")
    parser.add_argument(
        "--history-invoices-per-week",
        type=int,
        default=None,
        help="Historical invoices per week (default: invoices_per_week // 3).",
    )
    parser.add_argument("--companies", type=int, default=2, help="Number of synthetic companies.")
    parser.add_argument("--customers-per-company", type=int, default=10, help="Customers per company.")
    parser.add_argument("--groups-per-company", type=int, default=2, help="Groups per company.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic output.")
    parser.add_argument("--scenario-name", default=None, help="Scenario folder name.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root output directory.")
    parser.add_argument("--base-exposure-ratio", type=float, default=0.08, help="Base exposure / limit ratio.")
    parser.add_argument(
        "--facility-utilization-target",
        type=float,
        default=0.75,
        help="Target candidate utilization to derive facility limits.",
    )
    parser.add_argument(
        "--facility-limit-fraction-of-total",
        type=float,
        default=None,
        help="If provided, facility limit = this fraction * total candidate amount per company.",
    )
    parser.add_argument(
        "--customer-limit-fraction-of-facility",
        type=float,
        default=0.20,
        help="Customer limit fraction of facility.",
    )
    parser.add_argument(
        "--group-limit-fraction-of-facility",
        type=float,
        default=0.45,
        help="Group limit fraction of facility.",
    )
    parser.add_argument(
        "--lifetime-min-days",
        type=int,
        default=14,
        help="Minimum synthetic lifetime in days (summary->reconciliation).",
    )
    parser.add_argument(
        "--lifetime-max-days",
        type=int,
        default=112,
        help="Maximum synthetic lifetime in days (summary->reconciliation).",
    )
    parser.add_argument(
        "--due-min-days-after-offer",
        type=int,
        default=7,
        help="Minimum due-date offset from Offer File Date (days).",
    )
    parser.add_argument(
        "--due-max-days-after-offer",
        type=int,
        default=140,
        help="Maximum due-date offset from Offer File Date (days).",
    )
    parser.add_argument(
        "--enable-reconciliation-date",
        action="store_true",
        help="Generate candidate Reconciliation File Date (UTC) values.",
    )
    parser.add_argument(
        "--enable-reconciliation-dates",
        action="store_true",
        help="Alias of --enable-reconciliation-date.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        start_date=str(args.start_date),
        weeks=int(args.weeks),
        invoices_per_week=int(args.invoices_per_week),
        history_weeks=int(args.history_weeks),
        history_invoices_per_week=(
            None if args.history_invoices_per_week is None else int(args.history_invoices_per_week)
        ),
        companies=int(args.companies),
        customers_per_company=int(args.customers_per_company),
        groups_per_company=int(args.groups_per_company),
        seed=int(args.seed),
        base_exposure_ratio=float(args.base_exposure_ratio),
        facility_utilization_target=float(args.facility_utilization_target),
        facility_limit_fraction_of_total=(
            None
            if args.facility_limit_fraction_of_total is None
            else float(args.facility_limit_fraction_of_total)
        ),
        customer_limit_fraction_of_facility=float(args.customer_limit_fraction_of_facility),
        group_limit_fraction_of_facility=float(args.group_limit_fraction_of_facility),
        lifetime_min_days=int(args.lifetime_min_days),
        lifetime_max_days=int(args.lifetime_max_days),
        due_min_days_after_offer=int(args.due_min_days_after_offer),
        due_max_days_after_offer=int(args.due_max_days_after_offer),
        enable_reconciliation_date=bool(
            getattr(args, "enable_reconciliation_date", False)
            or getattr(args, "enable_reconciliation_dates", False)
        ),
        output_root=Path(args.output_root),
        scenario_name=args.scenario_name,
    )
    summary = generate_synthetic_package(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
