"""
Limits loading, validation, and resolution for optimizer constraints.

This module manages three tiers of credit limits that constrain the optimizer:

  1. Facility limits per Company Code -- the aggregate cap for a legal entity.
  2. Customer limits per Customer ID -- concentration cap per individual buyer.
  3. Group limits per customer group -- aggregate cap across related buyers.

Limits can be provided explicitly in the YAML config, or generated synthetically
from the candidate data when ``synthetic_generation.enabled = true``.

Precedence rules:
  - Explicit limits in the config always take priority over synthetic ones.
  - When no explicit facility limits are provided and synthetic generation is
    enabled, facility limits are computed as: alpha * sum(Purchase Price) per
    company code.
  - Customer and group limits for entities not listed explicitly are filled
    from effective default fractions:
      - synthetic mode: beta/gamma from synthetic_generation
      - manual mode: defaults.customer_limit_fraction_of_facility and
        defaults.group_limit_fraction_of_facility

All resolved limits are stored as integer cents to match the solver's internal
representation and avoid floating-point comparison issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yaml


@dataclass(frozen=True)
class ResolvedLimits:
    """Final resolved limits in integer cents, ready for the optimizer.

    All values are in cents (multiply by 100 from the original money amount).
    Use ``limits_to_money_dict()`` to convert back to human-readable floats.
    """
    facility_limits_by_company_code: Dict[str, int]  # company_code -> limit in cents
    customer_limits: Dict[str, int]                   # customer_id -> limit in cents
    group_limits: Dict[str, int]                      # group_id -> limit in cents
    customer_to_group: Dict[str, str]                 # customer_id -> group_id mapping
    base_exposure_facility: Dict[str, int] = field(default_factory=dict)  # company_code -> exposure in cents
    base_exposure_customer: Dict[str, int] = field(default_factory=dict)   # customer_id -> exposure in cents
    base_exposure_group: Dict[str, int] = field(default_factory=dict)      # group_id -> exposure in cents

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "facility_limits_by_company_code": self.facility_limits_by_company_code,
            "customer_limits": self.customer_limits,
            "group_limits": self.group_limits,
            "customer_to_group": self.customer_to_group,
            "base_exposure_facility": self.base_exposure_facility,
            "base_exposure_customer": self.base_exposure_customer,
            "base_exposure_group": self.base_exposure_group,
        }


def _to_cents(amount: float) -> int:
    """Convert a monetary float to integer cents for the solver."""
    return int(round(float(amount) * 100))


def _from_cents(amount: int) -> float:
    """Convert integer cents back to a monetary float for reporting."""
    return amount / 100.0


def load_limits_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate the limits YAML config, applying defaults for missing keys.

    The config file supports explicit limits (for production use) and synthetic
    generation parameters (for prototyping when real limits are not yet available).

    See ``config/limits_synthetic.yaml`` for the full schema with comments.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Limits config not found: {source}")

    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    # Apply defaults so downstream code can assume all keys exist.
    payload.setdefault("facility_limits_by_company_code", {})
    payload.setdefault("customer_limits", {})
    payload.setdefault("group_limits", {})
    payload.setdefault("customer_to_group", {})
    payload.setdefault("base_exposure", {})
    payload.setdefault("defaults", {})

    base_exposure = payload["base_exposure"]
    base_exposure.setdefault("facility", {})
    base_exposure.setdefault("customer", {})
    base_exposure.setdefault("group", {})

    defaults = payload["defaults"]
    defaults.setdefault("customer_limit_fraction_of_facility", 0.15)
    defaults.setdefault("group_limit_fraction_of_facility", 0.30)

    # Synthetic generation defaults: used when explicit limits are empty.
    payload.setdefault("synthetic_generation", {})
    syn = payload["synthetic_generation"]
    syn.setdefault("enabled", True)
    syn.setdefault("alpha", 0.85)   # facility = alpha * total_amount_per_company
    syn.setdefault("beta", defaults["customer_limit_fraction_of_facility"])  # customer = beta * facility
    syn.setdefault("gamma", defaults["group_limit_fraction_of_facility"])    # group = gamma * facility

    _validate_limits_payload(payload)
    return payload


def _validate_fraction(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not (0 < float(value) <= 1.5):
        raise ValueError(f"{name} must be in (0, 1.5]")


def _validate_limits_payload(payload: Dict[str, Any]) -> None:
    for key in (
        "facility_limits_by_company_code",
        "customer_limits",
        "group_limits",
        "customer_to_group",
    ):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"{key} must be a mapping")

    base_exposure = payload.get("base_exposure", {})
    if not isinstance(base_exposure, dict):
        raise ValueError("base_exposure must be a mapping")
    for key in ("facility", "customer", "group"):
        value = base_exposure.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"base_exposure.{key} must be a mapping")

    defaults = payload["defaults"]
    _validate_fraction(
        "defaults.customer_limit_fraction_of_facility",
        defaults["customer_limit_fraction_of_facility"],
    )
    _validate_fraction(
        "defaults.group_limit_fraction_of_facility",
        defaults["group_limit_fraction_of_facility"],
    )

    syn = payload["synthetic_generation"]
    if not isinstance(syn.get("enabled"), bool):
        raise ValueError("synthetic_generation.enabled must be a boolean")
    _validate_fraction("synthetic_generation.alpha", syn["alpha"])
    _validate_fraction("synthetic_generation.beta", syn["beta"])
    _validate_fraction("synthetic_generation.gamma", syn["gamma"])


def propose_synthetic_limits(
    candidates_df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, Dict[str, float]]:
    """Propose synthetic limits (in money units) from the candidate cohort profile.

    This is used when real limits from the lender are not yet available. The
    synthetic limits create a plausible constraint environment for testing:

      facility(company)  = alpha * sum(Purchase Price for that company)
      customer(customer)  = beta  * facility(customer's company)
      group(group)        = gamma * facility(group's company)   [currently empty]

    NOTE: For customers appearing under multiple Company Codes, the facility
    used is from the first company seen in the groupby iteration. This is a
    known simplification -- see resolve_limits() for the same pattern.

    Args:
        candidates_df: Eligible invoices for the cohort.
        alpha: Fraction of total company amount to use as facility limit (e.g. 0.85).
        beta: Fraction of facility limit to use as customer limit (e.g. 0.15).
        gamma: Fraction of facility limit to use as group limit (e.g. 0.30).

    Returns:
        Dict with 'facility_limits_by_company_code', 'customer_limits', 'group_limits'
        (all in money units, not cents).
    """
    if candidates_df.empty:
        return {
            "facility_limits_by_company_code": {},
            "customer_limits": {},
            "group_limits": {},
        }

    # Compute facility limits: total Purchase Price per Company Code * alpha.
    grouped = (
        candidates_df.groupby("Company Code", dropna=False)["Purchase Price"]
        .sum()
        .sort_index()
    )
    facility = {
        str(cc): float(amount) * float(alpha)
        for cc, amount in grouped.items()
        if pd.notna(cc)
    }

    # Compute customer limits: beta * facility limit of the customer's company.
    customer_limits: Dict[str, float] = {}
    for customer, grp in candidates_df.groupby("Customer", dropna=False):
        if pd.isna(customer):
            continue
        company = grp["Company Code"].iloc[0]
        customer_limits[str(customer)] = facility.get(str(company), 0.0) * float(beta)

    # Group limits are left empty in synthetic mode (no customer-to-group mapping available).
    group_limits: Dict[str, float] = {}
    return {
        "facility_limits_by_company_code": facility,
        "customer_limits": customer_limits,
        "group_limits": group_limits,
    }


def _resolve_facility_limits(
    candidates_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, int]:
    explicit = {
        str(k): _to_cents(v)
        for k, v in config["facility_limits_by_company_code"].items()
    }

    if explicit:
        return explicit

    syn = config["synthetic_generation"]
    if not syn["enabled"]:
        raise ValueError(
            "No facility limits provided and synthetic_generation.enabled=false"
        )

    proposed = propose_synthetic_limits(
        candidates_df,
        alpha=float(syn["alpha"]),
        beta=float(syn["beta"]),
        gamma=float(syn["gamma"]),
    )
    return {k: _to_cents(v) for k, v in proposed["facility_limits_by_company_code"].items()}


def resolve_limits(
    candidates_df: pd.DataFrame,
    limits_config: Dict[str, Any],
) -> ResolvedLimits:
    """Resolve all limit tiers into integer cents, merging explicit and default values.

    Resolution order:
      1. Facility limits: use explicit config values if provided, otherwise generate
         synthetic limits from the candidate data.
      2. Customer limits: use explicit config values for listed customers; for all
         other customers observed in the data, compute a default as a fraction of
         their company's facility limit (synthetic mode uses beta, manual mode
         uses defaults.customer_limit_fraction_of_facility).
      3. Group limits: use explicit config values for listed groups; for groups
         observed via customer_to_group mapping, compute a default as a fraction
         of the corresponding facility limit (synthetic mode uses gamma, manual
         mode uses defaults.group_limit_fraction_of_facility).

    NOTE: When computing default customer limits, the company code is taken from
    the first row of the customer's invoice group. If a customer has invoices
    under multiple company codes, only the first is used.

    Args:
        candidates_df: Eligible invoices for the cohort (used for synthetic generation
            and for discovering which customers/groups need limits).
        limits_config: Parsed YAML config dict (from load_limits_config).

    Returns:
        ResolvedLimits with all values in integer cents.
    """
    if candidates_df.empty:
        return ResolvedLimits({}, {}, {}, {})

    facility_limits = _resolve_facility_limits(candidates_df, limits_config)
    defaults = limits_config["defaults"]
    syn = limits_config.get("synthetic_generation", {})
    synthetic_enabled = bool(syn.get("enabled", False))
    customer_default_fraction = float(
        syn["beta"] if synthetic_enabled else defaults["customer_limit_fraction_of_facility"]
    )
    group_default_fraction = float(
        syn["gamma"] if synthetic_enabled else defaults["group_limit_fraction_of_facility"]
    )

    # Start with explicitly configured customer and group limits (in cents).
    explicit_customer = {
        str(k): _to_cents(v) for k, v in limits_config["customer_limits"].items()
    }
    explicit_group = {
        str(k): _to_cents(v) for k, v in limits_config["group_limits"].items()
    }
    customer_to_group = {
        str(k): str(v) for k, v in limits_config["customer_to_group"].items()
    }

    base_exposure_cfg = limits_config.get("base_exposure", {})
    base_exposure_facility = {
        str(k): _to_cents(v) for k, v in base_exposure_cfg.get("facility", {}).items()
    }
    base_exposure_customer = {
        str(k): _to_cents(v) for k, v in base_exposure_cfg.get("customer", {}).items()
    }
    base_exposure_group = {
        str(k): _to_cents(v) for k, v in base_exposure_cfg.get("group", {}).items()
    }

    # Fill in default customer limits for customers not explicitly listed.
    customer_limits = dict(explicit_customer)

    for customer, grp in candidates_df.groupby("Customer", dropna=False):
        if pd.isna(customer):
            continue
        customer_key = str(customer)
        if customer_key in customer_limits:
            continue  # Explicit limit takes precedence.

        # Default: fraction of the customer's company facility limit.
        company = str(grp["Company Code"].iloc[0])
        facility = facility_limits.get(company, 0)
        customer_limits[customer_key] = int(round(facility * customer_default_fraction))

    # Fill in default group limits for groups discovered via customer_to_group mapping.
    group_limits = dict(explicit_group)
    groups_seen = {customer_to_group.get(str(c)) for c in customer_limits.keys()}
    groups_seen = {g for g in groups_seen if g}

    for group_id in groups_seen:
        if group_id in group_limits:
            continue  # Explicit limit takes precedence.

        # Find all customers in this group and derive the company facility limit.
        mapped_customers = [
            cust for cust, grp_id in customer_to_group.items() if grp_id == group_id
        ]
        if not mapped_customers:
            continue
        company_candidates = candidates_df[candidates_df["Customer"].astype(str).isin(mapped_customers)]
        if company_candidates.empty:
            continue
        company = str(company_candidates["Company Code"].iloc[0])
        facility = facility_limits.get(company, 0)
        group_limits[group_id] = int(round(facility * group_default_fraction))

    return ResolvedLimits(
        facility_limits_by_company_code=facility_limits,
        customer_limits=customer_limits,
        group_limits=group_limits,
        customer_to_group=customer_to_group,
        base_exposure_facility=base_exposure_facility,
        base_exposure_customer=base_exposure_customer,
        base_exposure_group=base_exposure_group,
    )


def save_limits_config(path: str | Path, payload: Dict[str, Any]) -> None:
    """Persist a limits config dict back to YAML (for reproducibility of synthetic limits)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def limits_to_money_dict(limits: ResolvedLimits) -> Dict[str, Dict[str, float]]:
    """Convert resolved limits from cents back to money units for human-readable output."""
    return {
        "facility_limits_by_company_code": {
            k: _from_cents(v) for k, v in limits.facility_limits_by_company_code.items()
        },
        "customer_limits": {k: _from_cents(v) for k, v in limits.customer_limits.items()},
        "group_limits": {k: _from_cents(v) for k, v in limits.group_limits.items()},
        "customer_to_group": dict(limits.customer_to_group),
        "base_exposure": {
            "facility": {k: _from_cents(v) for k, v in limits.base_exposure_facility.items()},
            "customer": {k: _from_cents(v) for k, v in limits.base_exposure_customer.items()},
            "group": {k: _from_cents(v) for k, v in limits.base_exposure_group.items()},
        },
    }
