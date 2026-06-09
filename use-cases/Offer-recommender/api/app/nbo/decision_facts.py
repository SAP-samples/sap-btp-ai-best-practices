"""Typed fact computation for the executable decision engine."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from app.nbo.catalog import load_commercial_taxonomy, load_tariff_catalog
from app.nbo.config import (
    ANNUAL_USAGE_SMALL_BUSINESS_MAX,
    COL_BILL_PAID,
    COL_BILL_TOTAL,
    COL_DER_BATTERY_OWNERSHIP,
    COL_DER_BATTERY_PARTNER_BRAND,
    COL_DER_COMPATIBLE_BATTERY,
    COL_DER_DEMAND_MGMT_INCLUDED,
    COL_DER_ELIGIBLE_CONNECTED_DEVICE,
    COL_DER_ELIGIBLE_HOME_EV_CHARGER,
    COL_DER_NEW_ROOFTOP_SOLAR,
    COL_DER_PREFERRED_SOLAR_INSTALLER,
    COL_DER_PRIOR_REC_ASSIGNMENT,
    COL_DER_QUALIFYING_FACILITY_KW_AC,
    COL_DER_REC_RIGHTS_OWNED,
    COL_DER_SMART_THERMOSTAT_PURCHASE,
    COL_DER_SOLAR_OWNERSHIP,
    COL_DER_STORAGE_ONLY,
    COL_DER_THERMOSTAT_BRAND,
    COL_DER_THERMOSTAT_PROVIDER_ACCOUNT,
    COL_DER_THERMOSTAT_WIFI,
    COL_EVENT_DATE,
    COL_EVENT_PROGRAM_ID,
    COL_EVENT_TYPE,
    COL_INDUSTRY,
    COL_METER_USAGE,
    COL_NAIC_CODE,
    COL_OFF_PEAK,
    COL_ON_PEAK,
    COL_PROFILE_ACCOUNT_GOOD_STANDING,
    COL_PROFILE_ACCOUNT_NAME_TYPE,
    COL_PROFILE_CENTRAL_AC,
    COL_PROFILE_COOLING_SYSTEM_TYPE,
    COL_PROFILE_CONDITIONED_SQFT,
    COL_PROFILE_CONNECTED_UNIT_COUNT,
    COL_PROFILE_CURTAILMENT_CAPABILITY,
    COL_PROFILE_DWELLING_TYPE,
    COL_PROFILE_ELIGIBLE_CONTRACTOR,
    COL_PROFILE_ENERGY_STAR_MFNC,
    COL_PROFILE_LARGE_ENERGY_CONSUMER,
    COL_PROFILE_NEW_CONSTRUCTION,
    COL_PROFILE_OCCUPANCY_STATUS,
    COL_PROFILE_OWNERSHIP_STATUS,
    COL_PROFILE_PROJECT_EXCEEDS_BASELINE_10,
    COL_PROFILE_PROJECT_STAGE,
    COL_PROFILE_QUALIFIED_PRICE_PLAN,
    COL_PROFILE_SAME_ACCOUNT_HOLDER_12M,
    COL_PROFILE_SERVICE_ENTRANCE_AMPS,
    COL_PROFILE_SERVICE_START_DATE,
    COL_RATE_PLAN,
    COL_READ_DATE,
    COL_STATUS,
    DECLINE_SUPPRESSION_DAYS,
    DEMAND_RESPONSE_USAGE_THRESHOLD,
    SHORTFALL_HISTORY_THRESHOLD,
    TREND_RISE_FACTOR,
)
from app.nbo.data_loader import DataStore
from app.nbo.fact_registry import FACT_REGISTRY
from app.nbo.models import Confidence, FactSource, FactValue
from app.nbo.persona import get_persona_hints


SMALL_BUSINESS_RATE_PLANS = {"E32", "E34", "E36", "E47", "E48"}
SOLAR_RATE_PLANS = {"E13", "E14", "E15", "E27"}
EV_RATE_PLANS = {"E14", "E29"}
SUPPORTED_BATTERY_PARTNER_BRANDS = {"FRANKLINWH", "TESLA"}
SUPER_OFF_PEAK_SHARE_DEFAULT = 0.35
AVERAGE_WEEKDAYS_PER_MONTH = 21.7
E16_ON_PEAK_HOURS_PER_WEEKDAY = 5.0
SERVICE_CHARGE_TIER_UNAVAILABLE_REASON = (
    "Dwelling type and service entrance amps are both required to derive the monthly service charge tier."
)


def _fact(
    fact_id: str,
    value=None,
    source: FactSource = FactSource.DERIVED,
    confidence: Confidence = Confidence.MEDIUM,
    evidence: Iterable[str] | None = None,
    missing_reason: str | None = None,
) -> FactValue:
    definition = FACT_REGISTRY[fact_id]
    return FactValue(
        fact_id=fact_id,
        value=value,
        value_type=definition.value_type,
        source=source,
        confidence=confidence,
        evidence=list(evidence or []),
        missing_reason=missing_reason,
    )


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _safe_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _history_span_days(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    min_date = history[COL_READ_DATE].min()
    max_date = history[COL_READ_DATE].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return 0
    return int((max_date - min_date).days)


def _compute_bill_increase_yoy(history: pd.DataFrame) -> float | None:
    dated = history.dropna(subset=[COL_READ_DATE, COL_BILL_TOTAL]).copy()
    if len(dated) < 6:
        return None
    dated = dated.sort_values(COL_READ_DATE)
    if _history_span_days(dated) < 330:
        return None

    recent = dated.tail(3)
    recent_avg = recent[COL_BILL_TOTAL].mean()
    target_start = recent[COL_READ_DATE].min() - pd.DateOffset(months=12)
    target_end = recent[COL_READ_DATE].max() - pd.DateOffset(months=12)
    year_ago = dated[
        (dated[COL_READ_DATE] >= target_start - pd.DateOffset(months=1))
        & (dated[COL_READ_DATE] <= target_end + pd.DateOffset(months=1))
    ]
    if year_ago.empty:
        return None

    year_ago_avg = year_ago[COL_BILL_TOTAL].mean()
    if pd.isna(year_ago_avg) or year_ago_avg <= 0:
        return None
    return float((recent_avg - year_ago_avg) / year_ago_avg)


def _compute_usage_yoy(history: pd.DataFrame) -> bool | None:
    dated = history.dropna(subset=[COL_READ_DATE, COL_METER_USAGE]).copy()
    if len(dated) < 6:
        return None
    dated = dated.sort_values(COL_READ_DATE)
    if _history_span_days(dated) < 330:
        return None

    recent = dated.tail(3)
    recent_avg = dated.tail(3)[COL_METER_USAGE].mean()
    target_start = recent[COL_READ_DATE].min() - pd.DateOffset(months=12)
    target_end = recent[COL_READ_DATE].max() - pd.DateOffset(months=12)
    year_ago = dated[
        (dated[COL_READ_DATE] >= target_start - pd.DateOffset(months=1))
        & (dated[COL_READ_DATE] <= target_end + pd.DateOffset(months=1))
    ]
    if year_ago.empty:
        return None

    year_ago_avg = year_ago[COL_METER_USAGE].mean()
    if pd.isna(year_ago_avg) or year_ago_avg <= 0:
        return None
    return bool(recent_avg > year_ago_avg * TREND_RISE_FACTOR)


def _compute_three_day_cost(history: pd.DataFrame) -> float | None:
    dated = history.dropna(subset=[COL_READ_DATE, COL_BILL_TOTAL]).sort_values(COL_READ_DATE)
    if len(dated) < 2:
        return None
    latest = dated.iloc[-1]
    prev = dated.iloc[-2]
    total = _safe_float(latest[COL_BILL_TOTAL])
    if total is None or total <= 0:
        return None
    days = int((latest[COL_READ_DATE] - prev[COL_READ_DATE]).days)
    if days <= 0:
        return None
    return round((total / days) * 3, 2)


def _compute_annual_usage_estimate(history: pd.DataFrame) -> float | None:
    dated = history.dropna(subset=[COL_READ_DATE, COL_METER_USAGE]).sort_values(COL_READ_DATE)
    if dated.empty:
        return None
    span_days = _history_span_days(dated)
    total_usage = float(dated[COL_METER_USAGE].sum())
    if span_days >= 330:
        return round(total_usage, 2)
    if span_days <= 0:
        if len(dated) >= 3:
            return round(float(dated.tail(3)[COL_METER_USAGE].mean()) * 12, 2)
        return None
    return round(total_usage * 365.0 / span_days, 2)


def _infer_business_taxonomy(naic_code, industry: str | None) -> str:
    if pd.isna(naic_code):
        naic_text = ""
    else:
        try:
            naic_num = float(naic_code)
            naic_text = str(int(naic_num)) if naic_num.is_integer() else str(naic_code).strip()
        except (TypeError, ValueError):
            naic_text = str(naic_code).strip()
    industry_text = (industry or "").strip().lower()

    for rule in load_commercial_taxonomy():
        if any(naic_text.startswith(prefix) for prefix in rule["naic_prefixes"]):
            return rule["taxonomy"]
        if any(keyword in industry_text for keyword in rule["industry_keywords"]):
            return rule["taxonomy"]
    return "GENERAL_BUSINESS"


def _supported_rate_plans() -> set[str]:
    return {
        entry["rate_plan"]
        for entry in load_tariff_catalog()
        if entry.get("simulation_supported")
    }


def _derive_service_charge_tier(dwelling_type: str | None, amps: int | None) -> str | None:
    if dwelling_type is None or amps is None:
        return None
    if amps >= 226:
        return "tier3"
    if dwelling_type in {"ATTACHED_HOME", "MULTIFAMILY_UNIT"}:
        return "tier1"
    if dwelling_type in {"SINGLE_FAMILY", "MOBILE_HOME"}:
        return "tier2"
    return None


def _estimate_super_off_peak_usage(non_on_peak_kwh: float | None) -> float | None:
    if non_on_peak_kwh is None:
        return None
    return round(non_on_peak_kwh * SUPER_OFF_PEAK_SHARE_DEFAULT, 2)


def _estimate_avg_on_peak_daily_kw(avg_on_peak_kwh: float | None) -> float | None:
    if avg_on_peak_kwh is None:
        return None
    hours = E16_ON_PEAK_HOURS_PER_WEEKDAY * AVERAGE_WEEKDAYS_PER_MONTH
    if hours <= 0:
        return None
    return round(avg_on_peak_kwh / hours, 3)


def _profile_value(row: dict | None, key: str):
    if row is None:
        return None
    return row.get(key)


def _derive_customer_of_record_on_site(occupancy_status: str | None) -> bool | None:
    if occupancy_status is None:
        return None
    normalized = occupancy_status.strip().casefold()
    if any(token in normalized for token in {"primary", "owner", "occupied", "resident"}):
        return True
    if any(token in normalized for token in {"vacant", "non-owner", "off-site"}):
        return False
    return None


def compute_account_facts(billing_account: str, ds: DataStore) -> dict[str, FactValue]:
    """Build the typed fact map for a billing account."""
    facts: dict[str, FactValue] = {}
    history = ds.l3_account_history(billing_account)
    snapshot = ds.l2_current_snapshot(billing_account)
    customer_type = ds.l1_account_type(billing_account)
    segment_name = ds.l4_segment(billing_account)
    current_program_codes = ds.l8_current_program_codes(billing_account)
    usage_avgs = ds.l9_usage_averages(billing_account)
    summer_usage_avgs = ds.l9_summer_usage_averages(billing_account)
    span_days = _history_span_days(history)
    supported_rates = _supported_rate_plans()
    profile = ds.account_profile_row(billing_account)
    der_profile = ds.der_profile_row(billing_account)
    events = ds.program_event_history_rows(billing_account)

    facts["customer_type"] = _fact(
        "customer_type",
        customer_type,
        source=FactSource.WORKBOOK,
        evidence=["L1 account sheet membership"],
    )
    facts["has_current_snapshot"] = _fact(
        "has_current_snapshot",
        snapshot is not None,
        source=FactSource.DERIVED,
        evidence=["L2 latest dated snapshot"],
        confidence=Confidence.HIGH,
    )
    facts["segment_name"] = _fact(
        "segment_name",
        segment_name,
        source=FactSource.WORKBOOK,
        evidence=["L4 segment lookup"],
        missing_reason="Segment missing or NOT FOUND" if segment_name is None else None,
    )
    facts["current_program_codes"] = _fact(
        "current_program_codes",
        sorted(current_program_codes),
        source=FactSource.WORKBOOK,
        evidence=["L8 OTHER PROGRAMS"],
        confidence=Confidence.HIGH,
    )

    profile_dwelling = _safe_text(_profile_value(profile, COL_PROFILE_DWELLING_TYPE))
    profile_service_amps = _safe_int(_profile_value(profile, COL_PROFILE_SERVICE_ENTRANCE_AMPS))
    service_start_date = _profile_value(profile, COL_PROFILE_SERVICE_START_DATE)
    if pd.isna(service_start_date):
        service_start_date = None
    ownership_status = _safe_text(_profile_value(profile, COL_PROFILE_OWNERSHIP_STATUS))
    occupancy_status = _safe_text(_profile_value(profile, COL_PROFILE_OCCUPANCY_STATUS))
    account_name_type = _safe_text(_profile_value(profile, COL_PROFILE_ACCOUNT_NAME_TYPE))
    cooling_system_type = _safe_text(_profile_value(profile, COL_PROFILE_COOLING_SYSTEM_TYPE))
    central_ac = _safe_bool(_profile_value(profile, COL_PROFILE_CENTRAL_AC))
    new_construction_flag = _safe_bool(_profile_value(profile, COL_PROFILE_NEW_CONSTRUCTION))
    connected_unit_count = _safe_int(_profile_value(profile, COL_PROFILE_CONNECTED_UNIT_COUNT))
    project_stage = _safe_text(_profile_value(profile, COL_PROFILE_PROJECT_STAGE))
    conditioned_sqft = _safe_float(_profile_value(profile, COL_PROFILE_CONDITIONED_SQFT))
    account_good_standing = _safe_bool(_profile_value(profile, COL_PROFILE_ACCOUNT_GOOD_STANDING))
    same_account_holder_12m = _safe_bool(_profile_value(profile, COL_PROFILE_SAME_ACCOUNT_HOLDER_12M))
    qualified_price_plan = _safe_bool(_profile_value(profile, COL_PROFILE_QUALIFIED_PRICE_PLAN))
    large_energy_consumer = _safe_bool(_profile_value(profile, COL_PROFILE_LARGE_ENERGY_CONSUMER))
    curtailment_capability = _safe_bool(_profile_value(profile, COL_PROFILE_CURTAILMENT_CAPABILITY))
    eligible_contractor = _safe_bool(_profile_value(profile, COL_PROFILE_ELIGIBLE_CONTRACTOR))
    project_exceeds_baseline_10 = _safe_bool(
        _profile_value(profile, COL_PROFILE_PROJECT_EXCEEDS_BASELINE_10)
    )
    energy_star_multifamily = _safe_bool(_profile_value(profile, COL_PROFILE_ENERGY_STAR_MFNC))

    der_solar_ownership = _safe_bool(_profile_value(der_profile, COL_DER_SOLAR_OWNERSHIP))
    der_new_rooftop_solar = _safe_bool(_profile_value(der_profile, COL_DER_NEW_ROOFTOP_SOLAR))
    der_qf_kw_ac = _safe_float(_profile_value(der_profile, COL_DER_QUALIFYING_FACILITY_KW_AC))
    der_storage_only = _safe_bool(_profile_value(der_profile, COL_DER_STORAGE_ONLY))
    der_rec_rights_owned = _safe_bool(_profile_value(der_profile, COL_DER_REC_RIGHTS_OWNED))
    der_prior_rec_assignment = _safe_bool(_profile_value(der_profile, COL_DER_PRIOR_REC_ASSIGNMENT))
    der_preferred_installer = _safe_bool(_profile_value(der_profile, COL_DER_PREFERRED_SOLAR_INSTALLER))
    thermostat_brand = _safe_text(_profile_value(der_profile, COL_DER_THERMOSTAT_BRAND))
    thermostat_provider_account = _safe_bool(
        _profile_value(der_profile, COL_DER_THERMOSTAT_PROVIDER_ACCOUNT)
    )
    thermostat_wifi = _safe_bool(_profile_value(der_profile, COL_DER_THERMOSTAT_WIFI))
    eligible_connected_device = _safe_bool(
        _profile_value(der_profile, COL_DER_ELIGIBLE_CONNECTED_DEVICE)
    )
    smart_thermostat_purchase = _safe_bool(
        _profile_value(der_profile, COL_DER_SMART_THERMOSTAT_PURCHASE)
    )
    eligible_home_ev_charger = _safe_bool(
        _profile_value(der_profile, COL_DER_ELIGIBLE_HOME_EV_CHARGER)
    )
    battery_ownership = _safe_bool(_profile_value(der_profile, COL_DER_BATTERY_OWNERSHIP))
    battery_partner_brand = _safe_text(_profile_value(der_profile, COL_DER_BATTERY_PARTNER_BRAND))
    compatible_battery = _safe_bool(_profile_value(der_profile, COL_DER_COMPATIBLE_BATTERY))
    demand_mgmt_included = _safe_bool(_profile_value(der_profile, COL_DER_DEMAND_MGMT_INCLUDED))

    facts["dwelling_type"] = _fact(
        "dwelling_type",
        profile_dwelling,
        source=FactSource.WORKBOOK if profile_dwelling is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile dwelling type"] if profile_dwelling is not None else [],
        missing_reason="Dwelling type is not available in the account profile" if profile_dwelling is None else None,
    )
    facts["service_entrance_amps"] = _fact(
        "service_entrance_amps",
        profile_service_amps,
        source=FactSource.WORKBOOK if profile_service_amps is not None else FactSource.SYSTEM,
        evidence=["Account profile service entrance amps"] if profile_service_amps is not None else [],
        missing_reason="Service entrance amps are not available in the account profile" if profile_service_amps is None else None,
        confidence=Confidence.HIGH if profile_service_amps is not None else Confidence.MANUAL_REVIEW,
    )
    facts["service_start_date"] = _fact(
        "service_start_date",
        service_start_date,
        source=FactSource.WORKBOOK if service_start_date is not None else FactSource.SYSTEM,
        evidence=["Account profile service start date"] if service_start_date is not None else [],
        missing_reason="Service start date is not available in the account profile" if service_start_date is None else None,
    )
    facts["ownership_status"] = _fact(
        "ownership_status",
        ownership_status,
        source=FactSource.WORKBOOK if ownership_status is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile ownership status"] if ownership_status is not None else [],
        missing_reason="Ownership status is not available in the account profile" if ownership_status is None else None,
    )
    facts["home_ownership_status"] = _fact(
        "home_ownership_status",
        ownership_status,
        source=FactSource.WORKBOOK if ownership_status is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile ownership status"] if ownership_status is not None else [],
        missing_reason="Homeownership status is not available in the account profile" if ownership_status is None else None,
    )
    facts["occupancy_status"] = _fact(
        "occupancy_status",
        occupancy_status,
        source=FactSource.WORKBOOK if occupancy_status is not None else FactSource.SYSTEM,
        evidence=["Account profile occupancy status"] if occupancy_status is not None else [],
        missing_reason="Occupancy status is not available in the account profile" if occupancy_status is None else None,
    )
    facts["customer_of_record_on_site"] = _fact(
        "customer_of_record_on_site",
        _derive_customer_of_record_on_site(occupancy_status),
        source=FactSource.DERIVED if occupancy_status is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from occupancy status"] if occupancy_status is not None else [],
        missing_reason="Occupancy details are unavailable for this account" if occupancy_status is None else None,
    )
    facts["account_name_type"] = _fact(
        "account_name_type",
        account_name_type,
        source=FactSource.WORKBOOK if account_name_type is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile account-name type"] if account_name_type is not None else [],
        missing_reason="Account-name type is not available in the account profile" if account_name_type is None else None,
    )
    facts["is_company_name_account"] = _fact(
        "is_company_name_account",
        account_name_type == "COMPANY" if account_name_type is not None else None,
        source=FactSource.DERIVED if account_name_type is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from account-name type"] if account_name_type is not None else [],
        missing_reason="Account-name type is unavailable" if account_name_type is None else None,
    )
    facts["cooling_system_type"] = _fact(
        "cooling_system_type",
        cooling_system_type,
        source=FactSource.WORKBOOK if cooling_system_type is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile cooling system type"] if cooling_system_type is not None else [],
        missing_reason="Cooling system type is not available in the account profile" if cooling_system_type is None else None,
    )
    facts["central_air_conditioning"] = _fact(
        "central_air_conditioning",
        central_ac,
        source=FactSource.WORKBOOK if central_ac is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile central AC flag"] if central_ac is not None else [],
        missing_reason="Central-AC status is not available in the account profile" if central_ac is None else None,
    )
    facts["thermostat_controls_central_ac"] = _fact(
        "thermostat_controls_central_ac",
        True if central_ac is True and eligible_connected_device is True else None,
        source=FactSource.DERIVED if central_ac is not None and eligible_connected_device is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Connected thermostat with central AC"] if central_ac is True and eligible_connected_device is True else [],
        missing_reason="Thermostat control details are unavailable" if not (central_ac is True and eligible_connected_device is True) else None,
    )
    facts["thermostat_provider_account"] = _fact(
        "thermostat_provider_account",
        thermostat_provider_account,
        source=FactSource.WORKBOOK if thermostat_provider_account is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile thermostat provider-account flag"] if thermostat_provider_account is not None else [],
        missing_reason="Thermostat provider-account status is unavailable" if thermostat_provider_account is None else None,
    )
    facts["thermostat_wifi_connected"] = _fact(
        "thermostat_wifi_connected",
        thermostat_wifi,
        source=FactSource.WORKBOOK if thermostat_wifi is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile thermostat Wi-Fi flag"] if thermostat_wifi is not None else [],
        missing_reason="Thermostat Wi-Fi status is unavailable" if thermostat_wifi is None else None,
    )
    facts["thermostat_brand"] = _fact(
        "thermostat_brand",
        thermostat_brand,
        source=FactSource.WORKBOOK if thermostat_brand is not None else FactSource.SYSTEM,
        evidence=["DER profile thermostat brand"] if thermostat_brand is not None else [],
        missing_reason="Thermostat brand is not available in the DER profile" if thermostat_brand is None else None,
    )
    facts["smart_thermostat_purchase_eligible"] = _fact(
        "smart_thermostat_purchase_eligible",
        smart_thermostat_purchase,
        source=FactSource.WORKBOOK if smart_thermostat_purchase is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile thermostat-purchase eligibility"] if smart_thermostat_purchase is not None else [],
        missing_reason="Smart thermostat purchase eligibility is unavailable" if smart_thermostat_purchase is None else None,
    )
    facts["new_construction_flag"] = _fact(
        "new_construction_flag",
        new_construction_flag,
        source=FactSource.WORKBOOK if new_construction_flag is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile new-construction flag"] if new_construction_flag is not None else [],
        missing_reason="New-construction status is unavailable" if new_construction_flag is None else None,
    )
    facts["eligible_contractor"] = _fact(
        "eligible_contractor",
        eligible_contractor,
        source=FactSource.WORKBOOK if eligible_contractor is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Account profile eligible-contractor flag"] if eligible_contractor is not None else [],
        missing_reason="Eligible-contractor status is unavailable" if eligible_contractor is None else None,
    )
    facts["air_conditioning_measure_qualified"] = _fact(
        "air_conditioning_measure_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Air-conditioning equipment specifics are not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["insulation_measure_qualified"] = _fact(
        "insulation_measure_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Insulation measure specifics are not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["cool_roof_measure_qualified"] = _fact(
        "cool_roof_measure_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Cool-roof measure specifics are not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["hpwh_measure_qualified"] = _fact(
        "hpwh_measure_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Heat-pump water-heater equipment specifics are not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["window_measure_qualified"] = _fact(
        "window_measure_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Window product specifics are not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )

    if snapshot is None:
        missing_reason = "No dated billing snapshot is available for this account"
        for fact_id in (
            "current_status",
            "current_rate_plan",
            "avg_on_peak_kwh_3m",
            "avg_off_peak_kwh_3m",
            "avg_total_usage_3m",
            "latest_bill_shortfall",
        ):
            facts[fact_id] = _fact(fact_id, source=FactSource.WORKBOOK, missing_reason=missing_reason)
    else:
        current_status = (
            str(snapshot.get(COL_STATUS)).strip().upper()
            if pd.notna(snapshot.get(COL_STATUS))
            else None
        )
        current_rate_plan = (
            str(snapshot.get(COL_RATE_PLAN)).strip().upper()
            if pd.notna(snapshot.get(COL_RATE_PLAN))
            else None
        )
        total = _safe_float(snapshot.get(COL_BILL_TOTAL))
        paid = _safe_float(snapshot.get(COL_BILL_PAID))

        facts["current_status"] = _fact(
            "current_status",
            current_status,
            source=FactSource.WORKBOOK,
            evidence=["L2 STATUS"],
        )
        facts["snapshot_read_date"] = _fact(
            "snapshot_read_date",
            snapshot.get(COL_READ_DATE),
            source=FactSource.WORKBOOK,
            evidence=["L2 READ DATE"],
        )
        facts["current_rate_plan"] = _fact(
            "current_rate_plan",
            current_rate_plan,
            source=FactSource.WORKBOOK,
            evidence=["L2 RATE PLAN"],
        )

        avg_on_peak = _safe_float(usage_avgs["on_peak"])
        avg_off_peak = _safe_float(usage_avgs["off_peak"])
        avg_meter_usage = _safe_float(usage_avgs["meter_usage"])
        if avg_on_peak is not None and avg_off_peak is not None:
            avg_total = avg_on_peak + avg_off_peak
            non_on_peak = avg_off_peak
            evidence = ["L9 recent on-peak + off-peak average"]
        elif avg_meter_usage is not None:
            avg_total = avg_meter_usage
            non_on_peak = avg_meter_usage
            evidence = ["L9 recent meter usage average"]
        else:
            avg_total = None
            non_on_peak = None
            evidence = []

        facts["avg_on_peak_kwh_3m"] = _fact(
            "avg_on_peak_kwh_3m",
            avg_on_peak,
            source=FactSource.DERIVED,
            evidence=["L9 recent on-peak average"],
        )
        facts["avg_off_peak_kwh_3m"] = _fact(
            "avg_off_peak_kwh_3m",
            avg_off_peak,
            source=FactSource.DERIVED,
            evidence=["L9 recent off-peak average"],
        )
        facts["avg_super_off_peak_kwh_3m"] = _fact(
            "avg_super_off_peak_kwh_3m",
            _estimate_super_off_peak_usage(non_on_peak),
            source=FactSource.DERIVED,
            evidence=["Estimated from non-on-peak usage using open-plan time windows"] if non_on_peak is not None else [],
            missing_reason="Non-on-peak usage is unavailable for super-off-peak estimation" if non_on_peak is None else None,
        )
        facts["avg_total_usage_3m"] = _fact(
            "avg_total_usage_3m",
            avg_total,
            source=FactSource.DERIVED,
            evidence=evidence,
        )

        avg_on_peak_summer = _safe_float(summer_usage_avgs["on_peak"])
        avg_off_peak_summer = _safe_float(summer_usage_avgs["off_peak"])
        avg_meter_usage_summer = _safe_float(summer_usage_avgs["meter_usage"])
        if avg_on_peak_summer is not None and avg_off_peak_summer is not None:
            avg_total_summer = avg_on_peak_summer + avg_off_peak_summer
            summer_non_on_peak = avg_off_peak_summer
            summer_evidence = ["L9 summer on-peak + off-peak average (May-Oct)"]
        elif avg_meter_usage_summer is not None:
            avg_total_summer = avg_meter_usage_summer
            summer_non_on_peak = avg_meter_usage_summer
            summer_evidence = ["L9 summer meter usage average (May-Oct)"]
        else:
            avg_total_summer = None
            summer_non_on_peak = None
            summer_evidence = []

        facts["avg_on_peak_summer"] = _fact(
            "avg_on_peak_summer",
            avg_on_peak_summer,
            source=FactSource.DERIVED,
            evidence=["L9 summer on-peak average (May-Oct)"],
            missing_reason="No summer usage data available" if avg_on_peak_summer is None else None,
        )
        facts["avg_off_peak_summer"] = _fact(
            "avg_off_peak_summer",
            avg_off_peak_summer,
            source=FactSource.DERIVED,
            evidence=["L9 summer off-peak average (May-Oct)"],
            missing_reason="No summer usage data available" if avg_off_peak_summer is None else None,
        )
        facts["avg_super_off_peak_summer"] = _fact(
            "avg_super_off_peak_summer",
            _estimate_super_off_peak_usage(summer_non_on_peak),
            source=FactSource.DERIVED,
            evidence=["Estimated summer super-off-peak usage from non-on-peak consumption"] if summer_non_on_peak is not None else [],
            missing_reason="No summer usage data available for super-off-peak estimation" if summer_non_on_peak is None else None,
        )
        facts["avg_total_usage_summer"] = _fact(
            "avg_total_usage_summer",
            avg_total_summer,
            source=FactSource.DERIVED,
            evidence=summer_evidence,
            missing_reason="No summer usage data available" if avg_total_summer is None else None,
        )
        facts["avg_on_peak_daily_kw_3m"] = _fact(
            "avg_on_peak_daily_kw_3m",
            _estimate_avg_on_peak_daily_kw(avg_on_peak),
            source=FactSource.DERIVED,
            evidence=["Estimated from average monthly on-peak usage and documented E16 on-peak hours"] if avg_on_peak is not None else [],
            missing_reason="On-peak usage is unavailable for demand estimation" if avg_on_peak is None else None,
        )
        facts["latest_bill_shortfall"] = _fact(
            "latest_bill_shortfall",
            max(0.0, (total or 0.0) - (paid or 0.0)),
            source=FactSource.DERIVED,
            evidence=["L2 BILL TOTAL - BILL PAID"],
        )
        facts["is_mpower_enrolled"] = _fact(
            "is_mpower_enrolled",
            current_rate_plan in ds.mpower_rate_plans,
            source=FactSource.DERIVED,
            evidence=["L5 residential Prepay rate plans"],
            confidence=Confidence.HIGH,
        )
        facts["residential_price_plan_customer"] = _fact(
            "residential_price_plan_customer",
            customer_type == "RESIDENTIAL" and current_rate_plan is not None,
            source=FactSource.DERIVED,
            evidence=["Residential account with current rate plan"],
            confidence=Confidence.HIGH,
        )
        facts["current_rate_supported_for_optimization"] = _fact(
            "current_rate_supported_for_optimization",
            current_rate_plan in supported_rates,
            source=FactSource.DERIVED,
            evidence=["Source-controlled tariff catalog"],
            confidence=Confidence.HIGH,
        )
        solar_ownership = der_solar_ownership if der_solar_ownership is not None else (
            True if current_rate_plan in SOLAR_RATE_PLANS else None
        )
        solar_evidence = []
        solar_missing = None
        if der_solar_ownership is not None:
            solar_evidence = ["DER profile solar-ownership flag"]
        elif current_rate_plan in SOLAR_RATE_PLANS:
            solar_evidence = ["Current solar/export price plan"]
        else:
            solar_missing = "Solar ownership is not directly stored in the current data"
        facts["solar_ownership"] = _fact(
            "solar_ownership",
            solar_ownership,
            source=FactSource.WORKBOOK if der_solar_ownership is not None else FactSource.DERIVED,
            evidence=solar_evidence,
            missing_reason=solar_missing,
        )
        facts["solar_export_rate_plan"] = _fact(
            "solar_export_rate_plan",
            current_rate_plan in SOLAR_RATE_PLANS,
            source=FactSource.DERIVED,
            evidence=["Current solar/export price plan"] if current_rate_plan in SOLAR_RATE_PLANS else ["Current non-export price plan"],
            confidence=Confidence.HIGH,
        )
        facts["ev_ownership"] = _fact(
            "ev_ownership",
            True if current_rate_plan in EV_RATE_PLANS else None,
            source=FactSource.DERIVED,
            evidence=["Current EV-oriented price plan"] if current_rate_plan in EV_RATE_PLANS else [],
            missing_reason="EV ownership is not directly stored in the current data" if current_rate_plan not in EV_RATE_PLANS else None,
        )

    derived_service_tier = _derive_service_charge_tier(profile_dwelling, profile_service_amps)
    facts["service_charge_tier"] = _fact(
        "service_charge_tier",
        derived_service_tier,
        source=FactSource.DERIVED if derived_service_tier is not None else FactSource.SYSTEM,
        evidence=["Derived from dwelling type and service entrance amps"] if derived_service_tier is not None else [],
        missing_reason=SERVICE_CHARGE_TIER_UNAVAILABLE_REASON if derived_service_tier is None else None,
        confidence=Confidence.HIGH if derived_service_tier is not None else Confidence.MANUAL_REVIEW,
    )
    snapshot_date = facts.get("snapshot_read_date").value if "snapshot_read_date" in facts else None
    service_days = None
    if service_start_date is not None and snapshot_date is not None and not pd.isna(snapshot_date):
        service_days = int((snapshot_date - service_start_date).days)
    facts["service_at_address_over_12_months"] = _fact(
        "service_at_address_over_12_months",
        service_days >= 365 if service_days is not None else None,
        source=FactSource.DERIVED if service_days is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from service start date and current snapshot date"] if service_days is not None else [],
        missing_reason="Service tenure at the address is unavailable" if service_days is None else None,
    )

    recent = history.tail(6)
    if recent.empty:
        repeated_shortfalls = 0
    else:
        totals = pd.to_numeric(recent[COL_BILL_TOTAL], errors="coerce").fillna(0)
        paids = pd.to_numeric(recent[COL_BILL_PAID], errors="coerce").fillna(0)
        repeated_shortfalls = int(((totals - paids).clip(lower=0) > 0).sum())

    facts["repeated_shortfall_count_6m"] = _fact(
        "repeated_shortfall_count_6m",
        repeated_shortfalls,
        source=FactSource.DERIVED,
        evidence=["Recent bill shortfall count"],
    )
    facts["payment_distress_disconnected"] = _fact(
        "payment_distress_disconnected",
        facts.get("current_status", _fact("current_status")).value == "DISCONNECTED",
        source=FactSource.DERIVED,
        evidence=["Current disconnected status"],
        confidence=Confidence.HIGH,
    )
    facts["payment_distress_current_shortfall"] = _fact(
        "payment_distress_current_shortfall",
        (facts.get("latest_bill_shortfall").value or 0) > 0 if "latest_bill_shortfall" in facts and facts["latest_bill_shortfall"].is_known else None,
        source=FactSource.DERIVED,
        evidence=["Latest bill shortfall"],
        missing_reason="Latest bill shortfall unavailable" if not facts.get("latest_bill_shortfall", _fact("latest_bill_shortfall")).is_known else None,
    )
    facts["payment_distress_repeat_shortfalls"] = _fact(
        "payment_distress_repeat_shortfalls",
        repeated_shortfalls >= SHORTFALL_HISTORY_THRESHOLD,
        source=FactSource.DERIVED,
        evidence=["Recent bill shortfall history"],
    )
    facts["has_payment_distress_signal"] = _fact(
        "has_payment_distress_signal",
        bool(
            facts.get("payment_distress_disconnected").value
            or facts.get("payment_distress_current_shortfall").value
            or facts.get("payment_distress_repeat_shortfalls").value
        ),
        source=FactSource.DERIVED,
        evidence=["Composite payment distress: disconnected, current shortfall, or repeat shortfalls"],
        confidence=Confidence.HIGH,
    )

    yoy_supported = span_days >= 330 and len(history.dropna(subset=[COL_READ_DATE])) >= 6
    facts["service_history_days"] = _fact(
        "service_history_days",
        span_days,
        source=FactSource.DERIVED,
        evidence=["Billing history date span"],
    )
    facts["yoy_history_supported"] = _fact(
        "yoy_history_supported",
        yoy_supported,
        source=FactSource.DERIVED,
        evidence=["Billing history span"],
        confidence=Confidence.HIGH,
    )
    facts["bill_increase_yoy"] = _fact(
        "bill_increase_yoy",
        _compute_bill_increase_yoy(history),
        source=FactSource.DERIVED,
        evidence=["YoY bill comparison"],
        missing_reason="At least 330 days of bill history are required for YoY bill increase" if not yoy_supported else None,
    )
    usage_yoy = _compute_usage_yoy(history)
    facts["high_usage_yoy"] = _fact(
        "high_usage_yoy",
        usage_yoy,
        source=FactSource.DERIVED,
        evidence=["YoY usage comparison"],
        missing_reason="At least 330 days of usage history are required for YoY usage comparison" if usage_yoy is None else None,
    )
    facts["high_usage_percentile"] = _fact(
        "high_usage_percentile",
        (facts["avg_total_usage_3m"].value or 0) > ds.usage_p75 if facts["avg_total_usage_3m"].is_known else None,
        source=FactSource.DERIVED,
        evidence=["Three-month usage average vs. dataset percentile"],
        missing_reason="Three-month usage average unavailable" if not facts["avg_total_usage_3m"].is_known else None,
    )
    high_usage_signal = None
    if facts["high_usage_percentile"].is_known or facts["high_usage_yoy"].is_known:
        high_usage_signal = bool(
            facts["high_usage_percentile"].value or facts["high_usage_yoy"].value
        )
    facts["has_high_usage_signal"] = _fact(
        "has_high_usage_signal",
        high_usage_signal,
        source=FactSource.DERIVED,
        evidence=["Composite high-usage rule"],
        missing_reason="Usage signals unavailable" if high_usage_signal is None else None,
    )
    rising_bill_or_high_usage = None
    if facts["bill_increase_yoy"].is_known or facts["has_high_usage_signal"].is_known:
        rising_bill_or_high_usage = bool(
            (facts["bill_increase_yoy"].value or 0) > 0
            or facts["has_high_usage_signal"].value
        )
    facts["bill_increase_or_high_usage"] = _fact(
        "bill_increase_or_high_usage",
        rising_bill_or_high_usage,
        source=FactSource.DERIVED,
        evidence=["Composite NBO routing trigger"],
        missing_reason="Bill increase or usage trigger unavailable" if rising_bill_or_high_usage is None else None,
    )
    facts["annual_usage_estimate_kwh"] = _fact(
        "annual_usage_estimate_kwh",
        _compute_annual_usage_estimate(history),
        source=FactSource.DERIVED,
        evidence=["Billing history usage annualization"],
        missing_reason="Usage history unavailable for annualization" if history.empty else None,
    )
    facts["three_day_usage_cost_estimate"] = _fact(
        "three_day_usage_cost_estimate",
        _compute_three_day_cost(history),
        source=FactSource.DERIVED,
        evidence=["Latest billing period daily cost estimate"],
        missing_reason="At least two dated bill rows are required to estimate three-day usage cost" if len(history.dropna(subset=[COL_READ_DATE])) < 2 else None,
    )

    commercial_naic = snapshot.get(COL_NAIC_CODE) if snapshot is not None else None
    commercial_industry = snapshot.get(COL_INDUSTRY) if snapshot is not None else None
    business_taxonomy = (
        _infer_business_taxonomy(
            commercial_naic,
            str(commercial_industry) if pd.notna(commercial_industry) else None,
        )
        if customer_type == "COMMERCIAL" and snapshot is not None
        else None
    )
    normalized_naic = None
    if commercial_naic not in ("", None) and not pd.isna(commercial_naic):
        try:
            normalized_naic = int(float(commercial_naic))
        except (TypeError, ValueError):
            normalized_naic = None

    facts["commercial_naic_code"] = _fact(
        "commercial_naic_code",
        normalized_naic,
        source=FactSource.WORKBOOK,
        evidence=["Commercial snapshot NAIC"],
        missing_reason="NAIC code unavailable" if customer_type == "COMMERCIAL" and normalized_naic is None else None,
    )
    facts["commercial_industry"] = _fact(
        "commercial_industry",
        str(commercial_industry).strip() if customer_type == "COMMERCIAL" and pd.notna(commercial_industry) else None,
        source=FactSource.WORKBOOK,
        evidence=["Commercial snapshot industry"],
        missing_reason="Industry unavailable" if customer_type == "COMMERCIAL" and pd.isna(commercial_industry) else None,
    )
    facts["business_taxonomy"] = _fact(
        "business_taxonomy",
        business_taxonomy,
        source=FactSource.DERIVED if business_taxonomy else FactSource.SYSTEM,
        evidence=["Commercial taxonomy catalog"] if business_taxonomy else [],
        missing_reason="Business taxonomy only applies to commercial accounts" if customer_type != "COMMERCIAL" else None,
        confidence=Confidence.HIGH if business_taxonomy else Confidence.MEDIUM,
    )
    current_rate_plan = facts.get("current_rate_plan").value if "current_rate_plan" in facts else None
    facts["small_business_rate_plan_eligible"] = _fact(
        "small_business_rate_plan_eligible",
        current_rate_plan in SMALL_BUSINESS_RATE_PLANS if customer_type == "COMMERCIAL" and current_rate_plan else None,
        source=FactSource.DERIVED,
        evidence=["Small Business Solutions eligible rate plans"] if customer_type == "COMMERCIAL" and current_rate_plan else [],
        missing_reason="Commercial current rate plan unavailable" if customer_type == "COMMERCIAL" and not current_rate_plan else None,
    )
    facts["same_account_holder_12_months"] = _fact(
        "same_account_holder_12_months",
        same_account_holder_12m if same_account_holder_12m is not None else (service_days >= 365 if service_days is not None else None),
        source=FactSource.WORKBOOK if same_account_holder_12m is not None else FactSource.DERIVED if service_days is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile same-account-holder flag"] if same_account_holder_12m is not None else ["Derived from service tenure"] if service_days is not None else [],
        missing_reason="Same-account-holder history is unavailable" if same_account_holder_12m is None and service_days is None else None,
    )
    facts["facility_age_over_4_years"] = _fact(
        "facility_age_over_4_years",
        service_days >= 1460 if service_days is not None else None,
        source=FactSource.DERIVED if service_days is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from service start date as a facility-age proxy"] if service_days is not None else [],
        missing_reason="Facility age is unavailable in current data" if service_days is None else None,
    )
    facts["account_good_standing"] = _fact(
        "account_good_standing",
        account_good_standing,
        source=FactSource.WORKBOOK if account_good_standing is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile account-good-standing flag"] if account_good_standing is not None else [],
        missing_reason="Account good-standing status is unavailable" if account_good_standing is None else None,
    )
    facts["connected_unit_count"] = _fact(
        "connected_unit_count",
        connected_unit_count,
        source=FactSource.WORKBOOK if connected_unit_count is not None else FactSource.SYSTEM,
        evidence=["Profile connected-unit count"] if connected_unit_count is not None else [],
        missing_reason="Connected-unit count is unavailable" if connected_unit_count is None else None,
    )
    facts["connected_units_ge_4"] = _fact(
        "connected_units_ge_4",
        connected_unit_count >= 4 if connected_unit_count is not None else (True if business_taxonomy == "MULTIFAMILY" else None),
        source=FactSource.DERIVED if connected_unit_count is not None or business_taxonomy == "MULTIFAMILY" else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile connected-unit count"] if connected_unit_count is not None else ["Commercial taxonomy infers multifamily property"] if business_taxonomy == "MULTIFAMILY" else [],
        missing_reason="Connected-unit count requires property details" if connected_unit_count is None and business_taxonomy != "MULTIFAMILY" else None,
    )
    facts["new_construction_project"] = _fact(
        "new_construction_project",
        new_construction_flag if customer_type == "COMMERCIAL" else None,
        source=FactSource.WORKBOOK if new_construction_flag is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile new-construction flag"] if new_construction_flag is not None else [],
        missing_reason="Project new-construction status is unavailable" if customer_type == "COMMERCIAL" and new_construction_flag is None else None,
    )
    facts["project_stage"] = _fact(
        "project_stage",
        project_stage,
        source=FactSource.WORKBOOK if project_stage is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile project stage"] if project_stage is not None else [],
        missing_reason="Project stage is unavailable" if project_stage is None else None,
    )
    facts["project_early_design"] = _fact(
        "project_early_design",
        project_stage == "EARLY_DESIGN" if project_stage is not None else None,
        source=FactSource.DERIVED if project_stage is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from project stage"] if project_stage is not None else [],
        missing_reason="Project stage is unavailable" if project_stage is None else None,
    )
    facts["conditioned_sqft"] = _fact(
        "conditioned_sqft",
        conditioned_sqft,
        source=FactSource.WORKBOOK if conditioned_sqft is not None else FactSource.SYSTEM,
        evidence=["Profile conditioned square footage"] if conditioned_sqft is not None else [],
        missing_reason="Conditioned square footage is unavailable" if conditioned_sqft is None else None,
    )
    facts["project_sqft_qualifies"] = _fact(
        "project_sqft_qualifies",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Program-specific square-footage qualification still requires explicit screening",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["project_exceeds_baseline_by_10"] = _fact(
        "project_exceeds_baseline_by_10",
        project_exceeds_baseline_10,
        source=FactSource.WORKBOOK if project_exceeds_baseline_10 is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile baseline-performance flag"] if project_exceeds_baseline_10 is not None else [],
        missing_reason="Baseline performance screening is unavailable" if project_exceeds_baseline_10 is None else None,
    )
    facts["energy_star_multifamily_project"] = _fact(
        "energy_star_multifamily_project",
        energy_star_multifamily,
        source=FactSource.WORKBOOK if energy_star_multifamily is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile ENERGY STAR MFNC flag"] if energy_star_multifamily is not None else [],
        missing_reason="ENERGY STAR MFNC screening is unavailable" if energy_star_multifamily is None else None,
    )
    facts["qualified_price_plan"] = _fact(
        "qualified_price_plan",
        qualified_price_plan if qualified_price_plan is not None else bool(current_rate_plan) if customer_type == "COMMERCIAL" else None,
        source=FactSource.WORKBOOK if qualified_price_plan is not None else FactSource.DERIVED if customer_type == "COMMERCIAL" and current_rate_plan else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile qualified price-plan flag"] if qualified_price_plan is not None else ["Commercial current price plan"] if customer_type == "COMMERCIAL" and current_rate_plan else [],
        missing_reason="Qualified price-plan confirmation requires commercial project details" if customer_type == "COMMERCIAL" and qualified_price_plan is None and not current_rate_plan else None,
    )
    facts["large_energy_consumer"] = _fact(
        "large_energy_consumer",
        large_energy_consumer if large_energy_consumer is not None else (
            True if (facts["annual_usage_estimate_kwh"].value or 0) >= DEMAND_RESPONSE_USAGE_THRESHOLD else None
        ),
        source=FactSource.WORKBOOK if large_energy_consumer is not None else FactSource.DERIVED if (facts["annual_usage_estimate_kwh"].value or 0) >= DEMAND_RESPONSE_USAGE_THRESHOLD else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile large-energy-consumer flag"] if large_energy_consumer is not None else ["Usage-based large-consumer fallback screen"] if (facts["annual_usage_estimate_kwh"].value or 0) >= DEMAND_RESPONSE_USAGE_THRESHOLD else [],
        missing_reason="Large-consumer screening is unavailable" if large_energy_consumer is None and (facts["annual_usage_estimate_kwh"].value or 0) < DEMAND_RESPONSE_USAGE_THRESHOLD else None,
    )
    facts["can_participate_in_demand_response"] = _fact(
        "can_participate_in_demand_response",
        curtailment_capability,
        source=FactSource.WORKBOOK if curtailment_capability is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Profile curtailment-capability flag"] if curtailment_capability is not None else [],
        missing_reason="Curtailment capability is unavailable" if curtailment_capability is None else None,
    )
    facts["site_can_install_ev_charging"] = _fact(
        "site_can_install_ev_charging",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Site EV-charging readiness is not stored in the current profile data",
        confidence=Confidence.MANUAL_REVIEW,
    )

    prepay_events_this_month = 0
    if not events.empty and COL_EVENT_TYPE in events.columns and COL_EVENT_DATE in events.columns:
        prepay_rows = events[
            (events[COL_EVENT_PROGRAM_ID] == "prepay_advance")
            & (events[COL_EVENT_TYPE].astype(str).str.casefold() == "offered")
        ]
        if not prepay_rows.empty:
            latest_date = prepay_rows[COL_EVENT_DATE].max()
            month_mask = (
                prepay_rows[COL_EVENT_DATE].dt.year == latest_date.year
            ) & (
                prepay_rows[COL_EVENT_DATE].dt.month == latest_date.month
            )
            prepay_events_this_month = int(month_mask.sum())
    facts["prepay_advance_offers_this_month"] = _fact(
        "prepay_advance_offers_this_month",
        prepay_events_this_month if prepay_events_this_month or not events.empty else None,
        source=FactSource.EXTERNAL if not events.empty else FactSource.SYSTEM,
        evidence=["Program event history"] if not events.empty else [],
        missing_reason="Program event history is unavailable for this account" if events.empty else None,
        confidence=Confidence.HIGH if not events.empty else Confidence.MANUAL_REVIEW,
    )

    facts["new_rooftop_solar_installation"] = _fact(
        "new_rooftop_solar_installation",
        der_new_rooftop_solar,
        source=FactSource.WORKBOOK if der_new_rooftop_solar is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile rooftop-solar flag"] if der_new_rooftop_solar is not None else [],
        missing_reason="New-rooftop-solar status is unavailable" if der_new_rooftop_solar is None else None,
    )
    facts["preferred_solar_installer"] = _fact(
        "preferred_solar_installer",
        der_preferred_installer,
        source=FactSource.WORKBOOK if der_preferred_installer is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile preferred-installer flag"] if der_preferred_installer is not None else [],
        missing_reason="Preferred-installer status is unavailable" if der_preferred_installer is None else None,
    )
    facts["demand_management_system_included"] = _fact(
        "demand_management_system_included",
        demand_mgmt_included,
        source=FactSource.WORKBOOK if demand_mgmt_included is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile demand-management inclusion flag"] if demand_mgmt_included is not None else [],
        missing_reason="Demand-management system inclusion is unavailable" if demand_mgmt_included is None else None,
    )
    facts["qualifying_facility_kw_ac"] = _fact(
        "qualifying_facility_kw_ac",
        der_qf_kw_ac,
        source=FactSource.WORKBOOK if der_qf_kw_ac is not None else FactSource.SYSTEM,
        evidence=["DER profile qualifying-facility size"] if der_qf_kw_ac is not None else [],
        missing_reason="Qualifying-facility size is unavailable" if der_qf_kw_ac is None else None,
    )
    facts["qualifying_facility_le_100kw"] = _fact(
        "qualifying_facility_le_100kw",
        der_qf_kw_ac <= 100.0 if der_qf_kw_ac is not None else None,
        source=FactSource.DERIVED if der_qf_kw_ac is not None else FactSource.SYSTEM,
        evidence=["Derived from DER profile qualifying-facility size"] if der_qf_kw_ac is not None else [],
        missing_reason="Qualifying-facility size is unavailable" if der_qf_kw_ac is None else None,
    )
    facts["storage_only_configuration"] = _fact(
        "storage_only_configuration",
        der_storage_only,
        source=FactSource.WORKBOOK if der_storage_only is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile storage-only flag"] if der_storage_only is not None else [],
        missing_reason="Storage-only configuration is unavailable" if der_storage_only is None else None,
    )
    facts["eligible_connected_devices"] = _fact(
        "eligible_connected_devices",
        eligible_connected_device,
        source=FactSource.WORKBOOK if eligible_connected_device is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile eligible-connected-device flag"] if eligible_connected_device is not None else [],
        missing_reason="Eligible connected-device status is unavailable" if eligible_connected_device is None else None,
    )
    facts["tesla_vehicle_ownership"] = _fact(
        "tesla_vehicle_ownership",
        True if battery_partner_brand and battery_partner_brand.casefold() == "tesla" else None,
        source=FactSource.DERIVED if battery_partner_brand is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Tesla battery/brand signal in DER profile"] if battery_partner_brand and battery_partner_brand.casefold() == "tesla" else [],
        missing_reason="Tesla ownership is unavailable" if battery_partner_brand is None else None,
    )
    facts["eligible_home_ev_charger"] = _fact(
        "eligible_home_ev_charger",
        eligible_home_ev_charger,
        source=FactSource.WORKBOOK if eligible_home_ev_charger is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile eligible-home-EV-charger flag"] if eligible_home_ev_charger is not None else [],
        missing_reason="Home EV charger eligibility is unavailable" if eligible_home_ev_charger is None else None,
    )
    facts["battery_ownership"] = _fact(
        "battery_ownership",
        battery_ownership,
        source=FactSource.WORKBOOK if battery_ownership is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile battery-ownership flag"] if battery_ownership is not None else [],
        missing_reason="Battery ownership is unavailable" if battery_ownership is None else None,
    )
    facts["compatible_battery_configuration"] = _fact(
        "compatible_battery_configuration",
        compatible_battery,
        source=FactSource.WORKBOOK if compatible_battery is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile compatible-battery flag"] if compatible_battery is not None else [],
        missing_reason="Compatible battery configuration is unavailable" if compatible_battery is None else None,
    )
    facts["rec_rights_owned"] = _fact(
        "rec_rights_owned",
        der_rec_rights_owned,
        source=FactSource.WORKBOOK if der_rec_rights_owned is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile REC-rights flag"] if der_rec_rights_owned is not None else [],
        missing_reason="REC-rights ownership is unavailable" if der_rec_rights_owned is None else None,
    )
    facts["prior_rec_assignment_active"] = _fact(
        "prior_rec_assignment_active",
        der_prior_rec_assignment,
        source=FactSource.WORKBOOK if der_prior_rec_assignment is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["DER profile prior-REC-assignment flag"] if der_prior_rec_assignment is not None else [],
        missing_reason="Prior REC assignment status is unavailable" if der_prior_rec_assignment is None else None,
    )
    normalized_battery_brand = battery_partner_brand.upper() if battery_partner_brand else None
    facts["battery_partner_brand"] = _fact(
        "battery_partner_brand",
        normalized_battery_brand,
        source=FactSource.WORKBOOK if normalized_battery_brand is not None else FactSource.SYSTEM,
        evidence=["DER profile battery partner brand"] if normalized_battery_brand is not None else [],
        missing_reason="Battery partner brand is unavailable" if normalized_battery_brand is None else None,
    )
    facts["battery_partner_brand_supported"] = _fact(
        "battery_partner_brand_supported",
        normalized_battery_brand in SUPPORTED_BATTERY_PARTNER_BRANDS if normalized_battery_brand is not None else None,
        source=FactSource.DERIVED if normalized_battery_brand is not None else FactSource.CUSTOMER_ANSWER,
        evidence=["Derived from documented Battery Partner brands"] if normalized_battery_brand is not None else [],
        missing_reason="Battery partner brand is unavailable" if normalized_battery_brand is None else None,
    )
    facts["legal_right_to_plant"] = _fact(
        "legal_right_to_plant",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Property planting rights are not stored in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["completed_shade_tree_workshop"] = _fact(
        "completed_shade_tree_workshop",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Shade-tree workshop attendance is not stored in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["already_received_shade_tree"] = _fact(
        "already_received_shade_tree",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Prior shade-tree participation is not stored in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["household_income_qualified"] = _fact(
        "household_income_qualified",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Income qualification is not stored in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["income_assistance_auto_qualifier"] = _fact(
        "income_assistance_auto_qualifier",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Income-assistance auto-qualification is not stored in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["customer_requests_custom_due_date"] = _fact(
        "customer_requests_custom_due_date",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Customer intent is not present in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["customer_wants_followup"] = _fact(
        "customer_wants_followup",
        None,
        source=FactSource.CUSTOMER_ANSWER,
        missing_reason="Follow-up intent is not present in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )
    facts["has_60_day_arrears"] = _fact(
        "has_60_day_arrears",
        None,
        source=FactSource.SYSTEM,
        missing_reason="60-day arrears status is not present in the current data",
        confidence=Confidence.MANUAL_REVIEW,
    )

    for fact_id in FACT_REGISTRY:
        if fact_id not in facts:
            facts[fact_id] = _fact(
                fact_id,
                source=FactSource.SYSTEM,
                missing_reason="Fact not derivable from the current runtime state",
                confidence=Confidence.MANUAL_REVIEW,
            )

    facts["_persona_hints"] = FactValue(
        fact_id="_persona_hints",
        value_type="dict",
        source=FactSource.DERIVED,
        value=get_persona_hints(segment_name),
        confidence=Confidence.HIGH,
        evidence=["Persona matrix"],
    )
    facts["_program_event_history"] = FactValue(
        fact_id="_program_event_history",
        value_type="list[dict]",
        source=FactSource.EXTERNAL,
        value=events.to_dict(orient="records"),
        confidence=Confidence.HIGH if not events.empty else Confidence.MANUAL_REVIEW,
        evidence=["Program event history"] if not events.empty else [],
        missing_reason="Program event history unavailable" if events.empty else None,
    )
    facts["_decline_suppression_days"] = FactValue(
        fact_id="_decline_suppression_days",
        value_type="integer",
        source=FactSource.SYSTEM,
        value=DECLINE_SUPPRESSION_DAYS,
        confidence=Confidence.HIGH,
        evidence=["Configured decline suppression window"],
    )

    return facts
