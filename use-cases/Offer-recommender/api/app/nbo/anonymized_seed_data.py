"""Anonymized HANA-shaped seed data for public demo workbooks and tests."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.nbo.config import (
    COL_BILLING_ACCOUNT,
    COL_BILL_LANGUAGE,
    COL_BILL_PAID,
    COL_BILL_TOTAL,
    COL_BIZ_OFFERING_CODE,
    COL_BIZ_OFFERING_NAME,
    COL_COMM_BILLING_ACCOUNT,
    COL_COMM_SEGMENT,
    COL_COMM_SEGMENT_NAME,
    COL_COUNTY,
    COL_CREDIT_RATING,
    COL_CUSTOMER_TYPE,
    COL_DER_BATTERY_OWNERSHIP,
    COL_DER_BATTERY_PARTNER_BRAND,
    COL_DER_COMPATIBLE_BATTERY,
    COL_DER_DEMAND_MGMT_INCLUDED,
    COL_DER_ELIGIBLE_CONNECTED_DEVICE,
    COL_DER_ELIGIBLE_HOME_EV_CHARGER,
    COL_DER_EV_CHARGER_BRAND,
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
    COL_EVENT_PROGRAM_CODE,
    COL_EVENT_PROGRAM_ID,
    COL_EVENT_TYPE,
    COL_INDUSTRY,
    COL_METER_USAGE,
    COL_NAIC_CODE,
    COL_OFF_PEAK,
    COL_ON_PEAK,
    COL_OTHER_PROGRAMS,
    COL_PAYMENT_PLAN,
    COL_PC_PROGRAM,
    COL_PC_PROGRAM_OR_CONTRACT,
    COL_PC_SCREEN_NAME,
    COL_PC_SERVICE_OPTION,
    COL_PC_SERVICE_TYPE,
    COL_PROFILE_ACCOUNT_GOOD_STANDING,
    COL_PROFILE_ACCOUNT_NAME_TYPE,
    COL_PROFILE_CENTRAL_AC,
    COL_PROFILE_CONDITIONED_SQFT,
    COL_PROFILE_CONNECTED_UNIT_COUNT,
    COL_PROFILE_COOLING_SYSTEM_TYPE,
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
    COL_SEGMENT_NAME,
    COL_SEGMENT_NUM,
    COL_STATUS,
)


def synthetic_runtime_datasets() -> dict[str, pd.DataFrame]:
    """Return anonymized HANA-shaped DataFrames for backend tests.

    Output:
        A complete runtime dataset mapping with deterministic customer,
        offering, profile, DER, and program-history tables.
    """
    return {
        "residential": _residential_rows(),
        "res_segment": _residential_segments(),
        "commercial": _commercial_rows(),
        "comm_segment": _commercial_segments(),
        "active_offering": _active_offering_rows(),
        "program_contract": _program_contract_rows(),
        "program_samples": _program_sample_rows(),
        "account_profile": _account_profile_rows(),
        "der_profile": _der_profile_rows(),
        "program_event_history": _program_event_history_rows(),
    }


def _base_customer_row(account: str, rate_plan: str, read_date: datetime | None) -> dict:
    """Build one anonymized residential billing-history row."""
    return {
        COL_BILLING_ACCOUNT: account,
        COL_STATUS: "ACTIVE",
        COL_RATE_PLAN: rate_plan,
        COL_PAYMENT_PLAN: "",
        COL_CREDIT_RATING: "A",
        COL_COUNTY: "000",
        COL_BILL_LANGUAGE: "EN",
        COL_METER_USAGE: 900.0,
        COL_BILL_PAID: 125.0,
        COL_BILL_TOTAL: 135.0,
        COL_READ_DATE: read_date,
        COL_ON_PEAK: 100.0,
        COL_OFF_PEAK: 800.0,
        COL_OTHER_PROGRAMS: "",
    }


def _residential_rows() -> pd.DataFrame:
    """Create residential accounts used by recommendation regression tests."""
    rows: list[dict] = []
    for account, rate_plan in (("104", "E21"), ("6001", "E21"), ("100000", "E26")):
        for month in (3, 4, 5, 6, 7, 8):
            rows.append(
                {
                    **_base_customer_row(account, rate_plan, datetime(2025, month, 5)),
                    COL_METER_USAGE: 900.0 if account != "100000" else 1500.0,
                    COL_BILL_TOTAL: 135.0 if account != "100000" else 230.0,
                    COL_BILL_PAID: 125.0 if account != "100000" else 230.0,
                    COL_ON_PEAK: 100.0 if account != "100000" else 300.0,
                    COL_OFF_PEAK: 800.0 if account != "100000" else 1200.0,
                }
            )

    rows.append(_base_customer_row("103", "E23", None))
    disconnected = _base_customer_row("10106", "E24", datetime(2025, 6, 15))
    disconnected[COL_STATUS] = "DISCONNECTED"
    disconnected[COL_METER_USAGE] = 450.0
    disconnected[COL_BILL_TOTAL] = 75.0
    disconnected[COL_BILL_PAID] = 50.0
    rows.append(disconnected)
    return pd.DataFrame(rows)


def _residential_segments() -> pd.DataFrame:
    """Create anonymized residential segment assignments."""
    return pd.DataFrame(
        [
            {COL_BILLING_ACCOUNT: "104", COL_SEGMENT_NUM: 1, COL_SEGMENT_NAME: "Budget Planners"},
            {COL_BILLING_ACCOUNT: "6001", COL_SEGMENT_NUM: 2, COL_SEGMENT_NAME: "Digital Savers"},
            {COL_BILLING_ACCOUNT: "100000", COL_SEGMENT_NUM: 3, COL_SEGMENT_NAME: "Energy Managers"},
            {COL_BILLING_ACCOUNT: "10106", COL_SEGMENT_NUM: 0, COL_SEGMENT_NAME: "NOT FOUND"},
        ]
    )


def _commercial_rows() -> pd.DataFrame:
    """Create a commercial account with current billing history."""
    rows = []
    for month in (3, 4, 5, 6, 7, 8):
        rows.append(
            {
                COL_BILLING_ACCOUNT: "1004",
                COL_STATUS: "ACTIVE",
                COL_RATE_PLAN: "E36",
                COL_PAYMENT_PLAN: "",
                COL_CREDIT_RATING: "A",
                COL_COUNTY: "000",
                COL_BILL_LANGUAGE: "EN",
                COL_METER_USAGE: 2500.0,
                COL_BILL_PAID: 320.0,
                COL_BILL_TOTAL: 340.0,
                COL_READ_DATE: datetime(2025, month, 10),
                COL_ON_PEAK: 900.0,
                COL_OFF_PEAK: 1600.0,
                COL_OTHER_PROGRAMS: "",
                COL_NAIC_CODE: "221310",
                COL_INDUSTRY: "Water services",
            }
        )
    return pd.DataFrame(rows)


def _commercial_segments() -> pd.DataFrame:
    """Create anonymized commercial segment assignments."""
    return pd.DataFrame(
        [
            {
                COL_COMM_BILLING_ACCOUNT: "1004",
                COL_COMM_SEGMENT: "C1",
                COL_COMM_SEGMENT_NAME: "Small Operations",
            }
        ]
    )


def _active_offering_rows() -> pd.DataFrame:
    """Create anonymized rate-plan display rows."""
    rows = [
        ("E00", "E00-Inactive account", "RESIDENTIAL"),
        ("E21", "E21-Residential Time of Use 3-6", "RESIDENTIAL"),
        ("E22", "E22-Residential Time of Use 4-7", "RESIDENTIAL"),
        ("E23", "E23-Basic Residential Plan", "RESIDENTIAL"),
        ("E24", "E24-Prepay Plan", "RESIDENTIAL"),
        ("E26", "E26-Residential Time of Use", "RESIDENTIAL"),
        ("E28", "E28-Conserve 6-9 P.M.", "RESIDENTIAL"),
        ("E29", "E29-EV Time of Use", "RESIDENTIAL"),
        ("E13", "E13-Time of Use Export", "RESIDENTIAL"),
        ("E14", "E14-Customer Generation EV Export", "RESIDENTIAL"),
        ("E15", "E15-Customer Generation Average Demand", "RESIDENTIAL"),
        ("E16", "E16-Demand Saver 5-10 P.M.", "RESIDENTIAL"),
        ("E17", "E17-Residential Plan", "RESIDENTIAL"),
        ("E18", "E18-Prepay Legacy", "RESIDENTIAL"),
        ("E19", "E19-Residential Plan", "RESIDENTIAL"),
        ("E27", "E27-Customer Generation", "RESIDENTIAL"),
        ("E36", "E36-General Commercial Service", "COMMERCIAL"),
    ]
    return pd.DataFrame(
        [
            {
                COL_RATE_PLAN: rate_plan,
                COL_BIZ_OFFERING_CODE: rate_plan,
                COL_BIZ_OFFERING_NAME: name,
                COL_CUSTOMER_TYPE: customer_type,
            }
            for rate_plan, name, customer_type in rows
        ]
    )


def _program_contract_rows() -> pd.DataFrame:
    """Create anonymized program-code aliases used by catalog enrichment."""
    return pd.DataFrame(
        [
            {
                COL_PC_PROGRAM: "Household Assistance Discount",
                COL_PC_SCREEN_NAME: "Assistance Discount",
                COL_PC_SERVICE_OPTION: "HAD",
                COL_PC_SERVICE_TYPE: "PROGRAM",
                COL_PC_PROGRAM_OR_CONTRACT: "Program",
            },
            {
                COL_PC_PROGRAM: "Residential Battery Incentive",
                COL_PC_SCREEN_NAME: "Battery Partner",
                COL_PC_SERVICE_OPTION: "BATSI",
                COL_PC_SERVICE_TYPE: "PROGRAM",
                COL_PC_PROGRAM_OR_CONTRACT: "Program",
            },
        ]
    )


def _program_sample_rows() -> pd.DataFrame:
    """Create a minimal program-sample table for loader and datastore tests."""
    return pd.DataFrame(
        [
            {COL_BILLING_ACCOUNT: "104", "Program": "Household Assistance Discount"},
            {COL_BILLING_ACCOUNT: "6001", "Program": "Battery Partner"},
        ]
    )


def _account_profile_rows() -> pd.DataFrame:
    """Create profile rows with most optional facts intentionally blank."""
    columns = [
        COL_BILLING_ACCOUNT,
        COL_PROFILE_DWELLING_TYPE,
        COL_PROFILE_SERVICE_ENTRANCE_AMPS,
        COL_PROFILE_SERVICE_START_DATE,
        COL_PROFILE_OWNERSHIP_STATUS,
        COL_PROFILE_OCCUPANCY_STATUS,
        COL_PROFILE_COOLING_SYSTEM_TYPE,
        COL_PROFILE_CENTRAL_AC,
        COL_PROFILE_NEW_CONSTRUCTION,
        COL_PROFILE_CONNECTED_UNIT_COUNT,
        COL_PROFILE_PROJECT_STAGE,
        COL_PROFILE_CONDITIONED_SQFT,
        COL_PROFILE_ACCOUNT_GOOD_STANDING,
        COL_PROFILE_ACCOUNT_NAME_TYPE,
        COL_PROFILE_SAME_ACCOUNT_HOLDER_12M,
        COL_PROFILE_QUALIFIED_PRICE_PLAN,
        COL_PROFILE_LARGE_ENERGY_CONSUMER,
        COL_PROFILE_CURTAILMENT_CAPABILITY,
        COL_PROFILE_ELIGIBLE_CONTRACTOR,
        COL_PROFILE_PROJECT_EXCEEDS_BASELINE_10,
        COL_PROFILE_ENERGY_STAR_MFNC,
    ]
    return pd.DataFrame(columns=columns)


def _der_profile_rows() -> pd.DataFrame:
    """Create an empty DER profile table with runtime columns."""
    columns = [
        COL_BILLING_ACCOUNT,
        COL_DER_SOLAR_OWNERSHIP,
        COL_DER_NEW_ROOFTOP_SOLAR,
        COL_DER_QUALIFYING_FACILITY_KW_AC,
        COL_DER_STORAGE_ONLY,
        COL_DER_REC_RIGHTS_OWNED,
        COL_DER_PRIOR_REC_ASSIGNMENT,
        COL_DER_PREFERRED_SOLAR_INSTALLER,
        COL_DER_THERMOSTAT_BRAND,
        COL_DER_THERMOSTAT_PROVIDER_ACCOUNT,
        COL_DER_THERMOSTAT_WIFI,
        COL_DER_ELIGIBLE_CONNECTED_DEVICE,
        COL_DER_SMART_THERMOSTAT_PURCHASE,
        COL_DER_EV_CHARGER_BRAND,
        COL_DER_ELIGIBLE_HOME_EV_CHARGER,
        COL_DER_BATTERY_OWNERSHIP,
        COL_DER_BATTERY_PARTNER_BRAND,
        COL_DER_COMPATIBLE_BATTERY,
        COL_DER_DEMAND_MGMT_INCLUDED,
    ]
    return pd.DataFrame(columns=columns)


def _program_event_history_rows() -> pd.DataFrame:
    """Create an empty program event history table with runtime columns."""
    return pd.DataFrame(
        columns=[
            COL_BILLING_ACCOUNT,
            COL_EVENT_DATE,
            COL_EVENT_PROGRAM_ID,
            COL_EVENT_PROGRAM_CODE,
            COL_EVENT_TYPE,
        ]
    )
