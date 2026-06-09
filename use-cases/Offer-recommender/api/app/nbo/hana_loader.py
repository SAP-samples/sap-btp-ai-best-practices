from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.nbo.config import (
    COL_BILLING_ACCOUNT,
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
    PROGRAM_CODES_SHEET_CONTRACT,
    PROGRAM_CODES_SHEET_SAMPLES,
    SHEET_ACTIVE_OFFERING,
    SHEET_COMMERCIAL,
    SHEET_COMMERCIAL_SEGMENT,
    SHEET_RESIDENTIAL,
    SHEET_RESIDENTIAL_SEGMENT,
)
from app.nbo.hana import DATASET_TABLES, quote_identifier


def load_seed_datasets(
    customer_workbook,
    program_codes_workbook,
) -> dict[str, pd.DataFrame]:
    """Load explicit local seed workbooks for HANA bootstrap workflows.

    Inputs:
        customer_workbook: Path to the customer, segment, and active-offering
            workbook.
        program_codes_workbook: Path to the program contract and sample account
            workbook.

    Output:
        Runtime dataset names mapped to HANA-shaped DataFrames.
    """
    account_profile = pd.DataFrame(
        columns=[
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
    )
    der_profile = pd.DataFrame(
        columns=[
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
    )
    program_event_history = pd.DataFrame(
        columns=[
            COL_BILLING_ACCOUNT,
            COL_EVENT_DATE,
            COL_EVENT_PROGRAM_ID,
            COL_EVENT_PROGRAM_CODE,
            COL_EVENT_TYPE,
        ]
    )
    return {
        "residential": pd.read_excel(customer_workbook, sheet_name=SHEET_RESIDENTIAL),
        "res_segment": pd.read_excel(
            customer_workbook,
            sheet_name=SHEET_RESIDENTIAL_SEGMENT,
        ),
        "commercial": pd.read_excel(customer_workbook, sheet_name=SHEET_COMMERCIAL),
        "comm_segment": pd.read_excel(
            customer_workbook,
            sheet_name=SHEET_COMMERCIAL_SEGMENT,
        ),
        "active_offering": pd.read_excel(
            customer_workbook,
            sheet_name=SHEET_ACTIVE_OFFERING,
        ),
        "program_contract": pd.read_excel(
            program_codes_workbook,
            sheet_name=PROGRAM_CODES_SHEET_CONTRACT,
            header=1,
        ),
        "program_samples": pd.read_excel(
            program_codes_workbook,
            sheet_name=PROGRAM_CODES_SHEET_SAMPLES,
        ),
        "account_profile": account_profile,
        "der_profile": der_profile,
        "program_event_history": program_event_history,
    }


def hana_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    return "NVARCHAR(5000)"


def create_table_sql(table_name: str, df: pd.DataFrame) -> str:
    columns_sql = ", ".join(
        f"{quote_identifier(column)} {hana_column_type(df[column])}"
        for column in df.columns
    )
    return f"CREATE COLUMN TABLE {quote_identifier(table_name)} ({columns_sql})"


def insert_sql(table_name: str, columns: Iterable[str]) -> str:
    column_list = list(columns)
    quoted_columns = ", ".join(quote_identifier(column) for column in column_list)
    placeholders = ", ".join("?" for _ in column_list)
    return (
        f"INSERT INTO {quote_identifier(table_name)} ({quoted_columns}) "
        f"VALUES ({placeholders})"
    )


def dataframe_rows(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for record in df.itertuples(index=False, name=None):
        normalized: list[Any] = []
        for value in record:
            if pd.isna(value):
                normalized.append(None)
            elif isinstance(value, pd.Timestamp):
                normalized.append(value.to_pydatetime())
            elif isinstance(value, np.generic):
                normalized.append(value.item())
            elif isinstance(value, datetime):
                normalized.append(value)
            else:
                normalized.append(value)
        rows.append(tuple(normalized))
    return rows


def recreate_and_load_tables(
    connection,
    datasets: dict[str, pd.DataFrame],
) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    cursor = connection.cursor()
    try:
        for dataset_name, table_name in DATASET_TABLES.items():
            df = datasets[dataset_name]
            rows = dataframe_rows(df)
            try:
                cursor.execute(f"DROP TABLE {quote_identifier(table_name)}")
            except Exception:
                pass

            cursor.execute(create_table_sql(table_name, df))
            if rows:
                cursor.executemany(insert_sql(table_name, df.columns), rows)
            row_counts[table_name] = len(rows)

        connection.commit()
    finally:
        cursor.close()

    return row_counts
