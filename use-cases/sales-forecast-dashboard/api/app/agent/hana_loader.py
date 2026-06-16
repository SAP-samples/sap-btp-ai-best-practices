"""
HANA data loader module for the forecasting agent.

Provides centralized functions to load data from SAP HANA database.
This module replaces file-based loading with database queries while
maintaining the same API as the original io_utils functions.

Connection is configured via environment variables:
- hana_address: HANA server address
- hana_port: HANA port (default 443)
- hana_user: Database user
- hana_password: Database password
- hana_encrypt: Use encryption (default 'true')
- HANA_SCHEMA: Target schema (default 'AICOE')
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# HANA imports
from hana_ml import ConnectionContext


# =============================================================================
# Column Name Normalization for MODEL_B
# =============================================================================
# The CatBoost model was trained with TitleCase column names from CSV files,
# but HANA returns uppercase columns which get lowercased. These mappings
# convert specific columns to match the model's expected format.

MODEL_B_COLUMN_MAPPING = {
    'conversionrate_lag_1': 'ConversionRate_lag_1',
    'conversionrate_lag_4': 'ConversionRate_lag_4',
    'conversionrate_roll_mean_4': 'ConversionRate_roll_mean_4',
    'conversionrate_roll_mean_8': 'ConversionRate_roll_mean_8',
    'conversionrate_roll_mean_13': 'ConversionRate_roll_mean_13',
    'aov_roll_mean_8': 'AOV_roll_mean_8',
    'aov_roll_mean_13': 'AOV_roll_mean_13',
}


def _normalize_model_b_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize MODEL_B column names to match model expectations.

    HANA returns uppercase columns which get lowercased, but the CatBoost
    model was trained with TitleCase for ConversionRate and AOV columns.

    Args:
        df: DataFrame with lowercase column names from HANA

    Returns:
        DataFrame with normalized column names matching model expectations
    """
    # First lowercase all columns (existing behavior)
    df.columns = [col.lower() for col in df.columns]

    # Then apply specific mappings for TitleCase columns
    df = df.rename(columns=MODEL_B_COLUMN_MAPPING)

    return df


# =============================================================================
# HANA Fiscal Field Parsing Utilities
# =============================================================================
# HANA stores fiscal fields in string format ("Week 49", "Q1") rather than
# integers. These utilities provide robust parsing that handles both formats.

def parse_fiscal_week(value) -> Optional[int]:
    """
    Parse fiscal week from HANA format.

    HANA stores fiscal_week as "Week 49" (string) not just 49 (int).
    This handles both formats for robustness.

    Args:
        value: The fiscal week value (string "Week 49" or int 49)

    Returns:
        Integer week number or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Handle "Week 49" format from HANA
        cleaned = value.replace('Week', '').strip()
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def parse_fiscal_quarter(value) -> Optional[int]:
    """
    Parse fiscal quarter from HANA format.

    HANA stores fiscal_quarter as "Q1" (string) not just 1 (int).
    This handles both formats for robustness.

    Args:
        value: The fiscal quarter value (string "Q1" or int 1)

    Returns:
        Integer quarter (1-4) or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Handle "Q1", "Q2", "Q3", "Q4" format from HANA
        cleaned = value.replace('Q', '').replace('q', '').strip()
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def parse_fiscal_int(value) -> Optional[int]:
    """
    Parse a fiscal field that should be an integer but may be stored as string.

    Use for fiscal_year, fiscal_month, or other numeric fields that may
    come from HANA as strings.

    Args:
        value: The fiscal value (string or int)

    Returns:
        Integer value or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


# =============================================================================
# Connection Management
# =============================================================================

# Global connection cache
_connection: Optional[ConnectionContext] = None


def get_hana_connection() -> ConnectionContext:
    """
    Get or create HANA connection.

    Uses a cached connection for efficiency. The connection is created
    on first call and reused for subsequent calls.

    Returns:
        ConnectionContext: Active HANA connection

    Raises:
        ValueError: If required environment variables are not set
        Exception: If connection fails
    """
    global _connection

    if _connection is not None:
        # Check if connection is still alive
        try:
            _connection.hana_version()
            return _connection
        except Exception:
            # Connection is dead, recreate
            _connection = None

    hana_address = os.getenv('hana_address')
    if not hana_address:
        raise ValueError(
            "HANA connection not configured. "
            "Set hana_address, hana_user, and hana_password environment variables."
        )

    hana_port = int(os.getenv('hana_port', 443))
    hana_user = os.getenv('hana_user')
    hana_password = os.getenv('hana_password')
    hana_encrypt = os.getenv('hana_encrypt', 'true').lower() == 'true'
    hana_schema = os.getenv('HANA_SCHEMA', 'AICOE')

    if not hana_user or not hana_password:
        raise ValueError(
            "HANA credentials not configured. "
            "Set hana_user and hana_password environment variables."
        )

    _connection = ConnectionContext(
        address=hana_address,
        port=hana_port,
        user=hana_user,
        password=hana_password,
        encrypt=hana_encrypt,
        current_schema=hana_schema,
    )

    return _connection


def close_connection() -> None:
    """Close the cached HANA connection."""
    global _connection
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
        _connection = None


# =============================================================================
# Generic Query Functions
# =============================================================================

def query_to_dataframe(query: str) -> pd.DataFrame:
    """
    Execute SQL query and return results as pandas DataFrame.

    Args:
        query: SQL query string

    Returns:
        DataFrame with query results
    """
    cc = get_hana_connection()
    hdf = cc.sql(query)
    return hdf.collect()


def load_table(
    table_name: str,
    columns: Optional[List[str]] = None,
    where_clause: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load table from HANA into pandas DataFrame.

    Args:
        table_name: Name of the HANA table
        columns: Optional list of columns to select (default: all)
        where_clause: Optional WHERE clause (without 'WHERE' keyword)

    Returns:
        DataFrame with table data
    """
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    # Build column selection
    col_str = ", ".join(columns) if columns else "*"

    # Build query
    query = f'SELECT {col_str} FROM "{schema}"."{table_name}"'
    if where_clause:
        query += f" WHERE {where_clause}"

    hdf = cc.sql(query)
    return hdf.collect()


# =============================================================================
# Table-Specific Loaders
# =============================================================================

@lru_cache(maxsize=1)
def load_model_b() -> pd.DataFrame:
    """
    Load MODEL_B table from HANA.

    This is the main feature matrix with historical data for forecasting.

    Returns:
        DataFrame with columns normalized to match model expectations:
        - profit_center_nbr, dma, channel, origin_week_date, target_week_date, horizon
        - All feature columns (is_outlet, is_comp_store, rolling means, lags, etc.)
        - Label columns (label_log_sales, label_log_aov, label_logit_conversion)

    Note:
        Column names are normalized to match CatBoost model expectations,
        including TitleCase for ConversionRate and AOV columns.
    """
    df = load_table("MODEL_B")
    df = _normalize_model_b_columns(df)
    return df


@lru_cache(maxsize=1)
def load_predictions_bm() -> pd.DataFrame:
    """
    Load PREDICTIONS_BM table from HANA.

    Contains pre-computed B&M sales predictions with confidence intervals.

    Returns:
        DataFrame with all MODEL_B columns plus prediction columns (lowercase):
        - pred_log_sales, pred_log_aov, pred_log_orders, pred_logit_conversion
        - pred_sales_mean, pred_sales_p50, pred_sales_p90
        - pred_aov_mean, pred_aov_p50, pred_aov_p90
        - pred_traffic_p10, pred_traffic_p50, pred_traffic_p90
    """
    df = load_table("PREDICTIONS_BM")
    df.columns = [col.lower() for col in df.columns]
    return df


@lru_cache(maxsize=1)
def load_predictions_web() -> pd.DataFrame:
    """
    Load PREDICTIONS_WEB table from HANA.

    Contains pre-computed WEB sales predictions.

    Returns:
        DataFrame with feature columns and prediction columns (lowercase)
        (similar to PREDICTIONS_BM but for WEB channel)
    """
    df = load_table("PREDICTIONS_WEB")
    df.columns = [col.lower() for col in df.columns]
    return df


@lru_cache(maxsize=1)
def load_store_master() -> pd.DataFrame:
    """
    Load PROFIT_CENTER table from HANA (store master data).

    This replaces io_utils.load_store_master() and returns store data
    with coordinates and metadata.

    Returns:
        DataFrame with columns:
        - PROFIT_CENTER_NBR: Store number
        - profit_center_name: Store name (derived from PROFIT_CENTER columns)
        - store_address: Address
        - market_city: DMA/market city
        - latitude, longitude: Coordinates
        - date_opened: Opening date
        - is_outlet: Outlet store flag
        - merchandising_sf: Merchandising square footage
        - proforma_annual_sales: Expected annual sales
    """
    df = load_table("PROFIT_CENTER")

    # Normalize column names to match expected output from io_utils.load_store_master()
    # The HANA table has uppercase column names (sanitized during upload), we need to map them
    column_mapping = {
        'PROFIT_CENTER_NBR': 'profit_center_nbr',
        'PROFIT_CENTER_NAME': 'profit_center_name',
        'STORE_ADDRESS': 'store_address',
        'CITY': 'city',
        'STATE': 'state',
        'MARKET___CITY': 'market_city',  # 3 underscores from "Market - City" sanitization
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'DATE_OPENED': 'date_opened',
        'DATE_CLOSED': 'date_closed',
        'OUTLET': 'is_outlet',
        'MERCHANDISING_SF': 'merchandising_sf',
        'STORE_DESIGN_SF': 'store_design_sf',
        'PROFORMA_SALES': 'proforma_annual_sales',
    }

    # Rename columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert date columns
    if 'date_opened' in df.columns:
        df['date_opened'] = pd.to_datetime(df['date_opened'], errors='coerce')
    if 'date_closed' in df.columns:
        df['date_closed'] = pd.to_datetime(df['date_closed'], errors='coerce')

    return df


@lru_cache(maxsize=1)
def load_awareness_consideration() -> pd.DataFrame:
    """
    Load AWARENESS_CONSIDERATION table from HANA.

    This replaces io_utils.load_awareness_consideration() and returns
    weekly brand awareness and consideration metrics by market.

    Returns:
        DataFrame with columns:
        - market: Market name (mapped from yougov DMA)
        - week_start: Week start date
        - awareness: Awareness percentage (0-100)
        - consideration: Consideration percentage (0-100)
    """
    df = load_table("AWARENESS_CONSIDERATION")

    # Normalize column names
    column_mapping = {
        'COMPANY': 'company',
        'MARKET': 'market',
        'FISCALSTARTDATEWEEK': 'week_start',
        'AWARENESS': 'awareness',
        'CONSIDERATION': 'consideration',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert date column
    if 'week_start' in df.columns:
        df['week_start'] = pd.to_datetime(df['week_start'], errors='coerce')

    return df


@lru_cache(maxsize=1)
def load_yougov_dma_map() -> pd.DataFrame:
    """
    Load YOUGOV_DMA_MAP table from HANA.

    This replaces io_utils.load_yougov_dma_map() and provides mapping
    from YouGov market names to DMA/market_city.

    Returns:
        DataFrame with columns:
        - yougov_dma: YouGov market name
        - region: Region
        - market: Aggregate market name
        - market_city: DMA city name
    """
    df = load_table("YOUGOV_DMA_MAP")

    # Normalize column names to lowercase first (HANA may return uppercase)
    df.columns = [col.lower() for col in df.columns]

    # Map to expected column names
    column_mapping = {
        'yougovdma': 'yougov_dma',
        'region': 'region',
        'market': 'market',
        'market_city': 'market_city',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    return df


@lru_cache(maxsize=1)
def load_budget_data() -> pd.DataFrame:
    """
    Load BUDGET_MARKETING table from HANA.

    This replaces io_utils.load_budget_data() and returns normalized
    marketing budget data by DMA and month.

    The data is already normalized in HANA (transformed from pivot format
    during upload).

    Returns:
        DataFrame with columns:
        - dma_budget: Original DMA name from budget file
        - market_city: Extracted/mapped city name
        - year: Fiscal year
        - month: Month number (1-12)
        - budget: Monthly budget in dollars
    """
    df = load_table("BUDGET_MARKETING")

    # Normalize column names to lowercase first (HANA may return uppercase)
    df.columns = [col.lower() for col in df.columns]

    # Map to expected column names
    column_mapping = {
        'dma_budget': 'dma_budget',
        'market_city': 'market_city',
        'year': 'year',
        'month': 'month',
        'budget': 'budget',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Ensure numeric types
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
    if 'budget' in df.columns:
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)

    return df


@lru_cache(maxsize=1)
def load_calendar() -> pd.DataFrame:
    """
    Load CALENDAR table from HANA.

    Returns fiscal calendar with date dimension data.

    Returns:
        DataFrame with date dimension columns including:
        - date, fiscal_week, fiscal_month, fiscal_quarter, fiscal_year
        - holiday indicators
    """
    df = load_table("CALENDAR")

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    return df


@lru_cache(maxsize=1)
def load_ga_dma() -> pd.DataFrame:
    """
    Load GA_DMA table from HANA.

    Returns geographic/DMA mappings.

    Returns:
        DataFrame with DMA mapping columns
    """
    df = load_table("GA_DMA")

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    return df


# =============================================================================
# Filtered Loaders (Server-Side Filtering)
# =============================================================================

def load_model_b_filtered(
    profit_center_nbrs: Optional[List[int]] = None,
    channel: Optional[str] = None,
    origin_week_date: Optional[str] = None,
    max_horizon: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load MODEL_B with server-side filtering.

    Use this instead of load_model_b() when you need a subset of data.
    Filtering happens in HANA, reducing network transfer significantly
    for large tables.

    Args:
        profit_center_nbrs: Filter by specific store IDs
        channel: Filter by channel ('B&M' or 'WEB')
        origin_week_date: Filter by specific origin date (YYYY-MM-DD)
        max_horizon: Maximum horizon weeks to include

    Returns:
        DataFrame with filtered MODEL_B data (lowercase columns)
    """
    conditions = []

    if profit_center_nbrs:
        ids_str = ",".join(str(int(i)) for i in profit_center_nbrs)
        conditions.append(f"PROFIT_CENTER_NBR IN ({ids_str})")

    if channel:
        conditions.append(f"CHANNEL = '{channel.upper()}'")

    if origin_week_date:
        conditions.append(f"ORIGIN_WEEK_DATE = '{origin_week_date}'")

    if max_horizon is not None:
        conditions.append(f"HORIZON <= {int(max_horizon)}")

    where_clause = " AND ".join(conditions) if conditions else None

    df = load_table("MODEL_B", where_clause=where_clause)
    df = _normalize_model_b_columns(df)
    return df


def load_predictions_filtered(
    channel: str = "B&M",
    profit_center_nbrs: Optional[List[int]] = None,
    min_target_date: Optional[str] = None,
    max_horizon: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load predictions with server-side filtering.

    Args:
        channel: "B&M" or "WEB" to select which predictions table
        profit_center_nbrs: Filter by store IDs
        min_target_date: Minimum target_week_date (YYYY-MM-DD)
        max_horizon: Maximum horizon weeks to include
        columns: Optional list of columns to select (reduces memory usage)

    Returns:
        DataFrame with filtered predictions (lowercase columns)
    """
    table_name = "PREDICTIONS_BM" if channel.upper() == "B&M" else "PREDICTIONS_WEB"

    conditions = []

    if profit_center_nbrs:
        ids_str = ",".join(str(int(i)) for i in profit_center_nbrs)
        conditions.append(f"PROFIT_CENTER_NBR IN ({ids_str})")

    if min_target_date:
        conditions.append(f"TARGET_WEEK_DATE >= '{min_target_date}'")

    if max_horizon is not None:
        conditions.append(f"HORIZON <= {int(max_horizon)}")

    where_clause = " AND ".join(conditions) if conditions else None

    # Convert column names to uppercase for HANA query
    hana_columns = [col.upper() for col in columns] if columns else None

    df = load_table(table_name, columns=hana_columns, where_clause=where_clause)
    df.columns = [col.lower() for col in df.columns]
    return df


def get_unique_store_ids(
    channel: str = "B&M",
    min_target_date: Optional[str] = None,
    max_horizon: Optional[int] = None,
    dmas: Optional[List[str]] = None,
) -> List[int]:
    """
    Get list of unique store IDs from predictions table.

    This is a lightweight query that only fetches distinct store IDs,
    useful for batched processing to avoid loading all data at once.

    Args:
        channel: "B&M" or "WEB" to select which predictions table
        min_target_date: Minimum target_week_date filter (YYYY-MM-DD)
        max_horizon: Maximum horizon filter
        dmas: Optional list of DMA names to filter by

    Returns:
        List of unique profit_center_nbr values
    """
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')
    table_name = "PREDICTIONS_BM" if channel.upper() == "B&M" else "PREDICTIONS_WEB"

    conditions = []

    if min_target_date:
        conditions.append(f"TARGET_WEEK_DATE >= '{min_target_date}'")

    if max_horizon is not None:
        conditions.append(f"HORIZON <= {int(max_horizon)}")

    if dmas:
        dma_str = ", ".join(f"'{d}'" for d in dmas)
        conditions.append(f"DMA IN ({dma_str})")

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f'SELECT DISTINCT PROFIT_CENTER_NBR FROM "{schema}"."{table_name}"{where_clause} ORDER BY PROFIT_CENTER_NBR'

    hdf = cc.sql(query)
    df = hdf.collect()

    return df['PROFIT_CENTER_NBR'].tolist()


def load_store_master_filtered(
    profit_center_nbrs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load PROFIT_CENTER with optional store ID filter.

    Use this instead of load_store_master() when you only need specific stores.

    Args:
        profit_center_nbrs: Filter by specific store IDs

    Returns:
        DataFrame with filtered store master data
    """
    conditions = []

    if profit_center_nbrs:
        ids_str = ",".join(str(int(i)) for i in profit_center_nbrs)
        conditions.append(f"PROFIT_CENTER_NBR IN ({ids_str})")

    where_clause = " AND ".join(conditions) if conditions else None

    df = load_table("PROFIT_CENTER", where_clause=where_clause)

    # Apply same column mapping as load_store_master()
    column_mapping = {
        'PROFIT_CENTER_NBR': 'profit_center_nbr',
        'PROFIT_CENTER_NAME': 'profit_center_name',
        'STORE_ADDRESS': 'store_address',
        'CITY': 'city',
        'STATE': 'state',
        'MARKET___CITY': 'market_city',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'DATE_OPENED': 'date_opened',
        'DATE_CLOSED': 'date_closed',
        'OUTLET': 'is_outlet',
        'MERCHANDISING_SF': 'merchandising_sf',
        'STORE_DESIGN_SF': 'store_design_sf',
        'PROFORMA_SALES': 'proforma_annual_sales',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    if 'date_opened' in df.columns:
        df['date_opened'] = pd.to_datetime(df['date_opened'], errors='coerce')
    if 'date_closed' in df.columns:
        df['date_closed'] = pd.to_datetime(df['date_closed'], errors='coerce')

    return df


# =============================================================================
# Write Operations
# =============================================================================

logger = logging.getLogger(__name__)


def insert_predictions_bm(
    predictions_df: pd.DataFrame,
    batch_size: int = 1000,
    table_name: str = "PREDICTIONS_BM"
) -> int:
    """
    Insert predictions into PREDICTIONS_BM table in HANA.

    This function is used to populate missing prediction weeks (e.g., Nov-Dec 2025)
    that are not yet in the PREDICTIONS_BM table.

    Args:
        predictions_df: DataFrame with prediction columns matching PREDICTIONS_BM schema.
                        Column names should be lowercase (will be converted to uppercase for HANA).
        batch_size: Number of rows per INSERT batch (default 1000)
        table_name: Target table name (default "PREDICTIONS_BM")

    Returns:
        Number of rows inserted

    Raises:
        Exception: If insert fails (transaction is rolled back)

    Example:
        >>> df = pd.DataFrame({
        ...     'profit_center_nbr': [123, 123],
        ...     'channel': ['B&M', 'B&M'],
        ...     'origin_week_date': ['2025-10-27', '2025-10-27'],
        ...     'target_week_date': ['2025-11-03', '2025-11-10'],
        ...     'horizon': [1, 2],
        ...     'pred_sales_p50': [150000.0, 155000.0],
        ...     # ... other columns
        ... })
        >>> rows_inserted = insert_predictions_bm(df)
    """
    if predictions_df.empty:
        logger.warning("Empty DataFrame passed to insert_predictions_bm, no rows inserted")
        return 0

    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    # Get column names from DataFrame (convert to uppercase for HANA)
    columns = predictions_df.columns.tolist()
    column_str = ", ".join([f'"{col.upper()}"' for col in columns])
    placeholders = ", ".join(["?" for _ in columns])

    insert_sql = f'INSERT INTO "{schema}"."{table_name}" ({column_str}) VALUES ({placeholders})'

    logger.info(f"Inserting {len(predictions_df)} rows into {schema}.{table_name}")
    logger.debug(f"Insert SQL: {insert_sql}")

    total_inserted = 0
    cursor = cc.connection.cursor()

    try:
        for i in range(0, len(predictions_df), batch_size):
            batch = predictions_df.iloc[i:i + batch_size]

            # Convert DataFrame rows to list of tuples
            # Handle NaN values by converting to None
            rows = []
            for _, row in batch.iterrows():
                row_values = []
                for val in row.values:
                    if pd.isna(val):
                        row_values.append(None)
                    elif isinstance(val, pd.Timestamp):
                        row_values.append(val.strftime('%Y-%m-%d'))
                    else:
                        row_values.append(val)
                rows.append(tuple(row_values))

            cursor.executemany(insert_sql, rows)
            total_inserted += len(batch)

            batch_num = i // batch_size + 1
            total_batches = (len(predictions_df) + batch_size - 1) // batch_size
            logger.info(f"  Inserted batch {batch_num}/{total_batches}: {len(batch)} rows")

        cc.connection.commit()
        logger.info(f"Successfully inserted {total_inserted} rows into {table_name}")

    except Exception as e:
        logger.error(f"Error inserting into {table_name}: {e}")
        cc.connection.rollback()
        raise

    return total_inserted


def delete_predictions_by_date_range(
    table_name: str,
    min_target_date: str,
    max_target_date: Optional[str] = None,
    profit_center_nbrs: Optional[List[int]] = None,
) -> int:
    """
    Delete predictions within a date range (useful before re-inserting).

    Args:
        table_name: "PREDICTIONS_BM" or "PREDICTIONS_WEB"
        min_target_date: Minimum target_week_date (YYYY-MM-DD)
        max_target_date: Maximum target_week_date (YYYY-MM-DD, optional)
        profit_center_nbrs: Filter by store IDs (optional)

    Returns:
        Number of rows deleted
    """
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    conditions = [f"TARGET_WEEK_DATE >= '{min_target_date}'"]

    if max_target_date:
        conditions.append(f"TARGET_WEEK_DATE <= '{max_target_date}'")

    if profit_center_nbrs:
        ids_str = ",".join(str(int(i)) for i in profit_center_nbrs)
        conditions.append(f"PROFIT_CENTER_NBR IN ({ids_str})")

    where_clause = " AND ".join(conditions)
    delete_sql = f'DELETE FROM "{schema}"."{table_name}" WHERE {where_clause}'

    logger.info(f"Deleting rows from {table_name} where {where_clause}")

    cursor = cc.connection.cursor()
    try:
        cursor.execute(delete_sql)
        rows_deleted = cursor.rowcount
        cc.connection.commit()
        logger.info(f"Deleted {rows_deleted} rows from {table_name}")
        return rows_deleted
    except Exception as e:
        logger.error(f"Error deleting from {table_name}: {e}")
        cc.connection.rollback()
        raise


# =============================================================================
# Cache Management
# =============================================================================

def clear_cache() -> None:
    """Clear all cached data."""
    load_model_b.cache_clear()
    load_predictions_bm.cache_clear()
    load_predictions_web.cache_clear()
    load_store_master.cache_clear()
    load_awareness_consideration.cache_clear()
    load_yougov_dma_map.cache_clear()
    load_budget_data.cache_clear()
    load_calendar.cache_clear()
    load_ga_dma.cache_clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Connection
    "get_hana_connection",
    "close_connection",
    # Generic queries
    "query_to_dataframe",
    "load_table",
    # Table-specific loaders (cached, full tables)
    "load_model_b",
    "load_predictions_bm",
    "load_predictions_web",
    "load_store_master",
    "load_awareness_consideration",
    "load_yougov_dma_map",
    "load_budget_data",
    "load_calendar",
    "load_ga_dma",
    # Filtered loaders (server-side filtering, not cached)
    "load_model_b_filtered",
    "load_predictions_filtered",
    "load_store_master_filtered",
    "get_unique_store_ids",
    # Write operations
    "insert_predictions_bm",
    "delete_predictions_by_date_range",
    # Cache management
    "clear_cache",
    # HANA fiscal field parsing utilities
    "parse_fiscal_week",
    "parse_fiscal_quarter",
    "parse_fiscal_int",
]
