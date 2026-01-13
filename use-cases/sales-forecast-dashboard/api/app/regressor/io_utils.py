from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import holidays
from datetime import datetime, timedelta

from .paths import DATA_DIR


def load_store_master() -> pd.DataFrame:
    """Load PROFIT_CENTER from the master tables workbook.

    Returns a DataFrame with key columns standardized:
    - profit_center_nbr (int)
    - profit_center_name (str) - human-readable store name
    - store_address (str) - physical address of the store
    - market (str)
    - market_city (str)  # DMA-like label
    - ga_dma (str)
    - latitude (float)
    - longitude (float)
    - date_opened (datetime64[ns])
    - date_closed (datetime64[ns])
    - location_type (str)
    - dc_location (str)
    - merchandising_sf, store_design_sf (float)
    - is_outlet (bool) - outlet store indicator
    - proforma_annual_sales (float) - projected annual sales for new stores
    """
    path = DATA_DIR / "BDF Data Model Master Tables.xlsx"
    df = pd.read_excel(path, sheet_name="PROFIT_CENTER")

    df = df.rename(
        columns={
            "Profit Center #": "profit_center_nbr",
            "Profit Center Name": "profit_center_name",
            "Store Address": "store_address",
            "Market": "market",
            "Market - City": "market_city",
            "Google Analytics DMA": "ga_dma",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Date Opened": "date_opened",
            "Date Closed": "date_closed",
            "Location Type": "location_type",
            "DC Location": "dc_location",
            # Square footage fields used in screening
            "Merchandising SF": "merchandising_sf",
            "Store Design SF": "store_design_sf",
            # New fields
            "Outlet": "is_outlet",
            "Proforma Sales": "proforma_annual_sales",
        }
    )
    # Normalize key types
    df["profit_center_nbr"] = pd.to_numeric(df["profit_center_nbr"], errors="coerce").astype("Int64")

    # Coerce numeric SF fields if present
    for c in ("merchandising_sf", "store_design_sf"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert outlet to boolean
    if "is_outlet" in df.columns:
        # Handle various representations: 1/0, True/False, Yes/No, Y/N
        df["is_outlet"] = df["is_outlet"].fillna(0).astype(str).str.upper().isin(['1', 'TRUE', 'YES', 'Y'])

    # Convert proforma sales to float
    if "proforma_annual_sales" in df.columns:
        df["proforma_annual_sales"] = pd.to_numeric(df["proforma_annual_sales"], errors="coerce")

    return df


def load_market_region_map() -> pd.DataFrame:
    """Load MARKET sheet mapping Market -> Region."""
    path = DATA_DIR / "BDF Data Model Master Tables.xlsx"
    df = pd.read_excel(path, sheet_name="MARKET").rename(columns={"Market": "market", "Region": "region"})
    return df


def load_written_sales() -> pd.DataFrame:
    """Load Written Sales Data with cleaned column names/types.

    Returns columns:
    - profit_center_nbr (int)
    - channel (str)
    - fiscal_year, fiscal_month, fiscal_week (int)
    - fiscal_start_date_week (datetime64)
    - total_sales (float)
    - store_traffic (float, NaN if 'NULL')
    - order_qty, order_count (float) - order metrics
    - aur (float) - Average Unit Retail
    - employee_hours, unique_associates, avg_tenure_days (float) - staffing metrics
    - pct_white_glove, pct_threshold_delivery (float) - delivery mix
    - pct_omni_channel (float) - channel mix
    - pct_premium_product, pct_value_product (float) - product mix
    - pct_primary_financing, pct_secondary_financing, pct_tertiary_financing (float) - financing mix
    """
    path = DATA_DIR / "Written Sales Data.csv"
    # Use utf-8-sig to handle BOM
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "Profit Center Nbr": "profit_center_nbr",
            "Channel": "channel",
            "FiscalYear": "fiscal_year",
            "FiscalMonth": "fiscal_month",
            "FiscalWeek": "fiscal_week",
            "FiscalStartDateWeek": "fiscal_start_date_week",
            "Total Sales": "total_sales",
            "Store_Traffic": "store_traffic",
            # New columns
            "Order Qty": "order_qty",
            "Total Order Count": "order_count",
            "AUR": "aur",
            "EmployeeHours": "employee_hours",
            "Unique_Associates": "unique_associates",
            "Avg_Tenure_Days": "avg_tenure_days",
            "Percent_White_Gloves_Delivery": "pct_white_glove",
            "Percent_Threshold_Delivery": "pct_threshold_delivery",
            "Percent_Omni_Channel": "pct_omni_channel",
            "Percent_Value_Class_Best_Qty": "pct_premium_product",
            "Percent_Value_Class_Good_Qty": "pct_value_product",
            "Percent_Primary_Financing": "pct_primary_financing",
            "Percent_Secondary_Financing": "pct_secondary_financing",
            "Percent_Tertiary_Financing": "pct_tertiary_financing",
        }
    )
    # Standard type conversions
    df["profit_center_nbr"] = pd.to_numeric(df["profit_center_nbr"], errors="coerce").astype("Int64")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["fiscal_week"] = pd.to_numeric(df["fiscal_week"], errors="coerce").astype("Int64")
    df["fiscal_start_date_week"] = pd.to_datetime(df["fiscal_start_date_week"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "total_sales", "order_qty", "order_count", "aur",
        "employee_hours", "unique_associates", "avg_tenure_days",
        "pct_white_glove", "pct_threshold_delivery", "pct_omni_channel",
        "pct_premium_product", "pct_value_product",
        "pct_primary_financing", "pct_secondary_financing", "pct_tertiary_financing"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean Store_Traffic string 'NULL' to NaN
    if "store_traffic" in df.columns:
        df.loc[df["store_traffic"].astype(str).str.upper().eq("NULL"), "store_traffic"] = pd.NA
        df["store_traffic"] = pd.to_numeric(df["store_traffic"], errors="coerce")

    # Validate Percent_* columns: clip to [0, 100] (exclude negative values as instructed)
    pct_cols = [col for col in df.columns if col.startswith("pct_")]
    for col in pct_cols:
        if col in df.columns:
            # Log count of negative values before clipping (for monitoring)
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"Warning: {col} has {neg_count} negative values (will be clipped to 0)")

            # Clip to [0, 100]
            df[col] = df[col].clip(lower=0, upper=100)

    return df


def load_yougov_dma_map(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load YouGov DMA to Market/Market City mapping from master tables.

    Parameters
    ----------
    data_path : Optional[Path]
        Custom path to the master tables file. If None, uses default DATA_DIR.

    Returns columns:
    - yougov_dma (str)
    - region (str)
    - market (str)
    - market_city (str)
    """
    path = data_path if data_path else DATA_DIR / "BDF Data Model Master Tables.xlsx"
    df = pd.read_excel(path, sheet_name="YOUGOV_DMA_MAP")
    df = df.rename(
        columns={
            "YougovDMA": "yougov_dma",
            "Region": "region",
            "Market": "market",
            "Market City": "market_city",
        }
    )
    return df


def load_awareness_consideration(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load Awareness/Consideration weekly metrics by YouGov Market.

    Expected columns in the source file:
    - Company, Market, FiscalStartDateWeek, Awareness, Consideration

    Parameters
    ----------
    data_path : Optional[Path]
        Custom path to the awareness file. If None, uses default DATA_DIR.

    Returns normalized columns:
    - market (str)
    - week_start (datetime64)
    - awareness (float)
    - consideration (float)
    """
    path = data_path if data_path else DATA_DIR / "Awareness_Consideration_2022-2025.xlsx"
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "Market": "market",
            "FiscalStartDateWeek": "week_start",
            "Awareness": "awareness",
            "Consideration": "consideration",
        }
    )
    # Normalize types
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    for c in ("awareness", "consideration"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep only needed columns
    cols = [c for c in ["market", "week_start", "awareness", "consideration"] if c in df.columns]
    return df[cols]


def load_ecomm_traffic() -> pd.DataFrame:
    """Load Ecomm Traffic.csv with standardized columns.

    Returns columns:
    - profit_center_nbr (int)
    - market_city (str)
    - fiscal_start_date_week (datetime64)
    - merch_amt (float)
    - perc_sales (float)
    - unallocated_web_traffic (float)
    - allocated_web_traffic (float)
    """
    path = DATA_DIR / "Ecomm Traffic.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "Profit Center Nbr": "profit_center_nbr",
            "Market - City": "market_city",
            "FiscalStartDateWeek": "fiscal_start_date_week",
            "MerchAmt": "merch_amt",
            "PercSales": "perc_sales",
            "UnallocatedWebTraffic": "unallocated_web_traffic",
            "AllocatedWebTraffic": "allocated_web_traffic",
        }
    )
    df["profit_center_nbr"] = pd.to_numeric(df["profit_center_nbr"], errors="coerce").astype("Int64")
    df["fiscal_start_date_week"] = pd.to_datetime(df["fiscal_start_date_week"], errors="coerce")
    for c in ("merch_amt", "perc_sales", "unallocated_web_traffic", "allocated_web_traffic"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_competitor_counts() -> pd.DataFrame:
    """Load competitor store counts within 20 miles per store.

    Returns columns:
    - profit_center_nbr (int)
    - competitor_count_20mi (int)
    """
    path = DATA_DIR / "Competitor Store Data.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Profitcenter Nbr": "profit_center_nbr",
        "Competitive Stores within 20 Mile Radius": "competitor_count_20mi",
    })
    df["profit_center_nbr"] = pd.to_numeric(df["profit_center_nbr"], errors="coerce").astype("Int64")
    df["competitor_count_20mi"] = pd.to_numeric(df["competitor_count_20mi"], errors="coerce").fillna(0).astype(int)
    return df


def _clean_category_name(category_name: str) -> str:
    """Convert category names to valid column suffixes.

    Examples:
        "25 to 34" → "25_34"
        "$150k+" → "150k_plus"
        "Under $50k" → "under_50k"
        "Single Family Dwelling Unit" → "single_family_dwelling_unit"
    """
    if pd.isna(category_name) or category_name == "" or str(category_name).upper() == "NULL":
        return "unknown"

    # Convert to string and clean
    name = str(category_name).strip()

    # Handle special patterns
    name = name.replace("$", "")
    name = name.replace("+", "_plus")
    name = name.replace("%", "_pct")
    name = name.replace(" to ", "_")
    name = name.replace("-", "_")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = name.replace("&", "and")

    # Handle "Under X" pattern
    if name.lower().startswith("under "):
        name = "under_" + name[6:]

    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove multiple consecutive underscores
    while "__" in name:
        name = name.replace("__", "_")

    # Convert to lowercase and strip leading/trailing underscores
    name = name.lower().strip("_")

    return name


def load_demographics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load demographics data with static real estate features and time-varying CRM features.

    Demographics data contains two distinct types:
    1. Static real estate features (CV=0): Population, income, household metrics - constant over time
    2. Time-varying CRM features (CV>0.6): Customer age/income/household mix - varies weekly

    Returns:
        static_df: DataFrame with static features, one row per store
            Keys: profit_center_nbr
            Columns: population_20min, population_30min, median_income_20min,
                     total_households, drive_time_70pct, internal_stores_nearby, etc.

        crm_df: DataFrame with time-varying CRM customer mix features
            Keys: profit_center_nbr, channel_norm, week_start
            Columns: crm_age_25_34_pct, crm_income_150k_plus_pct,
                     crm_owner_renter_owner_pct, crm_children_y_pct,
                     crm_dwelling_single_family_dwelling_unit_pct, etc.
    """
    path = DATA_DIR / "Demographics (CRM + Real Estate).csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize column names and types
    df = df.rename(columns={
        "Profit Center Nbr": "profit_center_nbr",
        "Profit Center": "profit_center_nbr",  # Fallback for alternate name
        "Channel": "channel",
        "FiscalStartDateWeek": "week_start",
    })

    df["profit_center_nbr"] = pd.to_numeric(df["profit_center_nbr"], errors="coerce").astype("Int64")
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["channel"] = df["channel"].astype(str).str.upper().str.strip()

    # === STATIC FEATURES (Real Estate - CV=0) ===
    static_col_map = {
        'Pop20min': 'population_20min',
        'Pop30min': 'population_30min',
        'Pop40min': 'population_40min',
        'Median HH Income (20 Min)': 'median_income_20min',
        'Total HH': 'total_households',
        'Primary HH': 'primary_households',
        'Secondary HH': 'secondary_households',
        'Primary & Secondary HH': 'primary_secondary_hh',
        'Primary & Secondary HH (%)': 'primary_secondary_hh_pct',
        '70% TA Drive Time': 'drive_time_70pct',
        'Company X Store within 20 Miles': 'internal_stores_nearby'
    }

    # Extract static features - use latest value per store (they're constant, so any value works)
    static_cols = ['profit_center_nbr'] + [col for col in static_col_map.keys() if col in df.columns]
    static_df = df[static_cols].groupby('profit_center_nbr', as_index=False).last()
    static_df = static_df.rename(columns=static_col_map)

    # Convert static numeric columns
    for col in static_df.columns:
        if col != 'profit_center_nbr':
            static_df[col] = pd.to_numeric(static_df[col], errors="coerce")

    # === TIME-VARYING CRM FEATURES (Customer Mix - CV>0.6) ===
    # Parse categorical Primary/Secondary/Other structure into percentage columns

    # Define demographic dimensions and their column prefixes
    demographic_dimensions = {
        'AgeBand': 'crm_age',
        'IncomeBandGroup': 'crm_income',
        'OwnerRenter': 'crm_owner_renter',
        'PresenceofChildren': 'crm_children',
        'DwellingType': 'crm_dwelling',
        'gender': 'crm_gender',
        'HouseholdEducation': 'crm_education',
        'LengthofResidence': 'crm_residence',
        'MaritalStatus': 'crm_marital',
        'NumberOfChildren': 'crm_num_children',
    }

    # Extract base columns for CRM data
    crm_base_cols = ['profit_center_nbr', 'channel', 'week_start']
    crm_df = df[crm_base_cols].copy()

    # Process each demographic dimension
    for dimension, col_prefix in demographic_dimensions.items():
        # Identify columns for this dimension (Primary, Secondary, Other with IDs and Names)
        primary_col = f"{dimension}_Primary"
        primary_name_col = f"{dimension}_Primary_Name"
        secondary_col = f"{dimension}_Secondary"
        secondary_name_col = f"{dimension}_Secondary_Name"
        other_col = f"{dimension}_Other"
        other_name_col = f"{dimension}_Other_NAME"  # Note: inconsistent casing in CSV

        # Check if this dimension exists in the dataframe
        if primary_col not in df.columns:
            continue

        # For each row, collect category counts
        category_counts_list = []

        for _, row in df.iterrows():
            category_counts = {}

            # Extract Primary category
            if pd.notna(row.get(primary_name_col)):
                category = _clean_category_name(row[primary_name_col])
                count = pd.to_numeric(row.get(primary_col, 0), errors='coerce')
                if pd.notna(count) and count > 0:
                    category_counts[category] = category_counts.get(category, 0) + count

            # Extract Secondary category
            if pd.notna(row.get(secondary_name_col)):
                category = _clean_category_name(row[secondary_name_col])
                count = pd.to_numeric(row.get(secondary_col, 0), errors='coerce')
                if pd.notna(count) and count > 0:
                    category_counts[category] = category_counts.get(category, 0) + count

            # Extract Other category
            if pd.notna(row.get(other_name_col)):
                category = _clean_category_name(row[other_name_col])
                count = pd.to_numeric(row.get(other_col, 0), errors='coerce')
                if pd.notna(count) and count > 0:
                    category_counts[category] = category_counts.get(category, 0) + count

            category_counts_list.append(category_counts)

        # Convert counts to percentages
        # For each row, normalize counts so they sum to 100%
        all_categories = set()
        for counts in category_counts_list:
            all_categories.update(counts.keys())

        # Create percentage columns for each category
        for category in all_categories:
            col_name = f"{col_prefix}_{category}_pct"
            percentages = []

            for counts in category_counts_list:
                total = sum(counts.values())
                if total > 0 and category in counts:
                    pct = (counts[category] / total) * 100
                    percentages.append(pct)
                else:
                    percentages.append(0.0)

            crm_df[col_name] = percentages

    # Normalize channel values (B&M, Web → uppercase) and rename to channel_norm for merging
    crm_df['channel_norm'] = crm_df['channel'].astype(str).str.upper().str.strip()
    crm_df = crm_df.drop(columns=['channel'])

    # Remove duplicate rows if any
    crm_df = crm_df.drop_duplicates(subset=['profit_center_nbr', 'channel_norm', 'week_start'])

    return static_df, crm_df


def load_holiday_calendar() -> pd.DataFrame:
    """Generate holiday calendar with pre-holiday windows for major retail holidays.

    Creates fiscal week-level holiday indicators using US federal holidays.
    Major retail holidays get 3-week pre-holiday windows (t-3, t-2, t-1).
    Minor holidays get 1-week window only.

    Major retail holidays (3-week window):
    - Thanksgiving, Christmas, Memorial Day, Labor Day, July 4th

    Minor holidays (1-week window):
    - New Year's Day, MLK Day, Presidents Day, Easter

    Returns:
        DataFrame with columns:
        - week_start (datetime): Monday of fiscal week
        - is_holiday_week (bool): Week contains a holiday
        - is_pre_holiday_1wk (bool): 1 week before major holiday
        - is_pre_holiday_2wk (bool): 2 weeks before major holiday
        - is_pre_holiday_3wk (bool): 3 weeks before major holiday
        - weeks_to_holiday (int): Weeks until next major holiday (0 if holiday week, negative after)
        - holiday_name (str): Name of holiday (null if none)
        - holiday_type (str): 'major_retail' or 'minor'
    """
    # Generate US federal holidays for 2020-2026
    us_holidays = holidays.US(years=range(2020, 2027))

    # Define major retail holidays by name patterns
    major_patterns = ['Thanksgiving', 'Christmas', 'Memorial', 'Labor', 'Independence']
    minor_patterns = ["New Year", "Martin Luther King", "Washington's Birthday", "Presidents"]

    # Build holiday dataframe
    holiday_records = []
    for date, name in us_holidays.items():
        # Determine holiday type
        is_major = any(pattern in name for pattern in major_patterns)
        is_minor = any(pattern in name for pattern in minor_patterns)

        if is_major:
            hol_type = 'major_retail'
        elif is_minor:
            hol_type = 'minor'
        else:
            hol_type = 'minor'  # Default other federal holidays to minor

        holiday_records.append({
            'date': pd.to_datetime(date),
            'holiday_name': name,
            'holiday_type': hol_type
        })

    # Explicitly add Black Friday (day after Thanksgiving) as a major retail holiday
    bf_records = []
    for rec in holiday_records:
        if 'Thanksgiving' in rec['holiday_name']:
            bf_date = rec['date'] + timedelta(days=1)
            # Avoid duplicates if already present
            if not any((r['date'] == bf_date) and ('Black Friday' in r['holiday_name']) for r in holiday_records):
                bf_records.append({
                    'date': bf_date,
                    'holiday_name': 'Black Friday',
                    'holiday_type': 'major_retail'
                })
    holiday_records.extend(bf_records)

    hol_df = pd.DataFrame(holiday_records)

    # Create weekly calendar (Mondays from 2020-01-06 to 2026-12-28)
    start_date = pd.to_datetime('2020-01-06')  # First Monday of 2020
    end_date = pd.to_datetime('2026-12-28')  # Last Monday of 2026
    weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')

    calendar_df = pd.DataFrame({'week_start': weeks})

    # For each week, determine if it contains a holiday and capture the actual holiday date as anchor
    def week_contains_holiday(week_start, hol_df_inner):
        week_end = week_start + timedelta(days=6)
        week_holidays = hol_df_inner[
            (hol_df_inner['date'] >= week_start) &
            (hol_df_inner['date'] <= week_end)
        ]
        if len(week_holidays) > 0:
            # If Black Friday exists in this week, return it explicitly (takes priority)
            bf_mask = week_holidays['holiday_name'].str.contains('Black Friday', na=False) | \
                      week_holidays['holiday_name'].str.contains('Day after Thanksgiving', na=False)
            if bf_mask.any():
                bf_row = week_holidays[bf_mask].iloc[0]
                return 'Black Friday', 'major_retail', bf_row['date']

            # If multiple holidays in one week, prioritize major
            major_hols = week_holidays[week_holidays['holiday_type'] == 'major_retail']
            if len(major_hols) > 0:
                row = major_hols.iloc[0]
                return row['holiday_name'], row['holiday_type'], row['date']
            else:
                row = week_holidays.iloc[0]
                return row['holiday_name'], row['holiday_type'], row['date']
        return None, None, None

    calendar_df[['holiday_name', 'holiday_type', 'holiday_anchor_date']] = calendar_df['week_start'].apply(
        lambda w: pd.Series(week_contains_holiday(w, hol_df))
    )

    calendar_df['is_holiday_week'] = calendar_df['holiday_name'].notna()

    # Create pre-holiday windows for MAJOR holidays only
    # Get weeks with major holidays
    major_weeks = calendar_df[
        (calendar_df['holiday_type'] == 'major_retail')
    ]['week_start'].values

    # Initialize pre-holiday columns
    calendar_df['is_pre_holiday_1wk'] = False
    calendar_df['is_pre_holiday_2wk'] = False
    calendar_df['is_pre_holiday_3wk'] = False

    # For each major holiday week, mark the 3 weeks before
    for major_week in major_weeks:
        major_week_dt = pd.to_datetime(major_week)

        # 1 week before
        week_1_before = major_week_dt - timedelta(weeks=1)
        calendar_df.loc[calendar_df['week_start'] == week_1_before, 'is_pre_holiday_1wk'] = True

        # 2 weeks before
        week_2_before = major_week_dt - timedelta(weeks=2)
        calendar_df.loc[calendar_df['week_start'] == week_2_before, 'is_pre_holiday_2wk'] = True

        # 3 weeks before
        week_3_before = major_week_dt - timedelta(weeks=3)
        calendar_df.loc[calendar_df['week_start'] == week_3_before, 'is_pre_holiday_3wk'] = True

    # Calculate weeks_to_holiday (distance to next major holiday)
    major_week_list = sorted(major_weeks)

    def weeks_until_next_major(week_start, major_weeks_list):
        week_start_dt = pd.to_datetime(week_start)
        # Find next major holiday after current week
        future_holidays = [h for h in major_weeks_list if pd.to_datetime(h) >= week_start_dt]
        if len(future_holidays) > 0:
            next_holiday = pd.to_datetime(future_holidays[0])
            weeks_diff = (next_holiday - week_start_dt).days // 7
            return int(weeks_diff)
        else:
            # No future holidays in range, return NaN or large number
            return np.nan

    calendar_df['weeks_to_holiday'] = calendar_df['week_start'].apply(
        lambda w: weeks_until_next_major(w, major_week_list)
    )

    # Convert boolean columns to int for compatibility
    calendar_df['is_holiday_week'] = calendar_df['is_holiday_week'].astype(int)
    calendar_df['is_pre_holiday_1wk'] = calendar_df['is_pre_holiday_1wk'].astype(int)
    calendar_df['is_pre_holiday_2wk'] = calendar_df['is_pre_holiday_2wk'].astype(int)
    calendar_df['is_pre_holiday_3wk'] = calendar_df['is_pre_holiday_3wk'].astype(int)
    calendar_df['weeks_to_holiday'] = calendar_df['weeks_to_holiday'].astype('Int64')

    return calendar_df


def load_competitor_density() -> pd.DataFrame:
    """Compute internal store density metrics (distances to other company stores).

    Uses geo.py::compute_store_neighbors() to calculate pairwise distances
    between all internal stores.

    Returns:
        DataFrame with columns:
        - profit_center_nbr (int): Store ID
        - min_distance_to_internal (float): Miles to nearest internal store
        - internal_count_10mi (int): Count of internal stores within 10 miles
        - internal_count_20mi (int): Count of internal stores within 20 miles
        - internal_count_30mi (int): Count of internal stores within 30 miles
    """
    # Runtime import to avoid circular dependency (geo.py imports from io_utils)
    from .geo import compute_store_neighbors

    _, summary = compute_store_neighbors()

    # Select and rename columns
    result = summary[['profit_center_nbr', 'min_distance_miles', 'count_10mi', 'count_20mi', 'count_30mi']].copy()
    result = result.rename(columns={
        'min_distance_miles': 'min_distance_to_internal',
        'count_10mi': 'internal_count_10mi',
        'count_20mi': 'internal_count_20mi',
        'count_30mi': 'internal_count_30mi',
    })

    # Convert counts to int, filling NaN with 0
    for col in ['internal_count_10mi', 'internal_count_20mi', 'internal_count_30mi']:
        result[col] = result[col].fillna(0).astype(int)

    return result


# =============================================================================
# Marketing Budget Data
# =============================================================================

# Month name to number mapping for budget data
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Special cases for extracting market_city from budget DMA names
# These map Budget DMA -> market_city value that matches YOUGOV_DMA_MAP
BUDGET_DMA_TO_MARKET_CITY = {
    # Multi-word cities
    'Des-Moines-Ames, IA': 'Des Moines',
    'Fort Wayne, IN': 'Fort Wayne',
    'Grand Rapids, MI': 'Grand Rapids',
    'Green Bay-Appleton, WI': 'Green Bay',
    'Kansas City, MO': 'Kansas City',
    'Las Vegas, NV': 'Las Vegas',
    'Los Angeles, CA': 'Los Angeles',
    'New York, NY': 'New York',
    'Palm Springs, CA': 'Palm Springs',
    'San Diego, CA': 'San Diego',
    'South Bend, IN': 'South Bend',
    'Wilkes Barre, PA': 'Wilkes Barre',
    # Cities with different formats in YOUGOV_DMA_MAP
    'Boston, MA': 'Boston/NH',
    'Hartford & New Haven, CT': 'HARTFORD/NEW HAVEN',
    'St. Louis, MO': 'St. Louis',
    'Washington, DC': 'Washington DC',
    'Norfolk, VA': 'Norfolk / Newport News',
    'Davenport, IA': 'Davenport - Rock Island - Moline',
    'Greensboro, NC': 'GREENSBORO-HIGH POINT-WINSTON SALEM',
    # Hyphenated/combined DMAs
    'Fresno-Visalia, CA': 'Fresno',
    'Portland - Auburn, ME': 'Portland',
    'Springfield-Holyoke, MA': 'Springfield',
    # Skip
    'National': None,
}


def _extract_market_city(budget_dma: str) -> Optional[str]:
    """
    Extract market_city from budget DMA format.

    Examples:
        "Albany, NY" -> "Albany"
        "Des-Moines-Ames, IA" -> "Des Moines" (via lookup)
        "Hartford & New Haven, CT" -> "HARTFORD/NEW HAVEN" (via lookup)
    """
    if pd.isna(budget_dma):
        return None

    # Check special cases first
    if budget_dma in BUDGET_DMA_TO_MARKET_CITY:
        return BUDGET_DMA_TO_MARKET_CITY[budget_dma]

    # Default: extract first word before comma
    city = budget_dma.split(',')[0].strip()

    # Handle hyphenated names (take first part)
    if '-' in city and city not in ['Wilkes-Barre']:
        city = city.split('-')[0].strip()

    # Handle ampersand names (take first part)
    if '&' in city:
        city = city.split('&')[0].strip()

    return city


def load_budget_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load marketing budget data from budget_marketing.xlsx.

    Combines 2024 and 2025 sheets, maps Budget DMA to market_city for integration
    with YOUGOV_DMA_MAP.

    Parameters
    ----------
    data_path : Optional[Path]
        Custom path to the budget file. If None, uses default DATA_DIR.

    Returns:
        DataFrame with columns:
        - dma_budget (str): Original DMA name from budget file
        - market_city (str): Extracted/mapped city name
        - year (int): Fiscal year
        - month (int): Month number (1-12)
        - budget (float): Monthly budget in dollars
    """
    path = data_path if data_path else DATA_DIR / "budget_marketing.xlsx"
    dfs = []

    for year in [2024, 2025]:
        sheet_name = str(year)
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=2)

        dma_col = df.columns[0]

        df = df[df[dma_col].notna()]
        df = df[df[dma_col] != 'Grand Total']

        month_cols = [col for col in df.columns if col in MONTH_MAP]

        df_long = df[[dma_col] + month_cols].melt(
            id_vars=[dma_col],
            var_name='month_name',
            value_name='budget'
        )

        df_long['year'] = year
        df_long['month'] = df_long['month_name'].map(MONTH_MAP)
        df_long = df_long.rename(columns={dma_col: 'dma_budget'})

        # Extract market_city from budget DMA
        df_long['market_city'] = df_long['dma_budget'].apply(_extract_market_city)

        dfs.append(df_long[['dma_budget', 'market_city', 'year', 'month', 'budget']])

    result = pd.concat(dfs, ignore_index=True)
    result['budget'] = pd.to_numeric(result['budget'], errors='coerce').fillna(0)

    return result
