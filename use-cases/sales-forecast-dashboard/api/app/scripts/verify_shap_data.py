"""
Verify SHAP data for store 218, July 21, 2025 vs 2024.

This script queries HANA directly to compare:
1. Raw predictions for both years
2. TOP_FEATURES_PRED_LOG_SALES (raw SHAP string)
3. Origin week and horizon values
4. Business feature values
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Direct HANA connection
from hana_ml import ConnectionContext


def get_connection():
    """Create HANA connection from environment variables."""
    return ConnectionContext(
        address=os.getenv('hana_address'),
        port=int(os.getenv('hana_port', 443)),
        user=os.getenv('hana_user'),
        password=os.getenv('hana_password'),
        encrypt=os.getenv('hana_encrypt', 'true').lower() == 'true'
    )


def query_to_dataframe(query: str) -> pd.DataFrame:
    """Execute query and return DataFrame."""
    conn = get_connection()
    try:
        df = conn.sql(query).collect()
        return df
    finally:
        conn.close()


def parse_shap_string(shap_str: str) -> dict:
    """Parse SHAP string into dict of feature -> (value, impact)."""
    if pd.isna(shap_str) or shap_str == '' or not isinstance(shap_str, str):
        return {}

    features = {}
    for part in shap_str.split('; '):
        part = part.strip()
        if ':' in part and '=' in part:
            try:
                feature_val, impact = part.rsplit(':', 1)
                feature_name, value = feature_val.split('=', 1)
                features[feature_name.strip()] = {
                    'value': value.strip(),
                    'impact': float(impact)
                }
            except (ValueError, IndexError):
                continue
    return features


def query_store_data(store_id: int, target_date: str, channel: str = "B&M"):
    """Query HANA for store predictions on a specific date."""
    schema = os.getenv('HANA_SCHEMA', 'AICOE')
    table_name = "PREDICTIONS_BM" if channel == "B&M" else "PREDICTIONS_WEB"

    query = f"""
        SELECT
            PROFIT_CENTER_NBR,
            ORIGIN_WEEK_DATE,
            TARGET_WEEK_DATE,
            HORIZON,
            CHANNEL,
            PRED_SALES_P50,
            PRED_SALES_P90,
            TOP_FEATURES_PRED_LOG_SALES
        FROM "{schema}"."{table_name}"
        WHERE PROFIT_CENTER_NBR = {store_id}
          AND TARGET_WEEK_DATE = '{target_date}'
        ORDER BY ORIGIN_WEEK_DATE
    """

    df = query_to_dataframe(query)
    df.columns = [col.lower() for col in df.columns]
    return df


def main():
    store_id = 218

    # July 21, 2025 is ISO week 30
    # Find the equivalent week in 2024 (ISO week 30 = July 22, 2024)
    target_2025 = "2025-07-21"
    target_2024 = "2024-07-22"  # ISO week 30 in 2024

    print("=" * 80)
    print(f"SHAP Data Verification for Store {store_id}")
    print("=" * 80)

    # Query 2025 data
    print(f"\n--- 2025 Data (Target: {target_2025}) ---")
    df_2025 = query_store_data(store_id, target_2025, "B&M")

    if df_2025.empty:
        print("No 2025 data found!")
    else:
        print(f"Found {len(df_2025)} rows")
        for idx, row in df_2025.iterrows():
            print(f"\nRow {idx + 1}:")
            print(f"  Origin Week: {row['origin_week_date']}")
            print(f"  Target Week: {row['target_week_date']}")
            print(f"  Horizon: {row['horizon']}")
            print(f"  Pred Sales P50: ${row['pred_sales_p50']:,.2f}")
            print(f"  Pred Sales P90: ${row['pred_sales_p90']:,.2f}")

            shap_str = row.get('top_features_pred_log_sales', '')
            print(f"  Raw SHAP String: {shap_str[:200]}..." if len(str(shap_str)) > 200 else f"  Raw SHAP String: {shap_str}")

            features = parse_shap_string(shap_str)
            print(f"  Parsed Features ({len(features)}):")
            for feat, data in sorted(features.items(), key=lambda x: abs(x[1]['impact']), reverse=True):
                print(f"    - {feat} = {data['value']} (SHAP: {data['impact']:+.4f})")

    # Query 2024 data
    print(f"\n--- 2024 Data (Target: {target_2024}, ISO Week 30) ---")
    df_2024 = query_store_data(store_id, target_2024, "B&M")

    if df_2024.empty:
        # Try different dates around ISO week 30
        for alt_date in ["2024-07-21", "2024-07-22", "2024-07-23", "2024-07-15"]:
            df_2024 = query_store_data(store_id, alt_date, "B&M")
            if not df_2024.empty:
                print(f"Found data for alternate date: {alt_date}")
                break

    if df_2024.empty:
        print("No 2024 data found!")
    else:
        print(f"Found {len(df_2024)} rows")
        for idx, row in df_2024.iterrows():
            print(f"\nRow {idx + 1}:")
            print(f"  Origin Week: {row['origin_week_date']}")
            print(f"  Target Week: {row['target_week_date']}")
            print(f"  Horizon: {row['horizon']}")
            print(f"  Pred Sales P50: ${row['pred_sales_p50']:,.2f}")
            print(f"  Pred Sales P90: ${row['pred_sales_p90']:,.2f}")

            shap_str = row.get('top_features_pred_log_sales', '')
            print(f"  Raw SHAP String: {shap_str[:200]}..." if len(str(shap_str)) > 200 else f"  Raw SHAP String: {shap_str}")

            features = parse_shap_string(shap_str)
            print(f"  Parsed Features ({len(features)}):")
            for feat, data in sorted(features.items(), key=lambda x: abs(x[1]['impact']), reverse=True):
                print(f"    - {feat} = {data['value']} (SHAP: {data['impact']:+.4f})")

    # Compare features if we have both
    if not df_2025.empty and not df_2024.empty:
        print("\n" + "=" * 80)
        print("FEATURE COMPARISON (2025 vs 2024)")
        print("=" * 80)

        # Get the row with earliest origin (first forecast of year)
        row_2025 = df_2025.sort_values('origin_week_date').iloc[0]
        row_2024 = df_2024.sort_values('origin_week_date').iloc[0]

        feat_2025 = parse_shap_string(row_2025.get('top_features_pred_log_sales', ''))
        feat_2024 = parse_shap_string(row_2024.get('top_features_pred_log_sales', ''))

        all_features = set(feat_2025.keys()) | set(feat_2024.keys())

        print(f"\n2025 Origin: {row_2025['origin_week_date']}, Horizon: {row_2025['horizon']}")
        print(f"2024 Origin: {row_2024['origin_week_date']}, Horizon: {row_2024['horizon']}")
        print(f"\nSales: 2025=${row_2025['pred_sales_p50']:,.2f}, 2024=${row_2024['pred_sales_p50']:,.2f}")
        print(f"YoY Change: ${row_2025['pred_sales_p50'] - row_2024['pred_sales_p50']:,.2f} ({100*(row_2025['pred_sales_p50']/row_2024['pred_sales_p50'] - 1):.1f}%)")

        print(f"\n{'Feature':<45} {'2025 Value':<15} {'2024 Value':<15} {'Delta SHAP':<12}")
        print("-" * 90)

        for feat in sorted(all_features):
            val_2025 = feat_2025.get(feat, {}).get('value', 'N/A')
            val_2024 = feat_2024.get(feat, {}).get('value', 'N/A')
            shap_2025 = feat_2025.get(feat, {}).get('impact', 0)
            shap_2024 = feat_2024.get(feat, {}).get('impact', 0)
            delta = shap_2025 - shap_2024

            print(f"{feat:<45} {val_2025:<15} {val_2024:<15} {delta:+.4f}")


if __name__ == "__main__":
    main()
