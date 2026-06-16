#!/usr/bin/env python3
"""
Verify PREDICTIONS_WEB table existence and compare schema with PREDICTIONS_BM.

Run from the api directory:
    cd sales-forecast-dashboard/api
    source venv/bin/activate
    python -m app.scripts.verify_predictions_web

To export PREDICTIONS_WEB to CSV:
    python -m app.scripts.verify_predictions_web --export
    python -m app.scripts.verify_predictions_web --export --limit 10000
    python -m app.scripts.verify_predictions_web --export --store 202
"""

import argparse
import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, api_dir)

from dotenv import load_dotenv
load_dotenv()

from app.agent.hana_loader import get_hana_connection, close_connection


def get_table_schema(table_name: str) -> list:
    """Get column names and types for a table."""
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, IS_NULLABLE
        FROM SYS.TABLE_COLUMNS
        WHERE SCHEMA_NAME = '{schema}' AND TABLE_NAME = '{table_name}'
        ORDER BY POSITION
    """

    try:
        hdf = cc.sql(query)
        return hdf.collect()
    except Exception as e:
        print(f"Error querying schema for {table_name}: {e}")
        return None


def get_table_row_count(table_name: str) -> int:
    """Get row count for a table."""
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    query = f'SELECT COUNT(*) as cnt FROM "{schema}"."{table_name}"'

    try:
        hdf = cc.sql(query)
        result = hdf.collect()
        return result['CNT'].iloc[0]
    except Exception as e:
        print(f"Error counting rows in {table_name}: {e}")
        return -1


def get_sample_rows(table_name: str, limit: int = 3) -> None:
    """Get sample rows from a table."""
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    query = f'SELECT * FROM "{schema}"."{table_name}" LIMIT {limit}'

    try:
        hdf = cc.sql(query)
        return hdf.collect()
    except Exception as e:
        print(f"Error fetching sample from {table_name}: {e}")
        return None


def export_to_csv(table_name: str, output_path: str, limit: int = None, store_id: int = None):
    """Export table to CSV with optional filters."""
    cc = get_hana_connection()
    schema = os.getenv('HANA_SCHEMA', 'AICOE')

    conditions = []
    if store_id:
        conditions.append(f"PROFIT_CENTER_NBR = {store_id}")

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f" LIMIT {limit}" if limit else ""

    query = f'SELECT * FROM "{schema}"."{table_name}"{where_clause} ORDER BY PROFIT_CENTER_NBR, ORIGIN_WEEK_DATE, TARGET_WEEK_DATE, HORIZON{limit_clause}'

    print(f"Executing query: {query[:200]}...")

    try:
        hdf = cc.sql(query)
        df = hdf.collect()
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} rows to {output_path}")
        return len(df)
    except Exception as e:
        print(f"Error exporting {table_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Verify PREDICTIONS_WEB table and optionally export to CSV")
    parser.add_argument("--export", action="store_true", help="Export PREDICTIONS_WEB to CSV")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to export (default: all)")
    parser.add_argument("--store", type=int, default=None, help="Filter by store ID (PROFIT_CENTER_NBR)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: output/predictions_web.csv)")
    parser.add_argument("--table", type=str, default="PREDICTIONS_WEB", help="Table to export (default: PREDICTIONS_WEB)")
    args = parser.parse_args()

    # Handle export mode
    if args.export:
        output_dir = os.path.join(api_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        if args.output:
            output_path = args.output
        else:
            filename = f"{args.table.lower()}"
            if args.store:
                filename += f"_store_{args.store}"
            if args.limit:
                filename += f"_limit_{args.limit}"
            filename += ".csv"
            output_path = os.path.join(output_dir, filename)

        print(f"Exporting {args.table} to CSV...")
        if args.store:
            print(f"  Filter: store_id = {args.store}")
        if args.limit:
            print(f"  Limit: {args.limit} rows")
        print(f"  Output: {output_path}")
        print()

        try:
            export_to_csv(args.table, output_path, limit=args.limit, store_id=args.store)
        finally:
            close_connection()
        return

    print("=" * 70)
    print("PREDICTIONS_WEB vs PREDICTIONS_BM Table Comparison")
    print("=" * 70)

    try:
        # Get schemas
        print("\n[1] Fetching table schemas...")

        bm_schema = get_table_schema("PREDICTIONS_BM")
        web_schema = get_table_schema("PREDICTIONS_WEB")

        # Check if tables exist
        print("\n[2] Table existence check:")
        print(f"    PREDICTIONS_BM:  {'EXISTS' if bm_schema is not None and len(bm_schema) > 0 else 'NOT FOUND'}")
        print(f"    PREDICTIONS_WEB: {'EXISTS' if web_schema is not None and len(web_schema) > 0 else 'NOT FOUND'}")

        if web_schema is None or len(web_schema) == 0:
            print("\n    PREDICTIONS_WEB table does not exist in HANA.")
            return

        # Get row counts
        print("\n[3] Row counts:")
        bm_count = get_table_row_count("PREDICTIONS_BM")
        web_count = get_table_row_count("PREDICTIONS_WEB")
        print(f"    PREDICTIONS_BM:  {bm_count:,} rows")
        print(f"    PREDICTIONS_WEB: {web_count:,} rows")

        # Compare columns
        print("\n[4] Column comparison:")
        bm_cols = set(bm_schema['COLUMN_NAME'].tolist()) if bm_schema is not None else set()
        web_cols = set(web_schema['COLUMN_NAME'].tolist()) if web_schema is not None else set()

        common = bm_cols & web_cols
        only_bm = bm_cols - web_cols
        only_web = web_cols - bm_cols

        print(f"    Common columns:       {len(common)}")
        print(f"    Only in PREDICTIONS_BM:  {len(only_bm)}")
        print(f"    Only in PREDICTIONS_WEB: {len(only_web)}")

        if only_bm:
            print(f"\n    Columns ONLY in PREDICTIONS_BM:")
            for col in sorted(only_bm):
                print(f"      - {col}")

        if only_web:
            print(f"\n    Columns ONLY in PREDICTIONS_WEB:")
            for col in sorted(only_web):
                print(f"      - {col}")

        # Show full column list for both
        print("\n[5] Full column list (PREDICTIONS_BM):")
        if bm_schema is not None:
            for _, row in bm_schema.iterrows():
                print(f"    {row['COLUMN_NAME']:<40} {row['DATA_TYPE_NAME']}")

        print("\n[6] Full column list (PREDICTIONS_WEB):")
        if web_schema is not None:
            for _, row in web_schema.iterrows():
                print(f"    {row['COLUMN_NAME']:<40} {row['DATA_TYPE_NAME']}")

        # Sample data from WEB if it has rows
        if web_count > 0:
            print("\n[7] Sample rows from PREDICTIONS_WEB:")
            sample = get_sample_rows("PREDICTIONS_WEB", 3)
            if sample is not None:
                # Show just key columns
                key_cols = ['PROFIT_CENTER_NBR', 'CHANNEL', 'DMA', 'ORIGIN_WEEK_DATE',
                           'TARGET_WEEK_DATE', 'HORIZON', 'PRED_SALES_P50']
                display_cols = [c for c in key_cols if c in sample.columns]
                print(sample[display_cols].to_string(index=False))

        print("\n" + "=" * 70)
        print("Verification complete.")
        print("=" * 70)

    finally:
        close_connection()


if __name__ == "__main__":
    main()
