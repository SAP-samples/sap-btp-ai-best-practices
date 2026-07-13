"""
Validation for sister-store mapping (store-level fallback for short histories).

Checks:
    - Identifies stores with <min_years history
    - Maps each to nearest store, preferring same DMA then same region
    - Distances are finite and reasonable
    - Coverage: all low-history stores get a sister

Usage:
    python -m forecasting.regressor.validation.validate_sister_store
"""

import numpy as np
import pandas as pd

from app.regressor.sister_store import build_sister_store_map
from app.regressor.io_utils import load_written_sales, load_store_master, load_market_region_map


def main(min_years: int = 3):
    print("\n" + "=" * 80)
    print("SISTER STORE VALIDATION")
    print("=" * 80)

    # Build mapping
    sister_map = build_sister_store_map(min_years=min_years)

    if sister_map.empty:
        print("No stores require sister fallback (all meet history threshold).")
        return

    print(f"Stores requiring sister fallback: {len(sister_map)}")
    print("\nSample mappings:")
    print(sister_map.head(10).to_string(index=False))

    # Validate distances
    distances = sister_map["distance_miles"]
    print("\nDistance statistics:")
    print(f"  Mean: {distances.mean():.1f} mi")
    print(f"  Max:  {distances.max():.1f} mi")
    if np.isinf(distances).any() or distances.isna().any():
        print("✗ Invalid distances detected (inf/NaN)")
    else:
        print("✓ Distances finite")

    if distances.max() > 1000:
        print("⚠ Some sister distances exceed 1000 miles")

    # Coverage: ensure all low-history stores have mapping
    hist = load_written_sales()[["profit_center_nbr", "fiscal_year"]].dropna()
    years = hist.groupby("profit_center_nbr")["fiscal_year"].nunique().rename("years_of_history")
    low_hist = years[years < min_years].index
    mapped = sister_map["profit_center_nbr"].unique()
    missing = set(low_hist) - set(mapped)
    if missing:
        print(f"✗ Missing mappings for {len(missing)} low-history stores: {sorted(list(missing))[:5]}...")
    else:
        print("✓ All low-history stores have a sister mapping")

    # Preference stats
    same_dma_pct = sister_map["same_market_city"].mean() * 100
    same_region_pct = sister_map["same_region"].mean() * 100
    print(f"\nPreference adherence:")
    print(f"  Same DMA:    {same_dma_pct:.1f}%")
    print(f"  Same region: {same_region_pct:.1f}%")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
