"""
Check for negative effective costs in the vendor ranking data.
"""

import pandas as pd
import sys

def check_negative_costs(ranking_file):
    """Check for negative effective costs in ranking data."""
    
    print("Checking for negative effective costs...")
    
    # Load ranking data
    df = pd.read_csv(ranking_file)
    
    # Check for negative costs
    negative_costs = df[df['EffectiveCostPerUnit_USD'] < 0]
    
    if not negative_costs.empty:
        print(f"\nWARNING: Found {len(negative_costs)} items with NEGATIVE effective costs!")
        print("\nFirst 10 examples:")
        print(negative_costs[['LIFNR', 'NAME1', 'MAKTX', 'EffectiveCostPerUnit_USD']].head(10))
        
        # Group by vendor to see which vendors have negative costs
        vendors_with_neg = negative_costs.groupby(['LIFNR', 'NAME1']).agg({
            'EffectiveCostPerUnit_USD': ['count', 'min', 'mean'],
            'MAKTX': 'count'
        })
        vendors_with_neg.columns = ['neg_cost_count', 'min_cost', 'avg_neg_cost', 'items_count']
        vendors_with_neg = vendors_with_neg.sort_values('min_cost')
        
        print(f"\nVendors with negative costs (top 10 by most negative):")
        print(vendors_with_neg.head(10))
        
        # Check the components that lead to negative costs
        cost_cols = [col for col in df.columns if col.startswith('cost_')]
        if cost_cols:
            print("\nCost component analysis for items with most negative costs:")
            most_negative = negative_costs.nsmallest(5, 'EffectiveCostPerUnit_USD')
            for idx, row in most_negative.iterrows():
                print(f"\n{row['MAKTX']} (Vendor: {row['NAME1']}):")
                print(f"  Effective Cost: ${row['EffectiveCostPerUnit_USD']:.2f}")
                print(f"  Base Price: ${row.get('AvgUnitPriceUSD_raw', 0):.2f}")
                for col in cost_cols:
                    if col in row and row[col] != 0:
                        print(f"  {col}: ${row[col]:.2f}")
    else:
        print("No negative effective costs found.")
    
    # Also check for zero or very small positive costs
    small_costs = df[(df['EffectiveCostPerUnit_USD'] >= 0) & (df['EffectiveCostPerUnit_USD'] < 0.01)]
    if not small_costs.empty:
        print(f"\nNote: Found {len(small_costs)} items with very small costs (< $0.01)")
    
    # Summary statistics
    print("\nEffective Cost Statistics:")
    print(f"  Min: ${df['EffectiveCostPerUnit_USD'].min():.2f}")
    print(f"  Max: ${df['EffectiveCostPerUnit_USD'].max():.2f}")
    print(f"  Mean: ${df['EffectiveCostPerUnit_USD'].mean():.2f}")
    print(f"  Median: ${df['EffectiveCostPerUnit_USD'].median():.2f}")
    print(f"  Std Dev: ${df['EffectiveCostPerUnit_USD'].std():.2f}")
    
    # Check if this could cause optimization issues
    if (df['EffectiveCostPerUnit_USD'] < 0).any():
        print("\n⚠️  WARNING: Negative costs in a minimization problem can lead to:")
        print("   - Unbounded solutions (solver tries to allocate infinite quantity)")
        print("   - Numerical instability")
        print("   - Infeasibility if combined with capacity constraints")
        print("\nRECOMMENDATION: Review cost calculation logic or add non-negativity constraints.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ranking_file = sys.argv[1]
    else:
        ranking_file = "tables/vendor_maktx_ranking_tariff_values.csv"
    
    check_negative_costs(ranking_file)