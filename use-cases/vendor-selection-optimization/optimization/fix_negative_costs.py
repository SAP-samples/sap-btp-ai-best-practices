#!/usr/bin/env python3
"""
Fix negative costs in vendor ranking data to resolve optimization infeasibility.

This script addresses two main issues:
1. Extreme negative values in cost_Impact_PriceTrend
2. Extreme positive values in cost_Inefficiency_InFull

The fix involves:
- Capping extreme price trend slopes before cost calculation
- Adjusting inefficiency penalties to be more reasonable
- Ensuring all effective costs are positive
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

def fix_negative_costs(input_file, output_file=None, min_effective_cost=0.01):
    """
    Fix negative costs in vendor ranking data.
    
    Args:
        input_file: Path to the vendor ranking CSV file
        output_file: Path to save the fixed CSV file (defaults to input_file with _fixed suffix)
        min_effective_cost: Minimum allowed effective cost (default: $0.01)
    """
    
    # Load the ranking data
    print(f"Loading ranking data from: {input_file}")
    df = pd.read_csv(input_file)
    original_rows = len(df)
    
    print(f"Loaded {original_rows:,} vendor-material combinations")
    
    # Check for negative costs
    negative_costs = df[df['EffectiveCostPerUnit_USD'] < 0]
    print(f"\nFound {len(negative_costs):,} items with negative effective costs")
    
    if len(negative_costs) > 0:
        print("\nAnalyzing cost components...")
        
        # Show statistics for problematic cost components
        print("\nPrice Trend Impact statistics (before fix):")
        print(f"  Min: ${df['cost_Impact_PriceTrend'].min():,.2f}")
        print(f"  Max: ${df['cost_Impact_PriceTrend'].max():,.2f}")
        print(f"  Mean: ${df['cost_Impact_PriceTrend'].mean():,.2f}")
        
        print("\nInefficiency Cost statistics (before fix):")
        print(f"  Min: ${df['cost_Inefficiency_InFull'].min():,.2f}")
        print(f"  Max: ${df['cost_Inefficiency_InFull'].max():,.2f}")
        print(f"  Mean: ${df['cost_Inefficiency_InFull'].mean():,.2f}")
    
    # Fix 1: Cap extreme price trend slopes
    # Reasonable price changes should be within ±50% per year
    # So for 90 days (typical planning horizon), that's ±12.5%
    # Price trend slope is in $/day, so max reasonable slope = price * 0.5 / 365
    print("\nFixing extreme price trends...")
    
    # Calculate reasonable bounds for price trend slope based on unit price
    max_annual_change_rate = 0.5  # 50% per year
    df['max_reasonable_slope'] = df['AvgUnitPriceUSD_raw'] * max_annual_change_rate / 365
    df['min_reasonable_slope'] = -df['max_reasonable_slope']
    
    # Count items that need fixing
    extreme_trends = df[(df['PriceTrend_raw_slope'] > df['max_reasonable_slope']) | 
                       (df['PriceTrend_raw_slope'] < df['min_reasonable_slope'])]
    print(f"  Found {len(extreme_trends):,} items with extreme price trends")
    
    # Cap the slopes
    df['PriceTrend_raw_slope_fixed'] = df['PriceTrend_raw_slope'].clip(
        lower=df['min_reasonable_slope'],
        upper=df['max_reasonable_slope']
    )
    
    # Recalculate price trend impact with fixed slopes
    # Assuming 90 days planning horizon (should match the original calculation)
    planning_horizon_days = 90
    df['cost_Impact_PriceTrend_fixed'] = df['PriceTrend_raw_slope_fixed'] * (planning_horizon_days / 2.0)
    
    # Fix 2: Adjust extreme inefficiency costs
    print("\nFixing extreme inefficiency costs...")
    
    # For items with very low InFullRate, use a more reasonable penalty
    # Instead of 100x price, use 2x price as max penalty
    max_inefficiency_multiplier = 2.0
    
    # Recalculate inefficiency cost with cap
    df['InFullRate_raw_safe'] = df['InFullRate_raw'].replace(0, 0.01)  # Avoid division by zero
    df['cost_Inefficiency_InFull_fixed'] = np.minimum(
        df['AvgUnitPriceUSD_raw'] * ((1.0 / df['InFullRate_raw_safe']) - 1.0),
        df['AvgUnitPriceUSD_raw'] * max_inefficiency_multiplier
    )
    
    # Recalculate effective cost with fixed components
    print("\nRecalculating effective costs...")
    
    # Get all cost columns
    cost_columns = [col for col in df.columns if col.startswith('cost_') and not col.endswith('_fixed')]
    
    # Replace the problematic cost components with fixed versions
    df['EffectiveCostPerUnit_USD_fixed'] = 0.0
    for col in cost_columns:
        if col == 'cost_Impact_PriceTrend':
            df['EffectiveCostPerUnit_USD_fixed'] += df['cost_Impact_PriceTrend_fixed']
        elif col == 'cost_Inefficiency_InFull':
            df['EffectiveCostPerUnit_USD_fixed'] += df['cost_Inefficiency_InFull_fixed']
        else:
            df['EffectiveCostPerUnit_USD_fixed'] += df[col].fillna(0)
    
    # Ensure minimum effective cost
    df['EffectiveCostPerUnit_USD_fixed'] = df['EffectiveCostPerUnit_USD_fixed'].clip(lower=min_effective_cost)
    
    # Update the original columns with fixed values
    df['PriceTrend_raw_slope'] = df['PriceTrend_raw_slope_fixed']
    df['cost_Impact_PriceTrend'] = df['cost_Impact_PriceTrend_fixed']
    df['cost_Inefficiency_InFull'] = df['cost_Inefficiency_InFull_fixed']
    df['EffectiveCostPerUnit_USD'] = df['EffectiveCostPerUnit_USD_fixed']
    
    # Clean up temporary columns
    temp_columns = ['max_reasonable_slope', 'min_reasonable_slope', 'PriceTrend_raw_slope_fixed',
                   'cost_Impact_PriceTrend_fixed', 'InFullRate_raw_safe', 
                   'cost_Inefficiency_InFull_fixed', 'EffectiveCostPerUnit_USD_fixed']
    df.drop(columns=temp_columns, inplace=True, errors='ignore')
    
    # Show results
    print("\n=== RESULTS ===")
    print(f"\nEffective Cost statistics (after fix):")
    print(f"  Min: ${df['EffectiveCostPerUnit_USD'].min():,.2f}")
    print(f"  Max: ${df['EffectiveCostPerUnit_USD'].max():,.2f}")
    print(f"  Mean: ${df['EffectiveCostPerUnit_USD'].mean():,.2f}")
    
    negative_after = df[df['EffectiveCostPerUnit_USD'] < 0]
    print(f"\nItems with negative costs after fix: {len(negative_after):,}")
    
    # Save the fixed data
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
    
    df.to_csv(output_file, index=False)
    print(f"\nFixed data saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Fix negative costs in vendor ranking data")
    parser.add_argument('input_file', help='Path to the vendor ranking CSV file')
    parser.add_argument('--output-file', '-o', help='Path to save the fixed CSV file')
    parser.add_argument('--min-cost', type=float, default=0.01, 
                       help='Minimum allowed effective cost (default: 0.01)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    fix_negative_costs(args.input_file, args.output_file, args.min_cost)

if __name__ == "__main__":
    main()