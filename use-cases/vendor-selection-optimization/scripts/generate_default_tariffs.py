"""
Generate default tariff values JSON file using existing tariff data from vendor ranking file.

This script reads the vendor_maktx_ranking_tariff_values.csv file and extracts the
average tariff percentage by country. If this file doesn't exist, it falls back to
setting all countries to 0%.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

def generate_default_tariffs():
    """
    Generate default tariff values by extracting average tariffs by country from 
    the vendor ranking file. Falls back to 0% for all countries if the file doesn't exist.
    """
    print("Generating default country tariff values...")
    
    # Define file paths
    country_file = os.path.join('tables', 'SAP_VLY_IL_COUNTRY.csv')
    vendor_ranking_file = os.path.join('tables', 'vendor_maktx_ranking_tariff_values.csv')
    country_mapping_file = os.path.join('tables', 'vendor_with_direct_countries.csv')
    output_file = os.path.join('tables', 'tariff_values.json')
    
    try:
        # Load country data
        country_df = pd.read_csv(country_file, encoding='utf-8-sig')
        print(f"Loaded country data: {len(country_df)} records")
        
        # Extract unique country codes
        unique_countries = country_df['LAND1'].unique()
        print(f"Found {len(unique_countries)} unique country codes")
        
        # Default tariff values (all 0%)
        tariff_values = {country: 0.0 for country in unique_countries}
        
        # Try to extract existing tariff values from vendor ranking file
        try:
            # Prioritize vendor_maktx_ranking_tariff_values.csv if it exists
            if os.path.exists(vendor_ranking_file):
                print(f"Loading vendor data from {vendor_ranking_file}...")
                vendor_df = pd.read_csv(vendor_ranking_file)
                
                # Check if we have the country data directly in the vendor file
                if 'LAND1' in vendor_df.columns:
                    print("Found LAND1 column directly in vendor file")
                    country_mapping = vendor_df
                elif os.path.exists(country_mapping_file):
                    # If not, load the vendor-country mapping file
                    print(f"Loading country mapping from {country_mapping_file}...")
                    country_mapping = pd.read_csv(country_mapping_file)
                    
                    # Merge to get country codes for each vendor
                    if 'LIFNR' in vendor_df.columns and 'LIFNR' in country_mapping.columns:
                        vendor_df = pd.merge(vendor_df, 
                                            country_mapping[['LIFNR', 'LAND1']], 
                                            on='LIFNR', how='left')
                        print(f"Merged vendor data with country mapping. LAND1 column available: {'LAND1' in vendor_df.columns}")
                
                # Now calculate average tariff by country
                if 'LAND1' in vendor_df.columns and 'TariffImpact_raw_percent' in vendor_df.columns:
                    print("Calculating average tariffs by country...")
                    
                    # Group by country and calculate average tariff
                    tariff_by_country = vendor_df.groupby('LAND1')['TariffImpact_raw_percent'].mean()
                    
                    # Summary stats for debugging
                    tariff_min = vendor_df['TariffImpact_raw_percent'].min()
                    tariff_max = vendor_df['TariffImpact_raw_percent'].max()
                    tariff_mean = vendor_df['TariffImpact_raw_percent'].mean()
                    print(f"Tariff statistics: Min={tariff_min:.2f}%, Max={tariff_max:.2f}%, Mean={tariff_mean:.2f}%")
                    
                    # Convert to dictionary and update tariff_values
                    non_zero_count = 0
                    for country, tariff in tariff_by_country.items():
                        if pd.notna(country) and pd.notna(tariff):
                            # Round to 1 decimal place for user-friendliness
                            tariff_rounded = round(float(tariff), 1)
                            if tariff_rounded > 0.0:
                                non_zero_count += 1
                            # Update even if the country wasn't in the original list
                            tariff_values[country] = tariff_rounded
                    
                    print(f"Extracted tariff values for {len(tariff_by_country)} countries from vendor data ({non_zero_count} with non-zero values)")
                    
                    # Print top 5 countries with highest tariffs for verification
                    top_tariffs = tariff_by_country.sort_values(ascending=False).head(5)
                    if not top_tariffs.empty:
                        print("Top 5 countries by tariff value:")
                        for country, value in top_tariffs.items():
                            print(f"  {country}: {value:.2f}%")
                else:
                    print(f"Warning: Required columns not found in vendor data. LAND1: {'LAND1' in vendor_df.columns}, TariffImpact_raw_percent: {'TariffImpact_raw_percent' in vendor_df.columns}. Using default 0% tariffs.")
            else:
                print(f"Warning: Vendor ranking file not found at {vendor_ranking_file}. Using default 0% tariffs.")
        except Exception as e:
            import traceback
            print(f"Warning: Error while extracting tariffs from vendor data: {e}")
            traceback.print_exc()
            print("Using default 0% tariffs.")

        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(tariff_values, f, indent=2)
        
        print(f"Default tariff values saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error generating default tariff values: {e}")
        return False

if __name__ == "__main__":
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        success = generate_default_tariffs()
        if success:
            print("Default tariff configuration completed successfully.")
        else:
            print("Failed to generate default tariff configuration.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during tariff generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)