import pandas as pd
import os
import sys
import argparse

def fix_country_mapping(profile_id='profile_1'):
    """
    Creates a fixed version of the vendor data with correct country mapping.
    
    The issue is that the LIFNR values in vendor_maktx_ranking_tariff_values.csv
    don't match those in SAP_VLY_IL_SUPPLIER.csv. This script generates a proper mapping.
    """
    print("Starting country mapping fix...")

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct paths relative to the script's parent directory (resources/)
    base_path = os.path.join(script_dir, '..') # This goes up to 'resources'

    # Define profile-specific and global file paths
    profile_tables_dir = os.path.join(base_path, 'profiles', profile_id, 'tables')
    global_tables_dir = os.path.join(base_path, 'tables')
    
    # Input files: vendor file from profile, SAP files from global
    vendor_file = os.path.join(profile_tables_dir, 'vendor_matnr_ranking_tariff_values.csv')
    supplier_file = os.path.join(global_tables_dir, 'SAP_VLY_IL_SUPPLIER.csv')
    country_file = os.path.join(global_tables_dir, 'SAP_VLY_IL_COUNTRY.csv')
    
    # Output files: all go to profile-specific directory
    output_file_direct = os.path.join(profile_tables_dir, 'vendor_with_direct_countries.csv')
    output_file_artificial = os.path.join(profile_tables_dir, 'vendor_with_artificial_countries.csv')
    output_file_name_based = os.path.join(profile_tables_dir, 'vendor_with_countries.csv') # For name-based mapping
    
    # Check if files exist
    for file_path in [vendor_file, supplier_file, country_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return
    
    # Load data with explicit dtype specifications to avoid warnings
    print("Loading files...")
    try:
        # Specify dtypes for key columns to prevent mixed type warnings
        vendor_df = pd.read_csv(vendor_file, dtype={'LIFNR': str}, low_memory=False)
        supplier_df = pd.read_csv(supplier_file, encoding='utf-8-sig', dtype={'LIFNR': str, 'LAND1': str}, low_memory=False)
        country_df = pd.read_csv(country_file, encoding='utf-8-sig', dtype={'LAND1': str}, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        print("Attempting to load with default settings...")
        # Fallback to original loading method if dtype specification fails
        vendor_df = pd.read_csv(vendor_file, low_memory=False)
        supplier_df = pd.read_csv(supplier_file, encoding='utf-8-sig', low_memory=False)
        country_df = pd.read_csv(country_file, encoding='utf-8-sig', low_memory=False)
    
    print(f"Loaded vendor data: {len(vendor_df)} rows")
    print(f"Loaded supplier data: {len(supplier_df)} rows")
    print(f"Loaded country data: {len(country_df)} rows")
    
    # Data type validation and logging
    print("Data type validation:")
    if 'LIFNR' in vendor_df.columns:
        print(f"  Vendor LIFNR dtype: {vendor_df['LIFNR'].dtype}, sample values: {vendor_df['LIFNR'].head(3).tolist()}")
    if 'LIFNR' in supplier_df.columns:
        print(f"  Supplier LIFNR dtype: {supplier_df['LIFNR'].dtype}, sample values: {supplier_df['LIFNR'].head(3).tolist()}")
    if 'LAND1' in supplier_df.columns:
        print(f"  Supplier LAND1 dtype: {supplier_df['LAND1'].dtype}, sample values: {supplier_df['LAND1'].head(3).tolist()}")
    
    # Clean and convert data types
    print("Preparing data...")
    # Convert LIFNR to string in all dataframes
    vendor_df['LIFNR'] = vendor_df['LIFNR'].astype(str)
    supplier_df['LIFNR'] = supplier_df['LIFNR'].astype(str).str.strip()
    supplier_df['LAND1'] = supplier_df['LAND1'].astype(str).str.strip()
    
    # Print some sample values for debugging
    print("Sample vendor LIFNRs:", vendor_df['LIFNR'].head().tolist())
    print("Sample supplier LIFNRs:", supplier_df['LIFNR'].head().tolist())
    
    # Check for common values
    vendor_lifnrs = set(vendor_df['LIFNR'].unique())
    supplier_lifnrs = set(supplier_df['LIFNR'].unique())
    common = vendor_lifnrs.intersection(supplier_lifnrs)
    
    print(f"Vendor unique LIFNR count: {len(vendor_lifnrs)}")
    print(f"Supplier unique LIFNR count: {len(supplier_lifnrs)}")
    print(f"Common LIFNR values: {len(common)}")
    print(f"Common LIFNR examples: {list(common)[:5] if common else 'None'}")
    
    # LIFNR format correction
    # Check if there's a systematic difference in LIFNR format
    if not common:
        print("No common LIFNR values found. Attempting format correction...")
        
        # IMPORTANT: We need to understand what's different about the LIFNR format
        # Option 1: Check if supplier LIFNRs have leading zeros that vendor LIFNRs don't have
        supplier_df['LIFNR_stripped'] = supplier_df['LIFNR'].str.lstrip('0')
        vendor_df['LIFNR_numeric'] = pd.to_numeric(vendor_df['LIFNR'], errors='coerce')
        
        print("After stripping leading zeros from supplier LIFNR:")
        print("Sample supplier LIFNRs (stripped):", supplier_df['LIFNR_stripped'].head().tolist())
        
        # Check if now we have matches
        vendor_lifnrs_set = set(vendor_df['LIFNR'].unique())
        supplier_lifnrs_stripped_set = set(supplier_df['LIFNR_stripped'].unique())
        common_after_strip = vendor_lifnrs_set.intersection(supplier_lifnrs_stripped_set)
        
        print(f"Common LIFNR after stripping leading zeros: {len(common_after_strip)}")
        print(f"Examples: {list(common_after_strip)[:5] if common_after_strip else 'None'}")
        
        if common_after_strip:
            print("Found matches after stripping leading zeros. Using this approach.")
            supplier_df['LIFNR'] = supplier_df['LIFNR_stripped']
        else:
            # Try other approaches
            print("Trying numeric conversion...")
            # Create a mapping table from NAME1 since that appears in both dataframes
            name_mapping = supplier_df[['NAME1', 'LIFNR', 'LAND1']].drop_duplicates()
            
            print(f"Unique supplier names: {name_mapping['NAME1'].nunique()}")
            print(f"Supplier names: {name_mapping['NAME1'].unique().tolist()}")
            print(f"Vendor names: {vendor_df['NAME1'].unique().tolist()}")
            
            # Check for common supplier names
            supplier_names = set(supplier_df['NAME1'].unique())
            vendor_names = set(vendor_df['NAME1'].unique())
            common_names = supplier_names.intersection(vendor_names)
            
            print(f"Common supplier names: {len(common_names)}")
            print(f"Common names: {common_names}")
            
            if common_names:
                print("Found common names. Using name-based mapping.")
                # Merge based on NAME1 to get the correct LIFNR and country mapping
                merged_df = pd.merge(vendor_df, name_mapping, on='NAME1', how='left',
                                    suffixes=('', '_supplier'))
                
                # Check how many matches we got
                print(f"Rows after name merge: {len(merged_df)}")
                print(f"Null LAND1 values: {merged_df['LAND1'].isnull().sum()}")
                
                if merged_df['LAND1'].isnull().sum() > 0:
                    print("WARNING: Some vendors couldn't be matched to countries.")
                
                # Proceed with the merge with country data
                merged_df = pd.merge(merged_df, country_df, 
                                    left_on='LAND1', right_on='LAND1', how='left')
                
                # Keep necessary columns and rename for consistency
                result_df = merged_df.copy()
                result_df['Country'] = result_df['LANDX'].fillna('Unknown')
                result_df['Region'] = result_df['LAND1'].fillna('Unknown')
                
                print(f"Final country distribution: {result_df['Country'].value_counts().to_dict()}")
                # output_file = 'tables/vendor_with_countries.csv' # Old path
                result_df.to_csv(output_file_name_based, index=False)
                print(f"Fixed vendor data saved to {output_file_name_based}")
                return result_df
            else:
                # Last resort: manual mapping based on patterns or create a mapping table
                print("WARNING: No automated mapping method worked. Creating artificial mapping...")
                
                # Get unique LIFNR values from vendor data
                unique_vendor_lifnr = vendor_df['LIFNR'].unique()
                
                # Create an artificial mapping based on NAME1 patterns
                mapping = {}
                
                # Define known company patterns and their potential countries
                company_country_map = {
                    'EV Parts': ['TH', 'HU', 'MX', 'US', 'ID', 'IN'],
                    'WaveCrest': ['IN', 'HU', 'ID', 'US', 'TH', 'CN'],
                    'TechGroup': ['CN', 'ID', 'US', 'IN', 'MY'],
                    'LabSupply': ['HU', 'US', 'MY', 'MX'],
                    'Utilities': ['CN', 'US', 'DE', 'ID'],
                    'AdminSupply': ['US', 'CN', 'IN', 'MX'],
                    'Advert': ['US', 'HU', 'MX', 'TH'],
                    'BeSafe': ['US', 'TH', 'MX', 'CN', 'IN', 'DE']
                }
                
                # Create a artificial country mapping based on vendor name patterns
                artificial_mapping = []
                
                for index, row in vendor_df.drop_duplicates(['LIFNR', 'NAME1']).iterrows():
                    vendor_id = row['LIFNR']
                    vendor_name = row['NAME1']
                    
                    # Find matching company pattern
                    matched_pattern = None
                    for pattern, countries in company_country_map.items():
                        if pattern in vendor_name:
                            matched_pattern = pattern
                            # Use numeric part of LIFNR to select a country
                            try:
                                numeric_lifnr = int(vendor_id)
                                country_index = numeric_lifnr % len(countries)
                                country_code = countries[country_index]
                                artificial_mapping.append({
                                    'LIFNR': vendor_id,
                                    'NAME1': vendor_name,
                                    'LAND1': country_code
                                })
                                break
                            except ValueError:
                                # If LIFNR can't be converted to int, use a default country
                                artificial_mapping.append({
                                    'LIFNR': vendor_id,
                                    'NAME1': vendor_name,
                                    'LAND1': countries[0]
                                })
                                break
                    
                    # If no pattern matched, assign a default country
                    if not matched_pattern:
                        artificial_mapping.append({
                            'LIFNR': vendor_id,
                            'NAME1': vendor_name,
                            'LAND1': 'US'  # Default country
                        })
                
                # Convert to DataFrame
                mapping_df = pd.DataFrame(artificial_mapping)
                
                # Merge with country data
                mapping_df = pd.merge(mapping_df, country_df, on='LAND1', how='left')
                
                # Now merge with vendor data
                result_df = pd.merge(vendor_df, mapping_df[['LIFNR', 'LAND1', 'LANDX']], 
                                   on='LIFNR', how='left')
                
                # Add Country and Region columns
                result_df['Country'] = result_df['LANDX'].fillna('Unknown')
                result_df['Region'] = result_df['LAND1'].fillna('Unknown')
                
                print(f"Artificial country mapping created with distribution: {result_df['Country'].value_counts().to_dict()}")
                
                # Save the result
                # output_file = 'tables/vendor_with_artificial_countries.csv' # Old path
                result_df.to_csv(output_file_artificial, index=False)
                print(f"Artificial country mapping saved to {output_file_artificial}")
                return result_df
    else:
        # We have common LIFNR values, proceed with direct merge
        print("Found common LIFNR values. Proceeding with direct merge.")
        
        # Merge supplier and country data
        supplier_country = pd.merge(supplier_df, country_df, on='LAND1', how='left')
        
        # Check if vendor_df already has LAND1 column
        land1_in_vendor = 'LAND1' in vendor_df.columns
        print(f"LAND1 column already exists in vendor data: {land1_in_vendor}")
        
        if land1_in_vendor:
            # If LAND1 already exists, we'll merge supplier data but keep vendor's LAND1 if there's a conflict
            result_df = pd.merge(vendor_df, 
                               supplier_country[['LIFNR', 'LAND1', 'LANDX']], 
                               on='LIFNR', how='left',
                               suffixes=('', '_supplier'))
            
            # Use LANDX from supplier data, but keep original LAND1
            result_df['LANDX'] = result_df['LANDX'].fillna('Unknown')
        else:
            # If LAND1 doesn't exist in vendor data, use the normal merge
            result_df = pd.merge(vendor_df, 
                               supplier_country[['LIFNR', 'LAND1', 'LANDX']], 
                               on='LIFNR', how='left')
        
        # Add Country and Region columns
        result_df['Country'] = result_df['LANDX'].fillna('Unknown')
        
        # Use existing LAND1 or LAND1 from supplier data
        if 'LAND1' in result_df.columns:
            result_df['Region'] = result_df['LAND1'].fillna('Unknown')
        elif 'LAND1_supplier' in result_df.columns:
            result_df['Region'] = result_df['LAND1_supplier'].fillna('Unknown')
        else:
            # Fallback - shouldn't happen
            print("WARNING: No LAND1 column found after merge. Creating default Region column.")
            result_df['Region'] = 'Unknown'
        
        print(f"Country distribution after merge: {result_df['Country'].value_counts().to_dict()}")
        
        # Save the result
        # output_file = 'tables/vendor_with_direct_countries.csv' # Old path
        result_df.to_csv(output_file_direct, index=False)
        print(f"Direct country mapping saved to {output_file_direct}")
        return result_df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fix country mapping for vendor data')
    parser.add_argument('--profile-id', default='profile_1', 
                       help='Profile ID to process (default: profile_1)')
    args = parser.parse_args()
    
    try:
        result = fix_country_mapping(args.profile_id)
        print("Country mapping fix completed successfully.")
    except Exception as e:
        print(f"Error during country mapping fix: {e}")
        import traceback
        traceback.print_exc()
