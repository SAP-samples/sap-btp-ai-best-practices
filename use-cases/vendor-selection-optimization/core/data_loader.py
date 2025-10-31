"""
Data loading and processing functions for the Vendor Performance Dashboard.
Handles loading data from files and preparing it for analysis and visualization.
"""

import os
import json
import pandas as pd
import streamlit as st
from config import settings
from optimization.profile_manager import ProfileManager


# Function removed - no longer using automatic top-40 filtering

@st.cache_data(ttl=3600)
def get_all_materials():
    """
    Get all available materials (MATNR and MAKTX) from the vendor data
    
    Returns:
        dict: Dictionary with 'MATNR' and 'MAKTX' keys containing lists of unique values
    """
    # Try to load from the vendor data files
    for file_path in settings.VENDOR_FILES:
        if os.path.exists(file_path):
            try:
                # Load only the columns we need
                df = pd.read_csv(file_path, 
                               usecols=['MATNR', 'MAKTX'] if 'MATNR' in pd.read_csv(file_path, nrows=0).columns else ['MAKTX'],
                               dtype={'MATNR': 'str', 'MAKTX': 'str'}, 
                               low_memory=False)
                
                materials = {
                    'MAKTX': sorted(df['MAKTX'].dropna().unique().tolist())
                }
                
                # Add MATNR if it exists in the data
                if 'MATNR' in df.columns:
                    materials['MATNR'] = sorted(df['MATNR'].dropna().unique().tolist())
                else:
                    materials['MATNR'] = []
                
                print(f"Found {len(materials['MAKTX'])} unique material descriptions")
                if materials['MATNR']:
                    print(f"Found {len(materials['MATNR'])} unique material numbers")
                
                return materials
            except Exception as e:
                print(f"Error loading materials from {file_path}: {e}")
                continue
    
    # If no files found or all failed, return empty lists
    return {'MATNR': [], 'MAKTX': []}


def get_profile_aware_vendor_files(profile_id=None):
    """
    Get vendor file paths that are profile-aware.
    
    Args:
        profile_id: Profile ID to use. If None, uses the active profile.
    
    Returns:
        list: List of vendor file paths based on profile configuration
    """
    # Initialize ProfileManager
    profile_manager = ProfileManager(".")
    
    # Get profile ID
    if profile_id is None:
        profile_id = profile_manager.get_active_profile()
    
    # Get profile-specific paths
    profile_tables_dir = profile_manager.get_profile_tables_dir(profile_id)
    global_tables_dir = profile_manager.get_global_tables_dir()
    
    # Check for profile-specific vendor files first
    profile_vendor_files = [
        os.path.join(profile_tables_dir, 'vendor_with_direct_countries.csv'),
        os.path.join(profile_tables_dir, 'vendor_matnr_ranking_tariff_values.csv')
    ]
    
    # Check which files exist
    existing_files = []
    for file_path in profile_vendor_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    # If no profile-specific files exist, fall back to Profile 1 data as safety net
    if not existing_files and profile_id != 'profile_1':
        print(f"Warning: No vendor files found for {profile_id}, falling back to Profile 1 data")
        profile_1_tables_dir = profile_manager.get_profile_tables_dir('profile_1')
        profile_1_vendor_files = [
            os.path.join(profile_1_tables_dir, 'vendor_with_direct_countries.csv'),
            os.path.join(profile_1_tables_dir, 'vendor_matnr_ranking_tariff_values.csv')
        ]
        
        for file_path in profile_1_vendor_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
    
    # If still no files found (including Profile 1), fall back to global files (legacy fallback)
    if not existing_files:
        for file_path in settings.VENDOR_FILES:
            if os.path.exists(file_path):
                existing_files.append(file_path)
    
    return existing_files


@st.cache_data(ttl=3600)
def get_all_materials_profile_aware(profile_id=None):
    """
    Get all available materials (MATNR and MAKTX) from the vendor data with profile awareness.
    
    Args:
        profile_id: Profile ID to use. If None, uses the active profile.
    
    Returns:
        dict: Dictionary with 'MATNR' and 'MAKTX' keys containing lists of unique values
    """
    # Get profile-aware vendor files
    vendor_files = get_profile_aware_vendor_files(profile_id)
    
    # Try to load from the vendor data files
    for file_path in vendor_files:
        if os.path.exists(file_path):
            try:
                # Load only the columns we need
                df = pd.read_csv(file_path, 
                               usecols=['MATNR', 'MAKTX'] if 'MATNR' in pd.read_csv(file_path, nrows=0).columns else ['MAKTX'],
                               dtype={'MATNR': 'str', 'MAKTX': 'str'}, 
                               low_memory=False)
                
                materials = {
                    'MAKTX': sorted(df['MAKTX'].dropna().unique().tolist())
                }
                
                # Add MATNR if it exists in the data
                if 'MATNR' in df.columns:
                    materials['MATNR'] = sorted(df['MATNR'].dropna().unique().tolist())
                else:
                    materials['MATNR'] = []
                
                print(f"Found {len(materials['MAKTX'])} unique material descriptions (Profile: {profile_id})")
                if materials['MATNR']:
                    print(f"Found {len(materials['MATNR'])} unique material numbers (Profile: {profile_id})")
                
                return materials
            except Exception as e:
                print(f"Error loading materials from {file_path}: {e}")
                continue
    
    # If no files found or all failed, return empty lists
    return {'MATNR': [], 'MAKTX': []}


@st.cache_data(ttl=3600)
def get_material_relationships_profile_aware(profile_id=None):
    """
    Get the relationships between MATNR and MAKTX from the vendor data with profile awareness.
    
    Args:
        profile_id: Profile ID to use. If None, uses the active profile.
    
    Returns:
        dict: Dictionary with:
            - 'matnr_to_maktx': Dict mapping each MATNR to its MAKTX
            - 'maktx_to_matnrs': Dict mapping each MAKTX to list of MATNRs
            - 'maktx_counts': Dict with count of MATNRs per MAKTX
    """
    # Get profile-aware vendor files
    vendor_files = get_profile_aware_vendor_files(profile_id)
    
    # Try to load from the vendor data files
    for file_path in vendor_files:
        if os.path.exists(file_path):
            try:
                # Check if MATNR column exists
                columns = pd.read_csv(file_path, nrows=0).columns
                if 'MATNR' not in columns:
                    # If no MATNR column, return empty relationships
                    return {
                        'matnr_to_maktx': {},
                        'maktx_to_matnrs': {},
                        'maktx_counts': {}
                    }
                
                # Load only the columns we need
                df = pd.read_csv(file_path, 
                               usecols=['MATNR', 'MAKTX'],
                               dtype={'MATNR': 'str', 'MAKTX': 'str'}, 
                               low_memory=False)
                
                # Drop rows with missing values
                df = df.dropna(subset=['MATNR', 'MAKTX'])
                
                # Get unique MATNR-MAKTX combinations
                unique_materials = df[['MATNR', 'MAKTX']].drop_duplicates()
                
                # Build relationships
                matnr_to_maktx = dict(zip(unique_materials['MATNR'], unique_materials['MAKTX']))
                
                # Build reverse mapping (MAKTX to list of MATNRs)
                maktx_to_matnrs = {}
                for matnr, maktx in matnr_to_maktx.items():
                    if maktx not in maktx_to_matnrs:
                        maktx_to_matnrs[maktx] = []
                    maktx_to_matnrs[maktx].append(matnr)
                
                # Sort the MATNR lists for each MAKTX
                for maktx in maktx_to_matnrs:
                    maktx_to_matnrs[maktx] = sorted(maktx_to_matnrs[maktx])
                
                # Count MATNRs per MAKTX
                maktx_counts = {maktx: len(matnrs) for maktx, matnrs in maktx_to_matnrs.items()}
                
                print(f"Found {len(matnr_to_maktx)} MATNR-MAKTX relationships (Profile: {profile_id})")
                print(f"Materials with multiple variants: {sum(1 for count in maktx_counts.values() if count > 1)}")
                
                return {
                    'matnr_to_maktx': matnr_to_maktx,
                    'maktx_to_matnrs': maktx_to_matnrs,
                    'maktx_counts': maktx_counts
                }
                
            except Exception as e:
                print(f"Error loading material relationships from {file_path}: {e}")
                continue
    
    # If no files found or all failed, return empty relationships
    return {
        'matnr_to_maktx': {},
        'maktx_to_matnrs': {},
        'maktx_counts': {}
    }


@st.cache_data(ttl=3600)
def get_material_relationships():
    """
    Get the relationships between MATNR and MAKTX from the vendor data
    
    Returns:
        dict: Dictionary with:
            - 'matnr_to_maktx': Dict mapping each MATNR to its MAKTX
            - 'maktx_to_matnrs': Dict mapping each MAKTX to list of MATNRs
            - 'maktx_counts': Dict with count of MATNRs per MAKTX
    """
    # Try to load from the vendor data files
    for file_path in settings.VENDOR_FILES:
        if os.path.exists(file_path):
            try:
                # Check if MATNR column exists
                columns = pd.read_csv(file_path, nrows=0).columns
                if 'MATNR' not in columns:
                    # If no MATNR column, return empty relationships
                    return {
                        'matnr_to_maktx': {},
                        'maktx_to_matnrs': {},
                        'maktx_counts': {}
                    }
                
                # Load only the columns we need
                df = pd.read_csv(file_path, 
                               usecols=['MATNR', 'MAKTX'],
                               dtype={'MATNR': 'str', 'MAKTX': 'str'}, 
                               low_memory=False)
                
                # Drop rows with missing values
                df = df.dropna(subset=['MATNR', 'MAKTX'])
                
                # Get unique MATNR-MAKTX combinations
                unique_materials = df[['MATNR', 'MAKTX']].drop_duplicates()
                
                # Build relationships
                matnr_to_maktx = dict(zip(unique_materials['MATNR'], unique_materials['MAKTX']))
                
                # Build reverse mapping (MAKTX to list of MATNRs)
                maktx_to_matnrs = {}
                for matnr, maktx in matnr_to_maktx.items():
                    if maktx not in maktx_to_matnrs:
                        maktx_to_matnrs[maktx] = []
                    maktx_to_matnrs[maktx].append(matnr)
                
                # Sort the MATNR lists for each MAKTX
                for maktx in maktx_to_matnrs:
                    maktx_to_matnrs[maktx] = sorted(maktx_to_matnrs[maktx])
                
                # Count MATNRs per MAKTX
                maktx_counts = {maktx: len(matnrs) for maktx, matnrs in maktx_to_matnrs.items()}
                
                print(f"Found {len(matnr_to_maktx)} MATNR-MAKTX relationships")
                print(f"Materials with multiple variants: {sum(1 for count in maktx_counts.values() if count > 1)}")
                
                return {
                    'matnr_to_maktx': matnr_to_maktx,
                    'maktx_to_matnrs': maktx_to_matnrs,
                    'maktx_counts': maktx_counts
                }
                
            except Exception as e:
                print(f"Error loading material relationships from {file_path}: {e}")
                continue
    
    # If no files found or all failed, return empty relationships
    return {
        'matnr_to_maktx': {},
        'maktx_to_matnrs': {},
        'maktx_counts': {}
    }

@st.cache_data
def load_costs_config(profile_id=None):
    """Load cost configuration from costs.json with profile awareness"""
    # Initialize ProfileManager and get profile ID
    profile_manager = ProfileManager(".")
    if profile_id is None:
        profile_id = profile_manager.get_active_profile()
    
    # Get profile-specific config file path
    profile_config_dir = profile_manager.get_profile_config_dir(profile_id)
    costs_config_file = os.path.join(profile_config_dir, 'costs.json')
    
    try:
        with open(costs_config_file, 'r') as f:
            config_data = json.load(f)
        
        # Handle both old and new format
        if 'cost_components' in config_data:
            # New format - return just the cost components for backward compatibility
            return config_data['cost_components']
        else:
            # Old format - return as is
            return config_data
    except FileNotFoundError:
        st.error(f"⚠️ {costs_config_file} file not found. Using all cost components.")
        return None
    except json.JSONDecodeError:
        st.error(f"⚠️ Error reading {costs_config_file}. Using all cost components.")
        return None


@st.cache_data(max_entries=2, ttl=3600)  # Limit cache to prevent memory accumulation
def load_country_material_mapping():
    """Load and join SAP tables to get country information for each material - Memory Optimized"""
    import gc
    try:
        tables_dir = settings.TABLES_DIR
        
        # Define optimized data types to reduce memory by 50-70%
        material_dtypes = {'MATNR': 'category', 'MAKTX': 'string'}
        po_items_dtypes = {'MATNR': 'category', 'EBELN': 'category', 'EBELP': 'string'}
        po_header_dtypes = {'EBELN': 'category', 'LIFNR': 'category'}
        supplier_dtypes = {'LIFNR': 'category', 'LAND1': 'category', 'NAME1': 'string'}
        country_dtypes = {'LAND1': 'category', 'LANDX': 'string'}
        
        # Load only essential columns to minimize memory usage
        print("Loading material data...")
        material_df = pd.read_csv(f'{tables_dir}/SAP_VLY_IL_MATERIAL.csv', 
                                encoding=settings.FILE_ENCODING,
                                usecols=['MATNR', 'MAKTX'], 
                                dtype=material_dtypes)
        
        print("Loading PO items data...")
        po_items_df = pd.read_csv(f'{tables_dir}/SAP_VLY_IL_PO_ITEMS.csv', 
                                encoding=settings.FILE_ENCODING,
                                usecols=['MATNR', 'EBELN', 'EBELP'], 
                                dtype=po_items_dtypes)
        
        print("Loading PO header data...")
        po_header_df = pd.read_csv(f'{tables_dir}/SAP_VLY_IL_PO_HEADER.csv', 
                                 encoding=settings.FILE_ENCODING,
                                 usecols=['EBELN', 'LIFNR'], 
                                 dtype=po_header_dtypes)
        
        print("Loading supplier data...")
        supplier_df = pd.read_csv(f'{tables_dir}/SAP_VLY_IL_SUPPLIER.csv', 
                                encoding=settings.FILE_ENCODING,
                                usecols=['LIFNR', 'LAND1', 'NAME1'], 
                                dtype=supplier_dtypes)
        
        print("Loading country data...")
        country_df = pd.read_csv(f'{tables_dir}/SAP_VLY_IL_COUNTRY.csv', 
                               encoding=settings.FILE_ENCODING,
                               usecols=['LAND1', 'LANDX'], 
                               dtype=country_dtypes)
        
        # Memory-efficient merging with explicit cleanup
        print("Merging datasets...")
        
        # Step 1: Material to PO Items (cleanup intermediate results)
        material_po = pd.merge(material_df, po_items_df, on='MATNR', how='inner')
        del material_df, po_items_df  # Explicit cleanup
        gc.collect()
        
        # Step 2: Add PO Header information
        material_po_header = pd.merge(material_po, po_header_df[['EBELN', 'LIFNR']], on='EBELN', how='inner')
        del material_po, po_header_df  # Explicit cleanup
        gc.collect()
        
        # Step 3: Add Supplier information  
        material_supplier = pd.merge(material_po_header, supplier_df[['LIFNR', 'LAND1', 'NAME1']], on='LIFNR', how='inner')
        del material_po_header, supplier_df  # Explicit cleanup
        gc.collect()
        
        # Step 4: Add Country information
        material_country = pd.merge(material_supplier, country_df, on='LAND1', how='left')
        del material_supplier, country_df  # Explicit cleanup
        gc.collect()
        
        # Aggregate to get unique material-country combinations
        print("Aggregating final results...")
        material_country_agg = material_country.groupby(['MATNR', 'MAKTX', 'LAND1', 'LANDX'], observed=False).agg({
            'LIFNR': 'nunique',
            'EBELN': 'nunique'
        }).reset_index()
        
        del material_country  # Final cleanup
        gc.collect()
        
        material_country_agg.columns = ['MATNR', 'MAKTX', 'Country_Code', 'Country_Name', 'Supplier_Count', 'PO_Count']
        
        print(f"Memory optimized country-material mapping completed. Final size: {len(material_country_agg):,} rows")
        return material_country_agg
        
    except Exception as e:
        st.error(f"Error loading country-material mapping: {e}")
        return pd.DataFrame()


@st.cache_data(max_entries=10, ttl=3600)  # Increased cache entries to prevent premature eviction
def load_vendor_data(force_reload=False, material_filters=None, profile_id=None):
    """
    Load vendor data from the CSV file with profile awareness
    
    Args:
        force_reload: If True, bypass the cache and reload the data
        material_filters: Dict with 'MATNR' and/or 'MAKTX' keys for filtering
        profile_id: Profile ID to use. If None, uses the active profile.
    """
    # Initialize ProfileManager and get profile ID
    profile_manager = ProfileManager(".")
    if profile_id is None:
        profile_id = profile_manager.get_active_profile()
    
    # Get profile-aware file paths
    profile_tables_dir = profile_manager.get_profile_tables_dir(profile_id)
    
    # Get the current tariff values for cache invalidation checking
    tariff_path = os.path.join(profile_tables_dir, 'tariff_values.json')
    tariff_mtime = 0
    if os.path.exists(tariff_path):
        tariff_mtime = os.path.getmtime(tariff_path)
    
    # Also check modification times of key data files in profile directory
    vendor_file = os.path.join(profile_tables_dir, 'vendor_matnr_ranking_tariff_values.csv')
    vendor_country_file = os.path.join(profile_tables_dir, 'vendor_with_direct_countries.csv')
    
    vendor_mtime = 0
    country_mtime = 0
    
    if os.path.exists(vendor_file):
        vendor_mtime = os.path.getmtime(vendor_file)
    
    if os.path.exists(vendor_country_file):
        country_mtime = os.path.getmtime(vendor_country_file)
    
    # Store modification times in session_state for change detection
    if 'tariff_mtime' not in st.session_state:
        st.session_state.tariff_mtime = tariff_mtime
    
    if 'vendor_mtime' not in st.session_state:
        st.session_state.vendor_mtime = vendor_mtime
        
    if 'country_mtime' not in st.session_state:
        st.session_state.country_mtime = country_mtime
    
    # Force reload if any of the key files have changed since last load (with 1 second tolerance)
    if abs(tariff_mtime - st.session_state.tariff_mtime) > 1:
        print(f"Tariff values modified. Forcing data reload.")
        force_reload = True
        st.session_state.tariff_mtime = tariff_mtime
        
    if abs(vendor_mtime - st.session_state.vendor_mtime) > 1:
        print(f"Vendor data modified. Forcing data reload.")
        force_reload = True
        st.session_state.vendor_mtime = vendor_mtime
        
    if abs(country_mtime - st.session_state.country_mtime) > 1:
        print(f"Country mapping modified. Forcing data reload.")
        force_reload = True
        st.session_state.country_mtime = country_mtime
        
    # If force_reload is True, clear only this function's cache instead of all caches
    if force_reload:
        # Clear only the vendor data cache instead of all caches
        load_vendor_data.clear()
        print("Forced data reload requested - clearing vendor data cache only")
    
    # Use profile-aware vendor files
    vendor_files = get_profile_aware_vendor_files(profile_id)
    
    # First try the merged data file if it exists
    file_path = None
    for f in vendor_files:
        if os.path.exists(f):
            file_path = f
            print(f"Using vendor data file: {file_path} (Profile: {profile_id})")
            break
    
    if not file_path:
        print(f"Error: No vendor data file found for profile {profile_id}")
        return None
    
    try:
        # Memory optimization: Read with optimized dtypes
        dtype_dict = {
            'LIFNR': 'str',
            'NAME1': 'str', 
            'MAKTX': 'str',
            'LAND1': 'str'
        }
        
        df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
        
        # Memory optimization: Apply data type optimization after loading
        from core.utils import optimize_dataframe_memory, get_memory_usage_info
        
        original_memory = get_memory_usage_info(df)
        
        # Data preparation for performance dashboard
        df['Supplier_ID'] = df['LIFNR'].astype('str')
        df['Supplier_Name'] = df['NAME1'].astype('str')
        df['MedianLeadTimeDays'] = df['AvgLeadTimeDays_raw']
        df['OnTimeRate'] = df['OnTimeRate_raw']
        df['InFullRate'] = df['InFullRate_raw']
        df['Avg_OTIF_Rate'] = df['OnTimeRate'] * df['InFullRate']
        
        # Memory optimization: Optimize data types
        df = optimize_dataframe_memory(df)
        
        optimized_memory = get_memory_usage_info(df)
        memory_saved = original_memory['total_memory_mb'] - optimized_memory['total_memory_mb']
        
        if memory_saved > 0:
            print(f"Memory optimization: Saved {memory_saved:.1f} MB ({memory_saved/original_memory['total_memory_mb']*100:.1f}% reduction)")
        
        # If using the original data without countries, check for vendor_with_direct_countries.csv first
        # then try merging with supplier data if that's not available
        if file_path == os.path.join(profile_tables_dir, 'vendor_matnr_ranking_tariff_values.csv'):
            country_mapping_file = os.path.join(profile_tables_dir, 'vendor_with_direct_countries.csv')
            
            # First try using the dedicated country mapping file if it exists
            if os.path.exists(country_mapping_file):
                print(f"Found country mapping file: {country_mapping_file}")
                try:
                    # Load country mapping
                    mapping_df = pd.read_csv(country_mapping_file)
                    mapping_df['LIFNR'] = mapping_df['LIFNR'].astype(str).str.strip()
                    df['LIFNR'] = df['LIFNR'].astype(str).str.strip()
                    
                    # Merge with vendor data on LIFNR
                    merge_cols = ['LIFNR', 'LAND1', 'Country', 'Region'] 
                    merge_cols = [col for col in merge_cols if col in mapping_df.columns]
                    
                    if len(merge_cols) > 1:  # At minimum, we need LIFNR and one more column
                        df = pd.merge(df, mapping_df[merge_cols], on='LIFNR', how='left')
                        print(f"Merged with country mapping. Columns added: {merge_cols[1:]}")
                        
                        # Set default values for missing columns
                        if 'Country' not in df.columns and 'LANDX' in df.columns:
                            df['Country'] = df['LANDX'].fillna('Unknown')
                        elif 'Country' not in df.columns:
                            df['Country'] = 'Unknown'
                            
                        if 'Region' not in df.columns and 'LAND1' in df.columns:
                            df['Region'] = df['LAND1'].fillna('Unknown')
                        elif 'Region' not in df.columns:
                            df['Region'] = 'Unknown'
                        
                        print(f"Country data merged from mapping file. Found {df['Country'].nunique()} unique countries.")
                    else:
                        print(f"Warning: Country mapping file doesn't have required columns. Available: {mapping_df.columns.tolist()}")
                        # Fall back to supplier data method
                        raise ValueError("Insufficient columns in country mapping file")
                except Exception as e:
                    print(f"Error using country mapping file: {e}. Falling back to supplier data method.")
                    # Fall back to supplier data method
                    country_mapping_success = False
            
            # If country mapping file doesn't exist or failed, try the traditional supplier data method
            if not os.path.exists(country_mapping_file) or 'country_mapping_success' in locals() and not country_mapping_success:
                print("Using original vendor file. Attempting to merge with supplier data...")
                try:
                    global_tables_dir = profile_manager.get_global_tables_dir()
                    supplier_file = os.path.join(global_tables_dir, 'SAP_VLY_IL_SUPPLIER.csv')
                    country_file = os.path.join(global_tables_dir, 'SAP_VLY_IL_COUNTRY.csv')
                    
                    if os.path.exists(supplier_file) and os.path.exists(country_file):
                        # Load supplier and country data
                        supplier_df = pd.read_csv(supplier_file, encoding=settings.FILE_ENCODING)
                        country_df = pd.read_csv(country_file, encoding=settings.FILE_ENCODING)
                        
                        # Clean data types
                        supplier_df['LIFNR'] = supplier_df['LIFNR'].astype(str).str.strip()
                        supplier_df['LAND1'] = supplier_df['LAND1'].astype(str).str.strip()
                        country_df['LAND1'] = country_df['LAND1'].astype(str).str.strip()
                        df['LIFNR'] = df['LIFNR'].astype(str).str.strip()
                        
                        # Merge supplier with country data
                        supplier_country = pd.merge(supplier_df, country_df, on='LAND1', how='left')
                        
                        # Merge vendor data with supplier country data
                        df = pd.merge(df, supplier_country[['LIFNR', 'LAND1', 'LANDX']], 
                                     on='LIFNR', how='left')
                        
                        # Set country information
                        df['Country'] = df['LANDX'].fillna('Unknown')
                        df['Region'] = df['LAND1'].fillna('Unknown')
                        
                        print(f"Country data merged from supplier data. Found {df['Country'].nunique()} unique countries.")
                    else:
                        print("Supplier or country file not found. Using 'Unknown' for all countries.")
                        df['Country'] = 'Unknown'
                        df['Region'] = 'Unknown'
                except Exception as e:
                    print(f"Error merging country data: {e}")
                    df['Country'] = 'Unknown'
                    df['Region'] = 'Unknown'
        else:
            # If we're using a pre-merged file, make sure Country and Region exist
            if 'Country' not in df.columns:
                if 'LANDX' in df.columns:
                    df['Country'] = df['LANDX'].fillna('Unknown')
                else:
                    df['Country'] = 'Unknown'
                    
            if 'Region' not in df.columns:
                if 'LAND1' in df.columns:
                    df['Region'] = df['LAND1'].fillna('Unknown')
                else:
                    df['Region'] = 'Unknown'
                    
            print(f"Using pre-merged country data. Found {df['Country'].nunique()} unique countries.")
            print(f"Country distribution: {df['Country'].value_counts().to_dict()}")
                    
        # Clean and prepare other data
        df['MAKTX'] = df['MAKTX'].astype(str).fillna('')
        df['NAME1'] = df['NAME1'].astype(str).fillna('')
        df['LIFNR'] = df['LIFNR'].astype(str)
        df['LAND1'] = df['LAND1'].astype(str).fillna('')
        # Include country code in VendorFullID
        df['VendorFullID'] = df['NAME1'] + '-' + df['LIFNR'] + ' (' + df['LAND1'] + ')'
        
        # MATERIAL FILTERING: Apply user-selected material filters
        if material_filters:
            original_material_count = df['MAKTX'].nunique()
            original_row_count = len(df)
            
            # Apply MATNR filter if provided
            if material_filters.get('MATNR'):
                df = df[df['MATNR'] == material_filters['MATNR']].copy()
            
            # Apply MAKTX filter if provided
            if material_filters.get('MAKTX'):
                df = df[df['MAKTX'] == material_filters['MAKTX']].copy()
            
            filtered_material_count = df['MAKTX'].nunique()
            filtered_row_count = len(df)
            
            if filtered_row_count > 0:
                print(f"🎯 MATERIAL FILTER APPLIED:")
                print(f"   Materials: {original_material_count:,} → {filtered_material_count:,}")
                print(f"   Rows: {original_row_count:,} → {filtered_row_count:,}")
                if material_filters.get('MATNR'):
                    print(f"   Filtered by MATNR: {material_filters['MATNR']}")
                if material_filters.get('MAKTX'):
                    print(f"   Filtered by MAKTX: {material_filters['MAKTX']}")
            else:
                print("⚠️ No data found for selected material filters")
                return pd.DataFrame()  # Return empty dataframe
        
        return df
    except FileNotFoundError:
        print(f"Error: The data file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading vendor data: {e}")
        return None