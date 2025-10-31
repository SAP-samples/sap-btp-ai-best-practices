"""
Tariff Configuration UI Component.

This module provides the UI for setting country-specific tariff values.
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from subprocess import run
from datetime import datetime

# Updated imports for new structure
from optimization.profile_manager import ProfileManager

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_country_data():
    """Load country data from SAP_VLY_IL_COUNTRY.csv"""
    try:
        filepath = os.path.join('tables', 'SAP_VLY_IL_COUNTRY.csv')
        country_df = pd.read_csv(
            filepath, 
            encoding='utf-8-sig',
            keep_default_na=False,
            dtype={'LAND1': str, 'LANDX': str}
        )
        
        # Clean the data - trim whitespace and handle missing values
        country_df['LAND1'] = country_df['LAND1'].astype(str).str.strip()
        country_df['LANDX'] = country_df['LANDX'].astype(str).str.strip()
        
        # Remove any rows where country code is empty, 'nan', or 'None'
        country_df = country_df[
            (country_df['LAND1'] != '') & 
            (country_df['LAND1'].str.lower() != 'nan') &
            (country_df['LAND1'].str.lower() != 'none')
        ]
        
        # Create a clean dataframe with ISO country code and name
        countries = pd.DataFrame({
            'code': country_df['LAND1'],
            'name': country_df['LANDX']
        }).drop_duplicates(subset=['code']).sort_values('name')
        
        # Verify Namibia is present
        namibia_check = countries[countries['code'] == 'NA']
        if namibia_check.empty:
            print("WARNING: Namibia (NA) not found in country data!")
            print(f"Available codes starting with 'N': {countries[countries['code'].str.startswith('N', na=False)]['code'].tolist()}")
        else:
            print(f"✓ Namibia found: {namibia_check.iloc[0].to_dict()}")
        
        return countries
    except Exception as e:
        st.error(f"Error loading country data: {e}")
        import traceback
        print(f"Country data loading error: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=['code', 'name'])

def load_tariff_values(profile_id='profile_1'):
    """Load saved tariff values from JSON file"""
    profile_manager = ProfileManager(".")
    tariff_path = profile_manager.get_data_file_path(profile_id, 'tariff_values.json')
    
    if os.path.exists(tariff_path):
        try:
            with open(tariff_path, 'r') as f:
                tariff_values = json.load(f)
            return tariff_values
        except Exception as e:
            st.error(f"Error loading tariff values: {e}")
    
    # Return empty dictionary if file doesn't exist or there's an error
    return {}

def save_tariff_values(tariff_values, profile_id='profile_1'):
    """Save tariff values to JSON file"""
    profile_manager = ProfileManager(".")
    tariff_path = profile_manager.get_data_file_path(profile_id, 'tariff_values.json')
    os.makedirs(os.path.dirname(tariff_path), exist_ok=True)
    
    try:
        with open(tariff_path, 'w') as f:
            json.dump(tariff_values, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving tariff values: {e}")
        return False


def validate_tariff_csv(uploaded_df):
    """
    Validate uploaded CSV format and data integrity
    
    Args:
        uploaded_df (pd.DataFrame): The uploaded CSV as a DataFrame
        
    Returns:
        tuple: (is_valid, error_messages, validated_data)
    """
    error_messages = []
    validated_data = {}
    
    # Check required columns
    required_columns = ['Country Code', 'Country Name', 'Tariff Percentage']
    missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
    
    if missing_columns:
        error_messages.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, error_messages, {}
    
    # Load country master data for validation
    countries = load_country_data()
    valid_country_codes = countries['code'].tolist() if not countries.empty else []
    
    # Validate each row
    for index, row in uploaded_df.iterrows():
        country_code = str(row['Country Code']).strip().upper()
        tariff_percentage = str(row['Tariff Percentage']).strip()
        
        # Validate country code
        if country_code not in valid_country_codes and valid_country_codes:
            error_messages.append(f"Row {index + 1}: Invalid country code '{country_code}'")
            continue
        
        # Validate and convert tariff percentage
        try:
            # Remove % symbol if present and convert to float
            tariff_value = tariff_percentage.replace('%', '').strip()
            tariff_float = float(tariff_value)
            
            # Check for reasonable range (0-100%)
            if tariff_float < 0:
                error_messages.append(f"Row {index + 1}: Negative tariff value '{tariff_float}%' for {country_code}")
                continue
            elif tariff_float > 100:
                error_messages.append(f"Row {index + 1}: Tariff value '{tariff_float}%' exceeds 100% for {country_code}")
                continue
                
            validated_data[country_code] = tariff_float
            
        except ValueError:
            error_messages.append(f"Row {index + 1}: Invalid tariff percentage '{tariff_percentage}' for {country_code}")
            continue
    
    # Check if any valid data was found
    if not validated_data and not error_messages:
        error_messages.append("No valid tariff data found in the uploaded file")
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages, validated_data

def process_uploaded_tariff_csv(uploaded_file, profile_id='profile_1'):
    """
    Process uploaded CSV file and convert to internal tariff format
    
    Args:
        uploaded_file: Streamlit uploaded file object
        profile_id (str): Profile ID to apply changes to (for future use)
        
    Returns:
        tuple: (success, message, preview_data)
    """
    # Note: profile_id is kept for future extensibility
    try:
        # Read the uploaded CSV
        uploaded_df = pd.read_csv(uploaded_file)
        
        # Validate the CSV
        is_valid, error_messages, validated_data = validate_tariff_csv(uploaded_df)
        
        if not is_valid:
            return False, f"Validation failed:\n" + "\n".join(error_messages), None
        
        # Create preview data for user confirmation
        countries = load_country_data()
        preview_data = []
        
        for country_code, tariff_value in validated_data.items():
            country_name = countries[countries['code'] == country_code]['name'].iloc[0] if not countries[countries['code'] == country_code].empty else country_code
            preview_data.append({
                'Country Code': country_code,
                'Country Name': country_name,
                'Tariff Percentage': f"{tariff_value}%"
            })
        
        preview_df = pd.DataFrame(preview_data).sort_values('Country Name')
        
        return True, f"Successfully validated {len(validated_data)} tariff settings", {
            'validated_data': validated_data,
            'preview_df': preview_df
        }
        
    except Exception as e:
        return False, f"Error processing uploaded file: {str(e)}", None

def get_country_tariff_stats(profile_id='profile_1'):
    """Get statistics on tariff values by country from vendor data"""
    try:
        profile_manager = ProfileManager(".")
        vendor_file = profile_manager.get_data_file_path(profile_id, 'vendor_matnr_ranking_tariff_values.csv')
        mapping_file = profile_manager.get_data_file_path(profile_id, 'vendor_with_direct_countries.csv')
        
        if not os.path.exists(vendor_file):
            print(f"Warning: Vendor file not found at {vendor_file}")
            return pd.DataFrame(columns=['Country Code', 'Country Name', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count'])
        
        # Load vendor data
        df = pd.read_csv(vendor_file, low_memory=False)
        print(f"Loaded vendor data from {vendor_file}: {len(df)} rows")
        
        # Verify TariffImpact_raw_percent column exists
        if 'TariffImpact_raw_percent' not in df.columns:
            print(f"Warning: TariffImpact_raw_percent column not found in {vendor_file}")
            print(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=['Country Code', 'Country Name', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count'])
        
        # Check if country data is already in the vendor file
        if 'LAND1' not in df.columns and os.path.exists(mapping_file):
            # Load mapping and merge
            mapping = pd.read_csv(mapping_file)
            print(f"Loaded country mapping from {mapping_file}: {len(mapping)} rows")
            df = pd.merge(df, mapping[['LIFNR', 'LAND1', 'Country']], on='LIFNR', how='left')
            print(f"Merged vendor data with country mapping. LAND1 present: {'LAND1' in df.columns}")
        
        if 'LAND1' not in df.columns:
            print("Warning: LAND1 column not available after attempted merge")
            return pd.DataFrame(columns=['Country Code', 'Country Name', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count'])
        
        # Show tariff distribution for debugging
        tariff_min = df['TariffImpact_raw_percent'].min() if not df.empty else 0
        tariff_max = df['TariffImpact_raw_percent'].max() if not df.empty else 0
        tariff_mean = df['TariffImpact_raw_percent'].mean() if not df.empty else 0
        print(f"Tariff statistics: Min={tariff_min:.2f}%, Max={tariff_max:.2f}%, Mean={tariff_mean:.2f}%")
        
        # Calculate stats by country
        groupby_cols = ['LAND1', 'Country'] if 'Country' in df.columns else ['LAND1']
        stats = df.groupby(groupby_cols).agg(
            AvgTariff=('TariffImpact_raw_percent', 'mean'),
            MinTariff=('TariffImpact_raw_percent', 'min'),
            MaxTariff=('TariffImpact_raw_percent', 'max'),
            VendorCount=('LIFNR', 'nunique')
        ).reset_index()
        
        print(f"Generated tariff statistics for {len(stats)} countries")
        
        # Rename columns for display
        if 'Country' in df.columns:
            stats.columns = ['Country Code', 'Country Name', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count']
        else:
            stats.columns = ['Country Code', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count']
        
        # If country name is missing, add it from country data
        if 'Country Name' not in stats.columns:
            countries = load_country_data()
            stats = pd.merge(stats, 
                            countries.rename(columns={'code': 'Country Code', 'name': 'Country Name'}),
                            on='Country Code', how='left')
            stats['Country Name'] = stats['Country Name'].fillna('Unknown')
        
        # Format tariff values as percentages and round to 1 decimal place for readability
        for col in ['Avg Tariff', 'Min Tariff', 'Max Tariff']:
            stats[col] = stats[col].round(1)
        
        # Print top 5 countries by average tariff for verification
        top_tariffs = stats.sort_values('Avg Tariff', ascending=False).head(5)
        if not top_tariffs.empty:
            print("Top 5 countries by average tariff:")
            for _, row in top_tariffs.iterrows():
                print(f"  {row['Country Code']} ({row['Country Name']}): {row['Avg Tariff']}%")
        
        return stats.sort_values('Country Name')
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Error getting country tariff stats: {e}")
        return pd.DataFrame(columns=['Country Code', 'Country Name', 'Avg Tariff', 'Min Tariff', 'Max Tariff', 'Vendor Count'])

def run_fix_country_mapping():
    """Run the fix_country_mapping.py script to ensure country data is available"""
    try:
        print("Running fix_country_mapping.py to ensure country data is properly linked...")
        # Updated path to scripts directory
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'fix_country_mapping.py')
        if not os.path.exists(script_path):
            return False, f"Country mapping script not found at {script_path}"
            
        # Run the script
        process = run(["python", script_path], capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Warning: Country mapping script failed: {process.stderr}")
            return False, f"Country mapping failed: {process.stderr}"
            
        print("Country mapping completed successfully.")
        return True, "Country mapping completed successfully."
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running country mapping: {str(e)}")
        return False, f"Error running country mapping: {str(e)}"

def run_optimization_pipeline_script(profile_id='profile_1'):
    """Run the optimization pipeline shell script"""
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_optimization_pipeline.sh')
        if not os.path.exists(script_path):
            return False, f"Optimization script not found at {script_path}"
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run script with profile parameter
        process = run([script_path, profile_id], capture_output=True, text=True)
        
        if process.returncode != 0:
            return False, f"Optimization pipeline failed with error: {process.stderr}"
        
        # Ensure country mapping is updated
        mapping_success, mapping_message = run_fix_country_mapping()
        if not mapping_success:
            print(f"Warning: Country mapping step failed: {mapping_message}")
            # Continue despite mapping failure, since optimization was successful
            return True, "Optimization completed, but country mapping failed. Some visualizations may not display correctly."
        
        return True, "Optimization completed successfully. Refresh the Dashboard."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error running optimization pipeline: {str(e)}"

def refresh_data(profile_id='profile_1'):
    """Signal that the data should be refreshed"""
    # Create or update a file with the current timestamp (global for now)
    refresh_path = os.path.join('tables', '.refresh')
    with open(refresh_path, 'w') as f:
        f.write(str(int(time.time())))
    
    # Clear Streamlit's cache - including country data cache
    st.cache_data.clear()
    
    # Clear the specific country data cache if it exists
    try:
        load_country_data.clear()
    except:
        pass  # Cache might not exist yet
    
    # Update or invalidate any session state variables that cache data
    if 'vendor_data' in st.session_state:
        del st.session_state.vendor_data
    
    # Force update of tariff_mtime to ensure data reloading (profile-specific)
    profile_manager = ProfileManager(".")
    tariff_path = profile_manager.get_data_file_path(profile_id, 'tariff_values.json')
    if os.path.exists(tariff_path):
        st.session_state.tariff_mtime = os.path.getmtime(tariff_path)

def clear_country_data_cache():
    """Clear the country data cache to force reload"""
    try:
        load_country_data.clear()
        st.cache_data.clear()
        print("✓ Country data cache cleared")
    except Exception as e:
        print(f"Note: Cache clearing failed (this is normal): {e}")

def render_tariff_configuration_tab(tab, profile_id='profile_1'):
    """Render the tariff configuration tab"""
    with tab:
        st.subheader("Country Tariff Configuration")
        
        # Initialize ProfileManager
        profile_manager = ProfileManager(".")
        
        # Display current profile information
        st.info(f"Currently configuring: **{profile_id}**")
        
        # Get tariff statistics by country from vendor data
        tariff_stats = get_country_tariff_stats(profile_id)
        have_stats = not tariff_stats.empty
        
        # Load country data
        countries = load_country_data()
        
        if countries.empty:
            st.error("Could not load country data. Please check the file path.")
            return
        
        # Load saved tariff values
        tariff_values = load_tariff_values(profile_id)
        filtered_countries = countries.copy()  # Start with all countries

        # Create two columns for the UI
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Country List")
            if not filtered_countries.empty:
                # Create a selection grid for countries
                selected_country = st.selectbox(
                    "Select a country to configure tariff:",
                    options=filtered_countries['name'].tolist(),
                    index=0 if not filtered_countries.empty else None
                )
                
                # Get the country code for the selected country
                selected_code = filtered_countries[filtered_countries['name'] == selected_country]['code'].iloc[0]
                
                # Display information about the selected country
                with st.expander(f"Country Information: {selected_country} ({selected_code})"):
                    # Show country stats if available
                    if have_stats and selected_code in tariff_stats['Country Code'].values:
                        country_row = tariff_stats[tariff_stats['Country Code'] == selected_code].iloc[0]
                        st.markdown(f"**Current Tariff Statistics:**")
                        st.markdown(f"- Tariff Value: **{country_row['Avg Tariff']:.2f}%**")
                        st.markdown(f"- Vendor Count: {int(country_row['Vendor Count'])}")
                    else:
                        st.markdown(f"No tariff data available for {selected_country}")
            else:
                st.warning("No countries match your search criteria.")
        
        with col2:
            st.subheader("Tariff Configuration")
            if 'selected_country' in locals() and selected_country:
                # Get current tariff value
                # Priority: 1. Average from vendor tariff stats, 2. Saved value in JSON, 3. Default 0.0
                if have_stats and selected_code in tariff_stats['Country Code'].values:
                    # First priority: Use average from vendor stats
                    avg_tariff = float(tariff_stats[tariff_stats['Country Code'] == selected_code]['Avg Tariff'].iloc[0])
                    # Second priority: Use saved values from JSON if they exist
                    current_tariff = tariff_values.get(selected_code, avg_tariff)
                else:
                    # Third priority: Fallback to 0.0 if no stats or saved values
                    current_tariff = tariff_values.get(selected_code, 0.0)
                
                # Allow the user to modify the tariff value (no upper limit)
                new_tariff = st.number_input(
                    f"Tariff percentage for {selected_country} ({selected_code}):",
                    min_value=0.0,
                    value=float(current_tariff),
                    step=0.1,
                    format="%.1f",
                    help="Enter the tariff percentage (0 or above). This will be applied to all suppliers from this country."
                )
                
                # Update button
                if st.button(f"Update Tariff for {selected_country}"):
                    tariff_values[selected_code] = new_tariff
                    if save_tariff_values(tariff_values, profile_id):
                        st.success(f"Tariff for {selected_country} updated to {new_tariff}%")
                    else:
                        st.error("Failed to save tariff value")
        
        # Display current tariff settings
        st.subheader("Current Tariff Settings")
        
        if tariff_values:
            # Convert to dataframe for display
            tariff_df = pd.DataFrame([
                {"Country Code": code, "Country Name": countries[countries['code'] == code]['name'].iloc[0] if not countries[countries['code'] == code].empty else code, 
                 "Tariff Percentage": f"{value}%"}
                for code, value in tariff_values.items()
            ]).sort_values("Country Name")
            
            st.dataframe(tariff_df, use_container_width=True)
            
            # CSV Download and Refresh section
            col_download, col_refresh = st.columns(2)
            
            with col_download:
                # Generate CSV button
                csv = tariff_df.to_csv(index=False)
                st.download_button(
                    label="Download Current Settings",
                    data=csv,
                    file_name=f"tariff_settings_{profile_id}.csv",
                    mime="text/csv",
                    help="Download current tariff settings as CSV"
                )
                
            with col_refresh:
                # Cache refresh button
                if st.button("Refresh Data", help="Clear cache and reload country data"):
                    clear_country_data_cache()
                    st.success("Data cache cleared! Download will use fresh data.")
                    st.rerun()
        else:
            st.info("No tariff settings configured yet. Select a country above to set tariff values.")
            
            # Provide refresh functionality when no tariffs are set
            if st.button("Refresh Data", help="Clear cache and reload country data", key="refresh_no_tariffs"):
                clear_country_data_cache()
                st.success("Data cache cleared!")
                st.rerun()
        
        # CSV Upload section
        st.markdown("---")
        st.subheader("Upload Tariff Settings")
        
        with st.expander("Upload Instructions", expanded=False):
            st.markdown("""
            **Upload a CSV file to bulk update tariff settings:**
            
            1. **Required Format:** CSV file with exactly these columns:
               - `Country Code` (e.g., US, CN, DE)
               - `Country Name` (e.g., United States, China, Germany)
               - `Tariff Percentage` (e.g., 10.5% or 10.5)
            
            2. **Data Requirements:**
               - Country codes must match existing country master data
               - Tariff percentages must be numeric (0-100%)
               - Percentage symbols (%) are optional and will be removed automatically
            
            3. **Processing:**
               - File will be validated before applying changes
               - You'll see a preview of changes before confirmation
               - Only valid rows will be processed
               - Invalid rows will be reported as errors
            
            **Tip:** Download your current settings to use as a template for the correct format
            """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with tariff settings",
            key=f"tariff_upload_{profile_id}"
        )
        
        if uploaded_file is not None:
            # Process the uploaded file
            success, message, preview_data = process_uploaded_tariff_csv(uploaded_file, profile_id)
            
            if success and preview_data:
                st.success(message)
                
                # Show preview of changes
                st.subheader("Preview of Changes")
                st.dataframe(preview_data['preview_df'], use_container_width=True)
                
                # Confirmation controls
                col_confirm, col_cancel = st.columns(2)
                
                with col_confirm:
                    if st.button(
                        f"Apply Changes ({len(preview_data['validated_data'])} countries)",
                        type="primary",
                        key=f"apply_upload_{profile_id}"
                    ):
                        # Apply the changes
                        if save_tariff_values(preview_data['validated_data'], profile_id):
                            st.success(f"Successfully updated tariff settings for {len(preview_data['validated_data'])} countries!")
                            st.info("Changes applied. You may need to run optimization to see the effects.")
                            # Force reload the page to show updated values
                            st.rerun()
                        else:
                            st.error("Failed to save uploaded tariff settings")
                
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_upload_{profile_id}"):
                        st.info("Upload cancelled. No changes were made.")
                        st.rerun()
                        
            else:
                st.error("Upload Failed")
                st.error(message)
                
                # Show file contents for debugging
                if st.checkbox("Show file contents for debugging"):
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        debug_df = pd.read_csv(uploaded_file)
                        st.subheader("File Contents:")
                        st.dataframe(debug_df)
                        st.subheader("File Info:")
                        st.write(f"Shape: {debug_df.shape}")
                        st.write(f"Columns: {list(debug_df.columns)}")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
        
        if tariff_values:
            # Clear all tariffs button
            if st.button("Clear All Tariff Settings"):
                confirm = st.checkbox("I understand this will remove all tariff settings")
                if confirm:
                    tariff_values = {}
                    if save_tariff_values(tariff_values, profile_id):
                        st.success("All tariff settings cleared")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to clear tariff settings")
        else:
            st.info("No tariff settings configured yet. Select a country above to set tariff values.")
        
        # Run Optimization Pipeline section
        st.markdown("---")
        st.subheader("Run Procurement Optimization Pipeline")
        
        with st.expander("About the Optimization Pipeline"):
            st.markdown("""
            This process will:
            1. Update vendor performance metrics with the new tariff configuration
            2. Run the optimization algorithm to select the best vendors based on the updated metrics
            3. Generate a comparison between historical and optimized procurement
            4. Update the dashboard with the new data
            
            **Note:** This process may take several minutes to complete. After completion, all dashboard visualizations will automatically update with the new data.
            """)
        
        # Create a single button to update tariffs and run optimization
        if st.button("Update Tariffs & Run Optimization", type="primary"):
            if not tariff_values and not have_stats:
                st.warning("No tariff settings configured. The optimization will use 0% tariffs for all countries.")
                run_confirm = st.checkbox("Run anyway")
                if not run_confirm:
                    return
            
            # Skip regenerating default tariffs to prevent overwriting user changes
            st.info("Using current tariff settings for optimization...")
            
            # Force delete ALL optimization-related files to ensure everything gets regenerated
            st.info("Cleaning profile-specific optimization files...")
            profile_manager.clean_profile_data(profile_id)
            
            # Create a backup directory in the global tables directory 
            backup_dir = os.path.join('tables', 'backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(backup_dir, exist_ok=True)
            
            # Run the optimization pipeline
            with st.spinner("Running optimization pipeline with updated tariffs..."):
                success, message = run_optimization_pipeline_script(profile_id)
                
                if success:
                    st.success(f"{message}")
                    
                    # Double-check that vendor_with_direct_countries.csv exists in the profile
                    mapping_file = profile_manager.get_data_file_path(profile_id, 'vendor_with_direct_countries.csv')
                    if not os.path.exists(mapping_file):
                        st.warning("Country mapping file is missing. Running fix_country_mapping.py...")
                        # Run the fix_country_mapping.py script directly
                        mapping_success, mapping_message = run_fix_country_mapping()
                        if mapping_success:
                            st.success("Country mapping updated successfully.")
                        else:
                            st.warning(f"Country mapping issue: {mapping_message}")
                    
                    # Mark data for refresh
                    refresh_data(profile_id)
                    # Force reload by clearing the cache and session state
                    st.cache_data.clear()
                    if 'vendor_data' in st.session_state:
                        del st.session_state.vendor_data
                    if 'tariff_mtime' in st.session_state:
                        del st.session_state.tariff_mtime
                        
                    # Instruct the user to refresh
                    st.info("The tariff changes have been applied and optimization has been run.")
                    st.info("All dashboard tabs should now reflect the updated tariff values.")
                    # Create a button to manually refresh the dashboard
                    if st.button("Refresh Dashboard", type="primary"):
                        st.experimental_rerun()
                else:
                    st.error(f"{message}")
                    with st.expander("Error Details"):
                        st.code(message)

# Paths and configuration 
def run_optimization_pipeline():
    """
    Run the procurement optimization pipeline with current tariff settings.
    This is a wrapper around run_optimization_pipeline_script.
    """
    return run_optimization_pipeline_script()


def render_tariff_configuration_tab_standalone():
    """Standalone version for lazy loading"""
    class DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            # Args intentionally unused for context manager protocol
            pass
    render_tariff_configuration_tab(DummyTab())