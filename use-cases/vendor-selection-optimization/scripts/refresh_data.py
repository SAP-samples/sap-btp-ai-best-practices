"""
Utility script to force refresh the dashboard data cache.
This can be useful when changes to data files aren't being reflected in the dashboard.
"""

import os
import time
import argparse
import subprocess

def run_fix_country_mapping():
    """Run fix_country_mapping.py to regenerate country mapping"""
    print("Running fix_country_mapping.py...")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fix_country_mapping.py')
    if os.path.exists(script_path):
        try:
            result = subprocess.run(['python', script_path], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print("Country mapping regenerated successfully.")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running fix_country_mapping.py: {e}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            return False
    else:
        print(f"Country mapping script not found at {script_path}")
        return False

def refresh_data():
    """Force refresh of dashboard data"""
    # Update the .refresh file timestamp
    refresh_path = os.path.join('tables', '.refresh')
    with open(refresh_path, 'w') as f:
        f.write(str(int(time.time())))
    print(f"Created refresh marker at {refresh_path}")
    
    # Touch key data files to update their timestamps
    files_to_touch = [
        os.path.join('tables', 'tariff_values.json'),
        os.path.join('tables', 'vendor_maktx_ranking_tariff_values.csv'),
        os.path.join('tables', 'vendor_with_direct_countries.csv')
    ]
    
    for file_path in files_to_touch:
        if os.path.exists(file_path):
            # Update the modification time to current time
            os.utime(file_path, None)
            print(f"Updated timestamp for {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
            
    print("Data refresh markers created. Restart the dashboard or refresh the browser to see changes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh dashboard data cache")
    parser.add_argument('--fix-mapping', action='store_true', help='Regenerate country mapping')
    args = parser.parse_args()
    
    if args.fix_mapping:
        run_fix_country_mapping()
        
    refresh_data()
    print("Done! The dashboard should now display fresh data on next load.")