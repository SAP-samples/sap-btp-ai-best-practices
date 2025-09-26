import pandas as pd
import warnings
import os # Added for path joining
# Adjust import for top-level structure
# from config_chatbot import TABLES_PATH # Using hardcoded path below

# Define the path to the new tables directory
NEW_TABLES_PATH = 'new_tables/' # If running from resources
# NEW_TABLES_PATH = 'resources/new_tables/'  # If running from top directory

# Ignore specific pandas warnings if necessary (optional)
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(tables_path=NEW_TABLES_PATH): # Updated default path
    """Loads all required CSV files from the specified path and cleans them."""
    data = {}
    required_files = {
        'clientes': 'Cliente.csv',
        'unidades': 'Unidad.csv',
        'servicios': 'Servicios.csv',
        'operaciones': 'Operacion.csv',
        'campanas': 'Campanas.csv',
        'kits': 'Kits.csv',
        'materiales': 'Materiales.csv',
    }
    print(f"Attempting to load data from: {tables_path}")
    try:
        for key, filename in required_files.items():
            # Use os.path.join for better path handling
            full_path = os.path.join(tables_path, filename)
            print(f"Loading {key} from {full_path}...")
            try: # Wrap individual file loading in try-except
                if key == 'kits':
                    # Load Kits first
                    df = pd.read_csv(full_path)
                    # Clean PrecioKit: remove '$', ',', convert to float
                    # Using internal column name 'PrecioKit'
                    if 'PrecioKit' in df.columns:
                        df['PrecioKit'] = df['PrecioKit'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
                        # Coerce errors to NaN, then fill NaN with 0 or handle as needed
                        df['PrecioKit'] = pd.to_numeric(df['PrecioKit'], errors='coerce').fillna(0.0)
                        print(f"Cleaned PrecioKit in {filename}.")
                    else:
                        print(f"Warning: 'PrecioKit' column not found in {filename}.")

                    # Parse dates for Kits
                    # Using internal column names 'FechaInicio', 'FechaFin'
                    date_columns = ['FechaInicio', 'FechaFin']
                    valid_date_columns = [col for col in date_columns if col in df.columns]
                    for col in valid_date_columns:
                        df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')
                    data[key] = df

                elif key == 'unidades':
                    # Load Unidad
                    df = pd.read_csv(full_path)
                    # Clean Kilometraje: remove ',', convert to int
                    # Using internal column name 'Kilometraje'
                    if 'Kilometraje' in df.columns:
                        df['Kilometraje'] = df['Kilometraje'].astype(str).str.replace(',', '', regex=False).str.strip()
                        # Coerce errors to NaN, then fill NaN with 0 or handle as needed
                        df['Kilometraje'] = pd.to_numeric(df['Kilometraje'], errors='coerce').fillna(0).astype(int)
                        print(f"Cleaned Kilometraje in {filename}.")
                    else:
                         print(f"Warning: 'Kilometraje' column not found in {filename}.")
                    data[key] = df

                elif key == 'materiales':
                    # Load Materiales
                    df = pd.read_csv(full_path)
                    # Remove duplicates based on IdMaterial, keeping the first occurrence
                    # Using internal column name 'IdMaterial'
                    if 'IdMaterial' in df.columns:
                        initial_rows = len(df)
                        df.drop_duplicates(subset=['IdMaterial'], keep='first', inplace=True)
                        removed_count = initial_rows - len(df)
                        if removed_count > 0:
                            print(f"Removed {removed_count} duplicate rows based on IdMaterial in {filename}.")
                    else:
                        print(f"Warning: 'IdMaterial' column not found in {filename} for duplicate check.")
                    data[key] = df

                elif key == 'operaciones':
                    # Load Operacion
                    df = pd.read_csv(full_path)
                    # Remove duplicates based on IdOperacion, keeping the first occurrence
                    # Using internal column name 'IdOperacion'
                    if 'IdOperacion' in df.columns:
                        initial_rows = len(df)
                        df.drop_duplicates(subset=['IdOperacion'], keep='first', inplace=True)
                        removed_count = initial_rows - len(df)
                        if removed_count > 0:
                            print(f"Removed {removed_count} duplicate rows based on IdOperacion in {filename}.")
                    else:
                        print(f"Warning: 'IdOperacion' column not found in {filename} for duplicate check.")
                    data[key] = df

                elif key in ['servicios', 'campanas']:
                    # Handle date parsing for Servicios and Campanas
                    date_columns = []
                    if key == 'servicios':
                        # Using internal column names
                        date_columns = ['FechaApertura', 'FechaCierre', 'FechaFactura']
                    elif key == 'campanas':
                        # Using internal column names
                        date_columns = ['FechaInicio', 'FechaFin']

                    # Load the dataframe first
                    df = pd.read_csv(full_path)

                    valid_date_columns = [col for col in date_columns if col in df.columns]
                    # Explicitly parse dates with the correct format DD.MM.YYYY
                    for col in valid_date_columns:
                        df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')
                    data[key] = df

                else: # Load other files (like Cliente) without special cleaning/parsing
                    data[key] = pd.read_csv(full_path)

            except FileNotFoundError:
                 print(f"Error: File not found at {full_path}. Skipping {key}.")
                 # Optionally return None or raise an error if a file is critical
            except Exception as e:
                 print(f"Error loading or processing {filename}: {e}. Skipping {key}.")
                 # Optionally return None or raise an error

        # Final check if all required keys were loaded
        loaded_keys = data.keys()
        missing_keys = [k for k, f in required_files.items() if k not in loaded_keys]
        if missing_keys:
            print(f"Warning: Failed to load data for: {', '.join(missing_keys)}")
            # Decide if partial data is acceptable or return None
            # return None # Uncomment if all files are strictly required

        print("Data loading and cleaning process completed.") 
        # Example: Print shapes of cleaned dataframes
        # for name, df in data.items():
        #    print(f"Shape of {name}: {df.shape}")

    except Exception as e: # Catch errors during the loop setup itself
        print(f"An unexpected error occurred during data loading setup: {e}") 
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}") 
        return None
    return data

if __name__ == '__main__':
    # Example usage if run directly (for testing)
    print("Testing data loading from data_handler_chatbot.py...") 
    # Use the new default path for testing
    loaded_data = load_data()
    if loaded_data:
        print("\nData loaded successfully for testing.") 
        print("Available tables:", list(loaded_data.keys()))
        # Optional: Display head of cleaned dataframes
        # for name, df in loaded_data.items():
        #     print(f"\n--- {name} (head) ---")
        #     print(df.head())
    else:
        print("Data loading failed during testing.") 