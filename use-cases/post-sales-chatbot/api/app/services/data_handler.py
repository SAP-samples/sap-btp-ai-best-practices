"""Data handler service for loading and managing CSV data."""

import pandas as pd
import warnings
import os
from typing import Dict, Optional

# Ignore specific pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataHandler:
    """Singleton class for managing CSV data loading and access."""

    _instance: Optional['DataHandler'] = None
    _data: Optional[Dict[str, pd.DataFrame]] = None
    _loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_tables_path(self) -> str:
        """Get the tables path, resolving relative to api/ directory."""
        from ..config import TABLES_PATH

        # Get the api/ directory (two levels up from this file)
        # This file is at: api/app/services/data_handler.py
        # api/ is at: ../../ from here
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.dirname(os.path.dirname(current_dir))

        return os.path.join(api_dir, TABLES_PATH)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required CSV files and clean them.

        Returns:
            Dictionary containing all loaded DataFrames.
        """
        if self._loaded and self._data is not None:
            return self._data

        tables_path = self._get_tables_path()
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

        for key, filename in required_files.items():
            full_path = os.path.join(tables_path, filename)
            print(f"Loading {key} from {full_path}...")

            try:
                if key == 'kits':
                    df = pd.read_csv(full_path)
                    # Clean PrecioKit: remove '$', ',', convert to float
                    if 'PrecioKit' in df.columns:
                        df['PrecioKit'] = df['PrecioKit'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
                        df['PrecioKit'] = pd.to_numeric(df['PrecioKit'], errors='coerce').fillna(0.0)
                        print(f"Cleaned PrecioKit in {filename}.")

                    # Parse dates for Kits
                    date_columns = ['FechaInicio', 'FechaFin']
                    valid_date_columns = [col for col in date_columns if col in df.columns]
                    for col in valid_date_columns:
                        df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')
                    data[key] = df

                elif key == 'unidades':
                    df = pd.read_csv(full_path)
                    # Clean Kilometraje: remove ',', convert to int
                    if 'Kilometraje' in df.columns:
                        df['Kilometraje'] = df['Kilometraje'].astype(str).str.replace(',', '', regex=False).str.strip()
                        df['Kilometraje'] = pd.to_numeric(df['Kilometraje'], errors='coerce').fillna(0).astype(int)
                        print(f"Cleaned Kilometraje in {filename}.")
                    data[key] = df

                elif key == 'materiales':
                    df = pd.read_csv(full_path)
                    # Remove duplicates based on IdMaterial
                    if 'IdMaterial' in df.columns:
                        initial_rows = len(df)
                        df.drop_duplicates(subset=['IdMaterial'], keep='first', inplace=True)
                        removed_count = initial_rows - len(df)
                        if removed_count > 0:
                            print(f"Removed {removed_count} duplicate rows based on IdMaterial in {filename}.")
                    data[key] = df

                elif key == 'operaciones':
                    df = pd.read_csv(full_path)
                    # Remove duplicates based on IdOperacion
                    if 'IdOperacion' in df.columns:
                        initial_rows = len(df)
                        df.drop_duplicates(subset=['IdOperacion'], keep='first', inplace=True)
                        removed_count = initial_rows - len(df)
                        if removed_count > 0:
                            print(f"Removed {removed_count} duplicate rows based on IdOperacion in {filename}.")
                    data[key] = df

                elif key in ['servicios', 'campanas']:
                    date_columns = []
                    if key == 'servicios':
                        date_columns = ['FechaApertura', 'FechaCierre', 'FechaFactura']
                    elif key == 'campanas':
                        date_columns = ['FechaInicio', 'FechaFin']

                    df = pd.read_csv(full_path)
                    valid_date_columns = [col for col in date_columns if col in df.columns]
                    for col in valid_date_columns:
                        df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')
                    data[key] = df

                else:
                    data[key] = pd.read_csv(full_path)

            except FileNotFoundError:
                print(f"Error: File not found at {full_path}. Skipping {key}.")
            except Exception as e:
                print(f"Error loading or processing {filename}: {e}. Skipping {key}.")

        # Check for missing keys
        loaded_keys = data.keys()
        missing_keys = [k for k in required_files.keys() if k not in loaded_keys]
        if missing_keys:
            print(f"Warning: Failed to load data for: {', '.join(missing_keys)}")

        print("Data loading and cleaning process completed.")

        self._data = data
        self._loaded = True
        return data

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Get loaded data, loading if necessary."""
        if not self._loaded or self._data is None:
            return self.load_data()
        return self._data

    def get_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Get a specific dataframe by name."""
        return self.data.get(name)

    def reload(self) -> Dict[str, pd.DataFrame]:
        """Force reload all data."""
        self._loaded = False
        self._data = None
        return self.load_data()


# Singleton instance
data_handler = DataHandler()
