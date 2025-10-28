"""
Data loading and preprocessing module for pharmaceutical anomaly detection.

This module handles loading the dataset, basic preprocessing,
and train/test splitting based on temporal patterns.
"""

import pandas as pd
from typing import Tuple
from config.settings import CSV_FILENAME, DATE_COL, KEY_COL
from data.features import select_features


def load_and_preprocess_data(csv_file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Load the pharmaceutical dataset and perform initial preprocessing.
    
    Args:
        csv_file_path: Optional path to CSV file. If None, uses default from settings.
    
    Returns:
        tuple: (df_full, train_data, test_data, feature_columns)
    """
    print("Loading pharmaceutical dataset...")
    
    # Determine which file to use
    file_to_load = csv_file_path if csv_file_path else CSV_FILENAME
    
    # Load the dataset
    try:
        df = pd.read_csv(file_to_load)
        print(f"Successfully loaded dataset from {file_to_load}")
        print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_to_load}. Please check the file path.")
    
    # Convert date column to datetime - try multiple common formats
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce', dayfirst=False)
    
    # Handle missing values in date column
    missing_dates = df[DATE_COL].isnull().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} rows with missing dates will be excluded from time-based splitting")
        df = df.dropna(subset=[DATE_COL])
    
    # Check if we have any valid dates after parsing
    if len(df) == 0:
        raise ValueError("No valid dates found in the dataset. Please check the date format in the CSV file.")
    
    # Sort by date for proper train/test split
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    # Get date range
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Split data: last month for testing, rest for training
    test_cutoff_date = max_date - pd.DateOffset(months=1)
    
    train_data = df[df[DATE_COL] <= test_cutoff_date].copy()
    test_data = df[df[DATE_COL] > test_cutoff_date].copy()
    
    print(f"Training data: {train_data.shape[0]} rows (up to {test_cutoff_date.strftime('%Y-%m-%d')})")
    print(f"Test data: {test_data.shape[0]} rows (after {test_cutoff_date.strftime('%Y-%m-%d')})")
    
    # Define feature columns for anomaly detection
    feature_columns = select_features(df)
    
    return df, train_data, test_data, feature_columns