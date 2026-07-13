"""
Configuration settings for pharmaceutical anomaly detection.

This module contains all global constants and configuration parameters
used throughout the anomaly detection pipeline.
"""

# Data Configuration
CSV_FILENAME = 'datasets/merged_with_features_selected_ordered.csv'
RESULTS_DIR = 'anomaly_detection_results'
KEY_COL = 'Sales Document Number'  # Primary identifier
DATE_COL = 'Sales Document Created Date'
CUSTOMER_COL = 'Sold To number'  # Customer identifier

# Model Configuration
CONTAMINATION_RATE = 'auto'  # Auto-detect contamination rate (use 'auto' for both backends)
N_ESTIMATORS = 150           # Number of isolation trees
MAX_SAMPLES = 512            # Samples per tree
RANDOM_STATE = 42            # For reproducibility

# Customer Stratification Thresholds
LARGE_CUSTOMER_THRESHOLD = 100   # Orders needed to be considered a large customer
MEDIUM_CUSTOMER_THRESHOLD = 30   # Orders needed to be considered a medium customer (aligned with min training samples)
MIN_SAMPLES_FOR_TRAINING = 30    # Minimum samples needed to train a tier-specific model

# External Dependencies Availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# For HANA connection (optional - can run in standalone mode)
try:
    from dotenv import load_dotenv
    from hana_ml import ConnectionContext
    from hana_ml.dataframe import create_dataframe_from_pandas
    from hana_ml.algorithms.pal.preprocessing import IsolationForest as HanaIsolationForest
    HANA_AVAILABLE = True
    load_dotenv()
except ImportError:
    HANA_AVAILABLE = False