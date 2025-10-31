"""
Configuration settings for the Vendor Performance Dashboard application.
Contains constants, default values, and configuration parameters.
"""
import os

# File paths
TABLES_DIR = 'tables'
CONFIG_DIR = 'config'
COSTS_CONFIG_FILE = os.path.join(CONFIG_DIR, 'costs.json')

# Vendor data files (in order of priority)
VENDOR_FILES = [
    f'{TABLES_DIR}/vendor_with_direct_countries.csv',  # Use the fixed data file with country information
    f'{TABLES_DIR}/vendor_matnr_ranking_tariff_values.csv'  # Original data file (now using MATNR grouping)
]

# File encoding
FILE_ENCODING = 'utf-8-sig'

# Chart configuration
CHART_HEIGHT = 500
CHART_HEIGHT_LARGE = 600

# Chart color schemes
COLOR_SCALE_REVERSED = 'RdYlGn_r'  # Red to Green (reversed), lower values are green
COLOR_SCALE = 'RdYlGn'  # Red to Green, higher values are green
COLOR_DISCRETE = 'Set3'  # Qualitative color scale for categorical data

# Default display columns
DEFAULT_DISPLAY_COLUMNS = [
    'VendorFullID', 'EffectiveCostPerUnit_USD', 'TariffImpact_raw_percent', 
    'AvgLeadTimeDays_raw', 'OnTimeRate_raw', 'InFullRate_raw', 'POLineItemCount'
]

# Column renames for better display
COLUMN_RENAMES = {
    'VendorFullID': 'Vendor',
    'EffectiveCostPerUnit_USD': 'Effective Cost/Unit',
    'TariffImpact_raw_percent': 'Tariff Impact',
    'AvgLeadTimeDays_raw': 'Lead Time (Days)',
    'OnTimeRate_raw': 'On-Time Rate',
    'InFullRate_raw': 'In-Full Rate',
    'POLineItemCount': 'PO Count'
}

# Optimization settings
DEMAND_PERIOD_DAYS = 365  # Historical consumption period for demand calculation

# Application settings
PAGE_TITLE = "AI Supplier Sourcing Optimizer"
PAGE_ICON = "static/images/SAP_logo_square.png"