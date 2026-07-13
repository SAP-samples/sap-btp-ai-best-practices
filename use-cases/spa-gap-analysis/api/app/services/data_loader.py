"""
Data Loader: Load Parquet files with LRU caching

Uses functools.lru_cache to avoid repeated file reads.
Cache is process-level (resets on API restart).
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Runtime data directory. Normal and anonymized ETL both write this directory.
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@lru_cache(maxsize=1)
def is_anonymized_runtime_data() -> bool:
    """Return True when app/data/processed contains public anonymized data."""
    return (DATA_DIR / "_anonymization_report.json").exists()


def load_from_parquet(filename: str) -> pd.DataFrame:
    """
    Generic parquet loader - loads file from processed directory

    Args:
        filename: Name of parquet file (e.g., 'material_savings.parquet')

    Returns:
        DataFrame loaded from file
    """
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    logger.info(f"Loading {filename}")
    return pd.read_parquet(file_path)


@lru_cache(maxsize=1)
def load_customer_master() -> pd.DataFrame:
    """
    Load Customer Master with geography enrichment

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with customer profiles
    """
    file_path = DATA_DIR / "customer_master.parquet"
    logger.info(f"Loading customer_master from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} customer records")
    return df


@lru_cache(maxsize=1)
def load_transactions() -> pd.DataFrame:
    """
    Load S712 transactions

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with transaction history
    """
    file_path = DATA_DIR / "s712_transactions.parquet"
    logger.info(f"Loading transactions from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} transaction records")
    return df


@lru_cache(maxsize=1)
def load_qualifications() -> pd.DataFrame:
    """
    Load A701 Qualifications (Customer-SPA relationships)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with qualification data
    """
    file_path = DATA_DIR / "a701_qualifications.parquet"
    logger.info(f"Loading qualifications from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} qualification records")
    return df


@lru_cache(maxsize=1)
def load_customer_spa_assignments() -> pd.DataFrame:
    """
    Load canonical customer-SPA assignments.

    Returns:
        DataFrame with one active/inactive assignment row per customer-SPA pair
    """
    file_path = DATA_DIR / "customer_spa_assignments.parquet"
    logger.info(f"Loading customer SPA assignments from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} customer SPA assignments")
    return df


@lru_cache(maxsize=1)
def load_customer_current_metrics() -> pd.DataFrame:
    """
    Load canonical current-state customer metrics.

    Returns:
        DataFrame with Q4 rolling-12M current coverage and savings metrics
    """
    file_path = DATA_DIR / "customer_current_metrics.parquet"
    logger.info(f"Loading customer current metrics from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded current metrics for {len(df)} customers")
    return df


@lru_cache(maxsize=1)
def load_customer_material_current_pricing() -> pd.DataFrame:
    """
    Load canonical customer-material current pricing rows.

    Returns:
        DataFrame with customer-material pricing under current assigned SPAs
    """
    file_path = DATA_DIR / "customer_material_current_pricing.parquet"
    logger.info(f"Loading customer material current pricing from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} customer-material current pricing rows")
    return df


@lru_cache(maxsize=1)
def load_spa_guide_metadata() -> pd.DataFrame:
    """
    Load SPA Guide metadata used for eligibility/explainability.

    Returns:
        DataFrame with agreement area, plant, vendor, SPA type, category metadata
    """
    file_path = DATA_DIR / "spa_guide_metadata.parquet"
    logger.info(f"Loading SPA Guide metadata from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} SPA Guide metadata rows")
    return df


@lru_cache(maxsize=1)
def load_rfm_scores() -> pd.DataFrame:
    """
    Load RFM scores (Recency, Frequency, Monetary)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with RFM segmentation
    """
    file_path = DATA_DIR / "rfm_scores.parquet"
    logger.info(f"Loading RFM scores from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded RFM scores for {len(df)} customers")
    return df


@lru_cache(maxsize=1)
def load_customer_cogs() -> pd.DataFrame:
    """
    Load aggregated customer COGS

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with customer spending summary
    """
    file_path = DATA_DIR / "customer_cogs.parquet"
    logger.info(f"Loading customer COGS from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded COGS data for {len(df)} customers")
    return df


@lru_cache(maxsize=1)
def load_materials() -> pd.DataFrame:
    """
    Load A901 Materials (Material-SPA mappings)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with material-SPA coverage
    """
    file_path = DATA_DIR / "a901_materials.parquet"
    logger.info(f"Loading materials from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} material records")
    return df


@lru_cache(maxsize=1)
def load_sap_master() -> pd.DataFrame:
    """
    Load SAP Master (Material catalog)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with material catalog
    """
    file_path = DATA_DIR / "sap_master.parquet"
    logger.info(f"Loading SAP Master from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} material catalog records")
    return df


@lru_cache(maxsize=1)
def load_header_data() -> pd.DataFrame:
    """
    Load HEADER data (SPA master list)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with SPA definitions
    """
    file_path = DATA_DIR / "header_data.parquet"
    logger.info(f"Loading HEADER data from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} SPA header records")
    return df


@lru_cache(maxsize=1)
def load_area_plt_mapping() -> pd.DataFrame:
    """
    Load Area-PLT mapping (Plant → City/State)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with geographic mappings
    """
    file_path = DATA_DIR / "area_plt_mapping.parquet"
    logger.info(f"Loading Area-PLT mapping from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} plant mappings")
    return df


@lru_cache(maxsize=1)
def load_product_hierarchy() -> Optional[pd.DataFrame]:
    """
    Load Product Hierarchy (optional)

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with product hierarchy, or None if not available
    """
    file_path = DATA_DIR / "product_hierarchy.parquet"

    if not file_path.exists():
        logger.warning("Product Hierarchy file not found")
        return None

    logger.info(f"Loading Product Hierarchy from {file_path}")
    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} product hierarchy records")
    return df


@lru_cache(maxsize=1)
def load_a703_nets() -> pd.DataFrame:
    """
    Load A703-nets (SPA Materials linkage)

    Links SPAs to Materials with pricing

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with SPA-Material linkages
    """
    file_path = DATA_DIR / "a703_nets.parquet"
    logger.info(f"Loading A703-nets from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} A703-nets records")
    return df


@lru_cache(maxsize=1)
def load_a700_spa_vendors() -> pd.DataFrame:
    """
    Load A700 SPA Vendors (Vendor information per SPA)

    Links SPAs to Vendors

    Cached with LRU (loads only once per process)

    Returns:
        DataFrame with SPA-Vendor linkages
    """
    file_path = DATA_DIR / "a700_spa_vendors.parquet"
    logger.info(f"Loading A700 SPA Vendors from {file_path}")

    df = pd.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} A700 SPA Vendors records")
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all datasets at once

    Useful for initialization or testing

    Returns:
        Dictionary of all DataFrames keyed by name
    """
    logger.info("Loading all datasets")

    datasets = {
        'customer_master': load_customer_master(),
        'transactions': load_transactions(),
        'qualifications': load_qualifications(),
        'customer_spa_assignments': load_customer_spa_assignments(),
        'customer_current_metrics': load_customer_current_metrics(),
        'customer_material_current_pricing': load_customer_material_current_pricing(),
        'rfm_scores': load_rfm_scores(),
        'customer_cogs': load_customer_cogs(),
        'materials': load_materials(),
        'sap_master': load_sap_master(),
        'header_data': load_header_data(),
        'area_plt_mapping': load_area_plt_mapping(),
        'a703_nets': load_a703_nets(),  # NEW
        'a700_spa_vendors': load_a700_spa_vendors()  # NEW
    }

    # Optional dataset
    product_hierarchy = load_product_hierarchy()
    if product_hierarchy is not None:
        datasets['product_hierarchy'] = product_hierarchy

    logger.info(f"Loaded {len(datasets)} datasets")

    return datasets


def clear_cache():
    """
    Clear LRU cache (force reload on next call)

    Use when data files are updated
    """
    logger.info("Clearing data loader cache")

    load_customer_master.cache_clear()
    load_transactions.cache_clear()
    load_qualifications.cache_clear()
    load_customer_spa_assignments.cache_clear()
    load_customer_current_metrics.cache_clear()
    load_customer_material_current_pricing.cache_clear()
    load_rfm_scores.cache_clear()
    load_customer_cogs.cache_clear()
    load_materials.cache_clear()
    load_sap_master.cache_clear()
    load_header_data.cache_clear()
    load_area_plt_mapping.cache_clear()
    load_product_hierarchy.cache_clear()
    load_a703_nets.cache_clear()
    load_a700_spa_vendors.cache_clear()

    logger.info("Cache cleared")
