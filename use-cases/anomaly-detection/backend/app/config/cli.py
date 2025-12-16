"""
Command line interface module for pharmaceutical anomaly detection.

This module handles parsing and validation of command line arguments.
"""

import argparse
from typing import Union


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for backend selection.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Pharmaceutical Anomaly Detection using Isolation Forest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # HANA ML + auto contamination (default)
    python main.py --backend sklearn                  # scikit-learn + auto contamination  
    python main.py --contamination 0.05               # HANA ML + 5% contamination
    python main.py --backend sklearn --contamination 0.1  # scikit-learn + 10% contamination
    python main.py --backend sklearn --shap           # scikit-learn + SHAP explanations
    python main.py --backend sklearn --customer-stratified  # Customer-aware anomaly detection
    python main.py --backend sklearn --customer-stratified --shap  # Customer-stratified + SHAP
    python main.py --backend hana --shap --contamination 0.05  # HANA + SHAP + 5% contamination
    python main.py --backend sklearn --load-models             # Load previously saved sklearn models
    python main.py --backend sklearn --customer-stratified --load-models  # Load stratified models
    python main.py --file /path/to/dataset.csv        # Use custom dataset file
    python main.py --backend sklearn --file /path/to/dataset.csv --shap  # Custom dataset + sklearn + SHAP
    python main.py --backend sklearn --n-estimators 400 --max-samples 2048  # Custom model parameters
    python main.py --backend sklearn --n-estimators 200 --max-samples auto --shap  # Custom parameters + SHAP
        """
    )
    
    parser.add_argument(
        '--backend', 
        choices=['hana', 'sklearn'],
        default='hana',
        help='Choose ML backend: hana (SAP HANA ML) or sklearn (scikit-learn). Default: hana'
    )
    
    parser.add_argument(
        '--contamination',
        type=str,
        default='auto',
        help='Contamination rate: "auto" for automatic detection or float value (e.g., 0.05 for 5%%). Default: auto'
    )
    
    parser.add_argument(
        '--shap',
        action='store_true',
        help='Generate SHAP explanations for all samples (slower but provides detailed feature attributions). Default: False'
    )
    
    parser.add_argument(
        '--customer-stratified',
        action='store_true',
        help='Use customer-stratified anomaly detection (trains separate models for different customer tiers). Default: False'
    )
    
    parser.add_argument(
        '--load-models',
        action='store_true',
        help='Load previously saved models instead of training new ones (much faster for prediction). Default: False'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Path to the CSV file to use as dataset. If not specified, uses the default dataset from settings.'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=None,
        help='Number of estimators (trees) for Isolation Forest. If not specified, uses environment variable N_ESTIMATORS or default 150.'
    )
    
    parser.add_argument(
        '--max-samples',
        type=str,
        default=None,
        help='Maximum number of samples per tree for Isolation Forest. Can be integer or "auto". If not specified, uses environment variable MAX_SAMPLES or default 512.'
    )
    
    return parser.parse_args()


def validate_contamination_rate(contamination_str: str) -> Union[str, float]:
    """
    Validate and convert contamination rate from string to appropriate type.
    
    Args:
        contamination_str: String representation of contamination rate
        
    Returns:
        Either 'auto' string or float value
        
    Raises:
        ValueError: If contamination rate is invalid
    """
    if contamination_str.lower() == 'auto':
        return 'auto'
    
    try:
        contamination_rate = float(contamination_str)
        if not (0 < contamination_rate < 1):
            raise ValueError("Contamination rate must be between 0 and 1")
        return contamination_rate
    except ValueError as e:
        raise ValueError(f"Invalid contamination rate '{contamination_str}'. {e}")