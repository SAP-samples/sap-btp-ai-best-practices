"""
Directory management utilities for pharmaceutical anomaly detection.

This module handles directory creation and result folder naming.
"""

import os
from typing import Union


def generate_results_folder_name(
    backend: str, 
    contamination: Union[str, float], 
    use_shap: bool, 
    customer_stratified: bool
) -> str:
    """
    Generate a unique results folder name based on configuration flags.
    
    Args:
        backend: ML backend ('hana' or 'sklearn')
        contamination: Contamination rate ('auto' or float value)
        use_shap: Whether SHAP explanations are enabled
        customer_stratified: Whether customer stratification is enabled
        
    Returns:
        Formatted folder name
    """
    folder_parts = ['anomaly_detection_results']
    
    # Add backend
    folder_parts.append(f"backend_{backend}")
    
    # Add contamination
    if contamination == 'auto':
        folder_parts.append("contamination_auto")
    else:
        # Convert float to string with underscore (e.g., 0.05 -> "0_05")
        contamination_str = str(contamination).replace('.', '_')
        folder_parts.append(f"contamination_{contamination_str}")
    
    # Add optional flags
    if customer_stratified:
        folder_parts.append("customer_stratified")
    
    if use_shap:
        folder_parts.append("shap")
    
    return '_'.join(folder_parts)


def setup_directories(results_dir: str) -> None:
    """
    Create necessary directories for outputs.
    
    Args:
        results_dir: Path to the results directory
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")