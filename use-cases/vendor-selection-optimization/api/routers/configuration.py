"""Configuration Management API Router"""

import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
import logging

from ..config import settings
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["configuration"]
)


# Dependency to validate profile exists
def validate_profile(profile_id: str) -> str:
    """Validate that profile exists"""
    if not file_manager.profile_exists(profile_id):
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{profile_id}' not found"
        )
    return profile_id


@router.get("/profiles/{profile_id}/config")
async def get_profile_configuration(
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """
    Get all configuration for a profile.
    
    Returns:
    - costs.json: Cost components and economic impact parameters
    - tariff_values.json: Tariff configuration
    - table_map.json: Table mappings
    - column_map.json: Column mappings
    """
    config_dir = file_manager.get_profile_config_dir(profile_id)
    tables_dir = file_manager.get_profile_tables_dir(profile_id)
    
    configuration = {}
    
    # Load costs configuration
    costs_path = config_dir / "costs.json"
    if costs_path.exists():
        try:
            configuration["costs"] = json.loads(costs_path.read_text())
        except Exception as e:
            logger.error(f"Error loading costs.json: {e}")
            configuration["costs"] = None
    
    # Load tariff configuration
    tariff_path = tables_dir / "tariff_values.json"
    if tariff_path.exists():
        try:
            configuration["tariffs"] = json.loads(tariff_path.read_text())
        except Exception as e:
            logger.error(f"Error loading tariff_values.json: {e}")
            configuration["tariffs"] = None
    
    # Load table mappings
    table_map_path = config_dir / "table_map.json"
    if table_map_path.exists():
        try:
            configuration["table_mappings"] = json.loads(table_map_path.read_text())
        except Exception as e:
            logger.error(f"Error loading table_map.json: {e}")
            configuration["table_mappings"] = None
    
    # Load column mappings
    column_map_path = config_dir / "column_map.json"
    if column_map_path.exists():
        try:
            configuration["column_mappings"] = json.loads(column_map_path.read_text())
        except Exception as e:
            logger.error(f"Error loading column_map.json: {e}")
            configuration["column_mappings"] = None
    
    return configuration


@router.get("/profiles/{profile_id}/config/costs")
async def get_costs_configuration(
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """Get cost components and economic impact parameters configuration"""
    config_path = file_manager.get_profile_config_dir(profile_id) / "costs.json"
    
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Costs configuration not found"
        )
    
    try:
        return json.loads(config_path.read_text())
    except Exception as e:
        logger.error(f"Error loading costs configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load costs configuration"
        )


@router.put("/profiles/{profile_id}/config/costs")
async def update_costs_configuration(
    costs_config: Dict[str, Any],
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """Update cost components and economic impact parameters"""
    config_path = file_manager.get_profile_config_dir(profile_id) / "costs.json"
    
    try:
        # Validate structure
        if "cost_components" not in costs_config and "economic_impact_parameters" not in costs_config:
            # Old format - convert to new format
            new_config = {
                "cost_components": {},
                "economic_impact_parameters": {}
            }
            
            # Extract cost components
            cost_component_keys = [
                "cost_BasePrice", "cost_Tariff", "cost_Logistics",
                "cost_Holding_LeadTime", "cost_Holding_LTVariability",
                "cost_Holding_Lateness", "cost_Risk_PriceVolatility",
                "cost_Impact_PriceTrend"
            ]
            
            for key in cost_component_keys:
                if key in costs_config:
                    new_config["cost_components"][key] = costs_config[key]
            
            # Extract economic impact parameters
            for key, value in costs_config.items():
                if key not in cost_component_keys:
                    new_config["economic_impact_parameters"][key] = value
            
            costs_config = new_config
        
        # Save configuration
        config_path.write_text(json.dumps(costs_config, indent=2))
        
        return {
            "status": "success",
            "message": "Costs configuration updated",
            "configuration": costs_config
        }
        
    except Exception as e:
        logger.error(f"Error updating costs configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update costs configuration: {str(e)}"
        )


@router.get("/profiles/{profile_id}/config/tariffs")
async def get_tariffs_configuration(
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """Get tariff configuration"""
    tariff_path = file_manager.get_profile_tables_dir(profile_id) / "tariff_values.json"
    
    if not tariff_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Tariff configuration not found"
        )
    
    try:
        return json.loads(tariff_path.read_text())
    except Exception as e:
        logger.error(f"Error loading tariff configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load tariff configuration"
        )


@router.put("/profiles/{profile_id}/config/tariffs")
async def update_tariffs_configuration(
    tariff_config: Dict[str, float],
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """Update tariff configuration"""
    tariff_path = file_manager.get_profile_tables_dir(profile_id) / "tariff_values.json"
    
    try:
        # Validate that all values are numeric
        for country, rate in tariff_config.items():
            if not isinstance(rate, (int, float)):
                raise ValueError(f"Invalid tariff rate for {country}: must be numeric")
            if rate < 0:
                raise ValueError(f"Invalid tariff rate for {country}: must be non-negative")
        
        # Save configuration
        tariff_path.write_text(json.dumps(tariff_config, indent=2))
        
        return {
            "status": "success",
            "message": "Tariff configuration updated",
            "configuration": tariff_config
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating tariff configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update tariff configuration: {str(e)}"
        )


@router.get("/profiles/{profile_id}/config/metric-weights")
async def get_metric_weights(
    profile_id: str = Depends(validate_profile)
) -> Dict[str, float]:
    """Get current metric weights used for vendor evaluation"""
    # Default weights (these could be stored in a separate config file)
    default_weights = {
        "price_weight": 0.3,
        "lead_time_weight": 0.2,
        "reliability_weight": 0.2,
        "quality_weight": 0.15,
        "risk_weight": 0.15
    }
    
    # Check if custom weights file exists
    weights_path = file_manager.get_profile_config_dir(profile_id) / "metric_weights.json"
    
    if weights_path.exists():
        try:
            return json.loads(weights_path.read_text())
        except Exception as e:
            logger.warning(f"Error loading metric weights, using defaults: {e}")
    
    return default_weights


@router.put("/profiles/{profile_id}/config/metric-weights")
async def update_metric_weights(
    weights: Dict[str, float],
    profile_id: str = Depends(validate_profile)
) -> Dict[str, Any]:
    """Update metric weights for vendor evaluation"""
    weights_path = file_manager.get_profile_config_dir(profile_id) / "metric_weights.json"
    
    try:
        # Validate weights
        required_weights = ["price_weight", "lead_time_weight", "reliability_weight", 
                          "quality_weight", "risk_weight"]
        
        for weight_name in required_weights:
            if weight_name not in weights:
                raise ValueError(f"Missing required weight: {weight_name}")
            if not isinstance(weights[weight_name], (int, float)):
                raise ValueError(f"Invalid weight for {weight_name}: must be numeric")
            if weights[weight_name] < 0 or weights[weight_name] > 1:
                raise ValueError(f"Invalid weight for {weight_name}: must be between 0 and 1")
        
        # Validate sum equals 1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        # Save weights
        weights_path.write_text(json.dumps(weights, indent=2))
        
        return {
            "status": "success",
            "message": "Metric weights updated",
            "weights": weights
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating metric weights: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update metric weights: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    """Check configuration service health"""
    return {
        "status": "healthy",
        "service": "configuration",
        "version": settings.API_VERSION
    }