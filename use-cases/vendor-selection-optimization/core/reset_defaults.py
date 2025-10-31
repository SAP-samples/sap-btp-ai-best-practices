"""
Reset Defaults Module

This module provides functionality to reset all configuration values 
(EIP parameters, cost components, tariffs, and logistics factors) to their default values.
"""

import json
import os
import subprocess
from datetime import datetime
from config import settings


def reset_to_defaults():
    """Reset all EIP parameters, cost components, tariffs, and logistics factors to default values"""
    try:
        # 1. Reset costs.json to default values
        default_costs_config = {
            "cost_components": {
                "cost_BasePrice": "True",
                "cost_Tariff": "True",
                "cost_Holding_LeadTime": "True",
                "cost_Holding_LTVariability": "True",
                "cost_Holding_Lateness": "True",
                "cost_Inefficiency_InFull": "False",
                "cost_Risk_PriceVolatility": "True",
                "cost_Impact_PriceTrend": "True",
                "cost_Logistics": "True"
            },
            "economic_impact_parameters": {
                "EIP_ANNUAL_HOLDING_COST_RATE": 0.18,
                "EIP_SafetyStockMultiplierForLTVar_Param": 1.65,
                "EIP_RiskPremiumFactorForPriceVolatility_Param": 0.25,
                "EIP_PlanningHorizonDaysForPriceTrend_Param": 90,
                "PRICE_TREND_CUTOFF_DAYS": 180,
                "EIP_BaseLogisticsCostRate_Param": 0.15
            }
        }
        
        with open(settings.COSTS_CONFIG_FILE, 'w') as f:
            json.dump(default_costs_config, f, indent=4)
        
        # 2. Reset tariff_values.json by copying from backup
        backup_tariff_path = os.path.join('tables', 'tariff_values-backup.json')
        tariff_path = os.path.join('tables', 'tariff_values.json')
        try:
            with open(backup_tariff_path, 'r') as src, open(tariff_path, 'w') as dst:
                dst.write(src.read())
        except Exception as e:
            return False, f"Error resetting tariff values: {str(e)}"
        
        # 3. Reset logistics_factors.json to default values
        default_logistics = {
            "country_factors": {
                "CN": 1.0,
                "DE": 0.8,
                "HU": 0.85,
                "IN": 0.9,
                "ID": 1.0,
                "MY": 0.95,
                "MX": 0.3,
                "TH": 0.95,
                "US": 0.1
            },
            "material_factors": {
                "ECR-SENSOR": 0.3,
                "ECR-ZFRAME": 0.8,
                "EMN-BRAKES": 0.6,
                "EMN-HANDLE": 0.4,
                "EMN-MOTOR": 0.9,
                "EMN-THROTTLE": 0.35
            },
            "base_rate": 0.15,
            "combinations": {
                "CN_ECR-SENSOR": 0.332,
                "CN_ECR-ZFRAME": 0.654,
                "CN_EMN-BRAKES": 0.6312,
                "CN_EMN-HANDLE": 0.3758,
                "CN_EMN-MOTOR": 0.7564,
                "CN_EMN-THROTTLE": 0.3283,
                "DE_ECR-SENSOR": 0.214,
                "DE_ECR-ZFRAME": 0.5235,
                "DE_EMN-BRAKES": 0.4686,
                "DE_EMN-HANDLE": 0.3169,
                "DE_EMN-MOTOR": 0.7176,
                "DE_EMN-THROTTLE": 0.2635,
                "HU_ECR-SENSOR": 0.298,
                "HU_ECR-ZFRAME": 0.7165,
                "HU_EMN-BRAKES": 0.5016,
                "HU_EMN-HANDLE": 0.3276,
                "HU_EMN-MOTOR": 0.6822,
                "HU_EMN-THROTTLE": 0.3255,
                "IN_ECR-SENSOR": 0.265,
                "IN_ECR-ZFRAME": 0.6946,
                "IN_EMN-BRAKES": 0.6308,
                "IN_EMN-HANDLE": 0.3803,
                "IN_EMN-MOTOR": 0.886,
                "IN_EMN-THROTTLE": 0.351,
                "ID_ECR-SENSOR": 0.2549,
                "ID_ECR-ZFRAME": 0.8128,
                "ID_EMN-BRAKES": 0.6288,
                "ID_EMN-HANDLE": 0.3824,
                "ID_EMN-MOTOR": 0.9749,
                "ID_EMN-THROTTLE": 0.3119,
                "MY_ECR-SENSOR": 0.3167,
                "MY_ECR-ZFRAME": 0.9082,
                "MY_EMN-BRAKES": 0.6349,
                "MY_EMN-HANDLE": 0.3918,
                "MY_EMN-MOTOR": 0.9222,
                "MY_EMN-THROTTLE": 0.3822,
                "MX_ECR-SENSOR": 0.0837,
                "MX_ECR-ZFRAME": 0.2466,
                "MX_EMN-BRAKES": 0.2074,
                "MX_EMN-HANDLE": 0.1155,
                "MX_EMN-MOTOR": 0.2417,
                "MX_EMN-THROTTLE": 0.1207,
                "TH_ECR-SENSOR": 0.2447,
                "TH_ECR-ZFRAME": 0.8123,
                "TH_EMN-BRAKES": 0.4641,
                "TH_EMN-HANDLE": 0.4538,
                "TH_EMN-MOTOR": 0.9673,
                "TH_EMN-THROTTLE": 0.3042,
                "US_ECR-SENSOR": 0.0312,
                "US_ECR-ZFRAME": 0.0748,
                "US_EMN-BRAKES": 0.0693,
                "US_EMN-HANDLE": 0.0456,
                "US_EMN-MOTOR": 0.0859,
                "US_EMN-THROTTLE": 0.0404
            },
            "defaults": {
                "country_factor": 0.7,
                "material_factor": 0.5,
                "combined_factor": 0.35
            }
        }
        
        logistics_path = os.path.join('tables', 'logistics_factors.json')
        with open(logistics_path, 'w') as f:
            json.dump(default_logistics, f, indent=2)
        
        # 4. Clear optimization files to force regeneration
        files_to_clean = [
            os.path.join('tables', 'vendor_maktx_ranking_tariff_values.csv'),
            os.path.join('tables', 'vendor_with_direct_countries.csv'),
            os.path.join('tables', 'optimized_allocation_maktx_vendor_maktx_ranking_tariff_values.csv'),
            os.path.join('tables', 'comparison.csv')
        ]
        
        # Create backup directory
        backup_dir = os.path.join('tables', 'backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    os.rename(file_path, backup_path)
                except Exception as e:
                    pass  # Continue if backup fails
        
        # 5. Run the optimization pipeline to regenerate the files
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'run_optimization_pipeline.sh'
        )
        
        if not os.path.exists(script_path):
            return True, "Settings reset successfully, but optimization script not found. Please run the optimization manually."
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Run the optimization pipeline
        try:
            process = subprocess.run(
                [script_path], 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=300  # 5 minute timeout
            )
            
            if process.returncode != 0:
                return True, f"Settings reset successfully, but optimization pipeline failed: {process.stderr[:200]}..."
            
            return True, "All settings reset to default values and optimization pipeline completed successfully!"
            
        except subprocess.TimeoutExpired:
            return True, "Settings reset successfully, but optimization pipeline timed out. Please run it manually."
        except Exception as e:
            return True, f"Settings reset successfully, but optimization pipeline encountered an error: {str(e)[:200]}..."
        
    except Exception as e:
        return False, f"Error resetting to defaults: {str(e)}"