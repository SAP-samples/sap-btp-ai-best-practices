"""Vendor Evaluation Service Wrapper"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import logging

from ..config import settings
from ..models.jobs import JobType, JobStatus
from ..models.requests import EvaluateVendorsRequest
from .job_manager import job_manager
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)


class VendorEvaluator:
    """Service wrapper for vendor evaluation functionality"""
    
    def __init__(self):
        self.script_path = settings.OPTIMIZATION_DIR / "evaluate_vendor_material_with_country_tariffs.py"
        
    def estimate_result_size(
        self,
        profile_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int]:
        """Estimate result size (records and bytes) for evaluation"""
        # Get historical data to estimate
        tables_dir = file_manager.get_profile_tables_dir(profile_id).parent.parent / "tables"
        
        try:
            # Load PO Items to get material/supplier combinations
            po_items_path = tables_dir / "SAP_VLY_IL_PO_ITEMS.csv"
            
            if not po_items_path.exists():
                logger.warning(f"PO Items file not found: {po_items_path}")
                return 0, 0
            
            # Count unique combinations
            df = pd.read_csv(po_items_path, usecols=['LIFNR', 'MATNR', 'MATKL'])
            
            # Apply filters if provided
            if filters:
                if filters.get('materials'):
                    df = df[df['MATNR'].isin(filters['materials'])]
                if filters.get('material_groups'):
                    df = df[df['MATKL'].isin(filters['material_groups'])]
                if filters.get('suppliers'):
                    df = df[df['LIFNR'].isin(filters['suppliers'])]
            
            # Count unique vendor-material combinations
            unique_combinations = df.groupby(['LIFNR', 'MATNR']).size().count()
            
            # Estimate CSV size (approximately 500 bytes per row with all columns)
            estimated_bytes = file_manager.estimate_csv_size(
                rows=unique_combinations,
                columns=40,  # Approximate number of columns in output
                avg_cell_size=25
            )
            
            return unique_combinations, estimated_bytes
            
        except Exception as e:
            logger.error(f"Error estimating result size: {e}")
            return 0, 0
    
    def should_run_async(
        self,
        profile_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if evaluation should run asynchronously"""
        records, bytes_size = self.estimate_result_size(profile_id, filters)
        
        # Check against thresholds
        if records > settings.MAX_INLINE_RECORDS:
            logger.info(f"Async required: {records} records > {settings.MAX_INLINE_RECORDS}")
            return True
            
        if bytes_size > settings.MAX_INLINE_SIZE_MB * 1024 * 1024:
            size_mb = bytes_size / (1024 * 1024)
            logger.info(f"Async required: {size_mb:.2f} MB > {settings.MAX_INLINE_SIZE_MB} MB")
            return True
            
        return False
    
    def prepare_evaluation_args(
        self,
        profile_id: str,
        request: EvaluateVendorsRequest
    ) -> List[str]:
        """Prepare command line arguments for evaluation script"""
        profile_path = file_manager.get_profile_path(profile_id)
        profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
        profile_config_dir = file_manager.get_profile_config_dir(profile_id)
        
        # Base arguments
        args = [
            sys.executable,
            str(self.script_path),
            "--tariff-results-path", str(profile_tables_dir / "vendor_with_direct_countries.csv"),
            "--tables-dir", str(settings.TABLES_DIR),
            "--ranking-output-dir", str(profile_tables_dir),
            "--table-map", str(profile_config_dir / "table_map.json"),
            "--column-map", str(profile_config_dir / "column_map.json"),
            "--mode", request.mode,
            "--costs-config-path", str(profile_config_dir / "costs.json"),
            "--country-tariffs-path", str(profile_tables_dir / "tariff_values.json")
        ]
        
        # Add metric weights if provided
        if request.metric_weights:
            weights_file = file_manager.get_temp_file_path(f"weights_{profile_id}.json")
            weights_file.write_text(json.dumps(request.metric_weights.model_dump()))
            args.extend(["--metric-weights", str(weights_file)])
        
        return args
    
    def run_evaluation(
        self,
        profile_id: str,
        request: EvaluateVendorsRequest
    ) -> Dict[str, Any]:
        """Run vendor evaluation synchronously"""
        args = self.prepare_evaluation_args(profile_id, request)
        
        try:
            # Run the evaluation script
            logger.info(f"Running vendor evaluation for profile {profile_id}")
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get the output file path
            profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
            output_file = profile_tables_dir / f"vendor_{request.mode}_ranking_tariff_values.csv"
            
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            # Read results
            df_results = pd.read_csv(output_file)
            
            # Apply filters if specified
            if request.filters:
                if request.filters.materials:
                    df_results = df_results[df_results['MATNR'].isin(request.filters.materials)]
                if request.filters.material_groups:
                    df_results = df_results[df_results['MATKL'].isin(request.filters.material_groups)]
                if request.filters.suppliers:
                    df_results = df_results[df_results['LIFNR'].isin(request.filters.suppliers)]
            
            # Convert to response format
            return self._format_evaluation_results(df_results, request)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation script failed: {e.stderr}")
            raise RuntimeError(f"Vendor evaluation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            raise
    
    def run_evaluation_async(
        self,
        profile_id: str,
        request: EvaluateVendorsRequest,
        job_id: str
    ) -> None:
        """Run vendor evaluation asynchronously"""
        try:
            # Update job status to running
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            
            # Run evaluation
            results = self.run_evaluation(profile_id, request)
            
            # Save results
            result_file = file_manager.get_result_file_path(job_id, "evaluation_results.json")
            result_file.write_text(json.dumps(results, indent=2))
            
            # Update job with results
            job_manager.update_job_result(
                job_id=job_id,
                result_location=str(result_file),
                result_size_bytes=result_file.stat().st_size,
                result_summary={
                    "total_records": results["metadata"]["total_combinations"],
                    "materials_evaluated": results["metadata"]["total_materials_evaluated"],
                    "vendors_analyzed": results["metadata"]["total_vendors_evaluated"]
                }
            )
            
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, progress=100)
            
        except Exception as e:
            logger.error(f"Async evaluation failed for job {job_id}: {e}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                error_details={"type": type(e).__name__}
            )
    
    def _format_evaluation_results(
        self,
        df: pd.DataFrame,
        request: EvaluateVendorsRequest
    ) -> Dict[str, Any]:
        """Format evaluation results for API response"""
        # Prepare vendor list
        vendors = []
        
        for idx, row in df.iterrows():
            vendor = {
                "rank": int(row.get('Rank', idx + 1)),
                "supplier_id": str(row['LIFNR']),
                "supplier_name": row.get('NAME1', 'Unknown'),
                "material_id": str(row['MATNR']),
                "material_description": row.get('MAKTX', 'Unknown'),
                "country": row.get('LAND1', 'Unknown'),
                "effective_cost_per_unit": float(row['EffectiveCostPerUnit_USD']),
                "final_score": float(row.get('FinalScore', 0)),
                "po_line_item_count": int(row.get('POLineItemCount', 0)),
                "metrics": {
                    "avg_unit_price": float(row.get('AvgUnitPriceUSD_raw', 0)),
                    "tariff_impact_percent": float(row.get('TariffImpact_raw_percent', 0)),
                    "logistics_cost": float(row.get('cost_Logistics', 0)),
                    "lead_time_days": float(row.get('AvgLeadTimeDays_raw', 0)),
                    "on_time_rate": float(row.get('OnTimeRate_raw', 0)),
                    "in_full_rate": float(row.get('InFullRate_raw', 0))
                }
            }
            
            # Add cost components if requested
            if request.include_details:
                vendor["cost_components"] = {
                    "cost_BasePrice": float(row.get('cost_BasePrice', 0)),
                    "cost_Tariff": float(row.get('cost_Tariff', 0)),
                    "cost_Logistics": float(row.get('cost_Logistics', 0)),
                    "cost_Holding_LeadTime": float(row.get('cost_Holding_LeadTime', 0)),
                    "cost_Holding_LTVariability": float(row.get('cost_Holding_LTVariability', 0)),
                    "cost_Holding_Lateness": float(row.get('cost_Holding_Lateness', 0)),
                    "cost_Risk_PriceVolatility": float(row.get('cost_Risk_PriceVolatility', 0)),
                    "cost_Impact_PriceTrend": float(row.get('cost_Impact_PriceTrend', 0))
                }
            
            vendors.append(vendor)
        
        # Prepare metadata
        unique_vendors = df['LIFNR'].nunique()
        unique_materials = df['MATNR'].nunique()
        
        return {
            "status": "success",
            "metadata": {
                "total_vendors_evaluated": unique_vendors,
                "total_materials_evaluated": unique_materials,
                "total_combinations": len(df),
                "filters_applied": bool(request.filters),
                "evaluation_mode": request.mode,
                "pagination": {
                    "current_page": 1,
                    "total_pages": 1,
                    "page_size": len(df),
                    "total_records": len(df)
                }
            },
            "vendors": vendors
        }


# Global vendor evaluator instance
vendor_evaluator = VendorEvaluator()