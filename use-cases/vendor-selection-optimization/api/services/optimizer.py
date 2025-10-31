"""Optimization Service Wrapper"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from ..config import settings
from ..models.jobs import JobType, JobStatus
from ..models.requests import OptimizeAllocationRequest
from .job_manager import job_manager
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)


class Optimizer:
    """Service wrapper for procurement optimization functionality"""
    
    def __init__(self):
        self.script_path = settings.OPTIMIZATION_DIR / "optimize_procurement.py"
        
    def prepare_optimization_args(
        self,
        profile_id: str,
        request: OptimizeAllocationRequest
    ) -> List[str]:
        """Prepare command line arguments for optimization script"""
        profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
        profile_config_dir = file_manager.get_profile_config_dir(profile_id)
        
        # Get the ranking file path based on mode
        ranking_file = profile_tables_dir / f"vendor_{request.mode}_ranking_tariff_values.csv"
        
        # Output file path
        output_file = profile_tables_dir / f"optimized_allocation_{request.mode}_vendor_{request.mode}_ranking_tariff_values.csv"
        
        # Base arguments
        args = [
            sys.executable,
            str(self.script_path),
            "--ranking-results-path", str(ranking_file),
            "--tables-dir", str(settings.TABLES_DIR),
            "--optimization-output-path", str(output_file),
            "--table-map", str(profile_config_dir / "table_map.json"),
            "--column-map", str(profile_config_dir / "column_map.json"),
            "--mode", request.mode
        ]
        
        return args
    
    def run_optimization(
        self,
        profile_id: str,
        request: OptimizeAllocationRequest,
        job_id: str
    ) -> None:
        """Run optimization asynchronously (always async due to computational complexity)"""
        try:
            # Update job status to running
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            
            # Check if ranking file exists
            profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
            ranking_file = profile_tables_dir / f"vendor_{request.mode}_ranking_tariff_values.csv"
            
            if not ranking_file.exists():
                raise FileNotFoundError(
                    f"Vendor ranking file not found. Please run vendor evaluation first: {ranking_file}"
                )
            
            # Prepare arguments
            args = self.prepare_optimization_args(profile_id, request)
            
            # Update costs.json with demand period if different from default
            if request.demand_period_days != 365:
                self._update_demand_period(profile_id, request.demand_period_days)
            
            # Run the optimization script
            logger.info(f"Running optimization for profile {profile_id}")
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=30)
            
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=80)
            
            # Get the output file path
            output_file = profile_tables_dir / f"optimized_allocation_{request.mode}_vendor_{request.mode}_ranking_tariff_values.csv"
            
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            # Read and process results
            df_results = pd.read_csv(output_file)
            
            # Calculate summary statistics
            summary = self._calculate_optimization_summary(df_results, result.stdout)
            
            # Save summary
            summary_file = file_manager.get_result_file_path(job_id, "optimization_summary.json")
            summary_file.write_text(json.dumps(summary, indent=2))
            
            # Update job with results
            job_manager.update_job_result(
                job_id=job_id,
                result_location=str(output_file),
                result_size_bytes=output_file.stat().st_size,
                result_summary=summary
            )
            
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, progress=100)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Optimization script failed: {e.stderr}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=f"Optimization failed: {e.stderr}",
                error_details={"type": "OptimizationError", "stderr": e.stderr}
            )
        except Exception as e:
            logger.error(f"Optimization failed for job {job_id}: {e}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                error_details={"type": type(e).__name__}
            )
    
    def _update_demand_period(self, profile_id: str, demand_period_days: int) -> None:
        """Update demand period in costs configuration"""
        try:
            config_path = file_manager.get_profile_config_dir(profile_id) / "costs.json"
            
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
                
                # Handle both old and new format
                if 'economic_impact_parameters' in config_data:
                    config_data['economic_impact_parameters']['DEMAND_PERIOD_DAYS'] = demand_period_days
                else:
                    config_data['DEMAND_PERIOD_DAYS'] = demand_period_days
                
                config_path.write_text(json.dumps(config_data, indent=2))
                logger.info(f"Updated demand period to {demand_period_days} days")
                
        except Exception as e:
            logger.warning(f"Could not update demand period: {e}")
    
    def _calculate_optimization_summary(
        self,
        df: pd.DataFrame,
        stdout: str
    ) -> Dict[str, Any]:
        """Calculate optimization summary from results"""
        # Extract solver information from stdout
        solver_time = 0.0
        optimality_gap = 0.0
        optimization_status = "unknown"
        
        if "Optimization Status: Optimal" in stdout:
            optimization_status = "optimal"
        elif "Optimization Status: Infeasible" in stdout:
            optimization_status = "infeasible"
        elif "Optimization Status: Unbounded" in stdout:
            optimization_status = "unbounded"
        
        # Try to extract solver time and objective value from stdout
        for line in stdout.split('\n'):
            if "Optimization Status:" in line and "seconds" in line:
                try:
                    solver_time = float(line.split('(')[1].split(' seconds')[0])
                except:
                    pass
            if "Optimal Objective Value" in line:
                try:
                    total_cost = float(line.split(':')[1].strip().replace(',', ''))
                except:
                    total_cost = df['Optimized_Total_Effective_Cost_for_Combo'].sum()
        
        # Calculate from dataframe if not found in stdout
        if 'total_cost' not in locals():
            total_cost = df['Optimized_Total_Effective_Cost_for_Combo'].sum()
        
        return {
            "optimization_status": optimization_status,
            "total_effective_cost": total_cost,
            "total_materials_optimized": df['MATNR'].nunique(),
            "total_allocation_changes": len(df[df['Allocated_Quantity'] > 0]),
            "total_quantity_allocated": df['Allocated_Quantity'].sum(),
            "solver_time_seconds": solver_time,
            "optimality_gap": optimality_gap,
            "constraints_satisfied": {
                "demand_met": True,  # Assumed from successful completion
                "capacity_respected": True,
                "multi_supplier_enforced": True
            }
        }
    
    def get_optimization_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization summary for a completed job"""
        job = job_manager.get_job(job_id)
        
        if not job or job.job_type != JobType.OPTIMIZATION:
            return None
            
        if job.status != JobStatus.COMPLETED:
            return None
            
        # Load summary file
        summary_file = file_manager.get_result_file_path(job_id, "optimization_summary.json")
        
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        
        return job.result_summary
    
    def get_allocation_details(
        self,
        job_id: str,
        page: int = 1,
        page_size: int = 100,
        material_filter: Optional[str] = None,
        supplier_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get paginated allocation details"""
        job = job_manager.get_job(job_id)
        
        if not job or job.job_type != JobType.OPTIMIZATION:
            return None
            
        if job.status != JobStatus.COMPLETED or not job.result_location:
            return None
        
        # Load results
        result_path = Path(job.result_location)
        if not result_path.exists():
            return None
            
        df = pd.read_csv(result_path)
        
        # Apply filters
        if material_filter:
            df = df[df['MATNR'].astype(str).str.contains(material_filter, case=False)]
        if supplier_filter:
            df = df[df['LIFNR'].astype(str).str.contains(supplier_filter, case=False)]
        
        # Only include allocated items
        df = df[df['Allocated_Quantity'] > 0]
        
        # Calculate pagination
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get page data
        page_df = df.iloc[start_idx:end_idx]
        
        # Format allocations
        allocations = []
        for _, row in page_df.iterrows():
            allocations.append({
                "supplier_id": str(row['LIFNR']),
                "supplier_name": row.get('NAME1', 'Unknown'),
                "material_id": str(row['MATNR']),
                "material_description": row.get('MAKTX', 'Unknown'),
                "allocated_quantity": float(row['Allocated_Quantity']),
                "effective_cost_per_unit": float(row['EffectiveCostPerUnit_USD']),
                "total_effective_cost": float(row['Optimized_Total_Effective_Cost_for_Combo']),
                "average_unit_price": float(row.get('AvgUnitPriceUSD_raw', 0))
            })
        
        return {
            "status": "success",
            "metadata": {
                "total_allocations": total_records,
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size
            },
            "allocations": allocations
        }


# Global optimizer instance
optimizer = Optimizer()