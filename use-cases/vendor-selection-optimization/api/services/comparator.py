"""Comparison Service Wrapper"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from ..config import settings
from ..models.jobs import JobType, JobStatus
from ..models.requests import ComparePoliciesRequest
from .job_manager import job_manager
from ..utils.file_manager import file_manager

logger = logging.getLogger(__name__)


class Comparator:
    """Service wrapper for policy comparison functionality"""
    
    def __init__(self):
        self.script_path = settings.OPTIMIZATION_DIR / "compare_policies.py"
        
    def prepare_comparison_args(
        self,
        profile_id: str,
        request: ComparePoliciesRequest
    ) -> List[str]:
        """Prepare command line arguments for comparison script"""
        profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
        profile_config_dir = file_manager.get_profile_config_dir(profile_id)
        
        # Get the ranking and optimization file paths based on mode
        ranking_file = profile_tables_dir / f"vendor_{request.mode}_ranking_tariff_values.csv"
        optimization_file = profile_tables_dir / f"optimized_allocation_{request.mode}_vendor_{request.mode}_ranking_tariff_values.csv"
        
        # Output file path
        output_file = profile_tables_dir / "comparison.csv"
        
        # Base arguments
        args = [
            sys.executable,
            str(self.script_path),
            "--ranking-results-path", str(ranking_file),
            "--optimization-results-path", str(optimization_file),
            "--tables-dir", str(settings.TABLES_DIR),
            "--comparison-output-path", str(output_file),
            "--table-map", str(profile_config_dir / "table_map.json"),
            "--column-map", str(profile_config_dir / "column_map.json"),
            "--mode", request.mode,
            "--costs-config-path", str(profile_config_dir / "costs.json")
        ]
        
        return args
    
    def check_required_files(
        self,
        profile_id: str,
        request: ComparePoliciesRequest
    ) -> Optional[str]:
        """Check if required input files exist"""
        profile_tables_dir = file_manager.get_profile_tables_dir(profile_id)
        
        # Check ranking file
        ranking_file = profile_tables_dir / f"vendor_{request.mode}_ranking_tariff_values.csv"
        if not ranking_file.exists():
            return f"Vendor ranking file not found. Please run vendor evaluation first: {ranking_file}"
        
        # Check optimization file
        optimization_file = profile_tables_dir / f"optimized_allocation_{request.mode}_vendor_{request.mode}_ranking_tariff_values.csv"
        if not optimization_file.exists():
            return f"Optimization results file not found. Please run procurement optimization first: {optimization_file}"
        
        return None
    
    def run_comparison(
        self,
        profile_id: str,
        request: ComparePoliciesRequest,
        job_id: str
    ) -> None:
        """Run comparison asynchronously (always async due to data processing complexity)"""
        try:
            # Update job status to running
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            
            # Check if required files exist
            error_msg = self.check_required_files(profile_id, request)
            if error_msg:
                raise FileNotFoundError(error_msg)
            
            # Prepare arguments
            args = self.prepare_comparison_args(profile_id, request)
            
            # Run the comparison script
            logger.info(f"Running policy comparison for profile {profile_id}")
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=30)
            
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=80)
            
            # Get the output file path
            output_file = file_manager.get_profile_tables_dir(profile_id) / "comparison.csv"
            
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            # Read and process results
            df_results = pd.read_csv(output_file)
            
            # Extract summary statistics from stdout
            summary = self._extract_comparison_summary(result.stdout, df_results)
            
            # Save summary
            summary_file = file_manager.get_result_file_path(job_id, "comparison_summary.json")
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
            logger.error(f"Comparison script failed: {e.stderr}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=f"Comparison failed: {e.stderr}",
                error_details={"type": "ComparisonError", "stderr": e.stderr}
            )
        except Exception as e:
            logger.error(f"Comparison failed for job {job_id}: {e}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                error_details={"type": type(e).__name__}
            )
    
    def _extract_comparison_summary(
        self,
        stdout: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract comparison summary from script output and results"""
        summary = {
            "comparison_mode": "unknown",
            "demand_period_days": 365,
            "total_historical_quantity": 0.0,
            "total_optimized_quantity": 0.0,
            "total_historical_spend": 0.0,
            "total_optimized_spend_estimated": 0.0,
            "total_historical_effective_cost": 0.0,
            "total_optimized_effective_cost": 0.0,
            "net_economic_saving": 0.0,
            "percentage_saving": 0.0,
            "cost_component_changes": {},
            "suppliers_analyzed": 0,
            "materials_compared": 0,
            "allocation_changes": 0
        }
        
        # Parse stdout for summary values
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if "Comparison Mode:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    mode_info = parts[1].strip()
                    if mode_info.startswith("matkl"):
                        summary["comparison_mode"] = "material_group"
                    elif mode_info.startswith("matnr"):
                        summary["comparison_mode"] = "material_id"
                    elif mode_info.startswith("maktx"):
                        summary["comparison_mode"] = "material_description"
                    
                    # Extract demand period
                    if "period of" in mode_info and "days" in mode_info:
                        try:
                            days_part = mode_info.split("period of")[1].split("days")[0].strip()
                            summary["demand_period_days"] = int(days_part)
                        except:
                            pass
            
            elif "Total Allocated Quantity:" in line and i + 2 < len(lines):
                try:
                    hist_line = lines[i + 1]
                    opt_line = lines[i + 2]
                    if "Historical:" in hist_line:
                        summary["total_historical_quantity"] = float(hist_line.split(":")[1].replace("units", "").replace(",", "").strip())
                    if "Optimized:" in opt_line:
                        summary["total_optimized_quantity"] = float(opt_line.split(":")[1].replace("units", "").replace(",", "").strip())
                except:
                    pass
            
            elif "Total Actual Spend (USD):" in line and i + 2 < len(lines):
                try:
                    hist_line = lines[i + 1]
                    opt_line = lines[i + 2]
                    if "Historical:" in hist_line:
                        summary["total_historical_spend"] = float(hist_line.split(":")[1].replace(",", "").strip())
                    if "Optimized" in opt_line:
                        summary["total_optimized_spend_estimated"] = float(opt_line.split(":")[1].replace(",", "").strip())
                except:
                    pass
            
            elif "Total Effective Cost (USD):" in line and i + 3 < len(lines):
                try:
                    hist_line = lines[i + 1]
                    opt_line = lines[i + 2]
                    saving_line = lines[i + 3]
                    if "Historical:" in hist_line:
                        summary["total_historical_effective_cost"] = float(hist_line.split(":")[1].replace(",", "").strip())
                    if "Optimized:" in opt_line:
                        summary["total_optimized_effective_cost"] = float(opt_line.split(":")[1].replace(",", "").strip())
                    if "Net Economic Saving:" in saving_line:
                        saving_part = saving_line.split(":")[1]
                        amount = float(saving_part.split("(")[0].replace(",", "").strip())
                        percentage = float(saving_part.split("(")[1].split("%")[0].strip())
                        summary["net_economic_saving"] = amount
                        summary["percentage_saving"] = percentage
                except:
                    pass
            
            elif "Cost Component" in line and "Historical Total" in line:
                # Parse cost component breakdown
                cost_components = {}
                j = i + 2  # Skip header and divider
                while j < len(lines) and lines[j].strip() and not lines[j].startswith("SUM"):
                    try:
                        parts = lines[j].split("|")
                        if len(parts) >= 4:
                            component = parts[0].strip()
                            if "(disabled)" not in parts[1]:
                                hist_val = float(parts[1].replace(",", "").strip())
                                opt_val = float(parts[2].replace(",", "").strip())
                                change_val = float(parts[3].replace(",", "").strip())
                                cost_components[component] = {
                                    "historical": hist_val,
                                    "optimized": opt_val,
                                    "change": change_val
                                }
                    except:
                        pass
                    j += 1
                if cost_components:
                    summary["cost_component_changes"] = cost_components
        
        # Calculate from dataframe if not found in stdout
        if df is not None:
            if 'LIFNR' in df.columns:
                summary["suppliers_analyzed"] = df['LIFNR'].nunique()
            
            if 'MATNR' in df.columns:
                summary["materials_compared"] = df['MATNR'].nunique()
            
            # Count allocation changes
            if 'Delta_Allocated_Quantity' in df.columns:
                summary["allocation_changes"] = len(df[df['Delta_Allocated_Quantity'].abs() > 0.001])
        
        return summary
    
    def get_comparison_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get comparison summary for a completed job"""
        job = job_manager.get_job(job_id)
        
        if not job or job.job_type != JobType.COMPARISON:
            return None
            
        if job.status != JobStatus.COMPLETED:
            return None
            
        # Load summary file
        summary_file = file_manager.get_result_file_path(job_id, "comparison_summary.json")
        
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        
        return job.result_summary
    
    def get_comparison_details(
        self,
        job_id: str,
        page: int = 1,
        page_size: int = 100,
        supplier_filter: Optional[str] = None,
        material_filter: Optional[str] = None,
        change_type: Optional[str] = None  # "increased", "decreased", "new", "removed"
    ) -> Optional[Dict[str, Any]]:
        """Get paginated comparison details"""
        job = job_manager.get_job(job_id)
        
        if not job or job.job_type != JobType.COMPARISON:
            return None
            
        if job.status != JobStatus.COMPLETED or not job.result_location:
            return None
        
        # Load results
        result_path = Path(job.result_location)
        if not result_path.exists():
            return None
            
        df = pd.read_csv(result_path)
        
        # Apply filters
        if supplier_filter:
            df = df[df['LIFNR'].astype(str).str.contains(supplier_filter, case=False)]
        if material_filter:
            df = df[df['MATNR'].astype(str).str.contains(material_filter, case=False)]
        
        # Apply change type filter
        if change_type:
            if change_type == "increased":
                df = df[df['Delta_Allocated_Quantity'] > 0.001]
            elif change_type == "decreased":
                df = df[df['Delta_Allocated_Quantity'] < -0.001]
            elif change_type == "new":
                df = df[(df['Historical_Allocated_Quantity'] == 0) & (df['Optimized_Allocated_Quantity'] > 0)]
            elif change_type == "removed":
                df = df[(df['Historical_Allocated_Quantity'] > 0) & (df['Optimized_Allocated_Quantity'] == 0)]
        
        # Calculate pagination
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get page data
        page_df = df.iloc[start_idx:end_idx]
        
        # Format comparison entries
        comparisons = []
        for _, row in page_df.iterrows():
            comparison = {
                "supplier_id": str(row['LIFNR']),
                "supplier_name": row.get('NAME1', 'Unknown'),
                "material_id": str(row['MATNR']),
                "material_description": row.get('MAKTX', 'Unknown'),
                "historical": {
                    "quantity": float(row.get('Historical_Allocated_Quantity', 0)),
                    "actual_spend": float(row.get('Historical_Actual_Spend_USD', 0)),
                    "effective_cost": float(row.get('Historical_Total_Effective_Cost_for_Combo', 0)),
                    "unit_cost": float(row.get('Historical_EffectiveCostPerUnit_USD', 0))
                },
                "optimized": {
                    "quantity": float(row.get('Optimized_Allocated_Quantity', 0)),
                    "estimated_spend": float(row.get('Optimized_Actual_Spend_USD_Est_for_Combo', 0)),
                    "effective_cost": float(row.get('Optimized_Total_Effective_Cost_for_Combo', 0)),
                    "unit_cost": float(row.get('Optimized_EffectiveCostPerUnit_USD', 0))
                },
                "delta": {
                    "quantity": float(row.get('Delta_Allocated_Quantity', 0)),
                    "effective_cost": float(row.get('Delta_Total_Effective_Cost_for_Combo', 0))
                }
            }
            
            # Add change type
            if comparison["delta"]["quantity"] > 0.001:
                comparison["change_type"] = "increased"
            elif comparison["delta"]["quantity"] < -0.001:
                comparison["change_type"] = "decreased"
            elif comparison["historical"]["quantity"] == 0 and comparison["optimized"]["quantity"] > 0:
                comparison["change_type"] = "new"
            elif comparison["historical"]["quantity"] > 0 and comparison["optimized"]["quantity"] == 0:
                comparison["change_type"] = "removed"
            else:
                comparison["change_type"] = "unchanged"
            
            comparisons.append(comparison)
        
        return {
            "status": "success",
            "metadata": {
                "total_comparisons": total_records,
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size
            },
            "comparisons": comparisons
        }


# Global comparator instance
comparator = Comparator()