"""Pipeline Runner Service - Orchestrates the complete optimization pipeline"""

import sys
import json
import subprocess
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import logging
import pandas as pd

from ..config import settings
from ..models.jobs import JobType, JobStatus
from ..models.requests import RunPipelineRequest
from .job_manager import job_manager
from ..utils.file_manager import file_manager
from .vendor_evaluator import vendor_evaluator
from .optimizer import optimizer
from .comparator import comparator

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Service for running the complete procurement optimization pipeline"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)  # Single thread for sequential execution
        
    def validate_profile(self, profile_id: str) -> Optional[str]:
        """Validate that profile exists and has required configuration"""
        if not file_manager.profile_exists(profile_id):
            return f"Profile '{profile_id}' does not exist"
        
        # Check required configuration files
        config_dir = file_manager.get_profile_config_dir(profile_id)
        required_files = ["table_map.json", "column_map.json", "costs.json"]
        
        for file_name in required_files:
            if not (config_dir / file_name).exists():
                return f"Required configuration file missing: {file_name}"
        
        # Check tariff configuration
        tables_dir = file_manager.get_profile_tables_dir(profile_id)
        if not (tables_dir / "tariff_values.json").exists():
            return "Tariff configuration not found. Please configure tariffs first."
        
        return None
    
    async def run_pipeline(
        self,
        profile_id: str,
        request: RunPipelineRequest,
        job_id: str
    ) -> None:
        """Run the complete pipeline asynchronously"""
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._run_pipeline_sync,
            profile_id,
            request,
            job_id
        )
    
    def _run_pipeline_sync(
        self,
        profile_id: str,
        request: RunPipelineRequest,
        job_id: str
    ) -> None:
        """Synchronous pipeline execution"""
        try:
            # Update job status to running
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=5)
            
            # Validate profile
            error_msg = self.validate_profile(profile_id)
            if error_msg:
                raise ValueError(error_msg)
            
            # Update configuration if provided
            if request.metric_weights or request.demand_period_days != 365:
                self._update_configuration(profile_id, request)
            
            # Clean up old files if requested
            if request.clean_previous_results:
                self._clean_previous_results(profile_id, request.mode)
            
            # Step 1: Vendor Evaluation
            logger.info(f"Pipeline {job_id}: Starting vendor evaluation")
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=10)
            
            eval_result = self._run_vendor_evaluation(profile_id, request)
            if not eval_result["success"]:
                raise RuntimeError(f"Vendor evaluation failed: {eval_result.get('error', 'Unknown error')}")
            
            # Step 2: Procurement Optimization
            logger.info(f"Pipeline {job_id}: Starting procurement optimization")
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=40)
            
            opt_result = self._run_optimization(profile_id, request)
            if not opt_result["success"]:
                raise RuntimeError(f"Optimization failed: {opt_result.get('error', 'Unknown error')}")
            
            # Step 3: Policy Comparison
            logger.info(f"Pipeline {job_id}: Starting policy comparison")
            job_manager.update_job_status(job_id, JobStatus.RUNNING, progress=70)
            
            comp_result = self._run_comparison(profile_id, request)
            if not comp_result["success"]:
                raise RuntimeError(f"Comparison failed: {comp_result.get('error', 'Unknown error')}")
            
            # Collect summary results
            summary = self._create_pipeline_summary(
                profile_id, request, eval_result, opt_result, comp_result
            )
            
            # Copy CSV files to job results directory and update summary
            csv_files = self._copy_csv_files_to_results(
                job_id, eval_result, opt_result, comp_result
            )
            
            # Add download information to summary
            summary["downloads"] = {
                "summary": {
                    "url": f"/api/jobs/{job_id}/download",
                    "filename": "pipeline_summary.json",
                    "size_bytes": 0  # Will be updated after saving
                },
                "vendor_evaluation": csv_files.get("vendor_evaluation", {}),
                "optimization": csv_files.get("optimization", {}),
                "comparison": csv_files.get("comparison", {}),
                "all_results": {
                    "url": f"/api/jobs/{job_id}/download/all",
                    "filename": f"pipeline_results_{job_id}.zip",
                    "size_bytes": sum(f.get("size_bytes", 0) for f in csv_files.values())
                }
            }
            
            # Save summary
            summary_file = file_manager.get_result_file_path(job_id, "pipeline_summary.json")
            summary_file.write_text(json.dumps(summary, indent=2))
            
            # Update summary file size
            summary["downloads"]["summary"]["size_bytes"] = summary_file.stat().st_size
            summary["downloads"]["all_results"]["size_bytes"] += summary_file.stat().st_size
            
            # Save summary again with updated sizes
            summary_file.write_text(json.dumps(summary, indent=2))
            
            # Update job with results
            job_manager.update_job_result(
                job_id=job_id,
                result_location=str(summary_file),
                result_size_bytes=summary_file.stat().st_size,
                result_summary=summary
            )
            
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, progress=100)
            
            logger.info(f"Pipeline {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id}: {e}")
            job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                error_details={"type": type(e).__name__}
            )
    
    def _update_configuration(
        self,
        profile_id: str,
        request: RunPipelineRequest
    ) -> None:
        """Update configuration based on request parameters"""
        config_dir = file_manager.get_profile_config_dir(profile_id)
        
        # Update costs.json if metric weights provided
        if request.metric_weights:
            # Metric weights would be handled by vendor evaluation
            # For now, we'll pass them directly to the evaluation step
            pass
        
        # Update demand period if different from default
        if request.demand_period_days != 365:
            costs_path = config_dir / "costs.json"
            if costs_path.exists():
                try:
                    config_data = json.loads(costs_path.read_text())
                    
                    # Handle both old and new format
                    if 'economic_impact_parameters' in config_data:
                        config_data['economic_impact_parameters']['DEMAND_PERIOD_DAYS'] = request.demand_period_days
                    else:
                        config_data['DEMAND_PERIOD_DAYS'] = request.demand_period_days
                    
                    costs_path.write_text(json.dumps(config_data, indent=2))
                    logger.info(f"Updated demand period to {request.demand_period_days} days")
                except Exception as e:
                    logger.warning(f"Could not update demand period: {e}")
    
    def _clean_previous_results(
        self,
        profile_id: str,
        mode: str
    ) -> None:
        """Clean previous result files"""
        tables_dir = file_manager.get_profile_tables_dir(profile_id)
        
        files_to_delete = [
            f"vendor_{mode}_ranking_tariff_values.csv",
            f"optimized_allocation_{mode}_vendor_{mode}_ranking_tariff_values.csv",
            "comparison.csv",
            "vendor_with_direct_countries.csv"
        ]
        
        for file_name in files_to_delete:
            file_path = tables_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted previous result: {file_name}")
                except Exception as e:
                    logger.warning(f"Could not delete {file_name}: {e}")
    
    def _run_vendor_evaluation(
        self,
        profile_id: str,
        request: RunPipelineRequest
    ) -> Dict[str, Any]:
        """Run vendor evaluation step"""
        try:
            # Prepare evaluation request
            from ..models.requests import EvaluateVendorsRequest
            eval_request = EvaluateVendorsRequest(
                profile_id=profile_id,
                mode=request.mode,
                filters=request.filters,
                metric_weights=request.metric_weights,
                include_details=True
            )
            
            # Prepare arguments
            args = vendor_evaluator.prepare_evaluation_args(profile_id, eval_request)
            
            # Run the evaluation script
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check output file exists
            output_file = file_manager.get_profile_tables_dir(profile_id) / f"vendor_{request.mode}_ranking_tariff_values.csv"
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            return {
                "success": True,
                "output_file": str(output_file),
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Vendor evaluation failed: {e.stderr}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_optimization(
        self,
        profile_id: str,
        request: RunPipelineRequest
    ) -> Dict[str, Any]:
        """Run procurement optimization step"""
        try:
            # Prepare optimization request
            from ..models.requests import OptimizeAllocationRequest
            opt_request = OptimizeAllocationRequest(
                profile_id=profile_id,
                mode=request.mode,
                demand_period_days=request.demand_period_days
            )
            
            # Prepare arguments
            args = optimizer.prepare_optimization_args(profile_id, opt_request)
            
            # Run the optimization script
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check output file exists
            output_file = file_manager.get_profile_tables_dir(profile_id) / f"optimized_allocation_{request.mode}_vendor_{request.mode}_ranking_tariff_values.csv"
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            return {
                "success": True,
                "output_file": str(output_file),
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Optimization failed: {e.stderr}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_comparison(
        self,
        profile_id: str,
        request: RunPipelineRequest
    ) -> Dict[str, Any]:
        """Run policy comparison step"""
        try:
            # Prepare comparison request
            from ..models.requests import ComparePoliciesRequest
            comp_request = ComparePoliciesRequest(
                profile_id=profile_id,
                mode=request.mode,
                optimization_job_id="dummy"  # This is not used in the comparison script
            )
            
            # Prepare arguments
            args = comparator.prepare_comparison_args(profile_id, comp_request)
            
            # Run the comparison script
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check output file exists
            output_file = file_manager.get_profile_tables_dir(profile_id) / "comparison.csv"
            if not output_file.exists():
                raise FileNotFoundError(f"Expected output file not found: {output_file}")
            
            return {
                "success": True,
                "output_file": str(output_file),
                "stdout": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Comparison failed: {e.stderr}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_pipeline_summary(
        self,
        profile_id: str,
        request: RunPipelineRequest,
        eval_result: Dict[str, Any],
        opt_result: Dict[str, Any],
        comp_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create pipeline execution summary"""
        summary = {
            "pipeline_id": f"pipeline_{profile_id}_{request.mode}",
            "profile_id": profile_id,
            "mode": request.mode,
            "demand_period_days": request.demand_period_days,
            "steps_completed": ["vendor_evaluation", "procurement_optimization", "policy_comparison"],
            "results": {
                "vendor_evaluation": {
                    "output_file": eval_result.get("output_file", ""),
                    "summary": self._extract_eval_summary(eval_result.get("stdout", ""))
                },
                "procurement_optimization": {
                    "output_file": opt_result.get("output_file", ""),
                    "summary": self._extract_opt_summary(opt_result.get("stdout", ""))
                },
                "policy_comparison": {
                    "output_file": comp_result.get("output_file", ""),
                    "summary": self._extract_comp_summary(comp_result.get("stdout", ""))
                }
            }
        }
        
        # Extract key metrics
        comp_summary = summary["results"]["policy_comparison"]["summary"]
        summary["key_metrics"] = {
            "total_economic_saving": comp_summary.get("net_economic_saving", 0),
            "percentage_saving": comp_summary.get("percentage_saving", 0),
            "materials_optimized": comp_summary.get("materials_compared", 0),
            "suppliers_analyzed": comp_summary.get("suppliers_analyzed", 0),
            "allocation_changes": comp_summary.get("allocation_changes", 0)
        }
        
        return summary
    
    def _extract_eval_summary(self, stdout: str) -> Dict[str, Any]:
        """Extract summary from vendor evaluation output"""
        summary = {
            "vendors_evaluated": 0,
            "materials_evaluated": 0,
            "combinations_evaluated": 0
        }
        
        # Simple extraction from stdout
        lines = stdout.split('\n')
        for line in lines:
            if "unique vendor-material combinations" in line:
                try:
                    summary["combinations_evaluated"] = int(line.split(':')[0].strip())
                except:
                    pass
            elif "unique materials" in line:
                try:
                    summary["materials_evaluated"] = int(line.split(':')[0].strip())
                except:
                    pass
            elif "unique suppliers" in line:
                try:
                    summary["vendors_evaluated"] = int(line.split(':')[0].strip())
                except:
                    pass
        
        return summary
    
    def _extract_opt_summary(self, stdout: str) -> Dict[str, Any]:
        """Extract summary from optimization output"""
        summary = {
            "optimization_status": "unknown",
            "solver_time": 0.0,
            "objective_value": 0.0
        }
        
        lines = stdout.split('\n')
        for line in lines:
            if "Optimization Status: Optimal" in line:
                summary["optimization_status"] = "optimal"
                if "seconds" in line:
                    try:
                        summary["solver_time"] = float(line.split('(')[1].split(' seconds')[0])
                    except:
                        pass
            elif "Optimal Objective Value" in line:
                try:
                    summary["objective_value"] = float(line.split(':')[1].strip().replace(',', ''))
                except:
                    pass
        
        return summary
    
    def _extract_comp_summary(self, stdout: str) -> Dict[str, Any]:
        """Extract summary from comparison output"""
        # Try to load the comparison.csv for proper analysis
        try:
            # Get the comparison file path - we don't have profile_id here, so extract from stdout
            comparison_df = None
            for line in stdout.split('\n'):
                if "Detailed comparison saved successfully to:" in line:
                    output_path = line.split("Detailed comparison saved successfully to:")[1].strip()
                    if Path(output_path).exists():
                        comparison_df = pd.read_csv(output_path)
                    break
        except Exception as e:
            logger.warning(f"Could not load comparison CSV: {e}")
            comparison_df = None
        
        # Reuse comparator's extraction logic
        return comparator._extract_comparison_summary(stdout, comparison_df)
    
    def get_pipeline_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline summary for a completed job"""
        job = job_manager.get_job(job_id)
        
        if not job or job.job_type != JobType.PIPELINE:
            return None
            
        if job.status != JobStatus.COMPLETED:
            return None
            
        # Load summary file
        summary_file = file_manager.get_result_file_path(job_id, "pipeline_summary.json")
        
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        
        return job.result_summary


    def _copy_csv_files_to_results(
        self,
        job_id: str,
        eval_result: Dict[str, Any],
        opt_result: Dict[str, Any],
        comp_result: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Copy CSV files to job results directory and return file metadata"""
        csv_files = {}
        
        # Copy vendor evaluation CSV
        if eval_result.get("output_file"):
            source_path = Path(eval_result["output_file"])
            if source_path.exists():
                dest_path = file_manager.get_result_file_path(job_id, "vendor_evaluation.csv")
                shutil.copy2(source_path, dest_path)
                csv_files["vendor_evaluation"] = {
                    "url": f"/api/jobs/{job_id}/download/vendor-evaluation",
                    "filename": "vendor_evaluation.csv",
                    "size_bytes": dest_path.stat().st_size,
                    "original_path": str(source_path)
                }
                logger.info(f"Copied vendor evaluation CSV to {dest_path}")
        
        # Copy optimization CSV
        if opt_result.get("output_file"):
            source_path = Path(opt_result["output_file"])
            if source_path.exists():
                dest_path = file_manager.get_result_file_path(job_id, "optimization_allocation.csv")
                shutil.copy2(source_path, dest_path)
                csv_files["optimization"] = {
                    "url": f"/api/jobs/{job_id}/download/optimization",
                    "filename": "optimization_allocation.csv",
                    "size_bytes": dest_path.stat().st_size,
                    "original_path": str(source_path)
                }
                logger.info(f"Copied optimization CSV to {dest_path}")
        
        # Copy comparison CSV
        if comp_result.get("output_file"):
            source_path = Path(comp_result["output_file"])
            if source_path.exists():
                dest_path = file_manager.get_result_file_path(job_id, "comparison.csv")
                shutil.copy2(source_path, dest_path)
                csv_files["comparison"] = {
                    "url": f"/api/jobs/{job_id}/download/comparison",
                    "filename": "comparison.csv",
                    "size_bytes": dest_path.stat().st_size,
                    "original_path": str(source_path)
                }
                logger.info(f"Copied comparison CSV to {dest_path}")
        
        return csv_files


# Global pipeline runner instance
pipeline_runner = PipelineRunner()