#!/usr/bin/env python3
"""Basic test script to verify API endpoints"""

import sys
from pathlib import Path

# Add parent directory to path for imports if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import asyncio
import json
import time
from typing import Dict, Any, Optional
import os

# API base URL
BASE_URL = "http://localhost:8000"
PROFILE_ID = "profile_1"  # Using an existing profile
TEST_OUTPUT_DIR = "test_output"


def print_response(response: httpx.Response, title: str):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Response: {response.text}")
    print(f"{'='*60}\n")


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    params: Optional[Dict[str, Any]] = None
) -> bool:
    """Download a file from the API"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(TEST_OUTPUT_DIR, filename)
        
        print(f"\nDownloading to: {filepath}")
        
        response = await client.get(url, params=params, follow_redirects=True)
        
        if response.status_code == 200:
            # Check if it's a binary response
            content_type = response.headers.get("content-type", "")
            content_disposition = response.headers.get("content-disposition", "")
            
            print(f"Content-Type: {content_type}")
            print(f"Content-Disposition: {content_disposition}")
            print(f"Content-Length: {response.headers.get('content-length', 'Unknown')} bytes")
            
            # Write the file
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Successfully downloaded: {filename}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Download failed with status: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return False


async def test_health_endpoints():
    """Test health check endpoints"""
    async with httpx.AsyncClient() as client:
        # Main health check
        response = await client.get(f"{BASE_URL}/health")
        print_response(response, "Main Health Check")
        
        # Optimization service health
        response = await client.get(f"{BASE_URL}/api/optimization/health")
        print_response(response, "Optimization Service Health")
        
        # Jobs service health
        response = await client.get(f"{BASE_URL}/api/jobs/health")
        print_response(response, "Jobs Service Health")
        
        # Configuration service health
        response = await client.get(f"{BASE_URL}/api/configuration/health")
        print_response(response, "Configuration Service Health")


async def test_configuration_endpoints():
    """Test configuration management endpoints"""
    async with httpx.AsyncClient() as client:
        # Get all configuration
        response = await client.get(f"{BASE_URL}/api/configuration/profiles/{PROFILE_ID}/config")
        print_response(response, f"Get All Configuration for Profile '{PROFILE_ID}'")
        
        # Get costs configuration
        response = await client.get(f"{BASE_URL}/api/configuration/profiles/{PROFILE_ID}/config/costs")
        print_response(response, "Get Costs Configuration")
        
        # Get tariffs configuration
        response = await client.get(f"{BASE_URL}/api/configuration/profiles/{PROFILE_ID}/config/tariffs")
        print_response(response, "Get Tariffs Configuration")
        
        # Get metric weights
        response = await client.get(f"{BASE_URL}/api/configuration/profiles/{PROFILE_ID}/config/metric-weights")
        print_response(response, "Get Metric Weights")


async def test_job_management():
    """Test job management endpoints"""
    async with httpx.AsyncClient() as client:
        # List jobs
        response = await client.get(f"{BASE_URL}/api/jobs/")
        print_response(response, "List All Jobs")
        
        # List jobs with filters
        response = await client.get(f"{BASE_URL}/api/jobs/?profile_id={PROFILE_ID}&limit=5")
        print_response(response, f"List Jobs for Profile '{PROFILE_ID}'")


async def test_vendor_evaluation():
    """Test vendor evaluation endpoint"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "profile_id": PROFILE_ID,
            "mode": "matnr",
            "filters": None,
            "metric_weights": None,
            "include_details": False
        }
        
        response = await client.post(
            f"{BASE_URL}/api/optimization/evaluate-vendors?profile_id={PROFILE_ID}",
            json=request_data
        )
        print_response(response, "Vendor Evaluation Request")
        
        # If async, return job_id for further testing
        if response.status_code == 202:
            data = response.json()
            job_id = data.get("job_id")
            if job_id:
                print(f"\nVendor evaluation job created: {job_id}")
                return job_id
        elif response.status_code == 200:
            # Inline response - show summary
            data = response.json()
            if data.get("status") == "success":
                metadata = data.get("metadata", {})
                print(f"\n✅ Vendor Evaluation Completed:")
                print(f"   - Vendors evaluated: {metadata.get('total_vendors_evaluated', 0)}")
                print(f"   - Materials evaluated: {metadata.get('total_materials_evaluated', 0)}")
                print(f"   - Total combinations: {metadata.get('total_combinations', 0)}")
        return None


async def test_run_optimization():
    """Run procurement allocation optimization"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "profile_id": PROFILE_ID,
            "mode": "matnr",
            "demand_period_days": 365,
            "capacity_buffer_percent": 0.10,
            "constraints": {
                "enforce_multi_supplier": True,
                "min_suppliers_per_material": 2,
                "max_supplier_share": 0.80
            },
            "solver_options": {
                "timeout_seconds": 300,
                "gap_tolerance": 0.01
            }
        }
        
        response = await client.post(
            f"{BASE_URL}/api/optimization/optimize-allocation?profile_id={PROFILE_ID}",
            json=request_data
        )
        print_response(response, "Optimize Allocation Request")
        
        # Should always be async (202)
        if response.status_code == 202:
            data = response.json()
            job_id = data.get("job_id")
            if job_id:
                print(f"\nOptimization job created: {job_id}")
                # Store job ID for later use
                return job_id
        return None


async def test_optimization_summary(job_id: str):
    """Test optimization summary endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/jobs/{job_id}/summary")
        print_response(response, f"Optimization Summary for Job '{job_id}'")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                summary = data.get("summary", {})
                print(f"\n✅ Optimization Summary:")
                print(f"   - Total effective cost: ${summary.get('total_effective_cost', 0):,.2f}")
                print(f"   - Materials optimized: {summary.get('total_materials_optimized', 0)}")
                print(f"   - Allocation changes: {summary.get('total_allocation_changes', 0)}")
                print(f"   - Solver time: {summary.get('solver_time_seconds', 0):.1f}s")


async def test_optimization_allocations(job_id: str):
    """Test optimization allocation results endpoint"""
    async with httpx.AsyncClient() as client:
        # Test first page
        params = {
            "page": 1,
            "page_size": 10
        }
        
        response = await client.get(
            f"{BASE_URL}/api/jobs/{job_id}/allocations",
            params=params
        )
        print_response(response, f"Detailed Allocations for Job '{job_id}' (Page 1)")
        
        if response.status_code == 200:
            data = response.json()
            metadata = data.get("metadata", {})
            allocations = data.get("allocations", [])
            
            print(f"\n✅ Allocation Results:")
            print(f"   - Total allocations: {metadata.get('total_allocations', 0)}")
            print(f"   - Current page: {metadata.get('current_page', 0)}")
            print(f"   - Total pages: {metadata.get('total_pages', 0)}")
            print(f"   - Showing {len(allocations)} allocations")


async def test_compare_policies(optimization_job_id: str):
    """Test policy comparison endpoint"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "profile_id": PROFILE_ID,
            "optimization_job_id": optimization_job_id,
            "mode": "matnr",
            "output_format": "async"  # Use async for full comparison
        }
        
        response = await client.post(
            f"{BASE_URL}/api/optimization/compare-policies?profile_id={PROFILE_ID}",
            json=request_data
        )
        print_response(response, "Compare Policies Request")
        
        # Check if summary or async
        if response.status_code == 200:
            # Summary response
            data = response.json()
            summary = data.get("summary", {})
            print(f"\n✅ Comparison Summary:")
            print(f"   - Historical cost: ${summary.get('total_historical_cost', 0):,.2f}")
            print(f"   - Optimized cost: ${summary.get('total_optimized_cost', 0):,.2f}")
            print(f"   - Total savings: ${summary.get('total_savings', 0):,.2f}")
            print(f"   - Savings percentage: {summary.get('savings_percentage', 0):.1f}%")
            return None
        elif response.status_code == 202:
            # Async response
            data = response.json()
            job_id = data.get("job_id")
            if job_id:
                print(f"\nComparison job created: {job_id}")
                return job_id
        return None


async def test_retrieve_evaluation_results(job_id: str):
    """Test retrieving paginated evaluation results"""
    async with httpx.AsyncClient() as client:
        params = {
            "page": 1,
            "page_size": 5,
            "format": "json"
        }
        
        response = await client.get(
            f"{BASE_URL}/api/jobs/{job_id}/results",
            params=params
        )
        print_response(response, f"Evaluation Results for Job '{job_id}' (Page 1)")
        
        if response.status_code == 200:
            data = response.json()
            metadata = data.get("metadata", {})
            vendors = data.get("data", [])
            
            print(f"\n✅ Evaluation Results:")
            print(f"   - Total records: {metadata.get('total_records', 0)}")
            print(f"   - Current page: {metadata.get('current_page', 0)}")
            print(f"   - Total pages: {metadata.get('total_pages', 0)}")
            print(f"   - Showing {len(vendors)} vendor evaluations")


async def test_download_vendor_evaluation_results(job_id: str):
    """Test downloading vendor evaluation results"""
    async with httpx.AsyncClient() as client:
        print(f"\n{'='*60}")
        print(f"Download Vendor Evaluation Results for Job '{job_id}'")
        print(f"{'='*60}")
        
        # Download as CSV with gzip compression
        params = {
            "format": "csv",
            "compression": "gzip"
        }
        
        filename = f"vendor_evaluation_{job_id}.csv.gz"
        url = f"{BASE_URL}/api/jobs/{job_id}/download"
        
        success = await download_file(client, url, filename, params)
        
        if success:
            print(f"\n✅ Vendor evaluation results downloaded successfully")
        
        # Also try JSON format
        params["format"] = "json"
        filename = f"vendor_evaluation_{job_id}.json.gz"
        
        success = await download_file(client, url, filename, params)


async def test_download_optimization_results(job_id: str):
    """Test downloading optimization allocation results"""
    async with httpx.AsyncClient() as client:
        print(f"\n{'='*60}")
        print(f"Download Optimization Results for Job '{job_id}'")
        print(f"{'='*60}")
        
        filename = f"optimized_allocation_{job_id}.csv"
        url = f"{BASE_URL}/api/jobs/{job_id}/download"
        
        success = await download_file(client, url, filename)
        
        if success:
            print(f"\n✅ Optimization results downloaded successfully")


async def test_download_comparison_results(job_id: str):
    """Test downloading comparison results"""
    async with httpx.AsyncClient() as client:
        print(f"\n{'='*60}")
        print(f"Download Comparison Results for Job '{job_id}'")
        print(f"{'='*60}")
        
        filename = f"comparison_{job_id}.csv"
        url = f"{BASE_URL}/api/jobs/{job_id}/download"
        
        success = await download_file(client, url, filename)
        
        if success:
            print(f"\n✅ Comparison results downloaded successfully")


async def test_complete_pipeline():
    """Test complete pipeline execution"""
    async with httpx.AsyncClient() as client:
        request_data = {
            "profile_id": PROFILE_ID,
            "mode": "matnr",
            "force_regenerate": True
        }
        
        response = await client.post(
            f"{BASE_URL}/api/optimization/pipeline?profile_id={PROFILE_ID}",
            json=request_data
        )
        print_response(response, "Complete Pipeline Request")
        
        # Store initial response data for download URL
        initial_data = None
        if response.status_code == 200:
            initial_data = response.json()
            job_id = initial_data.get("job_id")
            if job_id:
                print(f"\nPipeline job created: {job_id}")
                # Poll for completion
                job_status = await poll_job_status(job_id, timeout=300)
                
                # If job completed successfully, download the results
                if job_status and job_status.get("status") == "completed":
                    download_url = initial_data.get("result_endpoints", {}).get("download")
                    if download_url:
                        print(f"\n{'='*60}")
                        print(f"Download Pipeline Results for Job '{job_id}'")
                        print(f"{'='*60}")
                        
                        # Download pipeline summary JSON
                        filename = f"pipeline_summary_{job_id}.json"
                        url = f"{BASE_URL}{download_url}"
                        
                        success = await download_file(client, url, filename)
                        
                        if success:
                            print(f"\n✅ Pipeline summary downloaded successfully")
                            
                            # Read the pipeline summary to get CSV download URLs
                            summary_path = os.path.join(TEST_OUTPUT_DIR, filename)
                            with open(summary_path, 'r') as f:
                                pipeline_summary = json.load(f)
                            
                            # Download individual CSV files
                            downloads = pipeline_summary.get("downloads", {})
                            
                            # Download vendor evaluation CSV
                            if "vendor_evaluation" in downloads:
                                csv_info = downloads["vendor_evaluation"]
                                if csv_info:
                                    print(f"\n{'='*60}")
                                    print(f"Download Vendor Evaluation CSV")
                                    print(f"{'='*60}")
                                    csv_url = f"{BASE_URL}{csv_info['url']}"
                                    csv_filename = f"vendor_evaluation_{job_id}.csv"
                                    success = await download_file(client, csv_url, csv_filename)
                            
                            # Download optimization CSV
                            if "optimization" in downloads:
                                csv_info = downloads["optimization"]
                                if csv_info:
                                    print(f"\n{'='*60}")
                                    print(f"Download Optimization CSV")
                                    print(f"{'='*60}")
                                    csv_url = f"{BASE_URL}{csv_info['url']}"
                                    csv_filename = f"optimization_allocation_{job_id}.csv"
                                    success = await download_file(client, csv_url, csv_filename)
                            
                            # Download comparison CSV
                            if "comparison" in downloads:
                                csv_info = downloads["comparison"]
                                if csv_info:
                                    print(f"\n{'='*60}")
                                    print(f"Download Comparison CSV")
                                    print(f"{'='*60}")
                                    csv_url = f"{BASE_URL}{csv_info['url']}"
                                    csv_filename = f"comparison_{job_id}.csv"
                                    success = await download_file(client, csv_url, csv_filename)
                            
                            # Download all results as ZIP
                            if "all_results" in downloads:
                                zip_info = downloads["all_results"]
                                if zip_info:
                                    print(f"\n{'='*60}")
                                    print(f"Download All Results as ZIP")
                                    print(f"{'='*60}")
                                    zip_url = f"{BASE_URL}{zip_info['url']}"
                                    zip_filename = f"pipeline_results_{job_id}.zip"
                                    success = await download_file(client, zip_url, zip_filename)


async def poll_job_status(job_id: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """Poll job status until completion"""
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        print(f"\nPolling job status for '{job_id}'...")
        
        while True:
            response = await client.get(f"{BASE_URL}/api/jobs/{job_id}")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                
                print(f"Job Status: {status} (Progress: {progress}%)")
                
                if status in ["completed", "failed", "cancelled"]:
                    print_response(response, f"Final Job Status for '{job_id}'")
                    
                    # If failed, show error details
                    if status == "failed":
                        error = data.get("error", "Unknown error")
                        error_details = data.get("error_details", {})
                        print(f"\n❌ Job Failed: {error}")
                        if error_details:
                            print(f"   Error Type: {error_details.get('type', 'Unknown')}")
                    
                    # If completed, try to get summary
                    elif status == "completed":
                        print(f"\n✅ Job Completed Successfully!")
                        summary_response = await client.get(f"{BASE_URL}/api/optimization/jobs/{job_id}/summary")
                        if summary_response.status_code == 200:
                            print_response(summary_response, "Job Summary")
                    
                    return data  # Return the job status data
            
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for job '{job_id}' to complete")
                return None
            
            await asyncio.sleep(2)  # Poll every 2 seconds


async def main():
    """Run all tests"""
    print("Starting API Tests...")
    print(f"Base URL: {BASE_URL}")
    print(f"Profile ID: {PROFILE_ID}")
    
    # Create test output directory
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    print(f"Test output directory: {TEST_OUTPUT_DIR}")
    
    # Test health endpoints
    print("\n1. Testing Health Endpoints...")
    await test_health_endpoints()
    
    # Test configuration endpoints
    print("\n2. Testing Configuration Endpoints...")
    await test_configuration_endpoints()
    
    # Test job management
    print("\n3. Testing Job Management...")
    await test_job_management()
    
    # Test vendor evaluation with full flow
    print("\n4. Testing Vendor Evaluation (Full Flow)...")
    eval_job_id = await test_vendor_evaluation()
    
    if eval_job_id:
        # Wait for evaluation to complete
        await poll_job_status(eval_job_id, timeout=120)
        
        # Test retrieving paginated results
        print("\n4a. Testing Retrieve Evaluation Results...")
        await test_retrieve_evaluation_results(eval_job_id)
        
        # Test downloading evaluation results
        print("\n4b. Testing Download Evaluation Results...")
        await test_download_vendor_evaluation_results(eval_job_id)
    
    # Run optimization
    print("\n5. Running Optimization Job...")
    opt_job_id = await test_run_optimization()
    
    if opt_job_id:
        # Wait for optimization to complete
        await poll_job_status(opt_job_id, timeout=300)
        
        # Test optimization summary
        print("\n5a. Testing Optimization Summary...")
        await test_optimization_summary(opt_job_id)
        
        # Test optimization allocations
        print("\n5b. Testing Optimization Allocations...")
        await test_optimization_allocations(opt_job_id)
        
        # Test downloading optimization results
        print("\n5c. Testing Download Optimization Results...")
        await test_download_optimization_results(opt_job_id)
        
        # Test policy comparison
        print("\n6. Testing Policy Comparison...")
        comp_job_id = await test_compare_policies(opt_job_id)
        
        if comp_job_id:
            # Wait for comparison to complete
            await poll_job_status(comp_job_id, timeout=120)
            
            # Test downloading comparison results
            print("\n6a. Testing Download Comparison Results...")
            await test_download_comparison_results(comp_job_id)
    
    # Test complete pipeline (optional - takes longer)
    print("\n7. Testing Complete Pipeline (Optional)...")
    response = input("Run complete pipeline test? This may take several minutes. (y/N): ")
    if response.lower() == 'y':
        await test_complete_pipeline()
    else:
        print("Skipping complete pipeline test.")
    
    print("\nAPI Tests Completed!")
    print(f"\nDownloaded files can be found in: {os.path.abspath(TEST_OUTPUT_DIR)}")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())