# Procurement Assistant API Capabilities

This document outlines the API capabilities of the Procurement Assistant system, detailing how each functionality can be externalized as callable API endpoints.

## Core Optimization Pipeline APIs

### 1. Run Complete Optimization Pipeline

**Endpoint:** `POST /api/optimization/pipeline`

**Description:** Executes the complete procurement optimization pipeline for a given profile. This orchestrates vendor evaluation, procurement optimization, and policy comparison in a single workflow. Always runs asynchronously and provides download endpoints for all generated files.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "mode": "matnr",  // Options: "matkl", "matnr", "maktx"
  "force_regenerate": true,  // Force regeneration of all derived files
  "metric_weights": {  // Optional: Custom metric weights for vendor evaluation
    "AvgUnitPriceUSD_Norm": 0.20,
    "PriceVolatility_Norm": 0.15
  },
  "demand_period_days": 365,  // Optional: Demand calculation period (default: 365)
  "filters": null,  // Optional: Filters for vendor evaluation
  "clean_previous_results": false  // Optional: Clean previous results before running
}
```

**Response:**
```json
{
  "status": "accepted",
  "job_id": "pipeline_20240115_123456_abc12345",
  "estimated_duration_seconds": 300,
  "result_endpoints": {
    "status": "/api/jobs/pipeline_20240115_123456_abc12345/status",
    "download": "/api/jobs/pipeline_20240115_123456_abc12345/download",
    "vendor_evaluation_csv": "/api/jobs/pipeline_20240115_123456_abc12345/download/vendor-evaluation",
    "optimization_csv": "/api/jobs/pipeline_20240115_123456_abc12345/download/optimization",
    "comparison_csv": "/api/jobs/pipeline_20240115_123456_abc12345/download/comparison",
    "all_results_zip": "/api/jobs/pipeline_20240115_123456_abc12345/download/all"
  }
}
```

### 1a. Pipeline Summary Structure

When the pipeline completes, the summary JSON contains:

```json
{
  "pipeline_id": "pipeline_profile_1_matnr",
  "profile_id": "profile_1",
  "mode": "matnr",
  "demand_period_days": 365,
  "steps_completed": ["vendor_evaluation", "procurement_optimization", "policy_comparison"],
  "results": {
    "vendor_evaluation": {
      "output_file": "/path/to/vendor_matnr_ranking_tariff_values.csv",
      "summary": {
        "vendors_evaluated": 45,
        "materials_evaluated": 150,
        "combinations_evaluated": 450
      }
    },
    "procurement_optimization": {
      "output_file": "/path/to/optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv",
      "summary": {
        "optimization_status": "optimal",
        "solver_time": 45.2,
        "objective_value": 8500000.00
      }
    },
    "policy_comparison": {
      "output_file": "/path/to/comparison.csv",
      "summary": {
        "comparison_mode": "material_id",
        "total_historical_cost": 10000000.00,
        "total_optimized_cost": 8500000.00,
        "net_economic_saving": 1500000.00,
        "percentage_saving": 15.0
      }
    }
  },
  "key_metrics": {
    "total_economic_saving": 1500000.00,
    "percentage_saving": 15.0,
    "materials_optimized": 150,
    "suppliers_analyzed": 45,
    "allocation_changes": 89
  },
  "downloads": {
    "summary": {
      "url": "/api/jobs/pipeline_20240115_123456_abc12345/download",
      "filename": "pipeline_summary.json",
      "size_bytes": 4567
    },
    "vendor_evaluation": {
      "url": "/api/jobs/pipeline_20240115_123456_abc12345/download/vendor-evaluation",
      "filename": "vendor_evaluation.csv",
      "size_bytes": 123456
    },
    "optimization": {
      "url": "/api/jobs/pipeline_20240115_123456_abc12345/download/optimization",
      "filename": "optimization_allocation.csv",
      "size_bytes": 234567
    },
    "comparison": {
      "url": "/api/jobs/pipeline_20240115_123456_abc12345/download/comparison",
      "filename": "comparison.csv",
      "size_bytes": 345678
    },
    "all_results": {
      "url": "/api/jobs/pipeline_20240115_123456_abc12345/download/all",
      "filename": "pipeline_results_pipeline_20240115_123456_abc12345.zip",
      "size_bytes": 708268
    }
  }
}
```

### 2. Evaluate Vendor Performance

**Endpoint:** `POST /api/optimization/evaluate-vendors`

**Description:** Evaluates vendor performance based on multiple metrics including cost, lead time, and reliability. Can evaluate all vendors/materials or apply specific filters. For large datasets, this returns a job ID for async processing.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "mode": "matnr",  // Required: "matkl", "matnr", or "maktx"
  "filters": {  // Optional: If omitted, evaluates ALL materials and vendors
    "material_groups": ["MATKL001", "MATKL002"],  // Optional: Filter by material groups
    "materials": ["MATNR001", "MATNR002"],  // Optional: Filter by specific materials
    "suppliers": ["LIFNR001", "LIFNR002"],  // Optional: Filter by specific suppliers
    "date_range": {  // Optional: Historical data range for evaluation
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  },
  "metric_weights": {  // Optional: Uses defaults if not provided
    "AvgUnitPriceUSD_Norm": 0.20,
    "PriceVolatility_Norm": 0.15,
    "PriceTrend_Norm": 0.10,
    "TariffImpact_Norm": 0.15,
    "AvgLeadTimeDays_Norm": 0.10,
    "LeadTimeVariability_Norm": 0.10,
    "OnTimeRate_Norm": 0.10,
    "InFullRate_Norm": 0.10
  },
  "cost_components": {  // Optional: All components enabled by default
    "cost_BasePrice": true,
    "cost_Tariff": true,
    "cost_Holding_LeadTime": true,
    "cost_Holding_LTVariability": true,
    "cost_Holding_Lateness": true,
    "cost_Risk_PriceVolatility": true,
    "cost_Impact_PriceTrend": true,
    "cost_Logistics": true
  },
  "output_format": "async",  // Options: "inline" (small datasets), "async" (large datasets, default)
  "include_details": true    // Include detailed cost components in results
}
```

**Response (Async mode - default for large datasets):**
```json
{
  "status": "accepted",
  "job_id": "eval_20240115_123456",
  "estimated_duration_seconds": 30,
  "result_endpoints": {
    "status": "/api/jobs/eval_20240115_123456/status",
    "results": "/api/jobs/eval_20240115_123456/results",
    "download": "/api/jobs/eval_20240115_123456/download"
  }
}
```

**Response (Inline mode - only for small datasets):**
```json
{
  "status": "success",
  "metadata": {
    "total_vendors_evaluated": 150,
    "total_materials_evaluated": 45,
    "total_combinations": 450,
    "filters_applied": false,  // true if any filters were used
    "evaluation_mode": "matnr",
    "pagination": {
      "current_page": 1,
      "total_pages": 1,
      "page_size": 1000,
      "total_records": 450
    }
  },
  "vendors": [
    {
      "rank": 1,
      "supplier_id": "LIFNR001",
      "supplier_name": "ABC Supplier",
      "material_id": "MATNR001",
      "material_description": "Steel Plates",
      "country": "US",
      "effective_cost_per_unit": 125.50,
      "final_score": 0.92,
      "po_line_item_count": 150,
      "metrics": {
        "avg_unit_price": 100.00,
        "tariff_impact_percent": 5.0,
        "logistics_cost": 15.00,
        "lead_time_days": 14,
        "on_time_rate": 0.95,
        "in_full_rate": 0.98
      },
      "cost_components": {
        "cost_BasePrice": 100.00,
        "cost_Tariff": 5.00,
        "cost_Logistics": 15.00,
        "cost_Holding_LeadTime": 2.50,
        "cost_Holding_LTVariability": 1.00,
        "cost_Holding_Lateness": 0.50,
        "cost_Risk_PriceVolatility": 1.00,
        "cost_Impact_PriceTrend": 0.50
      }
    }
  ]
}
```

**Example: Evaluate ALL vendors and materials**
```json
{
  "profile_id": "profile_1",
  "mode": "matnr"
  // No filters - evaluates entire dataset
}
```

**Example: Evaluate specific materials only**
```json
{
  "profile_id": "profile_1",
  "mode": "matnr",
  "filters": {
    "materials": ["MATNR001", "MATNR002", "MATNR003"]
  }
}
```

**Notes:**
- When no filters are provided, the API evaluates ALL vendor-material combinations in the dataset
- Large datasets (>1000 records or >1MB) automatically use async mode
- Results are ranked by effective cost per unit (lowest to highest) by default
- The evaluation considers only materials with multiple suppliers for optimization relevance

### 2a. Retrieve Evaluation Results

**Endpoint:** `GET /api/jobs/{job_id}/results`

**Description:** Retrieves paginated results from an async evaluation job.

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `page_size` (integer): Records per page (default: 100, max: 1000)
- `format` (string): Response format - "json" or "csv" (default: "json")

**Response:**
```json
{
  "status": "success",
  "job_id": "eval_20240115_123456",
  "metadata": {
    "total_records": 5000,
    "current_page": 1,
    "total_pages": 50,
    "page_size": 100
  },
  "data": [
    // ... vendor evaluation records ...
  ]
}
```

### 2b. Download Full Results

**Endpoint:** `GET /api/jobs/{job_id}/download`

**Description:** Downloads complete results as a file (CSV or JSON).

**Query Parameters:**
- `format` (string): File format - "csv" or "json" (default: "csv")
- `compression` (string): Optional compression - "gzip" or "none" (default: "gzip" for files >1MB)

**Response:** Binary file download with appropriate headers:
- `Content-Type: text/csv` or `application/json`
- `Content-Disposition: attachment; filename="vendor_evaluation_20240115_123456.csv.gz"`

### 3. Optimize Procurement Allocation

**Endpoint:** `POST /api/optimization/optimize-allocation`

**Description:** Optimizes supplier allocation using linear programming to minimize total effective cost while meeting demand constraints. Due to the computational intensity and large output size, this operation always runs asynchronously.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "mode": "matnr",
  "demand_period_days": 365,
  "capacity_buffer_percent": 0.10,
  "constraints": {
    "enforce_multi_supplier": true,
    "min_suppliers_per_material": 2,
    "max_supplier_share": 0.80
  },
  "solver_options": {
    "timeout_seconds": 300,  // Maximum solver time (default: 300)
    "gap_tolerance": 0.01    // Acceptable optimality gap (default: 0.01 = 1%)
  }
}
```

**Response:**
```json
{
  "status": "accepted",
  "job_id": "opt_20240115_234567",
  "estimated_duration_seconds": 120,
  "result_endpoints": {
    "status": "/api/jobs/opt_20240115_234567/status",
    "summary": "/api/jobs/opt_20240115_234567/summary",
    "allocations": "/api/jobs/opt_20240115_234567/allocations",
    "download": "/api/jobs/opt_20240115_234567/download"
  }
}
```

### 3a. Get Optimization Summary

**Endpoint:** `GET /api/jobs/{job_id}/summary`

**Description:** Retrieves high-level optimization results summary without detailed allocations.

**Response:**
```json
{
  "status": "success",
  "optimization_status": "optimal",
  "summary": {
    "total_effective_cost": 8500000.00,
    "total_materials_optimized": 150,
    "total_allocation_changes": 89,
    "total_quantity_allocated": 500000,
    "solver_time_seconds": 45.2,
    "optimality_gap": 0.008
  },
  "constraints_satisfied": {
    "demand_met": true,
    "capacity_respected": true,
    "multi_supplier_enforced": true
  },
  "top_changes": [
    {
      "material_id": "MATNR001",
      "description": "Steel Plates",
      "old_primary_supplier": "LIFNR002",
      "new_primary_supplier": "LIFNR001",
      "quantity_shifted": 5000,
      "cost_impact": -125000.00
    }
  ]
}
```

### 3b. Get Detailed Allocations

**Endpoint:** `GET /api/jobs/{job_id}/allocations`

**Description:** Retrieves paginated detailed allocation results.

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `page_size` (integer): Records per page (default: 100, max: 1000)
- `material_filter` (string): Optional material ID filter
- `supplier_filter` (string): Optional supplier ID filter

**Response:**
```json
{
  "status": "success",
  "metadata": {
    "total_allocations": 450,
    "current_page": 1,
    "total_pages": 5,
    "page_size": 100
  },
  "allocations": [
    {
      "supplier_id": "LIFNR001",
      "supplier_name": "ABC Supplier",
      "material_id": "MATNR001",
      "material_description": "Steel Plates",
      "allocated_quantity": 10000,
      "effective_cost_per_unit": 125.50,
      "total_effective_cost": 1255000.00,
      "average_unit_price": 100.00
    }
  ]
}
```

### 3c. Download Optimization Results

**Endpoint:** `GET /api/jobs/{job_id}/download`

**Description:** Downloads the optimization allocation results as CSV.

**Response:** Binary file download
- `Content-Type: text/csv`
- `Content-Disposition: attachment; filename="optimized_allocation_{job_id}.csv"`

**Note:** This only downloads the allocation results. To get comparison data, use the Compare Policies endpoint separately.

### 4. Compare Policies

**Endpoint:** `POST /api/optimization/compare-policies`

**Description:** Compares historical procurement policy with optimized allocation to calculate savings and improvements. Must be run after optimization completes. For large datasets, returns a job ID.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "optimization_job_id": "opt_20240115_234567",  // Required: Job ID from optimization run
  "mode": "matnr",
  "output_format": "async",  // Options: "summary" (lightweight), "async" (full comparison)
  "cost_components": {  // Optional: All components included by default
    "cost_BasePrice": true,
    "cost_Tariff": true,
    "cost_Holding_LeadTime": true,
    "cost_Holding_LTVariability": true,
    "cost_Holding_Lateness": true,
    "cost_Risk_PriceVolatility": true,
    "cost_Impact_PriceTrend": true,
    "cost_Logistics": true
  }
}
```

**Response (Summary mode):**
```json
{
  "status": "success",
  "summary": {
    "total_historical_cost": 10000000.00,
    "total_optimized_cost": 8500000.00,
    "total_savings": 1500000.00,
    "savings_percentage": 15.0,
    "allocation_changes": 45
  },
  "cost_breakdown": {
    "cost_BasePrice": {
      "historical": 7000000.00,
      "optimized": 6000000.00,
      "savings": 1000000.00
    },
    "cost_Tariff": {
      "historical": 700000.00,
      "optimized": 600000.00,
      "savings": 100000.00
    },
    "cost_Logistics": {
      "historical": 1050000.00,
      "optimized": 900000.00,
      "savings": 150000.00
    }
  },
  "top_improvements": [
    {
      "material_id": "MATNR001",
      "material_description": "Steel Plates",
      "historical_supplier": "LIFNR002",
      "optimized_supplier": "LIFNR001",
      "quantity_shift": 5000,
      "cost_reduction": 250000.00
    }
  ]
}
```

**Response (Async mode):**
```json
{
  "status": "accepted",
  "job_id": "comp_20240115_345678",
  "estimated_duration_seconds": 60,
  "result_endpoints": {
    "status": "/api/jobs/comp_20240115_345678/status",
    "summary": "/api/jobs/comp_20240115_345678/summary",
    "download": "/api/jobs/comp_20240115_345678/download"
  }
}
```

### 4a. Download Comparison Results

**Endpoint:** `GET /api/jobs/{job_id}/download`

**Description:** Downloads the full comparison CSV with detailed line-by-line historical vs optimized allocation.


**Response:** Binary file download
- `Content-Type: text/csv`
- `Content-Disposition: attachment; filename="comparison_{job_id}.csv"`
- Contains all vendor-material combinations with historical and optimized quantities, costs, and deltas

## Vendor Selection Assistant APIs

### 5. AI-Powered Vendor Recommendation

**Endpoint:** `POST /api/vendor-selection/ai-recommendation`

**Description:** Provides AI-powered vendor recommendations for a specific material, analyzing top vendors based on effective cost and other performance metrics.

**Request Body:**
```json
{
  "material_description": "EMN-MOTOR",  // Required: Material name/description
  "material_id": "MATNR001",  // Optional: Material ID for more specific identification
  "top_n_vendors": 5,  // Optional: Number of vendors to analyze (default: 5)
  "include_vendor_data": false  // Optional: Include raw vendor data in response (default: false)
}
```

**Response:**
```json
{
  "status": "success",
  "material": {
    "description": "EMN-MOTOR",
    "id": "MATNR001"
  },
  "recommendation_text": "Based on the provided data, the best vendor to procure the 'EMN-MOTOR' from is WaveCrest Labs-24 (US). Here's the reasoning:\n\nEffective Cost per Unit: WaveCrest Labs-24 (US) offers the lowest Effective Cost per Unit at $376.21, which is notably lower than all other vendors.\nBase Price Cost: The Base Price Cost is $335.00, which is competitive and only slightly higher than the lowest in the list, but this is offset by the absence of any Tariff Cost.\nTariff Cost: There is no Tariff Cost ($0.00), unlike some alternatives (e.g., WaveCrest Labs-23 (ID) and EV Parts Inc.-15 (ID), both with tariffs over $32).\nHolding Lead Time Cost: The Holding Lead Time Cost is $8.78, which is moderate and comparable to other vendors.\nOther Costs: Logistics cost is $15.24, and other holding and risk costs are in line with the rest of the field.\nAvg Lead Time Days: The Avg Lead Time Days is 53.13, which is among the shortest in the group.\nOn Time Rate & In Full Rate: The On Time Rate is 0.54 and the In Full Rate is 0.69. While these are not the highest, they are better than several alternatives (e.g., EV Parts Inc.-14 (US) and TechGroup, Inc-34 (US)), and only slightly lower than WaveCrest Labs-23 (ID), which is offset by the higher total cost due to tariffs.\n\nAlternatives:\n- EV Parts Inc.-14 (US) is the next closest in cost ($387.18), but it has a lower In Full Rate (0.41) and On Time Rate (0.43), which could impact supply reliability.\n- WaveCrest Labs-23 (ID) and EV Parts Inc.-15 (ID) have better On Time and In Full Rates, but their Effective Cost per Unit is higher due to significant Tariff Costs.\n\nConclusion:\nWaveCrest Labs-24 (US) is the most cost-effective option, offering the lowest total cost with reasonable delivery performance and no tariff burden. This makes it the best choice for procuring the EMN-MOTOR.",
  "vendor_data_included": false
}
```

**Response with vendor data included:**
```json
{
  "status": "success",
  "material": {
    "description": "EMN-MOTOR",
    "id": "MATNR001"
  },
  "recommendation_text": "...", // Full AI analysis text
  "vendor_data_included": true,
  "vendor_data": [
    {
      "VendorFullID": "WaveCrest Labs-24 (US)",
      "EffectiveCostPerUnit_USD": 376.21,
      "AvgLeadTimeDays_raw": 53.13,
      "OnTimeRate_raw": 0.54,
      "InFullRate_raw": 0.69,
      "cost_BasePrice": 335.00,
      "cost_Tariff": 0.00,
      "cost_Logistics": 15.24,
      "cost_Holding_LeadTime": 8.78,
      "cost_Holding_LTVariability": 3.45,
      "cost_Holding_Lateness": 9.01,
      "cost_Risk_PriceVolatility": 2.63,
      "cost_Impact_PriceTrend": 2.09
    }
    // ... up to 4 more vendors
  ]
}
```

**Notes:**
- The AI analyzes the top N vendors (by lowest effective cost) for the specified material
- Quantity is optional and informational only - it doesn't affect cost calculations
- The recommendation is a comprehensive text analysis discussing costs, performance metrics, and trade-offs
- Set `include_vendor_data: true` to receive the raw vendor data used in the analysis

### 6. Vendor Comparison Table

**Endpoint:** `GET /api/vendor-selection/compare`

**Description:** Generates a comprehensive comparison table of vendors for a specific material.

**Request Parameters:**
- `material_id` (string): Material identifier
- `top_n` (integer): Number of top vendors to compare (default: 5)
- `metrics` (array): List of metrics to include in comparison

**Response:**
```json
{
  "status": "success",
  "comparison": {
    "material": {
      "id": "MATNR001",
      "description": "Steel Plates"
    },
    "vendors": [
      {
        "rank": 1,
        "supplier_id": "LIFNR001",
        "supplier_name": "ABC Supplier",
        "country": "US",
        "metrics": {
          "effective_cost_per_unit": 125.50,
          "base_price_per_unit": 100.00,
          "tariff_impact_percent": 5.0,
          "logistics_cost": 15.00,
          "lead_time_days": 14,
          "on_time_rate": 0.95,
          "in_full_rate": 0.98,
          "po_count": 150
        },
        "is_recommended": true
      }
    ],
    "best_in_class": {
      "best_price": "LIFNR003",
      "fastest_delivery": "LIFNR001",
      "best_on_time": "LIFNR001",
      "best_in_full": "LIFNR002"
    }
  }
}
```

### 7. Cost Component Breakdown

**Endpoint:** `GET /api/vendor-selection/cost-breakdown`

**Description:** Provides detailed cost component breakdown for vendors supplying a specific material.

**Request Parameters:**
- `material_id` (string): Material identifier
- `supplier_ids` (array): List of supplier IDs to analyze
- `cost_components` (array): Specific cost components to include

**Response:**
```json
{
  "status": "success",
  "material": {
    "id": "MATNR001",
    "description": "Steel Plates"
  },
  "cost_breakdown": [
    {
      "supplier_id": "LIFNR001",
      "supplier_name": "ABC Supplier",
      "total_effective_cost": 125.50,
      "components": {
        "base_price": 100.00,
        "tariff": 5.00,
        "logistics": 15.00,
        "holding_lead_time": 2.50,
        "holding_variability": 1.00,
        "holding_lateness": 0.50,
        "risk_price_volatility": 1.00,
        "impact_price_trend": 0.50
      },
      "component_percentages": {
        "base_price": 79.68,
        "tariff": 3.98,
        "logistics": 11.95,
        "holding_lead_time": 1.99,
        "holding_variability": 0.80,
        "holding_lateness": 0.40,
        "risk_price_volatility": 0.80,
        "impact_price_trend": 0.40
      }
    }
  ]
}
```

### 8. Optimization Savings Breakdown

**Endpoint:** `GET /api/optimization/savings-breakdown`

**Description:** Provides detailed breakdown of cost savings achieved by switching from historical to optimal policy.

**Request Parameters:**
- `profile_id` (string): Profile identifier
- `material_id` (string, optional): Filter by specific material
- `supplier_id` (string, optional): Filter by specific supplier

**Response:**
```json
{
  "status": "success",
  "savings_summary": {
    "total_historical_cost": 10000000.00,
    "total_optimized_cost": 8500000.00,
    "total_savings": 1500000.00,
    "savings_percentage": 15.0
  },
  "savings_by_component": {
    "base_price": {
      "amount": 1000000.00,
      "percentage": 66.67
    },
    "tariff": {
      "amount": 100000.00,
      "percentage": 6.67
    },
    "logistics": {
      "amount": 150000.00,
      "percentage": 10.00
    },
    "holding_costs": {
      "amount": 200000.00,
      "percentage": 13.33
    },
    "risk_costs": {
      "amount": 50000.00,
      "percentage": 3.33
    }
  },
  "top_saving_opportunities": [
    {
      "material_id": "MATNR001",
      "material_description": "Steel Plates",
      "savings": 500000.00,
      "percentage_of_total": 33.33,
      "primary_factor": "Switching from high-tariff to domestic supplier"
    }
  ]
}
```

### 9. Actionable To-Do List

**Endpoint:** `GET /api/optimization/action-items`

**Description:** Generates an actionable to-do list showing allocation changes between historical and optimal policies for each material-vendor combination.

**Request Parameters:**
- `profile_id` (string): Profile identifier
- `material_filter` (string, optional): Filter by specific material ID
- `sort_by` (string, optional): Sort criteria - "impact" or "quantity_change" (default: "impact")

**Response:**
```json
{
  "status": "success",
  "action_items": [
    {
      "action_type": "increase",
      "vendor": {
        "LIFNR": "15",
        "name": "EV Parts Inc.",
        "country": "Indonesia"
      },
      "material": {
        "MATNR": "1385",
        "MAKTX": "EMN-MOTOR"
      },
      "allocation_change": {
        "historical": 384,
        "optimal": 1286,
        "delta": 902
      },
      "cost_impact": -359151.71
    },
    {
      "action_type": "increase",
      "vendor": {
        "LIFNR": "14",
        "name": "EV Parts Inc.",
        "country": "United States"
      },
      "material": {
        "MATNR": "1385",
        "MAKTX": "EMN-MOTOR"
      },
      "allocation_change": {
        "historical": 154,
        "optimal": 1064,
        "delta": 910
      },
      "cost_impact": -352281.12
    },
    {
      "action_type": "remove",
      "vendor": {
        "LIFNR": "32",
        "name": "TechGroup, Inc",
        "country": "Indonesia"
      },
      "material": {
        "MANTR": "1385",
        "MAKTX": "EMN-MOTOR"
      },
      "allocation_change": {
        "historical": 479,
        "optimal": 0,
        "delta": -479
      },
      "cost_impact": 215001.98
    }
  ],
  "summary": {
    "total_actions": 7,
    "increase_actions": 4,
    "remove_actions": 3,
    "materials_affected": 1,
    "total_cost_impact": -850329.48,
    "total_units_shifted": 3526
  }
}
```

**Notes:**
- Actions are sorted by cost impact (largest savings first)
- Negative impact values represent cost savings
- Positive impact values represent cost increases (from removing low-cost vendors)
- No implementation steps or timeline details - just the allocation changes

## Configuration APIs

### 10. Update Economic Impact Parameters

**Endpoint:** `PUT /api/configuration/economic-parameters`

**Description:** Updates economic impact parameters for cost calculations.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "parameters": {
    "EIP_ANNUAL_HOLDING_COST_RATE": 0.18,
    "EIP_SafetyStockMultiplierForLTVar_Param": 1.65,
    "EIP_RiskPremiumFactorForPriceVolatility_Param": 0.25,
    "EIP_PlanningHorizonDaysForPriceTrend_Param": 90,
    "PRICE_TREND_CUTOFF_DAYS": 180,
    "EIP_BaseLogisticsCostRate_Param": 0.15,
    "DEMAND_PERIOD_DAYS": 365
  }
}
```

**Response:**
```json
{
  "status": "success",
  "profile_id": "profile_1",
  "updated_parameters": {
    "EIP_ANNUAL_HOLDING_COST_RATE": 0.18,
    "EIP_SafetyStockMultiplierForLTVar_Param": 1.65,
    "EIP_RiskPremiumFactorForPriceVolatility_Param": 0.25,
    "EIP_PlanningHorizonDaysForPriceTrend_Param": 90,
    "PRICE_TREND_CUTOFF_DAYS": 180,
    "EIP_BaseLogisticsCostRate_Param": 0.15,
    "DEMAND_PERIOD_DAYS": 365
  }
}
```

### 11. Get Tariff Configuration

**Endpoint:** `GET /api/configuration/tariffs/{profile_id}`

**Description:** Retrieves current country-specific tariff rates for a given profile.

**Response:**
```json
{
  "status": "success",
  "profile_id": "profile_1",
  "tariffs": {
    "CN": 25.0,
    "US": 0.0,
    "DE": 5.5,
    "IN": 12.0,
    "ID": 10.0,
    "MX": 8.0,
    "TH": 10.0,
    "HU": 5.5,
    "MY": 7.5,
    "VN": 12.0,
    "JP": 0.0,
    "KR": 8.0,
    "SG": 0.0,
    "PH": 10.0,
    "BR": 14.0,
    "AR": 12.0,
    "CL": 6.0,
    "PE": 8.0,
    "CO": 10.0
  },
  "metadata": {
    "last_updated": "2024-01-15T10:30:00Z",
    "total_countries": 19,
    "default_tariff": 0.0
  }
}
```

### 12. Update Tariff Configuration

**Endpoint:** `PUT /api/configuration/tariffs`

**Description:** Updates country-specific tariff rates.

**Request Body:**
```json
{
  "profile_id": "profile_1",
  "tariffs": {
    "CN": 25.0,
    "US": 0.0,
    "DE": 5.5,
    "IN": 12.0,
    "MX": 8.0,
    "TH": 10.0,
    "HU": 5.5,
    "MY": 7.5
  }
}
```

**Response:**
```json
{
  "status": "success",
  "profile_id": "profile_1",
  "updated_tariffs": {
    "CN": 25.0,
    "US": 0.0,
    "DE": 5.5,
    "IN": 12.0,
    "MX": 8.0,
    "TH": 10.0,
    "HU": 5.5,
    "MY": 7.5
  }
}
```

### 13. Download Tariff Configuration

**Endpoint:** `GET /api/configuration/tariffs/{profile_id}/download`

**Description:** Downloads the tariff configuration as a JSON file.

**Response:** Binary file download
- `Content-Type: application/json`
- `Content-Disposition: attachment; filename="tariff_values_{profile_id}.json"`

### 14. Manage Profiles

**Endpoint:** `GET /api/profiles`

**Description:** List all available optimization profiles.

**Response:**
```json
{
  "status": "success",
  "profiles": [
    {
      "id": "profile_1",
      "name": "Standard Configuration",
      "created_at": "2024-01-01T00:00:00Z",
      "last_modified": "2024-01-15T10:30:00Z",
      "is_active": true
    },
    {
      "id": "profile_2",
      "name": "High Tariff Scenario",
      "created_at": "2024-01-10T00:00:00Z",
      "last_modified": "2024-01-12T14:20:00Z",
      "is_active": false
    }
  ],
  "active_profile": "profile_1"
}
```

## Implementation Notes

### Handling Large Datasets
- **Async by Default**: Operations producing >1MB or >1000 records automatically use async processing
- **Job-Based Architecture**: Large operations return job IDs for status tracking and result retrieval
- **Pagination**: All result endpoints support pagination with configurable page sizes
- **Compression**: Automatic gzip compression for downloads >1MB
- **Streaming**: Consider implementing streaming endpoints for real-time data processing
- **Result Storage**: Store results in object storage (S3, Azure Blob) with signed URLs for downloads
- **TTL**: Implement result expiration (e.g., 7 days) to manage storage

### Authentication & Authorization
- All endpoints should implement proper authentication (e.g., OAuth2, API keys)
- Role-based access control for sensitive operations (configuration changes, optimization runs)

### Rate Limiting
- Implement rate limiting for resource-intensive operations (optimization runs)
- Suggested limits: 10 optimization runs per hour, 100 evaluation requests per hour
- Different limits for async operations vs inline requests

### Async Processing
- Long-running operations (full pipeline, optimization, large evaluations) use async processing
- Standard job status endpoint: `GET /api/jobs/{job_id}/status`
- Job status responses include progress percentage when available

### Data Validation
- Validate all input parameters against defined schemas
- Check profile existence and permissions before operations
- Validate material and supplier IDs against master data

### Error Handling
Standard error response format:
```json
{
  "status": "error",
  "error_code": "INVALID_PROFILE",
  "message": "Profile 'profile_xyz' not found",
  "details": {
    "available_profiles": ["profile_1", "profile_2"]
  }
}
```

### Caching Strategy
- Cache vendor rankings for 5 minutes
- Cache optimization results for 15 minutes
- Invalidate cache on configuration changes
- Use cache headers for downloadable results: `Cache-Control: private, max-age=3600`
- Implement ETags for efficient cache validation

### Monitoring & Logging
- Log all API calls with request/response times
- Monitor optimization solver performance
- Track API usage by profile and user

## Example Integration Flow

```python
# 1. Update configuration
response = requests.put(
    f"{API_BASE_URL}/api/configuration/economic-parameters",
    json={
        "profile_id": "profile_1",
        "parameters": {
            "EIP_ANNUAL_HOLDING_COST_RATE": 0.20
        }
    },
    headers={"Authorization": f"Bearer {token}"}
)

# 2. Evaluate all vendors (large dataset)
response = requests.post(
    f"{API_BASE_URL}/api/optimization/evaluate-vendors",
    json={
        "profile_id": "profile_1",
        "mode": "matnr"
        # No filters - evaluates all
    },
    headers={"Authorization": f"Bearer {token}"}
)
eval_job_id = response.json()["job_id"]

# 3. Run optimization (always async)
response = requests.post(
    f"{API_BASE_URL}/api/optimization/optimize-allocation",
    json={
        "profile_id": "profile_1",
        "mode": "matnr",
        "demand_period_days": 365
    },
    headers={"Authorization": f"Bearer {token}"}
)
opt_job_id = response.json()["job_id"]

# 4. Poll for completion
def wait_for_job(job_id):
    while True:
        status = requests.get(
            f"{API_BASE_URL}/api/jobs/{job_id}/status",
            headers={"Authorization": f"Bearer {token}"}
        ).json()
        
        if status["status"] in ["completed", "failed"]:
            return status
        print(f"Progress: {status.get('progress', 0)}%")
        time.sleep(5)

eval_status = wait_for_job(eval_job_id)
opt_status = wait_for_job(opt_job_id)

# 5. Download large result sets
if eval_status["status"] == "completed":
    # Download compressed CSV
    response = requests.get(
        f"{API_BASE_URL}/api/jobs/{eval_job_id}/download",
        params={"format": "csv", "compression": "gzip"},
        headers={"Authorization": f"Bearer {token}"},
        stream=True
    )
    
    with open("vendor_evaluation.csv.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# 6. Get optimization summary (lightweight)
summary = requests.get(
    f"{API_BASE_URL}/api/jobs/{opt_job_id}/summary",
    headers={"Authorization": f"Bearer {token}"}
).json()

print(f"Total savings: ${summary['summary']['total_effective_cost']:,.2f}")

# 7. Get paginated detailed results
page = 1
while True:
    response = requests.get(
        f"{API_BASE_URL}/api/jobs/{opt_job_id}/allocations",
        params={"page": page, "page_size": 100},
        headers={"Authorization": f"Bearer {token}"}
    ).json()
    
    # Process allocations
    for allocation in response["allocations"]:
        print(f"{allocation['material_id']}: {allocation['allocated_quantity']}")
    
    if page >= response["metadata"]["total_pages"]:
        break
    page += 1
```

## Future Enhancements

1. **WebSocket Support**: Real-time updates for long-running operations
2. **Batch Operations**: Process multiple materials/suppliers in single request
3. **Webhook Notifications**: Notify external systems on optimization completion
4. **GraphQL API**: Alternative query interface for complex data relationships
5. **ML Model APIs**: Expose predictive models for demand forecasting and price trends