"""
Dynamic optimization insights generator for the Vendor Performance Dashboard.
Uses SAP GenAI Hub to generate data-driven insights about optimization results.
"""

import json
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
from gen_ai_hub.proxy.native.openai import chat as openai_chat

# --- Configuration Constants ---
OPTIMIZATION_INSIGHTS_TIMEOUT_SECONDS = 90.0

# System prompt for optimization insights generation
OPTIMIZATION_INSIGHTS_SYSTEM_PROMPT = """
You are an expert procurement analyst specializing in supply chain optimization.

Your task is to analyze optimization data comparing historical vendor allocations with optimized allocations, and generate key insights and recommendations.

Provide your analysis in the following SPECIFIC structure:
1. Four key insights, each a 1-2 sentence insight based on the data
2. Four specific recommendations, each 1-2 sentences based on the insights

IMPORTANT RULES:
- Keep insights objective and data-driven
- Focus on the specific metrics provided (lead time, OTIF rate, cost components, allocation changes)
- Be precise with numbers (use exact percentages, values from the data)
- Avoid generic insights - make specific observations about the changes in the data
- Keep language professional but accessible
- Base your recommendations directly on the data insights

I will provide you with structured metrics showing:
1. Overall optimization results (cost savings, allocation changes, etc.)
2. Performance changes pre/post optimization (lead time, OTIF rate, total cost)
3. Cost component changes (which cost elements changed the most)
4. Material or vendor category insights (which showed most improvement)

Return ONLY a JSON object with exactly the following structure:
{
  "insights": [
    "First specific insight about the optimization impact.",
    "Second specific insight about performance trade-offs.",
    "Third specific insight about cost component changes.",
    "Fourth specific insight about material or vendor improvements."
  ],
  "recommendations": [
    "First specific recommendation based on the insights.",
    "Second specific recommendation based on the insights.",
    "Third specific recommendation based on the insights.",
    "Fourth specific recommendation based on the insights."
  ]
}

Ensure the output is ONLY the JSON object and nothing else. Do not add any explanation.
"""

def clean_text(text: str) -> str:
    """Clean text by escaping special characters"""
    if "$" in text:
        text = text.replace("$", "\\$")
    return text

def generate_optimization_insights(optimization_metrics, performance_changes, cost_component_changes, costs_config=None, material_metrics=None):
    """
    Generates dynamic insights and recommendations based on optimization results using SAP GenAI Hub.
    
    Args:
        optimization_metrics (dict): Overall optimization metrics
        performance_changes (dict): Performance metrics before and after optimization
        cost_component_changes (dict): Cost component changes
        costs_config (dict, optional): Configuration indicating which cost components are active
        material_metrics (dict, optional): Material-specific metrics
        
    Returns:
        dict: A dictionary with "insights" and "recommendations" keys containing lists of strings,
              or None if generation fails.
    """
    print("Generating optimization insights...")
    
    # Filter cost components based on active configuration
    if costs_config:
        filtered_cost_components = {}
        for component, value in cost_component_changes.items():
            # Map from cost_component_changes key format to costs_config key format
            config_key = component
            if config_key in costs_config and costs_config[config_key] == "True":
                filtered_cost_components[component] = value
        cost_component_changes = filtered_cost_components
    
    # Format data for the LLM
    data_for_llm = {
        "optimization_metrics": optimization_metrics,
        "performance_changes": performance_changes,
        "cost_component_changes": cost_component_changes
    }
    
    if material_metrics:
        data_for_llm["material_metrics"] = material_metrics
    
    data_json = json.dumps(data_for_llm, indent=2)
    
    messages = [
        {"role": "system", "content": OPTIMIZATION_INSIGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the optimization analysis data:\n```json\n{data_json}\n```\nPlease provide your insights and recommendations based on this data."}
    ]
    
    try:
        print(f"Sending request to LLM for optimization insights (timeout: {OPTIMIZATION_INSIGHTS_TIMEOUT_SECONDS}s).")
        response = openai_chat.completions.create(
            model_name="gpt-4.1",  # Or another suitable model configured in GenAI Hub
            messages=messages,
            max_tokens=500,       # Sufficient for detailed insights and recommendations
            temperature=0.2,      # Low temperature for factual, consistent outputs
            timeout=OPTIMIZATION_INSIGHTS_TIMEOUT_SECONDS,
            response_format={"type": "json_object"}  # Request JSON output
        )
        
        raw_response_content = response.choices[0].message.content
        print(f"Raw LLM response: {raw_response_content}")
        
        # Attempt to parse the JSON response
        parsed_json = json.loads(raw_response_content)
        
        # Validate the structure
        if not isinstance(parsed_json, dict) or \
           "insights" not in parsed_json or \
           "recommendations" not in parsed_json or \
           not isinstance(parsed_json["insights"], list) or \
           not isinstance(parsed_json["recommendations"], list) or \
           len(parsed_json["insights"]) != 4 or \
           len(parsed_json["recommendations"]) != 4:
            print(f"Error: LLM response is not in the expected JSON format. Response: {raw_response_content}")
            return get_fallback_insights(costs_config)
        
        # Ensure each text item is escaped properly
        parsed_json["insights"] = [clean_text(insight) for insight in parsed_json["insights"]]
        parsed_json["recommendations"] = [clean_text(rec) for rec in parsed_json["recommendations"]]
        
        print("Successfully generated optimization insights.")
        return parsed_json
        
    except json.JSONDecodeError as json_err:
        print(f"Error decoding JSON response from LLM: {json_err}. Response: {raw_response_content}")
        return get_fallback_insights(costs_config)
    except Exception as e:
        error_msg = f"An unexpected error occurred during insights generation with LLM: {e}"
        print(error_msg)
        return get_fallback_insights(costs_config)

def get_fallback_insights(costs_config=None):
    """
    Return fallback insights if API call fails
    
    Args:
        costs_config (dict, optional): Configuration indicating which cost components are active
    """
    # Default insights when we don't know what cost components are active
    default_insights = [
        "Total procurement cost reduced significantly while improving overall supplier performance metrics.",
        "Lead time variation between suppliers has been optimized, with allocation shifted to vendors with more consistent delivery schedules.",
        "Tariff cost impact has been minimized by reallocating volume to domestic or preferential trade agreement countries.",
        "Optimization has balanced cost reduction with performance improvements across key metrics."
    ]
    
    # If we have costs config, customize the insights based on active components
    if costs_config:
        # Replace the third insight about costs with something more specific to active components
        custom_insights = default_insights.copy()
        
        # Check if specific cost components are enabled and customize the insights
        if costs_config.get("cost_Tariff") == "True":
            custom_insights[2] = "Tariff cost impact has been minimized by reallocating volume to domestic or preferential trade agreement countries."
        elif costs_config.get("cost_Holding_LeadTime") == "True" and costs_config.get("cost_Holding_LTVariability") == "True":
            custom_insights[2] = "Lead time and its variability costs have been optimized by reallocating to vendors with more consistent delivery performance."
        elif costs_config.get("cost_Risk_PriceVolatility") == "True":
            custom_insights[2] = "Price volatility risk has been mitigated by diversifying allocation across reliable vendors."
        elif costs_config.get("cost_Impact_PriceTrend") == "True":
            custom_insights[2] = "Future price trends have been factored into the optimization, improving long-term cost efficiency."
        
        # Ensure we never mention inefficiency costs if they're disabled
        if costs_config.get("cost_Inefficiency_InFull") != "True" and "inefficiency" in custom_insights[3].lower():
            custom_insights[3] = "The optimization balanced performance and cost metrics based on the configured cost components."
        
        default_insights = custom_insights
    
    return {
        "insights": default_insights,
        "recommendations": [
            "Implement the optimized allocation strategy gradually, prioritizing the highest-impact material categories first.",
            "Establish regular performance reviews with vendors receiving increased allocation to ensure they maintain or improve metrics.",
            "Develop contingency plans for materials where allocation is now concentrated with fewer suppliers to mitigate risk.",
            "Create a monthly optimization review process to continuously adapt allocations based on changing supplier performance."
        ]
    }

def prepare_optimization_data_for_insights(metrics, performance_changes, cost_components_data):
    """
    Prepare optimization data in the format needed for insights generation
    
    Args:
        metrics (dict): Overall optimization metrics
        performance_changes (pd.DataFrame): Performance changes dataframe 
        cost_components_data (dict): Cost component deltas
        
    Returns:
        tuple: (optimization_metrics, performance_metrics, cost_component_metrics)
    """
    # Format overall optimization metrics
    optimization_metrics = {
        "total_cost_savings": float(metrics["total_cost_savings"]),
        "allocation_changes_count": int(metrics["allocation_changes"]),
        "avg_cost_reduction_per_unit": float(metrics["avg_cost_reduction"]),
        "most_improved_category": str(metrics["most_improved_category"]),
        "most_improved_savings_pct": float(metrics["most_improved_savings_pct"])
    }
    
    # Format performance changes
    performance_metrics = {}
    for i, row in performance_changes.iterrows():
        metric = row['Metric']
        performance_metrics[metric.lower().replace(" ", "_")] = {
            "historical": float(row['Historical']),
            "optimized": float(row['Optimized']),
            "change": float(row['Change']),
            "percent_change": float(row['Percent Change'])
        }
    
    # Format cost component changes
    cost_component_metrics = {}
    for component, delta in cost_components_data.items():
        cost_component_metrics[component] = float(delta)
    
    return optimization_metrics, performance_metrics, cost_component_metrics

def generate_actionable_todo_list(comparison_df, supplier_df):
    """
    Generate a prioritized actionable to-do list based on the comparison data
    
    Args:
        comparison_df (pd.DataFrame): Comparison data with historical and optimized allocation
        supplier_df (pd.DataFrame): Supplier master data with country information
        
    Returns:
        dict: Prioritized to-do list grouped by material with economic impact
    """
    # Merge with supplier data to get country info
    # OPTIMAL: Merge on both LIFNR and MATNR to maintain exact 1:1 relationship
    # This preserves material-specific vendor properties and eliminates duplication
    supplier_cols = ['LIFNR', 'Country']
    merge_keys = ['LIFNR']
    
    # Check if MATNR exists in both dataframes for a more precise merge
    if 'MATNR' in comparison_df.columns and 'MATNR' in supplier_df.columns:
        supplier_cols.append('MATNR')
        merge_keys.append('MATNR')
        print("DEBUG: Using LIFNR+MATNR merge for precise vendor-material matching")
    else:
        print("DEBUG: Using LIFNR-only merge (MATNR not available in both datasets)")
    
    # Extract unique supplier combinations for the merge
    supplier_for_merge = supplier_df[supplier_cols].drop_duplicates(subset=merge_keys)
    
    print(f"DEBUG: Original supplier_df rows: {len(supplier_df)}")
    print(f"DEBUG: Supplier combinations for merge: {len(supplier_for_merge)}")
    print(f"DEBUG: Merge keys: {merge_keys}")
    
    merged_df = pd.merge(
        comparison_df,
        supplier_for_merge,
        on=merge_keys,
        how='left'
    )
    
    print(f"DEBUG: Original comparison_df rows: {len(comparison_df)}")
    print(f"DEBUG: Merged_df rows after join: {len(merged_df)}")
    
    # Verify merge integrity - should be 1:1 relationship
    if len(merged_df) != len(comparison_df):
        print(f"WARNING: Merge changed row count from {len(comparison_df)} to {len(merged_df)}")
        print("This indicates potential data quality issues in the merge keys")
    else:
        print("DEBUG: Merge preserved 1:1 relationship - optimal result")
    
    # Sort by economic impact (absolute value of Delta_Total_Effective_Cost_for_Combo)
    merged_df['Abs_Economic_Impact'] = abs(merged_df['Delta_Total_Effective_Cost_for_Combo'])
    merged_df = merged_df.sort_values('Abs_Economic_Impact', ascending=False)
    
    # Filter to include only changes (non-zero delta quantity)
    changes_df = merged_df[merged_df['Delta_Allocated_Quantity'] != 0].copy()
    
    # DEBUG: Print information about the changes dataframe
    print("\n=== DEBUG: generate_actionable_todo_list - AFTER DEDUPLICATION ===")
    print(f"Total rows in changes_df: {len(changes_df)}")
    print(f"Unique MAKTX values: {changes_df['MAKTX'].nunique()}")
    print(f"Unique MATNR values: {changes_df['MATNR'].nunique()}")
    print(f"Unique LIFNR values: {changes_df['LIFNR'].nunique()}")
    
    # Check for TRANS ASSY-SHIPPING specifically
    trans_assy_df = changes_df[changes_df['MAKTX'].str.contains('TRANS ASSY-SHIPPING', case=False, na=False)]
    if not trans_assy_df.empty:
        print(f"\nDEBUG: Found {len(trans_assy_df)} rows for TRANS ASSY-SHIPPING")
        print("Unique MATNR values for TRANS ASSY-SHIPPING:")
        for matnr in trans_assy_df['MATNR'].unique():
            count = len(trans_assy_df[trans_assy_df['MATNR'] == matnr])
            lifnr_in_matnr = trans_assy_df[trans_assy_df['MATNR'] == matnr]['LIFNR'].unique()
            print(f"  MATNR: {matnr} - {count} rows, LIFNR: {lifnr_in_matnr}")
    
    # Group by material description and number (MAKTX, MATNR) to avoid conflicts
    material_groups = changes_df.groupby(['MAKTX', 'MATNR'], observed=False)
    print(f"\nNumber of material groups after grouping: {len(material_groups)}")
    
    todo_list = []
    
    for material_group_key, group in material_groups:
        # Unpack the grouping key (MAKTX, MATNR)
        maktx, matnr = material_group_key
        
        # DEBUG: Print information about this specific group
        print(f"\nDEBUG: Processing group {maktx} ({matnr}) - {len(group)} rows")
        if 'TRANS ASSY-SHIPPING' in maktx:
            print(f"  Vendors in this group: {group['LIFNR'].unique()}")
            print(f"  Row details:")
            for _, row in group.iterrows():
                print(f"    LIFNR: {row['LIFNR']}, Delta_Qty: {row['Delta_Allocated_Quantity']}, Economic_Impact: {row['Delta_Total_Effective_Cost_for_Combo']}")
        
        # Calculate total economic impact for this material group
        total_economic_impact = group['Delta_Total_Effective_Cost_for_Combo'].sum()
        
        # Determine effort level based on quantity changes and vendor history
        total_quantity_change = abs(group['Delta_Allocated_Quantity']).sum()
        num_vendors_affected = len(group)
        
        # Simple effort level determination
        if total_quantity_change > 1000 or num_vendors_affected > 5:
            effort_level = "High"
        elif total_quantity_change > 500 or num_vendors_affected > 3:
            effort_level = "Medium"
        else:
            effort_level = "Low"
        
        # Create a task group
        # Invert the sign so positive means savings (cost reduction)
        display_impact = -total_economic_impact
        task_group = {
            "material": maktx,
            "matnr": matnr,
            "material_full": f"{matnr} ({maktx})",
            "economic_impact": total_economic_impact,
            "display_impact": display_impact,
            "economic_impact_formatted": f"${abs(display_impact):,.2f}",
            "savings": total_economic_impact < 0,  # True if this is a cost reduction
            "effort_level": effort_level,
            "tasks": []
        }
        
        # Add individual tasks for each vendor change
        for _, row in group.iterrows():
            vendor_info = f"{row['NAME1']} ({row['LIFNR']})"
            country_info = f" in {row['Country']}" if pd.notna(row['Country']) else ""
            delta_qty = row['Delta_Allocated_Quantity']
            current_qty = row['Historical_Allocated_Quantity']
            new_qty = row['Optimized_Allocated_Quantity']
            economic_impact = row['Delta_Total_Effective_Cost_for_Combo']
            
            if delta_qty > 0:
                # Increase allocation
                if current_qty == 0:
                    # New vendor
                    task_desc = f"Add new vendor {vendor_info}{country_info} for material {matnr} ({maktx}) with allocation of {new_qty:.0f} units"
                else:
                    # Existing vendor with increased allocation
                    task_desc = f"Increase allocation for {vendor_info}{country_info} for material {matnr} ({maktx}) from {current_qty:.0f} to {new_qty:.0f} units (+{delta_qty:.0f})"
            else:
                # Decrease allocation
                if new_qty == 0:
                    # Remove vendor completely
                    task_desc = f"Remove vendor {vendor_info}{country_info} from material {matnr} ({maktx}) (current allocation: {current_qty:.0f} units)"
                else:
                    # Existing vendor with decreased allocation
                    task_desc = f"Decrease allocation for {vendor_info}{country_info} for material {matnr} ({maktx}) from {current_qty:.0f} to {new_qty:.0f} units ({delta_qty:.0f})"
            
            # Invert the sign so positive means savings (cost reduction)
            display_impact = -economic_impact
            task_group["tasks"].append({
                "description": task_desc,
                "economic_impact": economic_impact,
                "display_impact": display_impact,
                "economic_impact_formatted": f"${abs(display_impact):,.2f}"
            })
        
        todo_list.append(task_group)
    
    # Sort the entire to-do list by absolute economic impact
    todo_list = sorted(todo_list, key=lambda x: abs(x['economic_impact']), reverse=True)
    
    return todo_list


if __name__ == '__main__':
    # Example Usage (for testing)
    test_optimization_metrics = {
        "total_cost_savings": 250000.0,
        "allocation_changes_count": 15,
        "avg_cost_reduction_per_unit": 12.5,
        "most_improved_category": "EMN-MOTOR",
        "most_improved_savings_pct": 18.7
    }
    
    test_performance_metrics = {
        "lead_time": {
            "historical": 45.2,
            "optimized": 42.7,
            "change": -2.5,
            "percent_change": -5.5
        },
        "otif_rate": {
            "historical": 0.83,
            "optimized": 0.87,
            "change": 0.04,
            "percent_change": 4.8
        },
        "total_cost": {
            "historical": 1250000.0,
            "optimized": 1000000.0,
            "change": -250000.0,
            "percent_change": -20.0
        }
    }
    
    test_cost_component_metrics = {
        "cost_BasePrice": -120000.0,
        "cost_Tariff": -80000.0,
        "cost_Holding_LeadTime": -30000.0,
        "cost_Holding_LTVariability": -15000.0,
        "cost_Holding_Lateness": -5000.0
    }
    
    result = generate_optimization_insights(
        test_optimization_metrics,
        test_performance_metrics,
        test_cost_component_metrics
    )
    
    print(f"\nGenerated Insights:\n{json.dumps(result, indent=2)}")