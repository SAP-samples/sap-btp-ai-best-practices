"""
Structured Cost Analyzer - Returns JSON for Dashboard

This module analyzes cost information from knowledge graphs and returns
structured JSON data with source traceability.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context


def filter_cost_related_data(kg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter KG to only include cost-related nodes and their direct connections.
    
    This significantly reduces the data sent to the LLM while preserving
    all relevant information for cost analysis.
    
    Strategy:
    1. Get all nodes with ID starting with "cost:" or type "Cost"
    2. Find GenericInformation nodes containing cost-related keywords
    2b. Include any nodes whose ID contains 'forecast' or 'volume' (e.g.,
        "genericinformation:volume_forecast_2027-2032")
    3. Get all relationships involving these cost/forecast nodes
    4. Get all nodes connected to these nodes (1-hop)
    5. Preserve metadata structure
    
    Args:
        kg_data: Full knowledge graph dictionary
        
    Returns:
        Filtered KG with same structure but only cost-related data
    """
    nodes = kg_data.get('nodes', [])
    relationships = kg_data.get('relationships', [])
    
    # Step 1: Find direct cost node IDs
    cost_node_ids = {
        n['id'] for n in nodes 
        if n.get('id', '').startswith('cost:') or n.get('type') == 'Cost'
    }
    
    # Step 2: Find GenericInformation nodes with cost-related keywords
    cost_keywords = [
        'cost', 'price', 'pricing', 'fee', 'payment', 'financial', 
        'budget', 'expense', 'surcharge', 'reimbursement', 'invoice',
        'monetary', 'charge', 'tariff', 'rate', 'quotation', 'billing',
        'compensation', 'refund', 'penalty', 'discount', 'rebate'
    ]
    
    for node in nodes:
        if node.get('type') == 'GenericInformation':
            props = node.get('properties', {})
            # Search across ALL property values (not specific fields)
            searchable_text = ' '.join([str(v) for v in props.values()]).lower()
            
            if any(keyword in searchable_text for keyword in cost_keywords):
                cost_node_ids.add(node['id'])

    # Step 2b: Include forecast/volume nodes by ID pattern regardless of keywords
    # These are generally GenericInformation nodes (e.g.,
    # "genericinformation:volume_forecast_2027-2032")
    for node in nodes:
        node_id = str(node.get('id', ''))
        id_lower = node_id.lower()
        if 'forecast' in id_lower or 'volume' in id_lower:
            cost_node_ids.add(node_id)
    
    if not cost_node_ids:
        print("Warning: No cost-related nodes found in the knowledge graph")
        return kg_data  # Return original if no cost nodes found
    
    # Step 3: Find all relationships involving cost nodes
    cost_relationships = []
    connected_node_ids = set()
    
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        
        # If either end connects to a cost node, include it
        if source in cost_node_ids or target in cost_node_ids:
            cost_relationships.append(rel)
            # Track the connected nodes
            connected_node_ids.add(source)
            connected_node_ids.add(target)
    
    # Step 4: Get all relevant nodes (cost nodes + connected nodes)
    relevant_nodes = [
        n for n in nodes 
        if n['id'] in connected_node_ids
    ]
    
    # Step 5: Return filtered KG with same structure
    return {
        'nodes': relevant_nodes,
        'relationships': cost_relationships,
        'metadata': kg_data.get('metadata', {}),
        'export_metadata': kg_data.get('export_metadata', {})
    }


def analyze_costs_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    use_cost_filter: bool = True,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze costs and return structured JSON data
    
    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for the analysis
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        use_cost_filter: Whether to filter KG to cost-related data (default: True)
        
    Returns:
        Structured dictionary with cost analysis
    """
    print(f"Loading KG from: {kg_json_path}")
    
    # Load the full KG JSON
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    original_nodes = len(kg_data.get('nodes', []))
    original_relationships = len(kg_data.get('relationships', []))
    original_size = len(json.dumps(kg_data))
    
    print(f"Loaded KG with {original_nodes} nodes and {original_relationships} relationships")
    
    # Apply cost filtering if enabled
    if use_cost_filter:
        print("Applying cost-focused filtering...")
        kg_data = filter_cost_related_data(kg_data)
        
        filtered_nodes = len(kg_data.get('nodes', []))
        filtered_relationships = len(kg_data.get('relationships', []))
        filtered_size = len(json.dumps(kg_data))
        
        reduction_pct = (1 - filtered_size/original_size) * 100
        node_reduction_pct = (1 - filtered_nodes/original_nodes) * 100
        
        print(f"After filtering to cost-related data:")
        print(f"  - Nodes: {filtered_nodes} (reduced by {node_reduction_pct:.1f}%)")
        print(f"  - Relationships: {filtered_relationships}")
        print(f"  - Data size reduction: {reduction_pct:.1f}%")
        print(f"  - Estimated token savings: ~{(original_size - filtered_size) // 4:,} tokens")
    
    # Build the prompt
    supplier_context = f"Focus on costs related to supplier: {supplier_name}\n" if supplier_name else ""
    
    # Adjust prompt intro based on whether filtering is applied
    if use_cost_filter:
        prompt_intro = """Analyze the knowledge graph containing COST-FOCUSED DATA and return a JSON response with cost information.

This is a filtered subset of the full knowledge graph, containing:
- All cost nodes and nodes with cost-related information
- Related parts, suppliers, organizations, and specifications  
- All relevant relationships for cost analysis

The data has been pre-filtered to focus on costs and their associated information.

STRICT CLIENT SCOPE FOR QUOTATION ANALYSIS:
- Only extract and use volumes/quantities for clients SEU and PurchasingOrganization2
- Explicitly EXCLUDE SLA and Navistar
- Do NOT use aggregate or 'Total' volumes that include other clients; prefer client-specific SEU/PurchasingOrganization2 values
- When both client-specific and aggregated totals exist, ignore the aggregates"""
    else:
        prompt_intro = """Analyze the knowledge graph and return a JSON response with cost information.

STRICT CLIENT SCOPE FOR QUOTATION ANALYSIS:
- Only extract and use volumes/quantities for clients SEU and PurchasingOrganization2
- Explicitly EXCLUDE SLA and Navistar
- Do NOT use aggregate or 'Total' volumes that include other clients; prefer client-specific SEU/PurchasingOrganization2 values
- When both client-specific and aggregated totals exist, ignore the aggregates"""
    
    base_prompt = f"""{prompt_intro}

{supplier_context}
Return ONLY valid JSON (no markdown, no explanations outside JSON) with this exact structure:
{{
  "total_cost": {{
    "amount": <number>,
    "currency": "EUR",
    "description": "explanation of what this cost includes and who is paying which part of the cost",
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
  }},
  "cost_breakdown": [
    // CRITICAL: Create SEPARATE entries for EACH individual cost category found
    // DO NOT group multiple costs under umbrella terms like "Project Services"
    // Example: If you find "Design: 147,000, Process: 24,000, Quality: 28,000"
    // Create THREE separate entries, NOT one combined entry
    {{
      "category": "specific category name (e.g., 'Design', 'Process', 'Quality', 'Testing', 'Tooling')",
      "amount": <number>,
      "currency": "EUR",
      "description": "what this specific cost covers",
      "percentage_of_total": <number>,
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
    // Repeat for EACH distinct cost category
  ],
  "unit_costs": {{
    "price_per_unit": <number>,
    "currency": "EUR",
    "volume_assumptions": "description of volume conditions",
    "price_variations": [
      {{
        "condition": "e.g., at 100% volume",
        "price": <number>,
        "sources": [{{"filename": "...", "chunk_id": "..."}}]
      }}
    ],
    "sources": [{{"filename": "...", "chunk_id": "..."}}]
  }},
  "volume_forecast": [
    {{
      "timeframe": "e.g., 2027",
      "quantity": <number>,
      "unit": "units",
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
  ],
  "estimated_total_product_cost": {{
    "amount": <number>,
    "currency": "EUR",
    "methodology": "explain how computed (e.g., sum of forecast quantities × unit price)",
    "sources": [{{"filename": "...", "chunk_id": "..."}}]
  }},
  "cost_dependencies": [
    {{
      "description": "explanation of cost dependencies or relationships",
      "impact": "how this affects total cost",
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
  ],
  "payment_terms": {{
    "description": "payment conditions and terms",
    "key_points": ["point 1", "point 2"],
    "sources": [{{"filename": "...", "chunk_id": "..."}}]
  }},
  "cost_risks": [
    {{
      "risk": "description of cost-related risk",
      "impact": "potential financial impact",
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
  ],
  "opportunities": [
    {{
      "opportunity": "cost reduction or negotiation opportunity",
      "potential_savings": "estimated savings if available",
      "approach": "how to achieve this",
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
  ],
  "summary": {{
    "total_project_cost": <number>,
    "cost_competitiveness": "assessment of cost position",
    "key_insights": ["insight 1", "insight 2", "insight 3"]
  }}
}}

IMPORTANT:
1. Extract the exact filename and chunk_id from the metadata in the knowledge graph
2. All monetary amounts must be numbers, not strings
3. Include sources for every data point
4. Perform a complete breakdown of costs, not just totals
5. If you need to calculate or estimate, explain the methodology in descriptions
6. Identify all cost components mentioned in the knowledge graph
7. For cost_breakdown: Create INDIVIDUAL entries for each distinct cost category
8. NEVER combine multiple costs under generic terms like "Project Services" or "Development"
9. If a document lists multiple costs (e.g., "Design: X EUR, Process: Y EUR, Quality: Z EUR"), 
   create SEPARATE entries for Design, Process, and Quality
10. Common individual categories include: Design, Process, Quality, Testing, Tooling, 
    Project Management, Purchasing, Logistics, etc.
11. Volumes/forecasts MUST be restricted to clients "SEU" and "PurchasingOrganization2" only (case-insensitive). Explicitly EXCLUDE "SLA" and "Navistar". Extract their per-year quantities into volume_forecast. Prefer per-client values; DO NOT use aggregate totals that include other clients. If multiple values exist for a client (e.g., offered vs total), choose the highest forecast within that client (not cross-client totals)
12. Use yearly forecasts, not timeframes like "2027-2032"
13. Compute estimated_total_product_cost as: sum(volume_forecast.quantity) × unit_costs.price_per_unit, if both are available. Provide methodology and sources.

Knowledge Graph:
{json.dumps(kg_data, indent=2)}

Return only the JSON object, no additional text."""
    
    # Apply PurchasingOrganization business context to the prompt
    prompt = format_prompt_with_business_context(base_prompt, analysis_type="cost")
    
    print(f"Prompt size: ~{len(prompt):,} characters")
    selected_model = model_name or "gemini-2.5-pro"
    print(f"Sending to {selected_model} for structured analysis...")
    
    # Get LLM and analyze
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        
        # Extract the text content from the response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Handle different response types
        if isinstance(content, list):
            # If content is a list, join it or take the first element
            if content:
                content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
            else:
                content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Clean the response to ensure it's valid JSON
        # Remove any markdown code blocks if present
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {content[:500]}...")
            # Return a fallback structure
            result = {
                "error": "Failed to parse LLM response",
                "raw_response": content,
                "total_cost": {"amount": 0, "currency": "EUR", "sources": []},
                "cost_breakdown": [],
                "unit_costs": {"price_per_unit": 0, "currency": "EUR", "sources": []},
                "volume_forecast": [],
                "estimated_total_product_cost": {"amount": 0, "currency": "EUR", "sources": []},
                "summary": {"total_project_cost": 0, "key_insights": ["Error in parsing"]}
            }

        # Fallback computation for estimated_total_product_cost if possible
        try:
            unit_costs = result.get('unit_costs', {}) or {}
            unit_price = unit_costs.get('price_per_unit')
            try:
                unit_price = float(unit_price) if unit_price is not None else None
            except Exception:
                unit_price = None

            volume_list = result.get('volume_forecast', []) or []
            total_units = 0.0
            for entry in volume_list:
                qty = entry.get('quantity')
                try:
                    qty_val = float(qty)
                except Exception:
                    qty_val = 0.0
                total_units += qty_val

            est_obj = result.get('estimated_total_product_cost') or {}
            est_amount = est_obj.get('amount') if isinstance(est_obj, dict) else None
            try:
                est_amount_val = float(est_amount) if est_amount is not None else 0.0
            except Exception:
                est_amount_val = 0.0

            if (not est_amount_val or est_amount_val == 0.0) and unit_price is not None and total_units > 0:
                combined_sources = []
                # Collect unit cost sources
                for s in unit_costs.get('sources', []) or []:
                    if isinstance(s, dict):
                        combined_sources.append({
                            "filename": s.get("filename", ""),
                            "chunk_id": s.get("chunk_id", "")
                        })
                # Collect volume sources
                for entry in volume_list:
                    for s in entry.get('sources', []) or []:
                        if isinstance(s, dict):
                            combined_sources.append({
                                "filename": s.get("filename", ""),
                                "chunk_id": s.get("chunk_id", "")
                            })

                computed_amount = unit_price * total_units
                result['estimated_total_product_cost'] = {
                    "amount": round(computed_amount, 2),
                    "currency": unit_costs.get('currency', 'EUR'),
                    "methodology": f"Computed as unit price (EUR {unit_price}) × total forecast quantity ({total_units})",
                    "sources": combined_sources
                }
        except Exception as e:
            print(f"Warning: Failed fallback computation for estimated_total_product_cost: {e}")
        
        print("Analysis complete!")
        
        # Save to file if requested
        if save_to_file and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Structured analysis saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        raise


def main():
    """Test the structured analyzer"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    kg_path = str(project_root / "docs/business/supplier 1/supplier_1_kg_image/unified_kg_20250723_195008.json")
    
    try:
        result = analyze_costs_structured(
            kg_json_path=kg_path,
            supplier_name="SupplierA",
            save_to_file=True,
            output_path="output/structured_cost_analysis_suppliera.json"
        )
        
        print("\n=== STRUCTURED COST ANALYSIS ===")
        print(f"Total Cost: {result['total_cost']['amount']} {result['total_cost']['currency']}")
        print(f"Cost Breakdown: {len(result.get('cost_breakdown', []))} categories")
        print(f"Unit Price: {result['unit_costs']['price_per_unit']} {result['unit_costs']['currency']}")
        
    except FileNotFoundError:
        print(f"File not found: {kg_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()