"""
Structured Supplier Comparator - Returns JSON for Dashboard

This module compares two suppliers using their knowledge graphs and analysis results,
returning structured JSON data for dashboard visualization.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context


def coerce_to_dict(value: Any) -> Dict[str, Any]:
    """Return a dict if possible; parse JSON string values; otherwise return {}.

    This helps robustly handle inputs that may be provided as serialized JSON strings
    or already-structured dicts, avoiding attribute errors when calling .get.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def coerce_to_list(value: Any) -> List[Any]:
    """Return a list if possible; parse JSON string lists; wrap other scalars; otherwise []."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [value]
    if isinstance(value, dict):
        return [value]
    if value is None:
        return []
    return [value]


def format_currency_eur(value: Any) -> str:
    """Format numeric-like values as Euro currency; fall back to raw string when not numeric."""
    try:
        # Allow strings like "1234.56" to be formatted as numbers
        num = float(value)
        if num.is_integer():
            return f"€{int(num):,}"
        return f"€{num:,.2f}"
    except Exception:
        return f"€{value}"


def format_analysis_summary(supplier_data: Dict[str, Any], supplier_name: str,
                            include_tqdcs_categories: Optional[Set[str]] = None,
                            include_cost: bool = True) -> str:
    """
    Format supplier analysis data for LLM consumption
    
    Args:
        supplier_data: Complete analysis data for the supplier
        supplier_name: Name of the supplier
        include_tqdcs_categories: If provided, only include these TQDCS categories
        include_cost: Whether to include cost context in the summary
        
    Returns:
        Formatted summary string
    """
    summary = f"\n=== {supplier_name} Analysis Summary ===\n"
    
    # TQDCS Scores
    tqdcs = coerce_to_dict(supplier_data.get('tqdcs'))
    overall_assessment = coerce_to_dict(tqdcs.get('overall_assessment'))
    if overall_assessment:
        avg_score = overall_assessment.get('average_score', 'N/A')
        summary += f"TQDCS Overall Score: {avg_score}/5\n"
        
        scores = coerce_to_dict(tqdcs.get('tqdcs_scores'))
        for category in ['technology', 'quality', 'delivery', 'cost', 'sustainability']:
            if include_tqdcs_categories is not None and category not in include_tqdcs_categories:
                continue
            if category in scores:
                cat_data = coerce_to_dict(scores.get(category, {})) or {}
                score = cat_data.get('score', 'N/A')
                summary += f"  - {category.capitalize()}: {score}/5\n"
                # Enrich with reasoning and key details
                reasoning = cat_data.get('reasoning')
                if reasoning:
                    reasoning_trim = str(reasoning).strip()
                    summary += f"    reasoning: {reasoning_trim}\n"
                strengths = coerce_to_list(cat_data.get('strengths'))
                if strengths:
                    strengths_trim = [str(s).strip() for s in strengths]
                    summary += f"    strengths: " + "; ".join(strengths_trim) + "\n"
                weaknesses = coerce_to_list(cat_data.get('weaknesses'))
                if weaknesses:
                    weaknesses_trim = [str(w).strip() for w in weaknesses]
                    summary += f"    weaknesses: " + "; ".join(weaknesses_trim) + "\n"
                key_findings = coerce_to_list(cat_data.get('key_findings'))
                if key_findings:
                    findings_trim = [str(k).strip() for k in key_findings]
                    summary += f"    key_findings: " + "; ".join(findings_trim) + "\n"
    
    # Risk Summary
    risks = coerce_to_dict(supplier_data.get('risks'))
    risk_summary = coerce_to_dict(risks.get('risk_summary'))
    if risk_summary:
        high_risks = risk_summary.get('high_risks_count', 0)
        medium_risks = risk_summary.get('medium_risks_count', 0)
        summary += f"Risk Profile: {high_risks} High, {medium_risks} Medium risks\n"
        
        # Top critical risks
        critical_risks = risk_summary.get('critical_risks', [])
        if isinstance(critical_risks, list) and critical_risks:
            summary += f"Critical Risks: {', '.join([str(r) for r in critical_risks[:3]])}\n"
        elif isinstance(critical_risks, str) and critical_risks.strip():
            summary += f"Critical Risks: {critical_risks.strip()}\n"
    
    # Cost Summary (conditionally include)
    if include_cost:
        cost = coerce_to_dict(supplier_data.get('cost'))
        if cost:
            total_cost = coerce_to_dict(cost.get('total_cost'))
            if total_cost:
                total = total_cost.get('amount', 'N/A')
                summary += f"Total Project Cost: {format_currency_eur(total)}\n"
            unit_costs = coerce_to_dict(cost.get('unit_costs'))
            if unit_costs:
                unit_price = unit_costs.get('price_per_unit', 'N/A')
                summary += f"Unit Price: {format_currency_eur(unit_price)}\n"
            # Use only volume_forecast numbers for demand/forecast figures
            forecast_list = coerce_to_list(cost.get('volume_forecast'))
            if forecast_list:
                try:
                    total_units = 0.0
                    details = []
                    for vf in forecast_list:
                        if isinstance(vf, dict):
                            tf = vf.get('timeframe', '—')
                            qty = vf.get('quantity')
                            try:
                                qty_val = float(qty)
                            except Exception:
                                qty_val = 0.0
                            total_units += qty_val
                            details.append(f"{tf}: {qty_val:,.0f}")
                    summary += f"Forecast (from cost.volume_forecast): total {total_units:,.0f} units; " + \
                               ("; ".join(details) if details else "") + "\n"
                except Exception:
                    pass
            cost_dependencies = coerce_to_list(cost.get("cost_dependencies"))
            if cost_dependencies:
                for dependency in cost_dependencies:
                    if isinstance(dependency, dict):
                        description = dependency.get('description', 'N/A')
                        impact = dependency.get('impact', 'N/A')
                        summary += f"Cost Dependency: {description} - {impact}\n"
                    else:
                        summary += f"Cost Dependency: {str(dependency).strip()}\n"
            cost_risks = coerce_to_list(cost.get("cost_risks"))
            if cost_risks:
                for cost_risk_item in cost_risks:
                    if isinstance(cost_risk_item, dict):
                        risk_title = cost_risk_item.get('risk', 'N/A')
                        impact = cost_risk_item.get('impact', 'N/A')
                        summary += f"Cost Risk: {risk_title} - {impact}\n"
                    else:
                        summary += f"Cost Risk: {str(cost_risk_item).strip()}\n"
                
    
    return summary


def generate_category_metrics(
    category: str,
    supplier1_name: str,
    supplier2_name: str,
    supplier1_analyses: Optional[Dict[str, Any]] = None,
    supplier2_analyses: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate comparison metrics for a specific TQDCS category using LLM and precomputed analyses.
    
    Args:
        category: TQDCS category ('Technology', 'Quality', 'Delivery', 'Cost', 'Sustainability')
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        supplier1_analyses: Optional analysis data for supplier 1 (tqdcs, risks, cost, parts)
        supplier2_analyses: Optional analysis data for supplier 2 (tqdcs, risks, cost, parts)
        
    Returns:
        List of comparison metrics for the category
    """
    print(f"  Generating {category} metrics...")
    
    # Build analysis context limited to the specific category
    analyses_context = ""
    allowed = {category.lower()}
    include_cost = (category.lower() == 'cost')
    if supplier1_analyses and supplier2_analyses:
        analyses_context = f"ANALYSIS DATA FOR {category.upper()}:\n"
        analyses_context += format_analysis_summary(supplier1_analyses, supplier1_name,
                                                    include_tqdcs_categories=allowed,
                                                    include_cost=include_cost)
        analyses_context += format_analysis_summary(supplier2_analyses, supplier2_name,
                                                    include_tqdcs_categories=allowed,
                                                    include_cost=include_cost)
        analyses_context += f"\nFocus on {category.lower()}-related aspects from this analysis data.\n"
    
    # Category-specific focus areas
    focus_areas = {
        "Technology": "Technical compliance, innovation, technology maturity, patents, R&D capabilities, system integration",
        "Quality": "Quality certifications, test results, quality systems, defect rates, compliance standards, audit results",
        "Delivery": "Lead times, delivery performance, capacity, flexibility, logistics, supply chain reliability", 
        "Cost": "Unit pricing, total cost of ownership, payment terms, cost volatility, hidden costs, value for money",
        "Sustainability": "Environmental certifications, sustainability commitments, carbon footprint, circular economy, social responsibility"
    }
    
    base_prompt = f"""Compare these two suppliers specifically on {category} aspects and generate exactly 2-5 focused comparison metrics.

Supplier 1: {supplier1_name}
Supplier 2: {supplier2_name}

{analyses_context}

FOCUS AREAS for {category}:
{focus_areas.get(category, "General comparison aspects")}

Return ONLY valid JSON (no markdown, no explanations) with this structure:
{{
  "metrics": [
    {{
      "metric": "Specific {category.lower()} metric name (e.g., 'Technical Compliance Rate', 'Quality Certification Status')",
      "category": "{category}",
      "supplier1_value": "Specific value or description for {supplier1_name}",
      "supplier2_value": "Specific value or description for {supplier2_name}",
      "winner": "{supplier1_name}|{supplier2_name}|Tie",
      "importance": "Critical|High|Medium|Low",
      "comparison_notes": "Brief explanation focusing on {category.lower()} implications",
      "sources": {{
        "supplier1": [{{\"filename\": \"analysis\", \"chunk_id\": \"{category}_context\"}}],
        "supplier2": [{{\"filename\": \"analysis\", \"chunk_id\": \"{category}_context\"}}]
      }}
    }}
  ]
}}

REQUIREMENTS:
1. Generate metrics focused on {category}
2. Base comparisons on the structured analyses provided (TQDCS, Risk, Cost, Parts)
3. Include specific source references (filename and chunk_id)
4. Make metrics highly relevant to {category} evaluation
5. Ensure objective winner determination
6. Focus on the most impactful {category.lower()} factors for this procurement

Return only the JSON object."""
    
    # Apply PurchasingOrganization business context to the prompt
    prompt = format_prompt_with_business_context(base_prompt, analysis_type="comparison")
    
    # Get LLM and analyze
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        
        # Extract content
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Handle different response types
        if isinstance(content, list):
            if content:
                content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
            else:
                content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Clean response
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
            metrics = result.get('metrics', [])
            
            # Add supplier names to each metric for UI display
            for metric in metrics:
                metric['supplier1_name'] = supplier1_name
                metric['supplier2_name'] = supplier2_name
            
            print(f"  Generated {len(metrics)} {category} metrics")
            return metrics
            
        except json.JSONDecodeError as e:
            print(f"  Failed to parse {category} metrics JSON: {e}")
            return []
            
    except Exception as e:
        print(f"  Error generating {category} metrics: {e}")
        return []


def compare_suppliers_structured(
    supplier1_name: str,
    supplier2_name: str,
    supplier1_analyses: Optional[Dict[str, Any]] = None,
    supplier2_analyses: Optional[Dict[str, Any]] = None,
    tqdcs_weights: Optional[Dict[str, float]] = None,
    generate_metrics: bool = True,
    generate_strengths_weaknesses: bool = True,
    generate_recommendation_and_split: bool = True,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two suppliers and return structured JSON data using only precomputed analyses.

    Args:
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        supplier1_analyses: Optional pre-computed analyses for supplier 1 (tqdcs, risks, cost, parts)
        supplier2_analyses: Optional pre-computed analyses for supplier 2 (tqdcs, risks, cost, parts)
        tqdcs_weights: Normalized weights mapping for TQDCS categories
        generate_metrics: Whether to generate comparison metrics
        generate_strengths_weaknesses: Whether to generate strengths/weaknesses
        generate_recommendation_and_split: Whether to generate recommendation and optimal split
    """
    print("Generating comparison from analyses only (no KG JSONs)...")

    # Determine allowed categories based solely on toggles (weights > 0 => enabled)
    base_cats = ['technology', 'quality', 'delivery', 'cost', 'sustainability']
    if tqdcs_weights:
        try:
            allowed_categories: Set[str] = {str(k).lower() for k, v in tqdcs_weights.items() if float(v) > 0.0}
            # Fallback to all if somehow none marked enabled
            if not allowed_categories:
                allowed_categories = set(base_cats)
        except Exception:
            allowed_categories = set(base_cats)
    else:
        allowed_categories = set(base_cats)
    include_cost_ctx = ('cost' in allowed_categories)

    # Generate comparison metrics from analyses (optional)
    categories = ["Technology", "Quality", "Delivery", "Cost", "Sustainability"]
    categories = [c for c in categories if c.lower() in allowed_categories]
    all_metrics: List[Dict[str, Any]] = []
    if generate_metrics:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_category = {
                executor.submit(
                    generate_category_metrics,
                    category,
                    supplier1_name,
                    supplier2_name,
                    supplier1_analyses,
                    supplier2_analyses,
                    model_name
                ): category
                for category in categories
            }
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    category_metrics = future.result()
                    all_metrics.extend(category_metrics)
                    print(f"  ✓ {category} metrics completed ({len(category_metrics)} metrics)")
                except Exception as e:
                    print(f"  ✗ Error generating {category} metrics: {e}")
        print(f"Generated {len(all_metrics)} comparison metrics across TQDCS categories")

    # Build analyses context text (filtered)
    analyses_context = ""
    if supplier1_analyses and supplier2_analyses:
        analyses_context = "COMPREHENSIVE ANALYSIS DATA:\n"
        analyses_context += format_analysis_summary(supplier1_analyses, supplier1_name,
                                                    include_tqdcs_categories=allowed_categories,
                                                    include_cost=include_cost_ctx)
        analyses_context += format_analysis_summary(supplier2_analyses, supplier2_name,
                                                    include_tqdcs_categories=allowed_categories,
                                                    include_cost=include_cost_ctx)
        analyses_context += "\nUSE THIS ANALYSIS DATA to inform your outputs.\n"

    # Create a summary of the generated metrics to inform the LLM
    metrics_summary = ""
    if generate_metrics and all_metrics:
        metrics_summary = "COMPARISON METRICS SUMMARY:\n"
        categories_count: Dict[str, int] = {}
        for metric in all_metrics:
            category = metric.get('category', 'Unknown')
            winner = metric.get('winner', 'Unknown')
            categories_count[category] = categories_count.get(category, 0) + 1
            metrics_summary += f"- {category}: {metric.get('metric')} → Winner: {winner}\n"
        metrics_summary += f"\nMetrics by category: {dict(categories_count)}\n"

    # Build TQDCS guidance based on enabled categories only (equal weighting semantics)
    weights_guidance = ""
    if allowed_categories:
        cats_list = ", ".join([c.capitalize() for c in sorted(allowed_categories)])
        weights_guidance = (
            "TQDCS CATEGORY INCLUSION:\n"
            f"- Enabled categories: {cats_list}\n"
            "- Treat all enabled categories as equally important; do not apply numeric weightings.\n"
            "- Ignore categories that are not listed as enabled.\n\n"
        )

    # Decide which sections to request from LLM
    sections_required: List[str] = []
    if generate_strengths_weaknesses:
        sections_required.append("strengths_weaknesses")
    if generate_recommendation_and_split:
        sections_required.extend(["recommendation", "optimal_split"])

    base_prompt = f"""Based on the structured analyses and optional comparison metrics, generate the requested sections.

Supplier 1: {supplier1_name}
Supplier 2: {supplier2_name}

{weights_guidance}{analyses_context}

{metrics_summary}

Generate ONLY the following sections: {', '.join(sections_required)}.
Return ONLY valid JSON (no markdown, no explanations outside JSON) with this exact structure (include only requested top-level keys):
{{
  "strengths_weaknesses": {{
    "supplier1": {{
      "name": "{supplier1_name}",
      "strengths": [
        {{
          "title": "Strength title",
          "description": "Detailed description",
          "impact": "Impact on project success",
          "sources": [{{"filename": "...", "chunk_id": "..."}}]
        }}
      ],
      "weaknesses": [
        {{
          "title": "Weakness title",
          "description": "Detailed description",
          "risk": "Associated risk",
          "sources": [{{"filename": "...", "chunk_id": "..."}}]
        }}
      ]
    }},
    "supplier2": {{
      "name": "{supplier2_name}",
      "strengths": [...],
      "weaknesses": [...]
    }}
  }},
  "recommendation": {{
    "preferred_supplier": "{supplier1_name}|{supplier2_name}",
    "confidence_level": "High|Medium|Low",
    "key_reasons": ["..."],
    "conditions": ["..."],
    "risk_considerations": ["..."]
  }},
  "optimal_split": {{
    "supplier1_name": "{supplier1_name}",
    "supplier2_name": "{supplier2_name}",
    "supplier1_percentage": <number 0-100>,
    "supplier2_percentage": <number 0-100>,
    "rationale": "Explanation considering the TQDCS analysis provided",
    "key_factors": ["..."],
    "implementation_phases": [{{"phase": "...", "description": "...", "timeline": "..."}}],
    "risk_mitigation": "...",
    "cost_benefits": "...",
    "flexibility_considerations": "..."
  }}
IMPORTANT GUIDELINES:
1. Include specific source references (filename and chunk_id)
2. For strengths and weaknesses, focus on the most impactful items in the analysis
3. Make a clear recommendation based on the analysis
4. For the optimal split:
   - ALWAYS use dual-sourcing strategy (never single-sourcing)
   - Consider minimum 20% for the weaker supplier
   - Consider the TQDCS analysis provided as well as risk levels, cost structure and strategic considerations
}}"""

    # Apply PurchasingOrganization business context to the prompt
    prompt = format_prompt_with_business_context(base_prompt, analysis_type="comparison")

    print(f"Prompt size: ~{len(prompt):,} characters")
    selected_model = model_name or "gemini-2.5-pro"
    print(f"Sending to {selected_model} for comparison analysis...")

    # Get LLM and analyze
    llm = create_llm(model_name=selected_model, temperature=0.1)

    try:
        response = llm.invoke(prompt)

        # Extract the text content
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        # Handle different response types
        if isinstance(content, list):
            if content:
                content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
            else:
                content = ""
        elif not isinstance(content, str):
            content = str(content)

        # Clean the response
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
            llm_result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {content[:500]}...")
            return {
                "error": "Failed to parse LLM response",
                "raw_response": content,
                "comparison_metrics": all_metrics if generate_metrics else [],
                "strengths_weaknesses": {},
                "recommendation": {},
                "optimal_split": {}
            }

        # Combine the generated metrics with the LLM result
        if generate_metrics:
            llm_result["comparison_metrics"] = all_metrics

        # Add supplier names to result
        llm_result["supplier1_name"] = supplier1_name
        llm_result["supplier2_name"] = supplier2_name
        # Persist TQDCS weights used for this computation (for UI restore)
        if tqdcs_weights is not None:
            try:
                # Normalize to floats with 1 decimal for compactness
                llm_result["tqdcs_weights"] = {
                    str(k): round(float(v), 1) for k, v in tqdcs_weights.items()
                }
            except Exception:
                llm_result["tqdcs_weights"] = tqdcs_weights

        # Ensure optimal_split exists when requested
        if generate_recommendation_and_split and "optimal_split" not in llm_result:
            llm_result["optimal_split"] = {
                "supplier1_percentage": 60,
                "supplier2_percentage": 40,
                "rationale": "Default split recommendation - LLM analysis incomplete",
                "key_factors": ["Analysis incomplete"],
                "risk_mitigation": "Standard dual-sourcing approach",
                "cost_benefits": "Maintains competitive pressure"
            }

        # Add comparison summary
        metrics = llm_result.get('comparison_metrics', []) if generate_metrics else []
        supplier1_wins = sum(1 for m in metrics if m.get('winner') == supplier1_name)
        supplier2_wins = sum(1 for m in metrics if m.get('winner') == supplier2_name)
        ties = sum(1 for m in metrics if m.get('winner') == 'Tie')

        llm_result["comparison_summary"] = {
            "total_metrics": len(metrics),
            f"{supplier1_name}_advantages": supplier1_wins,
            f"{supplier2_name}_advantages": supplier2_wins,
            "tied_metrics": ties,
            "clear_winner": supplier1_name if supplier1_wins > supplier2_wins else (
                supplier2_name if supplier2_wins > supplier1_wins else "No clear winner"
            )
        }

        print("Comparison analysis complete!")
        if generate_metrics:
            print(f"  Metrics compared: {len(metrics)}")
            print(f"  {supplier1_name} advantages: {supplier1_wins}")
            print(f"  {supplier2_name} advantages: {supplier2_wins}")

        # Save to file if requested
        if save_to_file and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(llm_result, f, indent=2, ensure_ascii=False)
            print(f"Structured comparison saved to: {output_path}")

        return llm_result

    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        raise


def main():
    """Test the structured comparator"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    
    # kg_path1 = str(project_root / "docs/business/supplier 1/supplier_1_kg_image/unified_kg_20250723_195008.json")
    # kg_path2 = str(project_root / "docs/business/supplier 2/supplier_2_kg_image/unified_kg_20250723_200647.json")
    
    try:
        # Test without pre-computed analyses (standalone comparison)
        result = compare_suppliers_structured(
            supplier1_name="SupplierA",
            supplier2_name="SupplierB",
            save_to_file=True,
            output_path="output/structured_comparison.json"
        )
        
        print("\n=== COMPARISON RESULTS ===")
        print(f"Preferred Supplier: {result.get('recommendation', {}).get('preferred_supplier', 'N/A')}")
        print(f"Optimal Split: {result.get('optimal_split', {}).get('supplier1_name')} {result.get('optimal_split', {}).get('supplier1_percentage')}% / {result.get('optimal_split', {}).get('supplier2_name')} {result.get('optimal_split', {}).get('supplier2_percentage')}%")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()