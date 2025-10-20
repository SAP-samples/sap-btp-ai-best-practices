"""
Structured TQDCS Analyzer - Returns JSON with TQDCS Scores

This module analyzes TQDCS (Technology, Quality, Delivery, Cost, Sustainability) aspects
from knowledge graphs and returns structured scores with reasoning.
Supports both serial and parallel analysis modes.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context
from .delivery_kpi_mock_data import get_delivery_kpis_for_supplier


def filter_tqdcs_category_data(kg_data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Filter KG to only include nodes for a specific TQDCS category and their connections.
    
    Args:
        kg_data: Full knowledge graph dictionary
        category: TQDCS category letter ('T', 'Q', 'D', 'C', or 'S')
        
    Returns:
        Filtered KG with same structure but only category-related data
    """
    nodes = kg_data.get('nodes', [])
    relationships = kg_data.get('relationships', [])
    
    # Step 1: Find all nodes with the specified TQDCS category
    category_node_ids = set()
    for node in nodes:
        tqdcs_categories = node.get('properties', {}).get('tqdcs_categories', [])
        if category in tqdcs_categories:
            category_node_ids.add(node['id'])
    
    if not category_node_ids:
        print(f"Warning: No nodes found for TQDCS category '{category}'")
        return kg_data  # Return original if no nodes found
    
    # Step 2: Find all relationships involving category nodes (1-hop)
    category_relationships = []
    connected_node_ids = set()
    
    for rel in relationships:
        source = rel.get('source')
        target = rel.get('target')
        
        # If either end connects to a category node, include it
        if source in category_node_ids or target in category_node_ids:
            category_relationships.append(rel)
            # Track all connected nodes
            connected_node_ids.add(source)
            connected_node_ids.add(target)
    
    # Step 3: Get all relevant nodes (category nodes + 1-hop connected nodes)
    relevant_nodes = [
        n for n in nodes 
        if n['id'] in connected_node_ids
    ]
    
    # Step 4: Return filtered KG with same structure
    return {
        'nodes': relevant_nodes,
        'relationships': category_relationships,
        'metadata': kg_data.get('metadata', {}),
        'export_metadata': kg_data.get('export_metadata', {})
    }


def compute_overall_assessment(
    tqdcs_scores: Dict[str, Dict[str, Any]], 
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compute overall assessment from TQDCS scores
    
    Args:
        tqdcs_scores: Dictionary of TQDCS scores
        weights: Optional weights for each category (must sum to 1.0)
                 Default is equal weighting
        
    Returns:
        Overall assessment with average score and analysis
    """
    # Default equal weights
    if weights is None:
        weights = {
            'technology': 0.2,
            'quality': 0.2,
            'delivery': 0.2,
            'cost': 0.2,
            'sustainability': 0.2
        }
    
    # Calculate weighted average
    weighted_sum = 0
    total_weight = 0
    scores_list = []
    
    for category, weight in weights.items():
        if category in tqdcs_scores and 'score' in tqdcs_scores[category]:
            score = tqdcs_scores[category]['score']
            weighted_sum += score * weight
            total_weight += weight
            scores_list.append((category, score))
    
    # Calculate average (weighted or simple)
    if total_weight > 0:
        average_score = round(weighted_sum / total_weight, 2)
    else:
        average_score = 0
    
    # Find strongest and weakest areas
    if scores_list:
        scores_list.sort(key=lambda x: x[1], reverse=True)
        strongest_area = scores_list[0][0]
        weakest_area = scores_list[-1][0]
        
        # Generate summary based on average score
        if average_score >= 4.5:
            summary = "Excellent supplier with outstanding performance across all TQDCS dimensions"
            recommendation = "Highly recommended - minimal risks identified"
        elif average_score >= 4.0:
            summary = "Strong supplier with good performance across most TQDCS dimensions"
            recommendation = "Recommended - minor areas for improvement"
        elif average_score >= 3.5:
            summary = "Competent supplier with adequate performance, some areas need attention"
            recommendation = "Acceptable with risk mitigation measures in place"
        elif average_score >= 3.0:
            summary = "Moderate supplier with mixed performance across TQDCS dimensions"
            recommendation = "Conditional recommendation - significant improvements needed"
        elif average_score >= 2.5:
            summary = "Below-average supplier with significant gaps in multiple areas"
            recommendation = "Not recommended without major improvements"
        else:
            summary = "Poor supplier performance with critical issues across multiple dimensions"
            recommendation = "Not recommended - high risk profile"
    else:
        strongest_area = "Unknown"
        weakest_area = "Unknown"
        summary = "Unable to assess - no scores available"
        recommendation = "Insufficient data for recommendation"
    
    return {
        "average_score": average_score,
        "weighted_average": total_weight < 1.0,  # Indicates if weighted calculation was used
        "strongest_area": strongest_area,
        "weakest_area": weakest_area,
        "summary": summary,
        "recommendation": recommendation,
        "score_distribution": {
            "excellent": len([s for _, s in scores_list if s >= 4.5]),
            "good": len([s for _, s in scores_list if 4.0 <= s < 4.5]),
            "adequate": len([s for _, s in scores_list if 3.0 <= s < 4.0]),
            "poor": len([s for _, s in scores_list if s < 3.0])
        }
    }


def analyze_technology(
    kg_data: Dict[str, Any], 
    supplier_name: str,
    parts_analysis: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze Technology dimension of TQDCS.
    
    Args:
        kg_data: Filtered KG with Technology nodes
        supplier_name: Name of the supplier
        parts_analysis: Optional parts analysis results for context
        
    Returns:
        Technology score with reasoning
    """
    # Build context from parts analysis if available
    parts_context = ""
    if parts_analysis:
        parts_summary = parts_analysis.get('parts_summary', {})
        supplier_capabilities = parts_analysis.get('supplier_capabilities', {})
        parts_context = f"""
PARTS ANALYSIS CONTEXT:
- Total parts: {parts_summary.get('total_parts_count', 0)}
- Manufacturing processes: {', '.join(supplier_capabilities.get('manufacturing_processes', [])[:5])}
- Testing capabilities: {', '.join(supplier_capabilities.get('testing_capabilities', [])[:5])}
- Special competencies: {', '.join(supplier_capabilities.get('special_competencies', [])[:3])}
"""
    
    prompt = f"""Analyze the TECHNOLOGY dimension for {supplier_name}.

{parts_context}

Knowledge Graph (Technology-focused data):
{json.dumps(kg_data, indent=2)}

Evaluate based on:
- Technical specifications and compliance
- R&D and innovation capabilities
- Patents and intellectual property
- Manufacturing technology and processes
- Testing and validation capabilities
- CAD/CAM systems and digital capabilities

STRICT FORMAT RULES (DO NOT VIOLATE):
- key_findings MUST be a list of plain strings only. Do NOT include objects in key_findings.
- All sources MUST be provided ONLY in the top-level "sources" array. Do NOT include sources inline inside key_findings or any other fields except the top-level "sources".
- Do NOT emit any extra keys other than those specified in the JSON schema below.

Return ONLY this JSON structure:
{{
    "score": <number 1-5>,
    "reasoning": "Detailed explanation of the technology score",
    "strengths": ["Key strength 1", "Key strength 2", "Key strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
}}

SCORING:
5: Cutting-edge technology, full compliance, strong innovation
4: Modern technology, mostly compliant, good capabilities
3: Adequate technology, some gaps but manageable
2: Outdated or significantly non-compliant technology
1: Major technology gaps, critical non-compliances
"""
    
    # Apply PurchasingOrganization context
    prompt = format_prompt_with_business_context(prompt, analysis_type="technology")
    
    # Get LLM response
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean and parse response
        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        data = json.loads(content.strip())
        # Normalize schema defensively: ensure key_findings are strings; move any inline sources up
        try:
            if isinstance(data, dict):
                findings = data.get('key_findings')
                if isinstance(findings, list):
                    normalized = []
                    moved_sources = []
                    for item in findings:
                        if isinstance(item, dict):
                            # Convert dict finding to text and migrate sources
                            text = item.get('finding') if 'finding' in item else json.dumps(item, ensure_ascii=False)
                            if text is not None:
                                normalized.append(str(text))
                            for s in (item.get('sources') or []):
                                if isinstance(s, dict):
                                    moved_sources.append({
                                        'filename': s.get('filename', ''),
                                        'chunk_id': s.get('chunk_id', '')
                                    })
                        elif isinstance(item, str):
                            normalized.append(item)
                        else:
                            normalized.append(str(item))
                    data['key_findings'] = normalized
                    if moved_sources:
                        existing = data.get('sources')
                        if not isinstance(existing, list):
                            existing = []
                        existing.extend(moved_sources)
                        data['sources'] = existing
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Error in technology analysis: {e}")
        return {
            "score": 3,
            "reasoning": "Error in analysis",
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
            "sources": []
        }


def analyze_quality(
    kg_data: Dict[str, Any],
    supplier_name: str,
    parts_analysis: Optional[Dict[str, Any]] = None,
    risk_analysis: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze Quality dimension of TQDCS.
    
    Args:
        kg_data: Filtered KG with Quality nodes
        supplier_name: Name of the supplier
        parts_analysis: Optional parts analysis for quality specs
        risk_analysis: Optional risk analysis for quality risks
        
    Returns:
        Quality score with reasoning
    """
    # Build context from prior analyses
    context = ""
    
    if parts_analysis:
        parts = parts_analysis.get('parts', [])
        quality_reqs = []
        for part in parts[:3]:  # Sample first 3 parts
            quality_reqs.extend(part.get('quality_requirements', []))
        context += f"""
PARTS QUALITY CONTEXT:
- Quality requirements identified: {len(quality_reqs)}
- Key certifications from parts: {', '.join(parts_analysis.get('supplier_capabilities', {}).get('certifications', [])[:5])}
"""
    
    if risk_analysis:
        quality_risks = [r for r in risk_analysis.get('risks', []) 
                        if 'quality' in r.get('category', '').lower()]
        context += f"""
QUALITY RISKS IDENTIFIED:
- Quality-related risks: {len(quality_risks)}
- High severity quality risks: {sum(1 for r in quality_risks if r.get('severity', '') == 'High')}
"""
    
    prompt = f"""Analyze the QUALITY dimension for {supplier_name}.

{context}

Knowledge Graph (Quality-focused data):
{json.dumps(kg_data, indent=2)}

Evaluate based on:
- Quality management systems and certifications (ISO 9001, IATF 16949, etc.)
- Test results and validation data
- Defect rates and quality metrics
- Quality control processes
- Compliance with quality standards
- Customer quality requirements

STRICT FORMAT RULES (DO NOT VIOLATE):
- key_findings MUST be a list of plain strings only. Do NOT include objects in key_findings.
- All sources MUST be provided ONLY in the top-level "sources" array. Do NOT include sources inline inside key_findings or any other fields except the top-level "sources".
- Do NOT emit any extra keys other than those specified in the JSON schema below.

Return ONLY this JSON structure:
{{
    "score": <number 1-5>,
    "reasoning": "Detailed explanation of the quality score",
    "strengths": ["Key strength 1", "Key strength 2", "Key strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
}}

SCORING:
5: Exceptional quality systems, all certifications current
4: Strong quality systems, minor gaps
3: Adequate quality, some concerns but manageable
2: Significant quality issues or missing certifications
1: Major quality concerns, critical gaps
"""
    
    prompt = format_prompt_with_business_context(prompt, analysis_type="quality")
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        data = json.loads(content.strip())
        # Normalize schema defensively: ensure key_findings are strings; move any inline sources up
        try:
            if isinstance(data, dict):
                findings = data.get('key_findings')
                if isinstance(findings, list):
                    normalized = []
                    moved_sources = []
                    for item in findings:
                        if isinstance(item, dict):
                            # Convert dict finding to text and migrate sources
                            text = item.get('finding') if 'finding' in item else json.dumps(item, ensure_ascii=False)
                            if text is not None:
                                normalized.append(str(text))
                            for s in (item.get('sources') or []):
                                if isinstance(s, dict):
                                    moved_sources.append({
                                        'filename': s.get('filename', ''),
                                        'chunk_id': s.get('chunk_id', '')
                                    })
                        elif isinstance(item, str):
                            normalized.append(item)
                        else:
                            normalized.append(str(item))
                    data['key_findings'] = normalized
                    if moved_sources:
                        existing = data.get('sources')
                        if not isinstance(existing, list):
                            existing = []
                        existing.extend(moved_sources)
                        data['sources'] = existing
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Error in quality analysis: {e}")
        return {
            "score": 3,
            "reasoning": "Error in analysis",
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
            "sources": []
        }


def analyze_delivery(
    kg_data: Dict[str, Any],
    supplier_name: str,
    risk_analysis: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    cost_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze Delivery dimension of TQDCS.
    
    Args:
        kg_data: Filtered KG with Delivery nodes
        supplier_name: Name of the supplier
        risk_analysis: Optional risk analysis for supply chain risks
        
    Returns:
        Delivery score with reasoning
    """
    # Build context from risk and cost analyses
    context = ""
    if risk_analysis:
        delivery_risks = [r for r in risk_analysis.get('risks', [])
                         if 'delivery' in r.get('category', '').lower() or 
                         'supply' in r.get('category', '').lower()]
        context = f"""
DELIVERY RISKS CONTEXT:
- Supply chain risks identified: {len(delivery_risks)}
- High severity delivery risks: {sum(1 for r in delivery_risks if r.get('severity', '') == 'High')}
"""
    
    # Include forecast volumes from cost analysis to assess deliverable quantities
    if cost_analysis:
        vols = cost_analysis.get('volume_forecast', []) or []
        total_units = 0.0
        brief = []
        for vf in vols:  
            timeframe = vf.get('timeframe', '—')
            qty = vf.get('quantity') or 0
            try:
                qty_val = float(qty)
            except Exception:
                qty_val = 0.0
            total_units += qty_val
            brief.append(f"{timeframe}: {qty_val:,.0f}")
        if vols:
            context += f"""
VOLUME FORECAST CONTEXT:
- Total forecast quantity: {total_units:,.0f} units
- Timeframes: {', '.join(brief)}
"""

    # Include authoritative external delivery KPIs (mocked integration)
    external_kpis = {}
    try:
        if supplier_name:
            external_kpis = get_delivery_kpis_for_supplier(supplier_name) or {}
    except Exception:
        external_kpis = {}

    if external_kpis:
        context += f"""
EXTERNAL DELIVERY KPI CONTEXT (authoritative):
- Country: {external_kpis.get('country', '—')}
- Lead time avg (days): {external_kpis.get('lead_time_avg_days', '—')}
- Lead time variability (days): {external_kpis.get('lead_time_variability_days', '—')}
- OTIF avg (%): {external_kpis.get('otif_avg_pct', '—')}
- OTIF variability (%-points): {external_kpis.get('otif_variability_pct', '—')}
- Country risk score (1-5): {external_kpis.get('country_risk_score_1_to_5', '—')}
"""
    else:
        context += """
EXTERNAL DELIVERY KPI CONTEXT:
- No external KPI data available for this supplier.
"""

    # Sanitize KG to remove any nodes/relationships that look like demand/volume/forecast values
    # We want demand/forecast numbers to come ONLY from the VOLUME FORECAST CONTEXT above
    try:
        nodes = kg_data.get('nodes', [])
        relationships = kg_data.get('relationships', [])
        remove_ids = {n.get('id') for n in nodes if isinstance(n.get('id'), str) and (
            'forecast' in n.get('id', '').lower() or 'volume' in n.get('id', '').lower()
        )}
        sanitized_nodes = [n for n in nodes if n.get('id') not in remove_ids]
        kept_ids = {n.get('id') for n in sanitized_nodes}
        sanitized_relationships = [r for r in relationships if r.get('source') in kept_ids and r.get('target') in kept_ids]
        kg_data_for_prompt = {
            'nodes': sanitized_nodes,
            'relationships': sanitized_relationships,
            'metadata': kg_data.get('metadata', {}),
            'export_metadata': kg_data.get('export_metadata', {})
        }
    except Exception:
        kg_data_for_prompt = kg_data
    
    prompt = f"""Analyze the DELIVERY dimension for {supplier_name}.

{context}

Knowledge Graph (Delivery-focused data):
{json.dumps(kg_data_for_prompt, indent=2)}

Evaluate based on:
- Production capacity and utilization
- Lead times and delivery performance
- Logistics capabilities (JIT, JIS, EDI)
- Supply chain flexibility and resilience
- Contingency planning and risk mitigation
- Geographic location and transportation
- Ability to meet forecast volume commitments and ramp-up needs

CRITICAL VOLUME RULE:
- For demand/forecast quantities, use ONLY the values provided in the VOLUME FORECAST CONTEXT above.
- Ignore any other demand/volume values that may appear in the knowledge graph or documents.
- You may use KG information for CAPACITY/capability checks, but do not replace the forecast values with other numbers.

CRITICAL DELIVERY KPI RULE:
- For lead time and OTIF, use ONLY the values provided in the EXTERNAL DELIVERY KPI CONTEXT above.
- Treat any lead time/OTIF mentions in the KG as supporting context; they must not override the external KPI values.
- Always report mean and variability for both lead time and OTIF when available.
- Consider the country risk score (1-5) as part of the delivery assessment and explicitly reference it in reasoning.

REQUIREMENTS FOR FORECAST/PEAK DEMAND REPORTING:
- If you aggregate forecast quantities from multiple entries, timeframes, plants, or variants, explicitly state this aggregation in the reasoning or key_findings.
- List each component quantity with its timeframe, show the summed total, and identify the peak demand considered (include timeframe).
- Include sources for every aggregated component and for the final aggregated figure.

STRICT FORMAT RULES (DO NOT VIOLATE):
- key_findings MUST be a list of plain strings only. Do NOT include objects in key_findings.
- All sources MUST be provided ONLY in the top-level "sources" array. Do NOT include sources inline inside key_findings or any other fields except the top-level "sources".
- Do NOT emit any extra keys other than those specified in the JSON schema below.

Return ONLY this JSON structure:
{{
    "score": <number 1-5>,
    "reasoning": "Detailed explanation of the delivery score",
    "strengths": ["Key strength 1", "Key strength 2", "Key strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
}}

SCORING:
5: Excellent delivery capability, short lead times, high flexibility
4: Good delivery performance, reasonable lead times
3: Adequate delivery, some constraints
2: Long lead times, capacity concerns
1: Critical delivery issues, major bottlenecks
"""
    
    prompt = format_prompt_with_business_context(prompt, analysis_type="delivery")
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        data = json.loads(content.strip())
        # Normalize schema defensively: ensure key_findings are strings; move any inline sources up
        try:
            if isinstance(data, dict):
                findings = data.get('key_findings')
                if isinstance(findings, list):
                    normalized = []
                    moved_sources = []
                    for item in findings:
                        if isinstance(item, dict):
                            # Convert dict finding to text and migrate sources
                            text = item.get('finding') if 'finding' in item else json.dumps(item, ensure_ascii=False)
                            if text is not None:
                                normalized.append(str(text))
                            for s in (item.get('sources') or []):
                                if isinstance(s, dict):
                                    moved_sources.append({
                                        'filename': s.get('filename', ''),
                                        'chunk_id': s.get('chunk_id', '')
                                    })
                        elif isinstance(item, str):
                            normalized.append(item)
                        else:
                            normalized.append(str(item))
                    data['key_findings'] = normalized
                    if moved_sources:
                        existing = data.get('sources')
                        if not isinstance(existing, list):
                            existing = []
                        existing.extend(moved_sources)
                        data['sources'] = existing
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Error in delivery analysis: {e}")
        return {
            "score": 3,
            "reasoning": "Error in analysis",
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
            "sources": []
        }


def analyze_cost(
    kg_data: Dict[str, Any],
    supplier_name: str,
    cost_analysis: Optional[Dict[str, Any]] = None,
    risk_analysis: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze Cost dimension of TQDCS.
    
    Args:
        kg_data: Filtered KG with Cost nodes
        supplier_name: Name of the supplier
        cost_analysis: Optional detailed cost analysis results
        risk_analysis: Optional risk analysis for financial risks
        
    Returns:
        Cost score with reasoning
    """
    # Build context from prior analyses
    context = ""
    
    if cost_analysis:
        total_cost = cost_analysis.get('total_cost', {})
        unit_costs = cost_analysis.get('unit_costs', {})
        est_total_product = cost_analysis.get('estimated_total_product_cost', {}) or {}
        vols = cost_analysis.get('volume_forecast', []) or []
        total_units = 0.0
        for vf in vols:
            try:
                total_units += float(vf.get('quantity') or 0)
            except Exception:
                pass
        context += f"""
COST ANALYSIS RESULTS:
- Total project cost: {total_cost.get('amount', 0)} {total_cost.get('currency', 'EUR')}
- Price per unit: {unit_costs.get('price_per_unit', 0)} {unit_costs.get('currency', 'EUR')}
- Estimated total product cost (unit × forecast): {est_total_product.get('amount', 0)} {est_total_product.get('currency', 'EUR')}
- Total forecast quantity considered: {total_units:,.0f} units
- Cost competitiveness: {cost_analysis.get('summary', {}).get('cost_competitiveness', 'Unknown')}
- Key insights: {', '.join(cost_analysis.get('summary', {}).get('key_insights', [])[:3])}
"""
    
    if risk_analysis:
        cost_risks = [r for r in risk_analysis.get('risks', [])
                     if 'cost' in r.get('category', '').lower() or 
                     'financial' in r.get('category', '').lower()]
        context += f"""
COST RISKS:
- Financial risks identified: {len(cost_risks)}
- High severity cost risks: {sum(1 for r in cost_risks if r.get('severity', '') == 'High')}
"""
    
    prompt = f"""Analyze the COST dimension for {supplier_name}.

{context}

Knowledge Graph (Cost-focused data):
{json.dumps(kg_data, indent=2)}

Evaluate based on:
- Price competitiveness vs market
- Total cost of ownership
- Payment terms and conditions
- Cost transparency and breakdown
- Volume pricing and discounts
- Financial stability
 - Estimated total product cost considering forecast volumes

STRICT FORMAT RULES (DO NOT VIOLATE):
- key_findings MUST be a list of plain strings only. Do NOT include objects in key_findings.
- All sources MUST be provided ONLY in the top-level "sources" array. Do NOT include sources inline inside key_findings or any other fields except the top-level "sources".
- Do NOT emit any extra keys other than those specified in the JSON schema below.

Return ONLY this JSON structure:
{{
    "score": <number 1-5>,
    "reasoning": "Detailed explanation of the cost score",
    "strengths": ["Key strength 1", "Key strength 2", "Key strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
}}

SCORING:
5: Highly competitive pricing, favorable terms
4: Competitive pricing, reasonable terms
3: Average pricing, standard terms
2: Above market pricing, unfavorable terms
1: Excessive pricing, very unfavorable terms
"""
    
    prompt = format_prompt_with_business_context(prompt, analysis_type="cost")
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        data = json.loads(content.strip())
        # Normalize schema defensively: ensure key_findings are strings; move any inline sources up
        try:
            if isinstance(data, dict):
                findings = data.get('key_findings')
                if isinstance(findings, list):
                    normalized = []
                    moved_sources = []
                    for item in findings:
                        if isinstance(item, dict):
                            # Convert dict finding to text and migrate sources
                            text = item.get('finding') if 'finding' in item else json.dumps(item, ensure_ascii=False)
                            if text is not None:
                                normalized.append(str(text))
                            for s in (item.get('sources') or []):
                                if isinstance(s, dict):
                                    moved_sources.append({
                                        'filename': s.get('filename', ''),
                                        'chunk_id': s.get('chunk_id', '')
                                    })
                        elif isinstance(item, str):
                            normalized.append(item)
                        else:
                            normalized.append(str(item))
                    data['key_findings'] = normalized
                    if moved_sources:
                        existing = data.get('sources')
                        if not isinstance(existing, list):
                            existing = []
                        existing.extend(moved_sources)
                        data['sources'] = existing
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Error in cost analysis: {e}")
        return {
            "score": 3,
            "reasoning": "Error in analysis",
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
            "sources": []
        }


def analyze_sustainability(
    kg_data: Dict[str, Any],
    supplier_name: str,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze Sustainability dimension of TQDCS.
    
    Args:
        kg_data: Filtered KG with Sustainability nodes
        supplier_name: Name of the supplier
        
    Returns:
        Sustainability score with reasoning
    """
    prompt = f"""Analyze the SUSTAINABILITY dimension for {supplier_name}.

Knowledge Graph (Sustainability-focused data):
{json.dumps(kg_data, indent=2)}

Evaluate based on:
- Environmental certifications (ISO 14001, etc.)
- Carbon footprint and emissions reduction
- Sustainable materials and processes
- Waste management and recycling
- Energy efficiency initiatives
- Social responsibility and ethics

STRICT FORMAT RULES (DO NOT VIOLATE):
- key_findings MUST be a list of plain strings only. Do NOT include objects in key_findings.
- All sources MUST be provided ONLY in the top-level "sources" array. Do NOT include sources inline inside key_findings or any other fields except the top-level "sources".
- Do NOT emit any extra keys other than those specified in the JSON schema below.

Return ONLY this JSON structure:
{{
    "score": <number 1-5>,
    "reasoning": "Detailed explanation of the sustainability score",
    "strengths": ["Key strength 1", "Key strength 2", "Key strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
}}

SCORING:
5: Industry leader in sustainability, clear commitments
4: Strong sustainability practices, good documentation
3: Basic sustainability compliance
2: Limited sustainability efforts
1: No clear sustainability commitment
"""
    
    prompt = format_prompt_with_business_context(prompt, analysis_type="sustainability")
    selected_model = model_name or "gemini-2.5-pro"
    llm = create_llm(model_name=selected_model, temperature=0.0)
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        data = json.loads(content.strip())
        # Normalize schema defensively: ensure key_findings are strings; move any inline sources up
        try:
            if isinstance(data, dict):
                findings = data.get('key_findings')
                if isinstance(findings, list):
                    normalized = []
                    moved_sources = []
                    for item in findings:
                        if isinstance(item, dict):
                            # Convert dict finding to text and migrate sources
                            text = item.get('finding') if 'finding' in item else json.dumps(item, ensure_ascii=False)
                            if text is not None:
                                normalized.append(str(text))
                            for s in (item.get('sources') or []):
                                if isinstance(s, dict):
                                    moved_sources.append({
                                        'filename': s.get('filename', ''),
                                        'chunk_id': s.get('chunk_id', '')
                                    })
                        elif isinstance(item, str):
                            normalized.append(item)
                        else:
                            normalized.append(str(item))
                    data['key_findings'] = normalized
                    if moved_sources:
                        existing = data.get('sources')
                        if not isinstance(existing, list):
                            existing = []
                        existing.extend(moved_sources)
                        data['sources'] = existing
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"Error in sustainability analysis: {e}")
        return {
            "score": 3,
            "reasoning": "Error in analysis",
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
            "sources": []
        }


def analyze_tqdcs_structured_parallel(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    prior_analyses: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parallel TQDCS analysis using category-specific filtering and prior analyses.
    
    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for the analysis
        prior_analyses: Optional dict with 'parts', 'cost', 'risk' analysis results
        weights: Optional weights for TQDCS categories (must sum to 1.0)
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        
    Returns:
        Structured dictionary with TQDCS analysis
    """
    print(f"Loading KG from: {kg_json_path}")
    
    # Load the full KG JSON once
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    original_nodes = len(kg_data.get('nodes', []))
    print(f"Loaded KG with {original_nodes} nodes")
    
    # Extract prior analyses if provided
    prior_analyses = prior_analyses or {}
    parts_analysis = prior_analyses.get('parts')
    cost_analysis = prior_analyses.get('cost')
    risk_analysis = prior_analyses.get('risk')
    
    print("Starting parallel TQDCS analysis across 5 dimensions...")
    
    # Prepare filtered KG data for each category
    filtered_data = {}
    categories = ['T', 'Q', 'D', 'C', 'S']
    category_names = ['technology', 'quality', 'delivery', 'cost', 'sustainability']
    
    for cat, name in zip(categories, category_names):
        filtered = filter_tqdcs_category_data(kg_data, cat)
        filtered_data[name] = filtered
        print(f"  {name.capitalize()}: {len(filtered['nodes'])} nodes after filtering")
    
    # Execute analyses in parallel
    results = {}
    max_workers = 5  # One worker per TQDCS category
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit all tasks
        futures[executor.submit(
            analyze_technology, 
            filtered_data['technology'], 
            supplier_name, 
            parts_analysis,
            model_name
        )] = 'technology'
        
        futures[executor.submit(
            analyze_quality,
            filtered_data['quality'],
            supplier_name,
            parts_analysis,
            risk_analysis,
            model_name
        )] = 'quality'
        
        futures[executor.submit(
            analyze_delivery,
            filtered_data['delivery'],
            supplier_name,
            risk_analysis,
            model_name,
            cost_analysis=cost_analysis
        )] = 'delivery'
        
        futures[executor.submit(
            analyze_cost,
            filtered_data['cost'],
            supplier_name,
            cost_analysis,
            risk_analysis,
            model_name
        )] = 'cost'
        
        futures[executor.submit(
            analyze_sustainability,
            filtered_data['sustainability'],
            supplier_name,
            model_name
        )] = 'sustainability'
        
        # Collect results as they complete
        for future in as_completed(futures):
            category = futures[future]
            try:
                results[category] = future.result()
                print(f"  ✓ {category.capitalize()} analysis complete")
            except Exception as e:
                print(f"  ✗ {category.capitalize()} analysis failed: {e}")
                results[category] = {
                    "score": 0,
                    "reasoning": f"Analysis failed: {str(e)}",
                    "strengths": [],
                    "weaknesses": [],
                    "key_findings": [],
                    "sources": []
                }
    
    # Retry failed analyses sequentially
    failed_categories = [cat for cat, result in results.items() 
                        if "Error in analysis" in result.get('reasoning', '') 
                        or "Analysis failed" in result.get('reasoning', '')
                        or result.get('score', 0) == 0]
    
    if failed_categories:
        print(f"Retrying {len(failed_categories)} failed analyses sequentially...")
        import time
        time.sleep(2)  # Brief pause before retry to avoid immediate connection issues
        
        for category in failed_categories:
            try:
                print(f"  Retrying {category} analysis...")
                if category == 'technology':
                    results[category] = analyze_technology(
                        filtered_data['technology'], 
                        supplier_name, 
                        parts_analysis
                    )
                elif category == 'quality':
                    results[category] = analyze_quality(
                        filtered_data['quality'], 
                        supplier_name, 
                        parts_analysis, 
                        risk_analysis
                    )
                elif category == 'delivery':
                    results[category] = analyze_delivery(
                        filtered_data['delivery'],
                        supplier_name,
                        risk_analysis,
                        None,
                        cost_analysis
                    )
                elif category == 'cost':
                    results[category] = analyze_cost(
                        filtered_data['cost'],
                        supplier_name,
                        cost_analysis,
                        risk_analysis
                    )
                elif category == 'sustainability':
                    results[category] = analyze_sustainability(
                        filtered_data['sustainability'],
                        supplier_name
                    )
                print(f"  ✓ {category.capitalize()} retry successful")
            except Exception as e:
                print(f"  ✗ {category.capitalize()} retry also failed: {e}")
                # Keep the error result structure for consistency
    
    print("Parallel analysis complete, combining results...")
    
    # Structure results in expected format
    tqdcs_scores = results
    
    # Compute overall assessment
    overall_assessment = compute_overall_assessment(tqdcs_scores, weights)
    
    # Extract detailed findings from the KG
    detailed_findings = {
        "compliance_status": {
            "technical_compliances": [],
            "non_compliances": [],
            "pending_items": [],
            "sources": []
        },
        "certifications": {
            "current": [],
            "expired_or_expiring": [],
            "missing_required": [],
            "sources": []
        },
        "capabilities": {
            "production": [],
            "testing": [],
            "innovation": [],
            "logistics": [],
            "sources": []
        }
    }
    
    # Extract certifications from nodes
    cert_nodes = [n for n in kg_data.get('nodes', []) 
                  if n.get('type') == 'Certification']
    for cert in cert_nodes:
        cert_name = cert.get('properties', {}).get('name', cert.get('id', ''))
        if cert_name:
            detailed_findings['certifications']['current'].append(cert_name)
    
    # Build scoring rationale
    scoring_rationale = {
        "methodology": "Parallel category-specific analysis with prior context integration",
        "key_factors": [
            f"Technology analysis included {len(filtered_data['technology']['nodes'])} nodes",
            f"Quality analysis included {len(filtered_data['quality']['nodes'])} nodes",
            f"Integrated results from parts, cost, and risk analyses where relevant"
        ],
        "data_quality": "Good - comprehensive KG with TQDCS categorization"
    }
    
    # Build final result
    result = {
        "tqdcs_scores": tqdcs_scores,
        "overall_assessment": overall_assessment,
        "detailed_findings": detailed_findings,
        "scoring_rationale": scoring_rationale
    }
    
    # Add weights information if custom weights were used
    if weights:
        result["scoring_weights"] = weights
    
    print("\nTQDCS Scores:")
    for category in ['technology', 'quality', 'delivery', 'cost', 'sustainability']:
        if category in tqdcs_scores:
            score = tqdcs_scores[category].get('score', 0)
            print(f"  {category.capitalize()}: {score}/5")
    print(f"  Average: {overall_assessment['average_score']}/5")
    
    # Save to file if requested
    if save_to_file and output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"TQDCS analysis saved to: {output_path}")
    
    return result


def analyze_tqdcs_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    use_parallel: bool = True,
    prior_analyses: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze TQDCS aspects and return structured JSON data with scores and reasoning
    
    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for the analysis
        weights: Optional weights for TQDCS categories (must sum to 1.0)
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        use_parallel: Whether to use parallel execution (default: True)
        prior_analyses: Optional dict with 'parts', 'cost', 'risk' analysis results
        
    Returns:
        Structured dictionary with TQDCS analysis
    """
    # Use parallel version if enabled and it's the default or explicitly requested
    if use_parallel:
        return analyze_tqdcs_structured_parallel(
            kg_json_path=kg_json_path,
            supplier_name=supplier_name,
            prior_analyses=prior_analyses,
            weights=weights,
            save_to_file=save_to_file,
            output_path=output_path,
            model_name=model_name
        )
    
    # Otherwise, use the original serial implementation
    print(f"Loading KG from: {kg_json_path}")
    
    # Load the full KG JSON
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    print(f"Loaded KG with {len(kg_data.get('nodes', []))} nodes and {len(kg_data.get('relationships', []))} relationships")
    
    # Build the prompt
    supplier_context = f"Focus on TQDCS evaluation for supplier: {supplier_name}\n" if supplier_name else ""
    
    base_prompt = f"""Analyze the knowledge graph and evaluate the supplier on TQDCS criteria.

{supplier_context}
Return ONLY valid JSON (no markdown, no explanations outside JSON) with this exact structure:
{{
  "tqdcs_scores": {{
    "technology": {{
      "score": <number between 1-5>,
      "reasoning": "Detailed explanation of the score",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"],
      "key_findings": ["Finding 1", "Finding 2"],
      "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
    }},
    "quality": {{
      "score": <number between 1-5>,
      "reasoning": "Detailed explanation of the score",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"],
      "key_findings": ["Finding 1", "Finding 2"],
      "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
    }},
    "delivery": {{
      "score": <number between 1-5>,
      "reasoning": "Detailed explanation of the score",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"],
      "key_findings": ["Finding 1", "Finding 2"],
      "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
    }},
    "cost": {{
      "score": <number between 1-5>,
      "reasoning": "Detailed explanation of the score",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"],
      "key_findings": ["Finding 1", "Finding 2"],
      "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
    }},
    "sustainability": {{
      "score": <number between 1-5>,
      "reasoning": "Detailed explanation of the score",
      "strengths": ["Strength 1", "Strength 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"],
      "key_findings": ["Finding 1", "Finding 2"],
      "sources": [{{"filename": "exact filename", "chunk_id": "page_X"}}]
    }}
  }},
  "detailed_findings": {{
    "compliance_status": {{
      "technical_compliances": ["List of technical compliances met"],
      "non_compliances": ["List of non-compliances identified"],
      "pending_items": ["Items pending confirmation"],
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }},
    "certifications": {{
      "current": ["List of current certifications (e.g., ISO 14001, IATF 16949)"],
      "expired_or_expiring": ["Certifications that are expired or expiring soon"],
      "missing_required": ["Required certifications that are missing"],
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }},
    "capabilities": {{
      "production": ["Production capabilities (e.g., capacity, facilities)"],
      "testing": ["Testing capabilities (e.g., test rigs, validation)"],
      "innovation": ["Innovation/R&D capabilities (e.g., patents, development)"],
      "logistics": ["Logistics capabilities (e.g., JIT, JIS, EDI)"],
      "sources": [{{"filename": "...", "chunk_id": "..."}}]
    }}
  }},
  "scoring_rationale": {{
    "methodology": "Brief explanation of how scores were determined",
    "key_factors": ["Primary factors that influenced the scoring"],
    "data_quality": "Assessment of the quality/completeness of available data"
  }}
}}

SCORING GUIDELINES:
1. Technology (1-5):
   - 5: Cutting-edge technology, full compliance, strong innovation, patents
   - 4: Modern technology, mostly compliant, good capabilities
   - 3: Adequate technology, some gaps but manageable
   - 2: Outdated or significantly non-compliant technology
   - 1: Major technology gaps, critical non-compliances

2. Quality (1-5):
   - 5: Exceptional quality systems, all certifications current, no issues
   - 4: Strong quality systems, minor gaps
   - 3: Adequate quality, some concerns but manageable
   - 2: Significant quality issues or missing certifications
   - 1: Major quality concerns, failed tests, expired certifications

3. Delivery (1-5):
   - 5: Excellent delivery capability, short lead times, high flexibility
   - 4: Good delivery performance, reasonable lead times
   - 3: Adequate delivery, some constraints
   - 2: Long lead times, capacity concerns
   - 1: Critical delivery issues, major bottlenecks

4. Cost (1-5):
   - 5: Highly competitive pricing, favorable terms, transparent
   - 4: Competitive pricing, reasonable terms
   - 3: Average pricing, standard terms
   - 2: Above market pricing, unfavorable terms
   - 1: Excessive pricing, very unfavorable terms

5. Sustainability (1-5):
   - 5: Industry leader in sustainability, clear commitments, certifications
   - 4: Strong sustainability practices, good documentation
   - 3: Basic sustainability compliance
   - 2: Limited sustainability efforts
   - 1: No clear sustainability commitment

IMPORTANT:
1. Base scores on actual evidence from the knowledge graph
2. Include specific examples in reasoning
3. Extract exact sources from metadata
4. Be objective and data-driven in scoring
5. Consider both positive and negative aspects
6. Look for specific mentions of:
   - Compliance/non-compliance statements
   - Test results and certifications
   - Lead times and capacity information
   - Pricing and cost structures
   - Environmental commitments
7. If information is limited for a category, reflect this in the scoring rationale
8. Order the strengths and weaknesses by significance. The first item should be the most significant.

Knowledge Graph:
{json.dumps(kg_data, indent=2)}

Return only the JSON object, no additional text."""
    
    # Apply PurchasingOrganization business context to the prompt
    prompt = format_prompt_with_business_context(base_prompt, analysis_type="tqdcs")

# """
# SCORING GUIDELINES:
# 1. Technology (1-5):
#    - 5: Cutting-edge technology, full compliance, strong innovation, patents
#    - 4: Modern technology, mostly compliant, good capabilities
#    - 3: Adequate technology, some gaps but manageable
#    - 2: Outdated or significantly non-compliant technology
#    - 1: Major technology gaps, critical non-compliances

# 2. Quality (1-5):
#    - 5: Exceptional quality systems, all certifications current, no issues
#    - 4: Strong quality systems, minor gaps
#    - 3: Adequate quality, some concerns but manageable
#    - 2: Significant quality issues or missing certifications
#    - 1: Major quality concerns, failed tests, expired certifications

# 3. Delivery (1-5):
#    - 5: Excellent delivery capability, short lead times, high flexibility
#    - 4: Good delivery performance, reasonable lead times
#    - 3: Adequate delivery, some constraints
#    - 2: Long lead times, capacity concerns
#    - 1: Critical delivery issues, major bottlenecks

# 4. Cost (1-5):
#    - 5: Highly competitive pricing, favorable terms, transparent
#    - 4: Competitive pricing, reasonable terms
#    - 3: Average pricing, standard terms
#    - 2: Above market pricing, unfavorable terms
#    - 1: Excessive pricing, very unfavorable terms

# 5. Sustainability (1-5):
#    - 5: Industry leader in sustainability, clear commitments, certifications
#    - 4: Strong sustainability practices, good documentation
#    - 3: Basic sustainability compliance
#    - 2: Limited sustainability efforts
#    - 1: No clear sustainability commitment
# """

    print(f"Prompt size: ~{len(prompt):,} characters")
    print("Sending to Gemini-2.5-pro for TQDCS analysis...")
    
    # Get LLM and analyze
    llm = create_llm(model_name="gemini-2.5-pro", temperature=0.0)
    
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
            # Return a fallback structure
            return {
                "error": "Failed to parse LLM response",
                "raw_response": content,
                "tqdcs_scores": {
                    "technology": {"score": 0, "reasoning": "Error in parsing", "sources": []},
                    "quality": {"score": 0, "reasoning": "Error in parsing", "sources": []},
                    "delivery": {"score": 0, "reasoning": "Error in parsing", "sources": []},
                    "cost": {"score": 0, "reasoning": "Error in parsing", "sources": []},
                    "sustainability": {"score": 0, "reasoning": "Error in parsing", "sources": []}
                },
                "overall_assessment": {
                    "average_score": 0,
                    "summary": "Unable to assess due to parsing error"
                }
            }
        
        # Extract TQDCS scores from LLM result
        tqdcs_scores = llm_result.get('tqdcs_scores', {})
        detailed_findings = llm_result.get('detailed_findings', {})
        scoring_rationale = llm_result.get('scoring_rationale', {})
        
        # Compute overall assessment programmatically
        overall_assessment = compute_overall_assessment(tqdcs_scores, weights)
        
        # Build final result
        result = {
            "tqdcs_scores": tqdcs_scores,
            "overall_assessment": overall_assessment,
            "detailed_findings": detailed_findings,
            "scoring_rationale": scoring_rationale
        }
        
        # Add weights information if custom weights were used
        if weights:
            result["scoring_weights"] = weights
        
        print("TQDCS analysis complete!")
        
        # Print summary
        print("\nTQDCS Scores:")
        for category in ['technology', 'quality', 'delivery', 'cost', 'sustainability']:
            if category in tqdcs_scores:
                score = tqdcs_scores[category].get('score', 0)
                print(f"  {category.capitalize()}: {score}/5")
        print(f"  Average: {overall_assessment['average_score']}/5")
        print(f"  Strongest: {overall_assessment['strongest_area']}")
        print(f"  Weakest: {overall_assessment['weakest_area']}")
        
        # Save to file if requested
        if save_to_file and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Structured TQDCS analysis saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        raise


def main():
    """Test the structured TQDCS analyzer"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    
    # Test with both suppliers
    suppliers = [
        ("SupplierA", "supplier 1/supplier_1_kg_image/unified_kg_20250723_195008.json"),
        ("SupplierB", "supplier 2/supplier_2_kg_image/unified_kg_20250723_200647.json")
    ]
    
    # Example of custom weights (optional)
    custom_weights = {
        'technology': 0.25,  # Higher weight on technology
        'quality': 0.25,     # Higher weight on quality
        'delivery': 0.20,
        'cost': 0.20,
        'sustainability': 0.10  # Lower weight on sustainability
    }
    
    for supplier_name, kg_relative_path in suppliers:
        kg_path = str(project_root / f"docs/business/{kg_relative_path}")
        
        try:
            print(f"\n{'='*50}")
            print(f"Analyzing {supplier_name}")
            print('='*50)
            
            # Test with default weights
            result = analyze_tqdcs_structured(
                kg_json_path=kg_path,
                supplier_name=supplier_name,
                weights=None,  # Use default equal weights
                save_to_file=True,
                output_path=f"output/structured_tqdcs_{supplier_name.replace(' ', '_').lower()}.json"
            )
            
            if 'overall_assessment' in result:
                print(f"\nOverall: {result['overall_assessment']['summary']}")
                print(f"Recommendation: {result['overall_assessment']['recommendation']}")
                
        except FileNotFoundError:
            print(f"File not found: {kg_path}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()