"""
Structured Risk Analyzer - Returns JSON with Mitigation Strategies

This module analyzes risks from knowledge graphs and returns structured JSON data
with severity levels, categories, and actionable mitigation strategies.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context


def compute_risk_summary(risks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute risk summary from the list of risks
    
    Args:
        risks: List of risk dictionaries
        
    Returns:
        Risk summary with counts and critical risks
    """
    high_risks = [r for r in risks if r.get('severity') == 'High']
    medium_risks = [r for r in risks if r.get('severity') == 'Medium']
    low_risks = [r for r in risks if r.get('severity') == 'Low']
    
    # Identify critical risks (High severity + High probability)
    critical_risks = [
        r['title'] for r in risks 
        if r.get('severity') == 'High' and r.get('probability') == 'High'
    ]
    
    # If no critical risks based on probability, take top high severity risks
    if not critical_risks:
        critical_risks = [r['title'] for r in high_risks[:5]]
    
    # Overall assessment based on risk profile
    total_risks = len(risks)
    if len(high_risks) > total_risks * 0.4:
        overall_assessment = "High risk supplier - significant concerns require immediate attention"
    elif len(high_risks) > total_risks * 0.2:
        overall_assessment = "Moderate-to-high risk supplier - several important issues need addressing"
    elif len(medium_risks) > total_risks * 0.5:
        overall_assessment = "Moderate risk supplier - manageable concerns with proper mitigation"
    else:
        overall_assessment = "Low-to-moderate risk supplier - standard risk profile"
    
    return {
        "high_risks_count": len(high_risks),
        "medium_risks_count": len(medium_risks),
        "low_risks_count": len(low_risks),
        "critical_risks": critical_risks[:5],  # Top 5 critical risks
        "overall_risk_assessment": overall_assessment
    }


def compute_risk_matrix(risks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize risks by category into a risk matrix
    
    Args:
        risks: List of risk dictionaries
        
    Returns:
        Risk matrix organized by category
    """
    categories = {
        "technical_risks": [],
        "financial_risks": [],
        "operational_risks": [],
        "legal_compliance_risks": [],
        "strategic_risks": [],
        "quality_risks": [],
        "supply_chain_risks": []
    }
    
    # Map categories to matrix keys
    category_mapping = {
        "Technical": "technical_risks",
        "Financial": "financial_risks",
        "Operational": "operational_risks",
        "Legal": "legal_compliance_risks",
        "Strategic": "strategic_risks",
        "Quality": "quality_risks",
        "Supply Chain": "supply_chain_risks"
    }
    
    for risk in risks:
        category = risk.get('category', 'Operational')
        matrix_key = category_mapping.get(category, "operational_risks")
        
        categories[matrix_key].append({
            "risk": risk['title'],
            "severity": risk['severity'],
            "probability": risk.get('probability', 'Medium'),
            "sources": risk.get('sources', [])
        })
    
    return categories


def analyze_risks_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze risks and return structured JSON data with mitigation strategies
    
    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for the analysis
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        
    Returns:
        Structured dictionary with risk analysis and mitigation strategies
    """
    print(f"Loading KG from: {kg_json_path}")
    
    # Load the full KG JSON
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    print(f"Loaded KG with {len(kg_data.get('nodes', []))} nodes and {len(kg_data.get('relationships', []))} relationships")
    
    # Build the prompt
    supplier_context = f"Focus on risks related to supplier: {supplier_name}\n" if supplier_name else ""
    
    base_prompt = f"""Analyze the knowledge graph and identify all risks associated with this supplier.

{supplier_context}
Return ONLY valid JSON (no markdown, no explanations outside JSON) with this exact structure:
{{
  "risks": [
    {{
      "title": "Brief, descriptive title of the risk",
      "severity": "High|Medium|Low",
      "category": "Technical|Financial|Operational|Legal|Strategic|Quality|Supply Chain",
      "description": "Detailed description of the risk and its implications",
      "impact": "Specific impact on the project if this risk materializes",
      "probability": "High|Medium|Low",
      "sources": [{{"filename": "exact filename from metadata", "chunk_id": "page_X"}}],
      "mitigation": "Specific, actionable mitigation strategy with clear steps",
      "mitigation_actions": [
        "Action 1: Specific action to take",
        "Action 2: Another specific action"
      ]
    }}
  ],
  "mitigation_plan": {{
    "immediate_actions": [
      {{
        "action": "Specific immediate action required",
        "responsible_party": "Who should take this action",
        "timeline": "When this should be completed",
        "related_risks": ["List of risk titles this addresses"]
      }}
    ],
    "short_term_actions": [
      {{
        "action": "Action to take within 3-6 months",
        "responsible_party": "Who should take this action",
        "timeline": "Specific timeline",
        "related_risks": ["Risk titles addressed"]
      }}
    ],
    "long_term_actions": [
      {{
        "action": "Strategic action for long-term risk mitigation",
        "responsible_party": "Who should take this action",
        "timeline": "Timeline (6+ months)",
        "related_risks": ["Risk titles addressed"]
      }}
    ]
  }}
}}

IMPORTANT GUIDELINES:
1. Extract exact filename and chunk_id from the knowledge graph metadata
2. Base severity on actual impact to project success:
   - High: Could cause project failure, major delays, or significant cost overruns
   - Medium: Could cause moderate delays or cost increases
   - Low: Minor impact, easily manageable
3. Base probability on likelihood of occurrence:
   - High: Very likely to occur based on current evidence
   - Medium: Possible occurrence
   - Low: Unlikely but possible
4. Mitigation strategies must be:
   - Specific and actionable (not generic advice)
   - Tailored to the supplier context
   - Based on industry best practices
   - Include concrete steps that can be implemented
5. Categories to use:
   - Technical: Non-compliance, performance issues, integration problems
   - Financial: Cost volatility, payment terms, hidden costs
   - Operational: Lead times, capacity, logistics
   - Legal: Compliance, certifications, contractual issues
   - Strategic: Long-term partnership risks, dependency
   - Quality: Quality control, testing, standards
   - Supply Chain: Material sourcing, bottlenecks, flexibility
6. Look for risks in:
   - Non-compliances mentioned in the documents
   - Missing certifications or pending confirmations
   - Long lead times or capacity constraints
   - Unfavorable contract terms
   - Quality issues or test failures
   - Geographic or political factors
   - Dependency on single sources
   - Cost escalation clauses
   - Production relocation rights
7. For the mitigation plan:
   - Immediate actions: Must be done before or at contract signing
   - Short-term actions: Within first 6 months of engagement
   - Long-term actions: Strategic initiatives for ongoing risk management

Knowledge Graph:
{json.dumps(kg_data, indent=2)}

Return only the JSON object, no additional text."""
    
    # Apply PurchasingOrganization business context to the prompt
    prompt = format_prompt_with_business_context(base_prompt, analysis_type="risk")
    
    print(f"Prompt size: ~{len(prompt):,} characters")
    selected_model = model_name or "gemini-2.5-pro"
    print(f"Sending to {selected_model} for structured risk analysis...")
    
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
                "risks": [],
                "risk_summary": {
                    "high_risks_count": 0,
                    "medium_risks_count": 0,
                    "low_risks_count": 0,
                    "critical_risks": ["Error in parsing"],
                    "overall_risk_assessment": "Unable to assess due to parsing error"
                },
                "risk_matrix": {},
                "mitigation_plan": {"immediate_actions": [], "short_term_actions": [], "long_term_actions": []}
            }
        
        # Extract risks and mitigation plan from LLM response
        risks = llm_result.get('risks', [])
        mitigation_plan = llm_result.get('mitigation_plan', {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": []
        })
        
        # Compute risk summary and matrix from the risks
        risk_summary = compute_risk_summary(risks)
        risk_matrix = compute_risk_matrix(risks)
        
        # Build final result
        result = {
            "risks": risks,
            "risk_summary": risk_summary,
            "risk_matrix": risk_matrix,
            "mitigation_plan": mitigation_plan
        }
        
        print("Risk analysis complete!")
        print(f"  - Identified {len(risks)} risks")
        print(f"  - High: {risk_summary['high_risks_count']}, Medium: {risk_summary['medium_risks_count']}, Low: {risk_summary['low_risks_count']}")
        
        # Save to file if requested
        if save_to_file and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Structured risk analysis saved to: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        raise


def main():
    """Test the structured risk analyzer"""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    kg_path = str(project_root / "docs/business/supplier 2/supplier_2_kg_image/unified_kg_20250723_200647.json")
    
    try:
        result = analyze_risks_structured(
            kg_json_path=kg_path,
            supplier_name="SupplierB",
            save_to_file=True,
            output_path="output/structured_risk_analysis_supplierb.json"
        )
        
        print("\n=== STRUCTURED RISK ANALYSIS ===")
        print(f"Total Risks Identified: {len(result['risks'])}")
        print(f"Risk Summary: {result['risk_summary']['overall_risk_assessment']}")
        print("\nRisk Distribution by Category:")
        for category, risks in result['risk_matrix'].items():
            if risks:
                print(f"  {category}: {len(risks)} risks")
        print("\nMitigation Plan:")
        print(f"  Immediate Actions: {len(result['mitigation_plan']['immediate_actions'])}")
        print(f"  Short-term Actions: {len(result['mitigation_plan']['short_term_actions'])}")
        print(f"  Long-term Actions: {len(result['mitigation_plan']['long_term_actions'])}")
        
    except FileNotFoundError:
        print(f"File not found: {kg_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()