"""
Structured TQDCS Analyzer - Returns JSON with TQDCS Scores

This module analyzes TQDCS (Technology, Quality, Delivery, Cost, Sustainability) aspects
from knowledge graphs and returns structured scores with reasoning.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

# Add parent directories for imports
current_file = Path(__file__).resolve()
kg_creation_dir = current_file.parent.parent.parent.parent
sys.path.append(str(kg_creation_dir))

from llm.factory import create_llm
from .business_context import format_prompt_with_business_context


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


def analyze_tqdcs_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze TQDCS aspects and return structured JSON data with scores and reasoning
    
    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for the analysis
        weights: Optional weights for TQDCS categories (must sum to 1.0)
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        
    Returns:
        Structured dictionary with TQDCS analysis
    """
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