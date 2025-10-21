"""
Structured Homepage Analyzer - RFQ Overview Extraction

This module ingests the FULL knowledge graph (no filtering) and asks an LLM
to extract structured RFQ overview information for display on the dashboard
homepage. The output is a structured JSON that can be cached and rendered
in Streamlit tables side-by-side for two suppliers.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from app.core.llm import create_llm
from .business_context import format_prompt_with_business_context


def _coerce_homepage_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce arbitrary LLM output into the strict homepage schema with value/sources.

    Any missing fields are filled with {"value": "Not found", "sources": []}.
    If a field is a plain string, it is wrapped into the object with empty sources.
    """
    sections: Dict[str, list[str]] = {
        "project_information": [
            "project_title",
            "project_description",
            "contracting_authority",
            "estimated_contract_value",
            "contract_duration",
            "location_of_works_services",
            "contact_person_details",
        ],
        "key_dates_and_deadlines": [
            "rfq_issue_date",
            "clarification_deadline",
            "submission_deadline",
            "contract_award_date",
            "contract_start_date",
            "contract_completion_date",
            "rfq_validity_period",
        ],
        "scope_and_technical_requirements": [
            "scope_of_work",
            "methodology_requirements",
            "technical_specifications",
            "required_outputs_deliverables",
            "equipment_or_materials",
            "site_access_requirements",
            "data_management_standards",
        ],
        "supplier_requirements": [
            "mandatory_supplier_information",
            "financial_thresholds",
            "insurance_requirements",
            "health_and_safety_compliance",
            "certifications_and_accreditations",
            "references_and_experience",
            "key_personnel_qualifications",
            "subcontractor_requirements",
            "equality_diversity_inclusion",
        ],
        "evaluation_criteria": [
            "evaluation_methodology",
            "technical_criteria",
            "commercial_criteria",
            "scoring_system",
            "award_decision_basis",
        ],
        "pricing_and_payment": [
            "pricing_format",
            "price_inclusions",
            "payment_terms",
            "invoicing_requirements",
        ],
        "legal_and_contractual": [
            "terms_and_conditions",
            "acceptance_of_terms",
            "confidentiality_requirements",
            "intellectual_property_rights",
            "freedom_of_information",
            "anti_corruption_and_bribery",
            "termination_clauses",
            "dispute_resolution",
        ],
        "compliance_and_exclusion_grounds": [
            "mandatory_exclusion_grounds",
            "discretionary_exclusion_grounds",
            "self_declaration_requirements",
            "evidence_submission_timing",
        ],
        "sustainability_and_social_value": [
            "sustainability_commitments",
            "social_value_requirements",
            "modern_slavery_compliance",
        ],
        "contract_management_and_reporting": [
            "contract_management_arrangements",
            "progress_reporting_requirements",
            "key_project_milestones",
            "quality_assurance_measures",
        ],
    }

    def normalize_field(field_value: Any) -> Dict[str, Any]:
        if isinstance(field_value, dict):
            val = field_value.get("value", "Not found")
            src = field_value.get("sources", [])
            if val is None or (isinstance(val, str) and not val.strip()):
                val = "Not found"
            if not isinstance(src, list):
                src = []
            return {"value": val, "sources": src}
        if isinstance(field_value, str):
            return {"value": field_value or "Not found", "sources": []}
        return {"value": "Not found", "sources": []}

    normalized: Dict[str, Any] = {}
    for section, keys in sections.items():
        src_obj = raw.get(section, {}) if isinstance(raw, dict) else {}
        norm_section: Dict[str, Any] = {}
        for key in keys:
            norm_section[key] = normalize_field(src_obj.get(key))
        normalized[section] = norm_section
    return normalized


def _extract_core_json(content: str) -> str:
    """Extract substring between first '{' and last '}' to increase parse success."""
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return content[start:end + 1]
    except Exception:
        pass
    return content


def _cleanup_common_json_issues(content: str) -> str:
    """Attempt to fix common LLM JSON issues before parsing."""
    # Replace known key typos
    fixed = content.replace('"filename_"', '"filename"')
    # Remove trailing commas before } or ]
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
    # Remove stray backslashes before quotes in keys
    fixed = re.sub(r"\\\"", '"', fixed)
    # Ensure proper quotes around keys (best-effort)
    return fixed

def analyze_homepage_structured(
    kg_json_path: str,
    supplier_name: Optional[str] = None,
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the full KG and extract RFQ overview fields for the homepage.

    Args:
        kg_json_path: Path to the unified KG JSON file
        supplier_name: Optional supplier name for context
        save_to_file: Whether to save the JSON to a file
        output_path: Optional path for saving the JSON
        model_name: Optional model override (defaults to gemini-2.5-pro)

    Returns:
        Structured dictionary with homepage RFQ overview
    """
    print(f"Loading KG for homepage overview from: {kg_json_path}")

    # Load the full KG JSON (no filtering)
    with open(kg_json_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)

    nodes_count = len(kg_data.get('nodes', []))
    rels_count = len(kg_data.get('relationships', []))
    approx_tokens = len(json.dumps(kg_data)) // 4
    print(f"Loaded KG with {nodes_count} nodes and {rels_count} relationships (~{approx_tokens:,} tokens)")

    supplier_context = (
        f"Focus on extracting information that applies to supplier: {supplier_name}\n"
        if supplier_name
        else ""
    )

    # Strict schema: ask for exact keys and instruct to use "Not found" if missing.
    # IMPORTANT: Each field must include value and sources[] extracted from node metadata.
    base_prompt = f"""
You are given a complete knowledge graph for an RFQ/procurement. Extract the RFQ overview
information EXACTLY in the JSON schema below. If a field cannot be found, set its value to
the string "Not found". Do not invent information.

{supplier_context}
Return ONLY valid JSON (no markdown) with EXACTLY these top-level keys and nested fields.
For EVERY field, provide an object with:
- "value": the extracted text (or "Not found")
- "sources": an array of source objects extracted from the knowledge graph node "metadata" entries, each as {{"filename":"...","chunk_id":"..."}}. If none found, use [].

{{
  "project_information": {{
    "project_title": {{"value":"...","sources":[{{"filename":"...","chunk_id":"..."}}]}},
    "project_description": {{"value":"...","sources":[...] }},
    "contracting_authority": {{"value":"...","sources":[...] }},
    "estimated_contract_value": {{"value":"...","sources":[...] }},  
    "contract_duration": {{"value":"...","sources":[...] }},
    "location_of_works_services": {{"value":"...","sources":[...] }},
    "contact_person_details": {{"value":"...","sources":[...] }}
  }},
  "key_dates_and_deadlines": {{
    "rfq_issue_date": {{"value":"...","sources":[...] }},
    "clarification_deadline": {{"value":"...","sources":[...] }},
    "submission_deadline": {{"value":"...","sources":[...] }},
    "contract_award_date": {{"value":"...","sources":[...] }},
    "contract_start_date": {{"value":"...","sources":[...] }},
    "contract_completion_date": {{"value":"...","sources":[...] }},
    "rfq_validity_period": {{"value":"...","sources":[...] }}
  }},
  "scope_and_technical_requirements": {{
    "scope_of_work": {{"value":"...","sources":[...] }},
    "methodology_requirements": {{"value":"...","sources":[...] }},
    "technical_specifications": {{"value":"...","sources":[...] }},
    "required_outputs_deliverables": {{"value":"...","sources":[...] }},
    "equipment_or_materials": {{"value":"...","sources":[...] }},
    "site_access_requirements": {{"value":"...","sources":[...] }},
    "data_management_standards": {{"value":"...","sources":[...] }}
  }},
  "supplier_requirements": {{
    "mandatory_supplier_information": {{"value":"...","sources":[...] }},
    "financial_thresholds": {{"value":"...","sources":[...] }},
    "insurance_requirements": {{"value":"...","sources":[...] }},
    "health_and_safety_compliance": {{"value":"...","sources":[...] }},
    "certifications_and_accreditations": {{"value":"...","sources":[...] }},
    "references_and_experience": {{"value":"...","sources":[...] }},
    "key_personnel_qualifications": {{"value":"...","sources":[...] }},
    "subcontractor_requirements": {{"value":"...","sources":[...] }},
    "equality_diversity_inclusion": {{"value":"...","sources":[...] }}
  }},
  "evaluation_criteria": {{
    "evaluation_methodology": {{"value":"...","sources":[...] }},
    "technical_criteria": {{"value":"...","sources":[...] }},
    "commercial_criteria": {{"value":"...","sources":[...] }},
    "scoring_system": {{"value":"...","sources":[...] }},
    "award_decision_basis": {{"value":"...","sources":[...] }}
  }},
  "pricing_and_payment": {{
    "pricing_format": {{"value":"...","sources":[...] }},
    "price_inclusions": {{"value":"...","sources":[...] }},
    "payment_terms": {{"value":"...","sources":[...] }},
    "invoicing_requirements": {{"value":"...","sources":[...] }}
  }},
  "legal_and_contractual": {{
    "terms_and_conditions": {{"value":"...","sources":[...] }},
    "acceptance_of_terms": {{"value":"...","sources":[...] }},
    "confidentiality_requirements": {{"value":"...","sources":[...] }},
    "intellectual_property_rights": {{"value":"...","sources":[...] }},
    "freedom_of_information": {{"value":"...","sources":[...] }},
    "anti_corruption_and_bribery": {{"value":"...","sources":[...] }},
    "termination_clauses": {{"value":"...","sources":[...] }},
    "dispute_resolution": {{"value":"...","sources":[...] }}
  }},
  "compliance_and_exclusion_grounds": {{
    "mandatory_exclusion_grounds": {{"value":"...","sources":[...] }},
    "discretionary_exclusion_grounds": {{"value":"...","sources":[...] }},
    "self_declaration_requirements": {{"value":"...","sources":[...] }},
    "evidence_submission_timing": {{"value":"...","sources":[...] }}
  }},
  "sustainability_and_social_value": {{
    "sustainability_commitments": {{"value":"...","sources":[...] }},
    "social_value_requirements": {{"value":"...","sources":[...] }},
    "modern_slavery_compliance": {{"value":"...","sources":[...] }}
  }},
  "contract_management_and_reporting": {{
    "contract_management_arrangements": {{"value":"...","sources":[...] }},
    "progress_reporting_requirements": {{"value":"...","sources":[...] }},
    "key_project_milestones": {{"value":"...","sources":[...] }},
    "quality_assurance_measures": {{"value":"...","sources":[...] }}
  }}
}}

Rules:
1) Extract ONLY from the provided knowledge graph content.
2) Use concise, human-readable sentences; include units/currency where applicable. Include relevant sources like quantities, numbers, phone numbers, email addresses, etc. when necessary.
3) If a value is unavailable, set it to "Not found".
3.5) If a value can be inferred from the knowledge graph, like total cost, compute it, explaining the methodology.
4) For sources: extract EXACT filename and chunk_id from node "metadata" arrays that support each answer. Prefer the most specific nodes.
5) Do not include any additional keys or commentary.

Knowledge Graph:
{json.dumps(kg_data, indent=2)}

Return only the JSON object, no additional text.
"""

    prompt = format_prompt_with_business_context(base_prompt, analysis_type="homepage")

    print(f"Prompt size: ~{len(prompt):,} characters")
    selected_model = model_name or "gemini-2.5-pro"
    print(f"Sending to {selected_model} for homepage overview extraction...")

    llm = create_llm(model_name=selected_model, temperature=0.0)

    try:
        response = llm.invoke(prompt)

        # Extract text content from potential SDK response types
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            content = content[0] if len(content) == 1 else ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)

        # Clean code fences if present
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        try:
            # Try direct parse first
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try extracting core JSON and cleaning common issues
                content_core = _extract_core_json(content)
                content_fixed = _cleanup_common_json_issues(content_core)
                result = json.loads(content_fixed)
            # Coerce to expected schema
            result = _coerce_homepage_schema(result)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response was: {content[:500]}...")
            # Fallback structure with "Not found" values
            result = _coerce_homepage_schema({})

        print("Homepage overview analysis complete!")

        if save_to_file and output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Homepage structured analysis saved to: {output_path}")

        return result

    except Exception as e:
        print(f"Error during homepage LLM analysis: {e}")
        raise


def main():
    """Manual test entrypoint for the structured homepage analyzer."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    kg_path = str(project_root / "docs/business/supplier 1/supplier_1_kg_image/unified_kg_20250723_195008.json")

    try:
        result = analyze_homepage_structured(
            kg_json_path=kg_path,
            supplier_name="SupplierA",
            save_to_file=True,
            output_path="output/structured_homepage_suppliera.json"
        )

        print("\n=== STRUCTURED HOMEPAGE OVERVIEW ===")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...")

    except FileNotFoundError:
        print(f"File not found: {kg_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


