"""
RFQ extraction schema definitions for structured information extraction.

This module defines the comprehensive schema for extracting RFQ information
directly from complete PDF documents using large context models.
"""

from typing import Dict, Any, List, Tuple

# Comprehensive RFQ extraction schema
RFQ_EXTRACTION_SCHEMA = {
    "project_information": {
        "project_title": "string",
        "project_description": "string", 
        "contracting_authority": "string",
        "country_of_contracting_authority": "string",
        "estimated_contract_value": "string",
        "contract_duration": "string",
        "location_of_works_services": "string",
        "contact_person_details": "string"
    },
    
    "key_dates_deadlines": {
        "rfq_issue_date": "string",
        "clarification_deadline": "string", 
        "submission_deadline": "string",
        "contract_award_date": "string",
        "contract_start_date": "string",
        "contract_completion_date": "string",
        "rfq_validity_period": "string"
    },
    
    "scope_technical_requirements": {
        "scope_of_work": "string",
        "methodology_requirements": "string",
        "technical_specifications": "string",
        "required_outputs_deliverables": "string",
        "equipment_or_materials": "string",
        "site_access_requirements": "string",
        "data_management_standards": "string"
    },
    
    "supplier_requirements": {
        "mandatory_supplier_information": "string",
        "financial_thresholds": "string",
        "insurance_requirements": "string",
        "health_and_safety_compliance": "string",
        "certifications_and_accreditations": "string",
        "references_and_experience": "string",
        "key_personnel_qualifications": "string",
        "subcontractor_requirements": "string",
        "equality_diversity_inclusion": "string"
    },
    
    "evaluation_criteria": {
        "evaluation_methodology": "string",
        "technical_criteria": "string",
        "commercial_criteria": "string",
        "scoring_system": "string",
        "award_decision_basis": "string"
    },
    
    "pricing_payment": {
        "pricing_format": "string",
        "price_inclusions": "string",
        "payment_terms": "string",
        "invoicing_requirements": "string"
    },
    
    "legal_contractual": {
        "terms_and_conditions": "string",
        "acceptance_of_terms": "string",
        "confidentiality_requirements": "string",
        "intellectual_property_rights": "string",
        "freedom_of_information": "string",
        "anti_corruption_and_bribery": "string",
        "termination_clauses": "string",
        "dispute_resolution": "string"
    },
    
    "compliance_exclusion_grounds": {
        "mandatory_exclusion_grounds": "string",
        "discretionary_exclusion_grounds": "string",
        "self_declaration_requirements": "string",
        "evidence_submission_timing": "string"
    },
    
    "sustainability_social_value": {
        "sustainability_commitments": "string",
        "social_value_requirements": "string",
        "modern_slavery_compliance": "string"
    },
    
    "contract_management_reporting": {
        "contract_management_arrangements": "string",
        "progress_reporting_requirements": "string",
        "key_project_milestones": "string",
        "quality_assurance_measures": "string"
    }
}
# RFI_EXTRACTION_SCHEMA = {
#     "company_information": {
#         "company_name_headquarters_offices": "string",
#         "years_in_business_experience": "string",
#         "relevant_certifications": "string",
#         "customer_references_case_studies": "string"
#     },
#     "solution_approach_technical_capabilities": {
#         "understanding_of_requirement_approach": "string",
#         "technologies_frameworks_used": "string",
#         "custom_vs_off_the_shelf_components": "string",
#         "ai_ml_automation_analytics_capabilities": "string"
#     },
#     "integration_interoperability": {
#         "experience_integrating_similar_systems": "string",
#         "api_capabilities_middleware_compatibility": "string",
#         "data_migration_synchronization_strategies": "string"
#     },
#     "implementation_delivery": {
#         "project_methodology": "string",
#         "estimated_timelines_milestones": "string",
#         "resource_skill_requirements": "string",
#         "post_implementation_support_maintenance": "string"
#     },
#     "pricing_licensing": {
#         "high_level_cost_estimates": "string",
#         "cost_components": "string",
#         "licensing_models": "string"
#     },
#     "compliance_security": {
#         "security_standards_data_protection_policies": "string",
#         "regulatory_compliance": "string",
#         "disaster_recovery_business_continuity_plans": "string"
#     },
#     "vendor_differentiation": {
#         "unique_value_proposition": "string",
#         "innovation_future_roadmap": "string",
#         "partnership_co_development_opportunities": "string"
#     },
#     "submission_details": {
#         "rfi_response_format": "string",
#         "submission_deadline": "string",
#         "contact_details_for_queries": "string",
#         "next_steps": "string"
#     }
# }


def get_flat_schema() -> Dict[str, str]:
    """
    Get a flattened version of the schema for direct extraction.
    
    Returns:
        Dictionary mapping field names to descriptions
    """
    flat_schema = {}
    
    for category, fields in RFQ_EXTRACTION_SCHEMA.items():
        for field_name, field_type in fields.items():
            flat_schema[field_name] = field_type
    
    return flat_schema


def get_extraction_instructions() -> str:
    """
    Get detailed instructions for RFQ information extraction.
    
    Returns:
        Comprehensive extraction instructions
    """
    
    return """
EXTRACTION INSTRUCTIONS:

1. PROJECT INFORMATION
   - project_title: The official name/title of the project or procurement
   - project_description: Detailed description of what work/services are required
   - contracting_authority: Name of the organization issuing the RFQ
   - estimated_contract_value: Budget or estimated value (include currency)
   - contract_duration: How long the contract will run (start to end dates or duration)
   - location_of_works_services: Where the work will be performed
   - contact_person_details: Name, email, phone of the contact person

2. KEY DATES & DEADLINES  
   - rfq_issue_date: When the RFQ was published/issued
   - clarification_deadline: Last date to ask questions about the RFQ
   - submission_deadline: Final deadline for submitting responses
   - contract_award_date: When the contract will be awarded
   - contract_start_date: When work is expected to begin
   - contract_completion_date: When work must be completed
   - rfq_validity_period: How long submitted quotes remain valid

3. SCOPE & TECHNICAL REQUIREMENTS
   - scope_of_work: Detailed breakdown of all work/services required
   - methodology_requirements: Required approaches, methods, or standards
   - technical_specifications: Detailed technical requirements and standards
   - required_outputs_deliverables: What must be delivered (reports, products, etc.)
   - equipment_or_materials: Specific equipment or materials required
   - site_access_requirements: Rules for accessing work sites
   - data_management_standards: Requirements for handling data

4. SUPPLIER REQUIREMENTS
   - mandatory_supplier_information: Required company information to provide
   - financial_thresholds: Minimum financial requirements (turnover, etc.)
   - insurance_requirements: Required insurance coverage and amounts
   - health_and_safety_compliance: H&S requirements and certifications
   - certifications_and_accreditations: Required industry certifications
   - references_and_experience: Required previous experience and references
   - key_personnel_qualifications: Required qualifications for key staff
   - subcontractor_requirements: Rules for using subcontractors
   - equality_diversity_inclusion: EDI requirements and commitments

5. EVALUATION CRITERIA
   - evaluation_methodology: How responses will be evaluated
   - technical_criteria: Technical scoring criteria and weightings
   - commercial_criteria: Price/commercial scoring criteria
   - scoring_system: How points/scores are calculated
   - award_decision_basis: Final decision criteria (e.g., lowest price, best value)

6. PRICING & PAYMENT
   - pricing_format: How to structure and present pricing
   - price_inclusions: What must be included in prices
   - payment_terms: When and how payments will be made
   - invoicing_requirements: How to submit invoices

7. LEGAL & CONTRACTUAL
   - terms_and_conditions: Reference to T&Cs that will apply
   - acceptance_of_terms: Requirements for accepting contract terms
   - confidentiality_requirements: Confidentiality and NDA requirements
   - intellectual_property_rights: IP ownership and usage rights
   - freedom_of_information: FOI disclosure requirements
   - anti_corruption_and_bribery: Anti-corruption policies and requirements
   - termination_clauses: Conditions under which contract can be terminated
   - dispute_resolution: Process for resolving contract disputes

8. COMPLIANCE & EXCLUSION GROUNDS
   - mandatory_exclusion_grounds: Reasons that automatically exclude suppliers
   - discretionary_exclusion_grounds: Reasons that may exclude suppliers
   - self_declaration_requirements: Required self-declarations from suppliers
   - evidence_submission_timing: When compliance evidence must be provided

9. SUSTAINABILITY & SOCIAL VALUE
   - sustainability_commitments: Environmental and sustainability requirements
   - social_value_requirements: Social value delivery requirements
   - modern_slavery_compliance: Modern slavery compliance requirements

10. CONTRACT MANAGEMENT & REPORTING
    - contract_management_arrangements: How the contract will be managed
    - progress_reporting_requirements: Required progress reports and frequency
    - key_project_milestones: Important project milestones and deadlines
    - quality_assurance_measures: Quality control and assurance requirements

EXTRACTION RULES:
- Extract information exactly as it appears in the document
- Use "Not Found" if information is not present
- For dates, preserve the original format
- For monetary values, include currency symbols
- Be comprehensive - check all sections of the document
- Look for information in tables, appendices, and footnotes
"""


def create_comparison_schema() -> Dict[str, Any]:
    """
    Create schema for comparing two extracted RFQ documents.
    
    Returns:
        Comparison schema structure
    """
    
    return {
        "comparison_metadata": {
            "document_1_name": "string",
            "document_2_name": "string", 
            "comparison_date": "string",
            "total_fields_compared": "number"
        },
        
        "field_by_field_comparison": {
            # This will be populated with each field from the extraction schema
            # showing values from both documents side by side
        },
        
        "summary_statistics": {
            "fields_found_in_both": "number",
            "fields_only_in_doc1": "number", 
            "fields_only_in_doc2": "number",
            "fields_not_found_in_either": "number",
            "completeness_doc1_percentage": "number",
            "completeness_doc2_percentage": "number"
        },
        
        "key_differences": [
            {
                "field_name": "string",
                "doc1_value": "string", 
                "doc2_value": "string",
                "difference_type": "string"  # "missing", "different_value", "different_format"
            }
        ],
        
        "recommendations": [
            "string"  # List of recommendations based on the comparison
        ]
    }


# Specific query templates for common RFQ questions
COMMON_RFQ_QUERIES = {
    "submission_deadline": "What is the deadline for submitting responses to this RFQ?",
    "estimated_value": "What is the estimated contract value or budget for this project?",
    "evaluation_criteria": "How will responses be evaluated? What are the scoring criteria?",
    "technical_requirements": "What are the key technical requirements and specifications?",
    "contact_information": "Who is the contact person for this RFQ and what are their details?",
    "contract_duration": "How long is the contract duration?",
    "scope_summary": "Can you provide a summary of the scope of work required?",
    "mandatory_requirements": "What are the mandatory requirements that suppliers must meet?",
    "payment_terms": "What are the payment terms and conditions?",
    "insurance_requirements": "What insurance requirements must suppliers meet?"
}


def get_query_template(query_type: str) -> str:
    """
    Get a predefined query template.
    
    Args:
        query_type: Type of query from COMMON_RFQ_QUERIES
        
    Returns:
        Query template string
    """
    
    return COMMON_RFQ_QUERIES.get(query_type, 
                                 "Please answer the following question about this RFQ document:")


def create_dynamic_schema(custom_features: List[str] = None, 
                         include_dynamic_extraction: bool = False) -> Dict[str, Any]:
    """
    Create a dynamic extraction schema by merging base schema with custom features.
    
    Args:
        custom_features: List of custom feature names to add
        include_dynamic_extraction: Whether to enable dynamic feature extraction
        
    Returns:
        Combined schema dictionary
    """
    # Start with base schema
    dynamic_schema = RFQ_EXTRACTION_SCHEMA.copy()
    
    # Add custom features if provided
    if custom_features:
        custom_category = {}
        for feature in custom_features:
            # Clean and format feature name
            clean_name = feature.lower().replace(' ', '_').replace('-', '_')
            # Remove special characters
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            custom_category[clean_name] = "string"
        
        if custom_category:
            dynamic_schema["manually_requested_features"] = custom_category
    
    # Add dynamic extraction category if enabled
    if include_dynamic_extraction:
        dynamic_schema["dynamically_fetched_features"] = {
            "additional_attributes": "string",
            "unique_requirements": "string",
            "special_conditions": "string",
            "other_relevant_information": "string"
        }
    
    return dynamic_schema


def create_filtered_schema_from_activation(base_schema: Dict[str, Dict[str, str]], 
                                         feature_activation: Dict[str, Dict[str, bool]]) -> Dict[str, Dict[str, str]]:
    """
    Create a filtered schema based on feature activation state.
    
    Args:
        base_schema: The complete base schema
        feature_activation: Dictionary tracking which features are activated
        
    Returns:
        Filtered schema with only activated features
    """
    filtered_schema = {}
    
    for category, fields in base_schema.items():
        active_fields = {}
        category_activation = feature_activation.get(category, {})
        
        for field_name, field_type in fields.items():
            if category_activation.get(field_name, False):
                active_fields[field_name] = field_type
        
        if active_fields:  # Only include categories with active fields
            filtered_schema[category] = active_fields
    
    return filtered_schema


def get_filtered_extraction_instructions(filtered_schema: Dict[str, Dict[str, str]]) -> str:
    """
    Generate extraction instructions for only the activated features in filtered schema.
    
    Args:
        filtered_schema: Schema containing only activated features
        
    Returns:
        Filtered extraction instructions
    """
    if not filtered_schema:
        return "No features selected for extraction."
    
    # Get base instructions
    base_instructions = get_extraction_instructions()
    
    # Create a mapping of category names to their section numbers and titles
    category_sections = {
        "project_information": ("1", "PROJECT INFORMATION"),
        "key_dates_deadlines": ("2", "KEY DATES & DEADLINES"),
        "scope_technical_requirements": ("3", "SCOPE & TECHNICAL REQUIREMENTS"),
        "supplier_requirements": ("4", "SUPPLIER REQUIREMENTS"),
        "evaluation_criteria": ("5", "EVALUATION CRITERIA"),
        "pricing_payment": ("6", "PRICING & PAYMENT"),
        "legal_contractual": ("7", "LEGAL & CONTRACTUAL"),
        "compliance_exclusion_grounds": ("8", "COMPLIANCE & EXCLUSION GROUNDS"),
        "sustainability_social_value": ("9", "SUSTAINABILITY & SOCIAL VALUE"),
        "contract_management_reporting": ("10", "CONTRACT MANAGEMENT & REPORTING"),
        "manually_requested_features": ("11", "MANUALLY REQUESTED FEATURES"),
        "dynamically_fetched_features": ("12", "DYNAMICALLY FETCHED ATTRIBUTES")
    }
    
    # Build filtered instructions
    filtered_instructions = []
    filtered_instructions.append("EXTRACTION INSTRUCTIONS:")
    filtered_instructions.append("")
    filtered_instructions.append("Extract ONLY the following selected features from the document:")
    filtered_instructions.append("")
    
    for category, fields in filtered_schema.items():
        if category in category_sections:
            section_num, section_title = category_sections[category]
            filtered_instructions.append(f"{section_num}. {section_title}")
            
            # Add field descriptions from base instructions by extracting them
            for field_name in fields.keys():
                field_title = field_name.replace('_', ' ').title()
                
                # Try to find field description in base instructions
                if category == "manually_requested_features":
                    filtered_instructions.append(f"   - {field_name}: Custom requested feature")
                elif category == "dynamically_fetched_features":
                    filtered_instructions.append(f"   - {field_name}: Dynamically extracted information")
                else:
                    # Extract description from base instructions if available
                    field_line = f"   - {field_name}:"
                    base_lines = base_instructions.split('\n')
                    for line in base_lines:
                        if line.strip().startswith(field_line):
                            filtered_instructions.append(line)
                            break
                    else:
                        # Fallback if not found in base instructions
                        filtered_instructions.append(f"   - {field_name}: {field_title}")
            
            filtered_instructions.append("")
    
    # Add extraction rules
    filtered_instructions.extend([
        "EXTRACTION RULES:",
        "- Extract information exactly as it appears in the document",
        "- Use \"Not Found\" if information is not present",
        "- For dates, preserve the original format", 
        "- For monetary values, include currency symbols",
        "- Be comprehensive - check all sections of the document",
        "- Look for information in tables, appendices, and footnotes",
        "- ONLY extract the features listed above - ignore all other information"
    ])
    
    return '\n'.join(filtered_instructions)


def get_dynamic_extraction_instructions(custom_features: List[str] = None,
                                      include_dynamic_extraction: bool = False) -> str:
    """
    Generate extraction instructions for dynamic schema including custom features.
    
    Args:
        custom_features: List of custom features added by user
        include_dynamic_extraction: Whether dynamic extraction is enabled
        
    Returns:
        Complete extraction instructions
    """
    
    base_instructions = get_extraction_instructions()
    
    additional_instructions = []
    
    # Add instructions for custom features
    if custom_features:
        additional_instructions.extend([
            "",
            "11. MANUALLY REQUESTED FEATURES",
            "The user has specifically requested the following additional information:",
            ""
        ])
        
        for feature in custom_features:
            clean_name = feature.lower().replace(' ', '_').replace('-', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            additional_instructions.append(f"   - {clean_name}: {feature}")
        
        additional_instructions.extend([
            "",
            "Look carefully through the document for any information related to these",
            "specific requirements. If information is not explicitly stated, use",
            "'Not Found' rather than making assumptions.",
            ""
        ])
    
    # Add instructions for dynamic extraction
    if include_dynamic_extraction:
        additional_instructions.extend([
            "",
            "12. DYNAMICALLY FETCHED ATTRIBUTES",
            "In addition to the predefined fields, identify and extract any other",
            "relevant information that might be important for RFQ analysis, including:",
            "",
            "   - additional_attributes: Any other requirements, conditions, or",
            "     specifications mentioned in the document that don't fit the",
            "     predefined categories",
            "   - unique_requirements: Special or unusual requirements specific",
            "     to this RFQ that bidders should be aware of",
            "   - special_conditions: Any special terms, conditions, or",
            "     circumstances that apply to this procurement",
            "   - other_relevant_information: Any other information that could",
            "     be valuable for comparison or decision-making",
            "",
            "IMPORTANT: Only include information that is explicitly stated in the",
            "document. Do not infer or assume information that is not clearly present.",
            ""
        ])
    
    # Combine all instructions
    if additional_instructions:
        complete_instructions = base_instructions + "\n".join(additional_instructions)
    else:
        complete_instructions = base_instructions
    
    return complete_instructions


def validate_schema_field_name(field_name: str) -> Tuple[bool, str]:
    """
    Validate a custom field name for schema inclusion.
    
    Args:
        field_name: The field name to validate
        
    Returns:
        Tuple of (is_valid, cleaned_name_or_error_message)
    """
    if not field_name or not field_name.strip():
        return False, "Field name cannot be empty"
    
    # Clean the field name
    cleaned = field_name.strip().lower().replace(' ', '_').replace('-', '_')
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
    
    if not cleaned:
        return False, "Field name contains no valid characters"
    
    if len(cleaned) > 50:
        return False, "Field name too long (max 50 characters)"
    
    # Check if it conflicts with existing fields
    all_existing_fields = set()
    for category_fields in RFQ_EXTRACTION_SCHEMA.values():
        all_existing_fields.update(category_fields.keys())
    
    if cleaned in all_existing_fields:
        return False, f"Field name '{cleaned}' conflicts with existing field"
    
    return True, cleaned


def get_schema_summary(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of the schema for display purposes.
    
    Args:
        schema: The schema to summarize
        
    Returns:
        Schema summary dictionary
    """
    summary = {
        "total_categories": len(schema),
        "total_fields": sum(len(fields) for fields in schema.values()),
        "categories": {},
        "has_custom_features": "manually_requested_features" in schema,
        "has_dynamic_extraction": "dynamically_fetched_features" in schema
    }
    
    for category, fields in schema.items():
        summary["categories"][category] = {
            "field_count": len(fields),
            "fields": list(fields.keys())
        }
    
    return summary


def merge_extraction_results(base_results: Dict[str, Any], 
                           dynamic_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Merge extraction results from base schema and dynamic extraction.
    
    Args:
        base_results: Results from base schema extraction
        dynamic_results: Optional results from dynamic extraction
        
    Returns:
        Merged results dictionary
    """
    merged = base_results.copy()
    
    if dynamic_results:
        # Merge dynamic results, avoiding conflicts
        for key, value in dynamic_results.items():
            if key not in merged or key.startswith('dynamically_'):
                merged[key] = value
    
    return merged