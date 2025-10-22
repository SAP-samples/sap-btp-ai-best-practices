"""
Generic Business Context - Centralized context for all LLM analyzers

This module provides a neutral business context about a purchasing organization
and its subsidiaries to ensure LLM analyzers understand organizational
structures and relationships without referencing real company names.
"""

from typing import Optional


def get_business_context() -> str:
    """
    Get the standard business context for LLM prompts

    Returns:
        Formatted context string to include in LLM prompts
    """
    return """BUSINESS CONTEXT:
This analysis is being conducted for PurchasingOrganization, a leading commercial vehicle manufacturer and member of ParentGroup.

Key PurchasingOrganization subsidiaries and brands include:
- SubsidiaryA: manufacturer of heavy trucks and buses
- SubsidiaryB: manufacturer of commercial vehicles
- SubsidiaryC: commercial vehicle manufacturer
- SubsidiaryD: commercial vehicle operations

Important context for analysis:
1. When documents reference SubsidiaryA, SubsidiaryB, SubsidiaryC, or SubsidiaryD, these are PurchasingOrganization subsidiaries
2. Cost allocations may be specified for individual subsidiaries (e.g., "this quantity should be paid by SubsidiaryA")
3. Technical requirements may vary by brand/subsidiary based on their specific product lines
4. Compliance requirements may differ by region/subsidiary (EU for SubsidiaryA/B, US for SubsidiaryC, etc.)
5. Volume commitments and pricing may be structured across multiple subsidiaries
6. Supply chain strategies should consider the global footprint of all PurchasingOrganization brands
7. Quality standards must meet the requirements of all applicable PurchasingOrganization subsidiaries

Consider this organizational structure when:
- Analyzing cost breakdowns and payment responsibilities
- Evaluating technical compliance across different vehicle platforms
- Assessing risk implications for multiple brands/regions
- Determining optimal supplier splits and volume allocations
- Understanding contractual terms that may apply to specific subsidiaries

"""


def get_analysis_guidelines() -> str:
    """
    Get specific analysis guidelines for organization-related evaluations

    Returns:
        Guidelines text to be appended to prompts when applicable
    """
    return """
ANALYSIS GUIDELINES:
- Always attribute costs/volumes to the specific subsidiary when explicitly mentioned
- If an item references multiple subsidiaries, clearly split and label the attribution
- When guidelines differ by region, use the region associated with the subsidiary
- When in doubt, prefer conservative interpretations and call out assumptions explicitly
"""


def format_prompt_with_business_context(base_prompt: str, analysis_type: str = "") -> str:
    """
    Format a prompt with business context at the top

    Args:
        base_prompt: The original LLM prompt body
        analysis_type: Optional hint to add guidelines for a specific analysis kind

    Returns:
        Formatted prompt with context at the beginning
    """
    context = get_business_context()

    if analysis_type.lower() in ["cost", "risk", "tqdcs", "comparison", "homepage", "technology", "quality", "delivery", "sustainability"]:
        context += get_analysis_guidelines()

    return f"{context}\n\n{base_prompt}" 


