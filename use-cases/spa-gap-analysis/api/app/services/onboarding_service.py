"""
Customer Onboarding Research Service

Uses Sonar-Pro to research new customer profiles and find similar existing customers.
"""

import logging
import time
from typing import Dict, List, Optional
import pandas as pd

from gen_ai_hub.orchestration_v2.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration_v2.models.template import Template, PromptTemplatingModuleConfig
from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
from gen_ai_hub.orchestration_v2.models.config import ModuleConfig, OrchestrationConfig
from gen_ai_hub.orchestration_v2.service import OrchestrationService

from app.services.data_loader import (
    load_transactions,
    load_customer_master,
    load_product_hierarchy
)
from app.observability.llm_usage_logging import emit_llm_usage_event, extract_token_counts

logger = logging.getLogger(__name__)


def research_customer_with_sonar(customer_name: str, location: Optional[str] = None) -> Dict:
    """
    Research a new customer using Sonar-Pro (Perplexity).

    Our customers are local equipment installers for schools, hospitals, data centers.

    Args:
        customer_name: Name of the customer to research
        location: Optional location (city, state)

    Returns:
        Dict with research results:
        - profile: Customer profile description
        - business_type: Type of business (school_contractor, hospital_contractor, etc.)
        - confidence: Confidence score (0-100)
        - materials_likely_needed: List of material categories
    """
    logger.info(f"Researching new customer: {customer_name}, location: {location}")

    # Build search query
    location_str = f" in {location}" if location else ""
    query = f"""Research this electrical contractor for our distribution business:

Company: {customer_name}{location_str}

BUSINESS CONTEXT:
We distribute electrical equipment to contractors who serve schools, hospitals, data centers, commercial buildings, and industrial facilities.

RESEARCH OBJECTIVES:
1. Company Profile & Services:
   - What electrical services do they provide? (installation, maintenance, design-build, etc.)
   - What types of facilities do they primarily serve? (educational, healthcare, data centers, commercial, industrial, residential)
   - Are they specialists or generalists?

2. Market Position:
   - Company size (employees, annual revenue estimates if available)
   - Geographic coverage (single location, multi-state, etc.)
   - Years in business
   - Key clients or notable projects (if public information available)

3. Material Needs Assessment:
   - Based on their services, what electrical materials do they likely purchase regularly?
   - Consider categories: Automation & Controls, Conduit & Fittings, Wire & Cable, Wiring Devices, Lighting, Switchgear
   - Do they handle specialized work requiring unique materials?

4. Alignment with Our Customer Base:
   - How well does this contractor match our target customer profile?
   - Confidence level: HIGH (perfect match), MEDIUM (partial match), or LOW (poor match)
   - Explain reasoning for confidence level

FORMATTING REQUIREMENTS:
- Use markdown headers (## for main sections, ### for subsections)
- Add a BLANK LINE after each header
- Add a BLANK LINE between each paragraph
- Add a BLANK LINE before and after bullet point lists
- Use bullet points (- or •) for lists
- DO NOT use markdown tables - convert table data into bullet point lists instead
- Keep individual paragraphs to 2-3 sentences maximum
- Use bold (**text**) for emphasis on key terms
- Structure your response with clear visual breaks for readability

EXAMPLE FORMAT:
## Section Title

First paragraph about the topic. Keep it concise.

Second paragraph with more details. Still concise.

- Bullet point one
- Bullet point two
- Bullet point three

Another paragraph after the list.

Provide detailed, structured response with citations."""

    try:
        # STEP 1: Query "sonar" model to get URLs/sources
        logger.info("Step 1: Fetching sources from 'sonar' model using Orchestrator")

        # Build sources query template
        sources_template = Template(
            template=[
                SystemMessage(content="You are a web research assistant. Find reliable sources about companies."),
                UserMessage(content=f"Find web sources about: {customer_name}{location_str}\n\nThis is an electrical contractor company. Find their official website, business profiles, and any public information.\nList the most relevant sources.")
            ]
        )

        # Configure sonar model
        sonar_llm = LLMModelDetails(
            name="sonar",
            params={
                "temperature": 0.2,
                "max_completion_tokens": 500,
            }
        )

        # Build orchestration config for sonar
        sonar_prompt_template = PromptTemplatingModuleConfig(
            prompt=sources_template,
            model=sonar_llm
        )

        sonar_module_config = ModuleConfig(
            prompt_templating=sonar_prompt_template
        )

        sonar_config = OrchestrationConfig(
            modules=sonar_module_config
        )

        # Execute sonar query
        sonar_service = OrchestrationService(config=sonar_config)
        start = time.perf_counter()
        try:
            sonar_result = sonar_service.run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(sonar_result)
            emit_llm_usage_event(
                route="internal:onboarding_service.research_customer_with_sonar.sources",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="sonar",
                llm_endpoint="OrchestrationService.run",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                outcome="success",
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            emit_llm_usage_event(
                route="internal:onboarding_service.research_customer_with_sonar.sources",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="sonar",
                llm_endpoint="OrchestrationService.run",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        # Extract sources from citations attribute
        sources = []
        if hasattr(sonar_result, 'final_result'):
            if hasattr(sonar_result.final_result, 'citations'):
                citations = sonar_result.final_result.citations
                if citations:
                    # Convert Citation objects to dictionaries
                    for citation in citations:
                        if hasattr(citation, 'url') and hasattr(citation, 'title'):
                            sources.append({
                                'url': citation.url,
                                'title': citation.title,
                                'ref_id': getattr(citation, 'ref_id', None)
                            })
        logger.info(f"Found {len(sources)} sources from 'sonar' model")

        # STEP 2: Query "sonar-pro" for detailed profile
        logger.info("Step 2: Fetching detailed profile from 'sonar-pro' model using Orchestrator")

        # Build detailed profile query template
        profile_template = Template(
            template=[
                SystemMessage(content="You are an expert business researcher for an electrical distribution company."),
                UserMessage(content=query)
            ]
        )

        # Configure sonar-pro model
        sonar_pro_llm = LLMModelDetails(
            name="sonar-pro",
            params={
                "temperature": 0.2,
                "max_completion_tokens": 1500,
            }
        )

        # Build orchestration config for sonar-pro
        sonar_pro_prompt_template = PromptTemplatingModuleConfig(
            prompt=profile_template,
            model=sonar_pro_llm
        )

        sonar_pro_module_config = ModuleConfig(
            prompt_templating=sonar_pro_prompt_template
        )

        sonar_pro_config = OrchestrationConfig(
            modules=sonar_pro_module_config
        )

        # Execute sonar-pro query
        sonar_pro_service = OrchestrationService(config=sonar_pro_config)
        start = time.perf_counter()
        try:
            sonar_pro_result = sonar_pro_service.run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(sonar_pro_result)
            emit_llm_usage_event(
                route="internal:onboarding_service.research_customer_with_sonar.profile",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                outcome="success",
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            emit_llm_usage_event(
                route="internal:onboarding_service.research_customer_with_sonar.profile",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="sonar-pro",
                llm_endpoint="OrchestrationService.run",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        # Extract profile text
        profile_text = sonar_pro_result.final_result.choices[0].message.content

        # Remove citation markers like [1], [2], [1][2], etc.
        import re
        profile_text = re.sub(r'\[\d+\]', '', profile_text)
        # Clean up multiple spaces on same line (but preserve newlines!)
        # Use [ \t]+ to match only spaces and tabs, NOT newlines
        profile_text = re.sub(r'[ \t]+', ' ', profile_text)
        # Clean up space before punctuation
        profile_text = re.sub(r' +([.,;:])', r'\1', profile_text)

        logger.info("Removed citation markers from profile text")

        # Simple heuristic parsing (in production, could use structured output)
        confidence = _extract_confidence(profile_text)
        business_type = _extract_business_type(profile_text)
        materials_needed = _extract_material_categories(profile_text)

        return {
            'profile': profile_text,
            'business_type': business_type,
            'confidence': confidence,
            'materials_likely_needed': materials_needed,
            'sources': sources,  # Sources from "sonar" model
            'success': True
        }

    except Exception as e:
        logger.error(f"Sonar research failed: {str(e)}", exc_info=True)
        return {
            'profile': f"Research failed: {str(e)}",
            'business_type': 'unknown',
            'confidence': 0,
            'materials_likely_needed': [],
            'success': False
        }


def _extract_confidence(text: str) -> int:
    """Extract confidence level from Sonar response"""
    text_lower = text.lower()
    if 'high confidence' in text_lower or 'very confident' in text_lower:
        return 85
    elif 'medium confidence' in text_lower or 'moderate' in text_lower:
        return 60
    elif 'low confidence' in text_lower:
        return 30
    else:
        return 50  # Default


def _extract_business_type(text: str) -> str:
    """Extract business type from Sonar response"""
    text_lower = text.lower()

    if 'school' in text_lower or 'education' in text_lower:
        return 'school_contractor'
    elif 'hospital' in text_lower or 'healthcare' in text_lower or 'medical' in text_lower:
        return 'hospital_contractor'
    elif 'data center' in text_lower or 'datacenter' in text_lower:
        return 'datacenter_contractor'
    elif 'commercial' in text_lower:
        return 'commercial_contractor'
    elif 'industrial' in text_lower:
        return 'industrial_contractor'
    else:
        return 'general_contractor'


def _extract_material_categories(text: str) -> List[str]:
    """Extract likely material categories from Sonar response"""
    categories = []
    text_lower = text.lower()

    # Check for common categories
    if 'automation' in text_lower or 'control' in text_lower:
        categories.append('1A')  # Automation and Controls
    if 'conduit' in text_lower or 'fitting' in text_lower:
        categories.append('1G')  # Conduit Fittings and Bodies
    if 'wiring' in text_lower or 'device' in text_lower:
        categories.append('1K')  # Wiring Devices
    if 'lighting' in text_lower or 'light' in text_lower:
        categories.append('1L')  # Lighting
    if 'wire' in text_lower or 'cable' in text_lower:
        categories.append('1N')  # Wire and Cable

    return categories if categories else ['1A', '1G']  # Default fallback


def find_similar_customers_by_profile(
    business_type: str,
    material_categories: List[str],
    top_n: int = 5
) -> List[Dict]:
    """
    Find existing customers similar to the new customer profile.

    Args:
        business_type: Type of business (school_contractor, etc.)
        material_categories: List of material category codes
        top_n: Number of similar customers to return

    Returns:
        List of similar customer dicts
    """
    logger.info(f"Finding similar customers for profile: {business_type}, categories: {material_categories}")

    try:
        # Load data
        transactions = load_transactions()
        customer_master = load_customer_master()
        hierarchy = load_product_hierarchy()

        if len(material_categories) == 0:
            logger.warning("No material categories specified, using default")
            material_categories = ['1A', '1G']

        # Join transactions to hierarchy to get Level 1 categories
        transactions['material_int'] = pd.to_numeric(transactions['material'], errors='coerce')
        transactions = transactions.dropna(subset=['material_int'])

        # Get SAP master for hierarchy mapping
        from app.services.data_loader import load_sap_master
        sap_master = load_sap_master()

        transactions = transactions.merge(
            sap_master[['material', 'product_hierarchy']],
            left_on='material_int',
            right_on='material',
            how='left',
            suffixes=('', '_sap')
        )

        transactions = transactions.merge(
            hierarchy[['product_hierarchy', 'level_1']],
            on='product_hierarchy',
            how='left'
        )

        # Calculate similarity: customers who buy from the same categories
        customer_categories = transactions.groupby('customer_id')['level_1'].apply(set).to_dict()
        target_categories = set(material_categories)

        # Calculate overlap scores
        similarity_scores = []
        for cust_id, cust_cats in customer_categories.items():
            if not cust_cats or len(cust_cats) == 0:
                continue

            overlap = len(target_categories & cust_cats)
            total = len(target_categories | cust_cats)

            if total > 0:
                similarity = (overlap / total) * 100
                if similarity > 20:  # Minimum threshold
                    similarity_scores.append({
                        'customer_id': str(cust_id),
                        'similarity_score': round(similarity, 1),
                        'matching_categories': list(target_categories & cust_cats),
                        'category_count': len(cust_cats)
                    })

        # Sort by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x['similarity_score'], reverse=True)[:top_n]

        # Enrich with customer info
        for item in similarity_scores:
            # Convert customer_id to string for comparison
            cust_id_str = str(item['customer_id'])
            # Try both string and numeric comparison
            cust_row = customer_master[
                (customer_master['customer_id'].astype(str) == cust_id_str)
            ]
            if len(cust_row) > 0:
                item['customer_name'] = str(cust_row['customer_name'].iloc[0])
                item['sales_office'] = str(cust_row['sales_office'].iloc[0]) if 'sales_office' in cust_row.columns else 'N/A'
                item['rfm_segment'] = str(cust_row['rfm_segment'].iloc[0]) if 'rfm_segment' in cust_row.columns else 'N/A'
            else:
                # Fallback: try to find by index if customer_id is numeric
                logger.warning(f"Customer {cust_id_str} not found in customer_master, using ID as name")
                item['customer_name'] = f"Customer {cust_id_str}"
                item['sales_office'] = 'N/A'
                item['rfm_segment'] = 'N/A'

        return similarity_scores

    except Exception as e:
        logger.error(f"Failed to find similar customers: {str(e)}", exc_info=True)
        return []
