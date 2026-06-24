"""
Customer Insights Service - LLM-powered analysis of customer savings

Uses GPT-5 via Generative AI Hub to provide business insights
"""

import logging
import hashlib
import time
from typing import Dict, Optional
import pandas as pd

from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService

from app.services.data_loader import is_anonymized_runtime_data, load_from_parquet
from app.observability.llm_usage_logging import emit_llm_usage_event, extract_token_counts

logger = logging.getLogger(__name__)


def generate_customer_savings_insight(customer_id: str) -> Dict:
    """
    Generate LLM-powered insights about customer's savings profile

    Args:
        customer_id: Customer ID to analyze

    Returns:
        Dict with:
        - insight: LLM-generated analysis text
        - recommendations: List of actionable recommendations
        - confidence: Confidence level (high/medium/low)
    """
    logger.info(f"Generating savings insights for customer {customer_id}")

    try:
        # Load customer data
        customer_current_metrics = load_from_parquet('customer_current_metrics.parquet')
        customer_master = load_from_parquet('customer_master.parquet')
        customer_material_current_pricing = load_from_parquet('customer_material_current_pricing.parquet')
        sap_master = load_from_parquet('sap_master_enhanced.parquet')

        # Load material descriptions for readable names
        try:
            material_descriptions = load_from_parquet('a901_materials.parquet')[['material', 'material_description']]
        except:
            material_descriptions = pd.DataFrame(columns=['material', 'material_description'])

        # Get customer record
        cust_metrics = customer_current_metrics[customer_current_metrics['customer_id'] == customer_id]
        if cust_metrics.empty:
            return {
                "insight": "No savings data available for this customer.",
                "recommendations": [],
                "confidence": "low"
            }

        cust_metrics = cust_metrics.iloc[0]

        # Get customer master data
        cust_master = customer_master[customer_master['customer_id'] == customer_id]
        customer_name = cust_master.iloc[0]['customer_name'] if not cust_master.empty else "Unknown"

        # Get current Q4 pricing rows for canonical current-state coverage and category breakdown
        cust_materials = customer_material_current_pricing[
            customer_material_current_pricing['customer_id'] == customer_id
        ].copy()
        if cust_materials.empty:
            return {
                "insight": "No current-state pricing data available for this customer.",
                "recommendations": [],
                "confidence": "low"
            }

        # Merge with material info
        cust_materials = cust_materials.merge(
            sap_master[['material', 'material_group', 'manufacturer']],
            on='material',
            how='left',
            suffixes=('', '_sap')
        )
        if 'material_group_sap' in cust_materials.columns:
            cust_materials['material_group'] = cust_materials.get('material_group').fillna(
                cust_materials['material_group_sap']
            )
            cust_materials = cust_materials.drop(columns=['material_group_sap'])
        if 'manufacturer_sap' in cust_materials.columns:
            cust_materials['manufacturer'] = cust_materials.get('manufacturer').fillna(
                cust_materials['manufacturer_sap']
            )
            cust_materials = cust_materials.drop(columns=['manufacturer_sap'])

        if 'material_group' not in cust_materials.columns:
            cust_materials['material_group'] = 'UNKNOWN'
        cust_materials['material_group'] = cust_materials['material_group'].fillna('UNKNOWN')

        # Group by material group
        category_summary = cust_materials.groupby('material_group').agg({
            'cogs_12m': 'sum',
            'material': 'count'
        }).sort_values('cogs_12m', ascending=False).head(5)

        # Build context from canonical current-state metrics
        total_cogs = float(cust_metrics.get('total_cogs_q4', 0.0) or 0.0)
        coverage_pct = float(cust_metrics.get('coverage_percent_q4', 0.0) or 0.0)
        savings_pct = float(cust_metrics.get('current_savings_pct_on_covered', 0.0) or 0.0)
        cogs_covered = float(cust_metrics.get('covered_cogs_q4', 0.0) or 0.0)
        cogs_not_covered = float(cust_metrics.get('uncovered_cogs_q4', 0.0) or 0.0)
        is_high_savings = bool(savings_pct > 60.0)

        # Format category breakdown with readable names
        category_lines = []

        # Helper function to decode material group name
        def get_category_name(group_code):
            """Decode material group codes into readable names"""
            group_code = str(group_code)
            if is_anonymized_runtime_data():
                digest = hashlib.sha256(group_code.encode("utf-8")).hexdigest()
                return f"Product Category {int(digest[:8], 16) % 1000000:06d}"

            # Wire & Cable
            if group_code.startswith('THHN'):
                return f"THHN Wire (Code: {group_code})"
            elif group_code.startswith('CARH'):
                return f"Cable/Harness (Code: {group_code})"

            # Conduit & Fittings
            elif group_code.startswith('EPVC'):
                return f"PVC Conduit (Code: {group_code})"

            # Lighting & Ballasts
            elif group_code.startswith('BLS'):
                return f"Lighting/Ballast (Code: {group_code})"
            elif group_code.startswith('BULW'):
                return f"Bulbs/Lamps (Code: {group_code})"

            # Circuit Protection & Controls
            elif group_code.startswith('SQD'):
                return f"Square D Products (Code: {group_code})"
            elif group_code.startswith('ALB') or group_code.startswith('ALCF'):
                return f"Allen-Bradley Controls (Code: {group_code})"
            elif group_code.startswith('BRA'):
                return f"Breakers (Code: {group_code})"

            # Enclosures & Boxes
            elif group_code.startswith('BSSC'):
                return f"Boxes/Steel (Code: {group_code})"

            # Tools & Hardware
            elif group_code.startswith('FASTNR'):
                return f"Fasteners (Code: {group_code})"
            elif group_code.startswith('HUTL'):
                return f"Hand Tools/Utilities (Code: {group_code})"
            elif group_code.startswith('ELTO'):
                return f"Electric Tools (Code: {group_code})"

            # Manufacturer-Specific
            elif group_code.startswith('ROI'):
                return f"ROI Products (Code: {group_code})"
            elif group_code.startswith('HUG'):
                return f"Hubbell Products (Code: {group_code})"
            elif group_code.startswith('RAB'):
                return f"RAB Lighting (Code: {group_code})"
            elif group_code.startswith('PENN'):
                return f"Penn Union (Code: {group_code})"
            elif group_code.startswith('CHN'):
                return f"Chain/Accessories (Code: {group_code})"

            # Generic Categories
            elif group_code.startswith('MISC'):
                return f"Miscellaneous (Code: {group_code})"
            elif group_code.startswith('RITT'):
                return f"Ritttal Products (Code: {group_code})"
            elif group_code.startswith('CNTRCT'):
                return f"Contract Items (Code: {group_code})"

            # Catch-all
            else:
                return f"Material Group: {group_code}"

        for cat, row in category_summary.iterrows():
            pct = (row['cogs_12m'] / total_cogs * 100) if total_cogs > 0 else 0
            readable_name = get_category_name(cat)
            category_lines.append(f"  - {readable_name}: ${row['cogs_12m']:,.0f} ({pct:.1f}%), {int(row['material'])} materials")

        category_text = "\n".join(category_lines) if category_lines else "  - No category data available"

        # Build prompt
        prompt_text = f"""Analyze this customer's SPA savings profile and provide strategic business insights:

CUSTOMER PROFILE:
- Name: {customer_name}
- ID: {customer_id}
- Annual Spend (COGS): ${total_cogs:,.0f}

SPA COVERAGE ANALYSIS:
- Coverage %: {coverage_pct:.1f}% (how much spending is covered by SPAs)
- Covered Amount: ${cogs_covered:,.0f}
- Not Covered: ${cogs_not_covered:,.0f}

SAVINGS PERFORMANCE:
- Current Savings %: {savings_pct:.1f}%{' (very high - likely normalized to 40% for projections)' if is_high_savings else ''}
- On Covered Materials: Getting {savings_pct:.1f}% discount

TOP SPENDING CATEGORIES (Last 12 Months):
{category_text}

BUSINESS CONTEXT:
- We are an electrical distributor serving contractors
- SPAs (Special Price Agreements) provide negotiated discounts on specific materials
- Coverage % shows adoption level - how much of their spending uses SPAs
- High coverage (>80%) = mature SPA program, low expansion opportunity
- Low coverage (<40%) = low adoption, high expansion opportunity
- Medium coverage (40-80%) = strategic expansion target

YOUR TASK:
Provide a concise (3-4 paragraphs) strategic analysis covering:

1. **SPA Adoption Status** - Assess their current coverage level:
   - Is it high (>80%), medium (40-80%), or low (<40%)?
   - What does this tell us about program maturity?

2. **Business Opportunity** - Based on coverage and spending:
   - If low coverage: emphasize expansion potential (${cogs_not_covered:,.0f} uncovered spending)
   - If high coverage: focus on relationship management and renewals
   - If medium: balanced approach - maintain existing + expand to uncovered categories

3. **Category-Specific Insights** - Using their top spending categories:
   - Which categories offer best expansion opportunities?
   - Are there patterns in their purchasing behavior?

4. **Action Priority** - What should the sales team do FIRST?
   - High coverage: "Maintain and renew existing SPAs"
   - Low coverage: "Expand SPA portfolio to uncovered categories"
   - Medium coverage: "Expand strategically while maintaining existing"

FORMATTING REQUIREMENTS:
- Use markdown with clear headers (##)
- Add a BLANK LINE after each header
- Keep paragraphs to 2-3 sentences maximum
- Use bullet points (- or •) for lists
- Add a BLANK LINE before and after lists
- Bold (**text**) key terms and numbers
- Be specific with dollar amounts and percentages
- Focus on actionable insights, not just data description

Write in a professional but direct business tone. The audience is sales managers who need clear action items."""

        # Build LLM request using Orchestration v1 API
        template = Template(
            messages=[
                SystemMessage(content="You are a strategic business analyst for an electrical distribution company. Provide actionable insights based on customer data."),
                UserMessage(content=prompt_text)
            ]
        )

        # Configure GPT-4.1 via SAP BTP AI Core
        llm = LLM(name="gpt-4.1")

        # Build orchestration config
        config = OrchestrationConfig(
            template=template,
            llm=llm
        )

        # Execute
        logger.info(f"Calling GPT-4.1 via SAP BTP AI Core for customer {customer_id} analysis")
        service = OrchestrationService(config=config)
        start = time.perf_counter()
        try:
            result = service.run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            input_tokens, output_tokens, total_tokens = extract_token_counts(result)
            emit_llm_usage_event(
                route="internal:customer_insights_service.generate_customer_savings_insight",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="gpt-4.1",
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
                route="internal:customer_insights_service.generate_customer_savings_insight",
                method="INTERNAL",
                actor_type="unknown",
                provider="sap-ai-core",
                model="gpt-4.1",
                llm_endpoint="OrchestrationService.run",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                outcome="error",
                latency_ms=latency_ms,
            )
            raise

        # Extract result using Orchestration v1 API structure
        insight_text = result.orchestration_result.choices[0].message.content

        if not insight_text:
            logger.warning("No insight text generated by LLM")
            insight_text = "Analysis unavailable at this time."

        logger.info(f"Generated insight for customer {customer_id}: {len(insight_text)} characters")

        # Determine confidence based on data completeness
        confidence = "high"
        if total_cogs == 0 or coverage_pct == 0:
            confidence = "low"
        elif len(category_summary) < 3:
            confidence = "medium"

        return {
            "insight": insight_text,
            "confidence": confidence,
            "data_summary": {
                "total_cogs": total_cogs,
                "coverage_percent": coverage_pct,
                "savings_percent": savings_pct,
                "top_categories": len(category_summary)
            }
        }

    except Exception as e:
        logger.error(f"Error generating insights for customer {customer_id}: {e}", exc_info=True)
        error_text = str(e)
        if "No credentials found" in error_text:
            return {
                "insight": (
                    "AI strategic analysis is temporarily unavailable in this environment "
                    "because SAP BTP AI Core credentials are not configured. "
                    "The customer savings and SPA calculations above are still valid."
                ),
                "confidence": "low",
                "error": "AI Core credentials not configured",
            }
        return {
            "insight": f"Error generating analysis: {str(e)}",
            "confidence": "low",
            "error": str(e)
        }
