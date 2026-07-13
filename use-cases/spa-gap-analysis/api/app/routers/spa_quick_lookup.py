"""
Quick Lookup API Router

Fast SPA gap analysis for a single customer
Tab 1 in UX: Customer ID input → Instant results
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import math

from app.models import (
    QuickLookupRequest,
    QuickLookupResponse,
    CustomerProfileSummary,
    SimilarCustomer,
    MissingSPA,
    SavingsAnalysis,
    ErrorResponse
)
from app.services import (
    get_customer_profile,
    find_similar_customers,
    detect_spa_gaps,
    calculate_confidence_score,
    load_from_parquet
)
from app.services.customer_insights_service import generate_customer_savings_insight
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spa", tags=["SPA Analysis"])


def clean_value(value):
    """Convert NaN/None to None, return value otherwise"""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def clean_bool(value):
    """Convert pandas/numpy boolean-ish values to JSON-safe bool/None."""
    value = clean_value(value)
    if value is None:
        return None
    return bool(value)


def clean_int(value):
    """Convert numeric values to int, preserving None for missing values."""
    value = clean_value(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_date(value):
    """Return an ISO-ish string for pandas/date values."""
    value = clean_value(value)
    if value is None:
        return None
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def normalize_identifier(value):
    """Normalize numeric and anonymized IDs for safe comparisons."""
    value = clean_value(value)
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def identifier_mask(series, value):
    """Compare parquet identifiers without assuming they are numeric."""
    normalized = normalize_identifier(value)
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True) == normalized


@router.get(
    "/health",
    summary="Health Check",
    description="Check if SPA API is healthy"
)
async def health_check():
    """Health check endpoint"""
    from app.services import load_customer_master

    try:
        # Test data loading
        customer_master = load_customer_master()
        customer_count = len(customer_master)

        return {
            "status": "healthy",
            "service": "SPA Gap Analysis API",
            "data_loaded": True,
            "customer_count": customer_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "SPA Gap Analysis API",
            "error": str(e)
        }


@router.post(
    "/quick-lookup",
    response_model=QuickLookupResponse,
    dependencies=[Depends(require_api_key)],
    summary="Quick SPA Gap Analysis",
    description="Analyze a customer and find missing SPAs based on similar customers"
)
async def quick_lookup(request: QuickLookupRequest) -> QuickLookupResponse:
    """
    Quick Lookup: Find missing SPAs for a customer

    Process:
    1. Get customer profile
    2. Find similar customers
    3. Detect SPA gaps
    4. Calculate confidence scores
    5. Return recommendations

    Returns comprehensive analysis in <200ms
    """
    logger.info(f"Quick lookup for customer: {request.customer_id}")

    try:
        # STEP 1: Get customer profile
        profile = get_customer_profile(request.customer_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Customer {request.customer_id} not found"
            )

        # Load savings data for customer
        savings_analysis = None
        try:
            customer_metrics = load_from_parquet('customer_current_metrics.parquet')
            savings_row = customer_metrics[customer_metrics['customer_id'] == request.customer_id]

            if not savings_row.empty:
                savings_data = savings_row.iloc[0]
                pricing_summary = profile.get('current_pricing_summary', {}) if profile else {}
                total_base_cost = float(savings_data.get('covered_base_estimated_spend_q4', 0) or 0)
                total_savings = float(savings_data.get('current_savings_value_q4', 0) or 0)
                total_spa_cost = float(savings_data.get('covered_cogs_q4', 0) or 0)
                covered_cogs = float(savings_data.get('covered_cogs_q4', 0) or 0)
                uncovered_cogs = float(savings_data.get('uncovered_cogs_q4', 0) or 0)
                coverage_percent = float(savings_data.get('coverage_percent_q4', 0) or 0)
                covered_material_count = pricing_summary.get('covered_material_count')
                total_material_count = pricing_summary.get('total_material_count')
                savings_analysis = SavingsAnalysis(
                    total_savings=total_savings,
                    total_base_cost=total_base_cost,
                    total_spa_cost=total_spa_cost,
                    material_count=int(covered_material_count) if covered_material_count is not None else None,
                    total_material_count=int(total_material_count) if total_material_count is not None else None,
                    covered_cogs=covered_cogs,
                    uncovered_cogs=uncovered_cogs,
                    coverage_percent=coverage_percent,
                    savings_percent=float(savings_data.get('current_savings_pct_on_covered', 0) or 0),
                    current_pricing_sources=pricing_summary.get('pricing_source_counts', {}),
                    pricing_source_note=pricing_summary.get('pricing_source_note')
                )
        except Exception as e:
            logger.warning(f"Could not load savings data for customer {request.customer_id}: {e}")

        # STEP 2: Find similar customers
        similar = find_similar_customers(
            request.customer_id,
            top_n=request.top_n_similar,
            include_rfm=request.include_rfm,
            include_price_group=request.include_price_group
        )

        if not similar:
            # No similar customers found
            return QuickLookupResponse(
                target_customer=CustomerProfileSummary(
                    customer_id=profile['customer_id'],
                    customer_name=clean_value(profile.get('customer_name')),
                    sales_office=clean_value(profile.get('sales_office')),
                    pl_type=clean_value(profile.get('pl_type')),
                    price_group=clean_value(profile.get('price_group')),
                    city=clean_value(profile.get('city')),
                    state=clean_value(profile.get('state')),
                    spa_count=profile.get('spa_count', 0),
                    current_spas=[str(spa) for spa in profile.get('spas', [])],
                    current_spa_details=profile.get('current_spa_details', []),
                    current_spa_count_unique=profile.get('current_spa_count_unique'),
                    current_spa_row_count=profile.get('current_spa_row_count'),
                    current_spa_count_rule=profile.get('current_spa_count_rule'),
                    snapshot_date=profile.get('snapshot_date'),
                    rfm_segment=(
                        profile.get('rfm', {}).get('segment')
                        if request.include_rfm and profile.get('rfm')
                        else None
                    ),
                    total_cogs=profile.get('spending', {}).get('total_cogs') if profile.get('spending') else None,
                    savings=savings_analysis
                ),
                similar_customers=[],
                missing_spas=[],
                summary={
                    'message': 'No similar customers found',
                    'reason': 'No customers with matching SOff and PLType'
                }
            )

        # STEP 3: Get missing SPAs from cache (use same data as Summary View)
        # This ensures consistency between Summary View and Quick Lookup
        try:
            cache = load_from_parquet('customer_summary_cache.parquet')
            cache_row = cache[cache['customer_id'] == request.customer_id]

            if not cache_row.empty:
                cache_data = cache_row.iloc[0]
                logger.info(f"Found customer {request.customer_id} in cache")

                # Load header data for SPA details
                header_data = load_from_parquet('spa_header_enhanced.parquet')

                # Get missing SPAs details from cache (JSON with full details)
                import json
                missing_spas_details_str = cache_data.get('missing_spas_details', '')

                # Parse SPA details from JSON
                spa_details_map = {}
                if missing_spas_details_str:
                    try:
                        spa_details_list = json.loads(missing_spas_details_str)
                        spa_details_map = {spa['sales_deal']: spa for spa in spa_details_list}
                        logger.info(f"Loaded details for {len(spa_details_map)} SPAs from cache")
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse missing_spas_details JSON: {parse_error}")
                        spa_details_map = {}

                # Fallback to old format if new format not available
                if not spa_details_map:
                    missing_spa_ids_str = cache_data.get('missing_spas_list', '')
                    if isinstance(missing_spa_ids_str, str) and missing_spa_ids_str:
                        missing_spa_ids = [int(x.strip()) for x in missing_spa_ids_str.split(',') if x.strip().isdigit()]
                        spa_details_map = {spa_id: {'sales_deal': spa_id} for spa_id in missing_spa_ids}
                        logger.info(f"Using fallback: parsed {len(spa_details_map)} SPA IDs from cache")

                # Build missing SPA list with details
                missing_spas_with_confidence = []

                for spa_id, spa_details in spa_details_map.items():
                    # Get SPA details from header
                    spa_info = header_data[header_data['agreement_id'] == str(spa_id)]

                    if not spa_info.empty:
                        spa_info = spa_info.iloc[0]

                        # Get similarity metrics from cached details
                        count_in_similar = spa_details.get('count_in_similar', 0)
                        percentage_in_similar = spa_details.get('percentage_in_similar', 0.0)

                        # Calculate confidence from percentage
                        if percentage_in_similar >= 50:
                            confidence_score = 85
                            confidence_level = 'High'
                        elif percentage_in_similar >= 25:
                            confidence_score = 70
                            confidence_level = 'Medium'
                        else:
                            confidence_score = 55
                            confidence_level = 'Low'

                        description_of_agreement = clean_value(spa_info.get('description_of_agreement'))
                        external_description = clean_value(spa_info.get('external_description'))
                        agreement_description = (
                            clean_value(spa_info.get('agreement_description'))
                            or description_of_agreement
                            or external_description
                        )
                        is_supplyforce = clean_bool(
                            spa_details.get('is_supplyforce')
                            if spa_details.get('is_supplyforce') is not None
                            else spa_info.get('is_supplyforce')
                        )

                        missing_spas_with_confidence.append(
                            MissingSPA(
                                sales_deal=str(spa_id),
                                vendor=spa_details.get('candidate_vendor_name') or spa_details.get('candidate_vendor_id'),
                                description=agreement_description,
                                description_of_agreement=description_of_agreement,
                                external_description=external_description,
                                agreement_description=agreement_description,
                                agreement_type=clean_value(spa_info.get('agreement_type')),
                                grouping=spa_details.get('agreement_grouping') or clean_value(spa_info.get('agreement_grouping')),
                                valid_from=clean_date(spa_info.get('valid_from')),
                                valid_to=clean_date(spa_info.get('valid_to')),
                                is_supplyforce=is_supplyforce,
                                customer_count=clean_int(spa_details.get('customer_count')),
                                a700_vendor_scope=(
                                    clean_bool(spa_details.get('is_vendor_in_a700_scope'))
                                    if spa_details.get('is_vendor_in_a700_scope') is not None
                                    else (False if spa_details.get('is_out_of_scope') else True)
                                ),
                                opportunity_scope=clean_value(spa_details.get('opportunity_scope')),
                                is_out_of_scope=clean_bool(spa_details.get('is_out_of_scope')),
                                count_in_similar=count_in_similar,
                                percentage_in_similar=percentage_in_similar,
                                confidence_score=confidence_score,
                                confidence_level=confidence_level,
                                eligibility_status=spa_details.get('eligibility_status'),
                                eligibility_label=spa_details.get('eligibility_label'),
                                eligibility_reason=spa_details.get('eligibility_reason'),
                                geo_relevance=spa_details.get('geo_relevance'),
                                is_addable_candidate=spa_details.get('is_addable_candidate'),
                                is_reference_only=spa_details.get('is_reference_only'),
                                is_out_of_area=spa_details.get('is_out_of_area'),
                                customer_area=spa_details.get('customer_area'),
                                candidate_areas=spa_details.get('candidate_areas') or [],
                                candidate_sales_offices=spa_details.get('candidate_sales_offices') or [],
                                candidate_plants=spa_details.get('candidate_plants') or [],
                                candidate_vendor_id=spa_details.get('candidate_vendor_id'),
                                candidate_vendor_name=spa_details.get('candidate_vendor_name'),
                                candidate_spa_type=spa_details.get('candidate_spa_type'),
                                candidate_primary_category=spa_details.get('candidate_primary_category')
                            )
                        )
                    else:
                        logger.warning(f"SPA {spa_id} not found in spa_header_enhanced")

                logger.info(f"Loaded {len(missing_spas_with_confidence)} missing SPAs from cache for customer {request.customer_id}")
            else:
                # Customer not in cache - no missing SPAs
                logger.warning(f"Customer {request.customer_id} not in cache")
                missing_spas_with_confidence = []

        except Exception as e:
            logger.error(f"Error loading missing SPAs from cache: {e}", exc_info=True)
            # Fallback to empty list
            missing_spas_with_confidence = []

        # STEP 4: Format response
        response = QuickLookupResponse(
            target_customer=CustomerProfileSummary(
                customer_id=profile['customer_id'],
                customer_name=clean_value(profile.get('customer_name')),
                sales_office=clean_value(profile.get('sales_office')),
                pl_type=clean_value(profile.get('pl_type')),
                price_group=clean_value(profile.get('price_group')),
                city=clean_value(profile.get('city')),
                state=clean_value(profile.get('state')),
                spa_count=profile.get('spa_count', 0),
                current_spas=[str(spa) for spa in profile.get('spas', [])],
                current_spa_details=profile.get('current_spa_details', []),
                current_spa_count_unique=profile.get('current_spa_count_unique'),
                current_spa_row_count=profile.get('current_spa_row_count'),
                current_spa_count_rule=profile.get('current_spa_count_rule'),
                snapshot_date=profile.get('snapshot_date'),
                rfm_segment=(
                    profile.get('rfm', {}).get('segment')
                    if request.include_rfm and profile.get('rfm')
                    else None
                ),
                total_cogs=profile.get('spending', {}).get('total_cogs') if profile.get('spending') else None,
                savings=savings_analysis
            ),
            similar_customers=[
                SimilarCustomer(
                    customer_id=sim['customer_id'],
                    customer_name=clean_value(sim.get('customer_name')),
                    sales_office=clean_value(sim.get('sales_office')),
                    city=clean_value(sim.get('city')),
                    state=clean_value(sim.get('state')),
                    similarity_score=sim['similarity_score'],
                    cogs_similarity=sim['cogs_similarity'],
                    rfm_boost=sim.get('rfm_boost', 0),
                    pg_boost=sim.get('pg_boost', 0),
                    rfm_segment=clean_value(sim.get('rfm_segment'))
                )
                for sim in similar
            ],
            missing_spas=missing_spas_with_confidence,
            summary={
                'target_spa_count': profile.get('spa_count', 0),
                'missing_spa_count': len(missing_spas_with_confidence),
                'similar_customers_analyzed': len(similar),
                'high_confidence_recommendations': sum(
                    1 for spa in missing_spas_with_confidence
                    if spa.confidence_level == 'High'
                )
            }
        )

        logger.info(f"Quick lookup complete for {request.customer_id}: {len(missing_spas_with_confidence)} missing SPAs")

        logger.info(
            f"Quick lookup complete: {len(missing_spas_with_confidence)} "
            f"missing SPAs found for {request.customer_id}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick lookup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/customer/{customer_id}",
    dependencies=[Depends(require_api_key)],
    summary="Get Customer Profile",
    description="Get complete customer profile with SPAs, RFM, and spending"
)
async def get_customer(customer_id: str):
    """Get customer profile"""
    logger.info(f"Getting customer profile: {customer_id}")

    try:
        profile = get_customer_profile(customer_id)

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Customer {customer_id} not found"
            )

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{spa_id}",
    dependencies=[Depends(require_api_key)],
    summary="Get SPA Details",
    description="Get details about a specific SPA including customers and validity dates"
)
async def get_spa_details(spa_id: str):
    """Get SPA details"""
    logger.info(f"Getting SPA details: {spa_id}")

    try:
        from app.services import load_qualifications
        import pandas as pd

        qualifications = load_qualifications()

        # Convert spa_id to int for comparison
        try:
            spa_id_int = int(spa_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SPA ID format: {spa_id}"
            )

        # Get all records for this SPA
        spa_records = qualifications[qualifications['sales_deal'] == spa_id_int]

        if spa_records.empty:
            raise HTTPException(
                status_code=404,
                detail=f"SPA {spa_id} not found"
            )

        # Get first record for dates
        first_record = spa_records.iloc[0]

        # Count unique customers (excluding NaN) - convert to int for JSON
        customer_count = int(spa_records['sold_to'].dropna().nunique())

        # Get customer list (convert numpy types to native Python for JSON serialization)
        customers = spa_records['sold_to'].dropna().unique().tolist()
        # Convert to int: handle both int and float types (e.g., '172.0')
        customers = [str(int(float(c))) for c in customers]

        # Get SPA details from HEADER data for vendor and grouping
        from app.services import load_header_data
        header_data = load_header_data()
        spa_header = header_data[header_data['sales_deal'] == spa_id_int]

        vendor = None
        grouping = None
        description = None

        if not spa_header.empty:
            spa_info = spa_header.iloc[0]
            # Convert numpy types to Python native types for JSON serialization
            vendor = int(spa_info['vendor']) if pd.notna(spa_info.get('vendor')) else None
            grouping = str(spa_info['grouping']) if pd.notna(spa_info.get('grouping')) else None
            description = str(spa_info['description']) if pd.notna(spa_info.get('description')) else None

        return {
            "spa_id": str(spa_id),  # Ensure string type
            "vendor": vendor,
            "grouping": grouping,
            "description": description,
            "valid_from": str(first_record.get('valid_from')) if pd.notna(first_record.get('valid_from')) else None,
            "valid_to": str(first_record.get('valid_to')) if pd.notna(first_record.get('valid_to')) else None,
            "expansion_type": str(first_record.get('expansion_type', 'direct')),
            "customer_count": int(customer_count),  # Ensure int type
            "customers": customers[:10]  # Limit to first 10 for display
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SPA details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{spa_id}/recommendation-breakdown",
    dependencies=[Depends(require_api_key)],
    summary="Get Recommendation Breakdown",
    description="Get detailed explanation of why this SPA is recommended for a customer"
)
async def get_recommendation_breakdown(spa_id: str, customer_id: str):
    """
    Get detailed breakdown of recommendation factors

    Returns:
    - Similarity factors (Sales Office, Customer Type matches)
    - SPA type (Blanket vs Customer Specific)
    - Material coverage (MOCK DATA with A703)
    """
    logger.info(f"Getting recommendation breakdown: SPA {spa_id} for customer {customer_id}")

    try:
        from app.services import (
            load_customer_master,
            load_qualifications,
            load_header_data,
            load_a703_nets,
            find_similar_customers
        )
        import pandas as pd

        # Get target customer profile
        customer_master = load_customer_master()
        target_customer = customer_master[customer_master['customer_id'] == customer_id]

        if target_customer.empty:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

        target_customer = target_customer.iloc[0]

        # Get similar customers
        similar_customers = find_similar_customers(customer_id, top_n=50)

        if not similar_customers:
            raise HTTPException(status_code=400, detail="No similar customers found")

        # Count how many similar customers have this SPA
        qualifications = load_qualifications()
        spa_key = normalize_identifier(spa_id)

        customers_with_spa = 0
        same_sales_office_count = 0
        same_pl_type_count = 0

        for sim_cust in similar_customers:
            sim_id = sim_cust['customer_id']

            # Convert to sold_to format (with .0)
            sim_sold_to = f"{sim_id}.0"

            # Check if this similar customer has the SPA
            has_spa = not qualifications[
                (qualifications['sold_to'] == sim_sold_to) &
                identifier_mask(qualifications['sales_deal'], spa_key)
            ].empty

            if has_spa:
                customers_with_spa += 1

                # Count matching factors
                sim_profile = customer_master[customer_master['customer_id'] == sim_id]
                if not sim_profile.empty:
                    sim_profile = sim_profile.iloc[0]

                    if sim_profile.get('sales_office') == target_customer.get('sales_office'):
                        same_sales_office_count += 1

                    if sim_profile.get('pl_type') == target_customer.get('pl_type'):
                        same_pl_type_count += 1

        # Get SPA info from HEADER
        header_data = load_header_data()
        spa_info = header_data[identifier_mask(header_data['sales_deal'], spa_key)]

        spa_type = "Unknown"
        spa_description = f"SPA {spa_id}"

        if not spa_info.empty:
            spa_info = spa_info.iloc[0]
            grouping = spa_info.get('grouping', '')
            spa_description = spa_info.get('description', spa_description)

            if 'Blanket' in grouping:
                spa_type = "Blanket SPA"
            elif 'Customer Specific' in grouping:
                spa_type = "Customer Specific"

        # Material Coverage - Mock Data from A703
        a703 = load_a703_nets()
        spa_materials = a703[identifier_mask(a703['agreement'], spa_key)]

        sample_materials = []
        for _, row in spa_materials.head(10).iterrows():
            # Convert all numpy types to native Python types for JSON serialization
            material_id = str(row['material']) if pd.notna(row.get('material')) else None
            description = str(row.get('material_description', 'N/A')) if pd.notna(row.get('material_description')) else 'N/A'
            unit_price = float(row['amount']) if pd.notna(row.get('amount')) else None
            uom = str(row.get('uom')) if pd.notna(row.get('uom')) else None

            sample_materials.append({
                "material_id": material_id,
                "description": description,
                "unit_price": unit_price,
                "uom": uom
            })

        # Get potential and materials count from cache
        import json
        potential_value = 0.0
        materials_count_for_customer = 0
        cogs_covered = 0.0

        try:
            cache = load_from_parquet('customer_summary_cache.parquet')
            cache_row = cache[cache['customer_id'] == customer_id]

            if not cache_row.empty:
                cache_data = cache_row.iloc[0]
                details_str = cache_data.get('missing_spas_details', '')

                if details_str:
                    spa_details_list = json.loads(details_str)
                    # Find this specific SPA
                    for spa_detail in spa_details_list:
                        if normalize_identifier(spa_detail.get('sales_deal')) == spa_key:
                            potential_value = float(spa_detail.get('potential', 0))
                            materials_count_for_customer = int(spa_detail.get('materials_count', 0))
                            cogs_covered = float(spa_detail.get('cogs_covered', 0))
                            break
        except Exception as e:
            logger.warning(f"Could not load potential from cache: {e}")

        # Build response - ensure all values are JSON-serializable native Python types
        breakdown = {
            "spa_id": str(spa_id),
            "spa_description": str(spa_description),
            "spa_type": str(spa_type),
            "customer_id": str(customer_id),
            "customer_name": str(target_customer.get('customer_name')) if pd.notna(target_customer.get('customer_name')) else None,
            "customer_location": f"{target_customer.get('city')}, {target_customer.get('state')}",
            "customer_sales_office": str(target_customer.get('sales_office')) if pd.notna(target_customer.get('sales_office')) else None,
            "customer_pl_type": str(target_customer.get('pl_type')) if pd.notna(target_customer.get('pl_type')) else None,
            "potential_value": round(potential_value, 2),  # NEW: Potential savings for this customer
            "materials_count_for_customer": materials_count_for_customer,  # NEW: Materials matched
            "cogs_covered": round(cogs_covered, 2),  # NEW: COGS covered
            "similarity_factors": {
                "total_similar_customers": int(len(similar_customers)),
                "customers_with_spa": int(customers_with_spa),
                "percentage": round(float(customers_with_spa) / float(len(similar_customers)) * 100, 1) if similar_customers else 0.0,
                "same_sales_office": int(same_sales_office_count),
                "same_customer_type": int(same_pl_type_count)
            },
            "material_coverage": {
                "status": "demo_data",
                "total_materials_in_spa": int(len(spa_materials)),
                "sample_materials": sample_materials,
                "disclaimer": "⚠️ Material Coverage: Demo Data\n→ Showing materials covered by this SPA (from A703)\n→ Customer-specific coverage calculation pending material mapping\n→ Material IDs: A703 format (not yet linked to customer purchases)"
            }
        }

        return breakdown

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation breakdown: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{spa_id}/materials",
    dependencies=[Depends(require_api_key)],
    summary="Get SPA Materials",
    description="Get list of materials covered by this SPA (from A703-nets)"
)
async def get_spa_materials(spa_id: str, limit: int = 100):
    """
    Get materials covered by SPA

    Data Source: A703-nets table
    Note: These are SPA catalog materials, NOT customer-specific overlap
    """
    logger.info(f"Getting materials for SPA {spa_id}")

    try:
        from app.services import load_a703_nets
        import pandas as pd

        a703 = load_a703_nets()
        spa_key = normalize_identifier(spa_id)

        spa_materials = a703[identifier_mask(a703['agreement'], spa_key)]

        if spa_materials.empty:
            raise HTTPException(status_code=404, detail=f"No materials found for SPA {spa_id}")

        materials = []
        for _, row in spa_materials.head(limit).iterrows():
            materials.append({
                "material_id": str(row['material']),
                "description": row.get('material_description', 'N/A'),
                "unit_price": float(row['amount']) if pd.notna(row.get('amount')) else None,
                "uom": row.get('uom'),
                "valid_from": str(row['valid_from']) if pd.notna(row.get('valid_from')) else None,
                "valid_to": str(row['valid_to']) if pd.notna(row.get('valid_to')) else None
            })

        return {
            "spa_id": spa_id,
            "materials": materials,
            "total_count": len(spa_materials),
            "showing_count": len(materials),
            "data_source": "A703-nets (SPA supplier catalog)",
            "disclaimer": "These are materials covered by the SPA. Customer-specific purchase overlap requires material mapping table."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SPA materials: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/customer-insights",
    dependencies=[Depends(require_api_key)],
    summary="Generate AI Insights for Customer",
    description="Use GPT-5.2 to analyze customer savings profile and provide strategic recommendations"
)
async def get_customer_insights(customer_id: str):
    """
    Generate LLM-powered insights about customer's savings profile

    Args:
        customer_id: Customer ID to analyze

    Returns:
        {
            "insight": "LLM-generated analysis text (markdown)",
            "confidence": "high/medium/low",
            "data_summary": {...}
        }
    """
    logger.info(f"Generating insights for customer {customer_id}")

    try:
        result = generate_customer_savings_insight(customer_id)
        return result
    except Exception as e:
        logger.error(f"Error generating insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


# Force reload
