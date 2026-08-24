"""
so_validator.py
---------------
Validates an extracted Purchase Order against S/4HANA master data.

Steps:
  1. Match customer name to a SAP Business Partner.
  2. Match each line item material to a SAP material code.
  3. Return a SOValidationResult indicating whether the PO is ready to create.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request

from matching.customer_api_matcher import search_customer_odata
from matching.product_api_matcher import search_material_odata
from SalesOrderProcess.so_models import ExtractedPurchaseOrder, SOValidationResult

logger = logging.getLogger(__name__)

# Score thresholds
_CUSTOMER_SCORE_THRESHOLD = 0.6
_MATERIAL_SCORE_THRESHOLD = 0.6


def validate_purchase_order(
    po: ExtractedPurchaseOrder,
    request: Optional[Request] = None,
) -> SOValidationResult:
    """
    Validate an extracted Purchase Order against S/4HANA master data.

    Args:
        po: The extracted purchase order from SAP Document AI.
        request: FastAPI request (carries X-SAP-* credential headers).

    Returns:
        SOValidationResult with resolved BPs, materials and readiness flag.
    """
    issues: list[str] = []
    items_validation: list[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Match customer name → S/4HANA Business Partner (fuzzy search)
    # ------------------------------------------------------------------
    customer_resolved = False
    customer_bp = ""
    customer_name_matched = ""
    customer_score = 0.0

    if po.customer_name:
        logger.info("Searching BP for customer_name=%r", po.customer_name)
        try:
            results = search_customer_odata(po.customer_name, top=5, request=request)
            if results:
                best = results[0]
                customer_score = best.get("score", 0.0)
                if customer_score >= _CUSTOMER_SCORE_THRESHOLD:
                    customer_resolved = True
                    customer_bp = best.get("business_partner", "")
                    customer_name_matched = best.get("customer_name", "")
                    logger.info(
                        "Customer matched | bp=%s | name=%r | score=%.2f",
                        customer_bp, customer_name_matched, customer_score,
                    )
                else:
                    msg = (
                        f"Customer not found: {po.customer_name!r} "
                        f"(best match: {best.get('customer_name')!r}, score={customer_score:.2f})"
                    )
                    issues.append(msg)
                    logger.warning(msg)
            else:
                msg = f"Customer not found: {po.customer_name!r} (no results returned)"
                issues.append(msg)
                logger.warning(msg)
        except Exception as exc:
            msg = f"Customer search failed: {exc}"
            issues.append(msg)
            logger.error(msg)
    else:
        msg = "Customer name is empty — cannot match Business Partner"
        issues.append(msg)
        logger.warning(msg)

    # ------------------------------------------------------------------
    # Step 2: Match each line item material → S/4HANA material code
    # ------------------------------------------------------------------
    all_items_matched = True

    for item in po.line_items:
        sap_material = ""
        matched = False
        item_score = 0.0
        item_description = item.description

        if item.material_code or item.description:
            # Try material_code first; fall back to description if no results
            mat_results = []
            if item.material_code:
                logger.info("Searching material by material_code=%r", item.material_code)
                try:
                    mat_results = search_material_odata(item.material_code, top=3, request=request)
                except Exception as exc:
                    logger.warning("Material search by code failed for %r: %s", item.material_code, exc)
            if not mat_results and item.description:
                logger.info("Searching material by description=%r", item.description)
                try:
                    mat_results = search_material_odata(item.description, top=3, request=request)
                except Exception as exc:
                    logger.warning("Material search by description failed for %r: %s", item.description, exc)

            search_term = item.material_code or item.description
            try:
                if mat_results:
                    best_mat = mat_results[0]
                    item_score = best_mat.get("score", 0.0)
                    if item_score >= _MATERIAL_SCORE_THRESHOLD:
                        matched = True
                        sap_material = best_mat.get("product", "")
                        item_description = best_mat.get("description") or item.description
                        logger.info(
                            "Material matched | extracted=%r | sap=%s | score=%.2f",
                            search_term,
                            sap_material,
                            item_score,
                        )
                    else:
                        msg = (
                            f"Material not matched: {search_term!r} "
                            f"(best: {best_mat.get('product')!r}, score={item_score:.2f})"
                        )
                        issues.append(msg)
                        logger.warning(msg)
                else:
                    msg = f"Material not matched: {search_term!r} (no results returned)"
                    issues.append(msg)
                    logger.warning(msg)
            except Exception as exc:
                msg = f"Material search failed for {search_term!r}: {exc}"
                issues.append(msg)
                logger.error(msg)
        else:
            msg = "Line item has neither material_code nor description — cannot match"
            issues.append(msg)
            logger.warning(msg)

        if not matched:
            all_items_matched = False

        items_validation.append(
            {
                "material_code_extracted": item.material_code,
                "sap_material": sap_material,
                "description": item_description,
                "matched": matched,
                "score": item_score,
            }
        )

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------
    ready_to_create = customer_resolved and all_items_matched and len(po.line_items) > 0

    logger.info(
        "Validation complete | customer_resolved=%s | items_matched=%s | ready=%s | issues=%d",
        customer_resolved,
        all_items_matched,
        ready_to_create,
        len(issues),
    )

    return SOValidationResult(
        customer_resolved=customer_resolved,
        customer_bp=customer_bp,
        customer_name_matched=customer_name_matched,
        customer_score=customer_score,
        items_validation=items_validation,
        ready_to_create=ready_to_create,
        issues=issues,
    )
