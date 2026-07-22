"""Read-only S/4 master-data lookup and deterministic fuzzy matching.

The module deliberately keeps matching deterministic. S/4 remains the source
of truth, while the extracted quote supplies search hints that may be absent
from the final purchase requisition payload.
"""

from __future__ import annotations

import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

from app.services.s4_pr_client import (
    PR_SERVICE_PATH,
    S4PRConfig,
    S4PRError,
    _request_context,
    _sanitize_error_text,
)

logger = logging.getLogger(__name__)

BUSINESS_PARTNER_SERVICE_PATH = "/sap/opu/odata/sap/API_BUSINESS_PARTNER"
PRODUCT_SERVICE_PATH = "/sap/opu/odata/sap/API_PRODUCT_SRV"
AUTO_APPLY_THRESHOLD = 88.0
REVIEW_THRESHOLD = 65.0
DEFAULT_CACHE_SECONDS = 300
DEFAULT_MAX_RECORDS = 2500

_catalog_cache: dict[str, Any] = {"loaded_at": 0.0, "business_partners": [], "products": []}


def _service_context(service_path: str) -> tuple[S4PRConfig, str, dict[str, Any], str | None]:
    config = S4PRConfig.from_env()
    pr_url, request_config, client, _destination = _request_context(config)
    if not pr_url.endswith(PR_SERVICE_PATH):
        raise S4PRError("Could not derive the S/4 backend URL for master-data lookup.")
    backend_url = pr_url[: -len(PR_SERVICE_PATH)]
    return config, backend_url + service_path, request_config, client


def _odata_results(
    service_path: str,
    entity_set: str,
    select_fields: list[str],
    *,
    filter_expression: str | None = None,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> list[dict[str, Any]]:
    _config, service_url, request_config, client = _service_context(service_path)
    session = requests.Session()
    page_size = 500
    rows: list[dict[str, Any]] = []

    while len(rows) < max_records:
        params = {
            "$format": "json",
            "$select": ",".join(select_fields),
            "$top": str(min(page_size, max_records - len(rows))),
            "$skip": str(len(rows)),
        }
        if client:
            params["sap-client"] = client
        if filter_expression:
            params["$filter"] = filter_expression
        try:
            response = session.get(
                f"{service_url}/{entity_set}",
                params=params,
                headers=request_config["headers"],
                proxies=request_config["proxies"],
                verify=request_config["verify"],
                timeout=request_config["timeout"],
            )
        except requests.RequestException as exc:
            raise S4PRError(
                f"Could not read S/4 master data from {entity_set}.",
                details=_sanitize_error_text(str(exc)),
            ) from exc
        if response.status_code >= 400:
            raise S4PRError(
                f"S/4 master-data request failed for {entity_set}.",
                status_code=response.status_code,
                details=_sanitize_error_text(response.text),
            )
        try:
            payload = response.json()
            page = payload.get("d", {}).get("results", [])
        except (TypeError, ValueError) as exc:
            raise S4PRError(f"S/4 returned an invalid response for {entity_set}.") from exc
        page = [item for item in page if isinstance(item, dict)]
        rows.extend(page)
        if len(page) < page_size:
            break
    return rows


def _load_catalogs_from_s4() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    partners = _odata_results(
        BUSINESS_PARTNER_SERVICE_PATH,
        "A_BusinessPartner",
        [
            "BusinessPartner",
            "BusinessPartnerFullName",
            "BusinessPartnerName",
            "OrganizationBPName1",
            "OrganizationBPName2",
            "SearchTerm1",
            "SearchTerm2",
            "Supplier",
        ],
    )
    partners = [row for row in partners if str(row.get("Supplier") or "").strip()]

    products = _odata_results(
        PRODUCT_SERVICE_PATH,
        "A_Product",
        [
            "Product",
            "ProductOldID",
            "ProductGroup",
            "BaseUnit",
            "ProductManufacturerNumber",
            "ManufacturerNumber",
            "SizeOrDimensionText",
        ],
    )
    descriptions = _odata_results(
        PRODUCT_SERVICE_PATH,
        "A_ProductDescription",
        ["Product", "Language", "ProductDescription"],
    )
    product_plants = _odata_results(
        PRODUCT_SERVICE_PATH,
        "A_ProductPlant",
        ["Product", "Plant"],
    )
    descriptions_by_product: dict[str, list[str]] = {}
    for row in descriptions:
        product = str(row.get("Product") or "").strip()
        description = str(row.get("ProductDescription") or "").strip()
        language = str(row.get("Language") or "").upper()
        if not product or not description:
            continue
        bucket = descriptions_by_product.setdefault(product, [])
        if language == "EN":
            bucket.insert(0, description)
        elif description not in bucket:
            bucket.append(description)
    plants_by_product: dict[str, set[str]] = {}
    for row in product_plants:
        product = str(row.get("Product") or "").strip()
        plant = str(row.get("Plant") or "").strip()
        if product and plant:
            plants_by_product.setdefault(product, set()).add(plant)
    for product in products:
        product_id = str(product.get("Product") or "")
        product["descriptions"] = list(dict.fromkeys(descriptions_by_product.get(product_id, [])))
        product["plants"] = sorted(plants_by_product.get(product_id, set()))
    return partners, products


def load_master_data_catalogs(*, force_refresh: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache_seconds = max(0, int(os.getenv("S4_MASTER_DATA_CACHE_SECONDS", str(DEFAULT_CACHE_SECONDS))))
    age = time.monotonic() - float(_catalog_cache.get("loaded_at") or 0.0)
    if not force_refresh and age < cache_seconds and _catalog_cache.get("products"):
        return list(_catalog_cache["business_partners"]), list(_catalog_cache["products"])
    partners, products = _load_catalogs_from_s4()
    _catalog_cache.update(
        {"loaded_at": time.monotonic(), "business_partners": partners, "products": products}
    )
    logger.info("Loaded read-only S/4 master data: %s suppliers, %s products", len(partners), len(products))
    return list(partners), list(products)


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()


def _compact_identifier(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _tokens(value: Any) -> set[str]:
    return {token for token in _normalized_text(value).split() if len(token) > 1}


def _text_similarity(left: Any, right: Any) -> float:
    left_text = _normalized_text(left)
    right_text = _normalized_text(right)
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 100.0
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    common = left_tokens & right_tokens
    if not common:
        return round(min(45.0, 100.0 * SequenceMatcher(None, left_text, right_text).ratio()), 1)
    if left_text in right_text or right_text in left_text:
        containment = 94.0
    else:
        containment = 0.0
    sequence = 100.0 * SequenceMatcher(None, left_text, right_text).ratio()
    token_f1 = 200.0 * len(common) / (len(left_tokens) + len(right_tokens))
    return round(max(containment, sequence, token_f1), 1)


def _identifier_similarity(left: Any, right: Any) -> float:
    left_id = _compact_identifier(left)
    right_id = _compact_identifier(right)
    if not left_id or not right_id:
        return 0.0
    if left_id == right_id:
        return 100.0
    if min(len(left_id), len(right_id)) >= 4 and (left_id in right_id or right_id in left_id):
        # A shared identifier fragment is useful evidence, but not an exact
        # material match. For example, 70B.28 and 132X70B.28 are distinct
        # door-hardware products and must remain human-review candidates.
        return 82.0
    return round(min(70.0, 100.0 * SequenceMatcher(None, left_id, right_id).ratio()), 1)


def _confidence(score: float) -> str:
    if score >= AUTO_APPLY_THRESHOLD:
        return "High"
    if score >= REVIEW_THRESHOLD:
        return "Medium"
    return "Low"


def _match_status(score: float) -> str:
    if score >= AUTO_APPLY_THRESHOLD:
        return "matched"
    if score >= REVIEW_THRESHOLD:
        return "review"
    return "no_reliable_match"


def rank_business_partners(
    vendor_name: Any,
    business_partners: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    query = str(vendor_name or "").strip()
    candidates: list[dict[str, Any]] = []
    name_fields = [
        "BusinessPartnerFullName",
        "BusinessPartnerName",
        "OrganizationBPName1",
        "OrganizationBPName2",
        "SearchTerm1",
        "SearchTerm2",
    ]
    for row in business_partners:
        scored_names = [(field, row.get(field), _text_similarity(query, row.get(field))) for field in name_fields]
        field, matched_name, score = max(scored_names, key=lambda item: item[2])
        candidates.append(
            {
                "business_partner": str(row.get("BusinessPartner") or ""),
                "supplier": str(row.get("Supplier") or ""),
                "name": str(row.get("BusinessPartnerFullName") or row.get("BusinessPartnerName") or matched_name or ""),
                "score": score,
                "confidence": _confidence(score),
                "reason": f"Closest supplier name match using {field}",
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["name"]), reverse=True)
    top = candidates[:limit]
    best_score = float(top[0]["score"]) if top else 0.0
    return {
        "query": query,
        "status": _match_status(best_score) if query else "missing_source_value",
        "auto_apply": bool(query and best_score >= AUTO_APPLY_THRESHOLD),
        "candidates": top,
    }


def _product_description_score(source: dict[str, Any], product: dict[str, Any]) -> float:
    source_description = source.get("description") or source.get("service_description")
    candidate_values = list(product.get("descriptions") or []) + [product.get("SizeOrDimensionText")]
    return max((_text_similarity(source_description, value) for value in candidate_values), default=0.0)


def _product_identifier_score(source: dict[str, Any], product: dict[str, Any]) -> tuple[float, str | None]:
    source_ids = [source.get("manufacturer_part_number"), source.get("vendor_material_number")]
    product_ids = [
        product.get("Product"),
        product.get("ProductOldID"),
        product.get("ProductManufacturerNumber"),
        product.get("ManufacturerNumber"),
    ]
    scored = [
        (_identifier_similarity(source_id, product_id), source_id, product_id)
        for source_id in source_ids
        for product_id in product_ids
        if source_id and product_id
    ]
    if not scored:
        return 0.0, None
    score, source_id, product_id = max(scored, key=lambda item: item[0])
    reason = f"Identifier {source_id} compared with {product_id}" if score else None
    return score, reason


def rank_materials(
    source_item: dict[str, Any],
    products: list[dict[str, Any]],
    *,
    required_plant: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for product in products:
        description_score = _product_description_score(source_item, product)
        identifier_score, identifier_reason = _product_identifier_score(source_item, product)
        if identifier_score >= 90:
            score = max(98.0 if identifier_score == 100 else 92.0, identifier_score * 0.85 + description_score * 0.15)
        elif identifier_score:
            score = max(description_score, identifier_score * 0.45 + description_score * 0.55)
        else:
            score = description_score
        score = round(min(100.0, score), 1)
        descriptions = list(product.get("descriptions") or [])
        reasons = [f"Description similarity {description_score:.0f}%"]
        if identifier_reason:
            reasons.append(f"{identifier_reason} ({identifier_score:.0f}%)")
        confidence = _confidence(score)
        if confidence == "High" and identifier_score < 90 and description_score < 96:
            confidence = "Medium"
        plants = [str(value) for value in product.get("plants") or [] if str(value).strip()]
        plant_ready = not required_plant or required_plant in plants
        if required_plant:
            reasons.append(
                f"Available in plant {required_plant}"
                if plant_ready
                else f"Not maintained in plant {required_plant}"
            )
        candidates.append(
            {
                "material": str(product.get("Product") or ""),
                "material_description": str(descriptions[0] if descriptions else product.get("SizeOrDimensionText") or ""),
                "material_group": str(product.get("ProductGroup") or ""),
                "base_unit": str(product.get("BaseUnit") or ""),
                "manufacturer_part_number": str(product.get("ProductManufacturerNumber") or ""),
                "score": score,
                "description_score": round(description_score, 1),
                "identifier_score": round(identifier_score, 1),
                "confidence": confidence,
                "plants": plants,
                "required_plant": required_plant or "",
                "plant_ready": plant_ready,
                "reason": "; ".join(reasons),
            }
        )
    candidates.sort(key=lambda item: (item["score"], item["material"]), reverse=True)
    top = candidates[:limit]
    best_score = float(top[0]["score"]) if top else 0.0
    best_has_strong_evidence = bool(
        top
        and (
            float(top[0].get("identifier_score") or 0) >= 90
            or float(top[0].get("description_score") or 0) >= 96
        )
    )
    best_is_plant_ready = bool(top and top[0].get("plant_ready", True))
    return {
        "line_number": source_item.get("line_number"),
        "description": source_item.get("description") or source_item.get("service_description") or "",
        "manufacturer_part_number": source_item.get("manufacturer_part_number") or "",
        "vendor_material_number": source_item.get("vendor_material_number") or "",
        "status": (
            _match_status(best_score)
            if best_has_strong_evidence and best_is_plant_ready
            else ("review" if best_score >= REVIEW_THRESHOLD else "no_reliable_match")
        ),
        "auto_apply": best_score >= AUTO_APPLY_THRESHOLD and best_has_strong_evidence and best_is_plant_ready,
        "candidates": top,
    }


def suggest_master_data(
    normalized: dict[str, Any],
    *,
    business_partners: list[dict[str, Any]] | None = None,
    products: list[dict[str, Any]] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    if business_partners is None or products is None:
        loaded_partners, loaded_products = load_master_data_catalogs(force_refresh=force_refresh)
        business_partners = loaded_partners if business_partners is None else business_partners
        products = loaded_products if products is None else products
    header = normalized.get("header") if isinstance(normalized.get("header"), dict) else {}
    source_items = normalized.get("line_items") if isinstance(normalized.get("line_items"), list) else []
    source_items = [item for item in source_items if isinstance(item, dict)]
    pr_mapping = normalized.get("pr_mapping") if isinstance(normalized.get("pr_mapping"), dict) else {}
    required_plant = str(
        pr_mapping.get("plant") or os.getenv("S4_PR_DEFAULT_PLANT") or "1710"
    ).strip()
    partner_match = rank_business_partners(header.get("vendor_name"), business_partners)
    material_matches = [
        rank_materials(item, products, required_plant=required_plant) for item in source_items
    ]

    # Matching is intentionally advisory in the customer workflow. Even a
    # high-confidence candidate may lack PR-specific organizational data.
    partner_match["auto_apply"] = False
    for match in material_matches:
        match["auto_apply"] = False
    recommended: dict[str, Any] = {"line_items": []}

    needs_review = (
        partner_match["status"] != "matched"
        or any(match["status"] != "matched" for match in material_matches)
    )
    return {
        "status": "review_required" if needs_review else "matched",
        "source": "SAP S/4HANA master data (read-only)",
        "business_partner": partner_match,
        "materials": material_matches,
        "recommended_overrides": recommended,
        "application_mode": "suggestions_only",
        "catalog": {
            "supplier_count": len(business_partners),
            "product_count": len(products),
            "cache_seconds": int(os.getenv("S4_MASTER_DATA_CACHE_SECONDS", str(DEFAULT_CACHE_SECONDS))),
            "required_plant": required_plant,
        },
        "guardrail": (
            "Matches are suggestions only. A supplier or plant-ready material changes the PR "
            "only after explicit human selection."
        ),
    }


def preflight_master_data_apis() -> dict[str, Any]:
    """Verify both master-data metadata endpoints without changing S/4."""

    checks: list[dict[str, Any]] = []
    for service_name, service_path, expected_entity in [
        ("API_BUSINESS_PARTNER", BUSINESS_PARTNER_SERVICE_PATH, "A_BusinessPartner"),
        ("API_PRODUCT_SRV", PRODUCT_SERVICE_PATH, "A_Product"),
    ]:
        config, service_url, request_config, client = _service_context(service_path)
        try:
            start = time.perf_counter()
            response = requests.get(
                f"{service_url}/$metadata",
                params={"sap-client": client} if client else None,
                headers={**request_config["headers"], "Accept": "application/xml"},
                proxies=request_config["proxies"],
                verify=request_config["verify"],
                timeout=request_config["timeout"],
            )
        except requests.RequestException as exc:
            raise S4PRError(
                f"S/4 metadata check failed for {service_name}.",
                details=_sanitize_error_text(str(exc)),
            ) from exc
        checks.append(
            {
                "service": service_name,
                "status_code": response.status_code,
                "available": response.status_code < 400 and expected_entity in response.text,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
        )
    return {
        "status": "available" if all(check["available"] for check in checks) else "unavailable",
        "connection_mode": "BTP Destination" if config.uses_destination else "Direct Basic Auth",
        "read_only": True,
        "services": checks,
    }
