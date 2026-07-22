from __future__ import annotations

from unittest.mock import patch

from app.routers import purchase_requisition as pr_router
from app.services.purchase_requisition_runner import _build_config
from app.services.s4_master_data import rank_business_partners, rank_materials, suggest_master_data
from app.services.s4_pr_client import S4PRConfig, _sap_date, build_pr_payload


S4_BOTH_MODES = {
    "S4_CONNECTION_MODE": "auto",
    "S4_BASE_URL": "https://s4.example.test",
    "S4_USERNAME": "local-user",
    "S4_PASSWORD": "local-password",
    "S4_DESTINATION_NAME": "S4_TEST",
    "DESTINATION_SERVICE_URI": "https://destination.example.test",
    "DESTINATION_TOKEN_BASE_URL": "https://token.example.test",
    "DESTINATION_CLIENT_ID": "destination-client",
    "DESTINATION_CLIENT_SECRET": "destination-secret",
}


def test_s4_auto_mode_uses_direct_credentials_locally() -> None:
    with patch.dict("os.environ", S4_BOTH_MODES, clear=True):
        config = S4PRConfig.from_env()

    assert config.uses_destination is False
    assert config.base_url == "https://s4.example.test"


def test_s4_auto_mode_uses_destination_in_cloud_foundry() -> None:
    with patch.dict("os.environ", {**S4_BOTH_MODES, "VCAP_APPLICATION": "{}"}, clear=True):
        config = S4PRConfig.from_env()

    assert config.uses_destination is True
    assert config.destination_name == "S4_TEST"


def test_customer_profile_builds_single_fast_extraction() -> None:
    config = _build_config(
        run_id="customer-run",
        documents=["quote.pdf"],
        mode="shortlist",
        manifest={
            "experiment_name": "customer-run",
            "include_docai": False,
            "include_llm": True,
            "selected_llm_models": ["gemini-2.5-flash"],
            "selected_llm_scenarios": ["detailed_static_prompt"],
            "selected_docai_scenarios": [],
            "approach_profile": "customer_fast_extraction",
        },
    )

    assert config.models == ["gemini-2.5-flash"]
    assert config.llm_scenarios == ["detailed_static_prompt"]
    assert config.docai_scenarios == []
    assert config.include_llm_summary is False


def test_successful_unscored_extraction_is_available_to_customer_ui() -> None:
    rows = [
        {
            "document": "quote.pdf",
            "method_family": "llm",
            "scenario": "detailed_static_prompt",
            "model": "gemini-2.5-flash",
            "status": "success",
            "quality_score": None,
            "confidence": None,
        }
    ]
    with patch.object(pr_router, "_rows", return_value=rows):
        candidates = pr_router._candidate_rows_for_document("quote.pdf", "customer-run")

    assert candidates == rows


def test_pr_payload_uses_plain_price_and_two_week_delivery_default() -> None:
    prepared = build_pr_payload(
        {
            "header": {"vendor_name": "Contoso Machine Services", "quote_number": "DEMO-136109", "currency": "USD"},
            "line_items": [
                {
                    "description": "Pump inspection",
                    "quantity": 1,
                    "unit_of_measure": "EA",
                    "unit_price": 1300.0,
                    "line_total": 1300.0,
                }
            ],
            "pr_mapping": {},
        }
    )

    item = prepared["payload"]["to_PurchaseReqnItem"]["results"][0]
    assert item["PurchaseRequisitionPrice"] == "1300"
    assert "E" not in item["PurchaseRequisitionPrice"]
    assert item["DeliveryDate"] == _sap_date(None, fallback_days=14)
    assert prepared["source_summary"]["defaulted_fields"]["delivery_date"] is True


def test_pr_payload_normalizes_each_alias_to_configured_sap_unit() -> None:
    prepared = build_pr_payload(
        {
            "header": {"currency": "USD"},
            "line_items": [{"description": "Part", "quantity": 1, "unit_of_measure": "eac", "unit_price": 10}],
            "pr_mapping": {},
        }
    )

    item = prepared["payload"]["to_PurchaseReqnItem"]["results"][0]
    assert item["BaseUnit"] == "PC"
    assert prepared["source_summary"]["defaulted_fields"]["base_unit"] is True
    assert prepared["source_summary"]["uom_normalizations"] == [{"item": 10, "source": "eac", "sap_value": "PC"}]


def test_pr_payload_base_unit_override_wins_over_extracted_unit() -> None:
    prepared = build_pr_payload(
        {
            "header": {"currency": "USD"},
            "line_items": [{"description": "Part", "quantity": 1, "unit_of_measure": "eac", "unit_price": 10}],
            "pr_mapping": {},
        },
        {"base_unit": "KG"},
    )

    item = prepared["payload"]["to_PurchaseReqnItem"]["results"][0]
    assert item["BaseUnit"] == "KG"
    assert prepared["source_summary"]["defaulted_fields"]["base_unit"] is False


def test_business_partner_fuzzy_match_auto_applies_only_high_confidence_supplier() -> None:
    result = rank_business_partners(
        "Northwind Industrial Supply",
        [
            {
                "BusinessPartner": "1000123",
                "Supplier": "1000123",
                "BusinessPartnerFullName": "Northwind Industrial Supply Inc",
            },
            {
                "BusinessPartner": "1000456",
                "Supplier": "1000456",
                "BusinessPartnerFullName": "Green Energy Ltd",
            },
        ],
    )

    assert result["status"] == "matched"
    assert result["auto_apply"] is True
    assert result["candidates"][0]["supplier"] == "1000123"
    assert result["candidates"][0]["confidence"] == "High"


def test_business_partner_unrelated_candidates_are_not_auto_applied() -> None:
    result = rank_business_partners(
        "Northwind Industrial Supply",
        [{"BusinessPartner": "1000456", "Supplier": "1000456", "BusinessPartnerFullName": "Green Energy Ltd"}],
    )

    assert result["status"] == "no_reliable_match"
    assert result["auto_apply"] is False


def test_material_identifier_match_outranks_description_only_candidate() -> None:
    result = rank_materials(
        {
            "description": "Push plate aluminum dull",
            "manufacturer_part_number": "70B.28",
            "vendor_material_number": "2RGL1",
        },
        [
            {
                "Product": "SP002",
                "ProductGroup": "YBPM01",
                "BaseUnit": "PC",
                "ProductManufacturerNumber": "70B-28",
                "descriptions": ["Door hardware push plate"],
            },
            {
                "Product": "OTHER",
                "ProductGroup": "YBPM01",
                "BaseUnit": "PC",
                "descriptions": ["Push plate aluminum"],
            },
        ],
    )

    assert result["auto_apply"] is True
    assert result["candidates"][0]["material"] == "SP002"
    assert result["candidates"][0]["score"] >= 92


def test_material_identifier_fragment_does_not_become_high_confidence_match() -> None:
    result = rank_materials(
        {
            "description": "Push Plate Aluminum Dull 3 1/2 x 15 In",
            "manufacturer_part_number": "70B.28",
        },
        [
            {
                "Product": "EXACT",
                "ProductManufacturerNumber": "70B.28",
                "descriptions": ["Push Plate Aluminum Dull 3 1/2 x 15 In"],
            },
            {
                "Product": "ASSEMBLY",
                "ProductManufacturerNumber": "132 X 70B.28",
                "descriptions": ["Pull Plate Oval Grip Dull Alum 3 1/2 x 15"],
            },
            {
                "Product": "OTHER_SIZE",
                "ProductManufacturerNumber": "70B.32",
                "descriptions": ["Push Plate Alum Dull 4 x 16 In"],
            },
        ],
    )

    assert result["candidates"][0]["material"] == "EXACT"
    assembly = next(candidate for candidate in result["candidates"] if candidate["material"] == "ASSEMBLY")
    assert assembly["identifier_score"] == 82
    assert assembly["confidence"] != "High"


def test_master_data_suggestions_do_not_invent_recommended_overrides() -> None:
    result = suggest_master_data(
        {
            "header": {"vendor_name": "Unknown Vendor"},
            "line_items": [{"description": "Unlisted custom assembly"}],
        },
        business_partners=[
            {"BusinessPartner": "1000", "Supplier": "1000", "BusinessPartnerFullName": "Green Energy"}
        ],
        products=[
            {"Product": "SP002", "ProductGroup": "YBPM01", "BaseUnit": "PC", "descriptions": ["Coupling"]}
        ],
    )

    assert result["status"] == "review_required"
    assert "supplier" not in result["recommended_overrides"]
    assert result["recommended_overrides"]["line_items"] == []


def test_pr_payload_accepts_per_line_master_data_overrides() -> None:
    prepared = build_pr_payload(
        {
            "header": {"currency": "USD"},
            "line_items": [
                {"description": "First item", "quantity": 1, "unit_price": 10},
                {"description": "Second item", "quantity": 2, "unit_price": 20},
            ],
            "pr_mapping": {},
        },
        {
            "line_items": [
                {"index": 0, "material": "MAT001", "material_group": "GROUP1", "base_unit": "PC"},
                {"index": 1, "material": "MAT002", "material_group": "GROUP2", "base_unit": "KG"},
            ]
        },
    )

    items = prepared["payload"]["to_PurchaseReqnItem"]["results"]
    assert [(item["Material"], item["MaterialGroup"], item["BaseUnit"]) for item in items] == [
        ("MAT001", "GROUP1", "PC"),
        ("MAT002", "GROUP2", "KG"),
    ]
