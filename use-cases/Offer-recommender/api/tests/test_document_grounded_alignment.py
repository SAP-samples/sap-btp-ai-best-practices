from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.nbo.config import (
    COL_BILLING_ACCOUNT,
    COL_PROFILE_DWELLING_TYPE,
    COL_PROFILE_SERVICE_ENTRANCE_AMPS,
)
from app.nbo.data_loader import DataStore
from app.nbo.decision_facts import compute_account_facts
from app.nbo.models import Confidence, DecisionStatus, FactSource, FactValue, OfferDecision
from app.services.recommendations import RecommendationService
from synthetic_runtime import synthetic_runtime_datasets


def test_program_catalog_enriches_aliases_from_program_contract() -> None:
    service = RecommendationService()

    assert "HAD" in service.catalog_index["income_qualified_discount"]["program_code_aliases"]
    assert "BATSI" in service.catalog_index["battery_partner"]["program_code_aliases"]


def test_service_charge_tier_derives_from_account_profile() -> None:
    datasets = synthetic_runtime_datasets()
    datasets["account_profile"] = pd.concat(
        [
            datasets["account_profile"],
            pd.DataFrame(
                [
                    {
                        COL_BILLING_ACCOUNT: "6001",
                        COL_PROFILE_DWELLING_TYPE: "ATTACHED_HOME",
                        COL_PROFILE_SERVICE_ENTRANCE_AMPS: 200,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    facts = compute_account_facts("6001", DataStore(datasets))

    assert facts["service_charge_tier"].value == "tier1"
    assert facts["service_charge_tier"].missing_reason is None


def test_recent_decline_detection_uses_90_day_window() -> None:
    service = RecommendationService()
    offer = OfferDecision(
        program_id="income_qualified_discount",
        display_name="Household Assistance Discount",
        status=DecisionStatus.ELIGIBLE,
        metadata={"program_code_aliases": ["HAD"]},
    )
    facts = {
        "_program_event_history": FactValue(
            fact_id="_program_event_history",
            value_type="list[dict]",
            source=FactSource.EXTERNAL,
            value=[
                {
                    "PROGRAM ID": "income_qualified_discount",
                    "PROGRAM CODE": "HAD",
                    "EVENT TYPE": "declined",
                    "EVENT DATE": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                }
            ],
            confidence=Confidence.HIGH,
        ),
        "_decline_suppression_days": FactValue(
            fact_id="_decline_suppression_days",
            value_type="integer",
            source=FactSource.SYSTEM,
            value=90,
            confidence=Confidence.HIGH,
        ),
    }

    assert service._recent_decline_detected(offer, facts) is True


def test_current_enrollment_detection_uses_alias_codes() -> None:
    service = RecommendationService()
    offer = OfferDecision(
        program_id="battery_partner",
        display_name="Battery Partner",
        status=DecisionStatus.ELIGIBLE,
        metadata={"program_code_aliases": ["BATSI"]},
    )
    facts = {
        "current_program_codes": FactValue(
            fact_id="current_program_codes",
            value_type="list[string]",
            source=FactSource.WORKBOOK,
            value=["BATSI"],
            confidence=Confidence.HIGH,
        )
    }

    assert service._current_enrollment_detected(offer, facts) is True
