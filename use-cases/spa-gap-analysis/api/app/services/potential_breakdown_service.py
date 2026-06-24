"""
Material-Level Potential Breakdown Service

Uses canonical opportunity-layer parquet outputs.
"""
import logging
import hashlib
from typing import Dict, Optional

import pandas as pd

from .data_loader import is_anonymized_runtime_data, load_from_parquet

logger = logging.getLogger(__name__)


def _get_category_name(group_code) -> str:
    """Decode material group into a readable category label."""
    if pd.isna(group_code) or group_code == "":
        return ""

    group_code = str(group_code)
    if is_anonymized_runtime_data():
        digest = hashlib.sha256(group_code.encode("utf-8")).hexdigest()
        return f"Product Category {int(digest[:8], 16) % 1000000:06d}"

    if group_code.startswith("THHN"):
        return f"THHN Wire ({group_code})"
    if group_code.startswith("CARH"):
        return f"Cable/Harness ({group_code})"
    if group_code.startswith("EPVC"):
        return f"PVC Conduit ({group_code})"
    if group_code.startswith("BLS"):
        return f"Lighting/Ballast ({group_code})"
    if group_code.startswith("BULW"):
        return f"Bulbs/Lamps ({group_code})"
    if group_code.startswith("SQD"):
        return f"Square D Products ({group_code})"
    if group_code.startswith("ALB") or group_code.startswith("ALCF"):
        return f"Allen-Bradley Controls ({group_code})"
    if group_code.startswith("BRA"):
        return f"Breakers ({group_code})"
    if group_code.startswith("BSSC"):
        return f"Boxes/Steel ({group_code})"
    if group_code.startswith("FASTNR"):
        return f"Fasteners ({group_code})"
    if group_code.startswith("HUTL"):
        return f"Hand Tools/Utilities ({group_code})"
    if group_code.startswith("ELTO"):
        return f"Electric Tools ({group_code})"
    if group_code.startswith("ROI"):
        return f"ROI Products ({group_code})"
    if group_code.startswith("HUG"):
        return f"Hubbell Products ({group_code})"
    if group_code.startswith("RAB"):
        return f"RAB Lighting ({group_code})"
    if group_code.startswith("PENN"):
        return f"Penn Union ({group_code})"
    if group_code.startswith("CHN"):
        return f"Chain/Accessories ({group_code})"
    if group_code.startswith("MISC"):
        return f"Miscellaneous ({group_code})"
    if group_code.startswith("RITT"):
        return f"Ritttal Products ({group_code})"
    if group_code.startswith("CNTRCT"):
        return f"Contract Items ({group_code})"
    return f"Material Group: {group_code}"


def _build_rough_estimate_breakdown(
    enriched_materials: pd.DataFrame,
    total_cogs: float,
    top_n: int = 10
) -> Dict:
    """Build conservative rough-estimate metrics from top spend opportunity materials."""
    if enriched_materials.empty:
        return {
            "rough_total_potential": 0.0,
            "rough_potential_percent": 0.0,
            "raw_full_potential": 0.0,
            "raw_full_potential_percent": 0.0,
            "potential_is_estimate": True,
            "estimate_scope": {
                "basis": f"top_{top_n}_spend_materials_uncovered_first",
                "materials_considered": 0,
                "uncovered_materials_considered": 0,
                "cogs_considered": 0.0,
                "is_capped": False,
                "cap_value": float(total_cogs * 0.5) if total_cogs > 0 else 0.0,
                "note": "Rough estimate based on top opportunity materials by spend.",
            },
            "top_estimate_materials": [],
            "top_categories": [],
        }

    df = enriched_materials.copy()
    df["cogs_12m"] = pd.to_numeric(df["cogs_12m"], errors="coerce").fillna(0.0)
    df["incremental_savings_value"] = pd.to_numeric(df["incremental_savings_value"], errors="coerce").fillna(0.0)
    df["opportunity_priority"] = (df.get("opportunity_type", "") != "uncovered").astype(int)
    for optional_col in ["candidate_pricing_source", "current_pricing_source"]:
        if optional_col not in df.columns:
            df[optional_col] = ""

    full_potential = float(df["incremental_savings_value"].sum())
    full_potential_percent = float(full_potential / total_cogs * 100) if total_cogs > 0 else 0.0

    estimate_materials = df.sort_values(
        by=["opportunity_priority", "cogs_12m", "incremental_savings_value"],
        ascending=[True, False, False]
    ).head(top_n).copy()

    rough_total_raw = float(estimate_materials["incremental_savings_value"].sum())
    cap_value = float(total_cogs * 0.5) if total_cogs > 0 else 0.0
    is_capped = total_cogs > 0 and rough_total_raw > cap_value
    rough_total = min(rough_total_raw, cap_value) if is_capped else rough_total_raw
    rough_percent = float(rough_total / total_cogs * 100) if total_cogs > 0 else 0.0

    top_categories_df = estimate_materials.groupby(
        ["material_group", "category_name"], as_index=False
    ).agg({
        "material": "nunique",
        "cogs_12m": "sum",
        "incremental_savings_value": "sum",
    }).rename(columns={
        "material": "materials_count",
        "incremental_savings_value": "potential_savings",
    }).sort_values(["cogs_12m", "potential_savings"], ascending=[False, False]).head(10)

    note = (
        f"Rough estimate based on top {len(estimate_materials)} opportunity materials by spend, "
        "prioritizing uncovered materials."
    )
    if is_capped:
        note += " Capped at 50% of customer COGS."

    return {
        "rough_total_potential": float(rough_total),
        "rough_potential_percent": float(rough_percent),
        "raw_full_potential": float(full_potential),
        "raw_full_potential_percent": float(full_potential_percent),
        "potential_is_estimate": True,
        "estimate_scope": {
            "basis": f"top_{top_n}_spend_materials_uncovered_first",
            "materials_considered": int(len(estimate_materials)),
            "uncovered_materials_considered": int((estimate_materials.get("opportunity_type", "") == "uncovered").sum()),
            "cogs_considered": float(estimate_materials["cogs_12m"].sum()),
            "is_capped": bool(is_capped),
            "cap_value": float(cap_value),
            "note": note,
        },
        "top_estimate_materials": estimate_materials[[
            "material",
            "material_group",
            "category_name",
            "material_description",
            "cogs_12m",
            "best_candidate_spa",
            "baseline_unit_price",
            "best_candidate_unit_price",
            "candidate_pricing_source",
            "current_pricing_source",
            "incremental_savings_pct",
            "incremental_savings_value",
            "opportunity_type",
        ]].rename(columns={
            "best_candidate_spa": "sales_deal",
            "baseline_unit_price": "base_cost",
            "best_candidate_unit_price": "spa_price",
            "candidate_pricing_source": "pricing_source",
            "incremental_savings_pct": "savings_percent",
            "incremental_savings_value": "potential_savings",
        }).fillna("").to_dict("records"),
        "top_categories": top_categories_df.fillna("").to_dict("records"),
    }


def get_potential_breakdown(customer_id: str) -> Dict:
    """
    Get detailed material-level breakdown of incremental potential savings.

    Returns the same high-level response shape as the legacy service, but
    values now come from canonical opportunity parquet outputs.
    """
    logger.info(f"Getting potential breakdown for customer {customer_id}")

    customer_id = str(customer_id)

    customer_master = load_from_parquet("customer_master.parquet")
    current_pricing = load_from_parquet("customer_material_current_pricing.parquet")
    current_metrics = load_from_parquet("customer_current_metrics.parquet")
    assignments = load_from_parquet("customer_spa_assignments.parquet")
    material_opportunities = load_from_parquet("customer_material_opportunities.parquet")
    bundle_recommendations = load_from_parquet("customer_spa_bundle_recommendations.parquet")
    sap_master = load_from_parquet("sap_master_enhanced.parquet")

    try:
        material_descriptions = load_from_parquet("a901_materials.parquet")[["material", "material_description"]]
    except Exception:
        material_descriptions = pd.DataFrame(columns=["material", "material_description"])

    customer_profile = customer_master[customer_master["customer_id"] == customer_id]
    customer_profile_name = (
        str(customer_profile.iloc[0].get("customer_name", "Unknown"))
        if not customer_profile.empty
        else "Unknown"
    )

    customer_current_pricing = current_pricing[current_pricing["customer_id"] == customer_id].copy()
    if customer_current_pricing.empty:
        raise ValueError(f"No current pricing data for customer {customer_id}")

    customer_metrics = current_metrics[current_metrics["customer_id"] == customer_id]
    total_cogs = (
        float(customer_metrics.iloc[0].get("total_cogs_q4", 0.0))
        if not customer_metrics.empty
        else float(customer_current_pricing["cogs_12m"].sum())
    )

    customer_assignments = assignments[
        (assignments["customer_id"] == customer_id) &
        (assignments["is_active"])
    ]
    current_spas_count = int(customer_assignments["agreement_id"].nunique())

    customer_opportunities = material_opportunities[
        material_opportunities["customer_id"] == customer_id
    ].copy()
    customer_bundle_recommendations = bundle_recommendations[
        bundle_recommendations["customer_id"] == customer_id
    ].copy()

    missing_spas_count = int(customer_bundle_recommendations["agreement_id"].nunique()) if not customer_bundle_recommendations.empty else 0

    covered_materials = set(
        customer_current_pricing.loc[customer_current_pricing["is_currently_covered"], "material"].tolist()
    )
    opportunity_materials = set(customer_opportunities["material"].tolist()) if not customer_opportunities.empty else set()
    materials_with_spa_pricing = covered_materials | opportunity_materials

    total_materials = int(customer_current_pricing["material"].nunique())
    materials_with_spa_pricing_count = len(materials_with_spa_pricing)
    materials_in_missing_spas = len(opportunity_materials)

    cogs_with_spa_pricing = float(
        customer_current_pricing[customer_current_pricing["material"].isin(materials_with_spa_pricing)]["cogs_12m"].sum()
    )
    cogs_in_missing_spas = float(customer_opportunities["cogs_12m"].sum()) if not customer_opportunities.empty else 0.0

    top_materials = []
    enriched_materials = pd.DataFrame()
    if not customer_opportunities.empty:
        enriched_materials = customer_opportunities.merge(
            sap_master[["material", "material_group"]].drop_duplicates(subset=["material"]),
            on="material",
            how="left",
            suffixes=("", "_sap"),
        )
        if "material_group_sap" in enriched_materials.columns:
            enriched_materials["material_group"] = enriched_materials["material_group"].fillna(
                enriched_materials["material_group_sap"]
            )
            enriched_materials = enriched_materials.drop(columns=["material_group_sap"])

        enriched_materials = enriched_materials.merge(
            material_descriptions,
            on="material",
            how="left",
        )
        enriched_materials["category_name"] = enriched_materials["material_group"].apply(_get_category_name)
        for optional_col in ["candidate_pricing_source", "current_pricing_source"]:
            if optional_col not in enriched_materials.columns:
                enriched_materials[optional_col] = ""

        top_materials_df = enriched_materials.nlargest(20, "incremental_savings_value")[
            [
                "material",
                "material_group",
                "category_name",
                "material_description",
                "cogs_12m",
                "best_candidate_spa",
                "baseline_unit_price",
                "best_candidate_unit_price",
                "candidate_pricing_source",
                "current_pricing_source",
                "incremental_savings_pct",
                "incremental_savings_value",
            ]
        ].rename(columns={
            "best_candidate_spa": "sales_deal",
            "baseline_unit_price": "base_cost",
            "best_candidate_unit_price": "spa_price",
            "candidate_pricing_source": "pricing_source",
            "incremental_savings_pct": "savings_percent",
            "incremental_savings_value": "potential_savings",
        })

        top_materials_df = top_materials_df.fillna("")
        top_materials_df["rebated_cost_status"] = top_materials_df["pricing_source"].apply(
            lambda source: (
                "A703 exact/netted/rebated cost"
                if str(source).upper() == "A703"
                else "A704 multiplier pricing is Phase 2, not exact POC opportunity"
                if str(source).upper() == "A704"
                else "Unknown pricing source"
            )
        )
        top_materials = top_materials_df.to_dict("records")

    rough_breakdown = _build_rough_estimate_breakdown(enriched_materials, total_cogs, top_n=10)
    for material in rough_breakdown["top_estimate_materials"]:
        source = str(material.get("pricing_source", "")).upper()
        if source == "A703":
            material["rebated_cost_status"] = "A703 exact/netted/rebated cost"
        elif source == "A704":
            material["rebated_cost_status"] = "A704 multiplier pricing is Phase 2, not exact POC opportunity"
        else:
            material["rebated_cost_status"] = "Unknown pricing source"

    spa_breakdown_list = []
    full_bundle_potential = 0.0
    if not customer_bundle_recommendations.empty:
        spa_breakdown = customer_bundle_recommendations[[
            "agreement_id",
            "new_materials_count",
            "new_covered_cogs",
            "incremental_savings_value_after_dedup",
            "avg_incremental_savings_pct",
            "bundle_rank",
        ]].rename(columns={
            "agreement_id": "spa_id",
            "new_materials_count": "materials_count",
            "new_covered_cogs": "cogs_covered",
            "incremental_savings_value_after_dedup": "potential_savings",
            "avg_incremental_savings_pct": "avg_savings_pct",
        }).sort_values(["bundle_rank", "potential_savings"], ascending=[True, False])
        full_bundle_potential = float(spa_breakdown["potential_savings"].sum())
        spa_breakdown_list = spa_breakdown.to_dict("records")
    full_bundle_potential_percent = float(full_bundle_potential / total_cogs * 100) if total_cogs > 0 else 0.0

    return {
        "customer_id": customer_id,
        "customer_name": customer_profile_name,
        "total_cogs": float(total_cogs),
        "total_potential": float(rough_breakdown["rough_total_potential"]),
        "potential_percent": float(rough_breakdown["rough_potential_percent"]),
        "full_potential": float(full_bundle_potential),
        "full_potential_percent": float(full_bundle_potential_percent),
        "raw_full_potential": float(rough_breakdown["raw_full_potential"]),
        "raw_full_potential_percent": float(rough_breakdown["raw_full_potential_percent"]),
        "potential_is_estimate": bool(rough_breakdown["potential_is_estimate"]),
        "estimate_scope": rough_breakdown["estimate_scope"],
        "current_spas_count": current_spas_count,
        "missing_spas_count": missing_spas_count,
        "coverage": {
            "total_materials": total_materials,
            "materials_with_spa_pricing": int(materials_with_spa_pricing_count),
            "materials_with_spa_pricing_pct": float(materials_with_spa_pricing_count / total_materials * 100) if total_materials > 0 else 0.0,
            "materials_in_missing_spas": int(materials_in_missing_spas),
            "cogs_with_spa_pricing": float(cogs_with_spa_pricing),
            "cogs_with_spa_pricing_pct": float(cogs_with_spa_pricing / total_cogs * 100) if total_cogs > 0 else 0.0,
            "cogs_in_missing_spas": float(cogs_in_missing_spas),
            "cogs_in_missing_spas_pct": float(cogs_in_missing_spas / total_cogs * 100) if total_cogs > 0 else 0.0,
        },
        "top_materials": top_materials,
        "top_estimate_materials": rough_breakdown["top_estimate_materials"],
        "top_categories": rough_breakdown["top_categories"],
        "spa_breakdown": spa_breakdown_list,
    }


def get_spa_materials(spa_id: str, customer_id: Optional[str] = None) -> Dict:
    """
    Get materials covered by a specific SPA.

    If customer_id is provided, use canonical opportunity candidates for that
    customer-SPA pairing. Otherwise, fall back to global material savings.
    """
    logger.info(f"Getting materials for SPA {spa_id}, customer {customer_id}")

    spa_id = str(spa_id)

    if customer_id:
        customer_id = str(customer_id)
        candidates = load_from_parquet("customer_material_opportunity_candidates.parquet")
        customer_candidates = candidates[
            (candidates["customer_id"] == customer_id) &
            (candidates["agreement_id"].astype(str) == spa_id)
        ].copy()

        if customer_candidates.empty:
            return {
                "spa_id": spa_id,
                "customer_id": customer_id,
                "total_materials": 0,
                "materials": [],
            }

        customer_candidates = customer_candidates.sort_values("incremental_savings_value", ascending=False).head(50)
        materials_list = []
        for _, row in customer_candidates.iterrows():
            pricing_source = str(row.get("candidate_pricing_source", "") or "")
            materials_list.append({
                "material": str(row["material"]),
                "base_cost": float(row["baseline_unit_price"]),
                "spa_price": float(row["candidate_unit_price"]),
                "pricing_source": pricing_source,
                "rebated_cost_status": (
                    "A703 exact/netted/rebated cost"
                    if pricing_source.upper() == "A703"
                    else "A704 multiplier pricing is Phase 2, not exact POC opportunity"
                    if pricing_source.upper() == "A704"
                    else "Unknown pricing source"
                ),
                "savings": float(row["incremental_unit_savings"]),
                "savings_percent": float(row["incremental_savings_pct"]),
                "customer_cogs": float(row["cogs_12m"]),
                "potential_savings": float(row["incremental_savings_value"]),
            })

        return {
            "spa_id": spa_id,
            "customer_id": customer_id,
            "total_materials": len(materials_list),
            "materials": materials_list,
        }

    mat_savings = load_from_parquet("material_savings.parquet")
    spa_materials = mat_savings[mat_savings["sales_deal"].astype(str) == spa_id].copy()

    if spa_materials.empty:
        return {
            "spa_id": spa_id,
            "total_materials": 0,
            "materials": [],
        }

    spa_materials = spa_materials.sort_values("savings_percent", ascending=False).head(50)
    materials_list = []
    for _, row in spa_materials.iterrows():
        materials_list.append({
            "material": str(row["material"]),
            "base_cost": float(row["base_cost"]),
            "spa_price": float(row["spa_price"]),
            "savings": float(row["savings"]),
            "savings_percent": float(row["savings_percent"]),
        })

    return {
        "spa_id": spa_id,
        "customer_id": customer_id,
        "total_materials": len(materials_list),
        "materials": materials_list,
    }
