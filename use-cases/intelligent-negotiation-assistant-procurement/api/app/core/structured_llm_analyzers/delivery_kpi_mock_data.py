"""
External mock delivery KPI data for the Delivery analysis.

This module simulates an external integration that provides delivery KPIs
per supplier and country. It is intentionally simple and file-based to
serve as an example of how real external signals would be injected into
the TQDCS pipeline.

Provided metrics per supplier:
- lead_time_avg_days: Average lead time in days
- lead_time_variability_days: Lead time variability (standard deviation) in days
- otif_avg_pct: On-Time In-Full average performance in percent (0-100)
- otif_variability_pct: OTIF variability in percent points (standard deviation)
- country: Supplier headquarter country

Country-level signal:
- country_risk_score_1_to_5: Country risk on a 1-5 scale, where 1 is low risk
  and 5 is high risk.

Note: All values are mocked for demonstration purposes only.
"""
from typing import Dict, Any


# Mocked per-supplier KPIs. Keys must match the supplier names used in analysis.
SUPPLIER_DELIVERY_KPIS: Dict[str, Dict[str, Any]] = {
    "SupplierA": {
        "country": "Germany",
        "lead_time_avg_days": 7.5,
        "lead_time_variability_days": 1.2,
        "otif_avg_pct": 96.3,
        "otif_variability_pct": 2.1,
    },
    "SupplierB": {
        "country": "France",
        "lead_time_avg_days": 9.0,
        "lead_time_variability_days": 2.0,
        "otif_avg_pct": 94.0,
        "otif_variability_pct": 3.0,
    },
}


# Mocked country risk scores on a 1-5 scale (1 = low risk, 5 = high risk).
COUNTRY_RISK_SCORES_1_TO_5: Dict[str, float] = {
    "Germany": 1.5,
    "France": 1.8,
}


def get_delivery_kpis_for_supplier(supplier_name: str) -> Dict[str, Any]:
    """
    Return a combined dict of delivery KPIs for a given supplier including
    a country-level risk score.

    Args:
        supplier_name: Exact supplier name (e.g., "SupplierA").

    Returns:
        Dictionary containing KPI fields. Returns an empty dict if the
        supplier is unknown in the mock data.
    """
    supplier_data = SUPPLIER_DELIVERY_KPIS.get(supplier_name)
    if not supplier_data:
        return {}

    country = supplier_data.get("country", "")
    risk_score = COUNTRY_RISK_SCORES_1_TO_5.get(country)

    merged = dict(supplier_data)
    merged["country_risk_score_1_to_5"] = risk_score
    return merged


