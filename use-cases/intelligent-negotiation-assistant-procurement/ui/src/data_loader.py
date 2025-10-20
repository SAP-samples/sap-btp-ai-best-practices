"""
Centralized data loading for the RFQ Analysis Dashboard.
This module handles loading supplier analysis data that is shared across all pages.
"""

import streamlit as st
import os
from src.api_client import (
    list_suppliers,
    analyze_ensure,
    analyze_compare,
    analyze_supplier_full,
)


@st.cache_data(show_spinner=False)
def load_supplier_data(
    force_refresh: bool = False,
    model_name: str | None = None,
    comparator_model_name: str | None = None,
    core_part_categories: list[str] | None = None,
    tqdcs_weights: dict | None = None,
    generate_metrics: bool = True,
    generate_strengths_weaknesses: bool = True,
    generate_recommendation_and_split: bool = True,
    need_comparison: bool = True,
):
    """
    Load supplier analysis data - shared across all pages.
    
    Args:
        force_refresh: If True, bypass cache and regenerate analysis
    
    Returns:
        Tuple of (supplier1_data, supplier2_data, comparison)
    """
    # Discover suppliers via API (packaged KGs) — cache in session to avoid repeated GETs
    if 'static_suppliers' not in st.session_state:
        supplier_listing = list_suppliers()
        # Surface API errors more clearly for troubleshooting
        if not isinstance(supplier_listing, dict):
            st.error("Failed to contact API. Please check API URL and network connectivity.")
            return None, None, None
        if supplier_listing.get("success") is False and supplier_listing.get("error"):
            # Fallback: construct suppliers from env if listing times out but we know IDs
            env_s1 = os.getenv("SUPPLIER1_ID")
            env_s2 = os.getenv("SUPPLIER2_ID")
            if supplier_listing.get("error", "").lower().startswith("request timed out") and env_s1 and env_s2:
                st.warning("Supplier list timed out; using SUPPLIER1_ID/SUPPLIER2_ID from environment.")
                st.session_state['static_suppliers'] = [
                    {"id": env_s1, "name": os.getenv("SUPPLIER1_NAME", env_s1)},
                    {"id": env_s2, "name": os.getenv("SUPPLIER2_NAME", env_s2)},
                ]
            else:
                st.error(f"API error: {supplier_listing.get('error')}")
                return None, None, None
        else:
            st.session_state['static_suppliers'] = supplier_listing.get("suppliers", [])
    suppliers = st.session_state.get('static_suppliers', [])
    if len(suppliers) < 2:
        st.error("No suppliers available from API. Ensure static KGs are packaged.")
        return None, None, None
    s1 = next((s for s in suppliers if s.get("id") in (os.getenv("SUPPLIER1_ID", "supplier1"),)), suppliers[0])
    s2 = next((s for s in suppliers if s.get("id") in (os.getenv("SUPPLIER2_ID", "supplier2"),)), suppliers[1])
    s1_id = s1.get("id")
    s2_id = s2.get("id")
    
    # Ensure consistent model defaults across pages when not explicitly provided
    # Match Home.py behavior: MODEL_NAME defaults to env LLM_MODEL or "gpt-5",
    # comparator defaults to env COMPARATOR_MODEL or "gemini-2.5-flash".
    effective_model_name = model_name or os.getenv("LLM_MODEL", "gpt-5")
    effective_comparator_model = comparator_model_name or os.getenv("COMPARATOR_MODEL", "gemini-2.5-flash")

    # Build a stable session key for current selection to persist across pages
    def _key_for_categories(cats: list[str] | None) -> str:
        if not cats:
            return "-"
        try:
            return ",".join(sorted([str(c).strip().lower() for c in cats if str(c).strip()]))
        except Exception:
            return "-"

    def _key_for_weights(w: dict | None) -> str:
        if not w:
            return "-"
        try:
            items = sorted([(str(k).strip().lower(), float(v)) for k, v in w.items()])
            return ";".join([f"{k}={v:.4f}" for k, v in items])
        except Exception:
            return "-"

    # If categories were not passed, inherit from session state or env
    if core_part_categories is None:
        try:
            core_part_categories = st.session_state.get("core_part_categories")
        except Exception:
            core_part_categories = None
        if core_part_categories is None:
            env_val = os.getenv("CORE_PART_CATEGORIES")
            if env_val:
                core_part_categories = [x.strip() for x in env_val.split(",") if x.strip()]

    analysis_key = "|".join([
        str(s1_id),
        str(s2_id),
        str(effective_model_name),
        str(effective_comparator_model),
        _key_for_categories(core_part_categories),
        _key_for_weights(tqdcs_weights),
    ])

    session_cache = st.session_state.setdefault('analysis_cache', {})
    session_force = bool(st.session_state.get('force_refresh_comparison', False) or force_refresh)

    # Fast path: reuse in-session results if available and no refresh requested
    if not session_force and analysis_key in session_cache:
        cached = session_cache.get(analysis_key)
        if isinstance(cached, tuple) and len(cached) == 3:
            if need_comparison:
                # If cached comparison is missing/empty, treat as cache miss to force compute
                if cached[2]:
                    return cached  # (supplier1_data, supplier2_data, comparison)
            else:
                # For pages that don't need comparison, return suppliers and empty comparison
                return cached[0], cached[1], {}

    try:
        # Determine whether we can do a fast compare-only recomputation using prior static analyses
        can_compare_only = False
        last_supplier_analyses = st.session_state.get('last_supplier_analyses')
        if last_supplier_analyses and last_supplier_analyses.get('s1') and last_supplier_analyses.get('s2'):
            # If we have prior analyses and a compare refresh was requested (Apply pressed) or weights provided,
            # run only the compare endpoint to recompute metrics/recommendations/split.
            if st.session_state.get('force_refresh_comparison') or tqdcs_weights is not None:
                can_compare_only = True

        if can_compare_only and need_comparison:
            with st.spinner("Recomputing comparison with updated RQDCE weights..."):
                # Use names from prior analyses if available; fall back to API listing
                s1_name = (
                    (last_supplier_analyses.get('s1_name') if isinstance(last_supplier_analyses, dict) else None)
                    or s1.get("name")
                    or str(s1_id)
                )
                s2_name = (
                    (last_supplier_analyses.get('s2_name') if isinstance(last_supplier_analyses, dict) else None)
                    or s2.get("name")
                    or str(s2_id)
                )

                compare_resp = analyze_compare(
                    supplier1_name=s1_name,
                    supplier2_name=s2_name,
                    supplier1_analyses=last_supplier_analyses.get('s1'),
                    supplier2_analyses=last_supplier_analyses.get('s2'),
                    tqdcs_weights=tqdcs_weights,
                    generate_metrics=bool(generate_metrics),
                    generate_strengths_weaknesses=bool(generate_strengths_weaknesses),
                    generate_recommendation_and_split=bool(generate_recommendation_and_split),
                    model=effective_comparator_model,
                )

                if not isinstance(compare_resp, dict) or 'comparison' not in compare_resp:
                    st.error("API did not return comparison results. Please check API logs.")
                    return None, None, None

                # Reuse prior supplier analyses; only comparison is updated
                supplier1_data = last_supplier_analyses.get('s1') or {}
                supplier2_data = last_supplier_analyses.get('s2') or {}
                comparison = compare_resp.get('comparison') or {}

                # Persist per-key cache
                session_cache[analysis_key] = (supplier1_data, supplier2_data, comparison)
                st.session_state['last_analysis_key'] = analysis_key

                # Clear trigger if set
                if st.session_state.get('force_refresh_comparison'):
                    st.session_state['force_refresh_comparison'] = False

                return supplier1_data, supplier2_data, comparison

        # If comparison is not needed, prefer per-supplier analysis path
        if not need_comparison:
            with st.spinner("Loading supplier analyses (without comparison)..."):
                s1_payload = analyze_supplier_full(
                    supplier_id=s1_id,
                    model=effective_model_name,
                    core_part_categories=core_part_categories,
                    force_refresh=session_force,
                )
                s2_payload = analyze_supplier_full(
                    supplier_id=s2_id,
                    model=effective_model_name,
                    core_part_categories=core_part_categories,
                    force_refresh=session_force,
                )
                if not isinstance(s1_payload, dict) or not isinstance(s2_payload, dict):
                    st.error("API did not return expected analysis payload. Please check API logs.")
                    return None, None, None
                supplier1_data = s1_payload
                supplier2_data = s2_payload
                comparison = {}

                # Cache suppliers without comparison
                session_cache[analysis_key] = (supplier1_data, supplier2_data, comparison)
                st.session_state['last_analysis_key'] = analysis_key
                st.session_state['last_supplier_analyses'] = {
                    's1_id': s1_id,
                    's2_id': s2_id,
                    's1_name': supplier1_data.get('supplier_name') or s1.get('name') or str(s1_id),
                    's2_name': supplier2_data.get('supplier_name') or s2.get('name') or str(s2_id),
                    's1': supplier1_data,
                    's2': supplier2_data,
                }
                if st.session_state.get('force_refresh_comparison'):
                    st.session_state['force_refresh_comparison'] = False
                return supplier1_data, supplier2_data, comparison

        # Fallback: ask backend to reuse cache or compute minimal work (includes comparison)
        with st.spinner("Loading and analyzing supplier data (this may take several minutes on first run)..."):
            payload = analyze_ensure(
                supplier1_id=s1_id,
                supplier2_id=s2_id,
                model=effective_model_name,
                comparator_model=effective_comparator_model,
                core_part_categories=core_part_categories,
                tqdcs_weights=tqdcs_weights if need_comparison else None,
                force_refresh=session_force,
                generate_metrics=bool(generate_metrics),
                generate_strengths_weaknesses=bool(generate_strengths_weaknesses),
                generate_recommendation_and_split=bool(generate_recommendation_and_split) if need_comparison else False,
            )
            if not isinstance(payload, dict) or not payload.get('supplier1') or not payload.get('supplier2'):
                st.error("API did not return expected analysis payload. Please check API logs.")
                return None, None, None
            supplier1_data = payload.get('supplier1') or {}
            supplier2_data = payload.get('supplier2') or {}
            comparison = payload.get('comparison') or {}
            if not need_comparison:
                # Drop comparison to avoid heavy downstream usage
                comparison = {}

            # Persist in session for reuse across pages
            session_cache[analysis_key] = (supplier1_data, supplier2_data, comparison)
            st.session_state['last_analysis_key'] = analysis_key
            st.session_state['last_supplier_analyses'] = {
                's1_id': s1_id,
                's2_id': s2_id,
                's1_name': supplier1_data.get('supplier_name') or s1.get('name') or str(s1_id),
                's2_name': supplier2_data.get('supplier_name') or s2.get('name') or str(s2_id),
                's1': supplier1_data,
                's2': supplier2_data,
            }

            # Clear the trigger after recompute initiated by weights change
            if st.session_state.get('force_refresh_comparison'):
                st.session_state['force_refresh_comparison'] = False
        return supplier1_data, supplier2_data, comparison
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
def get_data_processor():
    """Deprecated: analysis is handled by API now. Present for compatibility."""
    return None
