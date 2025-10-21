"""
TQDCS Dashboard Page
This page contains the EXACT content from the render_tqdcs_tab method in enhanced_dashboard.py
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys

# Add parent directory for imports
current_file = Path(__file__).resolve()
# Go up to prototype/: pages -> src -> template_dashboard -> prototype
prototype_dir = current_file.parent.parent.parent.parent
sys.path.append(str(prototype_dir))

# Import dashboard components
from dashboard_components import (
    create_tqdcs_spider_chart,
    create_optimal_split_chart,
    display_tqdcs_score_card,
    apply_metric_pill_styles,
    display_metric_pill,
    get_category_display_name
)

# Import utilities
from utils import load_css_files
from data_loader import load_supplier_data

# Page configuration
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title="Detailed Dashboard - PurchasingOrganization RFQ Analysis",
    page_icon=PAGE_ICON_PATH,
    layout="wide"
)

# Apply SAP SVG as app logo if available
try:
    st.logo(SAP_SVG_PATH)
except Exception:
    pass

# Load CSS
css_files = [
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "variables.css"),
    os.path.join(os.path.dirname(__file__), "..", "..", "static", "styles", "style.css"),
]
load_css_files(css_files)

# Inject metric pill styles
apply_metric_pill_styles()

# Load data (pass normalized weights if available to refresh recommendation/split)
weights_for_compare = st.session_state.get('tqdcs_weights')
with st.spinner("Loading analyses and computing insights. Please wait..."):
    supplier1_data, supplier2_data, comparison = load_supplier_data(
        tqdcs_weights=weights_for_compare,
        # Metrics can be skipped here to reduce tokens if desired
        generate_metrics=True,
        generate_strengths_weaknesses=True,
        generate_recommendation_and_split=True,
    )

if not all([supplier1_data, supplier2_data, comparison]):
    st.error("Failed to load analysis data. Please check the knowledge graph files.")
    st.stop()

# ===== EXACT CONTENT FROM render_tqdcs_tab() METHOD STARTS HERE =====

st.markdown('<h2 class="section-header">Detailed Analysis and Optimal Split Strategy</h2>', unsafe_allow_html=True)
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
st.markdown("## Detailed Analysis")
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Get TQDCS scores
tqdcs1 = supplier1_data.get('tqdcs', {}).get('tqdcs_scores', {})
tqdcs2 = supplier2_data.get('tqdcs', {}).get('tqdcs_scores', {})

col1, col2 = st.columns([3, 2])

with col1:
    # Spider chart
    fig = create_tqdcs_spider_chart(
        tqdcs1, tqdcs2,
        supplier1_data['supplier_name'],
        supplier2_data['supplier_name']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # --- RQDCE category toggles + Apply button ---
    # These toggles control which categories are considered. Disabled = weight 0.
    # When applied, enabled categories are equally weighted.
    st.markdown("### Score Toggles")

    # Determine current enabled categories from existing weights (if any)
    _weights_existing = st.session_state.get('tqdcs_weights', None)
    def _is_enabled(cat: str) -> bool:
        try:
            if isinstance(_weights_existing, dict):
                return float(_weights_existing.get(cat, 0.0)) > 0.0
        except Exception:
            pass
        return True  # default enabled when no prior state

    en_technology = st.toggle(get_category_display_name('technology'), value=_is_enabled('technology'), key="en_technology")
    en_quality = st.toggle(get_category_display_name('quality'), value=_is_enabled('quality'), key="en_quality")
    en_delivery = st.toggle(get_category_display_name('delivery'), value=_is_enabled('delivery'), key="en_delivery")
    en_cost = st.toggle(get_category_display_name('cost'), value=_is_enabled('cost'), key="en_cost")
    en_sustainability = st.toggle(get_category_display_name('sustainability'), value=_is_enabled('sustainability'), key="en_sustainability")

    enabled_map = {
        'technology': bool(en_technology),
        'quality': bool(en_quality),
        'delivery': bool(en_delivery),
        'cost': bool(en_cost),
        'sustainability': bool(en_sustainability),
    }
    enabled_count = sum(1 for v in enabled_map.values() if v)
    st.caption(f"Enabled categories: {enabled_count}/5")

    # Apply button deactivated - toggles are for display only
    if st.button("Apply", key="apply_tqdcs_toggles"):
        if enabled_count <= 0:
            st.error("Please enable at least one TQDCS category.")
        else:
            # Equal weights across enabled categories, zero for disabled
            per_cat = 100.0 / float(enabled_count)
            weights_norm = {k: (per_cat if enabled_map[k] else 0.0) for k in enabled_map.keys()}
            st.session_state['tqdcs_weights'] = weights_norm

            # Compute equal-weighted averages over enabled categories
            def _equal_weight_avg(tqdcs_scores: dict, enabled: dict) -> float:
                cats = [c for c, on in enabled.items() if on]
                if not cats:
                    return 0.0
                total = 0.0
                for cat in cats:
                    score = 0.0
                    if cat in tqdcs_scores:
                        score = float(tqdcs_scores[cat].get('score', 0) or 0)
                    total += score
                return round(total / float(len(cats)), 2)

            st.session_state['tqdcs_weighted_avg1'] = _equal_weight_avg(tqdcs1, enabled_map)
            st.session_state['tqdcs_weighted_avg2'] = _equal_weight_avg(tqdcs2, enabled_map)
            st.success("Applied TQDCS toggles with equal weighting. Recomputing insights...")

            # Rerun to refresh UI and force recompute of comparison (override cache)
            st.session_state['force_refresh_comparison'] = True
            st.rerun()

# ===== Score Analysis (below the top columns) =====
st.markdown("### Score Analysis")

# Overall scores (use weighted values if applied, else original averages)
avg1_default = supplier1_data.get('tqdcs', {}).get('overall_assessment', {}).get('average_score', 0)
avg2_default = supplier2_data.get('tqdcs', {}).get('overall_assessment', {}).get('average_score', 0)
avg1 = st.session_state.get('tqdcs_weighted_avg1', avg1_default)
avg2 = st.session_state.get('tqdcs_weighted_avg2', avg2_default)

pill_col1, pill_col2 = st.columns(2)

with pill_col1:
    st.markdown(f"### SupplierA")
    if avg1 > avg2 and avg2 > 0:
        delta_text = "Leading"
        status = "positive"
    elif avg1 < avg2 and avg2 > 0:
        percentage_diff = ((avg2 - avg1) / avg2) * 100
        delta_text = f"-{percentage_diff:.1f}% lower"
        status = "negative"
    else:
        delta_text = "Tied" if avg1 == avg2 else None
        status = "neutral"

    display_metric_pill(
        label="Overall Score",
        value=f"{avg1}/5" if isinstance(avg1, (int, float)) else str(avg1),
        vendor=None,
        delta=delta_text,
        status=status,
        icon=None
    )

with pill_col2:
    st.markdown(f"### SupplierB")
    if avg2 > avg1 and avg1 > 0:
        delta_text = "Leading"
        status = "positive"
    elif avg2 < avg1 and avg1 > 0:
        percentage_diff = ((avg1 - avg2) / avg1) * 100
        delta_text = f"-{percentage_diff:.1f}% lower"
        status = "negative"
    else:
        delta_text = "Tied" if avg1 == avg2 else None
        status = "neutral"

    display_metric_pill(
        label="Overall Score",
        value=f"{avg2}/5" if isinstance(avg2, (int, float)) else str(avg2),
        vendor=None,
        delta=delta_text,
        status=status,
        icon=None
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Cost & Delivery Snapshot using forecast volumes and estimated product cost
# st.markdown("### Cost & Delivery Snapshot")
# snap_col1, snap_col2 = st.columns(2)

# with snap_col1:
#     cost1 = supplier1_data.get('cost', {}) or {}
#     unit1 = float((cost1.get('unit_costs', {}) or {}).get('price_per_unit') or 0)
#     vols1 = cost1.get('volume_forecast', []) or []
#     total_units1 = 0.0
#     for vf in vols1:
#         try:
#             total_units1 += float(vf.get('quantity') or 0)
#         except Exception:
#             pass
#     est1 = cost1.get('estimated_total_product_cost') or {}
#     est_amt1 = est1.get('amount') if isinstance(est1, dict) else None
#     try:
#         est_amt1_val = float(est_amt1) if est_amt1 is not None else 0.0
#     except Exception:
#         est_amt1_val = 0.0
#     if not est_amt1_val and unit1 and total_units1:
#         est_amt1_val = unit1 * total_units1
#     st.markdown(f"#### {supplier1_data['supplier_name']}")
#     display_metric_pill(
#         label="Estimated Total Product Cost",
#         value=f"{est_amt1_val:,.2f}€",
#         vendor="forecast × unit price",
#         delta=None,
#         status="success" if est_amt1_val else "warning",
#         icon=None,
#     )
#     st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
#     display_metric_pill(
#         label="Forecast Quantity",
#         value=f"{total_units1:,.0f} units",
#         vendor=f"{len(vols1)} timeframe(s)",
#         delta=None,
#         status="info",
#         icon=None,
#     )

# with snap_col2:
#     cost2 = supplier2_data.get('cost', {}) or {}
#     unit2 = float((cost2.get('unit_costs', {}) or {}).get('price_per_unit') or 0)
#     vols2 = cost2.get('volume_forecast', []) or []
#     total_units2 = 0.0
#     for vf in vols2:
#         try:
#             total_units2 += float(vf.get('quantity') or 0)
#         except Exception:
#             pass
#     est2 = cost2.get('estimated_total_product_cost') or {}
#     est_amt2 = est2.get('amount') if isinstance(est2, dict) else None
#     try:
#         est_amt2_val = float(est_amt2) if est_amt2 is not None else 0.0
#     except Exception:
#         est_amt2_val = 0.0
#     if not est_amt2_val and unit2 and total_units2:
#         est_amt2_val = unit2 * total_units2
#     st.markdown(f"#### {supplier2_data['supplier_name']}")
#     display_metric_pill(
#         label="Estimated Total Product Cost",
#         value=f"{est_amt2:,.2f}€" if est_amt2 else f"{est_amt2_val:,.2f}€",
#         vendor="forecast × unit price",
#         delta=None,
#         status="success" if (est_amt2 or est_amt2_val) else "warning",
#         icon=None,
#     )
#     st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
#     display_metric_pill(
#         label="Forecast Quantity",
#         value=f"{total_units2:,.0f} units",
#         vendor=f"{len(vols2)} timeframe(s)",
#         delta=None,
#         status="info",
#         icon=None,
#     )

# Expandable details for each category
st.markdown("### Detailed Assessment")

# Category icons for visual cues on pills
category_icons = {
    'technology': None,
    'quality': None,
    'delivery': None,
    'cost': None,
    'sustainability': None,
}

category_order = ['technology', 'quality', 'delivery', 'cost', 'sustainability']

for category in category_order:
    display_label = get_category_display_name(category)
    with st.expander(f"{display_label} Analysis"):
        # First level: Show scores for both suppliers
        col1, col2 = st.columns(2)
        
        with col1:
            if category in tqdcs1:
                score = tqdcs1[category].get('score', 0)
                
                # Map score to remark and pill status
                if score >= 4.5:
                    remark = "Excellent 🟢"
                    pill_status = "positive"
                elif score >= 4.0:
                    remark = "Good 🟢"
                    pill_status = "positive"
                elif score >= 3.0:
                    remark = "Adequate 🟡"
                    pill_status = "info"
                else:
                    remark = "Poor 🔴"
                    pill_status = "negative"

                # Render as metric pill (Supplier, Category, Score, Remark)
                display_metric_pill(
                    label=display_label,
                    value=f"{score}/5",
                    vendor="SupplierA",
                    delta=remark,
                    status=pill_status,
                    icon=category_icons.get(category)
                )
        
        with col2:
            if category in tqdcs2:
                score = tqdcs2[category].get('score', 0)
                
                # Map score to remark and pill status
                if score >= 4.5:
                    remark = "Excellent 🟢"
                    pill_status = "positive"
                elif score >= 4.0:
                    remark = "Good 🟢"
                    pill_status = "positive"
                elif score >= 3.0:
                    remark = "Adequate 🟡"
                    pill_status = "info"
                else:
                    remark = "Poor 🔴"
                    pill_status = "negative"

                # Render as metric pill (Supplier, Category, Score, Remark)
                display_metric_pill(
                    label=display_label,
                    value=f"{score}/5",
                    vendor="SupplierB",
                    delta=remark,
                    status=pill_status,
                    icon=category_icons.get(category)
                )

        # Small space between the pills and the detailed analysis expander
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # Second level: Nested expandable for detailed information
        with st.expander("View Detailed Analysis"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"**SupplierA**")
                if category in tqdcs1:
                    display_tqdcs_score_card(category, tqdcs1[category], show_details=False)
            
            with detail_col2:
                st.markdown(f"**SupplierB**")
                if category in tqdcs2:
                    display_tqdcs_score_card(category, tqdcs2[category], show_details=False)


# ===== Optimal Split (moved from separate page) =====
# Inserted above the Detailed TQDCS Assessment section as requested
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
st.markdown('---')
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
st.markdown('## Optimal Split Strategy')
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
split = comparison.get('optimal_split', {})

col1_split, col2_split = st.columns([2, 1])

with col1_split:
    fig_split = create_optimal_split_chart(split)
    st.plotly_chart(fig_split, use_container_width=True)

with col2_split:
    st.markdown("### Split Analysis")

    p1 = split.get('supplier1_percentage', 0)
    p2 = split.get('supplier2_percentage', 0)

    # Supplier 1 pill (top)
    display_metric_pill(
        label=split.get('supplier1_name', 'Supplier 1'),
        value=f"{p1}%",
        vendor=("Primary" if p1 > 50 else "Secondary"),
        delta=None,
        status="positive" if p1 > 50 else "info",
        icon=None,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Supplier 2 pill (bottom)
    display_metric_pill(
        label=split.get('supplier2_name', 'Supplier 2'),
        value=f"{p2}%",
        vendor=("Primary" if p2 > 50 else "Secondary"),
        delta=None,
        status="positive" if p2 > 50 else "info",
        icon=None,
    )

# Split rationale
if split.get('rationale'):
    st.info(split['rationale'])

# Implementation phases
if split.get('implementation_phases'):
    st.markdown("### Delivery Roadmap")

    phases_data = []
    for phase in split['implementation_phases']:
        phases_data.append({
            'Phase': phase.get('phase', ''),
            'Description': phase.get('description', ''),
            'Timeline': phase.get('timeline', '')
        })

    if phases_data:
        df = pd.DataFrame(phases_data)
        st.dataframe(df, use_container_width=True)
