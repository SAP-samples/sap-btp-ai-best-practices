"""
Cost Breakdown Page
This page contains the EXACT content from the render_cost_tab method in enhanced_dashboard.py
"""

import streamlit as st
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
    create_source_badge,
    create_cost_breakdown_chart,
    display_metric_pill,
    apply_metric_pill_styles,
)

# Import utilities
from utils import load_css_files
from data_loader import load_supplier_data

# Page configuration
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title="Cost Breakdown - RFQ Analysis",
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

# Inject pill styles for consistent pill rendering
apply_metric_pill_styles()

# Load data (no comparison needed on this page)
supplier1_data, supplier2_data, comparison = load_supplier_data(need_comparison=False)

if not supplier1_data or not supplier2_data:
    st.error("Failed to load analysis data. Please check the knowledge graph files.")
    st.stop()

# ===== EXACT CONTENT FROM render_cost_tab() METHOD STARTS HERE =====

st.markdown('<h2 class="section-header">Cost Breakdown Analysis</h2>', unsafe_allow_html=True)
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
cost1 = supplier1_data.get('cost', {})
cost2 = supplier2_data.get('cost', {})

# Cost metric pills in a 2x2 grid
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"### SupplierA")
    total1 = cost1.get('total_cost', {}).get('amount') or 0
    display_metric_pill(
        label="Project Cost",
        value=f"€{total1:,}",
        # Show the cost description within the pill; remove supplier name
        vendor=cost1.get('total_cost', {}).get('description'),
        delta=None,
        status="info",
        icon=None,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    unit1 = cost1.get('unit_costs', {}).get('price_per_unit') or 0
    display_metric_pill(
        label="Product Cost",
        value=f"€{unit1:.2f}",
        # Replace supplier name with volume assumptions for product pricing context
        vendor=cost1.get('unit_costs', {}).get('volume_assumptions'),
        delta=None,
        status="info",
        icon=None,
    )
    # Compute and display Total Product Cost using forecast volumes × unit price
    vol_list1 = cost1.get('volume_forecast', []) or []
    total_units1 = 0.0
    for vf in vol_list1:
        try:
            total_units1 += float(vf.get('quantity') or 0)
        except Exception:
            pass
    est_obj1 = cost1.get('estimated_total_product_cost') or {}
    est_amount1 = est_obj1.get('amount') if isinstance(est_obj1, dict) else None
    try:
        est_amount1_val = float(est_amount1) if est_amount1 is not None else 0.0
    except Exception:
        est_amount1_val = 0.0
    computed_est1 = est_amount1_val if est_amount1_val else (float(unit1) * total_units1 if unit1 and total_units1 else 0.0)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    display_metric_pill(
        label="Total Product Cost",
        value=f"{computed_est1:,.2f}€",
        vendor="forecast quantities × unit price",
        delta=None,
        status="success" if computed_est1 else "warning",
        icon=None,
    )
    if cost1.get('total_cost', {}).get('sources'):
        create_source_badge(cost1['total_cost']['sources'])
    # Show sources for Total Product Cost if available, else use unit cost + all volume sources
    sources_est1 = []
    if isinstance(est_obj1, dict) and est_obj1.get('sources'):
        sources_est1 = est_obj1.get('sources') or []
    else:
        # Fallback: combine unit cost sources and all volume sources
        unit_sources1 = (cost1.get('unit_costs', {}) or {}).get('sources') or []
        vol_sources1 = []
        for vf in vol_list1:
            for s in vf.get('sources', []) or []:
                vol_sources1.append(s)
        sources_est1 = [*unit_sources1, *vol_sources1]
    if sources_est1:
        create_source_badge(sources_est1)

with c2:
    st.markdown(f"### SupplierB")
    total2 = cost2.get('total_cost', {}).get('amount') or 0
    display_metric_pill(
        label="Project Cost",
        value=f"€{total2:,}",
        # Show the cost description within the pill; remove supplier name
        vendor=cost2.get('total_cost', {}).get('description'),
        delta=None,
        status="info",
        icon=None,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    unit2 = cost2.get('unit_costs', {}).get('price_per_unit') or 0
    display_metric_pill(
        label="Product Cost",
        value=f"€{unit2:.2f}",
        # Replace supplier name with volume assumptions for product pricing context
        vendor=cost2.get('unit_costs', {}).get('volume_assumptions'),
        delta=None,
        status="info",
        icon=None,
    )
    # Compute and display Total Product Cost using forecast volumes × unit price
    vol_list2 = cost2.get('volume_forecast', []) or []
    total_units2 = 0.0
    for vf in vol_list2:
        try:
            total_units2 += float(vf.get('quantity') or 0)
        except Exception:
            pass
    est_obj2 = cost2.get('estimated_total_product_cost') or {}
    est_amount2 = est_obj2.get('amount') if isinstance(est_obj2, dict) else None
    try:
        est_amount2_val = float(est_amount2) if est_amount2 is not None else 0.0
    except Exception:
        est_amount2_val = 0.0
    computed_est2 = est_amount2_val if est_amount2_val else (float(unit2) * total_units2 if unit2 and total_units2 else 0.0)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    display_metric_pill(
        label="Total Product Cost",
        value=f"{computed_est2:,.2f}€",
        vendor="forecast quantities × unit price",
        delta=None,
        status="success" if computed_est2 else "warning",
        icon=None,
    )
    if cost2.get('total_cost', {}).get('sources'):
        create_source_badge(cost2['total_cost']['sources'])
    # Show sources for Total Product Cost if available, else use unit cost + all volume sources
    sources_est2 = []
    if isinstance(est_obj2, dict) and est_obj2.get('sources'):
        sources_est2 = est_obj2.get('sources') or []
    else:
        unit_sources2 = (cost2.get('unit_costs', {}) or {}).get('sources') or []
        vol_sources2 = []
        for vf in vol_list2:
            for s in vf.get('sources', []) or []:
                vol_sources2.append(s)
        sources_est2 = [*unit_sources2, *vol_sources2]
    if sources_est2:
        create_source_badge(sources_est2)

st.markdown("---")

# Cost breakdown charts
col1, col2 = st.columns(2)

with col1:
    if cost1.get('cost_breakdown'):
        fig1 = create_cost_breakdown_chart(
            cost1['cost_breakdown'],
            "SupplierA Cost Breakdown"
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if cost2.get('cost_breakdown'):
        fig2 = create_cost_breakdown_chart(
            cost2['cost_breakdown'],
            "SupplierB Cost Breakdown"
        )
        st.plotly_chart(fig2, use_container_width=True)

# Cost breakdown details
with st.expander("Cost Breakdown Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**SupplierA**")
        if cost1.get('cost_breakdown'):
            for item in cost1['cost_breakdown']:
                category = item.get('category', 'Unknown')
                amount = item.get('amount') or 0
                percentage = item.get('percentage_of_total') or 0
                description = item.get('description', 'No description available')
                
                # Display category information
                st.markdown(f"**{category}**")
                st.write(f"Amount: €{amount:,}")
                st.write(f"Percentage: {percentage:.1f}%")
                st.write(f"Description: {description}")
                
                # Display sources if available
                if item.get('sources'):
                    create_source_badge(item['sources'])  # Show all sources
                
                st.markdown("---")
    
    with col2:
        st.markdown(f"**SupplierB**")
        if cost2.get('cost_breakdown'):
            for item in cost2['cost_breakdown']:
                category = item.get('category', 'Unknown')
                amount = item.get('amount') or 0
                percentage = item.get('percentage_of_total') or 0
                description = item.get('description', 'No description available')
                
                # Display category information
                st.markdown(f"**{category}**")
                st.write(f"Amount: €{amount:,}")
                st.write(f"Percentage: {percentage:.1f}%")
                st.write(f"Description: {description}")
                
                # Display sources if available
                if item.get('sources'):
                    create_source_badge(item['sources'])  # Show all sources
                
                st.markdown("---")

# Volume forecast and estimation details
with st.expander("Volume Forecast and Product Cost Estimation"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**SupplierA**")
        unit_price = float(cost1.get('unit_costs', {}).get('price_per_unit') or 0)
        if unit_price:
            st.write(f"Unit Price: {unit_price:.2f}€")
            if (cost1.get('unit_costs', {}) or {}).get('sources'):
                create_source_badge(cost1['unit_costs']['sources'])
        vols = cost1.get('volume_forecast', []) or []
        total_units = 0.0
        for vf in vols:
            timeframe = vf.get('timeframe', '—')
            qty = vf.get('quantity') or 0
            try:
                qty_val = float(qty)
            except Exception:
                qty_val = 0.0
            total_units += qty_val
            st.write(f"{timeframe}: {qty_val:,.0f} units")
            if vf.get('sources'):
                create_source_badge(vf['sources'])
        if vols:
            st.write(f"Total Forecast Quantity: {total_units:,.0f} units")
        # Estimated total product cost (detailed)
        est = cost1.get('estimated_total_product_cost') or {}
        est_amt = est.get('amount') if isinstance(est, dict) else None
        try:
            est_val = float(est_amt) if est_amt is not None else 0.0
        except Exception:
            est_val = 0.0
        if not est_val and unit_price and total_units:
            est_val = unit_price * total_units
        if est_val:
            st.markdown(f"**Estimated Total Product Cost: {est_val:,.2f}€**")
            est_sources = (est.get('sources') if isinstance(est, dict) else None) or []
            if not est_sources:
                # fallback combine sources
                est_sources = ((cost1.get('unit_costs', {}) or {}).get('sources') or [])
                for vf in vols:
                    for s in vf.get('sources', []) or []:
                        est_sources.append(s)
            if est_sources:
                create_source_badge(est_sources)
    with col2:
        st.markdown(f"**SupplierB**")
        unit_price = float(cost2.get('unit_costs', {}).get('price_per_unit') or 0)
        if unit_price:
            st.write(f"Unit Price: {unit_price:.2f}€")
            if (cost2.get('unit_costs', {}) or {}).get('sources'):
                create_source_badge(cost2['unit_costs']['sources'])
        vols = cost2.get('volume_forecast', []) or []
        total_units = 0.0
        for vf in vols:
            timeframe = vf.get('timeframe', '—')
            qty = vf.get('quantity') or 0
            try:
                qty_val = float(qty)
            except Exception:
                qty_val = 0.0
            total_units += qty_val
            st.write(f"{timeframe}: {qty_val:,.0f} units")
            if vf.get('sources'):
                create_source_badge(vf['sources'])
        if vols:
            st.write(f"Total Forecast Quantity: {total_units:,.0f} units")
        est = cost2.get('estimated_total_product_cost') or {}
        est_amt = est.get('amount') if isinstance(est, dict) else None
        try:
            est_val = float(est_amt) if est_amt is not None else 0.0
        except Exception:
            est_val = 0.0
        if not est_val and unit_price and total_units:
            est_val = unit_price * total_units
        if est_val:
            st.markdown(f"**Estimated Total Product Cost: {est_val:,.2f}€**")
            est_sources = (est.get('sources') if isinstance(est, dict) else None) or []
            if not est_sources:
                est_sources = ((cost2.get('unit_costs', {}) or {}).get('sources') or [])
                for vf in vols:
                    for s in vf.get('sources', []) or []:
                        est_sources.append(s)
            if est_sources:
                create_source_badge(est_sources)

# Cost details
with st.expander("Cost Dependencies and Business Context"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**SupplierA**")
        if cost1.get('cost_dependencies'):
            for dep in cost1['cost_dependencies']:
                st.write(f"• {dep.get('description', '')}")
                if dep.get('sources'):
                    create_source_badge(dep['sources'])  # Show all sources
    
    with col2:
        st.markdown(f"**SupplierB**")
        if cost2.get('cost_dependencies'):
            for dep in cost2['cost_dependencies']:
                st.write(f"• {dep.get('description', '')}")
                if dep.get('sources'):
                    create_source_badge(dep['sources'])  # Show all sources

# Opportunities
with st.expander("Cost Optimization Opportunities"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**SupplierA**")
        if cost1.get('opportunities'):
            for opp in cost1['opportunities']:
                st.success(f"💡 {opp.get('opportunity', '')}")
                if opp.get('approach'):
                    st.caption(f"Approach: {opp['approach']}")
    with col2:
        st.markdown(f"**SupplierB**")
        if cost2.get('opportunities'):
            for opp in cost2['opportunities']:
                st.success(f"💡 {opp.get('opportunity', '')}")
                if opp.get('approach'):
                    st.caption(f"Approach: {opp['approach']}")

