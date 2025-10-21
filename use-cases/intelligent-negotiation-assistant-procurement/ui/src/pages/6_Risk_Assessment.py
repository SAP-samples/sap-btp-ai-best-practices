"""
Risk Assessment Page
This page contains the EXACT content from the render_risk_tab method in enhanced_dashboard.py
"""

import streamlit as st
import os
from pathlib import Path
import sys
from typing import Dict, Any, List
import plotly.graph_objects as go  # For radar (spider) plots

# Add parent directory for imports
current_file = Path(__file__).resolve()
# Go up to prototype/: pages -> src -> template_dashboard -> prototype
prototype_dir = current_file.parent.parent.parent.parent
sys.path.append(str(prototype_dir))

# Import dashboard components
from dashboard_components import (
    display_risk_card,
    create_risk_heatmap,
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
    page_title="Risk Assessment - RFQ Analysis",
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

# Inject pill styles for metric pills
apply_metric_pill_styles()

# Load data (no comparison needed on this page)
supplier1_data, supplier2_data, comparison = load_supplier_data(need_comparison=False)

if not supplier1_data or not supplier2_data:
    st.error("Failed to load analysis data. Please check the knowledge graph files.")
    st.stop()

# ===== HELPER METHOD FROM enhanced_dashboard.py =====

def _group_risks_by_category(risks_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group risks by category while preserving all risk data
    
    Args:
        risks_list: List of risk dictionaries
        
    Returns:
        Dictionary with risks grouped by category and sorted by severity within each category
    """
    grouped = {}
    for risk in risks_list:
        category = risk.get('category', 'Operational')
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(risk)
    
    # Sort within each category by severity (High → Medium → Low)
    for category in grouped:
        grouped[category].sort(
            key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}.get(
                x.get('severity', 'Medium')
            )
        )
    return grouped

# ===== EXACT CONTENT FROM render_risk_tab() METHOD STARTS HERE =====

st.markdown('<h2 class="section-header">Risk Assessment</h2>', unsafe_allow_html=True)
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

risks1 = supplier1_data.get('risks', {})
risks2 = supplier2_data.get('risks', {})

# Risk summary pills organized by supplier (two columns)
left_col, right_col = st.columns(2)

with left_col:
    st.markdown(f"### SupplierA")
    high1 = risks1.get('risk_summary', {}).get('high_risks_count', 0)
    display_metric_pill(
        label="High Risks",
        value=str(high1),
        vendor=None,
        delta=None,
        status="negative",
        icon=None,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    medium1 = risks1.get('risk_summary', {}).get('medium_risks_count', 0)
    display_metric_pill(
        label="Medium Risks",
        value=str(medium1),
        vendor=None,
        delta=None,
        status="negative",
        icon=None,
    )

with right_col:
    st.markdown(f"### SupplierB")
    high2 = risks2.get('risk_summary', {}).get('high_risks_count', 0)
    display_metric_pill(
        label="High Risks",
        value=str(high2),
        vendor=None,
        delta=None,
        status="negative",
        icon=None,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    medium2 = risks2.get('risk_summary', {}).get('medium_risks_count', 0)
    display_metric_pill(
        label="Medium Risks",
        value=str(medium2),
        vendor=None,
        delta=None,
        status="negative",
        icon=None,
    )

st.markdown("---")

# Risk heatmaps
col1, col2 = st.columns(2)

with col1:
    if risks1.get('risk_matrix'):
        fig1 = create_risk_heatmap(
            risks1['risk_matrix'],
            "SupplierA Risk Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if risks2.get('risk_matrix'):
        fig2 = create_risk_heatmap(
            risks2['risk_matrix'],
            "SupplierB Risk Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ===== Risk Supplier Profile (Spider Chart with Mocked Data) =====
# This section presents mocked supplier profile ratings as a radar chart.
st.markdown("### Supplier Risk Management")
st.caption("Note: The following supplier profile ratings are mocked data for demonstration purposes.")

# Define categories and mocked A/B/C ratings for both suppliers
profile_categories = [
    "S-Rating (Sustainability)",
    "L-Rating (Logistics)",
    "Q-Rating (Quality)",
    "F-Rating (Financial)",
    "FE-Rating (R&D)",
    "Cyber Security",
]

# Mapping where A is best, then B, then C
rating_to_score = {"A": 3, "B": 2, "C": 1}

# Mocked ratings from spec
supplier1_ratings = ["A", "C", "C", "A", "B", "C"]
supplier2_ratings = ["B", "A", "A", "C", "B", "A"]

# Convert ratings to numeric scores for the radar plot
s1_scores = [rating_to_score.get(r, 0) for r in supplier1_ratings]
s2_scores = [rating_to_score.get(r, 0) for r in supplier2_ratings]

# Build a radar (spider) chart similar in style to the TQDCS spider chart
fig_profile = go.Figure()

fig_profile.add_trace(go.Scatterpolar(
    r=s1_scores,
    theta=profile_categories,
    fill='toself',
    # Use Plotly default blue for Supplier 1
    line=dict(color='#1f77b4'),
    fillcolor='rgba(31, 119, 180, 0.20)',
    opacity=1.0,
    name="SupplierA",
))

fig_profile.add_trace(go.Scatterpolar(
    r=s2_scores,
    theta=profile_categories,
    fill='toself',
    # Use Plotly default orange for Supplier 2
    line=dict(color='#ff7f0e'),
    fillcolor='rgba(255, 127, 14, 0.20)',
    opacity=1.0,
    name="SupplierB",
))

fig_profile.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 3],   # A > B > C mapped to 3 > 2 > 1
            dtick=1,
        ),
    ),
    showlegend=True,
    margin=dict(l=10, r=10, t=10, b=10),
)

st.plotly_chart(fig_profile, use_container_width=True)

# Detailed risks grouped by category
st.markdown("### Risk Details by Category")

# Group risks for both suppliers
grouped_risks1 = _group_risks_by_category(risks1.get('risks', [])) if risks1.get('risks') else {}
grouped_risks2 = _group_risks_by_category(risks2.get('risks', [])) if risks2.get('risks') else {}

# Define category display order
category_order = [
    "Technical",
    "Supply Chain", 
    "Operational",
    "Legal",
    "Financial",
    "Quality",
    "Strategic"
]

# Get all categories present in the data
all_categories = set(grouped_risks1.keys()) | set(grouped_risks2.keys())

# Display categories in order, then any additional categories found
ordered_categories = [cat for cat in category_order if cat in all_categories]
ordered_categories.extend([cat for cat in all_categories if cat not in category_order])

for category in ordered_categories:
    # Count total risks in this category
    total_risks = len(grouped_risks1.get(category, [])) + len(grouped_risks2.get(category, []))
    
    if total_risks > 0:
        with st.expander(f"{category} Risks"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**SupplierA**")
                if category in grouped_risks1:
                    for risk in grouped_risks1[category]:
                        display_risk_card(risk, show_mitigation=True)
                else:
                    st.info("No risks in this category")
            
            with col2:
                st.markdown(f"**SupplierB**")
                if category in grouped_risks2:
                    for risk in grouped_risks2[category]:
                        display_risk_card(risk, show_mitigation=True)
                else:
                    st.info("No risks in this category")

