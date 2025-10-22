"""
Detailed Comparison Page
This page contains the EXACT content from the render_comparison_tab method in enhanced_dashboard.py
"""

import streamlit as st
import os
from pathlib import Path
import sys
import html
import pandas as pd

# Add parent directory for imports
current_file = Path(__file__).resolve()
# Go up to prototype/: pages -> src -> template_dashboard -> prototype
prototype_dir = current_file.parent.parent.parent.parent
sys.path.append(str(prototype_dir))

# Import helpers for category display names
from dashboard_components import get_category_display_name

# Import utilities
from utils import load_css_files
from data_loader import load_supplier_data

# Page configuration
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title="Detailed Comparison - RFQ Analysis",
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

# Load data
# We keep the Comparison Matrix unaffected by TQDCS toggles/weights, but update
# Final Recommendation and Strengths & Weaknesses when toggles are applied.
weights = st.session_state.get('tqdcs_weights')

# 1) Baseline comparison (provides metrics)
supplier1_data, supplier2_data, comparison_base = load_supplier_data(
    generate_metrics=True,
    generate_strengths_weaknesses=True,
    generate_recommendation_and_split=True,
)

# 2) If weights exist, recompute only recommendation/S&W with weights (no metrics)
comparison = comparison_base
if weights is not None:
    _, _, comparison_weighted = load_supplier_data(
        tqdcs_weights=weights,
        generate_metrics=False,  # do not regenerate metrics to keep them unaffected
        generate_strengths_weaknesses=True,
        generate_recommendation_and_split=True,
    )
    # Overlay weighted sections onto baseline metrics
    if isinstance(comparison_weighted, dict):
        comparison = {
            **(comparison_base or {}),
            **{  # only override the targeted sections
                'recommendation': comparison_weighted.get('recommendation', {}),
                'strengths_weaknesses': comparison_weighted.get('strengths_weaknesses', {}),
            }
        }

if not all([supplier1_data, supplier2_data, comparison]):
    st.error("Failed to load analysis data. Please check the knowledge graph files.")
    st.stop()

st.markdown('<h2 class="section-header">Detailed Comparison</h2>', unsafe_allow_html=True)
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# Comparison metrics
st.markdown("### Comparison Matrix")

metrics = comparison.get('comparison_metrics', [])
if metrics:
    # Inject compact table CSS once to guarantee four fixed columns within expanders
    st.markdown(
        """
        <style>
        table.comparison-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
        table.comparison-table th, table.comparison-table td { 
            border-bottom: 1px solid #e5e7eb; padding: 8px 10px; vertical-align: top; word-wrap: break-word;
        }
        table.comparison-table thead th { background: #f8fafc; font-weight: 600; }
        table.comparison-table .win { background: #eef8f0; color: #12803c; border-left: 4px solid #188944; padding: 6px 8px; border-radius: 4px; }
        table.comparison-table .caption { color: #6b7280; font-size: 12px; }
        table.comparison-table .note { color: #6b7280; font-size: 12px; margin-top: 4px; }
        table.comparison-table col.metric { width: 35%; }
        table.comparison-table col.s1 { width: 27%; }
        table.comparison-table col.s2 { width: 27%; }
        table.comparison-table col.winner { width: 11%; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Add supplier names to metrics for display
    for metric in metrics:
        metric['supplier1_name'] = supplier1_data['supplier_name']
        metric['supplier2_name'] = supplier2_data['supplier_name']
    
    # Group by category
    categories = {}
    for metric in metrics:
        cat = metric.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(metric)
    
    # Display by category using a fixed HTML table to avoid column wrapping
    for category, cat_metrics in categories.items():
        display_category = get_category_display_name(category)
        with st.expander(f"{display_category} Metrics ({len(cat_metrics)} parameters)"):
            # Build CSV rows for this category's table (for download button)
            csv_rows = []

            def make_csv_cell(value, sources):
                """Compose plain text cell with value and inline references for CSV."""
                try:
                    safe_value = str(value) if value is not None else "N/A"
                    parts = []
                    if isinstance(sources, list):
                        for src in sources:
                            fname = str(src.get('filename', 'Unknown'))
                            chunk = str(src.get('chunk_id', ''))
                            parts.append(f"{fname}:{chunk}")
                    return f"{safe_value} | {' | '.join(parts)}" if parts else safe_value
                except Exception:
                    return "N/A"

            table_html = [
                '<table class="comparison-table">',
                '<colgroup><col class="metric"/><col class="s1"/><col class="s2"/><col class="winner"/></colgroup>',
                '<thead>',
                '<tr>'
                f'<th>Metric</th>'
                f'<th>SupplierA</th>'
                f'<th>SupplierB</th>'
                f'<th>Winner</th>'
                '</tr>',
                '</thead>',
                '<tbody>'
            ]

            for m in cat_metrics:
                supplier1_name = m.get('supplier1_name', supplier1_data['supplier_name'])
                supplier2_name = m.get('supplier2_name', supplier2_data['supplier_name'])
                winner = m.get('winner', '')

                # Metric cell content
                metric_title = html.escape(str(m.get('metric', 'Unknown')))
                importance = m.get('importance', 'Medium')
                importance_html = ''
                if importance == 'Critical':
                    importance_html = '<div class="caption">⚠️ Critical</div>'
                elif importance == 'High':
                    importance_html = '<div class="caption">❗ High importance</div>'

                notes_html = ''
                if m.get('comparison_notes'):
                    notes_html = f"<div class='note'>💡 {html.escape(str(m['comparison_notes']))}</div>"

                sources_html = ''
                if m.get('sources') and m['sources'].get('supplier1'):
                    files = ", ".join(s.get('filename', '') for s in m['sources']['supplier1'][:2])
                    sources_html = f"<div class='caption'>Sources: {html.escape(files)}</div>"

                metric_cell = f"<div>{metric_title}</div>{importance_html}{notes_html}{sources_html}"

                # Supplier value cells with winner highlight
                v1 = html.escape(str(m.get('supplier1_value', 'N/A')))
                v2 = html.escape(str(m.get('supplier2_value', 'N/A')))

                # Collect sources for CSV download (raw dicts, not escaped)
                s1_sources = []
                s2_sources = []
                if isinstance(m.get('sources'), dict):
                    s1_sources = m['sources'].get('supplier1', []) or []
                    s2_sources = m['sources'].get('supplier2', []) or []

                if winner == supplier1_name:
                    v1_html = f"<div class='win'>✓ {v1}</div>"
                else:
                    v1_html = f"<div>{v1}</div>"

                if winner == supplier2_name:
                    v2_html = f"<div class='win'>✓ {v2}</div>"
                else:
                    v2_html = f"<div>{v2}</div>"

                # Winner cell text
                if winner == 'Tie':
                    win_cell = '<div>Tie</div>'
                elif winner:
                    win_cell = f"<div>→ {html.escape(str(winner))[:12]}</div>"
                else:
                    win_cell = '<div>-</div>'

                table_html.extend([
                    '<tr>',
                    f'<td>{metric_cell}</td>',
                    f'<td>{v1_html}</td>',
                    f'<td>{v2_html}</td>',
                    f'<td>{win_cell}</td>',
                    '</tr>'
                ])

                # Append row for CSV with plain text values and inline references
                csv_rows.append({
                    'Metric': str(m.get('metric', 'Unknown')),
                    supplier1_data['supplier_name']: make_csv_cell(m.get('supplier1_value', 'N/A'), s1_sources),
                    supplier2_data['supplier_name']: make_csv_cell(m.get('supplier2_value', 'N/A'), s2_sources),
                    'Winner': str(winner or ''),
                    'Importance': str(m.get('importance', '')),
                    'Notes': str(m.get('comparison_notes', '')),
                })

            table_html.extend(['</tbody>', '</table>'])
            st.markdown("".join(table_html), unsafe_allow_html=True)

            # Render per-category CSV download button mirroring Offers_Comparison
            try:
                df = pd.DataFrame(csv_rows)
                safe_category = "".join(
                    ch.lower() if ch.isalnum() else "_"
                    for ch in str(display_category)
                ).strip("_") or "category"
                st.download_button(
                    label=f"Download {display_category} Matrix as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{safe_category}_comparison_matrix.csv",
                    mime="text/csv",
                    key=f"download_{safe_category}_matrix_csv",
                )
            except Exception:
                # Silently ignore any CSV generation issues to not break UI
                pass

# Strengths and weaknesses
st.markdown("### Strengths & Weaknesses Analysis")

sw = comparison.get('strengths_weaknesses', {})

col1, col2 = st.columns(2)

with col1:
    if sw.get('supplier1'):
        st.markdown(f"#### {sw['supplier1'].get('name', 'Supplier 1')}")
        
        st.markdown("**Strengths:**")
        for strength in sw['supplier1'].get('strengths', [])[:5]:
            st.success(f"✓ {strength.get('title', '')}")
            if strength.get('description'):
                st.caption(strength['description'])
        
        st.markdown("**Weaknesses:**")
        for weakness in sw['supplier1'].get('weaknesses', [])[:5]:
            st.error(f"✗ {weakness.get('title', '')}")
            if weakness.get('description'):
                st.caption(weakness['description'])

with col2:
    if sw.get('supplier2'):
        st.markdown(f"#### {sw['supplier2'].get('name', 'Supplier 2')}")
        
        st.markdown("**Strengths:**")
        for strength in sw['supplier2'].get('strengths', [])[:5]:
            st.success(f"✓ {strength.get('title', '')}")
            if strength.get('description'):
                st.caption(strength['description'])
        
        st.markdown("**Weaknesses:**")
        for weakness in sw['supplier2'].get('weaknesses', [])[:5]:
            st.error(f"✗ {weakness.get('title', '')}")
            if weakness.get('description'):
                st.caption(weakness['description'])


# Final recommendation
rec = comparison.get('recommendation', {})
if rec:
    # st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
    st.markdown("### Final Recommendation")
    
    preferred = rec.get('preferred_supplier', 'N/A')
    confidence = rec.get('confidence_level', 'Medium')
    
    st.markdown(f"**Recommended Supplier:** {preferred}")
    st.markdown(f"**Confidence Level:** {confidence}")
    
    if rec.get('key_reasons'):
        st.markdown("**Key Reasons:**")
        for reason in rec['key_reasons']:
            st.write(f"• {reason}")
    
    if rec.get('conditions'):
        st.markdown("**Conditions:**")
        for condition in rec['conditions']:
            st.write(f"⚠️ {condition}")
    
    st.markdown('</div>', unsafe_allow_html=True)

