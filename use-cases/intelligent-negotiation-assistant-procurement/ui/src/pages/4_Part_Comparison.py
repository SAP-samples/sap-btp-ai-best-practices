"""
Part Comparison Page
This page shows the Parts Details view directly for streamlined comparison
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
    display_part_details_card,
    apply_metric_pill_styles,
    display_metric_pill
)

# Import utilities
from utils import load_css_files
from data_loader import load_supplier_data

# Page configuration
STATIC_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "static", "images")
PAGE_ICON_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo_square.png")
SAP_SVG_PATH = os.path.join(STATIC_IMAGES_DIR, "SAP_logo.svg")

st.set_page_config(
    page_title="Part Comparison - RFQ Analysis",
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

# Load data (no comparison needed on this page)
supplier1_data, supplier2_data, comparison = load_supplier_data(need_comparison=False)

if not supplier1_data or not supplier2_data:
    st.error("Failed to load analysis data. Please check the knowledge graph files.")
    st.stop()

# ===== PARTS COMPARISON =====

st.markdown('<h2 class="section-header">Parts Comparison</h2>', unsafe_allow_html=True)
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
# Get parts data (from LLM analysis)
parts1 = supplier1_data.get('parts', {}).get('parts', [])
parts2 = supplier2_data.get('parts', {}).get('parts', [])
# Get overall clutch system (system-level consolidated item)
overall1 = supplier1_data.get('parts', {}).get('overall_system') or supplier1_data.get('parts', {}).get('overall_clutch_system')
overall2 = supplier2_data.get('parts', {}).get('overall_system') or supplier2_data.get('parts', {}).get('overall_clutch_system')
# Get canonical parts data (from deterministic canonicalization)
parts1_canonical = supplier1_data.get('canonical_parts', {}).get('parts', [])
parts2_canonical = supplier2_data.get('canonical_parts', {}).get('parts', [])
concentrated1 = supplier1_data.get('parts', {}).get('concentrated_parts') or supplier1_data.get('concentrated_parts')
concentrated2 = supplier2_data.get('parts', {}).get('concentrated_parts') or supplier2_data.get('concentrated_parts')

# Summary metrics as metric pills
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### SupplierA")
    total1 = len(parts1)
    display_metric_pill(
        label="Parts",
        value=f"{total1}",
        vendor=None,
        status="info",
        icon=None
    )

with col2:
    st.markdown(f"### SupplierB")
    total2 = len(parts2)
    display_metric_pill(
        label="Parts",
        value=f"{total2}",
        vendor=None,
        status="info",
        icon=None
    )


st.markdown("### Parts Groups")

# Helper: extract categories
def _extract_categories(conc):
    cats = []
    if conc and conc.get('categories'):
        for c in conc['categories']:
            name = c.get('category') or 'Unknown'
            key = c.get('key') or name.lower().replace(' ', '_')
            cats.append((key, name))
    other = conc.get('other') if conc else None
    if other and other.get('items'):
        cats.append((other.get('key', 'other'), other.get('category', 'Other')))
    # Deduplicate preserving order
    seen = set()
    out = []
    for k, n in cats:
        if k not in seen:
            seen.add(k)
            out.append((k, n))
    return out

# Helper: build a quick index of detailed parts for description resolution
def _build_desc_index(parts):
    idx = {
        'by_num': {},
        'by_name': {},
    }
    for p in parts or []:
        num = (p.get('part_number') or '').upper()
        nam = (p.get('part_name') or '').upper()
        if num:
            idx['by_num'][num] = p.get('description')
        if nam:
            idx['by_name'][nam] = p.get('description')
    return idx

def _resolve_description(item, idx, parts):
    # 1) use description already present
    if item.get('description'):
        return item.get('description')
    # 2) try primary part number
    num = (item.get('primary_part_number') or '').upper()
    if num and num in idx['by_num'] and idx['by_num'][num]:
        return idx['by_num'][num]
    # 3) try synonyms against number and name indexes
    syns = item.get('synonyms') or []
    for s in syns:
        su = (s or '').upper()
        if su in idx['by_num'] and idx['by_num'][su]:
            return idx['by_num'][su]
        if su in idx['by_name'] and idx['by_name'][su]:
            return idx['by_name'][su]
    # 4) last resort: search detailed parts for same part_name
    pname = (item.get('part_name') or '').upper()
    if pname and pname in idx['by_name'] and idx['by_name'][pname]:
        return idx['by_name'][pname]
    return None

def _resolve_display_name(item, parts):
    """Prefer descriptive part_name from detailed parts for the same part."""
    # If item already has a non-numeric/longer name, keep it
    existing = item.get('part_name')
    primary = item.get('primary_part_number')
    if existing and existing != primary:
        return existing
    # Try exact part_number match in detailed parts
    for p in parts or []:
        if (p.get('part_number') or '').upper() == (primary or '').upper():
            pn = p.get('part_name')
            if pn and pn.upper() != (p.get('part_number') or '').upper():
                return pn
    # Try synonyms
    for s in item.get('synonyms') or []:
        su = (s or '').upper()
        for p in parts or []:
            if (p.get('part_number') or '').upper() == su or (p.get('part_name') or '').upper() == su:
                pn = p.get('part_name')
                if pn:
                    return pn
    # Fallback to primary number
    return existing or primary

cats1 = _extract_categories(concentrated1)
cats2 = _extract_categories(concentrated2)
merged_keys = []
names_by_key = {}
for k, n in cats1 + cats2:
    if k not in names_by_key:
        names_by_key[k] = n
        merged_keys.append(k)

# Precompute description indexes for both suppliers
desc_idx_1 = _build_desc_index(parts1)
desc_idx_2 = _build_desc_index(parts2)

def _get_items(conc, key):
    if not conc:
        return []
    for c in conc.get('categories', []) or []:
        if (c.get('key') == key) or (c.get('category', '').lower().replace(' ', '_') == key):
            return c.get('items', [])
    other = conc.get('other')
    if other and (other.get('key') == key or other.get('category', '').lower().replace(' ', '_') == key):
        return other.get('items', [])
    return []

if (concentrated1 or concentrated2) and merged_keys:
    for key in merged_keys:
        name = names_by_key[key]
        with st.expander(name, expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**SupplierA**")
                items1 = _get_items(concentrated1, key)
                if not items1:
                    st.caption("No items in this category")
                for it in items1:
                    desc = _resolve_description(it, desc_idx_1, parts1)
                    display_name = _resolve_display_name(it, parts1)
                    part_like = {
                        'part_number': it.get('primary_part_number') or it.get('canonical_part_id'),
                        'part_name': display_name,
                        'category': name,
                        'description': desc,
                        'technical_specifications': {},
                        'pricing': {},
                        'certifications': [],
                        'capacity': {},
                        'sources': it.get('sources', []),
                    }
                    display_part_details_card(part_like, "SupplierA")

            with col2:
                st.markdown(f"**SupplierB**")
                items2 = _get_items(concentrated2, key)
                if not items2:
                    st.caption("No items in this category")
                for it in items2:
                    desc = _resolve_description(it, desc_idx_2, parts2)
                    display_name = _resolve_display_name(it, parts2)
                    part_like = {
                        'part_number': it.get('primary_part_number') or it.get('canonical_part_id'),
                        'part_name': display_name,
                        'category': name,
                        'description': desc,
                        'technical_specifications': {},
                        'pricing': {},
                        'certifications': [],
                        'capacity': {},
                        'sources': it.get('sources', []),
                    }
                    display_part_details_card(part_like, "SupplierB")
else:
    st.info("Concentrated parts view not available; showing manual selection below.")

# --- Manual selection (bottom) remains the same ---
# Detailed Parts Information
st.markdown("### Detailed Parts Information")

# Create two columns for side-by-side comparison
col1, col2 = st.columns(2)

# Left column - Supplier 1
with col1:
    st.markdown(f"#### SupplierA")
    
    # Create display options for supplier 1 parts
    parts1_display = []
    parts1_map = {}
    
    # Include the overall clutch system as a selectable item
    if overall1:
        sys_ts = overall1.get('technical_specifications', {}) or {}
        # Build a part-like payload so the common renderer can display it
        system_part_1 = {
            'part_number': '',
            'part_name': overall1.get('title') or 'Overall Clutch System',
            'category': 'Clutch System',
            'description': overall1.get('description') or '',
            'technical_specifications': {
                'dimensions': sys_ts.get('dimensions', ''),
                'weight': sys_ts.get('weight', ''),
                'material': '',
                'torque_capacity': sys_ts.get('torque_capacity', ''),
                'operating_temperature': '',
                'other_specs': sys_ts.get('other_specs', []) or []
            },
            'pricing': {
                'unit_price': (overall1.get('pricing_summary') or {}).get('unit_price'),
                'currency': (overall1.get('pricing_summary') or {}).get('currency', 'EUR'),
                'volume_pricing': (overall1.get('pricing_summary') or {}).get('volume_pricing', []) or [],
                'cost_breakdown': []
            },
            'certifications': [],
            'capacity': {},
            'sources': overall1.get('sources', []) or []
        }
        parts1_display.append('Overall Clutch System')
        parts1_map['Overall Clutch System'] = system_part_1
    
    for part in parts1:
        part_num = part.get('part_number', '')
        if part_num:
            part_name = part.get('part_name', part_num)
            # Create a descriptive label
            label = f"{part_name} ({part_num})" if part_name != part_num else part_num
            parts1_display.append(label)
            parts1_map[label] = part
    
    if parts1_display:
        # Sort the display options alphabetically
        parts1_display.sort()
        
        # Show selectbox for supplier 1
        selected_label1 = st.selectbox(
            "Select a part",
            parts1_display,
            key="supplier1_parts_dropdown"
        )
        
        # Display the selected part details
        if selected_label1 and selected_label1 in parts1_map:
            st.markdown("---")
            display_part_details_card(parts1_map[selected_label1], "SupplierA")
    else:
        st.info("No parts available from this supplier")



# Right column - Supplier 2
with col2:
    st.markdown(f"#### SupplierB")
    
    # Create display options for supplier 2 parts
    parts2_display = []
    parts2_map = {}
    
    # Include the overall clutch system as a selectable item
    if overall2:
        sys_ts2 = overall2.get('technical_specifications', {}) or {}
        system_part_2 = {
            'part_number': '',
            'part_name': overall2.get('title') or 'Overall Clutch System',
            'category': 'Clutch System',
            'description': overall2.get('description') or '',
            'technical_specifications': {
                'dimensions': sys_ts2.get('dimensions', ''),
                'weight': sys_ts2.get('weight', ''),
                'material': '',
                'torque_capacity': sys_ts2.get('torque_capacity', ''),
                'operating_temperature': '',
                'other_specs': sys_ts2.get('other_specs', []) or []
            },
            'pricing': {
                'unit_price': (overall2.get('pricing_summary') or {}).get('unit_price'),
                'currency': (overall2.get('pricing_summary') or {}).get('currency', 'EUR'),
                'volume_pricing': (overall2.get('pricing_summary') or {}).get('volume_pricing', []) or [],
                'cost_breakdown': []
            },
            'certifications': [],
            'capacity': {},
            'sources': overall2.get('sources', []) or []
        }
        parts2_display.append('Overall Clutch System')
        parts2_map['Overall Clutch System'] = system_part_2
    
    for part in parts2:
        part_num = part.get('part_number', '')
        if part_num:
            part_name = part.get('part_name', part_num)
            # Create a descriptive label
            label = f"{part_name} ({part_num})" if part_name != part_num else part_num
            parts2_display.append(label)
            parts2_map[label] = part
    
    if parts2_display:
        # Sort the display options alphabetically
        parts2_display.sort()
        
        # Show selectbox for supplier 2
        selected_label2 = st.selectbox(
            "Select a part",
            parts2_display,
            key="supplier2_parts_dropdown"
        )
        
        # Display the selected part details
        if selected_label2 and selected_label2 in parts2_map:
            st.markdown("---")
            display_part_details_card(parts2_map[selected_label2], "SupplierB")
    else:
        st.info("No parts available from this supplier")