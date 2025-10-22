"""
Dashboard Components - Reusable UI components for the supplier dashboard

This module contains reusable Streamlit components for displaying
supplier analysis data with source traceability.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional
import html
import time

CATEGORY_DISPLAY_NAMES = {
    'technology': 'R&D',
    'quality': 'Quality',
    'delivery': 'Distribution',
    'cost': 'Cost',
    'sustainability': 'Eco',
}


def get_category_display_name(category: str) -> str:
    """
    Return the display label for a given analysis category.
    Defaults to capitalized input when no mapping exists.
    """
    if not isinstance(category, str):
        return str(category)
    return CATEGORY_DISPLAY_NAMES.get(category.lower(), category.capitalize())


def create_source_badge(sources: List[Dict[str, str]], max_display: int = None) -> None:
    """
    Create a formatted source reference display
    
    Args:
        sources: List of source dictionaries with 'filename' and 'chunk_id'
        max_display: Optional maximum number of sources to display (if None, shows all)
    """
    if not sources:
        return
    
    source_texts = []
    # If max_display is None, show all sources
    sources_to_show = sources[:max_display] if max_display else sources
    
    for source in sources_to_show:
        filename = source.get('filename', 'Unknown')
        chunk_id = source.get('chunk_id', '')
        # Shorten filename if too long
        # if len(filename) > 30:
        #     filename = "..." + filename[-27:]
        source_texts.append(f"📄 {filename}:{chunk_id}")
    
    # Only add "more" text if max_display is explicitly set and there are more sources
    if max_display and len(sources) > max_display:
        source_texts.append(f"... +{len(sources) - max_display} more")
    
    st.caption(" | ".join(source_texts))


def display_source_details(sources: List[Dict[str, str]], title: str = "Source Documents") -> None:
    """
    Display expandable source details
    
    Args:
        sources: List of source dictionaries
        title: Title for the expander
    """
    if not sources:
        return
    
    with st.expander(f"{title} ({len(sources)} sources)"):
        for i, source in enumerate(sources, 1):
            st.text(f"{i}. Document: {source.get('filename', 'Unknown')}")
            st.text(f"   Location: {source.get('chunk_id', 'N/A')}")


def create_tqdcs_spider_chart(
    supplier1_scores: Dict[str, Any],
    supplier2_scores: Dict[str, Any],
    supplier1_name: str,
    supplier2_name: str
) -> go.Figure:
    """
    Create a spider/radar chart comparing TQDCS scores
    
    Args:
        supplier1_scores: TQDCS scores for supplier 1
        supplier2_scores: TQDCS scores for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        
    Returns:
        Plotly figure object
    """
    category_order = ['technology', 'quality', 'delivery', 'cost', 'sustainability']
    categories = [get_category_display_name(cat) for cat in category_order]
    
    # Extract scores
    supplier1_values = []
    supplier2_values = []
    
    for cat in category_order:
        supplier1_values.append(supplier1_scores.get(cat, {}).get('score', 0))
        supplier2_values.append(supplier2_scores.get(cat, {}).get('score', 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add supplier 1 trace
    fig.add_trace(go.Scatterpolar(
        r=supplier1_values + [supplier1_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=supplier1_name,
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Add supplier 2 trace
    fig.add_trace(go.Scatterpolar(
        r=supplier2_values + [supplier2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=supplier2_name,
        line_color='#ff7f0e',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickmode='linear',
                tick0=0,
                dtick=1
            )
        ),
        showlegend=True,
        title="RQDCE Performance Comparison",
        height=500
    )
    
    return fig


def display_risk_card(risk: Dict[str, Any], show_mitigation: bool = True) -> None:
    """
    Display a risk card with severity-based coloring
    
    Args:
        risk: Risk dictionary with title, severity, description, etc.
        show_mitigation: Whether to show mitigation strategies
    """
    severity = risk.get('severity', 'Medium')
    title = risk.get('title', 'Unknown Risk')
    description = risk.get('description', '')
    
    # Define colors based on severity
    severity_colors = {
        'High': {'bg': '#ffcdd2', 'border': '#d32f2f', 'icon': '🔴'},
        'Medium': {'bg': '#ffe0b2', 'border': '#f57c00', 'icon': '🟡'},
        'Low': {'bg': '#c8e6c9', 'border': '#388e3c', 'icon': '🟢'}
    }
    
    colors = severity_colors.get(severity, severity_colors['Medium'])
    
    # Create risk card with HTML
    st.markdown(
        f"""<div style="
            background-color: {colors['bg']}; 
            padding: 10px; 
            border-left: 4px solid {colors['border']}; 
            margin: 8px 0; 
            border-radius: 4px;">
            <strong>{colors['icon']} {title}</strong><br>
            <em>{description}</em>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Show sources
    if risk.get('sources'):
        create_source_badge(risk['sources'])
    
    # Show mitigation if requested
    if show_mitigation and risk.get('mitigation'):
        with st.expander("Mitigation Strategy"):
            st.write(risk['mitigation'])
            if risk.get('mitigation_actions'):
                st.write("**Actions:**")
                for action in risk['mitigation_actions']:
                    st.write(f"• {action}")


def create_cost_breakdown_chart(cost_breakdown: List[Dict[str, Any]], title: str = "Cost Breakdown") -> go.Figure:
    """
    Create a pie chart for cost breakdown with enhanced hover tooltips
    
    Args:
        cost_breakdown: List of cost categories with amounts, descriptions, and percentages
        title: Chart title
        
    Returns:
        Plotly figure object with detailed hover information
    """
    if not cost_breakdown:
        return go.Figure()
    
    categories = []
    amounts = []
    
    for item in cost_breakdown:
        categories.append(item.get('category', 'Unknown'))
        amounts.append(item.get('amount', 0))
    
    # Create clean hover template without lengthy descriptions
    hover_template = (
        "<b>%{label}</b><br>"
        "<b>Amount:</b> €%{value:,.0f}<br>"
        "<b>Percentage:</b> %{percent}<br>"
        "<extra></extra>"  # Remove the default hover box
    )
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=amounts,
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
        hovertemplate=hover_template,  # Use clean hover template
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig


def apply_metric_pill_styles() -> None:
    """
    Inject CSS styles for reusable metric pills used across pages.
    This focuses only on the pill component styling and keeps other styles untouched.
    """
    st.markdown(
        """
        <style>
        /* Core metric pill styling */
        .metric-pill {
            background: white;
            border-radius: 20px;
            padding: 1rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .metric-pill:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            border-color: #3b82f6;
        }

        .metric-pill-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }

        .metric-pill-label {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .metric-pill-vendor {
            font-size: 0.85rem;
            color: #4b5563;
            font-weight: 500;
        }

        .metric-pill-delta {
            font-size: 0.85rem;
            font-weight: 600;
            color: #374151;
            margin-top: 0.25rem;
        }

        /* Optional status accents via border color */
        .metric-pill--positive { border-color: #22c55e; }
        .metric-pill--negative { border-color: #ef4444; }
        .metric-pill--info { border-color: #3b82f6; }
        .metric-pill--neutral { border-color: #e5e7eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_streaming_chat(
    state_key: str = "streaming_chat_messages",
    greeting: str = "Hi, I'm your RFQ assistant. How can I help?",
    placeholder: str = "Ask a question…",
    stream_delay_s: float = 0.01,
) -> None:
    """
    Render a streaming chat dock using Streamlit's built-in chat APIs.

    - Keeps messages in session state
    - Renders history with st.chat_message
    - Accepts input via st.chat_input
    - Streams assistant response token-by-token to reduce perceived reloads

    Args:
        state_key: Session state key to store chat messages
        greeting: Initial assistant greeting shown on first load
        placeholder: Input placeholder text
        stream_delay_s: Delay between streamed tokens (demo effect)
    """
    # Initialize chat history
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {"role": "assistant", "content": greeting}
        ]

    # Render message history
    for message in st.session_state[state_key]:
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Handle new input
    prompt = st.chat_input(placeholder)
    if prompt:
        # Append and show user message immediately
        st.session_state[state_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response (placeholder echo). Replace with LLM stream later.
        reply_text = f"(preview) Echo: {prompt}"

        def _token_generator():
            for ch in reply_text:
                yield ch
                if stream_delay_s:
                    time.sleep(stream_delay_s)

        with st.chat_message("assistant"):
            _ = st.write_stream(_token_generator())

        # Persist assistant message
        st.session_state[state_key].append({"role": "assistant", "content": reply_text})


def display_metric_pill(
    label: str,
    value: str,
    vendor: Optional[str] = None,
    delta: Optional[str] = None,
    status: str = "info",
    icon: Optional[str] = None,
) -> None:
    """
    Render a single metric pill with label, value, optional vendor and delta.

    Args:
        label: Short label describing the metric (e.g., "Score", "Parts").
        value: Main numeric/text value to display prominently.
        vendor: Optional vendor/supplier name rendered as a caption.
        delta: Optional delta text (e.g., "Leading", "-12.5% lower").
        status: Visual accent for the pill border: one of {positive, negative, info, neutral}.
        icon: Optional emoji/icon prefix for value to enhance visual scanning.
    """
    # Build the HTML in small pieces to avoid any chance of the Markdown renderer
    # treating parts of it as literal text (e.g. when nested f-strings inject tags).
    # All dynamic values are escaped to prevent injection; only our structural tags render.
    status_class = status if status in {"positive", "negative", "info", "neutral"} else "neutral"
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_vendor = html.escape(str(vendor)) if vendor is not None else None
    safe_delta = html.escape(str(delta)) if delta is not None else None
    safe_icon = html.escape(str(icon)) + " " if icon else ""

    lines = [
        f'<div class="metric-pill metric-pill--{status_class}">',
        f'  <div class="metric-pill-label">{safe_label}</div>',
        f'  <div class="metric-pill-value">{safe_icon}{safe_value}</div>',
    ]

    if safe_vendor:
        lines.append(f'  <div class="metric-pill-vendor">{safe_vendor}</div>')
    if safe_delta:
        lines.append(f'  <div class="metric-pill-delta">{safe_delta}</div>')

    lines.append('</div>')

    st.markdown("\n".join(lines), unsafe_allow_html=True)


def display_comparison_metric(metric: Dict[str, Any]) -> None:
    """
    Display a single comparison metric with clear column headers and winner indication
    
    Args:
        metric: Comparison metric dictionary
    """
    # Extract supplier names for headers
    supplier1_name = metric.get('supplier1_name', 'Supplier 1')
    supplier2_name = metric.get('supplier2_name', 'Supplier 2')
    
    # Display headers (always visible for clarity)
    header_cols = st.columns([2, 2, 2, 1])
    with header_cols[0]:
        st.markdown("**Metric**")
    with header_cols[1]:
        st.markdown(f"**{supplier1_name}**")
    with header_cols[2]:
        st.markdown(f"**{supplier2_name}**")
    with header_cols[3]:
        st.markdown("**Winner**")
    
    # Display metric values
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        st.write(f"{metric.get('metric', 'Unknown')}")
        importance = metric.get('importance', 'Medium')
        if importance == 'Critical':
            st.caption("⚠️ Critical")
        elif importance == 'High':
            st.caption("❗ High importance")
    
    with col2:
        value1 = metric.get('supplier1_value', 'N/A')
        winner = metric.get('winner', '')
        
        if winner == supplier1_name:
            st.success(f"✓ {value1}")
        else:
            st.write(value1)
    
    with col3:
        value2 = metric.get('supplier2_value', 'N/A')
        
        if winner == supplier2_name:
            st.success(f"✓ {value2}")
        else:
            st.write(value2)
    
    with col4:
        if winner == 'Tie':
            st.info("Tie")
        elif winner:
            st.write(f"→ {winner[:10]}")
    
    # Show comparison notes if available
    if metric.get('comparison_notes'):
        st.caption(f"💡 {metric['comparison_notes']}")
    
    # Show sources
    if metric.get('sources'):
        col1, col2 = st.columns(2)
        with col1:
            if metric['sources'].get('supplier1'):
                create_source_badge(metric['sources']['supplier1'])  # Show all sources
        with col2:
            if metric['sources'].get('supplier2'):
                create_source_badge(metric['sources']['supplier2'])  # Show all sources


def create_optimal_split_chart(
    split_data: Dict[str, Any]
) -> go.Figure:
    """
    Create a donut chart for optimal supplier split
    
    Args:
        split_data: Optimal split data with percentages
        
    Returns:
        Plotly figure object
    """
    supplier1_name = split_data.get('supplier1_name', 'Supplier 1')
    supplier2_name = split_data.get('supplier2_name', 'Supplier 2')
    supplier1_pct = split_data.get('supplier1_percentage', 50)
    supplier2_pct = split_data.get('supplier2_percentage', 50)
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{supplier1_name} ({supplier1_pct}%)", f"{supplier2_name} ({supplier2_pct}%)"],
        values=[supplier1_pct, supplier2_pct],
        hole=0.4,
        marker_colors=['#1f77b4', '#ff7f0e'],
        textfont_size=16,
        textposition='outside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Volume Share: %{value}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Recommended Volume Allocation",
        height=450,
        annotations=[dict(
            text='<b>Split<br>Strategy</b>',
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False
        )],
        showlegend=False
    )
    
    return fig


def display_tqdcs_score_card(
    category: str,
    score_data: Dict[str, Any],
    show_details: bool = True
) -> None:
    """
    Display a TQDCS score card with details
    
    Args:
        category: Category name (e.g., 'Technology')
        score_data: Score data including score, reasoning, strengths, weaknesses
        show_details: If True, wraps details in an expander. If False, shows details directly.
    """
    score = score_data.get('score', 0)
    
    # Color code based on score
    if score >= 4.5:
        color = "🟢"
        status = "Excellent"
    elif score >= 4.0:
        color = "🟢"
        status = "Good"
    elif score >= 3.0:
        color = "🟡"
        status = "Adequate"
    else:
        color = "🔴"
        status = "Poor"
    
    # Only display score metric if show_details is True (to avoid duplication)
    display_label = get_category_display_name(category)

    if show_details:
        st.metric(
            label=display_label,
            value=f"{score}/5",
            delta=f"{status} {color}"
        )
    
    # Display details either in expander or directly
    if show_details:
        # Wrap in expander
        with st.expander(f"Details for {display_label}"):
            # Key findings FIRST
            if score_data.get('key_findings'):
                st.write("**Key Points:**")
                for finding in score_data['key_findings']:
                    st.write(f"• {finding}")
            
            # Strengths and Weaknesses SECOND
            col1, col2 = st.columns(2)
            
            with col1:
                if score_data.get('strengths'):
                    st.write("**Strengths:**")
                    for strength in score_data['strengths']:
                        st.write(f"✓ {strength}")
            
            with col2:
                if score_data.get('weaknesses'):
                    st.write("**Weaknesses:**")
                    for weakness in score_data['weaknesses']:
                        st.write(f"✗ {weakness}")
            
            # Assessment (Reasoning) LAST
            if score_data.get('reasoning'):
                st.write("**Assessment:**")
                st.write(score_data['reasoning'])
            
            # Sources
            if score_data.get('sources'):
                display_source_details(score_data['sources'])
    else:
        # Show details directly without expander
        # Key findings FIRST
        if score_data.get('key_findings'):
            st.write("**Key Points:**")
            for finding in score_data['key_findings']:
                st.write(f"• {finding}")
        
        # Strengths and Weaknesses SECOND
        col1, col2 = st.columns(2)
        
        with col1:
            if score_data.get('strengths'):
                st.write("**Strengths:**")
                for strength in score_data['strengths']:
                    st.write(f"✓ {strength}")
        
        with col2:
            if score_data.get('weaknesses'):
                st.write("**Weaknesses:**")
                for weakness in score_data['weaknesses']:
                    st.write(f"✗ {weakness}")
        
        # Assessment (Reasoning) LAST
        if score_data.get('reasoning'):
            st.write("**Assessment:**")
            st.write(score_data['reasoning'])
        
        # Sources
        if score_data.get('sources'):
            display_source_details(score_data['sources'])


def create_risk_heatmap(
    risk_matrix: Dict[str, List[Dict[str, Any]]],
    title: str = "Risk Heatmap"
) -> go.Figure:
    """
    Create a heatmap visualization of risks by category and severity
    
    Args:
        risk_matrix: Risk matrix organized by category
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    categories = []
    high_counts = []
    medium_counts = []
    low_counts = []
    
    for category, risks in risk_matrix.items():
        # Format category name
        cat_name = category.replace('_risks', '').replace('_', ' ').title()
        categories.append(cat_name)
        
        # Count by severity
        high = sum(1 for r in risks if r.get('severity') == 'High')
        medium = sum(1 for r in risks if r.get('severity') == 'Medium')
        low = sum(1 for r in risks if r.get('severity') == 'Low')
        
        high_counts.append(high)
        medium_counts.append(medium)
        low_counts.append(low)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='High Risk',
        x=categories,
        y=high_counts,
        marker_color='#d32f2f'
    ))
    
    fig.add_trace(go.Bar(
        name='Medium Risk',
        x=categories,
        y=medium_counts,
        marker_color='#f57c00'
    ))
    
    fig.add_trace(go.Bar(
        name='Low Risk',
        x=categories,
        y=low_counts,
        marker_color='#388e3c'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title="Risk Category",
        yaxis_title="Number of Risks",
        height=400
    )
    
    return fig


def create_parts_comparison_table(
    parts1: List[Dict[str, Any]],
    parts2: List[Dict[str, Any]],
    supplier1_name: str,
    supplier2_name: str
) -> pd.DataFrame:
    """
    Create a comparison table for parts from two suppliers
    
    Args:
        parts1: Parts data for supplier 1
        parts2: Parts data for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        
    Returns:
        DataFrame with parts comparison
    """
    # Create dictionaries for quick lookup
    parts1_dict = {p.get("part_number", ""): p for p in parts1}
    parts2_dict = {p.get("part_number", ""): p for p in parts2}
    
    # Get all unique part numbers
    all_part_numbers = set(parts1_dict.keys()) | set(parts2_dict.keys())
    
    comparison_data = []
    for part_num in sorted(all_part_numbers):
        if not part_num:  # Skip empty part numbers
            continue
            
        part1 = parts1_dict.get(part_num, {})
        part2 = parts2_dict.get(part_num, {})
        
        comparison_data.append({
            "Part Number": part_num,
            f"{supplier1_name} Name": part1.get("part_name", "-"),
            f"{supplier2_name} Name": part2.get("part_name", "-"),
            f"{supplier1_name} Price": f"€{part1.get('pricing', {}).get('unit_price', 'N/A')}" if part1.get("pricing", {}).get("unit_price") else "-",
            f"{supplier2_name} Price": f"€{part2.get('pricing', {}).get('unit_price', 'N/A')}" if part2.get("pricing", {}).get("unit_price") else "-",
            f"{supplier1_name} Lead Time": part1.get("capacity", {}).get("lead_time", "-"),
            f"{supplier2_name} Lead Time": part2.get("capacity", {}).get("lead_time", "-"),
        })
    
    return pd.DataFrame(comparison_data)


def display_part_details_card(
    part: Dict[str, Any],
    supplier_name: str,
    show_sources: bool = True
) -> None:
    """
    Display detailed information for a single part
    
    Args:
        part: Part data dictionary
        supplier_name: Name of the supplier
        show_sources: Whether to show source references
    """
    # Create a descriptive header with part name and supplier
    part_name = part.get('part_name', part.get('part_number', 'Unknown Part'))
    header_text = f"{part_name} - {supplier_name}"
    
    st.markdown(f"### {header_text}")
    
    # Show part number as secondary information if we used the name in header
    if part.get('part_name') and part.get('part_number'):
        st.markdown(f"**Part Number:** {part.get('part_number')}")
    elif not part.get('part_name') and part.get('part_number'):
        # If we only have part number (already shown in header), show supplier
        st.markdown(f"**Supplier:** {supplier_name}")
    
    if part.get("description"):
        st.write(part["description"])
    
    # Technical specifications
    if part.get("technical_specifications"):
        specs = part["technical_specifications"]
        st.markdown("**Technical Specifications:**")
        
        # Display all specifications in a single column vertical list
        if specs.get("dimensions"):
            st.write(f"• Dimensions: {specs['dimensions']}")
        if specs.get("weight"):
            st.write(f"• Weight: {specs['weight']}")
        if specs.get("material"):
            st.write(f"• Material: {specs['material']}")
        if specs.get("torque_capacity"):
            st.write(f"• Torque Capacity: {specs['torque_capacity']}")
        if specs.get("operating_temperature"):
            st.write(f"• Operating Temp: {specs['operating_temperature']}")
        if specs.get("other_specs"):
            for spec in specs["other_specs"]:
                st.write(f"• {spec}")
    
    # Pricing information
    if part.get("pricing"):
        pricing = part["pricing"]
        st.markdown("**Pricing:**")
        if pricing.get("unit_price") is not None and pricing.get("unit_price") != "N/A":
            try:
                st.metric("Unit Price", f"€{float(pricing['unit_price']):,.2f}")
            except Exception:
                st.metric("Unit Price", f"€{pricing['unit_price']}")
        
        if pricing.get("volume_pricing"):
            st.markdown("*Volume Pricing:*")
            for vp in pricing["volume_pricing"]:
                ppu = vp.get('price_per_unit')
                price_text = f"€{ppu:,.2f}" if isinstance(ppu, (int, float)) else (f"€{ppu}" if ppu not in (None, "", "N/A") else "-")
                st.write(f"• {vp.get('volume', 'N/A')}: {price_text}")
    
    # Certifications
    if part.get("certifications"):
        st.markdown("**Certifications:**")
        # Display certifications in a single column vertical list
        for cert in part["certifications"]:
            st.info(f"📋 {cert.get('standard', 'Unknown')}")
    
    # Capacity information
    if part.get("capacity"):
        cap = part["capacity"]
        st.markdown("**Production Capacity:**")
        
        # Display all capacity information in a single column vertical list
        if cap.get("production_capacity"):
            st.write(f"• Capacity: {cap['production_capacity']}")
        if cap.get("lead_time"):
            st.write(f"• Lead Time: {cap['lead_time']}")
        if cap.get("current_utilization"):
            st.write(f"• Utilization: {cap['current_utilization']}")
        if cap.get("min_order_quantity"):
            st.write(f"• MOQ: {cap['min_order_quantity']}")
    
    # Source references
    if show_sources and part.get("sources"):
        create_source_badge(part["sources"])  # Show all sources without limit


def create_parts_specifications_chart(
    parts1: List[Dict[str, Any]],
    parts2: List[Dict[str, Any]],
    supplier1_name: str,
    supplier2_name: str,
    metric: str = "price"
) -> go.Figure:
    """
    Create a bar chart comparing parts specifications
    
    Args:
        parts1: Parts data for supplier 1
        parts2: Parts data for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        metric: Metric to compare ('price', 'lead_time', 'capacity')
        
    Returns:
        Plotly figure object
    """
    # Create dictionaries for quick lookup
    parts1_dict = {p.get("part_number", ""): p for p in parts1}
    parts2_dict = {p.get("part_number", ""): p for p in parts2}
    
    # Get common part numbers
    common_parts = set(parts1_dict.keys()) & set(parts2_dict.keys())
    common_parts = sorted([p for p in common_parts if p])[:10]  # Limit to 10 for visibility
    
    values1 = []
    values2 = []
    
    for part_num in common_parts:
        part1 = parts1_dict[part_num]
        part2 = parts2_dict[part_num]
        
        if metric == "price":
            val1 = part1.get("pricing", {}).get("unit_price", 0)
            val2 = part2.get("pricing", {}).get("unit_price", 0)
        elif metric == "lead_time":
            # Convert lead time to numeric days (simplified)
            lt1 = part1.get("capacity", {}).get("lead_time", "0 days")
            lt2 = part2.get("capacity", {}).get("lead_time", "0 days")
            val1 = int("".join(filter(str.isdigit, str(lt1))) or 0)
            val2 = int("".join(filter(str.isdigit, str(lt2))) or 0)
        else:
            val1 = 0
            val2 = 0
        
        values1.append(val1)
        values2.append(val2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=supplier1_name,
        x=common_parts,
        y=values1,
        marker_color='#1f77b4',
        text=values1,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name=supplier2_name,
        x=common_parts,
        y=values2,
        marker_color='#ff7f0e',
        text=values2,
        textposition='auto',
    ))
    
    title = {
        "price": "Unit Price Comparison (EUR)",
        "lead_time": "Lead Time Comparison (Days)",
        "capacity": "Production Capacity Comparison"
    }.get(metric, "Parts Comparison")
    
    fig.update_layout(
        title=title,
        xaxis_title="Part Number",
        yaxis_title=metric.replace("_", " ").title(),
        barmode="group",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def display_certification_badges(
    certifications: List[Dict[str, str]],
    max_display: int = 5
) -> None:
    """
    Display certification badges in a compact format
    
    Args:
        certifications: List of certification dictionaries
        max_display: Maximum number of certifications to display
    """
    if not certifications:
        st.write("No certifications available")
        return
    
    # Create columns for badges
    cols = st.columns(min(len(certifications), max_display))
    
    for i, cert in enumerate(certifications[:max_display]):
        with cols[i]:
            standard = cert.get("standard", "Unknown")
            description = cert.get("description", "")
            
            # Create a colored badge based on certification type
            if "ISO" in standard:
                color = '#4CAF50'  # Green for ISO
            elif "IATF" in standard:
                color = '#2196F3'  # Blue for IATF
            elif "CVS" in standard:
                color = '#FF9800'  # Orange for client-specific standards
            else:
                color = '#9E9E9E'  # Grey for others
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    text-align: center;
                    margin: 2px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {standard}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if description:
                st.caption(description[:50] + "..." if len(description) > 50 else description)
    
    if len(certifications) > max_display:
        st.caption(f"... +{len(certifications) - max_display} more certifications")


def create_parts_category_distribution(
    parts_data: Dict[str, Any],
    supplier_name: str
) -> go.Figure:
    """
    Create a pie chart showing distribution of parts by category
    
    Args:
        parts_data: Parts analysis data including summary
        supplier_name: Name of the supplier
        
    Returns:
        Plotly figure object
    """
    categories = {}
    
    for part in parts_data.get("parts", []):
        category = part.get("category", "Uncategorized")
        categories[category] = categories.get(category, 0) + 1
    
    if not categories:
        categories = {"No parts data": 1}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(categories.keys()),
        values=list(categories.values()),
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"{supplier_name} - Parts by Category",
        height=400,
        showlegend=True
    )
    
    return fig


"""
Dashboard Components - Reusable UI components for the supplier dashboard

This module contains reusable Streamlit components for displaying
supplier analysis data with source traceability.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List, Optional
import html


def create_source_badge(sources: List[Dict[str, str]], max_display: int = None) -> None:
    """
    Create a formatted source reference display
    
    Args:
        sources: List of source dictionaries with 'filename' and 'chunk_id'
        max_display: Optional maximum number of sources to display (if None, shows all)
    """
    if not sources:
        return
    
    source_texts = []
    # If max_display is None, show all sources
    sources_to_show = sources[:max_display] if max_display else sources
    
    for source in sources_to_show:
        filename = source.get('filename', 'Unknown')
        chunk_id = source.get('chunk_id', '')
        # Shorten filename if too long
        # if len(filename) > 30:
        #     filename = "..." + filename[-27:]
        source_texts.append(f"📄 {filename}:{chunk_id}")
    
    # Only add "more" text if max_display is explicitly set and there are more sources
    if max_display and len(sources) > max_display:
        source_texts.append(f"... +{len(sources) - max_display} more")
    
    st.caption(" | ".join(source_texts))


def display_source_details(sources: List[Dict[str, str]], title: str = "Source Documents") -> None:
    """
    Display expandable source details
    
    Args:
        sources: List of source dictionaries
        title: Title for the expander
    """
    if not sources:
        return
    
    with st.expander(f"{title} ({len(sources)} sources)"):
        for i, source in enumerate(sources, 1):
            st.text(f"{i}. Document: {source.get('filename', 'Unknown')}")
            st.text(f"   Location: {source.get('chunk_id', 'N/A')}")


def create_tqdcs_spider_chart(
    supplier1_scores: Dict[str, Any],
    supplier2_scores: Dict[str, Any],
    supplier1_name: str,
    supplier2_name: str
) -> go.Figure:
    """
    Create a spider/radar chart comparing TQDCS scores
    
    Args:
        supplier1_scores: TQDCS scores for supplier 1
        supplier2_scores: TQDCS scores for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        
    Returns:
        Plotly figure object
    """
    category_order = ['technology', 'quality', 'delivery', 'cost', 'sustainability']
    categories = [get_category_display_name(cat) for cat in category_order]
    
    # Extract scores
    supplier1_values = []
    supplier2_values = []
    
    for cat in category_order:
        supplier1_values.append(supplier1_scores.get(cat, {}).get('score', 0))
        supplier2_values.append(supplier2_scores.get(cat, {}).get('score', 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add supplier 1 trace
    fig.add_trace(go.Scatterpolar(
        r=supplier1_values + [supplier1_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=supplier1_name or "SupplierA",
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # Add supplier 2 trace
    fig.add_trace(go.Scatterpolar(
        r=supplier2_values + [supplier2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=supplier2_name or "SupplierB",
        line_color='#ff7f0e',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickmode='linear',
                tick0=0,
                dtick=1
            )
        ),
        showlegend=True,
        title="RQDCE Performance Comparison",
        height=500
    )
    
    return fig


def display_risk_card(risk: Dict[str, Any], show_mitigation: bool = True) -> None:
    """
    Display a risk card with severity-based coloring
    
    Args:
        risk: Risk dictionary with title, severity, description, etc.
        show_mitigation: Whether to show mitigation strategies
    """
    severity = risk.get('severity', 'Medium')
    title = risk.get('title', 'Unknown Risk')
    description = risk.get('description', '')
    
    # Define colors based on severity
    severity_colors = {
        'High': {'bg': '#ffcdd2', 'border': '#d32f2f', 'icon': '🔴'},
        'Medium': {'bg': '#ffe0b2', 'border': '#f57c00', 'icon': '🟡'},
        'Low': {'bg': '#c8e6c9', 'border': '#388e3c', 'icon': '🟢'}
    }
    
    colors = severity_colors.get(severity, severity_colors['Medium'])
    
    # Create risk card with HTML
    st.markdown(
        f"""<div style="
            background-color: {colors['bg']}; 
            padding: 10px; 
            border-left: 4px solid {colors['border']}; 
            margin: 8px 0; 
            border-radius: 4px;">
            <strong>{colors['icon']} {title}</strong><br>
            <em>{description}</em>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Show sources
    if risk.get('sources'):
        create_source_badge(risk['sources'])
    
    # Show mitigation if requested
    if show_mitigation and risk.get('mitigation'):
        with st.expander("Mitigation Strategy"):
            st.write(risk['mitigation'])
            if risk.get('mitigation_actions'):
                st.write("**Actions:**")
                for action in risk['mitigation_actions']:
                    st.write(f"• {action}")


def create_cost_breakdown_chart(cost_breakdown: List[Dict[str, Any]], title: str = "Cost Breakdown") -> go.Figure:
    """
    Create a pie chart for cost breakdown with enhanced hover tooltips
    
    Args:
        cost_breakdown: List of cost categories with amounts, descriptions, and percentages
        title: Chart title
        
    Returns:
        Plotly figure object with detailed hover information
    """
    if not cost_breakdown:
        return go.Figure()
    
    categories = []
    amounts = []
    
    for item in cost_breakdown:
        categories.append(item.get('category', 'Unknown'))
        amounts.append(item.get('amount', 0))
    
    # Create clean hover template without lengthy descriptions
    hover_template = (
        "<b>%{label}</b><br>"
        "<b>Amount:</b> €%{value:,.0f}<br>"
        "<b>Percentage:</b> %{percent}<br>"
        "<extra></extra>"  # Remove the default hover box
    )
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=amounts,
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
        hovertemplate=hover_template,  # Use clean hover template
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig


def apply_metric_pill_styles() -> None:
    """
    Inject CSS styles for reusable metric pills used across pages.
    This focuses only on the pill component styling and keeps other styles untouched.
    """
    st.markdown(
        """
        <style>
        /* Core metric pill styling */
        .metric-pill {
            background: white;
            border-radius: 20px;
            padding: 1rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .metric-pill:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            border-color: #3b82f6;
        }

        .metric-pill-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }

        .metric-pill-label {
            font-size: 0.9rem;
            color: #6b7280;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .metric-pill-vendor {
            font-size: 0.85rem;
            color: #4b5563;
            font-weight: 500;
        }

        .metric-pill-delta {
            font-size: 0.85rem;
            font-weight: 600;
            color: #374151;
            margin-top: 0.25rem;
        }

        /* Optional status accents via border color */
        .metric-pill--positive { border-color: #22c55e; }
        .metric-pill--negative { border-color: #ef4444; }
        .metric-pill--info { border-color: #3b82f6; }
        .metric-pill--neutral { border-color: #e5e7eb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_metric_pill(
    label: str,
    value: str,
    vendor: Optional[str] = None,
    delta: Optional[str] = None,
    status: str = "info",
    icon: Optional[str] = None,
) -> None:
    """
    Render a single metric pill with label, value, optional vendor and delta.

    Args:
        label: Short label describing the metric (e.g., "Score", "Parts").
        value: Main numeric/text value to display prominently.
        vendor: Optional vendor/supplier name rendered as a caption.
        delta: Optional delta text (e.g., "Leading", "-12.5% lower").
        status: Visual accent for the pill border: one of {positive, negative, info, neutral}.
        icon: Optional emoji/icon prefix for value to enhance visual scanning.
    """
    # Build the HTML in small pieces to avoid any chance of the Markdown renderer
    # treating parts of it as literal text (e.g. when nested f-strings inject tags).
    # All dynamic values are escaped to prevent injection; only our structural tags render.
    status_class = status if status in {"positive", "negative", "info", "neutral"} else "neutral"
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_vendor = html.escape(str(vendor)) if vendor is not None else None
    safe_delta = html.escape(str(delta)) if delta is not None else None
    safe_icon = html.escape(str(icon)) + " " if icon else ""

    lines = [
        f'<div class="metric-pill metric-pill--{status_class}">',
        f'  <div class="metric-pill-label">{safe_label}</div>',
        f'  <div class="metric-pill-value">{safe_icon}{safe_value}</div>',
    ]

    if safe_vendor:
        lines.append(f'  <div class="metric-pill-vendor">{safe_vendor}</div>')
    if safe_delta:
        lines.append(f'  <div class="metric-pill-delta">{safe_delta}</div>')

    lines.append('</div>')

    st.markdown("\n".join(lines), unsafe_allow_html=True)


def display_comparison_metric(metric: Dict[str, Any]) -> None:
    """
    Display a single comparison metric with clear column headers and winner indication
    
    Args:
        metric: Comparison metric dictionary
    """
    # Extract supplier names for headers
    supplier1_name = metric.get('supplier1_name', 'Supplier 1')
    supplier2_name = metric.get('supplier2_name', 'Supplier 2')
    
    # Display headers (always visible for clarity)
    header_cols = st.columns([2, 2, 2, 1])
    with header_cols[0]:
        st.markdown("**Metric**")
    with header_cols[1]:
        st.markdown(f"**{supplier1_name}**")
    with header_cols[2]:
        st.markdown(f"**{supplier2_name}**")
    with header_cols[3]:
        st.markdown("**Winner**")
    
    # Display metric values
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        st.write(f"{metric.get('metric', 'Unknown')}")
        importance = metric.get('importance', 'Medium')
        if importance == 'Critical':
            st.caption("⚠️ Critical")
        elif importance == 'High':
            st.caption("❗ High importance")
    
    with col2:
        value1 = metric.get('supplier1_value', 'N/A')
        winner = metric.get('winner', '')
        
        if winner == supplier1_name:
            st.success(f"✓ {value1}")
        else:
            st.write(value1)
    
    with col3:
        value2 = metric.get('supplier2_value', 'N/A')
        
        if winner == supplier2_name:
            st.success(f"✓ {value2}")
        else:
            st.write(value2)
    
    with col4:
        if winner == 'Tie':
            st.info("Tie")
        elif winner:
            st.write(f"→ {winner[:10]}")
    
    # Show comparison notes if available
    if metric.get('comparison_notes'):
        st.caption(f"💡 {metric['comparison_notes']}")
    
    # Show sources
    if metric.get('sources'):
        col1, col2 = st.columns(2)
        with col1:
            if metric['sources'].get('supplier1'):
                create_source_badge(metric['sources']['supplier1'])  # Show all sources
        with col2:
            if metric['sources'].get('supplier2'):
                create_source_badge(metric['sources']['supplier2'])  # Show all sources


def create_optimal_split_chart(
    split_data: Dict[str, Any]
) -> go.Figure:
    """
    Create a donut chart for optimal supplier split
    
    Args:
        split_data: Optimal split data with percentages
        
    Returns:
        Plotly figure object
    """
    supplier1_name = split_data.get('supplier1_name', 'Supplier 1')
    supplier2_name = split_data.get('supplier2_name', 'Supplier 2')
    supplier1_pct = split_data.get('supplier1_percentage', 50)
    supplier2_pct = split_data.get('supplier2_percentage', 50)
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{supplier1_name} ({supplier1_pct}%)", f"{supplier2_name} ({supplier2_pct}%)"],
        values=[supplier1_pct, supplier2_pct],
        hole=0.4,
        marker_colors=['#1f77b4', '#ff7f0e'],
        textfont_size=16,
        textposition='outside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Volume Share: %{value}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Recommended Volume Allocation",
        height=450,
        annotations=[dict(
            text='<b>Split<br>Strategy</b>',
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False
        )],
        showlegend=False
    )
    
    return fig


def display_tqdcs_score_card(
    category: str,
    score_data: Dict[str, Any],
    show_details: bool = True
) -> None:
    """
    Display a TQDCS score card with details
    
    Args:
        category: Category name (e.g., 'Technology')
        score_data: Score data including score, reasoning, strengths, weaknesses
        show_details: If True, wraps details in an expander. If False, shows details directly.
    """
    score = score_data.get('score', 0)
    
    # Color code based on score
    if score >= 4.5:
        color = "🟢"
        status = "Excellent"
    elif score >= 4.0:
        color = "🟢"
        status = "Good"
    elif score >= 3.0:
        color = "🟡"
        status = "Adequate"
    else:
        color = "🔴"
        status = "Poor"
    
    # Only display score metric if show_details is True (to avoid duplication)
    display_label = get_category_display_name(category)

    if show_details:
        st.metric(
            label=display_label,
            value=f"{score}/5",
            delta=f"{status} {color}"
        )
    
    # Display details either in expander or directly
    if show_details:
        # Wrap in expander
        with st.expander(f"Details for {display_label}"):
            # Key findings FIRST
            if score_data.get('key_findings'):
                st.write("**Key Points:**")
                for finding in score_data['key_findings']:
                    st.write(f"• {finding}")
            
            # Strengths and Weaknesses SECOND
            col1, col2 = st.columns(2)
            
            with col1:
                if score_data.get('strengths'):
                    st.write("**Strengths:**")
                    for strength in score_data['strengths']:
                        st.write(f"✓ {strength}")
            
            with col2:
                if score_data.get('weaknesses'):
                    st.write("**Weaknesses:**")
                    for weakness in score_data['weaknesses']:
                        st.write(f"✗ {weakness}")
            
            # Assessment (Reasoning) LAST
            if score_data.get('reasoning'):
                st.write("**Assessment:**")
                st.write(score_data['reasoning'])
            
            # Sources
            if score_data.get('sources'):
                display_source_details(score_data['sources'])
    else:
        # Show details directly without expander
        # Key findings FIRST
        if score_data.get('key_findings'):
            st.write("**Key Points:**")
            for finding in score_data['key_findings']:
                st.write(f"• {finding}")
        
        # Strengths and Weaknesses SECOND
        col1, col2 = st.columns(2)
        
        with col1:
            if score_data.get('strengths'):
                st.write("**Strengths:**")
                for strength in score_data['strengths']:
                    st.write(f"✓ {strength}")
        
        with col2:
            if score_data.get('weaknesses'):
                st.write("**Weaknesses:**")
                for weakness in score_data['weaknesses']:
                    st.write(f"✗ {weakness}")
        
        # Assessment (Reasoning) LAST
        if score_data.get('reasoning'):
            st.write("**Assessment:**")
            st.write(score_data['reasoning'])
        
        # Sources
        if score_data.get('sources'):
            display_source_details(score_data['sources'])


def create_risk_heatmap(
    risk_matrix: Dict[str, List[Dict[str, Any]]],
    title: str = "Risk Heatmap"
) -> go.Figure:
    """
    Create a heatmap visualization of risks by category and severity
    
    Args:
        risk_matrix: Risk matrix organized by category
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    categories = []
    high_counts = []
    medium_counts = []
    low_counts = []
    
    for category, risks in risk_matrix.items():
        # Format category name
        cat_name = category.replace('_risks', '').replace('_', ' ').title()
        categories.append(cat_name)
        
        # Count by severity
        high = sum(1 for r in risks if r.get('severity') == 'High')
        medium = sum(1 for r in risks if r.get('severity') == 'Medium')
        low = sum(1 for r in risks if r.get('severity') == 'Low')
        
        high_counts.append(high)
        medium_counts.append(medium)
        low_counts.append(low)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='High Risk',
        x=categories,
        y=high_counts,
        marker_color='#d32f2f'
    ))
    
    fig.add_trace(go.Bar(
        name='Medium Risk',
        x=categories,
        y=medium_counts,
        marker_color='#f57c00'
    ))
    
    fig.add_trace(go.Bar(
        name='Low Risk',
        x=categories,
        y=low_counts,
        marker_color='#388e3c'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title="Risk Category",
        yaxis_title="Number of Risks",
        height=400
    )
    
    return fig


def create_parts_comparison_table(
    parts1: List[Dict[str, Any]],
    parts2: List[Dict[str, Any]],
    supplier1_name: str,
    supplier2_name: str
) -> pd.DataFrame:
    """
    Create a comparison table for parts from two suppliers
    
    Args:
        parts1: Parts data for supplier 1
        parts2: Parts data for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        
    Returns:
        DataFrame with parts comparison
    """
    # Create dictionaries for quick lookup
    parts1_dict = {p.get("part_number", ""): p for p in parts1}
    parts2_dict = {p.get("part_number", ""): p for p in parts2}
    
    # Get all unique part numbers
    all_part_numbers = set(parts1_dict.keys()) | set(parts2_dict.keys())
    
    comparison_data = []
    for part_num in sorted(all_part_numbers):
        if not part_num:  # Skip empty part numbers
            continue
            
        part1 = parts1_dict.get(part_num, {})
        part2 = parts2_dict.get(part_num, {})
        
        comparison_data.append({
            "Part Number": part_num,
            f"{supplier1_name} Name": part1.get("part_name", "-"),
            f"{supplier2_name} Name": part2.get("part_name", "-"),
            f"{supplier1_name} Price": f"€{part1.get('pricing', {}).get('unit_price', 'N/A')}" if part1.get("pricing", {}).get("unit_price") else "-",
            f"{supplier2_name} Price": f"€{part2.get('pricing', {}).get('unit_price', 'N/A')}" if part2.get("pricing", {}).get("unit_price") else "-",
            f"{supplier1_name} Lead Time": part1.get("capacity", {}).get("lead_time", "-"),
            f"{supplier2_name} Lead Time": part2.get("capacity", {}).get("lead_time", "-"),
        })
    
    return pd.DataFrame(comparison_data)


def display_part_details_card(
    part: Dict[str, Any],
    supplier_name: str,
    show_sources: bool = True
) -> None:
    """
    Display detailed information for a single part
    
    Args:
        part: Part data dictionary
        supplier_name: Name of the supplier
        show_sources: Whether to show source references
    """
    # Create a descriptive header with part name and supplier
    part_name = part.get('part_name', part.get('part_number', 'Unknown Part'))
    header_text = f"{part_name} - {supplier_name}"
    
    st.markdown(f"### {header_text}")
    
    # Show part number as secondary information if we used the name in header
    if part.get('part_name') and part.get('part_number'):
        st.markdown(f"**Part Number:** {part.get('part_number')}")
    elif not part.get('part_name') and part.get('part_number'):
        # If we only have part number (already shown in header), show supplier
        st.markdown(f"**Supplier:** {supplier_name}")
    
    if part.get("description"):
        st.write(part["description"])
    
    # Technical specifications
    if part.get("technical_specifications"):
        specs = part["technical_specifications"]
        st.markdown("**Technical Specifications:**")
        
        # Display all specifications in a single column vertical list
        if specs.get("dimensions"):
            st.write(f"• Dimensions: {specs['dimensions']}")
        if specs.get("weight"):
            st.write(f"• Weight: {specs['weight']}")
        if specs.get("material"):
            st.write(f"• Material: {specs['material']}")
        if specs.get("torque_capacity"):
            st.write(f"• Torque Capacity: {specs['torque_capacity']}")
        if specs.get("operating_temperature"):
            st.write(f"• Operating Temp: {specs['operating_temperature']}")
        if specs.get("other_specs"):
            for spec in specs["other_specs"]:
                st.write(f"• {spec}")
    
    # Pricing information
    if part.get("pricing"):
        pricing = part["pricing"]
        st.markdown("**Pricing:**")
        if pricing.get("unit_price"):
            st.metric("Unit Price", f"€{pricing['unit_price']:,.2f}")
        
        if pricing.get("volume_pricing"):
            st.markdown("*Volume Pricing:*")
            for vp in pricing["volume_pricing"]:
                st.write(f"• {vp.get('volume', 'N/A')}: €{vp.get('price_per_unit', 'N/A')}")
    
    # Certifications
    if part.get("certifications"):
        st.markdown("**Certifications:**")
        # Display certifications in a single column vertical list
        for cert in part["certifications"]:
            st.info(f"📋 {cert.get('standard', 'Unknown')}")
    
    # Capacity information
    if part.get("capacity"):
        cap = part["capacity"]
        st.markdown("**Production Capacity:**")
        
        # Display all capacity information in a single column vertical list
        if cap.get("production_capacity"):
            st.write(f"• Capacity: {cap['production_capacity']}")
        if cap.get("lead_time"):
            st.write(f"• Lead Time: {cap['lead_time']}")
        if cap.get("current_utilization"):
            st.write(f"• Utilization: {cap['current_utilization']}")
        if cap.get("min_order_quantity"):
            st.write(f"• MOQ: {cap['min_order_quantity']}")
    
    # Source references
    if show_sources and part.get("sources"):
        create_source_badge(part["sources"])  # Show all sources without limit


def create_parts_specifications_chart(
    parts1: List[Dict[str, Any]],
    parts2: List[Dict[str, Any]],
    supplier1_name: str,
    supplier2_name: str,
    metric: str = "price"
) -> go.Figure:
    """
    Create a bar chart comparing parts specifications
    
    Args:
        parts1: Parts data for supplier 1
        parts2: Parts data for supplier 2
        supplier1_name: Name of supplier 1
        supplier2_name: Name of supplier 2
        metric: Metric to compare ('price', 'lead_time', 'capacity')
        
    Returns:
        Plotly figure object
    """
    # Create dictionaries for quick lookup
    parts1_dict = {p.get("part_number", ""): p for p in parts1}
    parts2_dict = {p.get("part_number", ""): p for p in parts2}
    
    # Get common part numbers
    common_parts = set(parts1_dict.keys()) & set(parts2_dict.keys())
    common_parts = sorted([p for p in common_parts if p])[:10]  # Limit to 10 for visibility
    
    values1 = []
    values2 = []
    
    for part_num in common_parts:
        part1 = parts1_dict[part_num]
        part2 = parts2_dict[part_num]
        
        if metric == "price":
            val1 = part1.get("pricing", {}).get("unit_price", 0)
            val2 = part2.get("pricing", {}).get("unit_price", 0)
        elif metric == "lead_time":
            # Convert lead time to numeric days (simplified)
            lt1 = part1.get("capacity", {}).get("lead_time", "0 days")
            lt2 = part2.get("capacity", {}).get("lead_time", "0 days")
            val1 = int("".join(filter(str.isdigit, str(lt1))) or 0)
            val2 = int("".join(filter(str.isdigit, str(lt2))) or 0)
        else:
            val1 = 0
            val2 = 0
        
        values1.append(val1)
        values2.append(val2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=supplier1_name,
        x=common_parts,
        y=values1,
        marker_color='#1f77b4',
        text=values1,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name=supplier2_name,
        x=common_parts,
        y=values2,
        marker_color='#ff7f0e',
        text=values2,
        textposition='auto',
    ))
    
    title = {
        "price": "Unit Price Comparison (EUR)",
        "lead_time": "Lead Time Comparison (Days)",
        "capacity": "Production Capacity Comparison"
    }.get(metric, "Parts Comparison")
    
    fig.update_layout(
        title=title,
        xaxis_title="Part Number",
        yaxis_title=metric.replace("_", " ").title(),
        barmode="group",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def display_certification_badges(
    certifications: List[Dict[str, str]],
    max_display: int = 5
) -> None:
    """
    Display certification badges in a compact format
    
    Args:
        certifications: List of certification dictionaries
        max_display: Maximum number of certifications to display
    """
    if not certifications:
        st.write("No certifications available")
        return
    
    # Create columns for badges
    cols = st.columns(min(len(certifications), max_display))
    
    for i, cert in enumerate(certifications[:max_display]):
        with cols[i]:
            standard = cert.get("standard", "Unknown")
            description = cert.get("description", "")
            
            # Create a colored badge based on certification type
            if "ISO" in standard:
                color = '#4CAF50'  # Green for ISO
            elif "IATF" in standard:
                color = '#2196F3'  # Blue for IATF
            elif "CVS" in standard:
                color = '#FF9800'  # Orange for client-specific standards
            else:
                color = '#9E9E9E'  # Grey for others
            
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    text-align: center;
                    margin: 2px;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {standard}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if description:
                st.caption(description[:50] + "..." if len(description) > 50 else description)
    
    if len(certifications) > max_display:
        st.caption(f"... +{len(certifications) - max_display} more certifications")


def create_parts_category_distribution(
    parts_data: Dict[str, Any],
    supplier_name: str
) -> go.Figure:
    """
    Create a pie chart showing distribution of parts by category
    
    Args:
        parts_data: Parts analysis data including summary
        supplier_name: Name of the supplier
        
    Returns:
        Plotly figure object
    """
    categories = {}
    
    for part in parts_data.get("parts", []):
        category = part.get("category", "Uncategorized")
        categories[category] = categories.get(category, 0) + 1
    
    if not categories:
        categories = {"No parts data": 1}
    
    fig = go.Figure(data=[go.Pie(
        labels=list(categories.keys()),
        values=list(categories.values()),
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f"{supplier_name} - Parts by Category",
        height=400,
        showlegend=True
    )
    
    return fig


def render_floating_chat(
    title: str = "APP SUPPORT",
    toggle_emoji: str = "💬",
    state_key: str = "floating_chat_messages",
    assistant_greeting: str = "Hi, I'm your RFQ assistant. How can I help?",
) -> None:
    """
    Render a CSS-only floating chat widget that can be invoked from any page.

    This component persists conversation in Streamlit session state and uses
    a simple query-parameter flow (name: "say") to submit messages without JS.

    Args:
        title: Header title displayed at the top of the chat panel
        toggle_emoji: Emoji/icon shown on the round floating toggle button
        state_key: Session state key for storing the chat messages
        assistant_greeting: Initial assistant greeting shown on first load

    Notes:
        - This is a mock implementation that echoes user input as a placeholder.
        - Later you can replace the echo with an actual LLM call.
    """
    # Initialize chat state with a greeting from the assistant
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {"role": "assistant", "content": assistant_greeting}
        ]

    # Read message from query param "say" and immediately clear it
    msg_value: Optional[str] = None

    try:
        # Streamlit >= 1.31 style
        param = getattr(st, "query_params", None)
        if param is not None:
            raw = param.get("say")
            if raw:
                if isinstance(raw, list):
                    msg_value = raw[0]
                else:
                    msg_value = str(raw)
    except Exception:
        # Fallback for older versions
        try:
            params_legacy = st.experimental_get_query_params()  # type: ignore[attr-defined]
            if "say" in params_legacy and params_legacy["say"]:
                msg_value = params_legacy["say"][0]
        except Exception:
            msg_value = None

    if msg_value is not None and str(msg_value).strip():
        user_text = str(msg_value).strip()
        st.session_state[state_key].append({"role": "user", "content": user_text})

        # Placeholder bot logic (echo). Replace with LLM call later.
        placeholder_reply = f"(preview) Echo: {user_text}"
        st.session_state[state_key].append({
            "role": "assistant",
            "content": placeholder_reply,
        })

        # Clear the query parameter to avoid duplicate handling on rerun
        try:
            if getattr(st, "query_params", None) is not None:
                st.query_params.clear()
            else:
                st.experimental_set_query_params()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Helper to render messages as simple HTML bubbles
    def _render_messages_html() -> str:
        chunks: List[str] = []
        for message in st.session_state[state_key]:
            role = message.get("role", "assistant")
            content = html.escape(str(message.get("content", "")))
            cls = "user" if role == "user" else "ai"
            chunks.append(
                f"<div class='msg {cls}'><div class='bubble'>{content}</div></div>"
            )
        return "\n".join(chunks)

    messages_html = _render_messages_html()

    # Prepare Joule logo image for the floating toggle as inline base64.
    # Falls back to the provided emoji if the image is unavailable.
    toggle_img_html = None
    try:
        import base64
        from pathlib import Path

        # Resolve path: prototype/dashboard_components.py -> template_dashboard/static/images/Joule_logo.webp
        base_dir = Path(__file__).resolve().parent
        joule_path = base_dir / "template_dashboard" / "static" / "images" / "icon2.png"
        # joule_path = base_dir / "template_dashboard" / "static" / "images" / "Joule_logo.webp"
        if joule_path.exists():
            with open(joule_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            toggle_img_html = (
                f"<img src='data:image/png;base64,{b64}' alt='Chat' "
                f"style='width:28px;height:28px;object-fit:contain;'/>"
            )
    except Exception:
        toggle_img_html = None

    # Inject minimal CSS and HTML for the floating chat
    st.markdown(
        f"""
<style>
/* Toggle checkbox is visually hidden */
#chat_toggle {{ display: none; }}

/* Floating circular toggle button */
.chat-toggle {{
  position: fixed; bottom: 20px; right: 20px;
  width: 56px; height: 56px; border-radius: 50%;
  border: 0; box-shadow: 0 8px 24px rgba(0,0,0,.2);
  background: #e5e7eb; color: #111827; font-size: 24px; cursor: pointer;
  z-index: 9999; display:flex; align-items:center; justify-content:center;
}}

/* Chat panel container */
.chatbox {{
  position: fixed; bottom: 88px; right: 20px; width: 340px; max-height: 65vh;
  background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
  box-shadow: 0 16px 40px rgba(0,0,0,.18); z-index: 9999;
  display:flex; flex-direction:column; overflow:hidden;
}}
.chatbox .header {{ padding: 10px 12px; font-weight: 600; background:#f8fafc; border-bottom:1px solid #eef2f7; }}
.chatbox .msgs {{ padding: 10px; overflow-y: auto; flex: 1; background:#fbfdff; }}
.msg {{ display:flex; margin: 8px 0; }}
.msg.user {{ justify-content: flex-end; }}
.bubble {{ max-width: 80%; padding: 8px 10px; border-radius: 12px; line-height: 1.3; }}
.msg.user .bubble {{ background:#e0e7ff; border:1px solid #c7d2fe; }}
.msg.ai .bubble   {{ background:#f1f5f9; border:1px solid #e2e8f0; }}
.chatbox .input {{ border-top:1px solid #eef2f7; padding: 8px; background:#fff; }}
.chatbox input[type=text] {{ width:100%; padding:10px 12px; border:1px solid #e5e7eb; border-radius: 10px; outline:none; }}

/* Pure CSS toggle using a hidden checkbox */
#chat_toggle:not(:checked) ~ .chatbox {{ display:none; }}
</style>

<input type="checkbox" id="chat_toggle">
<label class="chat-toggle" for="chat_toggle">{toggle_img_html or html.escape(toggle_emoji)}</label>

<div class="chatbox">
  <div class="header">{html.escape(title)}</div>
  <div class="msgs">
    {messages_html}
  </div>
  <div class="input">
    <form method="get">
      <input name="say" type="text" placeholder="Type and press Enter…" autocomplete="off">
    </form>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
