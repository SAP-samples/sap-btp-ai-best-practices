"""
Advanced Graph Explorer Page

This page provides an advanced interactive graph visualization using st-link-analysis
for exploring RFQ document relationships with enhanced features like node highlighting,
filtering, search, and detailed node information.
"""

import streamlit as st
import networkx as nx
from typing import Dict, List

# Import our graph processor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from graph_processor import GraphProcessor
from main import SimplifiedRFQComparator

# Import st-link-analysis for advanced graph visualization
try:
    from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle
except ImportError:
    st.error("st-link-analysis not installed. Please run: pip install st-link-analysis")
    st.stop()

st.set_page_config(
    page_title="Knowledge Graph Explorer - SAP RFQx",
    layout="wide"
)

# Page title
st.title("Knowledge Graph Explorer")

# Check if we have graph data in session state
if 'providers' not in st.session_state or not st.session_state.providers:
    st.warning("No document data found. Please process documents first in the main application.")
    if st.button("← Go to Main Application"):
        st.switch_page("app.py")
    st.stop()

# Get completed providers
completed_providers = [p for p in st.session_state.providers if p.get('extracted_data')]

if not completed_providers:
    st.warning("No processed documents found. Please complete document processing first.")
    if st.button("← Go to Main Application"):
        st.switch_page("app.py")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("Graph Controls")
    
    # Document filtering
    st.subheader("Document Filter")
    available_docs = [p['name'] for p in completed_providers]
    selected_docs = st.multiselect(
        "Select documents to display:",
        available_docs,
        default=available_docs,
        help="Choose which documents to include in the graph"
    )
    
    # Layout options
    st.subheader("Layout Settings")
    layout_options = {
        "cose": "Compound Spring Embedder (COSE)",
        "cola": "Constraint-Based Layout",
        "grid": "Grid Layout",
        "circle": "Circular Layout",
        "breadthfirst": "Breadth-First Tree",
        "concentric": "Concentric Circles"
    }
    
    selected_layout = st.selectbox(
        "Graph Layout:",
        list(layout_options.keys()),
        format_func=lambda x: layout_options[x],
        index=0
    )
    
    # Node filtering
    st.subheader("Node Filtering")
    node_types = ["Document", "ComplexEntity", "ComplexItem", "Value"]
    selected_node_types = st.multiselect(
        "Show node types:",
        node_types,
        default=node_types,
        help="Filter nodes by their type"
    )
    
    # Graph statistics
    st.subheader("Graph Statistics")

@st.cache_resource
def get_cached_comparator():
    """Get cached SimplifiedRFQComparator instance"""
    return SimplifiedRFQComparator()

@st.cache_resource  
def get_cached_graph_processor():
    """Get cached GraphProcessor instance"""
    return GraphProcessor(llm_client=get_cached_comparator().client)

def convert_networkx_to_cytoscape(selected_docs: List[str], selected_node_types: List[str]) -> Dict:
    """Convert NetworkX graphs to st-link-analysis format"""
    
    # Color mapping for documents
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    doc_colors = {doc: colors[i % len(colors)] for i, doc in enumerate(selected_docs)}
    
    nodes = []
    edges = []
    edge_count = 0
    node_count = 0
    
    # Process each selected document
    for doc_name in selected_docs:
        if doc_name not in [p['name'] for p in completed_providers]:
            continue
            
        # Find the provider data
        provider_data = next(p['extracted_data'] for p in completed_providers if p['name'] == doc_name)
        
        # Create graph from this document (use cached comparator)
        graph_processor = get_cached_graph_processor()
        graph = graph_processor.create_graph_from_json(provider_data)
        
        # Add nodes
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            
            # Apply node type filter
            if node_type not in selected_node_types:
                continue
                
            full_content = node_data.get('full_content', node_id)
            label = node_data.get('label', node_id)
            
            # Create node for st-link-analysis (no inline styling)
            nodes.append({
                'data': {
                    'id': f"{doc_name}_{node_id}",
                    'label': node_type,  # This will be used for NodeStyle matching
                    'name': label,       # Display name
                    'full_content': full_content,
                    'node_type': node_type,
                    'document': doc_name,
                    'doc_color': doc_colors.get(doc_name, '#666666')
                }
            })
            node_count += 1
        
        # Add edges
        for source, target, edge_data in graph.edges(data=True):
            # Check if both nodes are included (based on type filter)
            source_type = graph.nodes[source].get('type', 'Unknown')
            target_type = graph.nodes[target].get('type', 'Unknown')
            
            if source_type not in selected_node_types or target_type not in selected_node_types:
                continue
                
            relationship = edge_data.get('label', 'connected_to')
            
            # Create edge for st-link-analysis (no inline styling)
            edges.append({
                'data': {
                    'id': f"{doc_name}_{source}_{target}_{edge_count}",
                    'source': f"{doc_name}_{source}",
                    'target': f"{doc_name}_{target}",
                    'label': relationship,  # This will be used for EdgeStyle matching
                    'relationship': relationship,
                    'document': doc_name
                }
            })
            edge_count += 1
    
    # Update statistics in sidebar
    with st.sidebar:
        st.metric("Total Nodes", node_count)
        st.metric("Total Edges", edge_count)
        st.metric("Documents", len(selected_docs))
    
    # Create NodeStyle objects for styling
    node_styles = []
    edge_styles = []
    
    # Create styles for each node type
    for node_type in selected_node_types:
        if node_type == 'Document':
            node_styles.append(NodeStyle(node_type, '#2E86AB', 'name', 'description'))
        elif node_type == 'ComplexEntity':
            node_styles.append(NodeStyle(node_type, '#A23B72', 'name', 'person'))
        elif node_type == 'ComplexItem':
            node_styles.append(NodeStyle(node_type, '#F18F01', 'name', 'inventory'))
        elif node_type == 'Value':
            node_styles.append(NodeStyle(node_type, '#C73E1D', 'name', 'description'))
        else:
            node_styles.append(NodeStyle(node_type, '#666666', 'name', 'folder'))
    
    # Create styles for different relationship types
    relationship_types = set()
    for edge in edges:
        relationship_types.add(edge['data']['label'])
    
    for rel_type in relationship_types:
        edge_styles.append(EdgeStyle(rel_type, caption='label', directed=True))
    
    return {
        'elements': {
            'nodes': nodes,
            'edges': edges
        },
        'node_styles': node_styles,
        'edge_styles': edge_styles,
        'layout': selected_layout,
        'doc_colors': doc_colors
    }

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    
    if selected_docs:
        # Convert graph data to cytoscape format
        with st.spinner("Building interactive graph..."):
            graph_data = convert_networkx_to_cytoscape(
                selected_docs=selected_docs,
                selected_node_types=selected_node_types
            )
        
        # Display the interactive graph
        if graph_data['elements']['nodes']:
            try:
                selected_elements = st_link_analysis(
                    graph_data['elements'],
                    layout=graph_data['layout'],
                    node_styles=graph_data['node_styles'],
                    edge_styles=graph_data['edge_styles']
                )
            except Exception as e:
                st.error(f"Error displaying interactive graph: {str(e)}")
                st.info("Falling back to basic graph information display.")
                
                # Fallback: display basic graph information
                st.write(f"**Nodes:** {len(graph_data['elements']['nodes'])}")
                st.write(f"**Edges:** {len(graph_data['elements']['edges'])}")
                st.write("**Documents:**", ', '.join(selected_docs))
                selected_elements = None
            
            # Handle selected elements
            if selected_elements:
                with col2:
                    st.subheader("Selected Element")
                    
                    if 'nodes' in selected_elements and selected_elements['nodes']:
                        node = selected_elements['nodes'][0]
                        st.write("**Node Information:**")
                        st.write(f"**Label:** {node.get('label', 'N/A')}")
                        st.write(f"**Type:** {node.get('node_type', 'N/A')}")
                        st.write(f"**Document:** {node.get('document', 'N/A')}")
                        st.write(f"**Full Content:** {node.get('full_content', 'N/A')}")
                    
                    if 'edges' in selected_elements and selected_elements['edges']:
                        edge = selected_elements['edges'][0]
                        st.write("**Edge Information:**")
                        st.write(f"**Relationship:** {edge.get('relationship', 'N/A')}")
                        st.write(f"**Document:** {edge.get('document', 'N/A')}")
        else:
            st.warning("No graph elements to display with current filters.")
    else:
        st.warning("Please select at least one document to display the graph.")

with col2:
    if not (selected_docs and 'selected_elements' in locals() and selected_elements):
        st.write("")  # Empty space when no elements are selected

# Action buttons
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    if st.button("Back to Main App", type="secondary"):
        st.switch_page("app.py")

with col_b:
    if st.button("Refresh Graph"):
        st.rerun()
