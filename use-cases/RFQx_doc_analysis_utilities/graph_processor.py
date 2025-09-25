"""
Module for processing RFQ knowledge graphs.

This module contains the logic for:
1. Converting extracted data in JSON format to a knowledge graph (NetworkX).
2. Serializing the graph to a text format for injection into an LLM prompt.
3. Generating a comparative report using the LLM with the graph context.
"""

import hashlib
import json
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

# Assume that llm_client.py is in the same directory or in the python path
from llm_client import SimplifiedRAGClient
# Import the GenAI Hub interface for direct LLM calls
from gen_ai_hub.proxy.native.openai import chat
# Import the Country Risk Manager for risk analysis
from country_risk_manager import CountryRiskManager

class GraphProcessor:
    """
    Manages the creation of graphs from RFQ data and generates comparisons.
    """

    def __init__(self, llm_client: SimplifiedRAGClient):
        """
        Initializes the graph processor.

        Args:
            llm_client: An instance of the LLM client to make API calls.
        """
        if not isinstance(llm_client, SimplifiedRAGClient):
            raise TypeError("llm_client must be an instance of SimplifiedRAGClient")
        self.llm_client = llm_client
        self.country_risk_manager = CountryRiskManager()
        print("Initialized GraphProcessor.")

    def _truncate_label(self, text: str, max_length: int = 20) -> str:
        """
        Truncates text to a maximum length, adding ellipsis if needed.
        
        Args:
            text: The text to truncate.
            max_length: Maximum allowed length (default 20).
            
        Returns:
            Truncated text with ellipsis if needed.
        """
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    def create_graph_from_json(self, json_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Converts a single JSON dictionary of an RFQ into a directed graph.

        Args:
            json_data: The extracted data from an RFQ document.

        Returns:
            An nx.DiGraph object representing the RFQ.
        """
        graph = nx.DiGraph()
        # Use the file name as the main ID for consistency
        document_id = json_data.get("_metadata", {}).get("source_document", "unknown_doc")
        
        graph.add_node(document_id, type="Document", label=self._truncate_label(document_id), full_content=document_id)
        
        self._build_graph_recursively(graph, json_data, document_id)
        
        return graph

    def _build_graph_recursively(self, graph: nx.DiGraph, data: Dict[str, Any], parent_node: str):
        """
        Recursive helper function to build the graph from JSON.

        Args:
            graph: The networkx graph object to modify.
            data: The dictionary (or sub-dictionary) to process.
            parent_node: The ID of the parent node to which new nodes will be connected.
        """
        for key, value in data.items():
            if key.startswith('_'):  # Ignore internal metadata
                continue

            predicate = key.replace(" ", "_") # Name of the relationship (edge)

            if isinstance(value, dict):
                # For nested objects, we create an intermediate node that represents the entity.
                object_node_id = f"{parent_node}_{key}"
                graph.add_node(object_node_id, type="ComplexEntity", label=self._truncate_label(key), full_content=key)
                graph.add_edge(parent_node, object_node_id, label=predicate)
                self._build_graph_recursively(graph, value, object_node_id)
            
            elif isinstance(value, list):
                # For lists, create a relationship for each element
                for index, item in enumerate(value):
                    item_node_id = f"{parent_node}_{key}_{index}"
                    if isinstance(item, dict):
                        graph.add_node(item_node_id, type="ComplexItem", label=self._truncate_label(f"{key}_{index}"), full_content=f"{key}_{index}")
                        graph.add_edge(parent_node, item_node_id, label=f"{predicate}_item")
                        self._build_graph_recursively(graph, item, item_node_id)
                    else:
                        if str(item) != "Not Found":
                            graph.add_node(str(item), type="Value", label=self._truncate_label(str(item)), full_content=str(item))
                            graph.add_edge(parent_node, str(item), label=f"{predicate}_item")
            
            elif str(value) != "Not Found":
                # For simple values, the value is a node and the key is the relationship.
                value_node = str(value)
                graph.add_node(value_node, type="Value", label=self._truncate_label(value_node), full_content=value_node)
                graph.add_edge(parent_node, value_node, label=predicate)


    def serialize_graphs_to_text(self, graphs: Dict[str, nx.DiGraph]) -> str:
        """
        Converts a dictionary of graphs into a single string of triplets.

        Args:
            graphs: A dictionary that maps document names to their graphs.

        Returns:
            A formatted text string to be used in an LLM prompt.
        """
        serialized_string = ""
        for doc_name, graph in graphs.items():
            serialized_string += f"# Document data: {doc_name}\n"
            triplets = []
            for u, v, attrs in graph.edges(data=True):
                # u: subject, v: object, attrs['label']: predicate
                predicate = attrs.get('label', 'unknown_relation')
                # Clean line breaks in nodes for a clean format
                clean_u = str(u).replace('\n', ' ').strip()
                clean_v = str(v).replace('\n', ' ').strip()
                triplets.append(f"('{clean_u}', '{predicate}', '{clean_v}')")
            
            serialized_string += "\n".join(triplets)
            serialized_string += "\n\n"
            
        return serialized_string.strip()

    def generate_comparison_report_from_graphs_streaming(self, documents_data: Dict[str, Dict[str, Any]]):
        """
        Orchestrates the complete graph-based comparison process with streaming support.

        Args:
            documents_data: Dictionary with document names as keys and
                            their extracted JSON data as values.

        Yields:
            String chunks of the comparative report as it is generated.
        """
        print("--- Starting graph-based comparison process (streaming) ---")
        
        # 1. Create a graph for each document
        print("1. Creating graphs from JSON data...")
        graphs = {name: self.create_graph_from_json(data) for name, data in documents_data.items()}
        print(f"   - {len(graphs)} graphs have been created.")

        # 2. Serialize the graphs to text
        print("2. Serializing graphs to text format for the LLM...")
        graph_context_text = self.serialize_graphs_to_text(graphs)
        print("   - Serialization completed.")

        # 3. Build the prompt for the LLM
        print("3. Building the prompt for comparison...")
        prompt = self._build_comparison_prompt(graph_context_text, documents_data)

        # 4. Call the LLM to get the report with streaming
        print("4. Sending request to LLM to generate the report (streaming)...")
        
        system_prompt = "You are an expert contract analyst. Your task is to analyze a knowledge graph representing multiple RFQ documents and generate a detailed, clear, and concise comparative report in Markdown format."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Use the LLM client's streaming method
        try:
            for chunk in self.llm_client.generate_completion_streaming(
                messages=messages,
                temperature=0.1
            ):
                yield chunk
                
            print("   - Streaming report generation completed.")
            
        except Exception as e:
            error_msg = f"Error generating streaming report: {e}"
            print(f"   - {error_msg}")
            yield error_msg
        
        print("--- Comparison process completed ---")

    def generate_comparison_report_from_graphs(self, documents_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Orchestrates the complete graph-based comparison process.

        Args:
            documents_data: Dictionary with document names as keys and
                            their extracted JSON data as values.

        Returns:
            The comparative report generated by the LLM.
        """
        print("--- Starting graph-based comparison process ---")
        
        # 1. Create a graph for each document
        print("1. Creating graphs from JSON data...")
        graphs = {name: self.create_graph_from_json(data) for name, data in documents_data.items()}
        print(f"   - {len(graphs)} graphs have been created.")

        # 2. Serialize the graphs to text
        print("2. Serializing graphs to text format for the LLM...")
        graph_context_text = self.serialize_graphs_to_text(graphs)
        print("   - Serialization completed.")
        # print("Serialized graph context:\n", graph_context_text) # Uncomment for debugging

        # 3. Build the prompt for the LLM
        print("3. Building the prompt for comparison...")
        prompt = self._build_comparison_prompt(graph_context_text, documents_data)

        # 4. Call the LLM to get the report
        print("4. Sending request to LLM to generate the report...")
        
        system_prompt = "You are an expert contract analyst. Your task is to analyze a knowledge graph representing multiple RFQ documents and generate a detailed, clear, and concise comparative report in Markdown format."
        
        # Use the actual LLM client with retry logic
        max_retries = 3
        report = f"Error: Could not get a response from the LLM after {max_retries} attempts."
        for attempt in range(max_retries):
            try:
                response = chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                report = response.choices[0].message.content.strip()
                print("   - Report received from LLM.")
                break
            except Exception as e:
                print(f"   - LLM request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"   - All {max_retries} attempts failed.")
                    report = f"Error generating report after multiple retries: {e}"
        
        print("--- Comparison process completed ---")
        return report

    def create_interactive_graph(self, graphs: Dict[str, nx.DiGraph]) -> go.Figure:
        """
        Creates an interactive Plotly graph visualization from multiple NetworkX graphs.
        
        Args:
            graphs: Dictionary mapping document names to their NetworkX graphs.
            
        Returns:
            A Plotly Figure object with interactive features.
        """
        # Create a combined graph for visualization
        combined_graph = nx.DiGraph()
        # Use more distinct colors for better visibility
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = {}
        doc_map = {}  # Map nodes to their document
        
        for i, (doc_name, graph) in enumerate(graphs.items()):
            doc_color = colors[i % len(colors)]
            
            # Add all nodes from this document's graph
            for node, attrs in graph.nodes(data=True):
                combined_graph.add_node(node, **attrs, document=doc_name, color=doc_color)
                color_map[node] = doc_color
                doc_map[node] = doc_name
                
            # Add all edges from this document's graph
            for u, v, attrs in graph.edges(data=True):
                combined_graph.add_edge(u, v, **attrs, document=doc_name)
        
        # Calculate layout
        pos = nx.spring_layout(combined_graph, seed=42, k=2.0, iterations=100)
        
        # Create edge information for hover effects
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in combined_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_label = edge[2].get('label', '')
            # Provide hover text for the line segment, None for the gap
            hover_text = f"{edge[0]} → {edge[1]}<br>Relation: {edge_label}"
            edge_info.extend([hover_text, hover_text, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=2, color='#666', dash='solid'),  # Slightly thicker and darker edges
            hoverinfo='text',
            hovertext=edge_info,
            showlegend=False,
            name='edges',
            opacity=0.7  # Slight transparency for better layering
        )
        
        # Prepare node data - organized by document for legend
        node_data_by_doc = {}
        for doc_name in graphs.keys():
            node_data_by_doc[doc_name] = {
                'x': [],
                'y': [],
                'text': [],
                'hovertext': [],
                'colors': []
            }
        
        # Add nodes to appropriate data structures
        for node in combined_graph.nodes():
            x, y = pos[node]
            node_info = combined_graph.nodes[node]
            doc_name = node_info.get('document', 'unknown')
            
            # Get full content for hover
            full_content = node_info.get('full_content', node)
            node_type = node_info.get('type', 'Unknown')
            label = node_info.get('label', node)
            
            # Get connected nodes information for enhanced hover
            incoming_edges = list(combined_graph.predecessors(node))
            outgoing_edges = list(combined_graph.successors(node))
            
            # Create detailed hover text with connections
            hover_text = f"<b>{full_content}</b><br>Type: {node_type}<br>Document: {doc_name}"
            
            if incoming_edges:
                hover_text += f"<br><br><b>Incoming from:</b><br>"
                for pred in incoming_edges[:3]:  # Limit to first 3 to avoid clutter
                    edge_data = combined_graph[pred][node]
                    relation = edge_data.get('label', 'connected to')
                    hover_text += f"• {pred} ({relation})<br>"
                if len(incoming_edges) > 3:
                    hover_text += f"• ... and {len(incoming_edges) - 3} more<br>"
            
            if outgoing_edges:
                hover_text += f"<br><b>Outgoing to:</b><br>"
                for succ in outgoing_edges[:3]:  # Limit to first 3 to avoid clutter
                    edge_data = combined_graph[node][succ]
                    relation = edge_data.get('label', 'connected to')
                    hover_text += f"• {succ} ({relation})<br>"
                if len(outgoing_edges) > 3:
                    hover_text += f"• ... and {len(outgoing_edges) - 3} more<br>"
            
            if doc_name in node_data_by_doc:
                node_data_by_doc[doc_name]['x'].append(x)
                node_data_by_doc[doc_name]['y'].append(y)
                node_data_by_doc[doc_name]['text'].append(label)
                node_data_by_doc[doc_name]['hovertext'].append(hover_text)
                node_data_by_doc[doc_name]['colors'].append(color_map.get(node, 'gray'))
        
        # Create figure
        fig = go.Figure()
        
        # Add edge trace
        fig.add_trace(edge_trace)
        
        # Create and add node traces
        for doc_name, data in node_data_by_doc.items():
            if data['x']:  # Only create trace if there are nodes
                node_trace = go.Scatter(
                    x=data['x'],
                    y=data['y'],
                    text=data['text'],
                    mode='markers+text',
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=data['hovertext'],
                    name=doc_name,
                    marker=dict(
                        size=25,  # Slightly larger nodes for better visibility
                        color=data['colors'],
                        line=dict(width=2, color='DarkSlateGrey'),
                        opacity=0.9  # Slight transparency for visual appeal
                    )
                )
                fig.add_trace(node_trace)
        
        # Create annotations for edge labels and arrows
        annotations = []
        for edge in combined_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_label = edge[2].get('label', '')
            
            # Annotation for the edge label (if label exists)
            if edge_label:
                annotations.append(
                    dict(
                        x=(x0 + x1) / 2,  # Midpoint of edge
                        y=(y0 + y1) / 2,
                        xref="x", yref="y",
                        text=edge_label,
                        showarrow=False,
                        font=dict(
                            size=9,
                            color="black"
                        ),
                        bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                        borderpad=2
                    )
                )
            
            # Annotation for the arrow (direction indicator)
            annotations.append(
                dict(
                    x=x1,  # Arrow head points to the target node
                    y=y1,
                    ax=x0,  # Arrow starts from the source node
                    ay=y0,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#666',
                    standoff=12,  # Distance from target node to arrow head
                    startstandoff=12  # Distance from source node to arrow tail
                )
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive Knowledge Graph - RFQ Documents Comparison",
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=annotations,  # Add the edge labels and arrows
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        # Add interactivity for highlighting connected nodes
        fig.update_traces(
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig

    def _build_comparison_prompt(self, graph_context: str, documents_data: Dict[str, Dict[str, Any]] = None) -> str:
        """Creates the final prompt to send to the LLM, including country risk analysis if available."""
        
        # Build country risk context if available
        country_risk_context = ""
        if documents_data:
            # Look for country information in the documents
            countries_found = set()
            for doc_name, doc_data in documents_data.items():
                project_info = doc_data.get('project_information', {})
                country = project_info.get('country_of_contracting_authority')
                if country and country != "Not Found":
                    countries_found.add(country)
            
            if countries_found:
                country_risk_context = "\n\n[COUNTRY RISK INFORMATION]\n"
                for country in countries_found:
                    risk_context = self.country_risk_manager.get_risk_context_for_llm(country)
                    if risk_context:
                        country_risk_context += f"{risk_context}\n\n"
                country_risk_context += "[END OF COUNTRY RISK INFORMATION]\n"
        
        return f"""
Below is a knowledge graph in triplet format (subject, predicate, object). Each section of triplets corresponds to a different RFQ document.

[START OF KNOWLEDGE GRAPH]
{graph_context}
[END OF KNOWLEDGE GRAPH]
{country_risk_context}
Based EXCLUSIVELY on the information from the provided graph{' and country risk data' if country_risk_context else ''}, perform the following tasks:

1.  **Executive Summary**: Write a brief paragraph summarizing the most important differences between the documents.
2.  **Detailed Comparative Table**: Create a table in Markdown format that compares key attributes. Include columns for "Attribute", the document names, and an "Observations" column to highlight differences. Compare at least the following attributes if present:
    - `submission_deadline`
    - `contract_duration`
    - `estimated_contract_value`
    - `country_of_contracting_authority`
    - `warranty` (if it exists in the original schema)
    - `penalties` (if they exist)
3.  **Strengths and Weaknesses Analysis**: For each document, briefly describe its most favorable and least favorable points from the contractor's perspective.
{f'4.  **Country Risk Analysis**: If country information is available, analyze the risk implications for contracting in the identified countries, including governance, security, infrastructure, and reliability factors based on the provided risk data.' if country_risk_context else ''}
{f'5.  **Conclusion**: Offer a recommendation on which RFQ might be most advantageous and why, considering both contract terms and country risk factors.' if country_risk_context else '4.  **Conclusion**: Offer a recommendation on which RFQ might be most advantageous and why.'}

Ensure that your response is well-structured and easy to read. Use Markdown format.
"""

# --- Example block to test the module independently with REAL functions ---
if __name__ == '__main__':
    # Import main comparator class here to avoid circular dependencies if this module grows
    from main import SimplifiedRFQComparator

    print("--- Running GraphProcessor test with REAL data ---")

    # 1. Instantiate the main comparator to process PDFs
    comparator = SimplifiedRFQComparator()

    # Define paths to the actual PDF files
    # Assumes the script is run from the project's root directory
    pdf_paths = [
        "uploads/Humber south bank bird surveys - Jan to March 2024 RFQ.pdf",
        "uploads/MSH RFQ.pdf"
    ]

    all_docs_data = {}

    # 2. Process each PDF to extract structured data (JSON)
    for pdf_path in pdf_paths:
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"ERROR: PDF file not found at {pdf_path}. Make sure you are running the script from the project root directory.")
            continue
        
        print(f"\n--- Processing document: {path_obj.name} ---")
        # process_single_document performs the text extraction and the LLM call to obtain the JSON
        data = comparator.process_single_document(pdf_path)
        
        if "processing_error" in data or "extraction_error" in data:
            print(f"ERROR processing {path_obj.name}. Skipping this file.")
            print(f"Details: {data.get('processing_error') or data.get('extraction_error')}")
        else:
            print(f"Successfully processed and extracted data from {path_obj.name}.")
            all_docs_data[path_obj.name] = data

    # 3. Proceed only if we have data for at least two documents
    if len(all_docs_data) >= 2:
        # Instantiate the GraphProcessor with the real LLM client from the comparator
        graph_processor = GraphProcessor(llm_client=comparator.client)
        
        # Generate the final comparison report from the extracted data
        final_report = graph_processor.generate_comparison_report_from_graphs(all_docs_data)

        # Print the final result
        print("\n\n================ FINAL COMPARATIVE REPORT ================\n")
        print(final_report)
        print("\n=========================================================\n")
        
        # Optional: Visualize one of the graphs (requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            
            # Select the first successfully processed document to visualize
            first_doc_name = list(all_docs_data.keys())[0]
            first_doc_data = all_docs_data[first_doc_name]
            
            G = graph_processor.create_graph_from_json(first_doc_data)
            
            plt.figure(figsize=(16, 14))
            pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)
            
            node_labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
            edge_labels = nx.get_edge_attributes(G, 'label')

            nx.draw(G, pos, labels=node_labels, with_labels=True, node_color='skyblue', 
                    node_size=2000, font_size=8, edge_color='gray', arrowsize=15)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
            
            plt.title(f"Graph Visualization for {first_doc_name}")
            print("\nClose the graph window to finish the script.")
            plt.show()

        except ImportError:
            print("\nTo visualize the graph, install matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"\nCould not generate graph visualization: {e}")

    else:
        print("\nCould not proceed with comparison. Need at least two successfully processed documents.")

