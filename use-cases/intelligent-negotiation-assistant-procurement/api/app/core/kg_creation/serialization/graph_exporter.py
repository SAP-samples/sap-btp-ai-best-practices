"""
Graph serialization and export module for the Knowledge Graph Creation Pipeline.

This module implements Phase 5 of the pipeline, serializing the final
Knowledge Graph into various portable formats.
"""

import json
import gzip
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import networkx as nx

from ..models.kg_schema import KnowledgeGraph, Node, Relationship


logger = logging.getLogger(__name__)


class GraphExporter:
    """
    Exports Knowledge Graphs to various formats.
    
    Supports GraphML (compressed), JSON, and human-readable summaries.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the graph exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_graphml(self, 
                      knowledge_graph: KnowledgeGraph,
                      filename: str = "knowledge_graph.graphml.gz") -> str:
        """
        Export knowledge graph as compressed GraphML format.
        
        GraphML is a standard format supported by most graph analysis tools.
        
        Args:
            knowledge_graph: The KnowledgeGraph to export
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting to GraphML: {filename}")
        
        try:
            # Build NetworkX graph
            nx_graph = self._build_networkx_graph(knowledge_graph)
            
            # Prepare for GraphML export
            prepared_graph = self._prepare_graph_for_graphml(nx_graph)
            
            # Export with compression
            output_path = self.output_dir / filename
            with gzip.open(output_path, 'wb') as f:
                nx.write_graphml_lxml(
                    prepared_graph, 
                    f, 
                    infer_numeric_types=True, 
                    prettyprint=True
                )
            
            logger.info(f"GraphML export complete: {output_path}")
            logger.info(f"  Nodes: {len(knowledge_graph.nodes)}")
            logger.info(f"  Relationships: {len(knowledge_graph.relationships)}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"GraphML export failed: {str(e)}")
            raise
    
    def export_json(self,
                   knowledge_graph: KnowledgeGraph,
                   filename: str = "knowledge_graph.json") -> str:
        """
        Export knowledge graph as structured JSON.
        
        Args:
            knowledge_graph: The KnowledgeGraph to export
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting to JSON: {filename}")
        
        try:
            # Convert to dictionary format
            export_data = knowledge_graph.to_dict()
            
            # Add export metadata
            export_data["export_metadata"] = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_version": "1.0",
                "exporter": "KG Creation Pipeline"
            }
            
            # Write JSON file
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON export complete: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            raise
    
    def export_summary(self,
                      knowledge_graph: KnowledgeGraph,
                      filename: str = "knowledge_graph_summary.txt") -> str:
        """
        Export human-readable summary of the knowledge graph.
        
        Args:
            knowledge_graph: The KnowledgeGraph to export
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting summary: {filename}")
        
        try:
            summary = self._generate_text_summary(knowledge_graph)
            
            # Write summary file
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.info(f"Summary export complete: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Summary export failed: {str(e)}")
            raise
    
    def _build_networkx_graph(self, kg: KnowledgeGraph) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from the KnowledgeGraph.
        
        Args:
            kg: The KnowledgeGraph to convert
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in kg.nodes:
            # Prepare node attributes
            node_attrs = {
                'node_type': node.type.value,
                **node.properties  # Include all properties
            }
            
            # Add metadata as a serialized list
            if node.metadata:
                node_attrs['source_metadata'] = json.dumps([
                    {"filename": m.filename, "chunk_id": m.chunk_id}
                    for m in node.metadata
                ])
            
            G.add_node(node.id, **node_attrs)
        
        # Add edges with attributes
        for rel in kg.relationships:
            # Prepare edge attributes
            edge_attrs = {
                'label': rel.label,
                **rel.properties  # Include all properties
            }
            
            # Add metadata
            if rel.metadata:
                edge_attrs['source_metadata'] = json.dumps({
                    "filename": rel.metadata.filename,
                    "chunk_id": rel.metadata.chunk_id
                })
            
            G.add_edge(rel.source, rel.target, **edge_attrs)
        
        return G
    
    def _prepare_graph_for_graphml(self, graph: nx.Graph, prop_prefix: str = "prop_") -> nx.Graph:
        """
        Prepare graph for GraphML export by flattening complex attributes.
        
        GraphML has limitations on attribute types, so we need to:
        - Flatten nested dictionaries
        - Convert complex types to strings
        - Handle None values
        
        Args:
            graph: NetworkX graph to prepare
            prop_prefix: Prefix for flattened properties
            
        Returns:
            Prepared graph safe for GraphML export
        """
        export_graph = deepcopy(graph)
        
        # Process nodes
        for node_id, data in export_graph.nodes(data=True):
            self._flatten_attributes(data, prop_prefix)
        
        # Process edges
        for u, v, data in export_graph.edges(data=True):
            self._flatten_attributes(data, prop_prefix)
        
        return export_graph
    
    def _flatten_attributes(self, attrs: Dict[str, Any], prefix: str = "prop_") -> None:
        """
        Flatten complex attributes in-place for GraphML compatibility.
        
        Args:
            attrs: Dictionary of attributes to flatten
            prefix: Prefix for nested properties
        """
        keys_to_process = list(attrs.keys())
        
        for key in keys_to_process:
            value = attrs[key]
            
            if value is None:
                attrs[key] = ""  # Convert None to empty string
                
            elif isinstance(value, dict):
                # Special handling for StructuredValue objects
                if "value" in value and "unit" in value:
                    attrs[key] = f"{value['value']} {value['unit']}"
                else:
                    # Flatten nested dictionary
                    for sub_key, sub_value in value.items():
                        new_key = f"{prefix}{key}_{sub_key}"
                        attrs[new_key] = str(sub_value) if sub_value is not None else ""
                    del attrs[key]
                    
            elif isinstance(value, list):
                # Convert lists to JSON strings
                attrs[key] = json.dumps(value)
                
            elif isinstance(value, bool):
                # Convert booleans to strings
                attrs[key] = str(value)
            
            # Other types (str, int, float) are GraphML-compatible
    
    def _generate_text_summary(self, kg: KnowledgeGraph) -> str:
        """
        Generate a comprehensive text summary of the knowledge graph.
        
        Args:
            kg: The KnowledgeGraph to summarize
            
        Returns:
            Text summary
        """
        lines = []
        
        # Header
        lines.append("KNOWLEDGE GRAPH SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")
        
        # Overview statistics
        lines.append("OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"Total Nodes: {len(kg.nodes)}")
        lines.append(f"Total Relationships: {len(kg.relationships)}")
        lines.append("")
        
        # Node type breakdown
        node_types = {}
        for node in kg.nodes:
            node_type = node.type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        lines.append("NODE TYPES")
        lines.append("-" * 40)
        for node_type, count in sorted(node_types.items()):
            lines.append(f"  {node_type}: {count}")
        lines.append("")
        
        # Relationship type breakdown
        rel_types = {}
        for rel in kg.relationships:
            rel_types[rel.label] = rel_types.get(rel.label, 0) + 1
        
        lines.append("RELATIONSHIP TYPES")
        lines.append("-" * 40)
        for rel_type, count in sorted(rel_types.items()):
            lines.append(f"  {rel_type}: {count}")
        lines.append("")
        
        # Detailed node listing by type
        lines.append("DETAILED NODE LISTING")
        lines.append("=" * 80)
        
        # Group nodes by type
        nodes_by_type = {}
        for node in kg.nodes:
            if node.type.value not in nodes_by_type:
                nodes_by_type[node.type.value] = []
            nodes_by_type[node.type.value].append(node)
        
        # List nodes by type
        for node_type in sorted(nodes_by_type.keys()):
            nodes = sorted(nodes_by_type[node_type], key=lambda n: n.id)
            
            lines.append(f"\n{node_type.upper()} NODES ({len(nodes)})")
            lines.append("-" * 40)
            
            for node in nodes:
                lines.append(f"\n• {node.id}")
                
                # Properties
                if node.properties:
                    for key, value in sorted(node.properties.items()):
                        # Format value
                        if isinstance(value, dict) and "value" in value and "unit" in value:
                            formatted_value = f"{value['value']} {value['unit']}"
                        else:
                            formatted_value = str(value)
                            if len(formatted_value) > 100:
                                formatted_value = formatted_value[:97] + "..."
                        
                        lines.append(f"  - {key}: {formatted_value}")
                
                # Source metadata
                if node.metadata:
                    sources = [f"{m.filename}:{m.chunk_id}" for m in node.metadata]
                    lines.append(f"  - Sources: {', '.join(sources)}")
        
        # Relationship details
        lines.append("\n\nRELATIONSHIP DETAILS")
        lines.append("=" * 80)
        
        # Group relationships by type
        rels_by_type = {}
        for rel in kg.relationships:
            if rel.label not in rels_by_type:
                rels_by_type[rel.label] = []
            rels_by_type[rel.label].append(rel)
        
        # List relationships by type
        for rel_type in sorted(rels_by_type.keys()):
            rels = rels_by_type[rel_type]
            
            lines.append(f"\n{rel_type} ({len(rels)} instances)")
            lines.append("-" * 40)
            
            # Show up to 10 examples
            for rel in rels[:10]:
                lines.append(f"  {rel.source} → {rel.target}")
                
                if rel.properties:
                    for key, value in sorted(rel.properties.items()):
                        lines.append(f"    - {key}: {value}")
            
            if len(rels) > 10:
                lines.append(f"  ... and {len(rels) - 10} more")
        
        # Special sections for requirements
        requirements = [n for n in kg.nodes if n.type.value == "Requirement"]
        if requirements:
            lines.append("\n\nREQUIREMENT ANALYSIS")
            lines.append("=" * 80)
            
            # Strategy breakdown
            matching_strategies = {}
            comparison_strategies = {}
            tqsdc_categories = {}
            
            for req in requirements:
                props = req.properties
                
                ms = props.get("matching_strategy", "Unknown")
                matching_strategies[ms] = matching_strategies.get(ms, 0) + 1
                
                cs = props.get("comparison_strategy", "Unknown")
                comparison_strategies[cs] = comparison_strategies.get(cs, 0) + 1
                
                cat = props.get("tqsdc_category", "Unknown")
                tqsdc_categories[cat] = tqsdc_categories.get(cat, 0) + 1
            
            lines.append("Matching Strategies:")
            for strategy, count in sorted(matching_strategies.items()):
                lines.append(f"  {strategy}: {count}")
            
            lines.append("\nComparison Strategies:")
            for strategy, count in sorted(comparison_strategies.items()):
                lines.append(f"  {strategy}: {count}")
            
            lines.append("\nTQSDC Categories:")
            for category, count in sorted(tqsdc_categories.items()):
                lines.append(f"  {category}: {count}")
        
        return "\n".join(lines)


def export_knowledge_graph(knowledge_graph: KnowledgeGraph,
                         output_dir: str = "./output",
                         formats: List[str] = ["graphml", "json", "summary"]) -> Dict[str, str]:
    """
    Convenience function to export a knowledge graph to multiple formats.
    
    Args:
        knowledge_graph: The graph to export
        output_dir: Output directory
        formats: List of formats to export
        
    Returns:
        Dictionary mapping format to output file path
    """
    exporter = GraphExporter(output_dir)
    output_files = {}
    
    if "graphml" in formats:
        output_files["graphml"] = exporter.export_graphml(knowledge_graph)
    
    if "json" in formats:
        output_files["json"] = exporter.export_json(knowledge_graph)
    
    if "summary" in formats:
        output_files["summary"] = exporter.export_summary(knowledge_graph)
    
    return output_files