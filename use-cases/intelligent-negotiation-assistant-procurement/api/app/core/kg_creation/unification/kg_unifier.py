"""
Knowledge Graph Unifier

This module provides functionality to unify multiple knowledge graphs
into a single consolidated graph. It handles:
- Merging nodes with the same ID
- Deduplicating relationships
- Preserving all source metadata
- Combining properties intelligently
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

from ..models.kg_schema import KnowledgeGraph, Node, Relationship, SourceMetadata

logger = logging.getLogger(__name__)


class KGUnifier:
    """
    Unifies multiple knowledge graphs into a single consolidated graph.
    
    This class provides methods to:
    1. Unify knowledge graphs from memory
    2. Load and unify knowledge graphs from JSON files
    3. Handle conflicts in node properties
    4. Preserve all source metadata
    """
    
    def __init__(self, conflict_resolution: str = "last_wins"):
        """
        Initialize the KG unifier.
        
        Args:
            conflict_resolution: Strategy for resolving property conflicts
                - "last_wins": Later values override earlier ones (default)
                - "first_wins": Earlier values are kept
                - "merge_all": Keep all values as lists
        """
        self.conflict_resolution = conflict_resolution
        self.unification_stats = {
            "total_input_graphs": 0,
            "total_input_nodes": 0,
            "total_input_relationships": 0,
            "unified_nodes": 0,
            "unified_relationships": 0,
            "duplicate_nodes_merged": 0,
            "duplicate_relationships_removed": 0,
            "property_conflicts_resolved": 0
        }
    
    def unify_knowledge_graphs(self, 
                             knowledge_graphs: List[KnowledgeGraph],
                             unified_name: str = "unified") -> KnowledgeGraph:
        """
        Unify multiple knowledge graphs into a single graph.
        
        Args:
            knowledge_graphs: List of KnowledgeGraph objects to unify
            unified_name: Name for the unified graph (used in logging)
            
        Returns:
            Unified KnowledgeGraph
        """
        if not knowledge_graphs:
            logger.warning("No knowledge graphs provided for unification")
            return KnowledgeGraph(nodes=[], relationships=[])
        
        # Reset statistics
        self._reset_stats()
        self.unification_stats["total_input_graphs"] = len(knowledge_graphs)
        
        # Count input statistics
        for kg in knowledge_graphs:
            self.unification_stats["total_input_nodes"] += len(kg.nodes)
            self.unification_stats["total_input_relationships"] += len(kg.relationships)
        
        logger.info(f"Unifying {len(knowledge_graphs)} knowledge graphs into '{unified_name}'")
        
        # If only one graph, return it as-is
        if len(knowledge_graphs) == 1:
            return knowledge_graphs[0]
        
        # Use the first graph as base and merge others into it
        unified_kg = knowledge_graphs[0]
        for kg in knowledge_graphs[1:]:
            unified_kg = self._merge_two_graphs(unified_kg, kg)
        
        # Update final statistics
        self.unification_stats["unified_nodes"] = len(unified_kg.nodes)
        self.unification_stats["unified_relationships"] = len(unified_kg.relationships)
        
        self._log_unification_stats()
        
        return unified_kg
    
    def unify_from_json_files(self, 
                            json_files: List[str],
                            unified_name: str = "unified") -> KnowledgeGraph:
        """
        Load and unify knowledge graphs from JSON files.
        
        Args:
            json_files: List of paths to JSON files containing knowledge graphs
            unified_name: Name for the unified graph
            
        Returns:
            Unified KnowledgeGraph
        """
        knowledge_graphs = []
        
        for file_path in json_files:
            try:
                kg = self._load_kg_from_json(file_path)
                if kg:
                    knowledge_graphs.append(kg)
                    logger.info(f"Loaded KG from {file_path}: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph from {file_path}: {e}")
        
        if not knowledge_graphs:
            logger.error("No valid knowledge graphs could be loaded from the provided files")
            return KnowledgeGraph(nodes=[], relationships=[])
        
        return self.unify_knowledge_graphs(knowledge_graphs, unified_name)
    
    def _merge_two_graphs(self, kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> KnowledgeGraph:
        """
        Merge two knowledge graphs using intelligent deduplication.
        
        Args:
            kg1: First knowledge graph
            kg2: Second knowledge graph to merge into the first
            
        Returns:
            Merged knowledge graph
        """
        # Create node map for efficient lookup
        node_map: Dict[str, Node] = {node.id: node for node in kg1.nodes}
        
        # Merge nodes from kg2
        for node in kg2.nodes:
            if node.id in node_map:
                # Merge duplicate node
                existing_node = node_map[node.id]
                merged_node = self._merge_nodes(existing_node, node)
                node_map[node.id] = merged_node
                self.unification_stats["duplicate_nodes_merged"] += 1
            else:
                # Add new node
                node_map[node.id] = node
        
        # Merge relationships
        relationship_set: Set[Tuple[str, str, str]] = set()
        merged_relationships: List[Relationship] = []
        
        # Add relationships from kg1
        for rel in kg1.relationships:
            rel_key = (rel.source, rel.target, rel.label)
            relationship_set.add(rel_key)
            merged_relationships.append(rel)
        
        # Add unique relationships from kg2
        for rel in kg2.relationships:
            rel_key = (rel.source, rel.target, rel.label)
            if rel_key not in relationship_set:
                merged_relationships.append(rel)
                relationship_set.add(rel_key)
            else:
                self.unification_stats["duplicate_relationships_removed"] += 1
        
        return KnowledgeGraph(
            nodes=list(node_map.values()),
            relationships=merged_relationships
        )
    
    def _merge_nodes(self, node1: Node, node2: Node) -> Node:
        """
        Merge two nodes with the same ID.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Merged node with combined properties and metadata
        """
        # Merge properties based on conflict resolution strategy
        merged_properties = self._merge_properties(
            node1.properties, 
            node2.properties
        )
        
        # Merge metadata lists, avoiding duplicates
        merged_metadata = self._merge_metadata(node1.metadata, node2.metadata)
        
        return Node(
            id=node1.id,
            type=node1.type,  # Type should be the same for nodes with same ID
            properties=merged_properties,
            metadata=merged_metadata
        )
    
    def _merge_properties(self, 
                         props1: Dict[str, Any], 
                         props2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two property dictionaries based on conflict resolution strategy.
        
        Args:
            props1: First property dictionary
            props2: Second property dictionary
            
        Returns:
            Merged property dictionary
        """
        if self.conflict_resolution == "first_wins":
            # Start with props2 and update with props1 (props1 wins)
            merged = props2.copy()
            merged.update(props1)
            return merged
        
        elif self.conflict_resolution == "last_wins":
            # Start with props1 and update with props2 (props2 wins)
            merged = props1.copy()
            for key, value in props2.items():
                if key in merged and merged[key] != value:
                    self.unification_stats["property_conflicts_resolved"] += 1
                merged[key] = value
            return merged
        
        elif self.conflict_resolution == "merge_all":
            # Keep all values as lists
            merged = {}
            all_keys = set(props1.keys()) | set(props2.keys())
            
            for key in all_keys:
                values = []
                
                if key in props1:
                    if isinstance(props1[key], list):
                        values.extend(props1[key])
                    else:
                        values.append(props1[key])
                
                if key in props2:
                    if isinstance(props2[key], list):
                        values.extend(props2[key])
                    else:
                        values.append(props2[key])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_values = []
                for v in values:
                    v_str = json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v)
                    if v_str not in seen:
                        seen.add(v_str)
                        unique_values.append(v)
                
                if len(unique_values) == 1:
                    merged[key] = unique_values[0]
                else:
                    merged[key] = unique_values
                    if key in props1 and key in props2:
                        self.unification_stats["property_conflicts_resolved"] += 1
            
            return merged
        
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution}")
    
    def _merge_metadata(self, 
                       metadata1: List[SourceMetadata], 
                       metadata2: List[SourceMetadata]) -> List[SourceMetadata]:
        """
        Merge metadata lists, avoiding duplicates.
        
        Args:
            metadata1: First metadata list
            metadata2: Second metadata list
            
        Returns:
            Merged metadata list without duplicates
        """
        # Use set of (filename, chunk_id) tuples to track unique metadata
        seen_metadata = set()
        merged_metadata = []
        
        for meta_list in [metadata1, metadata2]:
            for meta in meta_list:
                meta_key = (meta.filename, meta.chunk_id)
                if meta_key not in seen_metadata:
                    seen_metadata.add(meta_key)
                    merged_metadata.append(meta)
        
        return merged_metadata
    
    def _load_kg_from_json(self, file_path: str) -> Optional[KnowledgeGraph]:
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            KnowledgeGraph object or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create nodes
            nodes = []
            for node_data in data.get('nodes', []):
                # Convert metadata if present
                metadata = []
                if 'metadata' in node_data:
                    for meta in node_data['metadata']:
                        metadata.append(SourceMetadata(
                            filename=meta['filename'],
                            chunk_id=meta['chunk_id']
                        ))
                
                nodes.append(Node(
                    id=node_data['id'],
                    type=node_data['type'],
                    properties=node_data.get('properties', {}),
                    metadata=metadata
                ))
            
            # Create relationships
            relationships = []
            for rel_data in data.get('relationships', []):
                # Convert metadata
                meta = rel_data.get('metadata', {})
                metadata = SourceMetadata(
                    filename=meta.get('filename', 'unknown'),
                    chunk_id=meta.get('chunk_id', 'unknown')
                )
                
                relationships.append(Relationship(
                    source=rel_data['source'],
                    target=rel_data['target'],
                    label=rel_data['label'],
                    properties=rel_data.get('properties', {}),
                    metadata=metadata
                ))
            
            return KnowledgeGraph(nodes=nodes, relationships=relationships)
            
        except Exception as e:
            logger.error(f"Error loading KG from {file_path}: {e}")
            return None
    
    def _reset_stats(self):
        """Reset unification statistics."""
        for key in self.unification_stats:
            self.unification_stats[key] = 0
    
    def _log_unification_stats(self):
        """Log unification statistics."""
        stats = self.unification_stats
        logger.info(f"Unification complete:")
        logger.info(f"  - Input graphs: {stats['total_input_graphs']}")
        logger.info(f"  - Input nodes: {stats['total_input_nodes']} → Output nodes: {stats['unified_nodes']}")
        logger.info(f"  - Input relationships: {stats['total_input_relationships']} → Output relationships: {stats['unified_relationships']}")
        logger.info(f"  - Duplicate nodes merged: {stats['duplicate_nodes_merged']}")
        logger.info(f"  - Duplicate relationships removed: {stats['duplicate_relationships_removed']}")
        logger.info(f"  - Property conflicts resolved: {stats['property_conflicts_resolved']}")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get the current unification statistics.
        
        Returns:
            Dictionary of statistics from the last unification operation
        """
        return self.unification_stats.copy()