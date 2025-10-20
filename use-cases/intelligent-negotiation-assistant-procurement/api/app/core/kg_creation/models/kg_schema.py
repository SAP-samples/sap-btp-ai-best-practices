"""
Core Pydantic models for the Knowledge Graph Creation Pipeline.

This module defines the strict schema used throughout the pipeline as specified
in the Engineering Specification document.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class NodeType(str, Enum):
    """Canonical entity types for nodes in the knowledge graph (V2 spec)."""
    PART = "Part"
    COST = "Cost"
    SUPPLIER = "Supplier"
    DATE = "Date"
    LOCATION = "Location"
    ORGANIZATION = "Organization"
    DRAWING = "Drawing"
    CERTIFICATION = "Certification"
    GENERIC_INFORMATION = "GenericInformation"




class SourceMetadata(BaseModel):
    """Contains metadata about the origin of a piece of information."""
    filename: str = Field(
        description="The name of the source file (e.g., 'Supplier_A_Offer.xlsx')."
    )
    chunk_id: str = Field(
        description="The specific chunk the data was extracted from (e.g., 'page_5', 'sheet_Technical_Specs')."
    )
    
    class Config:
        frozen = True  # Make immutable for use as dict keys if needed


class StructuredValue(BaseModel):
    """A model for numerical values that have an associated unit, to avoid parsing ambiguity."""
    value: float = Field(
        description="The numerical value."
    )
    unit: str = Field(
        description="The unit of measurement (e.g., 'EUR', 'USD', 'kg', 'days')."
    )
    
    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Ensure unit is not empty and normalize common units."""
        if not v or not v.strip():
            raise ValueError("Unit cannot be empty")
        
        # Normalize common units
        unit_map = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            'kilogram': 'kg',
            'kilograms': 'kg',
            'gram': 'g',
            'grams': 'g',
            'day': 'days',
            'week': 'weeks',
            'month': 'months',
            'year': 'years'
        }
        
        normalized = v.strip().lower()
        return unit_map.get(normalized, v.strip())


class Node(BaseModel):
    """Represents a single canonical entity in the knowledge graph."""
    id: str = Field(
        description="A unique, normalized identifier for the node (e.g., 'Requirement:ISO-27001', 'Cost:Annual-License')."
    )
    type: NodeType = Field(
        description="The canonical type of the entity."
    )
    properties: Dict[str, Any] = Field(
        description=(
            "A dictionary of all attributes associated with the node. "
            "For numerical properties, this should use the 'StructuredValue' model."
        ),
        default_factory=dict
    )
    metadata: List[SourceMetadata] = Field(
        description="A list of all source chunks where this node was mentioned or inferred.",
        default_factory=list
    )
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows the expected format."""
        if ':' not in v:
            raise ValueError("Node ID must be in format 'Type:Identifier'")
        return v
    


class RelationshipLabel(str, Enum):
    """Standard relationship types in the knowledge graph (V2 spec)."""
    HAS_COST = "HAS_COST"
    HAS_PART = "HAS_PART"
    HAS_RATING = "HAS_RATING"
    HAS_LOCATION = "HAS_LOCATION"
    HAS_CERTIFICATION = "HAS_CERTIFICATION"
    DELIVERS_ON = "DELIVERS_ON"
    CAN_SUPPLY = "CAN_SUPPLY"
    HAS_CAPACITY = "HAS_CAPACITY"
    HAS_DRAWING = "HAS_DRAWING"


class Relationship(BaseModel):
    """Represents a directed connection between two nodes."""
    source: str = Field(
        description="The unique ID of the source node."
    )
    target: str = Field(
        description="The unique ID of the target node."
    )
    label: str = Field(
        description="The type of relationship, expressed as a verb phrase (e.g., 'HAS_COST', 'REQUIRES_CERTIFICATION', 'DELIVERS_ON')."
    )
    properties: Dict[str, Any] = Field(
        description=(
            "A dictionary of attributes for the relationship. "
            "For numerical properties, this should use the 'StructuredValue' model."
        ),
        default_factory=dict
    )
    metadata: SourceMetadata = Field(
        description="Metadata about the specific source of this relationship."
    )
    
    @field_validator('source', 'target')
    @classmethod
    def validate_node_id_format(cls, v: str) -> str:
        """Ensure node IDs follow the expected format."""
        if ':' not in v:
            raise ValueError("Node ID must be in format 'Type:Identifier'")
        return v


class KnowledgeGraph(BaseModel):
    """The top-level container for the graph extracted from one or more chunks."""
    nodes: List[Node] = Field(
        description="A list of all entities identified in the text.",
        default_factory=list
    )
    relationships: List[Relationship] = Field(
        description="A list of all relationships between entities.",
        default_factory=list
    )
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Find a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.type == node_type]
    
    def get_relationships_by_label(self, label: str) -> List[Relationship]:
        """Get all relationships with a specific label."""
        return [rel for rel in self.relationships if rel.label == label]
    
    def merge_with(self, other: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """
        Merge another KnowledgeGraph into this one, handling duplicates.
        
        Returns a new KnowledgeGraph with merged content.
        """
        # Create a mapping of node IDs to nodes for efficient lookup
        node_map = {node.id: node for node in self.nodes}
        
        # Merge nodes
        for other_node in other.nodes:
            if other_node.id in node_map:
                # Merge properties and metadata
                existing_node = node_map[other_node.id]
                
                # Merge properties (new properties are added, conflicts use last-write-wins)
                merged_properties = existing_node.properties.copy()
                merged_properties.update(other_node.properties)
                
                # Merge metadata lists
                merged_metadata = existing_node.metadata.copy()
                # Add unique metadata entries
                for meta in other_node.metadata:
                    if not any(m.filename == meta.filename and m.chunk_id == meta.chunk_id 
                              for m in merged_metadata):
                        merged_metadata.append(meta)
                
                # Create updated node
                node_map[other_node.id] = Node(
                    id=existing_node.id,
                    type=existing_node.type,
                    properties=merged_properties,
                    metadata=merged_metadata
                )
            else:
                # Add new node
                node_map[other_node.id] = other_node
        
        # Merge relationships (allow duplicates as this is a multi-graph)
        merged_relationships = self.relationships.copy()
        merged_relationships.extend(other.relationships)
        
        return KnowledgeGraph(
            nodes=list(node_map.values()),
            relationships=merged_relationships
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "nodes": [node.model_dump() for node in self.nodes],
            "relationships": [rel.model_dump() for rel in self.relationships],
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_relationships": len(self.relationships),
                "node_types": list(set(node.type.value for node in self.nodes)),
                "relationship_types": list(set(rel.label for rel in self.relationships))
            }
        }


# Additional models for pipeline processing

class Chunk(BaseModel):
    """Represents a chunk of text from a document."""
    chunk_id: str = Field(
        description="Unique identifier for the chunk (e.g., 'document.pdf:page_5')."
    )
    chunk_text: str = Field(
        description="The text content of the chunk."
    )
    metadata: SourceMetadata = Field(
        description="Source metadata for the chunk."
    )


class RawAssertion(BaseModel):
    """Represents a raw factual assertion extracted from text."""
    subject: str = Field(description="The subject of the assertion.")
    predicate: str = Field(description="The relationship or action.")
    object: str = Field(description="The object of the assertion.")