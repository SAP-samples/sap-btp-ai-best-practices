# Knowledge Graph Models

This directory contains the core Pydantic models that define the strict schema for the Knowledge Graph Creation Pipeline. These models ensure data consistency and type safety throughout the pipeline.

## Overview

The models in this directory implement the V2 specification for knowledge graph structures, providing:
- Strong type validation through Pydantic
- Immutable metadata for data provenance tracking
- Support for structured numerical values with units
- Built-in merge capabilities for graph combination
- JSON serialization support

## Core Models

### Node

Represents a single canonical entity in the knowledge graph.

**Attributes:**
- `id` (str): Unique identifier in format "Type:Identifier" (e.g., "Part:XYZ-123", "Supplier:ACME")
- `type` (NodeType): The canonical type of the entity
- `properties` (Dict[str, Any]): Attributes associated with the node
- `metadata` (List[SourceMetadata]): Source tracking for data provenance

**Example:**
```python
from kg_schema import Node, NodeType, SourceMetadata

node = Node(
    id="Part:BOLT-A325",
    type=NodeType.PART,
    properties={
        "name": "A325 Structural Bolt",
        "material": "High-strength steel",
        "diameter": {"value": 20, "unit": "mm"}
    },
    metadata=[
        SourceMetadata(
            filename="supplier_catalog.xlsx",
            chunk_id="sheet_Parts"
        )
    ]
)
```

### Relationship

Represents a directed connection between two nodes.

**Attributes:**
- `source` (str): ID of the source node
- `target` (str): ID of the target node
- `label` (str): Type of relationship (verb phrase)
- `properties` (Dict[str, Any]): Attributes for the relationship
- `metadata` (SourceMetadata): Source information for this relationship

**Example:**
```python
from kg_schema import Relationship, SourceMetadata

relationship = Relationship(
    source="Supplier:ACME",
    target="Part:BOLT-A325",
    label="CAN_SUPPLY",
    properties={
        "lead_time": {"value": 14, "unit": "days"},
        "minimum_order_quantity": {"value": 1000, "unit": "pieces"}
    },
    metadata=SourceMetadata(
        filename="supplier_catalog.xlsx",
        chunk_id="sheet_Availability"
    )
)
```

### KnowledgeGraph

Top-level container for nodes and relationships.

**Attributes:**
- `nodes` (List[Node]): All entities in the graph
- `relationships` (List[Relationship]): All connections between entities

**Methods:**
- `get_node_by_id(node_id)`: Find a node by its ID
- `get_nodes_by_type(node_type)`: Get all nodes of a specific type
- `get_relationships_by_label(label)`: Get all relationships with a specific label
- `merge_with(other)`: Merge another KnowledgeGraph into this one
- `to_dict()`: Convert to dictionary for JSON serialization

**Example:**
```python
from kg_schema import KnowledgeGraph, Node, Relationship

# Create a knowledge graph
kg = KnowledgeGraph()

# Add nodes
kg.nodes.append(supplier_node)
kg.nodes.append(part_node)

# Add relationships
kg.relationships.append(supply_relationship)

# Query the graph
parts = kg.get_nodes_by_type(NodeType.PART)
supply_rels = kg.get_relationships_by_label("CAN_SUPPLY")

# Merge with another graph
merged_kg = kg.merge_with(other_kg)
```

## Enumerations

### NodeType

Canonical entity types for nodes in the knowledge graph (V2 spec):
- `PART`: Component or material
- `COST`: Cost information
- `SUPPLIER`: Supplier organization
- `DATE`: Date reference
- `LOCATION`: Geographic location
- `ORGANIZATION`: Any organization entity
- `DRAWING`: Technical drawing reference
- `CERTIFICATION`: Certification or compliance standard
- `GENERIC_INFORMATION`: General information not fitting other categories

### RelationshipLabel

Standard relationship types in the knowledge graph:
- `HAS_COST`: Entity has associated cost
- `HAS_PART`: Entity contains or requires a part
- `HAS_RATING`: Entity has a rating or score
- `HAS_LOCATION`: Entity is located at a place
- `HAS_CERTIFICATION`: Entity has certification
- `DELIVERS_ON`: Supplier delivers on a date
- `CAN_SUPPLY`: Supplier can provide a part
- `HAS_CAPACITY`: Entity has capacity information
- `HAS_DRAWING`: Entity has associated technical drawing

## Special Models

### StructuredValue

Model for numerical values with units to avoid parsing ambiguity.

**Attributes:**
- `value` (float): The numerical value
- `unit` (str): The unit of measurement

**Features:**
- Automatic unit normalization (e.g., "$" → "USD", "kilogram" → "kg")
- Validation to ensure units are not empty

**Example:**
```python
from kg_schema import StructuredValue

price = StructuredValue(value=150.50, unit="EUR")
weight = StructuredValue(value=2.5, unit="kg")
```

### SourceMetadata

Contains metadata about the origin of information.

**Attributes:**
- `filename` (str): Name of the source file
- `chunk_id` (str): Specific chunk/section identifier

**Features:**
- Immutable (frozen) for use as dictionary keys
- Used for data provenance tracking

**Example:**
```python
from kg_schema import SourceMetadata

metadata = SourceMetadata(
    filename="technical_specs.pdf",
    chunk_id="page_15"
)
```

## Additional Models

### Chunk

Represents a chunk of text from a document.

**Attributes:**
- `chunk_id` (str): Unique identifier for the chunk
- `chunk_text` (str): The text content
- `metadata` (SourceMetadata): Source metadata

### RawAssertion

Represents a raw factual assertion extracted from text.

**Attributes:**
- `subject` (str): The subject of the assertion
- `predicate` (str): The relationship or action
- `object` (str): The object of the assertion

## Usage Examples

### Creating a Complete Knowledge Graph

```python
from kg_schema import (
    KnowledgeGraph, Node, Relationship, 
    NodeType, SourceMetadata, StructuredValue
)

# Create nodes
supplier = Node(
    id="Supplier:TechCorp",
    type=NodeType.SUPPLIER,
    properties={
        "name": "TechCorp Industries",
        "established": 1995
    },
    metadata=[SourceMetadata(filename="suppliers.csv", chunk_id="row_15")]
)

part = Node(
    id="Part:SENSOR-X1",
    type=NodeType.PART,
    properties={
        "name": "Temperature Sensor X1",
        "weight": StructuredValue(value=50, unit="g").model_dump()
    },
    metadata=[SourceMetadata(filename="parts_catalog.xlsx", chunk_id="sheet_Sensors")]
)

cost = Node(
    id="Cost:SENSOR-X1-UNIT",
    type=NodeType.COST,
    properties={
        "amount": StructuredValue(value=25.99, unit="USD").model_dump(),
        "cost_type": "unit_price"
    },
    metadata=[SourceMetadata(filename="pricing.xlsx", chunk_id="sheet_2024")]
)

# Create relationships
supply_rel = Relationship(
    source="Supplier:TechCorp",
    target="Part:SENSOR-X1",
    label="CAN_SUPPLY",
    properties={
        "availability": "in_stock",
        "lead_time": StructuredValue(value=7, unit="days").model_dump()
    },
    metadata=SourceMetadata(filename="suppliers.csv", chunk_id="row_15")
)

cost_rel = Relationship(
    source="Part:SENSOR-X1",
    target="Cost:SENSOR-X1-UNIT",
    label="HAS_COST",
    properties={
        "valid_from": "2024-01-01",
        "valid_to": "2024-12-31"
    },
    metadata=SourceMetadata(filename="pricing.xlsx", chunk_id="sheet_2024")
)

# Build the knowledge graph
kg = KnowledgeGraph(
    nodes=[supplier, part, cost],
    relationships=[supply_rel, cost_rel]
)

# Export to JSON
import json
json_data = json.dumps(kg.to_dict(), indent=2)
```

### Merging Knowledge Graphs

```python
# Create two separate graphs from different sources
kg1 = extract_from_supplier_a()  # Returns KnowledgeGraph
kg2 = extract_from_supplier_b()  # Returns KnowledgeGraph

# Merge them
merged_kg = kg1.merge_with(kg2)

# The merge operation:
# - Combines nodes with the same ID, merging their properties
# - Preserves all metadata for data provenance
# - Keeps all relationships (allows multi-graph structure)
```

## Design Patterns and Best Practices

### 1. Node ID Format

Always use the "Type:Identifier" format for node IDs:
```python
# Good
"Supplier:ACME-Corp"
"Part:BOLT-123"
"Cost:Annual-License-2024"

# Bad
"acme_supplier"
"bolt123"
"cost_annual"
```

### 2. Using StructuredValue

For all numerical properties that have units, use StructuredValue:
```python
# Good
properties = {
    "weight": StructuredValue(value=2.5, unit="kg").model_dump(),
    "price": StructuredValue(value=99.99, unit="EUR").model_dump()
}

# Bad
properties = {
    "weight": "2.5 kg",
    "price": 99.99  # Missing unit!
}
```

### 3. Metadata Tracking

Always include SourceMetadata for data provenance:
```python
# Every node and relationship should have metadata
node = Node(
    id="Part:XYZ",
    type=NodeType.PART,
    properties={},
    metadata=[
        SourceMetadata(
            filename="source_document.pdf",
            chunk_id="page_10"
        )
    ]
)
```

### 4. Property Naming Conventions

Use consistent, descriptive property names:
- Use snake_case for property keys
- Be specific (e.g., "unit_price" not just "price")
- Include temporal context when relevant (e.g., "price_2024_q1")

### 5. Relationship Labels

Use clear, action-oriented labels:
- Use UPPER_SNAKE_CASE
- Start with a verb (HAS_, CAN_, REQUIRES_, etc.)
- Be specific about the relationship type

### 6. Error Handling

The models include built-in validation:
```python
try:
    # This will raise a validation error
    node = Node(
        id="invalid-format",  # Missing colon separator
        type=NodeType.PART,
        properties={}
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### 7. Graph Querying

Use the built-in query methods for efficient access:
```python
# Instead of manual iteration
suppliers = [n for n in kg.nodes if n.type == NodeType.SUPPLIER]

# Use the provided method
suppliers = kg.get_nodes_by_type(NodeType.SUPPLIER)
```

## Integration with Pipeline

These models are designed to work seamlessly with the knowledge graph creation pipeline:

1. **Extraction Phase**: Raw text is converted to Chunk objects
2. **Entity Recognition**: Chunks are processed to create Node objects
3. **Relationship Extraction**: Connections between nodes create Relationship objects
4. **Graph Construction**: Nodes and relationships are assembled into KnowledgeGraph
5. **Merging**: Multiple KnowledgeGraph instances can be merged
6. **Export**: The graph can be serialized to JSON or other formats

## Future Considerations

The models are designed with extensibility in mind:
- New NodeType values can be added to the enum
- New RelationshipLabel values can be added
- The properties dictionaries allow flexible attribute storage
- The merge functionality supports incremental graph building