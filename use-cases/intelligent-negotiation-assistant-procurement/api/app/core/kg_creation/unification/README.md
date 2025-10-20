# Knowledge Graph Unification Module

## Overview

The Knowledge Graph Unification module provides a sophisticated framework for merging multiple knowledge graphs into a single, consolidated graph. It intelligently handles duplicate entities, resolves property conflicts, and preserves all source metadata to maintain traceability.

The unification process is essential when:
- Processing documents from multiple suppliers or sources
- Combining knowledge graphs generated from different document chunks
- Integrating information from various file formats (Excel, PDF, images)
- Building a comprehensive view from distributed data sources

## Key Features

- **Intelligent Node Deduplication**: Automatically merges nodes with identical IDs while preserving all metadata
- **Relationship Deduplication**: Removes duplicate relationships based on source, target, and label
- **Configurable Conflict Resolution**: Multiple strategies for handling property conflicts
- **Source Metadata Preservation**: Maintains complete traceability of information origins
- **Comprehensive Statistics**: Detailed reporting on the unification process
- **JSON File Support**: Direct loading and unification from JSON files

## Architecture

### Core Classes

#### KGUnifier
The main class that orchestrates the unification process.

```python
from resources.kg_creation.unification import KGUnifier

# Initialize with conflict resolution strategy
unifier = KGUnifier(conflict_resolution="last_wins")
```

### Data Models

The unifier works with these core data structures from the `kg_schema` module:

- **Node**: Represents entities (Part, Cost, Supplier, etc.)
- **Relationship**: Directed connections between nodes
- **SourceMetadata**: Tracks origin of information (filename, chunk_id)
- **KnowledgeGraph**: Container for nodes and relationships

## Conflict Resolution Strategies

When merging nodes with the same ID but different properties, the unifier offers three strategies:

### 1. Last Wins (Default)
```python
unifier = KGUnifier(conflict_resolution="last_wins")
```
- Later values override earlier ones
- Useful when newer data is more accurate
- Property conflicts are counted in statistics

### 2. First Wins
```python
unifier = KGUnifier(conflict_resolution="first_wins")
```
- Earlier values are preserved
- Useful when the first source is most authoritative
- New properties are added, but existing ones are not modified

### 3. Merge All
```python
unifier = KGUnifier(conflict_resolution="merge_all")
```
- Keeps all values as lists
- No information is lost
- Single values remain as scalars
- Multiple different values become lists
- Automatically deduplicates identical values

## Deduplication Algorithms

### Node Deduplication
Nodes are considered duplicates if they have the same ID. When duplicates are found:
1. Properties are merged according to the conflict resolution strategy
2. Metadata lists are combined, removing duplicate (filename, chunk_id) pairs
3. The node type is preserved (should be consistent for same ID)

### Relationship Deduplication
Relationships are considered duplicates if they have the same:
- Source node ID
- Target node ID  
- Relationship label

Duplicate relationships are removed, keeping only the first occurrence.

## Usage Examples

### Example 1: Unifying Knowledge Graphs from Memory

```python
from resources.kg_creation.unification import KGUnifier
from resources.kg_creation.models.kg_schema import KnowledgeGraph, Node, Relationship, SourceMetadata

# Create sample knowledge graphs
kg1 = KnowledgeGraph(
    nodes=[
        Node(
            id="Part:ENGINE-01",
            type="Part",
            properties={"name": "V8 Engine", "weight": 250},
            metadata=[SourceMetadata(filename="supplier_a.xlsx", chunk_id="sheet_1")]
        )
    ],
    relationships=[
        Relationship(
            source="Part:ENGINE-01",
            target="Cost:ENGINE-COST-01",
            label="HAS_COST",
            properties={"valid_from": "2024-01-01"},
            metadata=SourceMetadata(filename="supplier_a.xlsx", chunk_id="sheet_1")
        )
    ]
)

kg2 = KnowledgeGraph(
    nodes=[
        Node(
            id="Part:ENGINE-01",  # Same part, different properties
            type="Part",
            properties={"name": "V8 Engine", "weight": 255, "manufacturer": "ACME"},
            metadata=[SourceMetadata(filename="supplier_b.pdf", chunk_id="page_5")]
        )
    ],
    relationships=[
        Relationship(
            source="Part:ENGINE-01",
            target="Supplier:ACME",
            label="CAN_SUPPLY",
            properties={},
            metadata=SourceMetadata(filename="supplier_b.pdf", chunk_id="page_5")
        )
    ]
)

# Unify with different strategies
unifier = KGUnifier(conflict_resolution="merge_all")
unified_kg = unifier.unify_knowledge_graphs([kg1, kg2], unified_name="combined_suppliers")

# Access statistics
stats = unifier.get_statistics()
print(f"Merged {stats['duplicate_nodes_merged']} duplicate nodes")
print(f"Removed {stats['duplicate_relationships_removed']} duplicate relationships")
```

### Example 2: Unifying from JSON Files

```python
from resources.kg_creation.unification import KGUnifier

# Initialize unifier
unifier = KGUnifier(conflict_resolution="last_wins")

# List of JSON files to unify
json_files = [
    "/path/to/supplier_a_kg.json",
    "/path/to/supplier_b_kg.json",
    "/path/to/technical_specs_kg.json"
]

# Unify all graphs
unified_kg = unifier.unify_from_json_files(
    json_files=json_files,
    unified_name="rfq_analysis"
)

# Save unified graph
import json
with open("unified_knowledge_graph.json", "w") as f:
    json.dump(unified_kg.to_dict(), f, indent=2)
```

### Example 3: Processing Multiple Document Chunks

```python
from resources.kg_creation.unification import KGUnifier
import os
import glob

# Find all chunk KGs
chunk_files = glob.glob("output/chunks/*/knowledge_graph.json")

# Unify all chunks with merge_all to preserve all information
unifier = KGUnifier(conflict_resolution="merge_all")
unified_kg = unifier.unify_from_json_files(
    json_files=chunk_files,
    unified_name="all_chunks"
)

# Print detailed statistics
stats = unifier.get_statistics()
print(f"Unification Statistics:")
print(f"  Input graphs: {stats['total_input_graphs']}")
print(f"  Input nodes: {stats['total_input_nodes']}")
print(f"  Output nodes: {stats['unified_nodes']}")
print(f"  Nodes merged: {stats['duplicate_nodes_merged']}")
print(f"  Property conflicts: {stats['property_conflicts_resolved']}")
```

## Statistics and Reporting

The unifier tracks detailed statistics during the unification process:

```python
stats = unifier.get_statistics()
```

Available statistics:
- `total_input_graphs`: Number of input knowledge graphs
- `total_input_nodes`: Total nodes across all input graphs
- `total_input_relationships`: Total relationships across all inputs
- `unified_nodes`: Number of nodes in the unified graph
- `unified_relationships`: Number of relationships in the unified graph
- `duplicate_nodes_merged`: Number of duplicate nodes that were merged
- `duplicate_relationships_removed`: Number of duplicate relationships removed
- `property_conflicts_resolved`: Number of property conflicts encountered

## Best Practices for Merging Graphs

### 1. Choose the Right Conflict Resolution Strategy

- **Use `last_wins`** when processing updates or corrections to existing data
- **Use `first_wins`** when the initial source is most authoritative
- **Use `merge_all`** when you need to preserve all variations for analysis

### 2. Maintain Consistent Node IDs

Ensure node IDs follow the format `Type:Identifier` across all sources:
```python
# Good
"Part:ENGINE-01"
"Cost:Annual-License-2024"
"Supplier:ACME-Corp"

# Bad
"engine01"  # Missing type prefix
"Part-ENGINE-01"  # Wrong separator
```

### 3. Process in Logical Order

When order matters (e.g., with `last_wins` or `first_wins`), process graphs in the appropriate sequence:
```python
# Process base data first, then updates
graphs = [base_kg, update1_kg, update2_kg]
unified = unifier.unify_knowledge_graphs(graphs)
```

### 4. Handle Large-Scale Unification

For large numbers of graphs, consider batching:
```python
def unify_in_batches(all_files, batch_size=10):
    unifier = KGUnifier(conflict_resolution="merge_all")
    
    # Process in batches
    batches = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]
    batch_graphs = []
    
    for batch in batches:
        batch_kg = unifier.unify_from_json_files(batch)
        batch_graphs.append(batch_kg)
    
    # Final unification
    return unifier.unify_knowledge_graphs(batch_graphs)
```

### 5. Validate Unified Graphs

After unification, validate the results:
```python
def validate_unified_graph(kg):
    # Check for orphaned relationships
    node_ids = {node.id for node in kg.nodes}
    for rel in kg.relationships:
        if rel.source not in node_ids or rel.target not in node_ids:
            print(f"Warning: Orphaned relationship {rel.source} -> {rel.target}")
    
    # Check for expected merges
    for node in kg.nodes:
        if len(node.metadata) > 1:
            print(f"Node {node.id} merged from {len(node.metadata)} sources")
```

### 6. Preserve Metadata

Always ensure source metadata is preserved for traceability:
```python
# Check which sources contributed to a node
for node in unified_kg.nodes:
    sources = [m.filename for m in node.metadata]
    print(f"Node {node.id} came from: {sources}")
```

## Error Handling

The unifier includes robust error handling:

- Invalid JSON files are logged and skipped
- Missing files generate warnings but don't stop the process
- Malformed nodes or relationships are reported in logs
- Empty input lists return empty knowledge graphs

## Performance Considerations

- **Memory Usage**: The unifier loads all graphs into memory. For very large graphs, consider processing in batches
- **Deduplication Complexity**: O(n) for nodes, O(n²) worst case for relationships
- **JSON Loading**: Large JSON files may take time to parse

## Integration with Pipeline

The unifier integrates seamlessly with other pipeline components:

```python
from resources.kg_creation.unification import KGUnifier
from resources.kg_creation.pipeline import KGCreationPipeline

# Generate KGs from multiple sources
pipeline = KGCreationPipeline()
kg1 = pipeline.process_document("supplier_a.xlsx")
kg2 = pipeline.process_document("supplier_b.pdf")

# Unify the results
unifier = KGUnifier(conflict_resolution="merge_all")
unified_kg = unifier.unify_knowledge_graphs([kg1, kg2])
```

## Troubleshooting

### Common Issues

1. **"Node ID must be in format 'Type:Identifier'"**
   - Ensure all node IDs include the type prefix
   - Check that ':' is used as separator, not '-' or '_'

2. **High number of property conflicts**
   - Review your conflict resolution strategy
   - Consider using `merge_all` to preserve all values
   - Check for inconsistent property names across sources

3. **Missing relationships after unification**
   - Verify that source and target node IDs exist
   - Check for typos in node IDs
   - Ensure relationship deduplication isn't too aggressive

### Debug Logging

Enable detailed logging to troubleshoot issues:
```python
import logging

# Set logging level
logging.getLogger('resources.kg_creation.unification').setLevel(logging.DEBUG)

# Unification will now show detailed progress
unified_kg = unifier.unify_from_json_files(files)
```