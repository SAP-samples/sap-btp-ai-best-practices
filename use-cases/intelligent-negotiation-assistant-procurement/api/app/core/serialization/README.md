# Knowledge Graph Serialization System

## Overview

The serialization system is responsible for exporting the final Knowledge Graph into various portable formats. The system provides flexibility for different use cases, from machine-readable formats for automated analysis to human-readable summaries for documentation and review.

The `GraphExporter` class handles all serialization operations, supporting multiple export formats while preserving the complete graph structure, relationships, and metadata.

## Export Formats

### 1. GraphML (Compressed)

GraphML is a comprehensive, XML-based graph format that preserves the complete graph structure and is compatible with most graph analysis tools.

**Features:**
- Standard format supported by NetworkX, Gephi, Cytoscape, and other graph analysis tools
- Compressed with gzip to reduce file size
- Preserves all node and edge attributes
- Maintains source metadata for traceability

**File Extension:** `.graphml.gz`

### 2. JSON

A structured JSON format that provides a clean, hierarchical representation of the Knowledge Graph.

**Features:**
- Human and machine-readable
- Preserves complete graph structure
- Includes export metadata (timestamp, version)
- Suitable for web applications and APIs

**File Extension:** `.json`

### 3. Summary (Text)

A comprehensive human-readable text report that provides insights into the graph structure and content.

**Features:**
- Overview statistics (node and relationship counts)
- Detailed breakdowns by entity types
- Complete node listings with properties
- Relationship examples
- Special analysis sections for requirements

**File Extension:** `.txt`

## Usage Examples

### Basic Usage

```python
from resources.kg_creation.serialization.graph_exporter import GraphExporter
from resources.kg_creation.models.kg_schema import KnowledgeGraph

# Initialize exporter
exporter = GraphExporter(output_dir="./output")

# Export to GraphML
graphml_path = exporter.export_graphml(knowledge_graph, "my_graph.graphml.gz")

# Export to JSON
json_path = exporter.export_json(knowledge_graph, "my_graph.json")

# Export summary
summary_path = exporter.export_summary(knowledge_graph, "my_graph_summary.txt")
```

### Export to Multiple Formats

```python
from resources.kg_creation.serialization.graph_exporter import export_knowledge_graph

# Export to all formats at once
output_files = export_knowledge_graph(
    knowledge_graph=kg,
    output_dir="./output",
    formats=["graphml", "json", "summary"]
)

# Access exported file paths
print(f"GraphML: {output_files['graphml']}")
print(f"JSON: {output_files['json']}")
print(f"Summary: {output_files['summary']}")
```

### Custom Output Directory

```python
# Specify custom output directory
exporter = GraphExporter(output_dir="/path/to/custom/output")

# All exports will be saved to the custom directory
exporter.export_graphml(kg)
```

## Technical Implementation Details

### GraphML Export Process

1. **NetworkX Conversion**: The KnowledgeGraph is converted to a NetworkX DiGraph
2. **Attribute Flattening**: Complex attributes are flattened for GraphML compatibility:
   - Nested dictionaries are converted to flat key-value pairs with prefixes
   - StructuredValue objects are formatted as "value unit" strings
   - Lists are converted to JSON strings
   - None values become empty strings
3. **Compression**: The GraphML is compressed with gzip to reduce file size

### JSON Export Structure

The JSON export includes:
```json
{
  "nodes": [...],
  "relationships": [...],
  "export_metadata": {
    "export_timestamp": "2024-01-15T10:30:00",
    "export_version": "1.0",
    "exporter": "KG Creation Pipeline"
  }
}
```

### Summary Report Sections

1. **Overview Statistics**: Total counts of nodes and relationships
2. **Node Type Breakdown**: Count of each entity type
3. **Relationship Type Breakdown**: Count of each relationship type
4. **Detailed Node Listing**: All nodes grouped by type with properties
5. **Relationship Details**: Examples of each relationship type
6. **Requirement Analysis**: Special section analyzing requirement strategies and categories

## Integration with Analysis Tools

### GraphML Integration

The GraphML format enables integration with various analysis tools:

```python
# Load in NetworkX
import networkx as nx
import gzip

with gzip.open('knowledge_graph.graphml.gz', 'rb') as f:
    G = nx.read_graphml(f)

# Analyze graph properties
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G)}")
```

### JSON Integration

The JSON format is ideal for programmatic access:

```python
import json

with open('knowledge_graph.json', 'r') as f:
    kg_data = json.load(f)

# Access nodes
for node in kg_data['nodes']:
    print(f"Node: {node['id']} (Type: {node['type']})")
    
# Access relationships
for rel in kg_data['relationships']:
    print(f"Relationship: {rel['source']} -> {rel['target']} ({rel['label']})")
```

## Best Practices

### 1. Choose the Right Format

- **GraphML**: Use for graph analysis tools, visualization, and when you need complete preservation of structure
- **JSON**: Use for web applications, APIs, or when you need programmatic access
- **Summary**: Use for documentation, reviews, or understanding the graph content

### 2. Handle Large Graphs

For large Knowledge Graphs:
- GraphML compression significantly reduces file size
- Consider exporting only specific subgraphs if needed
- Use the summary format to understand the graph structure before detailed analysis

### 3. Preserve Metadata

Always maintain source metadata for traceability:
```python
# Node metadata is preserved in all formats
node.metadata = [
    SourceMetadata(filename="supplier_doc.pdf", chunk_id="page_5"),
    SourceMetadata(filename="rfq_response.xlsx", chunk_id="sheet_1")
]
```

### 4. Error Handling

Always wrap export operations in try-except blocks:
```python
try:
    path = exporter.export_graphml(kg)
    print(f"Export successful: {path}")
except Exception as e:
    logger.error(f"Export failed: {e}")
```

### 5. Validate Before Export

Ensure your Knowledge Graph is valid before exporting:
```python
# Check for required attributes
assert len(kg.nodes) > 0, "Knowledge Graph has no nodes"
assert all(node.id for node in kg.nodes), "All nodes must have IDs"
```

## Output Directory Structure

The exporter creates the following directory structure:
```
output/
├── knowledge_graph.graphml.gz
├── knowledge_graph.json
└── knowledge_graph_summary.txt
```

## Logging

The serialization system uses Python's logging module:
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Export operations will log progress
exporter = GraphExporter()
exporter.export_graphml(kg)  # Logs export progress
```

## Error Handling

Common errors and solutions:

1. **Permission Denied**: Ensure write permissions for the output directory
2. **Memory Issues**: For very large graphs, consider increasing available memory
3. **Invalid Attributes**: Ensure all node/edge attributes are serializable
4. **Missing Dependencies**: Install required packages: `networkx`, `lxml`