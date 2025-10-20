# Knowledge Graph Validation Module

## Overview

The Knowledge Graph Validation module provides comprehensive validation and enhancement capabilities for extracted knowledge graphs. It performs a second pass over the source text to ensure completeness and accuracy of the extracted information, identifying missing nodes, relationships, and properties while standardizing existing data according to defined schemas.

## Key Features

- **Completeness Validation**: Compares extracted knowledge graphs against source text to identify missing information
- **Property Standardization**: Automatically corrects and standardizes property names across all nodes
- **TQDCS Validation**: Ensures TQDCS (Technology, Quality, Delivery, Cost, Sustainability) categories are properly formatted
- **LLM-based Enhancement**: Uses language models to intelligently identify missing entities and relationships
- **Metadata Preservation**: Maintains source metadata for traceability
- **Batch Processing**: Efficiently validates large knowledge graphs

## Property Standardization Rules

The validator automatically standardizes property names to maintain consistency across the knowledge graph:

### Common Property Mappings

- `TQDCS` → `tqdcs_categories`
- `tqdcs` → `tqdcs_categories`
- `TQDCS_categories` → `tqdcs_categories`
- `TQDCS_CATEGORIES` → `tqdcs_categories`

### Standardization Process

1. **Property Name Normalization**: Non-standard property names are automatically converted to their canonical forms
2. **Duplicate Handling**: When multiple versions of the same property exist, values are merged (for lists) or the latest value is kept
3. **Case Sensitivity**: Property names are case-sensitive, with lowercase being the standard

## TQDCS Validation

The validator ensures TQDCS categories follow strict formatting rules:

### TQDCS Format Rules

1. **Single Letter Format**: Only single uppercase letters are allowed: T, Q, D, C, S
2. **Full Name Conversion**: Full category names are automatically converted:
   - `technology` → `T`
   - `quality` → `Q`
   - `delivery` → `D`
   - `cost` → `C`
   - `sustainability` → `S`

3. **Node Type Restrictions**: Certain node types should not have TQDCS categories:
   - `ORGANIZATION`
   - `LOCATION`
   - `DATE`
   - `SUPPLIER`

4. **Generic Information Nodes**: TQDCS categories are removed from nodes that appear to be organizational or contact information

### TQDCS Content Validation

The validator performs the following checks:
- Ensures `tqdcs_categories` is always a list
- Removes duplicate categories
- Sorts categories alphabetically
- Validates that only valid TQDCS letters are present

## LLM-based Enhancement

The validation process uses language models to identify missing information:

### Enhancement Process

1. **Knowledge Graph Summary**: The existing KG is summarized by node and relationship types
2. **Text Comparison**: The LLM compares the original text against the summarized KG
3. **Missing Entity Detection**: Identifies entities mentioned in text but not captured as nodes
4. **Missing Relationship Detection**: Finds connections described in text but not represented as relationships
5. **Property Enhancement**: Discovers additional properties for existing nodes

### LLM Configuration

- **Default Model**: gpt-4.1
- **Temperature**: 0.0 (for consistent, deterministic results)
- **Prompt Templates**: Customizable system and human prompts for validation

## Usage Examples

### Basic Validation

```python
from resources.kg_creation.validation.kg_validator import KGValidator
from resources.kg_creation.models.kg_schema import KnowledgeGraph, SourceMetadata

# Initialize validator
validator = KGValidator(llm_model="gpt-4.1", temperature=0.0)

# Create source metadata
metadata = SourceMetadata(
    filename="supplier_quote.pdf",
    chunk_id="page_3"
)

# Validate knowledge graph
result = validator.validate_knowledge_graph(
    kg=existing_kg,
    original_text=source_text,
    metadata=metadata
)

# Access validation results
print(f"Nodes added: {result.nodes_added}")
print(f"Relationships added: {result.relationships_added}")
print(f"Properties enhanced: {result.properties_enhanced}")
print(f"TQDCS fields corrected: {result.tqdcs_fields_corrected}")
print(f"Property names standardized: {result.property_names_standardized}")
```

### Custom Validation with Different Model

```python
# Use a different LLM model
validator = KGValidator(llm_model="gpt-4", temperature=0.1)

# Perform validation
result = validator.validate_knowledge_graph(
    kg=my_kg,
    original_text=document_text,
    metadata=source_metadata
)

# Check if validation was successful
if result.validation_successful:
    enhanced_kg = result.validated_kg
    print(f"Validation notes: {result.notes}")
else:
    print(f"Validation failed: {result.notes}")
```

## Validation Results Interpretation

The `ValidationResult` object contains comprehensive information about the validation process:

### Result Fields

- **`validated_kg`**: The enhanced knowledge graph with all corrections and additions
- **`nodes_added`**: Number of new nodes identified and added
- **`relationships_added`**: Number of new relationships discovered
- **`properties_enhanced`**: Number of existing nodes that received additional properties
- **`validation_successful`**: Boolean indicating if validation completed without errors
- **`notes`**: Textual description of validation findings and any issues
- **`tqdcs_fields_corrected`**: Number of TQDCS field name corrections made
- **`property_names_standardized`**: Total number of property names standardized

### Interpreting Results

1. **High Node/Relationship Additions**: May indicate the initial extraction missed significant information
2. **Many Property Enhancements**: Suggests the initial extraction captured entities but missed some attributes
3. **TQDCS Corrections**: Shows how many nodes had improperly formatted TQDCS categories
4. **Property Standardizations**: Indicates consistency issues in the original extraction

### Error Handling

If validation fails, the original knowledge graph is returned unchanged:
- Check `validation_successful` flag
- Review `notes` field for error details
- Common failures include LLM API errors or malformed input data

## Best Practices

1. **Always Validate**: Run validation on all extracted knowledge graphs to ensure completeness
2. **Review Additions**: Manually review significant additions to ensure accuracy
3. **Monitor Standardizations**: High numbers of standardizations may indicate issues with the extraction process
4. **Use Appropriate Models**: Choose LLM models based on complexity and accuracy requirements
5. **Preserve Metadata**: Always provide source metadata for new nodes and relationships
6. **Batch Processing**: For multiple documents, process validations in sequence to avoid rate limits

## Integration with KG Pipeline

The validator is designed to be used as the final step in the knowledge graph creation pipeline:

1. **Text Extraction**: Extract text from source documents
2. **Initial KG Creation**: Create knowledge graph from extracted text
3. **Validation**: Use KGValidator to enhance and standardize the graph
4. **Post-processing**: Apply any domain-specific rules or transformations
5. **Storage**: Save the validated knowledge graph

## Troubleshooting

### Common Issues

1. **Missing Prompts**: If prompt files are not found, default prompts are used automatically
2. **LLM Timeouts**: For large texts, consider chunking the validation process
3. **Memory Issues**: For very large KGs, process in batches
4. **Inconsistent Results**: Ensure temperature is set to 0.0 for deterministic outputs

### Debug Logging

Enable debug logging to see detailed validation steps:

```python
import logging
logging.getLogger('resources.kg_creation.validation.kg_validator').setLevel(logging.DEBUG)
```

This will show:
- Property standardization details
- TQDCS corrections
- Node and relationship processing
- LLM prompt construction