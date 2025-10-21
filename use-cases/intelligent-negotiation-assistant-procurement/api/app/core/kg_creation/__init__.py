"""
Knowledge Graph Creation Pipeline

A unified system for extracting structured knowledge graphs from business documents.

Main Features:
- TQDCS Framework for categorization (Technology, Quality, Delivery, Cost, Sustainability)
- Dynamic pattern discovery and adaptive validation
- Support for PDF and Excel documents
- Parallel processing for large documents
- Export to JSON and GraphML formats

Basic Usage:
    from kg_creation import create_knowledge_graph
    
    result = create_knowledge_graph(
        file_path="document.pdf",
        output_dir="./output"
    )
"""

from .extractor import (
    KGExtractor,
    create_knowledge_graph,
    DocumentTypeConfig,
    TQDCSFramework
)

from .kg_pipeline import KGPipeline

from .models.kg_schema import (
    KnowledgeGraph,
    Node,
    Relationship,
    SourceMetadata,
    NodeType,
    RelationshipLabel,
    StructuredValue
)

__version__ = "2.0.0"  # Unified version after migration

__all__ = [
    # Main functions
    "create_knowledge_graph",
    
    # Core classes
    "KGExtractor",
    "KGPipeline",
    "TQDCSFramework",
    "DocumentTypeConfig",
    
    # Data models
    "KnowledgeGraph",
    "Node",
    "Relationship",
    "SourceMetadata",
    "NodeType",
    "RelationshipLabel",
    "StructuredValue",
]