"""
Knowledge Graph Pipeline Orchestrator

This module handles the full pipeline process:
1. Document extraction (PDF/Excel)
2. Chunking
3. Knowledge graph extraction from chunks
4. Merging and deduplication
5. Export to various formats
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models.kg_schema import KnowledgeGraph, SourceMetadata, NodeType
from .extractor import KGExtractor
from .validation.kg_validator import KGValidator

logger = logging.getLogger(__name__)


class KGPipeline:
    """
    Pipeline orchestrator for knowledge graph extraction.
    """
    
    def __init__(self, 
                 output_dir: str = "./kg_output",
                 llm_model: str = "gpt-4.1",
                 enable_validation: bool = True,
                 parallel_processing: bool = True,
                 max_workers: Optional[int] = None):
        """Initialize the pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.extractor = KGExtractor(
            llm_model=llm_model,
            enable_validation=enable_validation
        )
        
        # Parallel processing settings
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document with the pipeline."""
        from .extractors.pdf_extractor import extract_pdf_for_kg_pipeline
        from .extractors.excel_extractor import extract_excel_for_kg_pipeline
        from .extractors.markdown_extractor import extract_markdown_for_kg_pipeline
        from .chunking.context_chunker import chunk_extraction_result
        from .serialization.graph_exporter import GraphExporter
        
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        if file_path.suffix.lower() == '.pdf':
            extraction_result = extract_pdf_for_kg_pipeline(str(file_path))
        elif file_path.suffix.lower() == '.md':
            extraction_result = extract_markdown_for_kg_pipeline(str(file_path))
        else:
            extraction_result = extract_excel_for_kg_pipeline(str(file_path))
        
        # Chunk document
        chunks = chunk_extraction_result(extraction_result, merge_small_chunks=False)
        
        # Extract KGs from each chunk
        chunk_graphs = []
        
        if self.parallel_processing and len(chunks) > 1:
            # Parallel extraction
            logger.info(f"Processing {len(chunks)} chunks in parallel with {self.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {}
                for chunk in chunks:
                    # Extract simple chunk_id without filename prefix
                    simple_chunk_id = chunk.chunk_id.split(':', 1)[-1] if ':' in chunk.chunk_id else chunk.chunk_id
                    metadata = SourceMetadata(
                        filename=extraction_result['filename'],
                        chunk_id=simple_chunk_id
                    )
                    future = executor.submit(
                        self._extract_chunk_safe,
                        chunk,
                        metadata
                    )
                    future_to_chunk[future] = (chunk, metadata)
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk, metadata = future_to_chunk[future]
                    try:
                        kg = future.result()
                        if kg:
                            chunk_graphs.append(kg)
                    except Exception as e:
                        logger.error(f"Failed to extract from chunk {chunk.chunk_id}: {e}")
        else:
            # Sequential extraction
            logger.info(f"Processing {len(chunks)} chunks sequentially")
            for chunk in chunks:
                # Extract simple chunk_id without filename prefix
                simple_chunk_id = chunk.chunk_id.split(':', 1)[-1] if ':' in chunk.chunk_id else chunk.chunk_id
                metadata = SourceMetadata(
                    filename=extraction_result['filename'],
                    chunk_id=simple_chunk_id
                )
                
                try:
                    kg = self.extractor.extract_knowledge_graph(chunk.chunk_text, metadata)
                    chunk_graphs.append(kg)
                except Exception as e:
                    logger.error(f"Failed to extract from chunk {chunk.chunk_id}: {e}")
        
        # Merge chunk graphs with proper entity resolution
        final_kg = self._merge_knowledge_graphs(chunk_graphs)
        
        # Perform document-level validation if enabled
        if self.extractor.enable_validation:
            logger.info("Performing document-level validation on merged knowledge graph")
            
            # Create validator for document-level validation
            document_validator = KGValidator(
                llm_model=self.extractor.llm_model,
                temperature=self.extractor.temperature
            )
            
            # Combine all chunk texts for full document validation
            full_text = "\n\n".join([chunk.chunk_text for chunk in chunks])
            
            # Validate against the full document
            validation_result = document_validator.validate_knowledge_graph(
                kg=final_kg,
                original_text=full_text,
                metadata=SourceMetadata(
                    filename=extraction_result['filename'],
                    chunk_id="document_validation"
                )
            )
            
            if validation_result.validation_successful:
                final_kg = validation_result.validated_kg
                logger.info(
                    f"Document-level validation complete: "
                    f"+{validation_result.nodes_added} nodes, "
                    f"+{validation_result.relationships_added} relationships, "
                    f"{validation_result.properties_enhanced} properties enhanced"
                )
            else:
                logger.warning(f"Document-level validation failed: {validation_result.notes}")
        
        # Export results
        exporter = GraphExporter(self.output_dir)
        output_files = {
            'json': exporter.export_json(final_kg, f"{file_path.stem}_kg.json"),
            'graphml': exporter.export_graphml(final_kg, f"{file_path.stem}_kg.graphml.gz")
        }
        
        return {
            'knowledge_graph': final_kg,
            'output_files': output_files,
            'statistics': self._generate_statistics(final_kg)
        }
    
    def _extract_chunk_safe(self, chunk, metadata: SourceMetadata) -> Optional[KnowledgeGraph]:
        """Safely extract KG from a chunk (for parallel processing)."""
        try:
            return self.extractor.extract_knowledge_graph(chunk.chunk_text, metadata)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {metadata.chunk_id}: {str(e)}")
            return None
    
    def _merge_knowledge_graphs(self, graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
        """Merge multiple KGs with proper entity resolution and metadata deduplication."""
        merged = KnowledgeGraph(nodes=[], relationships=[])
        node_map = {}  # Track nodes by ID for deduplication
        
        for kg in graphs:
            # Merge nodes
            for node in kg.nodes:
                if node.id in node_map:
                    # Merge properties
                    existing = node_map[node.id]
                    existing.properties.update(node.properties)
                    
                    # Merge metadata with deduplication
                    if node.metadata:
                        # Track existing metadata by (filename, chunk_id) tuple
                        existing_metadata_keys = {
                            (m.filename, m.chunk_id) for m in existing.metadata
                        }
                        
                        # Add only new metadata
                        for new_meta in node.metadata:
                            meta_key = (new_meta.filename, new_meta.chunk_id)
                            if meta_key not in existing_metadata_keys:
                                existing.metadata.append(new_meta)
                                existing_metadata_keys.add(meta_key)
                else:
                    node_map[node.id] = node
                    merged.nodes.append(node)
            
            # Add relationships
            merged.relationships.extend(kg.relationships)
        
        # Deduplicate relationships
        unique_rels = []
        seen_rels = set()
        for rel in merged.relationships:
            rel_key = (rel.source, rel.target, rel.label)
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)
        
        merged.relationships = unique_rels
        
        return merged
    
    def _generate_statistics(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Generate statistics about the extracted KG."""
        from collections import Counter
        
        node_types = Counter(node.type.value for node in kg.nodes)
        rel_types = Counter(rel.label for rel in kg.relationships)
        
        return {
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships),
            'node_types': dict(node_types),
            'relationship_types': dict(rel_types),
            'parts_found': len([n for n in kg.nodes if n.type == NodeType.PART]),
            'costs_found': len([n for n in kg.nodes if n.type == NodeType.COST]),
            'structured_values': sum(
                1 for node in kg.nodes
                for value in node.properties.values()
                if isinstance(value, dict) and 'value' in value and 'unit' in value
            )
        }