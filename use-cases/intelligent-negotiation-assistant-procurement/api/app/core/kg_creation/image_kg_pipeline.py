"""
Image-based Knowledge Graph Pipeline

This module provides an alternative pipeline that processes PDFs directly
as images, bypassing text extraction for improved accuracy with complex layouts.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .models.kg_schema import KnowledgeGraph
from .kg_pipeline import KGPipeline
from .extractors.pdf_image_extractor import extract_pdf_content_for_nodes
from .extractors.image_kg_extractor import process_images_to_kg
from .serialization.graph_exporter import GraphExporter

logger = logging.getLogger(__name__)


class ImageKGPipeline(KGPipeline):
    """
    Pipeline for image-based knowledge graph extraction.
    
    This pipeline converts PDFs to images and extracts knowledge graphs
    directly from the visual content without text transcription.
    """
    
    def __init__(self,
                 output_dir: str = "./kg_output",
                 llm_model: str = "gemini-2.5-pro",
                 enable_validation: bool = True,
                 parallel_processing: bool = True,
                 max_workers: Optional[int] = None,
                 image_batch_size: int = 1,
                 dpi: int = 200,
                 max_image_dimension: int = 2048):
        """
        Initialize the image-based pipeline.
        
        Args:
            output_dir: Directory for output files
            llm_model: LLM model to use (should support vision)
            enable_validation: Enable knowledge graph validation
            parallel_processing: Enable parallel processing
            max_workers: Maximum parallel workers
            image_batch_size: Number of images to process together
            dpi: DPI for PDF to image conversion
            max_image_dimension: Maximum dimension for images
        """
        super().__init__(
            output_dir=output_dir,
            llm_model=llm_model,
            enable_validation=enable_validation,
            parallel_processing=parallel_processing,
            max_workers=max_workers
        )
        
        # Store attributes that parent class might not store
        self.llm_model = llm_model
        self.enable_validation = enable_validation
        self.image_batch_size = image_batch_size
        self.dpi = dpi
        self.max_image_dimension = max_image_dimension
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document using image-based extraction.
        
        Args:
            file_path: Path to the PDF document
            
        Returns:
            Dictionary with extraction results
        """
        file_path = Path(file_path)
        logger.info(f"Processing document with image-based extraction: {file_path}")
        
        # Currently only supports PDF
        if file_path.suffix.lower() != '.pdf':
            logger.warning(f"Image-based extraction only supports PDF files, falling back to text-based extraction")
            return super().process_document(str(file_path))
        
        try:
            # Extract PDF as images
            logger.info("Converting PDF to images...")
            image_extraction_result = extract_pdf_content_for_nodes(
                str(file_path),
                dpi=self.dpi,
                max_dimension=self.max_image_dimension,
                parallel_processing=self.parallel_processing,
                max_workers=self.max_workers
            )
            
            logger.info(f"Extracted {image_extraction_result['page_count']} pages as images")
            
            # Process images to extract knowledge graph
            logger.info("Extracting knowledge graph from images...")
            final_kg = process_images_to_kg(
                image_extraction_result,
                llm_model=self.llm_model,
                temperature=0.1,
                parallel_processing=self.parallel_processing,
                max_workers=self.max_workers,
                batch_size=self.image_batch_size
            )
            
            # Perform document-level validation if enabled
            if self.enable_validation:
                logger.info("Performing document-level validation on extracted knowledge graph")
                # For image-based extraction, we validate structure only
                # since we don't have text to validate against
                final_kg = self._validate_image_kg_structure(final_kg)
            
            # Export results
            exporter = GraphExporter(self.output_dir)
            output_files = {
                'json': exporter.export_json(final_kg, f"{file_path.stem}_kg.json"),
                'graphml': exporter.export_graphml(final_kg, f"{file_path.stem}_kg.graphml.gz")
            }
            
            # Generate statistics
            statistics = self._generate_statistics(final_kg)
            statistics['extraction_method'] = 'image-direct'
            statistics['pages_processed'] = image_extraction_result['page_count']
            
            return {
                'knowledge_graph': final_kg,
                'output_files': output_files,
                'statistics': statistics,
                'extraction_method': 'image-direct'
            }
            
        except Exception as e:
            logger.error(f"Image-based extraction failed: {str(e)}")
            logger.info("Attempting fallback to text-based extraction...")
            return super().process_document(str(file_path))
    
    def _validate_image_kg_structure(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Validate the structure of an image-extracted knowledge graph.
        
        This performs consistency checks without reference to original text.
        """
        # Count issues for logging
        issues_fixed = 0
        
        # Ensure all nodes have valid IDs
        for node in kg.nodes:
            if ':' not in node.id:
                node.id = f"{node.type.value}:{node.id}"
                issues_fixed += 1
        
        # Validate relationships reference existing nodes
        node_ids = {node.id for node in kg.nodes}
        valid_relationships = []
        
        for rel in kg.relationships:
            if rel.source in node_ids and rel.target in node_ids:
                valid_relationships.append(rel)
            else:
                logger.warning(f"Removing orphaned relationship: {rel.source} -> {rel.target}")
                issues_fixed += 1
        
        kg.relationships = valid_relationships
        
        # Ensure TQDCS categories are properly formatted
        for node in kg.nodes:
            if 'tqdcs_categories' in node.properties:
                original = node.properties['tqdcs_categories']
                fixed = self._fix_tqdcs_categories(node.properties['tqdcs_categories'])
                if original != fixed:
                    node.properties['tqdcs_categories'] = fixed
                    issues_fixed += 1
        
        if issues_fixed > 0:
            logger.info(f"Fixed {issues_fixed} structural issues in knowledge graph")
        
        return kg
    
    def _fix_tqdcs_categories(self, categories: Any) -> list:
        """Fix TQDCS categories to ensure they are single letters."""
        if not isinstance(categories, list):
            return []
        
        fixed = []
        category_map = {
            'technology': 'T', 'tech': 'T', 't': 'T',
            'quality': 'Q', 'qual': 'Q', 'q': 'Q',
            'delivery': 'D', 'del': 'D', 'd': 'D',
            'cost': 'C', 'price': 'C', 'c': 'C',
            'sustainability': 'S', 'sustain': 'S', 's': 'S'
        }
        
        for cat in categories:
            if isinstance(cat, str):
                cat_lower = cat.lower().strip()
                if cat_lower in ['t', 'q', 'd', 'c', 's']:
                    fixed.append(cat_lower.upper())
                elif cat_lower in category_map:
                    fixed.append(category_map[cat_lower])
        
        return sorted(list(set(fixed)))


def create_image_based_kg(
    file_path: str,
    output_dir: str = "./kg_output",
    llm_model: str = "gemini-2.5-pro",
    parallel_processing: bool = True,
    max_workers: Optional[int] = 5,
    image_batch_size: int = 1,
    dpi: int = 200
) -> Dict[str, Any]:
    """
    Create a knowledge graph using image-based extraction.
    
    This is a convenience function that creates an ImageKGPipeline
    and processes the document.
    
    Args:
        file_path: Path to the PDF document
        output_dir: Directory for output files
        llm_model: LLM model to use (should support vision)
        parallel_processing: Enable parallel processing
        max_workers: Maximum parallel workers
        image_batch_size: Number of images to process together
        dpi: DPI for PDF to image conversion
        
    Returns:
        Dictionary with extraction results
    """
    pipeline = ImageKGPipeline(
        output_dir=output_dir,
        llm_model=llm_model,
        enable_validation=True,
        parallel_processing=parallel_processing,
        max_workers=max_workers,
        image_batch_size=image_batch_size,
        dpi=dpi
    )
    
    return pipeline.process_document(file_path)