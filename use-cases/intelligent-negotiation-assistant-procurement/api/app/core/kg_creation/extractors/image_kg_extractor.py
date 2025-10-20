"""
Image-based Knowledge Graph Extractor

This module provides direct image-to-knowledge-graph extraction capabilities,
bypassing text transcription for improved accuracy with complex documents.
"""

import json
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from ..models.kg_schema import (
    KnowledgeGraph, Node, Relationship, SourceMetadata, 
    StructuredValue, NodeType
)
from ..llm import create_llm
from ..extractor import KGExtractor, TQDCSFramework
from ..validation.kg_validator import KGValidator

logger = logging.getLogger(__name__)


class ImageKGExtractor(KGExtractor):
    """
    Knowledge Graph extractor that works directly with images.
    
    This extractor processes document images directly to extract entities
    and relationships without intermediate text transcription.
    """
    
    def __init__(self, 
                 llm_model: str = "gemini-2.5-pro",  # Default to Gemini for vision
                 temperature: float = 0.1,
                 enable_validation: bool = True,
                 custom_patterns: Optional[Dict[str, Any]] = None):
        """Initialize the image-based extractor."""
        # Initialize parent class
        super().__init__(
            llm_model=llm_model,
            temperature=temperature,
            enable_validation=enable_validation,
            custom_patterns=custom_patterns
        )
        
        # Override parser for image extraction
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
        # Load image-specific prompts
        self.image_prompts = self._load_image_prompts()
    
    def extract_knowledge_graph_from_image(self, 
                                          image_path: str,
                                          page_metadata: SourceMetadata,
                                          additional_context: Optional[Dict[str, Any]] = None) -> KnowledgeGraph:
        """
        Extract a knowledge graph directly from an image.
        
        Args:
            image_path: Path to the image file
            page_metadata: Source metadata for this page
            additional_context: Optional context from previous pages
            
        Returns:
            KnowledgeGraph object with extracted entities and relationships
        """
        logger.info(f"Extracting KG from image: {page_metadata.chunk_id}")
        
        try:
            # Load and prepare image
            image_data = self._prepare_image(image_path)
            
            # Create extraction prompt
            prompt = self._create_image_extraction_prompt(additional_context)
            
            # Build multimodal message
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
            
            message = HumanMessage(content=content)
            
            # Extract knowledge graph
            response = self.llm.invoke([message])
            
            # Parse response
            kg = self._parse_image_extraction_response(response.content, page_metadata)
            
            # Post-process and validate
            kg = self._post_process_kg(kg, page_metadata)
            
            if self.enable_validation:
                # Validate without original text (image-based validation)
                kg = self._validate_image_extracted_kg(kg, image_path)
            
            return kg
            
        except Exception as e:
            logger.error(f"Image extraction failed for {page_metadata.chunk_id}: {str(e)}")
            raise
    
    def extract_from_image_batch(self,
                                image_paths: List[str],
                                base_metadata: SourceMetadata,
                                batch_context: Optional[Dict[str, Any]] = None) -> KnowledgeGraph:
        """
        Extract knowledge graph from a batch of images (e.g., multi-page tables).
        
        Args:
            image_paths: List of image file paths
            base_metadata: Base source metadata
            batch_context: Context for the batch
            
        Returns:
            Merged KnowledgeGraph from all images
        """
        logger.info(f"Extracting KG from batch of {len(image_paths)} images")
        
        # Prepare all images
        image_data_list = []
        for path in image_paths:
            image_data_list.append(self._prepare_image(path))
        
        # Create batch extraction prompt
        prompt = self._create_batch_extraction_prompt(len(image_paths), batch_context)
        
        # Build multimodal message with multiple images
        content = [{"type": "text", "text": prompt}]
        for i, image_data in enumerate(image_data_list):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })
        
        message = HumanMessage(content=content)
        
        # Extract knowledge graph
        response = self.llm.invoke([message])
        
        # Parse and process
        kg = self._parse_image_extraction_response(response.content, base_metadata)
        kg = self._post_process_kg(kg, base_metadata)
        
        return kg
    
    def _prepare_image(self, image_path: str, max_dimension: int = 2048) -> str:
        """
        Prepare image for LLM processing.
        
        Args:
            image_path: Path to the image file
            max_dimension: Maximum dimension to resize to
            
        Returns:
            Base64 encoded image string
        """
        with Image.open(image_path) as img:
            # Resize if too large
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {img.size} to fit {max_dimension}")
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            image_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _create_image_extraction_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for single image extraction."""
        base_prompt = self.image_prompts['single_image']
        
        # Add TQDCS categories description
        tqdcs_info = self._format_tqdcs_for_prompt()
        
        # Add format instructions
        format_instructions = self.parser.get_format_instructions()
        
        # Add context if available
        context_str = ""
        if context:
            context_str = f"\n\nContext from previous pages:\n{json.dumps(context, indent=2)}"
        
        return base_prompt.format(
            tqdcs_categories=tqdcs_info,
            format_instructions=format_instructions,
            context=context_str
        )
    
    def _create_batch_extraction_prompt(self, num_images: int, context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for batch image extraction."""
        base_prompt = self.image_prompts['batch_images']
        
        # Add TQDCS categories description
        tqdcs_info = self._format_tqdcs_for_prompt()
        
        # Add format instructions
        format_instructions = self.parser.get_format_instructions()
        
        # Add context if available
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        return base_prompt.format(
            num_images=num_images,
            tqdcs_categories=tqdcs_info,
            format_instructions=format_instructions,
            context=context_str
        )
    
    def _format_tqdcs_for_prompt(self) -> str:
        """Format TQDCS categories for prompt inclusion."""
        tqdcs_lines = []
        for cat_key, category in self.tqdcs.categories.items():
            tqdcs_lines.append(
                f"- {cat_key[0]} ({category.name}): {category.description}"
            )
        return "\n".join(tqdcs_lines)
    
    def _parse_image_extraction_response(self, 
                                       response_content: str, 
                                       metadata: SourceMetadata) -> KnowledgeGraph:
        """Parse LLM response into KnowledgeGraph object."""
        try:
            # Try to parse as JSON first
            if response_content.strip().startswith('{'):
                kg_dict = json.loads(response_content)
                
                # Clean up node data - fix metadata format if needed
                cleaned_nodes = []
                for node_data in kg_dict.get('nodes', []):
                    # If metadata contains tqdcs, move it to properties
                    if isinstance(node_data.get('metadata'), dict) and 'tqdcs' in node_data['metadata']:
                        if 'properties' not in node_data:
                            node_data['properties'] = {}
                        node_data['properties']['tqdcs_categories'] = node_data['metadata']['tqdcs']
                        node_data['metadata'] = []  # Reset to empty list
                    elif not isinstance(node_data.get('metadata'), list):
                        node_data['metadata'] = []  # Ensure it's a list
                    
                    cleaned_nodes.append(node_data)
                
                # Clean up relationship data
                cleaned_relationships = []
                for rel_data in kg_dict.get('relationships', []):
                    # Ensure metadata is a dict with required fields
                    if not isinstance(rel_data.get('metadata'), dict):
                        rel_data['metadata'] = {}
                    
                    # If metadata is empty or missing required fields, populate them
                    if 'filename' not in rel_data['metadata'] or 'chunk_id' not in rel_data['metadata']:
                        rel_data['metadata'] = {
                            'filename': metadata.filename,
                            'chunk_id': metadata.chunk_id
                        }
                    
                    cleaned_relationships.append(rel_data)
                
                # Convert to KnowledgeGraph object
                kg = KnowledgeGraph(
                    nodes=[Node(**node_data) for node_data in cleaned_nodes],
                    relationships=[Relationship(**rel_data) for rel_data in cleaned_relationships]
                )
            else:
                # Use the parser
                kg = self.parser.parse(response_content)
            
            # Add proper metadata to all nodes and relationships
            for node in kg.nodes:
                if not node.metadata:
                    node.metadata = [metadata]
                elif not any(m.chunk_id == metadata.chunk_id for m in node.metadata):
                    node.metadata.append(metadata)
            
            for rel in kg.relationships:
                # Only set metadata if it's completely missing or invalid
                if not rel.metadata or not hasattr(rel.metadata, 'filename') or not hasattr(rel.metadata, 'chunk_id'):
                    rel.metadata = metadata
            
            return kg
            
        except Exception as e:
            logger.error(f"Failed to parse image extraction response: {e}")
            logger.debug(f"Response content: {response_content[:500]}...")
            # Return empty KG on parse failure
            return KnowledgeGraph(nodes=[], relationships=[])
    
    def _validate_image_extracted_kg(self, kg: KnowledgeGraph, image_path: str) -> KnowledgeGraph:
        """
        Validate knowledge graph extracted from image.
        
        Since we don't have text to validate against, this performs
        structural validation and consistency checks.
        """
        # Ensure all nodes have required properties
        for node in kg.nodes:
            # Ensure TQDCS categories are properly formatted
            if 'tqdcs_categories' in node.properties:
                node.properties['tqdcs_categories'] = self._fix_tqdcs_categories(node)
            
            # Ensure numerical values have units
            for key, value in list(node.properties.items()):
                if isinstance(value, (int, float)):
                    # Try to infer unit from property name
                    unit = self._infer_unit_from_property(key, value, node)
                    node.properties[key] = {"value": value, "unit": unit}
        
        # Validate relationships
        node_ids = {node.id for node in kg.nodes}
        valid_relationships = []
        
        for rel in kg.relationships:
            if rel.source in node_ids and rel.target in node_ids:
                valid_relationships.append(rel)
            else:
                logger.warning(f"Removing invalid relationship: {rel.source} -> {rel.target}")
        
        kg.relationships = valid_relationships
        
        # Deduplicate nodes
        kg.nodes = self._intelligent_deduplication(kg.nodes)
        
        return kg
    
    def _infer_unit_from_property(self, property_name: str, value: Any, node: Node) -> str:
        """Infer unit based on property name and node context."""
        property_lower = property_name.lower()
        
        # Currency indicators
        if any(term in property_lower for term in ['price', 'cost', 'fee', 'charge', 'amount']):
            return 'EUR'  # Default currency
        
        # Percentage indicators
        if any(term in property_lower for term in ['rate', 'percentage', 'percent', 'ratio']):
            return 'percent'
        
        # Time indicators
        if any(term in property_lower for term in ['time', 'duration', 'period', 'lead']):
            if 'day' in property_lower:
                return 'days'
            elif 'week' in property_lower:
                return 'weeks'
            return 'days'
        
        # Quantity indicators
        if any(term in property_lower for term in ['quantity', 'volume', 'count', 'pieces']):
            return 'units'
        
        # Weight indicators
        if any(term in property_lower for term in ['weight', 'mass']):
            if value < 1000:
                return 'kg'
            else:
                return 'tonnes'
        
        return ""  # No unit if unclear
    
    def _load_image_prompts(self) -> Dict[str, str]:
        """Load image-specific extraction prompts."""
        prompts = {}
        
        # Single image extraction prompt
        prompts['single_image'] = """Analyze this document image and extract a knowledge graph following the TQDCS framework.

TQDCS Categories:
{tqdcs_categories}

CRITICAL SCHEMA REQUIREMENTS - Your output MUST follow these exact Pydantic schemas:

Node Schema:
{{
  "id": "string",         // Format: "type:identifier" (e.g., "part:ABC-123")
  "type": "string",       // One of: Part, Cost, Supplier, Date, Location, Organization, Drawing, Certification, GenericInformation
  "properties": {{         // Dictionary with all attributes
    "tqdcs_categories": ["T", "Q", "D", "C", "S"],  // REQUIRED for most nodes
    // ... other properties
  }},
  "metadata": []          // MUST be empty list [] - will be filled automatically
}}

Relationship Schema:
{{
  "source": "string",     // Node ID that exists in nodes list
  "target": "string",     // Node ID that exists in nodes list
  "label": "string",      // Relationship type (e.g., "HAS_COST", "SUPPLIES")
  "properties": {{}},       // Optional properties dictionary
  "metadata": {{           // REQUIRED - MUST be a dict with these exact fields:
    "filename": "document.pdf",  // Placeholder - will be filled automatically
    "chunk_id": "page_X"         // Placeholder - will be filled automatically
  }}
}}

WARNING: Relationship metadata CANNOT be empty {{}} - it MUST include filename and chunk_id fields!

COMPLETE VALID EXAMPLE:
{{
  "nodes": [
    {{
      "id": "part:F-SC-P19_170",
      "type": "Part",
      "properties": {{
        "part_number": "F SC P19_170",
        "description": "DW5 clutch system",
        "tqdcs_categories": ["T", "Q", "D", "C"]
      }},
      "metadata": []
    }},
    {{
      "id": "cost:F-SC-P19_170-price",
      "type": "Cost",
      "properties": {{
        "amount": {{"value": 1500.0, "unit": "EUR"}},
        "cost_type": "unit_price",
        "tqdcs_categories": ["C"]
      }},
      "metadata": []
    }}
  ],
  "relationships": [
    {{
      "source": "part:F-SC-P19_170",
      "target": "cost:F-SC-P19_170-price",
      "label": "HAS_COST",
      "properties": {{}},
      "metadata": {{"filename": "doc.pdf", "chunk_id": "page_1"}}
    }}
  ]
}}

Instructions:
1. Identify ALL entities visible in the image:
   - Parts/Products (with part numbers, descriptions)
   - Suppliers/Organizations (company names, departments)
   - Costs/Prices (with amounts and currencies)
   - Technical specifications (measurements, standards)
   - Quality certifications (ISO standards, ratings)
   - Delivery information (lead times, quantities)
   - Sustainability data (emissions, renewable percentages)

2. Extract relationships between entities:
   - HAS_COST: Part/Service -> Cost
   - SUPPLIES: Supplier -> Part
   - HAS_SPECIFICATION: Part -> Technical_Specification
   - MEETS_STANDARD: Product -> Certification
   - LOCATED_AT: Organization -> Location

3. For each entity:
   - Create descriptive IDs (e.g., "part:ABC-123", "supplier:CompanyName")
   - Store TQDCS categories in properties field
   - Extract all visible properties and values
   - Use StructuredValue format for numerical values: {{"value": 123.45, "unit": "EUR"}}

4. CRITICAL RULES:
   - Node metadata: MUST be empty list []
   - Relationship metadata: MUST be dict with filename AND chunk_id (not empty {{}})
   - TQDCS categories go in properties, NEVER in metadata
   - Organization/Location nodes: "tqdcs_categories": []
   - Part/Cost/Technical nodes: "tqdcs_categories": ["T", "C"] etc.
   - All numerical values must use StructuredValue format

{context}

{format_instructions}

Extract the knowledge graph from the image now:"""
        
        # Batch images extraction prompt
        prompts['batch_images'] = """Analyze these {num_images} document images that form a continuous section (e.g., multi-page table or related content).

TQDCS Categories:
{tqdcs_categories}

CRITICAL SCHEMA REQUIREMENTS - Your output MUST follow these exact Pydantic schemas:

Node Schema:
{{
  "id": "string",         // Format: "type:identifier" (e.g., "part:ABC-123")
  "type": "string",       // One of: Part, Cost, Supplier, Date, Location, Organization, Drawing, Certification, GenericInformation
  "properties": {{         // Dictionary with all attributes
    "tqdcs_categories": ["T", "Q", "D", "C", "S"],  // REQUIRED for most nodes
    // ... other properties
  }},
  "metadata": []          // MUST be empty list [] - will be filled automatically
}}

Relationship Schema:
{{
  "source": "string",     // Node ID that exists in nodes list
  "target": "string",     // Node ID that exists in nodes list
  "label": "string",      // Relationship type (e.g., "HAS_COST", "SUPPLIES")
  "properties": {{}},       // Optional properties dictionary
  "metadata": {{           // REQUIRED - MUST be a dict with these exact fields:
    "filename": "document.pdf",  // Placeholder - will be filled automatically
    "chunk_id": "page_X"         // Placeholder - will be filled automatically
  }}
}}

WARNING: Relationship metadata CANNOT be empty {{}} - it MUST include filename and chunk_id fields!

Instructions:
1. Treat the images as a continuous document section
2. Identify entities that may span across images
3. Maintain consistency in entity IDs across pages
4. Link related information from different images

CRITICAL RULES - Same as single image:
- Node metadata: MUST be empty list []
- Relationship metadata: MUST be dict with filename AND chunk_id (not empty {{}})
- TQDCS categories go in properties, NEVER in metadata
- All numerical values must use StructuredValue format

{context}

{format_instructions}

Extract the unified knowledge graph from all images:"""
        
        return prompts


def process_images_to_kg(
    image_extraction_result: Dict[str, Any],
    llm_model: str = "gemini-2.5-pro",
    temperature: float = 0.1,
    parallel_processing: bool = True,
    max_workers: int = 5,
    batch_size: int = 1
) -> KnowledgeGraph:
    """
    Process extracted images to create a knowledge graph.
    
    Args:
        image_extraction_result: Result from extract_pdf_content_for_nodes
        llm_model: LLM model to use for extraction
        temperature: Temperature for LLM
        parallel_processing: Enable parallel processing
        max_workers: Maximum parallel workers
        batch_size: Number of images to process together
        
    Returns:
        Complete KnowledgeGraph extracted from all images
    """
    from .pdf_image_extractor import get_image_batch_groups, cleanup_temp_images
    
    logger.info(f"Processing {len(image_extraction_result['pages'])} images to extract knowledge graph")
    
    # Initialize extractor
    extractor = ImageKGExtractor(
        llm_model=llm_model,
        temperature=temperature,
        enable_validation=True
    )
    
    # Group pages into batches if needed
    if batch_size > 1:
        page_batches = get_image_batch_groups(
            image_extraction_result['pages'],
            max_batch_size=batch_size
        )
    else:
        # Each page is its own batch
        page_batches = [[page] for page in image_extraction_result['pages']]
    
    # Extract KGs from each batch
    batch_kgs = []
    
    if parallel_processing and len(page_batches) > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            
            for batch_idx, batch in enumerate(page_batches):
                # Create metadata for batch
                if len(batch) == 1:
                    metadata = SourceMetadata(
                        filename=image_extraction_result['filename'],
                        chunk_id=f"page_{batch[0]['page_number']}"
                    )
                else:
                    page_nums = [p['page_number'] for p in batch]
                    metadata = SourceMetadata(
                        filename=image_extraction_result['filename'],
                        chunk_id=f"pages_{min(page_nums)}-{max(page_nums)}"
                    )
                
                # Submit extraction task
                future = executor.submit(
                    _extract_batch_safe,
                    extractor,
                    batch,
                    metadata
                )
                future_to_batch[future] = (batch_idx, batch)
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    kg = future.result()
                    if kg:
                        batch_kgs.append(kg)
                        logger.info(f"Extracted {len(kg.nodes)} nodes from batch {batch_idx + 1}")
                except Exception as e:
                    logger.error(f"Failed to extract from batch {batch_idx + 1}: {e}")
    else:
        # Sequential processing
        for batch_idx, batch in enumerate(page_batches):
            try:
                # Create metadata
                if len(batch) == 1:
                    metadata = SourceMetadata(
                        filename=image_extraction_result['filename'],
                        chunk_id=f"page_{batch[0]['page_number']}"
                    )
                    # Single image extraction
                    kg = extractor.extract_knowledge_graph_from_image(
                        batch[0]['image_path'],
                        metadata
                    )
                else:
                    # Batch extraction
                    page_nums = [p['page_number'] for p in batch]
                    metadata = SourceMetadata(
                        filename=image_extraction_result['filename'],
                        chunk_id=f"pages_{min(page_nums)}-{max(page_nums)}"
                    )
                    image_paths = [p['image_path'] for p in batch]
                    kg = extractor.extract_from_image_batch(
                        image_paths,
                        metadata
                    )
                
                batch_kgs.append(kg)
                logger.info(f"Extracted {len(kg.nodes)} nodes from batch {batch_idx + 1}")
                
            except Exception as e:
                logger.error(f"Failed to extract from batch {batch_idx + 1}: {e}")
    
    # Merge all batch KGs
    final_kg = _merge_knowledge_graphs(batch_kgs)
    
    # Clean up temporary images
    cleanup_temp_images(image_extraction_result)
    
    logger.info(f"Final knowledge graph: {len(final_kg.nodes)} nodes, {len(final_kg.relationships)} relationships")
    
    return final_kg


def _extract_batch_safe(extractor: ImageKGExtractor, 
                       batch: List[Dict[str, Any]], 
                       metadata: SourceMetadata) -> Optional[KnowledgeGraph]:
    """Safely extract KG from a batch (for parallel processing)."""
    try:
        if len(batch) == 1:
            # Single image
            return extractor.extract_knowledge_graph_from_image(
                batch[0]['image_path'],
                metadata
            )
        else:
            # Multiple images
            image_paths = [p['image_path'] for p in batch]
            return extractor.extract_from_image_batch(
                image_paths,
                metadata
            )
    except Exception as e:
        logger.error(f"Batch extraction failed: {str(e)}")
        return None


def _merge_knowledge_graphs(graphs: List[KnowledgeGraph]) -> KnowledgeGraph:
    """Merge multiple knowledge graphs with deduplication."""
    merged = KnowledgeGraph(nodes=[], relationships=[])
    node_map = {}
    
    for kg in graphs:
        # Merge nodes
        for node in kg.nodes:
            if node.id in node_map:
                # Merge properties
                existing = node_map[node.id]
                existing.properties.update(node.properties)
                
                # Merge metadata
                if node.metadata:
                    existing_metadata_keys = {
                        (m.filename, m.chunk_id) for m in existing.metadata
                    }
                    
                    for new_meta in node.metadata:
                        meta_key = (new_meta.filename, new_meta.chunk_id)
                        if meta_key not in existing_metadata_keys:
                            existing.metadata.append(new_meta)
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