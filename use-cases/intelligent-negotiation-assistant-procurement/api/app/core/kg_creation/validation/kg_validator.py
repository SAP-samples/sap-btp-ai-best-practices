"""
Knowledge Graph Validator

This module provides validation functionality to ensure that the extracted
knowledge graph captures all relevant information from the source text.
It performs a second pass over the text with the existing KG to identify
missing nodes, relationships, and properties.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models.kg_schema import (
    KnowledgeGraph, Node, Relationship, SourceMetadata, NodeType
)
from ..llm import create_llm

logger = logging.getLogger(__name__)


class ValidationFindings(BaseModel):
    """Model for validation findings from the LLM."""
    missing_nodes: List[Node] = Field(
        default_factory=list,
        description="Nodes that should be added to the knowledge graph"
    )
    missing_relationships: List[Relationship] = Field(
        default_factory=list,
        description="Relationships that should be added to the knowledge graph"
    )
    enhanced_properties: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Additional properties to add to existing nodes. Format: [{'node_id': 'xxx', 'properties': {...}}]"
    )
    validation_notes: str = Field(
        default="",
        description="Notes about the validation process and any issues found"
    )


@dataclass
class ValidationResult:
    """Result of knowledge graph validation."""
    validated_kg: KnowledgeGraph
    nodes_added: int
    relationships_added: int
    properties_enhanced: int
    validation_successful: bool
    notes: str
    tqdcs_fields_corrected: int = 0
    property_names_standardized: int = 0


class KGValidator:
    """
    Validates and enhances knowledge graphs by comparing them against source text.
    
    This validator performs a second pass over the text to ensure completeness
    of the extracted knowledge graph. It identifies missing nodes, relationships,
    and properties without modifying existing node IDs.
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4.1",
                 temperature: float = 0.0):
        """
        Initialize the validator.
        
        Args:
            llm_model: The LLM model to use for validation
            temperature: Temperature for LLM generation
        """
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = create_llm(
            model_name=llm_model,
            temperature=temperature
        )
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=ValidationFindings)
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load validation prompt templates from files."""
        prompts_dir = Path(__file__).parent.parent / "llm" / "prompts"
        prompts = {}
        
        # Define prompt files to load
        prompt_files = {
            'validation_system': 'validation_system_prompt.txt',
            'validation_human': 'validation_human_prompt.txt'
        }
        
        for key, filename in prompt_files.items():
            file_path = prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompts[key] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load validation prompt {filename}: {e}")
                    prompts[key] = None
            else:
                logger.warning(f"Validation prompt file not found: {file_path}")
                prompts[key] = None
        
        # Use default prompts if files not found
        if not prompts.get('validation_system'):
            prompts['validation_system'] = self._get_default_validation_system_prompt()
        if not prompts.get('validation_human'):
            prompts['validation_human'] = self._get_default_validation_human_prompt()
        
        return prompts
    
    def validate_knowledge_graph(self,
                               kg: KnowledgeGraph,
                               original_text: str,
                               metadata: Optional[SourceMetadata] = None) -> ValidationResult:
        """
        Validate and enhance a knowledge graph against the source text.
        
        Args:
            kg: The knowledge graph to validate
            original_text: The original text that was processed
            metadata: Optional source metadata for new nodes/relationships
            
        Returns:
            ValidationResult containing the enhanced KG and statistics
        """
        logger.info(f"Validating knowledge graph with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
        
        # First, standardize TQDCS field names and other properties
        tqdcs_corrections = self._validate_tqdcs_field_names(kg)
        
        # Standardize all node properties
        total_property_standardizations = 0
        for node in kg.nodes:
            total_property_standardizations += self._standardize_node_properties(node)
        
        if total_property_standardizations > 0:
            logger.info(f"Standardized {total_property_standardizations} property names across all nodes")
        
        try:
            # Create validation prompt
            prompt = self._create_validation_prompt()
            
            # Prepare KG summary for validation
            kg_summary = self._summarize_kg_for_validation(kg)
            
            # Invoke LLM for validation
            chain = prompt | self.llm | self.parser
            
            findings = chain.invoke({
                "original_text": original_text,
                "existing_nodes": kg_summary["nodes"],
                "existing_relationships": kg_summary["relationships"],
                "node_count": len(kg.nodes),
                "relationship_count": len(kg.relationships),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Apply findings to create enhanced KG
            enhanced_kg = self._apply_validation_findings(kg, findings, metadata)
            
            # Calculate statistics
            nodes_added = len(enhanced_kg.nodes) - len(kg.nodes)
            relationships_added = len(enhanced_kg.relationships) - len(kg.relationships)
            properties_enhanced = len(findings.enhanced_properties)
            
            logger.info(f"Validation complete: Added {nodes_added} nodes, {relationships_added} relationships, enhanced {properties_enhanced} properties")
            
            return ValidationResult(
                validated_kg=enhanced_kg,
                nodes_added=nodes_added,
                relationships_added=relationships_added,
                properties_enhanced=properties_enhanced,
                validation_successful=True,
                notes=findings.validation_notes,
                tqdcs_fields_corrected=tqdcs_corrections,
                property_names_standardized=total_property_standardizations
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            # Return original KG if validation fails
            return ValidationResult(
                validated_kg=kg,
                nodes_added=0,
                relationships_added=0,
                properties_enhanced=0,
                validation_successful=False,
                notes=f"Validation failed: {str(e)}",
                tqdcs_fields_corrected=tqdcs_corrections,
                property_names_standardized=total_property_standardizations
            )
    
    def _summarize_kg_for_validation(self, kg: KnowledgeGraph) -> Dict[str, str]:
        """
        Create a concise summary of the KG for validation prompt.
        
        Args:
            kg: The knowledge graph to summarize
            
        Returns:
            Dictionary with node and relationship summaries
        """
        # Summarize nodes by type
        nodes_summary = []
        nodes_by_type = {}
        
        for node in kg.nodes:
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            
            # Create concise node representation
            node_info = {
                "id": node.id,
                "type": node.type.value,
                "key_properties": self._extract_key_properties(node)
            }
            nodes_by_type[node.type].append(node_info)
        
        # Format nodes summary
        for node_type, nodes in nodes_by_type.items():
            nodes_summary.append(f"\n{node_type.value} ({len(nodes)} nodes):")
            for node in nodes[:5]:  # Show first 5 of each type
                props_str = ", ".join([f"{k}={v}" for k, v in node["key_properties"].items()])
                nodes_summary.append(f"  - {node['id']}: {props_str}")
            if len(nodes) > 5:
                nodes_summary.append(f"  ... and {len(nodes) - 5} more")
        
        # Summarize relationships by type
        relationships_summary = []
        rels_by_type = {}
        
        for rel in kg.relationships:
            if rel.label not in rels_by_type:
                rels_by_type[rel.label] = []
            rels_by_type[rel.label].append(rel)
        
        # Format relationships summary
        for rel_type, rels in rels_by_type.items():
            relationships_summary.append(f"\n{rel_type} ({len(rels)} relationships):")
            for rel in rels[:5]:  # Show first 5 of each type
                relationships_summary.append(f"  - {rel.source} → {rel.target}")
            if len(rels) > 5:
                relationships_summary.append(f"  ... and {len(rels) - 5} more")
        
        return {
            "nodes": "\n".join(nodes_summary),
            "relationships": "\n".join(relationships_summary)
        }
    
    def _extract_key_properties(self, node: Node) -> Dict[str, Any]:
        """Extract key properties from a node for summary."""
        key_props = {}
        
        # Priority properties to show
        priority_keys = ['name', 'part_number', 'value', 'unit_price', 'description', 
                        'tqdcs_categories', 'location', 'capacity', 'certification']
        
        for key in priority_keys:
            if key in node.properties:
                value = node.properties[key]
                # Format structured values
                if isinstance(value, dict) and 'value' in value and 'unit' in value:
                    key_props[key] = f"{value['value']} {value['unit']}"
                elif isinstance(value, list):
                    key_props[key] = ', '.join(str(v) for v in value)
                else:
                    key_props[key] = str(value)
        
        # Add other properties if we have room
        if len(key_props) < 3:
            for key, value in node.properties.items():
                if key not in priority_keys and len(key_props) < 3:
                    if isinstance(value, dict) and 'value' in value and 'unit' in value:
                        key_props[key] = f"{value['value']} {value['unit']}"
                    else:
                        key_props[key] = str(value)[:50]  # Truncate long values
        
        return key_props
    
    def _create_validation_prompt(self) -> ChatPromptTemplate:
        """Create the validation prompt template."""
        system_message = self.prompts['validation_system']
        human_message = self.prompts['validation_human']
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _apply_validation_findings(self, 
                                 original_kg: KnowledgeGraph,
                                 findings: ValidationFindings,
                                 metadata: Optional[SourceMetadata]) -> KnowledgeGraph:
        """
        Apply validation findings to create an enhanced knowledge graph.
        
        Args:
            original_kg: The original knowledge graph
            findings: Validation findings from the LLM
            metadata: Source metadata for new elements
            
        Returns:
            Enhanced knowledge graph
        """
        # Create a copy of the original KG
        enhanced_kg = KnowledgeGraph(
            nodes=original_kg.nodes.copy(),
            relationships=original_kg.relationships.copy()
        )
        
        # Track existing node IDs for validation
        existing_node_ids = {node.id for node in enhanced_kg.nodes}
        
        # Add missing nodes
        for node in findings.missing_nodes:
            # Ensure the node ID doesn't conflict with existing ones
            if node.id not in existing_node_ids:
                # Standardize properties before adding
                self._standardize_node_properties(node)
                # Validate TQDCS content if present
                if 'tqdcs_categories' in node.properties:
                    self._validate_tqdcs_content(node)
                # Add metadata if provided
                if metadata and not node.metadata:
                    node.metadata = [metadata]
                enhanced_kg.nodes.append(node)
                existing_node_ids.add(node.id)
            else:
                logger.warning(f"Skipping node with conflicting ID: {node.id}")
        
        # Add missing relationships
        for rel in findings.missing_relationships:
            # Validate that both endpoints exist
            if rel.source in existing_node_ids and rel.target in existing_node_ids:
                # Add metadata if provided and not already set
                if metadata and not rel.metadata:
                    rel.metadata = metadata
                enhanced_kg.relationships.append(rel)
            else:
                logger.warning(f"Skipping relationship with non-existent endpoints: {rel.source} → {rel.target}")
        
        # Enhance existing node properties
        node_map = {node.id: node for node in enhanced_kg.nodes}
        for enhancement in findings.enhanced_properties:
            node_id = enhancement.get('node_id')
            new_properties = enhancement.get('properties', {})
            
            if node_id in node_map and new_properties:
                node = node_map[node_id]
                
                # Standardize property names in the enhancement
                standardized_properties = {}
                property_aliases = self._get_property_aliases()
                
                for key, value in new_properties.items():
                    # Check if this is a non-standard property name
                    if key in property_aliases:
                        standardized_key = property_aliases[key]
                        standardized_properties[standardized_key] = value
                    else:
                        standardized_properties[key] = value
                
                # Add new properties without overwriting existing ones
                for key, value in standardized_properties.items():
                    if key not in node.properties:
                        node.properties[key] = value
                        # Validate TQDCS content if we just added it
                        if key == 'tqdcs_categories':
                            self._validate_tqdcs_content(node)
                    else:
                        logger.debug(f"Skipping existing property '{key}' for node {node_id}")
        
        return enhanced_kg
    
    def _get_default_validation_system_prompt(self) -> str:
        """Return default validation system prompt."""
        return """You are an expert at validating knowledge graphs against source documents.

Your task is to review an existing knowledge graph and the original text to identify:
1. Missing entities (nodes) that were mentioned but not captured
2. Missing relationships between entities
3. Additional properties that should be added to existing nodes

Guidelines:
- Only identify genuinely missing information that appears in the text
- Do not duplicate existing nodes or relationships
- Maintain the same ID format as existing nodes (Type:Identifier)
- For numerical values, use StructuredValue format: {{"value": number, "unit": "unit"}}
- Follow the TQDCS categorization rules from the original extraction

When creating new nodes:
- Use the same NodeType enum values as the existing graph
- Ensure IDs are unique and descriptive
- Include all relevant properties found in the text

When creating new relationships:
- Use consistent relationship labels (HAS_COST, SUPPLIES, etc.)
- Ensure both source and target nodes exist

{format_instructions}"""
    
    def _get_default_validation_human_prompt(self) -> str:
        """Return default validation human prompt."""
        return """Review this knowledge graph extraction against the original text:

ORIGINAL TEXT:
{original_text}

EXISTING KNOWLEDGE GRAPH:
Total Nodes: {node_count}
Total Relationships: {relationship_count}

Nodes by Type:
{existing_nodes}

Relationships by Type:
{existing_relationships}

Identify any missing information from the original text that should be in the knowledge graph.
Focus on entities, relationships, and properties that were clearly mentioned but not captured."""
    
    def _validate_tqdcs_field_names(self, kg: KnowledgeGraph) -> int:
        """
        Standardize TQDCS field names across all nodes.
        
        Args:
            kg: The knowledge graph to validate
            
        Returns:
            Number of corrections made
        """
        corrections_made = 0
        
        for node in kg.nodes:
            # Check for uppercase TQDCS field
            if 'TQDCS' in node.properties:
                # Get the value from the incorrect field
                tqdcs_value = node.properties.pop('TQDCS')
                
                # If tqdcs_categories already exists, merge the values
                if 'tqdcs_categories' in node.properties:
                    existing = node.properties['tqdcs_categories']
                    if isinstance(existing, list) and isinstance(tqdcs_value, list):
                        # Merge and deduplicate
                        merged = list(set(existing + tqdcs_value))
                        node.properties['tqdcs_categories'] = sorted(merged)
                    else:
                        # Replace with the TQDCS value
                        node.properties['tqdcs_categories'] = tqdcs_value
                else:
                    # Simply move the value to the correct field
                    node.properties['tqdcs_categories'] = tqdcs_value
                
                corrections_made += 1
                logger.info(f"Corrected TQDCS field name for node {node.id}")
            
            # Validate tqdcs_categories content if present
            if 'tqdcs_categories' in node.properties:
                self._validate_tqdcs_content(node)
            
            # Check if this node type should have TQDCS categories
            if not self._should_have_tqdcs_categories(node):
                # Remove TQDCS categories for nodes that shouldn't have them
                if 'tqdcs_categories' in node.properties:
                    node.properties['tqdcs_categories'] = []
                    logger.info(f"Cleared TQDCS categories for {node.type.value} node {node.id}")
        
        if corrections_made > 0:
            logger.info(f"Standardized {corrections_made} TQDCS field names")
        
        return corrections_made
    
    def _validate_tqdcs_content(self, node: Node) -> None:
        """
        Validate and fix TQDCS category content to ensure single letters only.
        
        Args:
            node: The node to validate
        """
        if 'tqdcs_categories' not in node.properties:
            return
        
        categories = node.properties['tqdcs_categories']
        
        # Ensure it's a list
        if not isinstance(categories, list):
            if isinstance(categories, str):
                categories = [categories]
            else:
                categories = []
        
        # Map full names to single letters
        name_to_letter = {
            'technology': 'T',
            'quality': 'Q',
            'delivery': 'D',
            'cost': 'C',
            'sustainability': 'S',
            't': 'T',
            'q': 'Q',
            'd': 'D',
            'c': 'C',
            's': 'S'
        }
        
        fixed_categories = []
        for cat in categories:
            if isinstance(cat, str):
                cat_lower = cat.lower().strip()
                if cat_lower in name_to_letter:
                    fixed_categories.append(name_to_letter[cat_lower])
                elif cat.upper() in ['T', 'Q', 'D', 'C', 'S']:
                    fixed_categories.append(cat.upper())
        
        # Remove duplicates and sort
        node.properties['tqdcs_categories'] = sorted(list(set(fixed_categories)))
    
    def _get_property_aliases(self) -> Dict[str, str]:
        """
        Return mapping of non-standard to standard property names.
        
        Returns:
            Dictionary mapping non-standard names to standard names
        """
        return {
            'TQDCS': 'tqdcs_categories',
            'tqdcs': 'tqdcs_categories',
            'TQDCS_categories': 'tqdcs_categories',
            'TQDCS_CATEGORIES': 'tqdcs_categories',
            # Add more property mappings as needed in the future
        }
    
    def _standardize_node_properties(self, node: Node) -> int:
        """
        Standardize all property names for a node.
        
        Args:
            node: The node to standardize
            
        Returns:
            Number of property names standardized
        """
        property_aliases = self._get_property_aliases()
        changes_made = 0
        
        # Create a list of properties to change (can't modify dict during iteration)
        properties_to_change = []
        for prop_name in node.properties:
            if prop_name in property_aliases:
                properties_to_change.append((prop_name, property_aliases[prop_name]))
        
        # Apply the changes
        for old_name, new_name in properties_to_change:
            value = node.properties.pop(old_name)
            
            # Handle case where the standard property already exists
            if new_name in node.properties:
                if new_name == 'tqdcs_categories':
                    # Special handling for TQDCS categories - merge lists
                    existing = node.properties[new_name]
                    if isinstance(existing, list) and isinstance(value, list):
                        merged = list(set(existing + value))
                        node.properties[new_name] = sorted(merged)
                    else:
                        node.properties[new_name] = value
                else:
                    # For other properties, last value wins
                    node.properties[new_name] = value
            else:
                node.properties[new_name] = value
            
            changes_made += 1
            logger.debug(f"Standardized property '{old_name}' to '{new_name}' for node {node.id}")
        
        return changes_made
    
    def _should_have_tqdcs_categories(self, node: Node) -> bool:
        """
        Determine if a node type should have TQDCS categories.
        
        Args:
            node: The node to check
            
        Returns:
            True if the node type should have TQDCS categories, False otherwise
        """
        # Node types that should NOT have TQDCS categories
        no_tqdcs_types = [
            NodeType.ORGANIZATION,
            NodeType.LOCATION,
            NodeType.DATE,
            NodeType.SUPPLIER,
        ]
        
        # Check if it's a type that shouldn't have TQDCS
        if node.type in no_tqdcs_types:
            return False
        
        # For GENERIC_INFORMATION nodes, check the content
        if node.type == NodeType.GENERIC_INFORMATION:
            # Check if it looks like organization/location/contact info
            node_text = json.dumps(node.properties).lower()
            org_indicators = ['address', 'contact', 'person', 'company', 'corporation', 
                            'gmbh', 'ag', 'ltd', 'inc', 'email', 'phone']
            if any(indicator in node_text for indicator in org_indicators):
                return False
        
        # All other node types can have TQDCS categories
        return True