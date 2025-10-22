"""
Flexible Knowledge Graph Extractor with TQDCS Framework

This enhanced version provides:
1. Dynamic pattern discovery using LLM
2. Configurable TQDCS-based extraction
3. Adaptive validation
4. Extensible domain patterns
"""

import json
import logging
import re
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

from .models.kg_schema import (
    KnowledgeGraph, Node, Relationship, SourceMetadata, 
    StructuredValue, NodeType
)
from .llm import create_llm
from .validation.kg_validator import KGValidator

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class TQDCSCategory:
    """Defines a TQDCS category with its patterns and indicators."""
    name: str
    description: str
    keywords: List[str]
    value_patterns: List[str]
    common_attributes: List[str]


class TQDCSFramework:
    """Framework for Technology, Quality, Delivery, Cost, Sustainability categorization."""
    
    def __init__(self):
        # Load TQDCS categories from JSON file
        prompts_dir = Path(__file__).parent / "llm" / "prompts"
        tqdcs_file = prompts_dir / "tqdcs_categories.json"
        
        if tqdcs_file.exists():
            try:
                with open(tqdcs_file, 'r', encoding='utf-8') as f:
                    tqdcs_data = json.load(f)
                
                self.categories = {}
                for key, data in tqdcs_data.items():
                    self.categories[key] = TQDCSCategory(
                        name=data["name"],
                        description=data["description"],
                        keywords=data["keywords"],
                        value_patterns=data["value_patterns"],
                        common_attributes=data["common_attributes"]
                    )
            except Exception as e:
                logger.warning(f"Failed to load TQDCS categories from file: {e}")
                self._load_default_categories()
        else:
            logger.warning(f"TQDCS categories file not found at {tqdcs_file}")
            self._load_default_categories()
    
    def _load_default_categories(self):
        """Load default categories if file loading fails."""
        self.categories = {
            "Technology": TQDCSCategory(
                name="Technology",
                description="Technical specifications, capabilities, features, innovation",
                keywords=["technical", "specification", "capability", "feature", "technology", 
                         "innovation", "R&D", "design", "engineering", "performance"],
                value_patterns=["power", "speed", "capacity", "efficiency", "dimensions"],
                common_attributes=["specifications", "features", "capabilities", "performance_metrics"]
            ),
            "Quality": TQDCSCategory(
                name="Quality",
                description="Quality standards, certifications, ratings, compliance",
                keywords=["quality", "certification", "ISO", "standard", "compliance", 
                         "warranty", "reliability", "defect", "audit", "inspection"],
                value_patterns=["ISO-\\d+", "rating", "grade", "compliance", "warranty period"],
                common_attributes=["certifications", "standards", "ratings", "warranty_terms"]
            ),
            "Delivery": TQDCSCategory(
                name="Delivery",
                description="Delivery capabilities, lead times, logistics, capacity",
                keywords=["delivery", "lead time", "capacity", "volume", "logistics", 
                         "shipment", "production", "availability", "schedule", "location"],
                value_patterns=["days", "weeks", "units", "quantity", "location"],
                common_attributes=["lead_time", "capacity", "location", "availability"]
            ),
            "Cost": TQDCSCategory(
                name="Cost",
                description="Pricing, costs, payment terms, financial aspects",
                keywords=["price", "cost", "EUR", "USD", "payment", "discount", 
                         "fee", "charge", "rate", "budget", "quotation"],
                value_patterns=["€", "$", "EUR", "USD", "price", "cost", "\\d+\\.\\d+"],
                common_attributes=["unit_price", "total_cost", "currency", "payment_terms"]
            ),
            "Sustainability": TQDCSCategory(
                name="Sustainability",
                description="Environmental impact, sustainability measures, green initiatives",
                keywords=["sustainability", "environmental", "carbon", "emission", "green", 
                         "renewable", "recycling", "eco", "footprint", "sustainable",
                         "climate", "neutral", "sdg", "global compact", "climate neutral"],
                value_patterns=["CO2", "carbon", "emission", "renewable", "%"],
                common_attributes=["carbon_footprint", "certifications", "renewable_percentage"]
            )
        }


class DocumentPattern:
    """Discovered pattern in a document."""
    def __init__(self, pattern_type: str, pattern: str, category: str, confidence: float):
        self.pattern_type = pattern_type  # 'header', 'value', 'structure'
        self.pattern = pattern
        self.category = category  # TQDCS category
        self.confidence = confidence


class KGExtractor:
    """
    Knowledge Graph extractor that adapts to different document types.
    
    Key improvements:
    1. Dynamic pattern discovery
    2. TQDCS-based categorization
    3. Adaptive validation
    4. Configurable extraction rules
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4.1",
                 temperature: float = 0.0,
                 enable_validation: bool = True,
                 custom_patterns: Optional[Dict[str, Any]] = None):
        """Initialize the extractor."""
        self.llm_model = llm_model
        self.temperature = temperature
        self.enable_validation = enable_validation
        
        # Initialize LLM using factory
        self.llm = create_llm(
            model_name=llm_model,
            temperature=temperature
        )
        
        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
        # Initialize TQDCS framework
        self.tqdcs = TQDCSFramework()
        
        # Merge custom patterns if provided
        self.domain_patterns = self._initialize_base_patterns()
        if custom_patterns:
            self._merge_custom_patterns(custom_patterns)
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
        
        # Initialize validator if enabled
        if self.enable_validation:
            self.validator = KGValidator(
                llm_model=llm_model,
                temperature=temperature
            )
    
    def _initialize_base_patterns(self) -> Dict[str, Any]:
        """Initialize base patterns that work across documents."""
        return {
            'table_indicators': [
                r'\s*\|\s*',  # Pipe-separated tables
                r'\t+',       # Tab-separated
                r'\s{2,}',    # Multiple spaces
                r'[-=]{3,}',  # Table borders
            ],
            'header_patterns': [
                r'^[A-Z][A-Za-z\s]+:',  # "Section Name:"
                r'^\d+\.\s*[A-Z]',      # "1. Section"
                r'^#{1,3}\s*',          # Markdown headers
            ],
            'value_patterns': {
                'currency': [r'[\$€£¥]\s*[\d,]+\.?\d*', r'[\d,]+\.?\d*\s*(?:EUR|USD|GBP|JPY)'],
                'percentage': [r'[\d,]+\.?\d*\s*%', r'[\d,]+\.?\d*\s*percent'],
                'quantity': [r'[\d,]+\s*(?:units?|pieces?|items?|kg|tons?|liters?)'],
                'date': [r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'],
                'identifier': [r'[A-Z]{2,}-?\d{3,}', r'\d{6,}', r'[A-Z0-9]{8,}'],
            },
            'tqdcs_patterns': {}  # Will be populated dynamically
        }
    
    def _merge_custom_patterns(self, custom_patterns: Dict[str, Any]):
        """Merge custom patterns with base patterns."""
        for key, value in custom_patterns.items():
            if key in self.domain_patterns and isinstance(value, list):
                self.domain_patterns[key].extend(value)
            elif key in self.domain_patterns and isinstance(value, dict):
                self.domain_patterns[key].update(value)
            else:
                self.domain_patterns[key] = value
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        prompts_dir = Path(__file__).parent / "llm" / "prompts"
        prompts = {}
        
        # Define prompt files to load
        prompt_files = {
            'discovery': 'discovery_prompt.txt',
            'system': 'system_prompt.txt',
            'human': 'human_prompt.txt'
        }
        
        for key, filename in prompt_files.items():
            file_path = prompts_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompts[key] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load prompt template {filename}: {e}")
                    prompts[key] = None
            else:
                logger.warning(f"Prompt template file not found: {file_path}")
                prompts[key] = None
        
        # Load default prompts if file loading failed
        if not prompts.get('discovery'):
            prompts['discovery'] = self._get_default_discovery_prompt()
        if not prompts.get('system'):
            prompts['system'] = self._get_default_system_prompt()
        if not prompts.get('human'):
            prompts['human'] = self._get_default_human_prompt()
        
        return prompts
    
    def extract_knowledge_graph(self, 
                               chunk_text: str,
                               chunk_metadata: SourceMetadata) -> KnowledgeGraph:
        """Extract a knowledge graph with adaptive pattern discovery."""
        logger.info(f"Extracting KG from chunk: {chunk_metadata.chunk_id}")
        
        # Step 1: Discover patterns dynamically
        discovered_patterns = self._discover_document_patterns(chunk_text)
        
        # Step 2: Analyze structure with discovered patterns
        doc_structure = self._analyze_document_structure(chunk_text, discovered_patterns)
        
        # Step 3: Create TQDCS-aware prompt
        prompt = self._create_tqdcs_aware_prompt(doc_structure, discovered_patterns)
        
        # Step 4: Extract knowledge graph
        try:
            chain = prompt | self.llm | self.parser
            
            # Build discovered patterns string for the prompt
            pattern_examples = defaultdict(list)
            for p in discovered_patterns:
                if p.category and p.pattern:
                    pattern_examples[p.category].append(p.pattern)
            
            discovered_patterns_str = "\n".join([f"- {cat}: {', '.join(ex.replace('{', '{{').replace('}', '}}') for ex in examples[:3])}" 
                                               for cat, examples in pattern_examples.items() if examples])
            
            kg = chain.invoke({
                "chunk_text": chunk_text,
                "chunk_id": chunk_metadata.chunk_id,
                "filename": chunk_metadata.filename,
                "format_instructions": self.parser.get_format_instructions(),
                "doc_structure": self._escape_template_vars(json.dumps(doc_structure, indent=2)),
                "document_type": doc_structure.get('document_type', 'unknown'),
                "has_tables": doc_structure.get('has_tables', False),
                "discovered_patterns": discovered_patterns_str,
                "tqdcs_categories": self._escape_template_vars(json.dumps({
                    cat: {
                        "description": info.description,
                        "keywords": info.keywords
                    } for cat, info in self.tqdcs.categories.items()
                }, indent=2))
            })
            
            # Post-process and validate
            kg = self._post_process_kg(kg, chunk_metadata)
            
            if self.enable_validation:
                # First do adaptive validation and fixing
                kg = self._adaptive_validate_and_fix_kg(kg, chunk_text, discovered_patterns)
                
                # Then do comprehensive validation against original text
                validation_result = self.validator.validate_knowledge_graph(
                    kg=kg,
                    original_text=chunk_text,
                    metadata=chunk_metadata
                )
                
                if validation_result.validation_successful:
                    kg = validation_result.validated_kg
                    if validation_result.nodes_added > 0 or validation_result.relationships_added > 0:
                        logger.info(f"Validation enhanced KG: +{validation_result.nodes_added} nodes, +{validation_result.relationships_added} relationships")
                else:
                    logger.warning(f"Validation failed: {validation_result.notes}")
            
            return kg
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_metadata.chunk_id}: {str(e)}")
            raise
    
    def _post_process_kg(self, kg: KnowledgeGraph, metadata: SourceMetadata) -> KnowledgeGraph:
        """Post-process the KG to ensure consistency and add metadata."""
        # Add source metadata to all nodes and relationships
        for node in kg.nodes:
            if not node.metadata:
                node.metadata = [metadata]
            elif not any(m.chunk_id == metadata.chunk_id for m in node.metadata):
                node.metadata.append(metadata)
        
        for rel in kg.relationships:
            if not rel.metadata:
                rel.metadata = metadata
        
        # Track ID changes for updating relationships
        id_mapping = {}
        
        # Ensure proper ID formats
        for node in kg.nodes:
            original_id = node.id
            
            if ':' not in node.id:
                node.id = f"{node.type.value}:{node.id}"
            
            # Clean up IDs (remove spaces, special characters)
            cleaned_id = re.sub(r'[^\w:\-]', '-', node.id)
            
            # Track the mapping if ID changed
            if cleaned_id != original_id:
                id_mapping[original_id] = cleaned_id
                # Also track the intermediate ID if we added type prefix
                if ':' not in original_id and node.id != original_id:
                    id_mapping[node.id] = cleaned_id
            
            node.id = cleaned_id
        
        # Update relationship references using the ID mapping
        for rel in kg.relationships:
            # Update source if it was changed
            if rel.source in id_mapping:
                rel.source = id_mapping[rel.source]
            
            # Update target if it was changed
            if rel.target in id_mapping:
                rel.target = id_mapping[rel.target]
        
        # Validate relationships after fixing references
        node_ids = {node.id for node in kg.nodes}
        for rel in kg.relationships:
            # Ensure relationship endpoints exist
            if rel.source not in node_ids or rel.target not in node_ids:
                logger.warning(f"Relationship references non-existent node: {rel.source} -> {rel.target}")
        
        return kg
    
    def _discover_document_patterns(self, text: str) -> List[DocumentPattern]:
        """Use LLM to discover patterns in the document."""
        # Use loaded prompt template
        discovery_prompt = self.prompts['discovery'].format(text=text)

        try:
            response = self.llm.invoke(discovery_prompt)
            # Parse the JSON response
            discovered = json.loads(response.content)
            
            patterns = []
            for p in discovered.get("patterns", []):
                pattern_str = p.get("pattern", "")
                # Validate the pattern before using it
                if pattern_str:
                    try:
                        # Test compile the regex
                        re.compile(pattern_str)
                        patterns.append(DocumentPattern(
                            pattern_type=p.get("type", "value"),
                            pattern=pattern_str,
                            category=p.get("category", ""),
                            confidence=0.8  # Default confidence
                        ))
                    except re.error as e:
                        logger.warning(f"Invalid regex pattern from LLM '{pattern_str}': {e}")
                        continue
            
            # Store document type for context
            self._current_doc_type = discovered.get("document_type", "unknown")
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern discovery failed, using defaults: {e}")
            return []
    
    def _analyze_document_structure(self, text: str, patterns: List[DocumentPattern]) -> Dict[str, Any]:
        """Analyze document structure using discovered patterns."""
        structure = {
            'document_type': getattr(self, '_current_doc_type', 'unknown'),
            'has_tables': False,
            'table_sections': [],
            'tqdcs_sections': defaultdict(list),
            'identified_values': defaultdict(list),
            'key_patterns': defaultdict(list)
        }
        
        lines = text.split('\n')
        
        # Check for tables
        for i, line in enumerate(lines):
            # Use base table indicators
            for indicator in self.domain_patterns['table_indicators']:
                try:
                    if re.search(indicator, line):
                        structure['has_tables'] = True
                        structure['table_sections'].append({
                            'line': i,
                            'content': line[:100]  # First 100 chars
                        })
                        break
                except re.error as e:
                    logger.warning(f"Invalid table indicator pattern '{indicator}': {e}")
                    continue
        
        # Apply discovered patterns
        for pattern in patterns:
            if pattern.pattern:
                try:
                    matches = re.findall(pattern.pattern, text, re.IGNORECASE)
                    if matches:
                        structure['tqdcs_sections'][pattern.category].extend(matches[:5])
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern.pattern}': {e}")
                    continue
        
        # Extract values by type
        for value_type, patterns in self.domain_patterns['value_patterns'].items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text)
                    if matches:
                        structure['identified_values'][value_type].extend(matches[:10])
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}' for {value_type}: {e}")
                    continue
        
        return structure
    
    def _escape_template_vars(self, text: str) -> str:
        """Escape curly braces in text to prevent template variable interpretation."""
        return text.replace('{', '{{').replace('}', '}}')
    
    def _create_tqdcs_aware_prompt(self, 
                                   doc_structure: Dict[str, Any], 
                                   patterns: List[DocumentPattern]) -> ChatPromptTemplate:
        """Create a prompt that uses TQDCS framework."""
        
        # Use loaded prompt templates
        system_message = self.prompts['system']
        human_message = self.prompts['human']

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def _adaptive_validate_and_fix_kg(self, 
                                     kg: KnowledgeGraph, 
                                     original_text: str,
                                     patterns: List[DocumentPattern]) -> KnowledgeGraph:
        """Adaptive validation that doesn't assume specific node structures."""
        issues_found = []
        
        # 1. Check node consistency (not assuming specific properties)
        for node in kg.nodes:
            # Validate based on node type
            if node.type == NodeType.COST:
                # Don't assume 'part' exists - check what references exist
                references = self._find_node_references(node)
                if not references and not self._node_has_identifying_info(node):
                    issues_found.append(f"Cost node {node.id} lacks identifying information")
                    # Try to extract identifying info from ID or properties
                    self._add_identifying_info(node, original_text)
            
            # Fix TQDCS categories
            if 'tqdcs_categories' in node.properties:
                # Ensure single letters only and filter by node type
                node.properties['tqdcs_categories'] = self._fix_tqdcs_categories(node)
            else:
                # Only add TQDCS for applicable node types
                if self._should_have_tqdcs(node):
                    categories = self._infer_tqdcs_categories(node, patterns)
                    if categories:
                        node.properties['tqdcs_categories'] = categories
                else:
                    # Explicitly set empty list for non-applicable types
                    node.properties['tqdcs_categories'] = []
        
        # 2. Validate and fix numerical values
        for node in kg.nodes:
            for key, value in list(node.properties.items()):
                if isinstance(value, (int, float)):
                    # Infer unit based on property name and context
                    unit = self._context_aware_unit_inference(key, value, node, original_text)
                    node.properties[key] = {"value": value, "unit": unit}
        
        # 3. Ensure relationships are valid
        node_ids = {node.id for node in kg.nodes}
        valid_relationships = []
        
        for rel in kg.relationships:
            if rel.source in node_ids and rel.target in node_ids:
                valid_relationships.append(rel)
            else:
                issues_found.append(f"Invalid relationship: {rel.source} -> {rel.target}")
        
        kg.relationships = valid_relationships
        
        # 4. Deduplicate nodes intelligently
        kg.nodes = self._intelligent_deduplication(kg.nodes)
        
        if issues_found:
            logger.info(f"Fixed {len(issues_found)} validation issues")
            for issue in issues_found[:5]:
                logger.debug(f"  - {issue}")
        
        return kg
    
    def _find_node_references(self, node: Node) -> List[str]:
        """Find what this node references (flexible approach)."""
        references = []
        
        # Check common reference patterns
        ref_keys = ['part', 'product', 'item', 'component', 'service', 
                   'part_number', 'reference', 'refers_to', 'for']
        
        for key in ref_keys:
            if key in node.properties:
                references.append(str(node.properties[key]))
        
        # Check ID for references
        id_parts = node.id.split(':')
        if len(id_parts) > 1:
            # ID might contain reference info
            references.extend(id_parts[1].split('-'))
        
        return references
    
    def _node_has_identifying_info(self, node: Node) -> bool:
        """Check if node has sufficient identifying information."""
        identifying_keys = ['part_number', 'reference', 'id', 'code', 'sku', 
                          'model', 'type', 'name', 'description']
        
        for key in identifying_keys:
            if key in node.properties and node.properties[key]:
                return True
        
        return False
    
    def _add_identifying_info(self, node: Node, context: str):
        """Try to add identifying information to a node."""
        # Extract from node ID if possible
        if ':' in node.id:
            parts = node.id.split(':')[1].split('-')
            if parts:
                node.properties['extracted_reference'] = '-'.join(parts)
        
        # TODO: Could use context parameter to extract more info if needed
    
    def _should_have_tqdcs(self, node: Node) -> bool:
        """Determine if a node type should have TQDCS categories."""
        # Node types that should have TQDCS categories
        tqdcs_applicable = [
            NodeType.PART, NodeType.COST, NodeType.SUPPLIER,
            NodeType.CERTIFICATION, NodeType.DRAWING
        ]
        
        # Node types that should NOT have TQDCS categories
        tqdcs_not_applicable = [
            NodeType.ORGANIZATION, NodeType.LOCATION, NodeType.DATE
        ]
        
        # Check if it's explicitly not applicable
        if node.type in tqdcs_not_applicable:
            return False
        
        # Check if it's explicitly applicable
        if node.type in tqdcs_applicable:
            return True
        
        # Default: check GENERIC_INFORMATION content
        if node.type == NodeType.GENERIC_INFORMATION:
            # Check node properties to determine if it should have TQDCS
            node_text = json.dumps(node.properties).lower()
            # If it looks like organization/location info, no TQDCS
            if any(term in node_text for term in ['address', 'contact', 'person', 'company', 'corporation', 'gmbh', 'ag', 'ltd']):
                return False
            # Otherwise, it likely represents technical/cost/quality info
            return True
        
        return False
    
    def _fix_tqdcs_categories(self, node: Node) -> List[str]:
        """Fix TQDCS categories to ensure single letters only."""
        if not self._should_have_tqdcs(node):
            return []
        
        current_categories = node.properties.get('tqdcs_categories', [])
        fixed_categories = []
        
        # Map of full names to single letters
        category_map = {
            'technology': 'T', 'tech': 'T', 't': 'T',
            'quality': 'Q', 'qual': 'Q', 'q': 'Q',
            'delivery': 'D', 'del': 'D', 'd': 'D',
            'cost': 'C', 'price': 'C', 'c': 'C',
            'sustainability': 'S', 'sustain': 'S', 's': 'S'
        }
        
        for cat in current_categories:
            if isinstance(cat, str):
                cat_lower = cat.lower().strip()
                # Check if it's already a single letter
                if cat_lower in ['t', 'q', 'd', 'c', 's']:
                    fixed_categories.append(cat_lower.upper())
                # Try to map full name to letter
                elif cat_lower in category_map:
                    fixed_categories.append(category_map[cat_lower])
        
        # Remove duplicates and sort
        return sorted(list(set(fixed_categories)))
    
    def _infer_tqdcs_categories(self, node: Node, patterns: List[DocumentPattern]) -> List[str]:
        """Infer TQDCS categories for a node based on its content."""
        if not self._should_have_tqdcs(node):
            return []
        
        categories = []
        
        # Check node properties against TQDCS keywords
        node_text = json.dumps(node.properties).lower()
        
        for cat_name, category in self.tqdcs.categories.items():
            score = 0
            
            # Check keywords
            for keyword in category.keywords:
                if keyword.lower() in node_text:
                    score += 1
            
            # Check if any discovered patterns match
            for pattern in patterns:
                if pattern.category == cat_name[0]:  # First letter
                    if pattern.pattern.lower() in node_text:
                        score += 2
            
            if score > 0:
                categories.append(cat_name[0])  # T, Q, D, C, or S
        
        return categories
    
    def _context_aware_unit_inference(self, 
                                     property_name: str, 
                                     value: Any,
                                     node: Node,
                                     context: str) -> str:
        """Infer unit based on property, node type, and context."""
        
        # Check if property name contains unit hints
        property_lower = property_name.lower()
        
        # Currency detection
        if any(term in property_lower for term in ['price', 'cost', 'fee', 'charge']):
            # Look for currency in context
            if '€' in context or 'EUR' in context:
                return 'EUR'
            elif '$' in context or 'USD' in context:
                return 'USD'
            elif '£' in context or 'GBP' in context:
                return 'GBP'
            # Default based on document origin if known
            return 'EUR'  # Default
        
        # Percentage detection
        if any(term in property_lower for term in ['rate', 'percentage', 'percent', 'ratio']):
            return 'percent'
        
        # Time detection
        if any(term in property_lower for term in ['time', 'duration', 'period', 'days', 'weeks']):
            if 'day' in property_lower:
                return 'days'
            elif 'week' in property_lower:
                return 'weeks'
            elif 'month' in property_lower:
                return 'months'
            elif 'year' in property_lower:
                return 'years'
            return 'days'  # Default time unit
        
        # Quantity detection
        if any(term in property_lower for term in ['quantity', 'volume', 'amount', 'capacity']):
            if any(term in property_lower for term in ['kg', 'kilogram']):
                return 'kg'
            elif any(term in property_lower for term in ['ton', 'tonne']):
                return 'tonnes'
            elif any(term in property_lower for term in ['liter', 'litre']):
                return 'liters'
            return 'units'  # Default quantity
        
        # Check node's TQDCS categories for hints
        if 'tqdcs_categories' in node.properties:
            categories = node.properties['tqdcs_categories']
            if 'C' in categories:  # Cost category
                return 'EUR'  # Assume currency
            elif 'D' in categories and isinstance(value, (int, float)) and value > 100:
                return 'units'  # Likely quantity
        
        return ""  # No unit if unclear
    
    def _intelligent_deduplication(self, nodes: List[Node]) -> List[Node]:
        """Deduplicate nodes intelligently, merging properties."""
        node_groups = defaultdict(list)
        
        # Group nodes by ID
        for node in nodes:
            node_groups[node.id].append(node)
        
        # Merge nodes with same ID
        unique_nodes = []
        for node_id, group in node_groups.items():
            if len(group) == 1:
                unique_nodes.append(group[0])
            else:
                # Merge properties from all nodes in group
                merged = group[0]
                for node in group[1:]:
                    # Merge properties
                    for key, value in node.properties.items():
                        if key not in merged.properties:
                            merged.properties[key] = value
                        elif isinstance(merged.properties[key], list):
                            if value not in merged.properties[key]:
                                merged.properties[key].append(value)
                        # For conflicts, keep first value
                    
                    # Merge metadata with deduplication
                    if node.metadata:
                        # Track existing metadata by (filename, chunk_id) tuple
                        existing_metadata_keys = {
                            (m.filename, m.chunk_id) for m in merged.metadata
                        }
                        
                        # Add only new metadata
                        for new_meta in node.metadata:
                            meta_key = (new_meta.filename, new_meta.chunk_id)
                            if meta_key not in existing_metadata_keys:
                                merged.metadata.append(new_meta)
                                existing_metadata_keys.add(meta_key)
                
                unique_nodes.append(merged)
        
        return unique_nodes
    
    def _get_default_discovery_prompt(self) -> str:
        """Return default discovery prompt if file loading fails."""
        return """Analyze this document excerpt and identify patterns for TQDCS categories.

TQDCS Categories:
- Technology (T): Technical specs, capabilities, features
- Quality (Q): Standards, certifications, quality measures  
- Delivery (D): Lead times, capacity, logistics
- Cost (C): Prices, payment terms, financial data
- Sustainability (S): Environmental measures, green initiatives

Document excerpt:
{text}  

Identify:
1. What type of document is this? (quotation, specification, contract, etc.)
2. Key patterns for each TQDCS category found
3. Table structures if any
4. Identifier formats (part numbers, reference codes)

Return as JSON with structure:
{{
    "document_type": "type",
    "patterns": [
        {{"pattern": "pattern_text", "category": "T/Q/D/C/S", "type": "header/value/structure", "example": "example_from_text"}}
    ],
    "table_structure": {{"has_tables": true/false, "format": "description"}},
    "identifiers": ["examples of IDs found"]
}}"""
    
    def _get_default_system_prompt(self) -> str:
        """Return default system prompt if file loading fails."""
        return """You are an expert at extracting structured knowledge from business documents using the TQDCS framework.

TQDCS Categories for classification:
{tqdcs_categories}

Extract information into a KnowledgeGraph following these rules:

1. NODE CREATION:
   - Create specific nodes for each entity (Part, Supplier, Cost, etc.)
   - Use descriptive IDs that include key identifiers
   - Each distinct item/price/specification gets its own node

2. TQDCS CLASSIFICATION - CRITICAL RULES:
   - ONLY use single letters: T, Q, D, C, S (NOT full words like "Technology" or "Delivery")
   - Tag applicable nodes with relevant TQDCS categories
   - Use "tqdcs_categories" property as a list of single letters
   - Nodes that should NOT have TQDCS categories (use empty list []):
     * Organization nodes (companies, departments)
     * Location nodes (addresses, facilities)
     * Contact/Person nodes
   - Nodes that SHOULD have TQDCS categories:
     * Part, Cost, Quality, Technical_Specification, Process, Material nodes
   - Examples:
     * Part node: "tqdcs_categories": ["T", "Q"]
     * Cost node: "tqdcs_categories": ["C"]
     * Organization node: "tqdcs_categories": []

3. STRUCTURED VALUES:
   - ALL numerical values must use StructuredValue format: {{"value": 123.45, "unit": "EUR"}}
   - Infer units from context when not explicit

4. RELATIONSHIPS:
   - Create meaningful relationships between nodes
   - Common relationships: HAS_COST, HAS_SPECIFICATION, MEETS_STANDARD, SUPPLIES, LOCATED_AT

5. PROPERTY NAMING:
   - Use consistent, descriptive property names
   - Include source identifiers (part numbers, reference codes)

Document type: {document_type}
Document has tables: {has_tables}

Discovered patterns by category:
{discovered_patterns}

{format_instructions}"""
    
    def _get_default_human_prompt(self) -> str:
        """Return default human prompt if file loading fails."""
        return """Extract a knowledge graph from this document text:

Chunk ID: {chunk_id}
Filename: {filename}

Document structure analysis:
{doc_structure}

--- TEXT ---
{chunk_text}
--- END TEXT ---

Create nodes and relationships following the TQDCS framework. Ensure proper classification."""


# Configuration builder for specific document types
class DocumentTypeConfig:
    """Build configuration for specific document types."""
    
    @staticmethod
    def quotation_config() -> Dict[str, Any]:
        """Configuration optimized for quotations/RFQs."""
        return {
            'value_patterns': {
                'price': [r'[\d,]+\.?\d*\s*€', r'€\s*[\d,]+\.?\d*'],
                'volume': [r'[\d,]+\s*(?:pieces?|units?|pcs)'],
                'discount': [r'[\d,]+\.?\d*\s*%\s*(?:discount|reduction)'],
            },
            'tqdcs_emphasis': {
                'C': 1.5,  # Emphasize cost extraction
                'D': 1.2,  # Emphasize delivery
            }
        }
    
    @staticmethod
    def specification_config() -> Dict[str, Any]:
        """Configuration optimized for technical specifications."""
        return {
            'value_patterns': {
                'measurement': [r'[\d,]+\.?\d*\s*(?:mm|cm|m|kg|g|l)'],
                'tolerance': [r'[±+-]\s*[\d,]+\.?\d*\s*%?'],
                'standard': [r'(?:ISO|DIN|EN)\s*[\d\-]+'],
            },
            'tqdcs_emphasis': {
                'T': 1.5,  # Emphasize technology
                'Q': 1.3,  # Emphasize quality
            }
        }
    
    @staticmethod
    def sustainability_report_config() -> Dict[str, Any]:
        """Configuration optimized for sustainability documents."""
        return {
            'value_patterns': {
                'emission': [r'[\d,]+\.?\d*\s*(?:kg|t|ton)\s*CO2'],
                'renewable': [r'[\d,]+\.?\d*\s*%\s*renewable'],
                'certification': [r'(?:ISO\s*14001|LEED|BREEAM)'],
            },
            'tqdcs_emphasis': {
                'S': 2.0,  # Strongly emphasize sustainability
            }
        }


# Updated pipeline creation function
def create_knowledge_graph(
    file_path: str,
    output_dir: str = "./kg_output",
    llm_model: str = "gpt-4.1",
    document_type: Optional[str] = None,
    custom_patterns: Optional[Dict[str, Any]] = None,
    parallel_processing: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a knowledge graph using the extraction pipeline.
    
    Args:
        file_path: Path to the input document
        output_dir: Directory for output files
        llm_model: LLM model to use
        document_type: Type of document ('quotation', 'specification', 'sustainability')
        custom_patterns: Custom patterns to use
        parallel_processing: Enable parallel chunk processing
        max_workers: Max parallel workers (None = auto)
        
    Returns:
        Dictionary with extraction results
    """
    # Get configuration based on document type
    config = {}
    if document_type == 'quotation':
        config = DocumentTypeConfig.quotation_config()
    elif document_type == 'specification':
        config = DocumentTypeConfig.specification_config()
    elif document_type == 'sustainability':
        config = DocumentTypeConfig.sustainability_report_config()
    
    # Merge with custom patterns
    if custom_patterns:
        config.update(custom_patterns)
    
    # Create extractor with configuration
    extractor = KGExtractor(
        llm_model=llm_model,
        enable_validation=True,
        custom_patterns=config
    )
    
    # Use the pipeline structure
    from .kg_pipeline import KGPipeline
    
    pipeline = KGPipeline(
        output_dir=output_dir,
        llm_model=llm_model,
        enable_validation=True,
        parallel_processing=parallel_processing,
        max_workers=max_workers
    )
    
    # Replace the extractor with our custom configured one
    pipeline.extractor = extractor
    
    return pipeline.process_document(file_path)
