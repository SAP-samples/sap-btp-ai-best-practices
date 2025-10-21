"""
Metric Extraction with Static Subcategory Classification

This module provides reusable functionality for extracting metrics from 
TQSDC-categorized nodes and classifying them into predefined static 
subcategories for consistent supplier comparison.
"""

import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field

# Add parent directories for imports
sys.path.append('../')
sys.path.append('../../')

# Import existing schemas
from models.kg_schema import (
    NodeType, 
    SourceMetadata as BaseSourceMetadata, 
    Node as KGNodeSchema,
    Relationship as KGRelationshipSchema,
    KnowledgeGraph as BaseKnowledgeGraph
)

# Import static classification modules
from static_subcategories import (
    TQSDC_SUBCATEGORIES,
    get_subcategories_for_category,
    get_subcategory_names_for_category
)
from llm_classifier import SubcategoryClassifier, ClassificationResult

# Pydantic models
from pydantic import BaseModel, Field


# Extended data models
class SourceMetadata(BaseSourceMetadata):
    """Extended metadata for metrics"""
    extraction_timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class EnhancedMetric(BaseModel):
    """Enhanced metric with additional context for clustering"""
    metric_name: str
    category: str  # TQDCS category (T, Q, D, C, S)
    source_node_id: str
    node_type: NodeType  # Using the enum from kg_schema
    value: Any
    unit: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[str] = Field(default_factory=list)
    description_text: str
    metadata: List[SourceMetadata] = Field(default_factory=list)
    
    def get_clustering_text(self) -> str:
        """Generate text representation for clustering"""
        parts = [self.metric_name]
        
        # Add node type context
        parts.append(f"Type: {self.node_type.value}")
        
        # Add value and unit if present
        if self.value is not None:
            value_str = f"Value: {self.value}"
            if self.unit:
                value_str += f" {self.unit}"
            parts.append(value_str)
        
        # Add description
        if self.description_text:
            parts.append(self.description_text)
        
        # Add key properties
        important_props = ['cost_type', 'description', 'standard', 'certification_type']
        for prop in important_props:
            if prop in self.properties:
                parts.append(f"{prop}: {self.properties[prop]}")
        
        return " | ".join(parts)


class MetricCluster(BaseModel):
    """A cluster of related metrics within a TQDCS category"""
    cluster_id: str
    name: str  # Subcategory name
    category: str  # Parent TQDCS category
    metrics: List[EnhancedMetric]
    size: int = 0
    representative_metric: Optional[EnhancedMetric] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        self.size = len(self.metrics)
        if self.metrics and not self.representative_metric:
            self.representative_metric = self.metrics[0]
        
    def get_summary(self) -> Dict[str, Any]:
        """Get cluster summary statistics"""
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "category": self.category,
            "size": self.size,
            "metric_types": list(set(m.node_type.value for m in self.metrics)),
            "sample_metrics": [m.metric_name for m in self.metrics[:3]]
        }


# Extended KnowledgeGraph
class KnowledgeGraph(BaseKnowledgeGraph):
    """Extended Knowledge Graph with clustering-specific methods"""
    
    def get_node_by_id(self, node_id: str) -> Optional[KGNodeSchema]:
        """Find a node by its ID."""
        return super().get_node_by_id(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[KGNodeSchema]:
        """Get all nodes of a specific type."""
        return super().get_nodes_by_type(node_type)
    
    def get_related_nodes(self, node_id: str) -> List[Tuple[KGNodeSchema, KGRelationshipSchema]]:
        """Get all nodes related to a given node"""
        related = []
        for rel in self.relationships:
            if rel.source == node_id:
                target_node = self.get_node_by_id(rel.target)
                if target_node:
                    related.append((target_node, rel))
            elif rel.target == node_id:
                source_node = self.get_node_by_id(rel.source)
                if source_node:
                    related.append((source_node, rel))
        return related
    
    @property
    def nodes_with_tqdcs(self) -> List[KGNodeSchema]:
        """Get all nodes that have TQDCS categories"""
        return [n for n in self.nodes if n.properties.get('tqdcs_categories', [])]


class MetricExtractor:
    """Extracts metrics from TQDCS-categorized nodes with clustering support."""
    
    def __init__(self):
        self.metrics_count = 0
        
    def extract_metrics_with_context(self, kg: KnowledgeGraph) -> Dict[str, List[EnhancedMetric]]:
        """
        Extract metrics with rich context for clustering.
        
        Returns:
            Dict mapping TQDCS categories to enhanced metrics with context
        """
        metrics = {
            "T": [],  # Technology
            "Q": [],  # Quality
            "D": [],  # Delivery
            "C": [],  # Cost
            "S": [],  # Sustainability
        }
        
        # Extract metrics from categorized nodes with enhanced context
        for node in kg.nodes:
            tqdcs_categories = node.properties.get('tqdcs_categories', [])
            
            if not tqdcs_categories:
                continue
                
            for category in tqdcs_categories:
                if category in metrics:
                    # Extract with additional context for clustering
                    extracted_metrics = self._extract_enhanced_metrics(node, category, kg)
                    metrics[category].extend(extracted_metrics)
        
        # Print extraction summary
        print("\nMetric Extraction Summary:")
        for category, category_metrics in metrics.items():
            print(f"  {category}: {len(category_metrics)} metrics")
        
        return metrics
    
    def _extract_enhanced_metrics(self, node: KGNodeSchema, category: str, kg: KnowledgeGraph) -> List[EnhancedMetric]:
        """Extract enhanced metrics from a node based on its type and properties."""
        metrics = []
        
        # Get related nodes for context
        related_nodes = kg.get_related_nodes(node.id)
        relationship_context = [f"{rel.label}:{rnode.id}" for rnode, rel in related_nodes]
        
        # Extract based on node type
        if node.type == NodeType.COST:
            metrics.extend(self._extract_cost_metrics(node, category, relationship_context))
        elif node.type == NodeType.PART:
            metrics.extend(self._extract_part_metrics(node, category, relationship_context))
        elif node.type == NodeType.CERTIFICATION:
            metrics.extend(self._extract_certification_metrics(node, category, relationship_context))
        elif node.type == NodeType.GENERIC_INFORMATION:
            metrics.extend(self._extract_generic_metrics(node, category, relationship_context))
        elif node.type == NodeType.LOCATION:
            metrics.extend(self._extract_location_metrics(node, category, relationship_context))
        else:
            # Generic extraction for other node types
            metrics.extend(self._extract_generic_metrics(node, category, relationship_context))
        
        return metrics
    
    def _extract_cost_metrics(self, node: KGNodeSchema, category: str, relationships: List[str]) -> List[EnhancedMetric]:
        """Extract metrics from Cost nodes."""
        metrics = []
        
        # Try different attribute names for cost value
        value = None
        unit = 'EUR'  # Default unit
        
        # Check for 'amount' attribute
        if 'amount' in node.properties:
            amount_info = node.properties['amount']
            if isinstance(amount_info, dict):
                value = amount_info.get('value', 0)
                unit = amount_info.get('unit', 'EUR')
            else:
                value = amount_info
        
        # Check for 'value' attribute if amount not found
        elif 'value' in node.properties:
            value_info = node.properties['value']
            if isinstance(value_info, dict):
                value = value_info.get('value', 0)
                unit = value_info.get('unit', 'EUR')
            else:
                value = value_info
        
        # Check for 'price' attribute if neither amount nor value found
        elif 'price' in node.properties:
            price_info = node.properties['price']
            if isinstance(price_info, dict):
                value = price_info.get('value', 0)
                unit = price_info.get('unit', 'EUR')
            else:
                value = price_info
        
        # If still no value found, try to find any numeric property
        if value is None:
            for key, val in node.properties.items():
                if key not in ['tqdcs_categories', 'cost_type', 'description', 'name']:
                    if isinstance(val, (int, float)):
                        value = val
                        break
                    elif isinstance(val, dict) and 'value' in val:
                        value = val['value']
                        unit = val.get('unit', 'EUR')
                        break
        
        # Default to 0 if no value found
        if value is None:
            value = 0
        
        cost_type = node.properties.get('cost_type', 'General Cost')
        description = node.properties.get('description', '')
        
        # Create metric name
        metric_name = f"{cost_type} - {value} {unit}"
        
        # Build description text for clustering
        description_parts = [cost_type]
        if description:
            description_parts.append(description)
        description_parts.extend(relationships)
        
        metric = EnhancedMetric(
            metric_name=metric_name,
            category=category,
            source_node_id=node.id,
            node_type=node.type,  # This is already a NodeType enum
            value=value,
            unit=unit,
            properties=node.properties,
            relationships=relationships,
            description_text=" | ".join(description_parts),
            metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
        )
        
        metrics.append(metric)
        self.metrics_count += 1
        
        return metrics
    
    def _extract_part_metrics(self, node: KGNodeSchema, category: str, relationships: List[str]) -> List[EnhancedMetric]:
        """Extract metrics from Part nodes with focus on part numbers and key attributes."""
        metrics = []
        
        part_name = node.properties.get('name', node.id)
        part_model = node.properties.get('model', '')
        part_type = node.properties.get('type', '')
        
        # 1. First priority: Extract part numbers using pattern matching
        part_number_patterns = ['part_number', 'pn', 'drawing_number']
        part_numbers = {}
        
        # Check each property for part number patterns
        for prop_name, prop_value in node.properties.items():
            prop_name_lower = prop_name.lower()
            
            # Check if property contains any part number pattern
            for pattern in part_number_patterns:
                if pattern in prop_name_lower:
                    # Extract the classifier from the property name
                    classifier = prop_name.replace('_', ' ')
                    classifier = classifier.replace('-', ' ').title()
                    
                    # Handle both single values and lists
                    if isinstance(prop_value, list):
                        value_str = ', '.join(str(v) for v in prop_value)
                    else:
                        value_str = str(prop_value)
                    
                    part_numbers[classifier] = value_str
        
        # Create part number cross-reference metric if any part numbers found
        if part_numbers:
            # Build a comprehensive part identification metric
            pn_text = []
            for classifier, value in part_numbers.items():
                pn_text.append(f"{classifier}: {value}")
            
            metric_name = f"{part_name} - Part Numbers"
            if part_model:
                metric_name = f"{part_name} ({part_model}) - Part Numbers"
            
            description_parts = [f"Part: {part_name}"]
            if part_model:
                description_parts.append(f"Model: {part_model}")
            if part_type:
                description_parts.append(f"Type: {part_type}")
            description_parts.extend(pn_text)
            description_parts.extend(relationships)
            
            metrics.append(EnhancedMetric(
                metric_name=metric_name,
                category=category,
                source_node_id=node.id,
                node_type=node.type,
                value='; '.join(pn_text),
                unit=None,
                properties=node.properties,
                relationships=relationships,
                description_text=" | ".join(description_parts),
                metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
            ))
            self.metrics_count += 1
        
        # 2. Extract other properties (excluding already processed ones)
        skip_properties = {'name', 'model', 'type', 'tqdcs_categories', 'id', 'metadata'}
        # Also skip properties we've already processed as part numbers
        for prop_name in node.properties.keys():
            prop_name_lower = prop_name.lower()
            for pattern in part_number_patterns:
                if pattern in prop_name_lower:
                    skip_properties.add(prop_name)
                    break
        
        # Process remaining properties
        for prop_name, prop_value in node.properties.items():
            if prop_name in skip_properties or prop_value is None:
                continue
            
            # Create metric based on property type
            prop_display = prop_name.replace('_', ' ').title()
            metric_name = f"{part_name} - {prop_display}"
            if part_model:
                metric_name = f"{part_name} ({part_model}) - {prop_display}"
            
            # Handle different value types
            if isinstance(prop_value, dict) and 'value' in prop_value:
                # Structured value with unit
                value = prop_value.get('value')
                unit = prop_value.get('unit', '')
                
                description_parts = [
                    f"Part: {part_name}",
                    f"{prop_display}: {value} {unit}".strip()
                ]
                if part_model:
                    description_parts.insert(1, f"Model: {part_model}")
                description_parts.extend(relationships)
                
                metrics.append(EnhancedMetric(
                    metric_name=metric_name,
                    category=category,
                    source_node_id=node.id,
                    node_type=node.type,
                    value=value,
                    unit=unit,
                    properties=node.properties,
                    relationships=relationships,
                    description_text=" | ".join(description_parts),
                    metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
                ))
                self.metrics_count += 1
                
            elif isinstance(prop_value, list):
                # List values
                if all(isinstance(item, (str, int, float)) for item in prop_value):
                    value_str = ', '.join(str(v) for v in prop_value)
                else:
                    value_str = str(prop_value)
                
                description_parts = [
                    f"Part: {part_name}",
                    f"{prop_display}: {value_str}"
                ]
                if part_model:
                    description_parts.insert(1, f"Model: {part_model}")
                description_parts.extend(relationships)
                
                metrics.append(EnhancedMetric(
                    metric_name=metric_name,
                    category=category,
                    source_node_id=node.id,
                    node_type=node.type,
                    value=value_str,
                    unit=None,
                    properties=node.properties,
                    relationships=relationships,
                    description_text=" | ".join(description_parts),
                    metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
                ))
                self.metrics_count += 1
                
            else:
                # Simple values
                description_parts = [
                    f"Part: {part_name}",
                    f"{prop_display}: {prop_value}"
                ]
                if part_model:
                    description_parts.insert(1, f"Model: {part_model}")
                description_parts.extend(relationships)
                
                metrics.append(EnhancedMetric(
                    metric_name=metric_name,
                    category=category,
                    source_node_id=node.id,
                    node_type=node.type,
                    value=prop_value,
                    unit=None,
                    properties=node.properties,
                    relationships=relationships,
                    description_text=" | ".join(description_parts),
                    metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
                ))
                self.metrics_count += 1
        
        # If no metrics were created, create a basic metric
        if not metrics:
            description = node.properties.get('description', '')
            metric_name = f"{part_name} - General Info"
            
            description_parts = [f"Part: {part_name}"]
            if part_model:
                description_parts.append(f"Model: {part_model}")
            if part_type:
                description_parts.append(f"Type: {part_type}")
            if description:
                description_parts.append(f"Description: {description}")
            description_parts.extend(relationships)
            
            metrics.append(EnhancedMetric(
                metric_name=metric_name,
                category=category,
                source_node_id=node.id,
                node_type=node.type,
                value=part_name,
                unit=None,
                properties=node.properties,
                relationships=relationships,
                description_text=" | ".join(description_parts),
                metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
            ))
            self.metrics_count += 1
        
        return metrics
    
    def _extract_certification_metrics(self, node: KGNodeSchema, category: str, relationships: List[str]) -> List[EnhancedMetric]:
        """Extract metrics from Certification nodes."""
        metrics = []
        
        # 1. First priority: Extract name and description
        cert_name = node.properties.get('name', node.properties.get('certification_type', 'Unknown Certification'))
        description = node.properties.get('description', '')
        
        # 2. Search for standard and validity period using pattern matching
        standard = ''
        validity = ''
        
        # Pattern matching for properties
        for prop_name, prop_value in node.properties.items():
            prop_name_lower = prop_name.lower()
            
            # Check for standard patterns
            if 'standard' in prop_name_lower and not standard:
                standard = str(prop_value)
            
            # Check for validity/period patterns
            elif any(pattern in prop_name_lower for pattern in ['validity', 'period', 'expiry', 'expiration']):
                if not validity:
                    if isinstance(prop_value, dict) and 'value' in prop_value:
                        validity = f"{prop_value['value']} {prop_value.get('unit', '')}".strip()
                    else:
                        validity = str(prop_value)
        
        metric_name = f"Certification: {cert_name}"
        
        # Build description parts
        description_parts = [cert_name]
        if description:
            description_parts.append(f"Description: {description}")
        if standard:
            description_parts.append(f"Standard: {standard}")
        if validity:
            description_parts.append(f"Validity: {validity}")
        
        # Add any other interesting properties not yet captured
        skip_properties = {'name', 'description', 'certification_type', 'tqdcs_categories', 'id', 'metadata'}
        for prop_name in node.properties:
            if prop_name in skip_properties:
                continue
            prop_name_lower = prop_name.lower()
            # Skip if already processed
            if any(pattern in prop_name_lower for pattern in ['standard', 'validity', 'period', 'expiry', 'expiration']):
                continue
            # Add other relevant properties
            prop_value = node.properties[prop_name]
            if prop_value and prop_value not in ['', None, [], {}]:
                prop_display = prop_name.replace('_', ' ').title()
                if isinstance(prop_value, dict) and 'value' in prop_value:
                    description_parts.append(f"{prop_display}: {prop_value['value']} {prop_value.get('unit', '')}".strip())
                else:
                    description_parts.append(f"{prop_display}: {prop_value}")
        
        description_parts.extend(relationships)
        
        metric = EnhancedMetric(
            metric_name=metric_name,
            category=category,
            source_node_id=node.id,
            node_type=node.type,
            value=cert_name,
            unit=None,
            properties=node.properties,
            relationships=relationships,
            description_text=" | ".join(description_parts),
            metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
        )
        
        metrics.append(metric)
        self.metrics_count += 1
        
        return metrics
    
    def _extract_location_metrics(self, node: KGNodeSchema, category: str, relationships: List[str]) -> List[EnhancedMetric]:
        """Extract metrics from Location nodes."""
        metrics = []
        
        # 1. Get the location name from 'name' attribute
        location_name = node.properties.get('name', '')
        
        # 2. Extract location-related attributes using pattern matching
        location_info = {}
        location_patterns = {
            'address': ['address', 'street', 'addr'],
            'city': ['city', 'town', 'municipality'],
            'state': ['state', 'province', 'region'],
            'country': ['country', 'nation'],
            'postal_code': ['postal_code', 'zip', 'postcode', 'zip_code'],
            'url': ['url', 'website', 'link'],
            'coordinates': ['coordinates', 'latitude', 'longitude', 'gps']
        }
        
        # Search through properties for location patterns
        for prop_name, prop_value in node.properties.items():
            prop_name_lower = prop_name.lower()
            
            for info_type, patterns in location_patterns.items():
                if any(pattern in prop_name_lower for pattern in patterns):
                    if prop_value and prop_value not in ['', None, [], {}]:
                        location_info[info_type] = str(prop_value)
        
        metric_name = f"Delivery Location: {location_name}"
        
        # Build comprehensive description
        description_parts = [f"Location: {location_name}"]
        
        # Add location details in a structured way
        if location_info.get('address'):
            description_parts.append(f"Address: {location_info['address']}")
        if location_info.get('city'):
            description_parts.append(f"City: {location_info['city']}")
        if location_info.get('state'):
            description_parts.append(f"State/Region: {location_info['state']}")
        if location_info.get('country'):
            description_parts.append(f"Country: {location_info['country']}")
        if location_info.get('postal_code'):
            description_parts.append(f"Postal Code: {location_info['postal_code']}")
        if location_info.get('url'):
            description_parts.append(f"URL: {location_info['url']}")
        if location_info.get('coordinates'):
            description_parts.append(f"Coordinates: {location_info['coordinates']}")
        
        # Add any other properties not captured by patterns
        skip_properties = {'name', 'tqdcs_categories', 'id', 'metadata'}
        skip_properties.update([prop for prop in node.properties.keys() 
                                if any(pattern in prop.lower() 
                                    for patterns in location_patterns.values() 
                                    for pattern in patterns)])
        
        for prop_name, prop_value in node.properties.items():
            if prop_name in skip_properties:
                continue
            if prop_value and prop_value not in ['', None, [], {}]:
                prop_display = prop_name.replace('_', ' ').title()
                description_parts.append(f"{prop_display}: {prop_value}")
        
        description_parts.extend(relationships)
        
        # Create location value as a formatted string
        location_value = location_name
        if location_info:
            # Create a structured location string
            location_parts = [location_name]
            if location_info.get('city'):
                location_parts.append(location_info['city'])
            if location_info.get('country'):
                location_parts.append(location_info['country'])
            location_value = ', '.join(location_parts)
        
        metric = EnhancedMetric(
            metric_name=metric_name,
            category=category,
            source_node_id=node.id,
            node_type=node.type,
            value=location_value,
            unit=None,
            properties=node.properties,
            relationships=relationships,
            description_text=" | ".join(description_parts),
            metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
        )
        
        metrics.append(metric)
        self.metrics_count += 1
        
        return metrics
    
    def _extract_generic_metrics(self, node: KGNodeSchema, category: str, relationships: List[str]) -> List[EnhancedMetric]:
        """Extract metrics from GenericInformation and other nodes."""
        metrics = []
        
        # Extract description or title as metric name
        metric_name = (node.properties.get('description', '') or 
                      node.properties.get('title', '') or 
                      node.properties.get('name', node.id))
        
        # Build comprehensive description
        description_parts = []
        
        # Add all text-based properties
        text_props = ['description', 'information', 'details', 'notes', 'terms']
        for prop in text_props:
            if prop in node.properties and node.properties[prop]:
                description_parts.append(f"{prop}: {node.properties[prop]}")
        
        # Add relationships
        description_parts.extend(relationships)
        
        # Look for any quantitative values
        value = None
        unit = None
        for key, val in node.properties.items():
            if isinstance(val, (int, float)):
                value = val
                metric_name = f"{metric_name} - {key}: {val}"
                break
            elif isinstance(val, dict) and 'value' in val:
                value = val['value']
                unit = val.get('unit', '')
                metric_name = f"{metric_name} - {key}: {value} {unit}"
                break
        
        if description_parts or value is not None:
            metric = EnhancedMetric(
                metric_name=metric_name[:200],  # Limit length
                category=category,
                source_node_id=node.id,
                node_type=node.type,
                value=value,
                unit=unit,
                properties=node.properties,
                relationships=relationships,
                description_text=" | ".join(description_parts),
                metadata=[SourceMetadata(**m.model_dump()) for m in node.metadata] if node.metadata else []
            )
            
            metrics.append(metric)
            self.metrics_count += 1
        
        return metrics


class StaticMetricClassificationPipeline:
    """
    Complete pipeline for extracting metrics and classifying them 
    into static subcategories.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4.1",
        max_workers: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the classification pipeline.
        
        Args:
            llm_model: LLM model to use for classification
            max_workers: Maximum parallel workers
            verbose: Whether to print progress
        """
        self.extractor = MetricExtractor()
        self.classifier = SubcategoryClassifier(
            llm_model=llm_model,
            max_workers=max_workers,
            verbose=verbose
        )
        self.verbose = verbose
    
    def process_knowledge_graph(
        self,
        kg: KnowledgeGraph
    ) -> Dict[str, List[MetricCluster]]:
        """
        Process a knowledge graph to extract and classify metrics.
        
        Args:
            kg: Knowledge graph to process
            
        Returns:
            Dict of TQSDC category -> list of metric clusters
        """
        # Step 1: Extract metrics
        if self.verbose:
            print("Extracting metrics from knowledge graph...")
        metrics_by_category = self.extractor.extract_metrics_with_context(kg)
        
        # Step 2: Classify into static subcategories
        if self.verbose:
            print("\nClassifying metrics into static subcategories...")
        classified_metrics = self.classifier.classify_metrics(metrics_by_category)
        
        # Step 3: Convert to MetricCluster format
        if self.verbose:
            print("\nConverting to MetricCluster format...")
        clustered_metrics = {}
        
        for category in ['T', 'Q', 'D', 'C', 'S']:
            clusters = self.classifier.create_metric_clusters(classified_metrics, category)
            clustered_metrics[category] = clusters
            
            if self.verbose:
                print(f"Category {category}: {len(clusters)} subcategories")
                for cluster in clusters:
                    print(f"  - {cluster.name}: {cluster.size} metrics")
        
        return clustered_metrics
    
    def load_and_process_kg_file(
        self,
        kg_file_path: str
    ) -> Tuple[Dict[str, List[MetricCluster]], KnowledgeGraph]:
        """
        Load a knowledge graph from file and process it.
        
        Args:
            kg_file_path: Path to the KG JSON file
            
        Returns:
            Tuple of (clustered_metrics, knowledge_graph)
        """
        # Load the JSON file
        if self.verbose:
            print(f"Loading knowledge graph from: {kg_file_path}")
        with open(kg_file_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        # Parse into data models
        nodes = [KGNodeSchema(**node) for node in kg_data['nodes']]
        relationships = [KGRelationshipSchema(**rel) for rel in kg_data['relationships']]
        
        # Create KnowledgeGraph object
        kg = KnowledgeGraph(nodes=nodes, relationships=relationships)
        
        if self.verbose:
            print(f"Knowledge Graph loaded successfully!")
            print(f"Total nodes: {len(kg.nodes)}")
            print(f"Total relationships: {len(kg.relationships)}")
        
        # Process the knowledge graph
        clustered_metrics = self.process_knowledge_graph(kg)
        
        return clustered_metrics, kg
    
    def generate_hierarchical_json(
        self,
        clustered_metrics: Dict[str, List[MetricCluster]], 
        kg: KnowledgeGraph
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate hierarchical JSON output with structure:
        {
            "T": {
                "Subcategory Name": {
                    "cluster_metadata": {...},
                    "nodes": [...]
                },
                ...
            },
            ...
        }
        """
        hierarchical_output = {}
        
        # Create a mapping of node IDs to full node data for quick lookup
        node_map = {node.id: node for node in kg.nodes}
        
        for category, clusters in clustered_metrics.items():
            if not clusters:
                continue
                
            category_data = {}
            
            for cluster in clusters:
                cluster_data = {
                    "cluster_metadata": {
                        "cluster_id": cluster.cluster_id,
                        "cluster_name": cluster.name,
                        "size": cluster.size,
                        "category": cluster.category,
                        "representative_metric": cluster.representative_metric.metric_name if cluster.representative_metric else None
                    },
                    "nodes": []
                }
                
                # For each metric in the cluster, get the full node information
                for metric in cluster.metrics:
                    # Get the original node from the KG
                    source_node = node_map.get(metric.source_node_id)
                    
                    if source_node:
                        # Create a comprehensive node entry
                        node_entry = {
                            "node_id": source_node.id,
                            "node_type": source_node.type.value if hasattr(source_node.type, 'value') else str(source_node.type),
                            "metric_name": metric.metric_name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "properties": source_node.properties,
                            "metric_properties": metric.properties,
                            "relationships": metric.relationships,
                            "description_text": metric.description_text,
                            "metadata": [m.dict() if hasattr(m, 'dict') else m for m in (source_node.metadata or [])]
                        }
                        cluster_data["nodes"].append(node_entry)
                    else:
                        # Fallback if node not found in KG
                        node_entry = {
                            "node_id": metric.source_node_id,
                            "node_type": metric.node_type.value if hasattr(metric.node_type, 'value') else str(metric.node_type),
                            "metric_name": metric.metric_name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "properties": metric.properties,
                            "relationships": metric.relationships,
                            "description_text": metric.description_text,
                            "metadata": []
                        }
                        cluster_data["nodes"].append(node_entry)
                
                # Add cluster to category using cluster name as key
                category_data[cluster.name] = cluster_data
            
            hierarchical_output[category] = category_data
        
        return hierarchical_output