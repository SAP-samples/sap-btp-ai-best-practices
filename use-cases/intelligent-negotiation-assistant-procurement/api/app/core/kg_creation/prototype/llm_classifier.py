"""
LLM-based Subcategory Classifier for TQSDC Metrics

This module implements parallel LLM classification of metrics into 
predefined static subcategories for consistent supplier comparison.
"""

import sys
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import json
from collections import defaultdict

# Add parent directories for imports
sys.path.append('../')
sys.path.append('../../')

# Import LLM factory
from llm.factory import create_llm

# Import static subcategories
from static_subcategories import (
    Subcategory,
    get_subcategories_for_category,
    create_classification_prompt,
    find_subcategory_by_name
)

# Import data models
from models.kg_schema import Node as KGNodeSchema


@dataclass
class ClassificationResult:
    """Result of classifying a metric into a subcategory."""
    node_id: str
    category: str
    subcategory_name: str
    confidence: float
    reasoning: Optional[str] = None
    error: Optional[str] = None


class SubcategoryClassifier:
    """
    Classifies metrics into predefined static subcategories using LLM.
    Supports parallel processing for efficiency.
    """
    
    def __init__(
        self, 
        llm_model: str = "gpt-4.1",
        max_workers: int = 10,
        temperature: float = 0.1,
        max_retries: int = 3,
        verbose: bool = True
    ):
        """
        Initialize the subcategory classifier.
        
        Args:
            llm_model: LLM model to use for classification
            max_workers: Maximum parallel workers
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Maximum retries for failed classifications
            verbose: Whether to print progress
        """
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Create LLM instance
        self.llm = create_llm(model_name=llm_model, temperature=temperature)
        
        # Cache for classification results
        self.classification_cache = {}
        
    def classify_metrics(
        self,
        metrics_by_category: Dict[str, List['EnhancedMetric']]
    ) -> Dict[str, Dict[str, List['EnhancedMetric']]]:
        """
        Classify all metrics into static subcategories.
        
        Args:
            metrics_by_category: Dict of TQSDC category -> list of metrics
            
        Returns:
            Dict of TQSDC category -> subcategory name -> list of metrics
        """
        classified_metrics = {}
        
        for category, metrics in metrics_by_category.items():
            if self.verbose:
                print(f"\nClassifying {len(metrics)} metrics in category {category}...")
            
            # Get subcategories for this category
            subcategories = get_subcategories_for_category(category)
            if not subcategories:
                print(f"Warning: No subcategories defined for category {category}")
                continue
            
            # Classify metrics in parallel
            category_results = self._classify_category_parallel(
                category, metrics, subcategories
            )
            
            # Group by subcategory
            classified_metrics[category] = self._group_by_subcategory(
                metrics, category_results
            )
            
            if self.verbose:
                print(f"Category {category} classification complete:")
                for subcat_name, subcat_metrics in classified_metrics[category].items():
                    print(f"  - {subcat_name}: {len(subcat_metrics)} metrics")
        
        return classified_metrics
    
    def _classify_category_parallel(
        self,
        category: str,
        metrics: List['EnhancedMetric'],
        subcategories: List[Subcategory]
    ) -> Dict[str, ClassificationResult]:
        """
        Classify all metrics in a category in parallel.
        
        Returns:
            Dict mapping metric node_id to classification result
        """
        results = {}
        
        # Create tasks for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit classification tasks
            future_to_metric = {}
            
            for metric in metrics:
                # Check cache first
                cache_key = f"{category}_{metric.source_node_id}"
                if cache_key in self.classification_cache:
                    results[metric.source_node_id] = self.classification_cache[cache_key]
                    continue
                
                # Submit new classification task
                future = executor.submit(
                    self._classify_single_metric,
                    metric,
                    category,
                    subcategories
                )
                future_to_metric[future] = metric
            
            # Process completed classifications
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    result = future.result()
                    results[metric.source_node_id] = result
                    
                    # Cache result
                    cache_key = f"{category}_{metric.source_node_id}"
                    self.classification_cache[cache_key] = result
                    
                except Exception as e:
                    print(f"Error classifying metric {metric.metric_name}: {e}")
                    # Create error result
                    results[metric.source_node_id] = ClassificationResult(
                        node_id=metric.source_node_id,
                        category=category,
                        subcategory_name="Unclassified",
                        confidence=0.0,
                        error=str(e)
                    )
        
        return results
    
    def _classify_single_metric(
        self,
        metric: 'EnhancedMetric',
        category: str,
        subcategories: List[Subcategory]
    ) -> ClassificationResult:
        """
        Classify a single metric into a subcategory using LLM.
        """
        # Prepare node content for classification
        node_content = self._prepare_node_content(metric)
        
        # Create classification prompt
        prompt = create_classification_prompt(node_content, category, subcategories)
        
        # Try classification with retries
        for attempt in range(self.max_retries):
            try:
                # Call LLM
                response = self.llm.invoke(prompt)
                
                # Extract subcategory name from response
                if hasattr(response, 'content'):
                    subcategory_name = response.content.strip()
                else:
                    subcategory_name = str(response).strip()
                
                # Validate the response
                subcategory_name = self._validate_subcategory_name(
                    subcategory_name, subcategories
                )
                
                # Create result
                return ClassificationResult(
                    node_id=metric.source_node_id,
                    category=category,
                    subcategory_name=subcategory_name,
                    confidence=0.95  # High confidence for direct LLM classification
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    raise e
        
        # Should not reach here
        raise Exception("Classification failed after all retries")
    
    def _prepare_node_content(self, metric: 'EnhancedMetric') -> str:
        """
        Prepare comprehensive node content for classification.
        """
        parts = []
        
        # Add metric name
        parts.append(f"Metric: {metric.metric_name}")
        
        # Add node type
        parts.append(f"Node Type: {metric.node_type.value if hasattr(metric.node_type, 'value') else metric.node_type}")
        
        # Add value if present
        if metric.value is not None:
            value_str = f"Value: {metric.value}"
            if metric.unit:
                value_str += f" {metric.unit}"
            parts.append(value_str)
        
        # Add key properties
        important_props = [
            'description', 'cost_type', 'name', 'type', 'model',
            'certification_type', 'standard', 'requirement'
        ]
        for prop in important_props:
            if prop in metric.properties:
                parts.append(f"{prop}: {metric.properties[prop]}")
        
        # Add relationships for context
        if metric.relationships:
            parts.append(f"Relationships: {', '.join(metric.relationships[:5])}")
        
        # Use description text if available
        if hasattr(metric, 'description_text') and metric.description_text:
            parts.append(f"Context: {metric.description_text[:200]}")
        
        return " | ".join(parts)
    
    def _validate_subcategory_name(
        self,
        response: str,
        subcategories: List[Subcategory]
    ) -> str:
        """
        Validate and normalize the subcategory name from LLM response.
        """
        # Get valid subcategory names
        valid_names = [sub.name for sub in subcategories]
        
        # Direct match
        if response in valid_names:
            return response
        
        # Case-insensitive match
        response_lower = response.lower()
        for name in valid_names:
            if name.lower() == response_lower:
                return name
        
        # Partial match (if response contains the subcategory name)
        for name in valid_names:
            if name.lower() in response_lower or response_lower in name.lower():
                return name
        
        # If no match found, use first subcategory as default
        print(f"Warning: Could not match '{response}' to valid subcategories. Using default.")
        return valid_names[0] if valid_names else "Unclassified"
    
    def _group_by_subcategory(
        self,
        metrics: List['EnhancedMetric'],
        classification_results: Dict[str, ClassificationResult]
    ) -> Dict[str, List['EnhancedMetric']]:
        """
        Group metrics by their classified subcategory.
        """
        grouped = defaultdict(list)
        
        for metric in metrics:
            result = classification_results.get(metric.source_node_id)
            if result and not result.error:
                subcategory_name = result.subcategory_name
            else:
                subcategory_name = "Unclassified"
            
            grouped[subcategory_name].append(metric)
        
        return dict(grouped)
    
    def create_metric_clusters(
        self,
        classified_metrics: Dict[str, Dict[str, List['EnhancedMetric']]],
        category: str
    ) -> List['MetricCluster']:
        """
        Convert classified metrics into MetricCluster objects for compatibility.
        """
        from dataclasses import dataclass
        from typing import List, Optional
        
        @dataclass
        class MetricCluster:
            cluster_id: str
            name: str
            category: str
            metrics: List['EnhancedMetric']
            size: int = 0
            representative_metric: Optional['EnhancedMetric'] = None
            
            def __post_init__(self):
                self.size = len(self.metrics)
                if self.metrics and not self.representative_metric:
                    # Use first metric as representative
                    self.representative_metric = self.metrics[0]
        
        clusters = []
        
        if category not in classified_metrics:
            return clusters
        
        for idx, (subcategory_name, metrics) in enumerate(
            classified_metrics[category].items()
        ):
            if not metrics:
                continue
            
            cluster = MetricCluster(
                cluster_id=f"{category}_{idx}",
                name=subcategory_name,
                category=category,
                metrics=metrics
            )
            clusters.append(cluster)
        
        return clusters
    
    def save_classification_cache(self, filepath: str):
        """Save classification cache to file for reuse."""
        with open(filepath, 'w') as f:
            json.dump(self.classification_cache, f, indent=2)
    
    def load_classification_cache(self, filepath: str):
        """Load classification cache from file."""
        try:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
                # Convert to ClassificationResult objects
                for key, value in cache_data.items():
                    self.classification_cache[key] = ClassificationResult(**value)
        except FileNotFoundError:
            print(f"Cache file {filepath} not found. Starting with empty cache.")
        except Exception as e:
            print(f"Error loading cache: {e}. Starting with empty cache.")