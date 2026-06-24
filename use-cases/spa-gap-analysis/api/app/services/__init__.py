"""
Services Module: Business Logic for SPA Gap Analysis

This module contains core business logic services:
- Data loading with LRU caching
- Customer profiling and search
- Similar customer finding (similarity engine)
- SPA gap detection
- Material coverage analysis
- RFM segmentation
- Confidence scoring
"""

from .data_loader import (
    load_customer_master,
    load_transactions,
    load_qualifications,
    load_rfm_scores,
    load_customer_cogs,
    load_header_data,
    load_a703_nets,
    load_a700_spa_vendors,
    load_all_datasets,
    load_from_parquet
)

from .customer_service import (
    get_customer_profile,
    search_customers,
    get_customer_spas,
    get_customer_materials
)

from .similarity_engine import (
    find_similar_customers,
    calculate_similarity_score
)

from .gap_detector import (
    detect_spa_gaps,
    get_gap_recommendations
)

from .material_matcher import (
    check_material_coverage,
    get_uncovered_materials
)

from .rfm_analyzer import (
    get_rfm_segment,
    get_segment_customers,
    calculate_customer_value,
    get_rfm_distribution
)

from .confidence_scorer import (
    calculate_confidence_score,
    calculate_simple_confidence,
    get_recommendation_details
)

from .summary_aggregator import (
    aggregate_customer_summaries,
    generate_and_cache_summary,
    save_summary_cache
)

__all__ = [
    # Data Loader
    'load_customer_master',
    'load_transactions',
    'load_qualifications',
    'load_rfm_scores',
    'load_customer_cogs',
    'load_header_data',
    'load_a703_nets',
    'load_a700_spa_vendors',
    'load_all_datasets',
    'load_from_parquet',
    # Customer Service
    'get_customer_profile',
    'search_customers',
    'get_customer_spas',
    'get_customer_materials',
    # Similarity Engine
    'find_similar_customers',
    'calculate_similarity_score',
    # Gap Detector
    'detect_spa_gaps',
    'get_gap_recommendations',
    # Material Matcher
    'check_material_coverage',
    'get_uncovered_materials',
    # RFM Analyzer
    'get_rfm_segment',
    'get_segment_customers',
    'calculate_customer_value',
    'get_rfm_distribution',
    # Confidence Scorer
    'calculate_confidence_score',
    'calculate_simple_confidence',
    'get_recommendation_details',
    # Summary Aggregator
    'aggregate_customer_summaries',
    'generate_and_cache_summary',
    'save_summary_cache'
]
