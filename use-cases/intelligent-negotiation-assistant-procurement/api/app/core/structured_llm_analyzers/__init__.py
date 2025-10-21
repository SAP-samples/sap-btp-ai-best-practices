"""
Structured LLM Analyzers for Dashboard

These analyzers generate structured JSON output from knowledge graphs,
designed specifically for dashboard consumption while preserving source traceability.
"""

from .structured_cost_analyzer import analyze_costs_structured
from .structured_risk_analyzer import analyze_risks_structured
from .structured_tqdcs_analyzer import analyze_tqdcs_structured
from .structured_comparator import compare_suppliers_structured
from .structured_parts_analyzer import analyze_parts_structured
from .structured_homepage import analyze_homepage_structured

__all__ = [
    'analyze_costs_structured',
    'analyze_risks_structured', 
    'analyze_tqdcs_structured',
    'compare_suppliers_structured',
    'analyze_parts_structured',
    'analyze_homepage_structured'
]