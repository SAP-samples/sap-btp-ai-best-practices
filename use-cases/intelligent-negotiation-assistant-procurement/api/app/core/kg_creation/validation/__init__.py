"""
Knowledge Graph validation module.

This module provides validation capabilities to ensure completeness
of extracted knowledge graphs by comparing them against the original text.
"""

from .kg_validator import KGValidator, ValidationResult

__all__ = [
    "KGValidator",
    "ValidationResult"
]