"""
Database Abstraction Layer

Provides a dual-backend (SQLite / SAP HANA Cloud) database interface.
Backend is selected automatically based on environment variables.
"""

from .backend import BackendType, DatabaseBackend, get_backend

__all__ = ["BackendType", "DatabaseBackend", "get_backend"]
