"""Persistence layer for the ECN impact pipeline.

This package owns:

* ``table_contracts`` — the immutable seven-table HANA catalog with allowlisted
  identifiers, column-level structural contracts, and deterministic DDL
  rendering.
* ``hana`` — safe SQLAlchemy engine construction from validated environment
  credentials, plus UTC timestamp helpers.

New modules in this package must use :mod:`app.persistence.hana` for database
connectivity.
"""
