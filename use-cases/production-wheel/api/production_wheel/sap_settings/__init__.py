"""SAP settings generation from a selected production-wheel scenario.

Ports the customer's "Wheel settings for SAP" Power Query template to Python:
- policy.py: plant policy rules (site exceptions as data, not code);
- roc.py: coverage (RoC) adjustments and SAP lot-size key mapping;
- dc_lot_sizes.py, calendar.py, recipes.py, kpis.py: one module per SAP tab;
- generate.py: runs every tab and reports ready / blocked / not applicable.

See docs/sap-settings-generation.md.
"""
