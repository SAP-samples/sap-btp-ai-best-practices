"""Access to the (mocked) SAP ECC system.

In this build the functions read and write the ``*_POAGENT`` HANA tables. In production the
same functions are re-implemented to call SAP Integration Suite iFlows (see docs/ecc-integration.md);
callers do not change.
"""
