"""
LangChain tool wrappers exposing UC-02 helpers to the agent runtime.

Each function decorated with @tool becomes a LangChain BaseTool whose name,
description, and input schema are derived from the function signature and
docstring.  The HANA engine is bound via closure for customer / rules tools
so the agent never handles the connection object directly.

The three full-document tools (document_statistics, list_invoice_references,
fetch_invoices) read the active payment advice from the
ContextVar managed by ``app.deduction_agent.document_context`` instead of
accepting a payload argument.  This keeps tool signatures clean and lets the
agent focus on business logic rather than data routing.

build_uc02_tools(engine) returns exactly 8 tools:
  - find_customers            — fuzzy search customers by display name
  - get_customer              — look up one customer by normalized client_key
  - get_deduction_rules       — read a client's playbook
  - save_deduction_rules      — stage or persist a playbook revision
  - delete_deduction_rules    — preview or hard-delete a playbook
  - document_statistics       — aggregate stats for the bound document
  - list_invoice_references   — paginated list of invoice_reference values
  - fetch_invoices            — retrieve lines by invoice_reference(s)

Example usage:
    from app.deduction_agent.tools.uc02_tools import build_uc02_tools
    from app.deduction_agent.document_context import set_current_document, reset_current_document

    tools = build_uc02_tools(engine)
    token = set_current_document(payload)
    try:
        result = tools_dict["document_statistics"].invoke({})
    finally:
        reset_current_document(token)
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from sqlalchemy.engine import Engine

from app.deduction_agent.document_context import get_current_document
from app.payment_advice import customers as customers_mod
from app.payment_advice import deduction_rules as rules_mod
from app.payment_advice import deductions


def build_uc02_tools(engine: Engine) -> list[BaseTool]:
    """
    Return the UC-02 tool set with a HANA engine bound into each tool via closure.

    The returned list contains exactly 8 tools: 5 customer/rules tools that
    receive their data via explicit arguments, and 3 full-document tools that
    read the active payment advice payload from the ContextVar managed by
    ``app.deduction_agent.document_context``.

    Args:
        engine: SQLAlchemy Engine connected to the HANA instance that holds the
                PAYMENT_ADVICE_CUSTOMERS and DEDUCTION_RULES tables.

    Returns:
        A list of 8 LangChain BaseTool instances ready to pass to a ReAct agent.
    """

    # ------------------------------------------------------------------
    # Customer tools
    # ------------------------------------------------------------------

    @tool
    def find_customers(query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fuzzy-search known customers by display name. Returns key, name, score."""
        return [
            {"client_key": k, "display_name": n, "score": s}
            for k, n, s in customers_mod.find_customers(engine, query, limit)
        ]

    @tool
    def get_customer(client_key: str) -> dict[str, Any] | None:
        """Fetch one customer's record by normalized client_key."""
        c = customers_mod.get_customer(engine, client_key)
        return None if c is None else c.__dict__

    # ------------------------------------------------------------------
    # Deduction rules (playbook) tools
    # ------------------------------------------------------------------

    @tool
    def get_deduction_rules(client_key: str) -> dict | None:
        """Read a client's deduction playbook (skill catalog). None if not set."""
        pb = rules_mod.get_playbook(engine, client_key)
        return None if pb is None else pb.__dict__

    @tool
    def save_deduction_rules(
        client_key: str,
        playbook_text: str,
        anchors: dict,
        user_confirmed: bool = False,
        updated_by: str = "",
    ) -> dict:
        """Stage a playbook (user_confirmed=False) or persist a new revision (True)."""
        return rules_mod.save_playbook(
            engine, client_key, playbook_text, anchors, user_confirmed, updated_by or None
        )

    @tool
    def delete_deduction_rules(client_key: str, user_confirmed: bool = False) -> dict:
        """Preview (user_confirmed=False) or delete (True) a client's playbook."""
        return rules_mod.delete_playbook(engine, client_key, user_confirmed)

    # ------------------------------------------------------------------
    # Full-document tools — read payload from ContextVar, no payload arg
    # ------------------------------------------------------------------

    @tool
    def document_statistics() -> dict:
        """Return aggregate statistics for all line items in the bound document.

        Reads the payment advice payload from the current execution context
        (bound via set_current_document).  Returns a dict with keys:
        total_lines, anomalous_count, plain_count, negative_count,
        with_deduction_reason_count, gross_total, net_total, currency,
        distinct_customer_account_references.

        Call this tool first to understand the document's size, sign
        distribution, and currency before deciding which invoices to fetch.
        """
        return deductions.document_statistics(get_current_document())

    @tool
    def list_invoice_references(offset: int = 0, limit: int = 50) -> dict:
        """Return a paginated list of invoice_reference values from the bound document.

        Reads the payment advice payload from the current execution context
        (bound via set_current_document).  Only lines with a non-empty
        invoice_reference are included.  Returns a dict with keys:
        total, offset, limit, references (list[str]), has_more.

        Use offset / limit to page through large documents.  The total field
        tells you how many distinct references are available.
        """
        return deductions.list_invoice_references(
            get_current_document(), offset, limit
        )

    @tool
    def fetch_invoices(references: list[str] | str) -> dict:
        """Retrieve line items by invoice_reference(s) from the bound document.

        Reads the payment advice payload from the current execution context
        (bound via set_current_document).  Accepts a single reference string
        or a list of strings.  Returns a dict with keys:
        invoices (list of matched lines, each augmented with line_index),
        not_found (references that matched no line).

        Use this tool to pull the full detail of one or more invoice lines
        once you know their references from list_invoice_references.
        """
        return deductions.fetch_invoices(get_current_document(), references)

    return [
        find_customers,
        get_customer,
        get_deduction_rules,
        save_deduction_rules,
        delete_deduction_rules,
        document_statistics,
        list_invoice_references,
        fetch_invoices,
    ]
