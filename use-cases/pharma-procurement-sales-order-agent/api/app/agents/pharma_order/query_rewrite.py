"""Deterministic query rewrite rules for Pharma Procurement Sales Order Agent.

The goal is not to change business intent. The goal is to make action-style
requests explicit as preview-only requests so Joule/DAS and the backend agent
route them to the right read/preview tools without implying SAP write-back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRewriteResult:
    original_question: str
    rewritten_question: str
    rule_id: str | None = None
    rationale: str | None = None

    @property
    def changed(self) -> bool:
        return self.rewritten_question != self.original_question


@dataclass(frozen=True)
class QueryRewriteRule:
    rule_id: str
    pattern: re.Pattern[str]
    replacement: str
    rationale: str


_RULES: tuple[QueryRewriteRule, ...] = (
    QueryRewriteRule(
        rule_id="preview_release_delivery_block",
        pattern=re.compile(r"\b(release|remove|clear|unblock)\b.*\b(delivery\s+blocks?|blocks?)\b", re.IGNORECASE),
        replacement=(
            "Preview the release of delivery blocks for the requested open sales orders. "
            "Do not update SAP. Identify affected orders, block reason, release prerequisites, "
            "and the next action required from the service representative."
        ),
        rationale="Convert delivery-block action request into preview-only analysis.",
    ),
    QueryRewriteRule(
        rule_id="preview_create_sales_order",
        pattern=re.compile(r"\b(create|enter|submit)\b.*\b(sales\s+order|order)\b", re.IGNORECASE),
        replacement=(
            "Review the sales order request and explain what data is required to create the order. "
            "Do not create or update SAP. Check pricing, availability, compliance, duplicate PO, "
            "and missing order-entry fields if possible."
        ),
        rationale="Convert create-order request into read-only order readiness review.",
    ),
    QueryRewriteRule(
        rule_id="preview_reject_order_lines",
        pattern=re.compile(r"\b(reject|cancel)\b.*\b(order\s+lines?|orders?)\b", re.IGNORECASE),
        replacement=(
            "Preview which order lines would be rejected and what confirmation is required. "
            "Do not update SAP. Identify candidate orders or lines, rejection reason, risks, "
            "and next action for the service representative."
        ),
        rationale="Convert reject/cancel action into preview-only analysis.",
    ),
    QueryRewriteRule(
        rule_id="preview_partner_function_change",
        pattern=re.compile(r"\b(add|replace|change|update)\b.*\b(partner\s+function|freight\s+forwarder)\b", re.IGNORECASE),
        replacement=(
            "Preview the partner-function change for the requested sales order. Do not update SAP. "
            "Identify the target order, current context if available, required partner role, "
            "and next action for an authorized service representative."
        ),
        rationale="Convert partner-function update into preview-only analysis.",
    ),
    QueryRewriteRule(
        rule_id="preview_expedite_order",
        pattern=re.compile(r"\b(expedite|rush)\b.*\b(order|sales\s+order)\b", re.IGNORECASE),
        replacement=(
            "Check whether the requested order can be expedited and what route or shipping-condition "
            "changes would be required. Do not update SAP. Summarize blockers, shipment status, "
            "and next action."
        ),
        rationale="Convert expedite action into feasibility and preview analysis.",
    ),
    QueryRewriteRule(
        rule_id="preview_send_order_acknowledgement",
        pattern=re.compile(r"\b(send|email)\b.*\b(order\s+acknowledg(e)?ment|acknowledg(e)?ment\s+pdf)\b", re.IGNORECASE),
        replacement=(
            "Find order acknowledgement context and explain how the acknowledgement PDF would be sent. "
            "Do not send email or generate a document. Summarize available order evidence and next action."
        ),
        rationale="Convert email/send request into preview-only acknowledgement guidance.",
    ),
    QueryRewriteRule(
        rule_id="preview_invoice_pdf_generation",
        pattern=re.compile(r"\b(generate|print|retrieve|get)\b.*\b(invoice\s+pdfs?|pdf\s+copies|invoice)\b", re.IGNORECASE),
        replacement=(
            "Find invoice PDF metadata and explain which invoice PDFs would be retrieved or generated. "
            "Do not generate or download binary PDFs. Summarize invoice identifiers, availability, "
            "and next action."
        ),
        rationale="Convert invoice PDF action into metadata lookup and guidance.",
    ),
)


def rewrite_order_question(question: str) -> QueryRewriteResult:
    """Return a deterministic preview-safe rewrite when a known action pattern is found."""
    original = (question or "").strip()
    if not original:
        return QueryRewriteResult(original_question=original, rewritten_question=original)

    for rule in _RULES:
        if rule.pattern.search(original):
            rewritten = f"{rule.replacement}\n\nOriginal user request: {original}"
            return QueryRewriteResult(
                original_question=original,
                rewritten_question=rewritten,
                rule_id=rule.rule_id,
                rationale=rule.rationale,
            )

    return QueryRewriteResult(original_question=original, rewritten_question=original)
