"""
document_classifier.py
-----------------------
Classifies a document as "invoice" or "purchase_order" BEFORE sending to
SAP Document AI, so the correct schema is used from the start.

Strategy (in order):
1. Fast PDF text scan — deterministic keyword matching (no LLM cost)
2. LLM fallback only if keywords are ambiguous

Key insight: In a customer Purchase Order, AI4U is the VENDOR/recipient.
The buyer puts AI4U in "TO:" or "Vendor:" — DocAI may confuse sender/receiver.
We classify by DOCUMENT TYPE keywords, not by party names.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Keywords that unambiguously identify a Purchase Order ──────────────────
_PO_KEYWORDS = [
    r"\bpurchase\s+order\b",
    r"\bbestellung\b",
    r"\bbestellnummer\b",
    r"\borden\s+de\s+compra\b",
    r"\bbon\s+de\s+commande\b",
    r"\bpo\s+number\b",
    r"\bpo\s*#",
    r"\border\s+number\b",          # generic "Order Number" (not invoice)
    r"\bship\s*-?\s*to\b",
    r"\brequested\s+delivery\b",
    r"\blieferdatum\b",             # German delivery date
    r"\bbestellpositionen\b",       # German PO line items
]

# ── Keywords that unambiguously identify a Payment Advice ─────────────────
_PA_KEYWORDS = [
    r"\bpayment\s+advice\b",
    r"\bremittance\s+advice\b",
    r"\bremittance\s+slip\b",
    r"\bzahlungsavis\b",
    r"\bavis\s+de\s+paiement\b",
    r"\baviso\s+de\s+pago\b",
    r"\bpayment\s+notification\b",
    r"\bgutschriftanzeige\b",
    r"\bremittance\s+information\b",
]

# ── Keywords that unambiguously identify an Invoice ────────────────────────
_INVOICE_KEYWORDS = [
    r"\binvoice\s+number\b",
    r"\binvoice\s+no\.?\b",
    r"\brechnungsnummer\b",
    r"\bnumero\s+de\s+factura\b",
    r"\bfactura\b",
    r"\bamount\s+due\b",
    r"\bplease\s+remit\b",
    r"\bbank\s+transfer\b",
    r"\biban\b",
    r"\bpayment\s+due\b",
    r"\brechnung\b",                # German invoice
    r"\bzahlbar\b",                 # German "payable"
]


def _extract_pdf_text(file_path: Path) -> str:
    """Extract plain text from PDF for keyword scanning."""
    try:
        import pdfplumber
        with pdfplumber.open(str(file_path)) as pdf:
            return " ".join(
                page.extract_text() or ""
                for page in pdf.pages[:3]   # scan first 3 pages max
            ).lower()
    except Exception:
        pass
    # Fallback: try PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(file_path))
        text = " ".join(doc[i].get_text() for i in range(min(3, len(doc))))
        doc.close()
        return text.lower()
    except Exception:
        return ""


def _keyword_classify(text: str) -> str | None:
    """
    Classify by keyword score.
    Returns "payment_advice", "purchase_order", "invoice", or None (ambiguous).
    Payment Advice takes priority — its keywords are very specific.
    """
    if not text:
        return None

    pa_score  = sum(1 for pat in _PA_KEYWORDS if re.search(pat, text))
    po_score  = sum(1 for pat in _PO_KEYWORDS if re.search(pat, text))
    inv_score = sum(1 for pat in _INVOICE_KEYWORDS if re.search(pat, text))

    logger.info(
        "Keyword classification | pa_score=%d | po_score=%d | invoice_score=%d",
        pa_score, po_score, inv_score,
    )

    # Payment Advice wins if it has any match and more than invoice/PO
    if pa_score > 0 and pa_score >= po_score and pa_score >= inv_score:
        return "payment_advice"
    if po_score > 0 and inv_score == 0:
        return "purchase_order"
    if inv_score > 0 and po_score == 0:
        return "invoice"
    if po_score > inv_score:
        return "purchase_order"
    if inv_score > po_score:
        return "invoice"

    return None  # ambiguous — fall through to LLM


_LLM_PROMPT = """Look at this document. Answer with EXACTLY one word.

Answer "purchase_order" if:
- The main title says "Purchase Order", "Bestellung", "Orden de Compra", or similar
- A company is ORDERING goods FROM a vendor
- Has fields like "PO Number", "Ship-To", "Requested Delivery Date"

Answer "invoice" if:
- The main title says "Invoice", "Rechnung", "Factura", or similar
- A seller is requesting PAYMENT
- Has fields like "Invoice Number", "Amount Due", "IBAN", "Bank Account"

Answer with only one word: purchase_order or invoice"""


def classify_document(file_path: Path) -> str:
    """
    Classify a document as "invoice" or "purchase_order".

    1. Fast keyword scan of PDF text
    2. LLM fallback for ambiguous/image-only PDFs
    Returns "purchase_order" or "invoice" (default fallback).
    """
    # Step 1: keyword scan (fast, free, deterministic)
    if file_path.suffix.lower() == ".pdf":
        text = _extract_pdf_text(file_path)
        result = _keyword_classify(text)
        if result:
            logger.info("Classified by keywords: %s | file=%s", result, file_path.name)
            return result
        logger.info("Keywords ambiguous — falling back to LLM | file=%s", file_path.name)

    # Step 2: LLM (for image PDFs or ambiguous docs)
    try:
        from modules.genai.llm_client import ask_llm_multimodal
        raw = ask_llm_multimodal(_LLM_PROMPT, file_path)
        cleaned = raw.strip().lower().replace('"', "").replace("'", "").split()[0]
        if "purchase" in cleaned:
            logger.info("LLM classified as: purchase_order | file=%s", file_path.name)
            return "purchase_order"
        logger.info("LLM classified as: invoice | file=%s | raw=%r", file_path.name, raw)
        return "invoice"
    except Exception as exc:
        logger.warning("LLM classification failed (%s) — defaulting to invoice", exc)
        return "invoice"
