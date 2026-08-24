"""
pa_extractor.py
---------------
Extracts structured Payment Advice data from a PDF using SAP Document AI
with the "SAP_paymentAdvice_schema" schema.

Reuses InvoiceProcessor from modules/invoice/process_invoice.py — only the
schema name and field mapping differ.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from modules.invoice.process_invoice import InvoiceProcessor, InvoiceProcessingError
from PaymentAdviceProcess.pa_models import ExtractedPaymentAdvice, PaymentAdviceLine

logger = logging.getLogger(__name__)

PA_SCHEMA_NAME = "SAP_paymentAdvice_schema"
PA_DOCUMENT_TYPE = "paymentAdvice"

# SAP Document AI field name aliases for payment advice header
_HEADER_ALIASES: dict[str, list[str]] = {
    "payer_name":         ["payerName", "senderName", "payerCompany", "remitterName"],
    "payment_date":       ["paymentDate", "documentDate", "valueDate"],
    "total_amount":       ["totalAmount", "grossAmount", "paymentAmount", "netAmount"],
    "currency":           ["currencyCode", "currency"],
    "bank_reference":     ["bankReference", "transactionReference", "wireReference",
                           "bankTransactionId", "paymentReference"],
    "our_reference":      ["ourReference", "receiverReference", "creditorReference"],
    "payment_advice_note": ["note", "memo", "paymentNote", "remittanceInfo"],
}

# SAP Document AI field name aliases for payment advice line items
_LINE_ALIASES: dict[str, list[str]] = {
    "invoice_number":       ["invoiceNumber", "documentNumber", "referenceDocument",
                             "invoiceReference", "assignmentReference"],
    "invoice_date":         ["invoiceDate", "documentDate", "accountingDocumentCreationDate"],
    "gross_amount":         ["grossAmount", "amount", "invoiceAmount",
                             "grossAmountInPaymentCurrency", "paymentAmount"],
    "discount_amount":      ["discountAmount", "discount", "deductionAmount",
                             "cashDiscountAmount", "cashDiscountAmountInPaytCrcy"],
    "net_payment_amount":   ["netAmount", "paymentAmount", "settlementAmount",
                             "netPaymentAmount", "netPaymentAmountInPaytCurrency",
                             "paidAmountInPaytCurrency"],
    "currency":             ["currencyCode", "currency", "paymentCurrency"],
    "payment_reference":    ["paymentReference", "assignmentReference", "reference",
                             "documentReferenceID"],
}


def _first_value(fields_dict: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        val = fields_dict.get(alias)
        if val is not None and val != "":
            return val
    return None


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


class PaymentAdviceExtractor:
    """
    Extracts Payment Advice data from a PDF using SAP Document AI
    with the SAP_paymentAdvice_schema schema.
    """

    def __init__(self) -> None:
        self._processor = InvoiceProcessor()

    def extract(self, pdf_path: Path, client_id: str = "default") -> ExtractedPaymentAdvice:
        logger.info(
            "Submitting Payment Advice for extraction | file=%s | schema=%s",
            pdf_path.name, PA_SCHEMA_NAME,
        )

        job_id = self._processor.submit_document(
            pdf_path,
            schema_name=PA_SCHEMA_NAME,
            client_id=client_id,
            document_type=PA_DOCUMENT_TYPE,
        )
        logger.info("Document AI job | job_id=%s", job_id)

        result = self._processor.poll_until_done(job_id)
        self._processor.save_result(job_id, result)

        return self._map_result(result)

    def _map_result(self, result: dict[str, Any]) -> ExtractedPaymentAdvice:
        extraction = result.get("extraction") or result.get("document") or {}
        header_fields_raw: list[dict] = extraction.get("headerFields") or []
        line_items_raw: list[Any] = extraction.get("lineItems") or []

        header_dict: dict[str, Any] = {
            f.get("name", ""): (f.get("value") if f.get("value") is not None else f.get("rawValue"))
            for f in header_fields_raw if f.get("name")
        }

        payer_name    = str(_first_value(header_dict, _HEADER_ALIASES["payer_name"]) or "")
        payment_date  = str(_first_value(header_dict, _HEADER_ALIASES["payment_date"]) or "")
        total_amount  = _safe_float(_first_value(header_dict, _HEADER_ALIASES["total_amount"]))
        currency      = str(_first_value(header_dict, _HEADER_ALIASES["currency"]) or "")
        bank_ref      = str(_first_value(header_dict, _HEADER_ALIASES["bank_reference"]) or "")
        our_ref       = str(_first_value(header_dict, _HEADER_ALIASES["our_reference"]) or "")
        note          = str(_first_value(header_dict, _HEADER_ALIASES["payment_advice_note"]) or "")

        line_items = self._map_line_items(line_items_raw, currency)

        logger.info(
            "PA extracted | payer=%r | date=%r | amount=%s %s | items=%d",
            payer_name, payment_date, total_amount, currency, len(line_items),
        )

        return ExtractedPaymentAdvice(
            payer_name=payer_name,
            payment_date=payment_date,
            total_amount=total_amount,
            currency=currency,
            bank_reference=bank_ref,
            our_reference=our_ref,
            payment_advice_note=note,
            line_items=line_items,
            raw_sap_fields=dict(header_dict),
        )

    def _map_line_items(
        self, line_items_raw: list[Any], fallback_currency: str
    ) -> list[PaymentAdviceLine]:
        items: list[PaymentAdviceLine] = []
        for idx, raw_item in enumerate(line_items_raw):
            if isinstance(raw_item, list):
                fields: list[dict] = raw_item
            elif isinstance(raw_item, dict):
                fields = [{"name": k, "value": v} for k, v in raw_item.items()]
            else:
                logger.warning("Unexpected line item format at index %d", idx)
                continue

            fd: dict[str, Any] = {
                f.get("name", ""): (f.get("value") if f.get("value") is not None else f.get("rawValue"))
                for f in fields if isinstance(f, dict) and f.get("name")
            }

            items.append(PaymentAdviceLine(
                invoice_number      = str(_first_value(fd, _LINE_ALIASES["invoice_number"]) or ""),
                invoice_date        = str(_first_value(fd, _LINE_ALIASES["invoice_date"]) or ""),
                gross_amount        = _safe_float(_first_value(fd, _LINE_ALIASES["gross_amount"])),
                discount_amount     = _safe_float(_first_value(fd, _LINE_ALIASES["discount_amount"])),
                net_payment_amount  = _safe_float(_first_value(fd, _LINE_ALIASES["net_payment_amount"])),
                currency            = str(_first_value(fd, _LINE_ALIASES["currency"]) or fallback_currency),
                payment_reference   = str(_first_value(fd, _LINE_ALIASES["payment_reference"]) or ""),
            ))
        return items
