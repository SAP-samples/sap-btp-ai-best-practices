"""SAP Document AI batch runner for benchmark method artifacts."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from dox_client.models import ParsedExtraction
from dox_client.sap_dox_client import SapDoxClient

from .artifacts import BenchmarkRunConfig, MethodArtifactKey, RunStore
from .diagnostics import classify_docai_failure, inspect_pdf_text_layer
from .extraction_schema import HEADER_FIELDS, LINE_ITEM_FIELDS, normalize_quote_payload
from .settings import AppPaths


class DocumentAIAdapter(Protocol):
    def extract_quote(
        self,
        *,
        pdf_path: Path,
        scenario_key: str,
    ) -> "DocumentAICallResult":
        ...


@dataclass
class DocumentAICallResult:
    """Result of a single SAP Document AI extraction call."""

    scenario_key: str
    raw_payload: dict[str, Any]
    normalized_json: dict[str, Any]
    latency_ms: int
    outcome: str
    job_id: str | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentAITask:
    """One cacheable SAP Document AI extraction unit."""

    document_name: str
    scenario_key: str
    schema_name: str | None = None

    def artifact_key(self) -> MethodArtifactKey:
        return MethodArtifactKey(
            document_name=self.document_name,
            method_family="docai",
            scenario_key=self.scenario_key,
            model="SAP Document AI",
            prompt_version=None,
            schema_name=self.schema_name,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentAITaskResult:
    task: DocumentAITask
    status: str
    method_dir: str
    latency_ms: int = 0
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_json_dict(),
            "status": self.status,
            "method_dir": self.method_dir,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class DocumentAIBatchResult:
    run_id: str
    task_results: list[DocumentAITaskResult]
    started_at: str
    finished_at: str

    @property
    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.task_results:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_counts": self.task_counts,
            "task_results": [item.to_json_dict() for item in self.task_results],
        }


class SapDocumentAIAdapter:
    """Live SAP Document AI adapter using the local reusable DOX client."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = paths or AppPaths.for_project()
        self.client = SapDoxClient.from_service_key(str(self.paths.docai_service_key))

    def extract_quote(self, *, pdf_path: Path, scenario_key: str) -> DocumentAICallResult:
        started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        last_error: str | None = None
        for attempt in docai_attempts_for_scenario(scenario_key):
            try:
                upload = self.client.upload_document(
                    str(pdf_path),
                    client_id="default",
                    header_fields=attempt["header_fields"],
                    line_item_fields=attempt["line_item_fields"],
                    document_type=attempt.get("document_type"),
                    custom_label=_docai_custom_label(pdf_path.stem, scenario_key, attempt["name"]),
                )
                job_id = _extract_job_id(upload)
                result = self.client.wait_for_result(
                    job_id,
                    timeout_seconds=240,
                    poll_interval_seconds=3,
                    extracted_values=True,
                    return_null_values=True,
                )
                normalized = normalize_docai_payload(result)
                attempts.append({"attempt": attempt, "upload": upload, "result": result})
                if not _is_empty_extraction(normalized) or attempt.get("final"):
                    return DocumentAICallResult(
                        scenario_key=scenario_key,
                        raw_payload={"attempts": attempts, "selected_attempt": attempt["name"]},
                        normalized_json=normalized,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        outcome="success",
                        job_id=job_id,
                    )
            except Exception as exc:
                last_error = str(exc)
                attempts.append({"attempt": attempt, "error": last_error})

        return DocumentAICallResult(
            scenario_key=scenario_key,
            raw_payload={"attempts": attempts},
            normalized_json=normalize_quote_payload({"warnings": [last_error] if last_error else []}),
            latency_ms=int((time.perf_counter() - started) * 1000),
            outcome="error",
            error=last_error,
        )


def build_docai_batch_plan(config: BenchmarkRunConfig) -> list[DocumentAITask]:
    tasks: list[DocumentAITask] = []
    for document_name in config.document_names:
        for scenario_key in config.docai_scenarios:
            tasks.append(DocumentAITask(document_name=document_name, scenario_key=scenario_key))
    return tasks


def run_docai_batch(
    *,
    run_id: str,
    config: BenchmarkRunConfig,
    paths: AppPaths | None = None,
    store: RunStore | None = None,
    adapter: DocumentAIAdapter | None = None,
    progress_callback: Callable[[int, int, DocumentAITask, str], None] | None = None,
) -> DocumentAIBatchResult:
    """Run selected SAP Document AI scenarios and persist comparable artifacts."""

    paths = paths or AppPaths.for_project()
    store = store or RunStore(paths)
    adapter = adapter or SapDocumentAIAdapter(paths)
    tasks = build_docai_batch_plan(config)
    results: list[DocumentAITaskResult] = []
    started_at = _utc_now_text()

    for index, task in enumerate(tasks, start=1):
        key = task.artifact_key()
        method_dir = store.method_dir(run_id, key)
        if config.use_cached_results and not config.force_rerun and store.method_artifact_exists(run_id, key):
            status = "cached"
            results.append(DocumentAITaskResult(task=task, status=status, method_dir=str(method_dir)))
            if progress_callback:
                progress_callback(index, len(tasks), task, status)
            continue

        if progress_callback:
            progress_callback(index, len(tasks), task, "running")
        result = adapter.extract_quote(pdf_path=paths.data_dir / task.document_name, scenario_key=task.scenario_key)
        diagnostic = classify_docai_failure(error=result.error).to_json_dict() if result.outcome == "error" or result.error else None
        saved_dir = store.save_method_artifacts(
            run_id,
            key,
            raw=result.raw_payload if result.raw_payload else {"error": result.error},
            normalized=result.normalized_json,
            metrics={
                "usage": {},
                "latency_ms": result.latency_ms,
                "outcome": result.outcome,
                "provider_family": "sap_document_ai",
                "job_id": result.job_id,
                "error": result.error,
                "diagnostic_category": diagnostic["category"] if diagnostic else None,
                "diagnostic_root_cause": diagnostic["root_cause"] if diagnostic else None,
            },
            metadata=result.to_json_dict(),
        )
        if diagnostic:
            diagnostic["pdf_text_layer"] = inspect_pdf_text_layer(paths.data_dir / task.document_name)
            (saved_dir / "diagnostics.json").write_text(
                json.dumps(
                    {
                        "document_name": task.document_name,
                        "model": "SAP Document AI",
                        "scenario_key": task.scenario_key,
                        "diagnostic": diagnostic,
                        "attempts": result.raw_payload.get("attempts", []) if isinstance(result.raw_payload, dict) else [],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        status = "success" if result.outcome == "success" else "error"
        results.append(
            DocumentAITaskResult(
                task=task,
                status=status,
                method_dir=str(saved_dir),
                latency_ms=result.latency_ms,
                error=result.error,
            )
        )
        if progress_callback:
            progress_callback(index, len(tasks), task, status)

    finished_at = _utc_now_text()
    batch_result = DocumentAIBatchResult(run_id=run_id, task_results=results, started_at=started_at, finished_at=finished_at)
    store.save_json(run_id, "docai_batch_result.json", batch_result.to_json_dict())
    return batch_result


def fields_for_docai_scenario(scenario_key: str) -> tuple[list[str], list[str]]:
    """Return ad-hoc Document AI fields for each benchmark scenario."""

    first_attempt = docai_attempts_for_scenario(scenario_key)[0]
    return list(first_attempt["header_fields"]), list(first_attempt["line_item_fields"])


def docai_attempts_for_scenario(scenario_key: str) -> list[dict[str, Any]]:
    """Return ordered Document AI attempts for static and dynamic scenarios."""

    default_headers = [
        "vendor_name",
        "quote_number",
        "quote_date",
        "quote_expiration_date",
        "currency",
        "subtotal_amount",
        "tax_amount",
        "total_amount",
        "payment_terms",
        "ship_to_name",
        "ship_to_address",
    ]
    default_lines = ["description", "quantity", "unit_of_measure", "unit_price", "line_total", "expected_delivery_date"]
    wide_headers = [field for field in HEADER_FIELDS if field != "document_type"]
    wide_lines = list(LINE_ITEM_FIELDS)
    predefined_headers = [
        "documentNumber",
        "documentDate",
        "senderName",
        "senderAddress",
        "receiverName",
        "receiverAddress",
        "grossAmount",
        "netAmount",
        "taxAmount",
        "currencyCode",
        "purchaseOrderNumber",
        "paymentTerms",
        "shippingAmount",
        "deliveryDate",
        "dueDate",
    ]
    predefined_lines = ["description", "quantity", "unitPrice", "netAmount", "materialNumber", "unitOfMeasure"]
    purchase_order_headers = [
        "documentNumber",
        "documentDate",
        "senderName",
        "senderAddress",
        "receiverName",
        "receiverAddress",
        "shipToName",
        "shipToAddress",
        "grossAmount",
        "netAmount",
        "currencyCode",
        "paymentTerms",
        "deliveryDate",
    ]
    purchase_order_lines = [
        "description",
        "quantity",
        "unitPrice",
        "netAmount",
        "supplierMaterialNumber",
        "customerMaterialNumber",
        "unitOfMeasure",
    ]

    if scenario_key == "default":
        return [
            _docai_attempt("custom_default", None, default_headers, default_lines),
            _docai_attempt("invoice_default_fallback", "invoice", predefined_headers, predefined_lines),
            _docai_attempt("purchase_order_default_fallback", "purchaseOrder", purchase_order_headers, purchase_order_lines, final=True),
        ]
    if scenario_key == "wide_attributes":
        return [
            _docai_attempt("custom_wide", None, wide_headers, wide_lines),
            _docai_attempt("invoice_wide_fallback", "invoice", predefined_headers, predefined_lines),
            _docai_attempt("purchase_order_wide_fallback", "purchaseOrder", purchase_order_headers, purchase_order_lines, final=True),
        ]
    if scenario_key == "dynamic_attributes":
        return [
            _docai_attempt("dynamic_quote_fields", None, dynamic_docai_header_fields(), dynamic_docai_line_fields()),
            _docai_attempt(
                "dynamic_invoice_alias_fallback",
                "invoice",
                predefined_headers,
                predefined_lines,
            ),
            _docai_attempt(
                "dynamic_purchase_order_fallback",
                "purchaseOrder",
                purchase_order_headers,
                purchase_order_lines,
                final=True,
            ),
        ]
    if scenario_key == "dynamic_attributes_judge_loop":
        return [
            _docai_attempt("judge_loop_wide_probe", None, wide_headers, wide_lines),
            _docai_attempt("judge_loop_invoice_retry", "invoice", predefined_headers, predefined_lines),
            _docai_attempt("judge_loop_purchase_order_retry", "purchaseOrder", purchase_order_headers, purchase_order_lines, final=True),
        ]
    raise KeyError(f"Unsupported Document AI scenario: {scenario_key}")


def _docai_attempt(
    name: str,
    document_type: str | None,
    header_fields: list[str],
    line_item_fields: list[str],
    *,
    final: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "document_type": document_type,
        "header_fields": _dedupe(header_fields),
        "line_item_fields": _dedupe(line_item_fields),
        "final": final,
    }


def dynamic_docai_header_fields(include_invoice_aliases: bool = False) -> list[str]:
    fields = [field for field in HEADER_FIELDS if field != "document_type"]
    fields.extend(
        [
            "supplier_name",
            "supplier_address",
            "supplier_email",
            "quotation_number",
            "valid_until",
            "quote_validity_date",
            "total",
            "grand_total",
            "shipping",
            "freight",
        ]
    )
    if include_invoice_aliases:
        fields.extend(
            [
                "documentNumber",
                "documentDate",
                "senderName",
                "senderAddress",
                "receiverName",
                "receiverAddress",
                "grossAmount",
                "netAmount",
                "taxAmount",
                "currencyCode",
                "purchaseOrderNumber",
                "paymentTerms",
                "shippingAmount",
                "deliveryDate",
                "dueDate",
            ]
        )
    return _dedupe(fields)


def dynamic_docai_line_fields(include_invoice_aliases: bool = False) -> list[str]:
    fields = list(LINE_ITEM_FIELDS)
    fields.extend(["item_description", "item_quantity", "price", "amount", "part_number", "material_number"])
    if include_invoice_aliases:
        fields.extend(["unitPrice", "netAmount", "materialNumber", "supplierMaterialNumber", "customerMaterialNumber", "unitOfMeasure"])
    return _dedupe(fields)


def normalize_docai_payload(job_payload: dict[str, Any]) -> dict[str, Any]:
    """Map SAP Document AI extraction fields into the canonical quote schema."""

    parsed = ParsedExtraction.from_job(job_payload)
    header = {_canonical_header_name(field.name): _field_value(field.model_dump()) for field in parsed.header_fields}
    line_items: list[dict[str, Any]] = []
    for row in parsed.line_items:
        line_items.append({_canonical_line_name(field.name): _field_value(field.model_dump()) for field in row})
    warnings = []
    if not parsed.header_fields and not parsed.line_items:
        warnings.append("SAP Document AI returned no extracted fields.")
    return normalize_quote_payload(
        {
            "header": header,
            "line_items": line_items,
            "warnings": warnings,
            "evidence": _docai_evidence(parsed),
        }
    )


def _docai_evidence(parsed: ParsedExtraction) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for field in parsed.header_fields:
        payload = field.model_dump()
        evidence[_canonical_header_name(field.name)] = {
            "confidence": payload.get("confidence"),
            "page": payload.get("page"),
            "rawValue": payload.get("rawValue"),
        }
    return evidence


def _field_value(field: dict[str, Any]) -> Any:
    value = field.get("value")
    if value is None:
        value = field.get("rawValue")
    return value


def _canonical_header_name(name: str) -> str:
    normalized = str(name or "").strip()
    compact = normalized.replace(" ", "").replace("-", "").replace("_", "").lower()
    aliases = {
        "vendorname": "vendor_name",
        "suppliername": "vendor_name",
        "sendername": "vendor_name",
        "vendoraddress": "vendor_address",
        "supplieraddress": "vendor_address",
        "senderaddress": "vendor_address",
        "invoicenumber": "quote_number",
        "documentnumber": "quote_number",
        "quotationnumber": "quote_number",
        "invoicedate": "quote_date",
        "documentdate": "quote_date",
        "validuntil": "quote_expiration_date",
        "quotevaliditydate": "quote_expiration_date",
        "currencycode": "currency",
        "grossamount": "total_amount",
        "grandtotal": "total_amount",
        "total": "total_amount",
        "netamount": "subtotal_amount",
        "taxamount": "tax_amount",
        "deliveryaddress": "ship_to_address",
        "shiptoaddress": "ship_to_address",
        "shiptoname": "ship_to_name",
        "receivername": "customer_name",
        "receiveraddress": "sold_to_address",
        "purchaseordernumber": "customer_number",
        "shippingamount": "shipping_amount",
        "duedate": "quote_expiration_date",
        "deliverydate": "requested_delivery_date",
    }
    if compact in aliases:
        return aliases[compact]
    return normalized.lower().replace(" ", "_").replace("-", "_")


def _canonical_line_name(name: str) -> str:
    normalized = str(name or "").strip()
    compact = normalized.replace(" ", "").replace("-", "").replace("_", "").lower()
    aliases = {
        "unitprice": "unit_price",
        "unitofmeasure": "unit_of_measure",
        "netamount": "line_total",
        "materialnumber": "vendor_material_number",
        "suppliermaterialnumber": "vendor_material_number",
        "customermaterialnumber": "manufacturer_part_number",
        "itemdescription": "description",
        "itemquantity": "quantity",
        "amount": "line_total",
        "price": "unit_price",
        "partnumber": "manufacturer_part_number",
    }
    if compact in aliases:
        return aliases[compact]
    return normalized.lower().replace(" ", "_").replace("-", "_")


def _is_empty_extraction(normalized: dict[str, Any]) -> bool:
    header_values = [value for value in (normalized.get("header") or {}).values() if value not in (None, "", [])]
    line_items = normalized.get("line_items") or []
    line_values = [
        value
        for item in line_items
        if isinstance(item, dict)
        for value in item.values()
        if value not in (None, "", [])
    ]
    return not header_values and not line_values


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _docai_custom_label(document_stem: str, scenario_key: str, attempt_name: str) -> str:
    """Return a unique SAP Document AI custom label under the service length limit."""

    suffix = uuid.uuid4().hex[:8]
    base = f"benchmark_{document_stem}_{scenario_key}_{attempt_name}"
    return f"{base[:71]}_{suffix}"[:80]


def _extract_job_id(upload_payload: dict[str, Any]) -> str:
    for key in ("id", "jobId", "documentId"):
        value = upload_payload.get(key)
        if value:
            return str(value)
    payload = upload_payload.get("payload")
    if isinstance(payload, dict):
        for key in ("id", "jobId", "documentId"):
            value = payload.get(key)
            if value:
                return str(value)
    raise RuntimeError(f"Document AI upload response did not include a job id: {upload_payload}")


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
