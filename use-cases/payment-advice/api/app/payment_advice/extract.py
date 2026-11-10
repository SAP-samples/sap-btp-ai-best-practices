"""
Extraction orchestration for the Payment Advice Extractor (UC-01).

Ties the splitter and the SAP Document AI client together:

    probe/split -> upload each part -> poll each job -> recombine (concat mode)

The Document AI client is injected through a small ``DocumentAIClient`` protocol so
the whole orchestration is testable with a fake (no live SAP). The real
``dox_client.SapDoxClient`` satisfies the protocol.

Recombination uses ``aggregate_documents(..., dedupe_line_items=False)`` so the
disjoint parts of one split document are concatenated in order without merging.
"""

from __future__ import annotations

import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .aggregation import aggregate_documents
from .config import MAX_COLUMNS, MAX_LINE_ITEMS, MAX_PAGES
from .splitter import probe, split_document

# Terminal SAP Document AI job statuses.
_SUCCESS_STATUSES = {"DONE"}
_FAILURE_STATUSES = {"FAILED", "ERROR", "RETRY_FAILED"}

# Explicit MIME types for the formats we send; fall back to mimetypes / octet-stream.
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".eml": "message/rfc822",
}


class ExtractionError(RuntimeError):
    """Raised when a Document AI job fails or does not complete in time."""


class DocumentAIClient(Protocol):
    """Minimal SAP Document AI surface used by the orchestrator (fake-able)."""

    def upload_document_stream(
        self, file_name: str, file_obj: Any, mime_type: str, **options: Any
    ) -> dict[str, Any]: ...

    def get_job(
        self,
        job_id: str,
        *,
        extracted_values: bool | None = None,
        return_null_values: bool | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class ExtractResult:
    """Outcome of extracting one advice (possibly split into parts)."""

    aggregate: dict[str, Any]
    part_names: list[str] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    oversized: bool = False
    size_reason: str = ""


def mime_for(path: str | Path) -> str:
    """Return the MIME type to send for a file, by extension."""
    ext = Path(path).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _job_id(upload_response: dict[str, Any]) -> str:
    """Extract the job id from an upload response, tolerating key variants."""
    job_id = upload_response.get("id") or upload_response.get("jobId")
    if not job_id:
        raise ExtractionError(f"Document AI upload returned no job id: {upload_response!r}")
    return str(job_id)


def _poll_extraction(
    client: DocumentAIClient,
    job_id: str,
    *,
    poll_interval: float,
    poll_timeout: float,
    sleep: Callable[[float], None],
    now: Callable[[], float],
) -> dict[str, Any]:
    """
    Poll a Document AI job until terminal; return its extraction payload.

    Args:
        client: Injected Document AI client.
        job_id: The job to poll.
        poll_interval: Seconds between polls.
        poll_timeout: Maximum seconds to wait before failing.
        sleep / now: Injectable time seams for testing.

    Returns:
        The job's ``extraction`` dict (``{"headerFields": [...], "lineItems": [...]}``).

    Raises:
        ExtractionError: On a failed job or a timeout.
    """
    deadline = now() + poll_timeout
    while True:
        # Null values are requested too: a cell Document AI cannot normalize (value=null)
        # still carries its printed rawValue, which customer rules may need.
        job = client.get_job(job_id, extracted_values=True, return_null_values=True)
        status = str(job.get("status") or "").upper()
        if status in _SUCCESS_STATUSES:
            return job.get("extraction") or {}
        if status in _FAILURE_STATUSES:
            raise ExtractionError(f"Document AI job {job_id} failed with status {status}")
        if now() >= deadline:
            raise ExtractionError(
                f"Document AI job {job_id} did not finish within {poll_timeout:.0f}s "
                f"(last status {status or 'unknown'})"
            )
        sleep(poll_interval)


def extract_document(
    path: str | Path,
    *,
    client: DocumentAIClient,
    schema: dict[str, Any],
    dox_client_id: str,
    out_dir: str | Path,
    schema_id: str | None = None,
    schema_version: str | int | None = None,
    max_pages: int = MAX_PAGES,
    max_line_items: int = MAX_LINE_ITEMS,
    max_columns: int = MAX_COLUMNS,
    poll_interval: float = 2.0,
    poll_timeout: float = 300.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> ExtractResult:
    """
    Extract one advice end-to-end: split if needed, upload, poll, recombine.

    Args:
        path: Input advice file.
        client: Injected Document AI client (real or fake).
        schema: Schema field definitions used for number/date normalization during
            recombination (may be ``{}`` to keep raw string values).
        dox_client_id: SAP Document AI client id for the upload options.
        out_dir: Directory for split chunk files.
        schema_id / schema_version: Bound schema to extract against. When omitted,
            the client's own defaults apply (mainly for tests/ad-hoc use).
        max_pages / max_line_items / max_columns: Document AI limits for the splitter.
        poll_interval / poll_timeout / sleep / now: Polling controls and time seams.

    Returns:
        An ``ExtractResult`` with the recombined aggregate, part names, and job ids.

    Raises:
        ColumnLimitExceeded: From the splitter for >max_columns tabular input.
        ExtractionError: On job failure or timeout.
    """
    path = Path(path)
    report = probe(
        path, max_pages=max_pages, max_line_items=max_line_items, max_columns=max_columns
    )
    parts = split_document(
        path,
        out_dir,
        max_pages=max_pages,
        max_line_items=max_line_items,
        max_columns=max_columns,
    )

    documents: list[dict[str, Any]] = []
    job_ids: list[str] = []
    mime_type = mime_for(path)  # parts keep the source extension, so one MIME applies
    upload_options: dict[str, Any] = {"client_id": dox_client_id}
    if schema_id:
        upload_options["schema_id"] = schema_id
    if schema_version is not None:
        upload_options["schema_version"] = schema_version

    for index, part in enumerate(parts):
        with Path(part).open("rb") as handle:
            upload_response = client.upload_document_stream(
                file_name=Path(part).name,
                file_obj=handle,
                mime_type=mime_type,
                **upload_options,
            )
        job_id = _job_id(upload_response)
        job_ids.append(job_id)
        extraction = _poll_extraction(
            client,
            job_id,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
            sleep=sleep,
            now=now,
        )
        documents.append(
            {"id": f"{Path(part).stem}#{index}", "file_name": Path(part).name, "extraction": extraction}
        )

    # Concat mode: parts are disjoint slices of one document; never dedup lines.
    aggregate = aggregate_documents(schema, documents, dedupe_line_items=False)
    return ExtractResult(
        aggregate=aggregate,
        part_names=[Path(p).name for p in parts],
        job_ids=job_ids,
        oversized=report.oversized,
        size_reason=report.reason,
    )


def build_client(service_key_data: dict[str, Any]) -> DocumentAIClient:
    """Build the real SAP Document AI client from parsed service-key data."""
    from dox_client import SapDoxClient  # local import keeps requests optional for pure tests

    return SapDoxClient.from_service_key_data(service_key_data)
