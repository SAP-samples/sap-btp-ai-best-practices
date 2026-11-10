"""Atomic upload/row persistence with retry identity separate from source identity."""

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from ...models.workspace import RevisionConflict
from .schema import decode_json, ensure_schema, transaction


def input_digest(content, analysis_date, settings):
    """Hash source bytes and effective analysis settings to detect key misuse."""
    metadata = json.dumps({"date": str(analysis_date), "settings": settings}, sort_keys=True)
    return hashlib.sha256(content + metadata.encode()).hexdigest()


def filter_rows(rows, filters):
    """Filter table visibility without changing an optimizer's saved candidate scope."""
    result = []
    for row in rows:
        invoice = row["invoice"]
        status = filters.get("status")
        if status in ("eligible", "not_eligible") and row["eligible"] != (status == "eligible"):
            continue
        if any(filters.get(key) and str(invoice.get(key) or "") != str(filters[key])
               for key in ("seller_id", "debtor_id", "programa", "insurer_id", "original_currency")):
            continue
        search = str(filters.get("search") or "").casefold()
        if search and search not in " ".join(str(value or "") for value in invoice.values()).casefold():
            continue
        result.append(row)
    return result


class AnalysisStore:
    """Save original bytes, rule outcomes and source positions in one transaction."""

    def __init__(self, backend, db_path=None):
        """Use the supplied backend, initializing its required tables once per store."""
        self.backend, self.db_path = backend, db_path
        ensure_schema(backend, db_path)

    def create(self, content, filename, analysis_date, settings, rows, request_key):
        """Save a new analysis or return the original response for an identical retry."""
        digest = input_digest(content, analysis_date, settings)
        source_hash = hashlib.sha256(content).hexdigest()
        metadata = dict(analysis_id=str(uuid4()), filename=filename, source_hash=source_hash,
                        analysis_date=str(analysis_date), settings=settings,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        total_invoices=len(rows), eligible_count=sum(row["eligible"] for row in rows))
        metadata["not_eligible_count"] = len(rows) - metadata["eligible_count"]
        metadata["eligible_row_ids"] = [row["row_id"] for row in rows if row["eligible"]]
        try:
            with transaction(self.backend, self.db_path) as cursor:
                cursor.execute("SELECT input_hash, metadata FROM RECEIVABLES_ANALYSES WHERE request_key = ?", (request_key,))
                if saved := cursor.fetchone():
                    if saved[0] != digest:
                        raise RevisionConflict("Idempotency key already used with different inputs")
                    return decode_json(saved[1])
                cursor.execute("INSERT INTO RECEIVABLES_ANALYSES "
                               "(analysis_id, request_key, input_hash, source_hash, created_at, metadata, content) "
                               "VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (metadata["analysis_id"], request_key, digest, source_hash,
                                metadata["created_at"], json.dumps(metadata), content))
                # One driver batch avoids a network round trip for every invoice while
                # retaining the same all-or-nothing upload transaction.
                cursor.executemany("INSERT INTO RECEIVABLES_SOURCE_ROWS (analysis_id, row_id, row_number, payload) "
                                   "VALUES (?, ?, ?, ?)", [(metadata["analysis_id"], row["row_id"],
                                                          row["source_row_number"], json.dumps(row)) for row in rows])
            return metadata
        except Exception:
            # A concurrent identical retry may win the unique key while this insert waits.
            with transaction(self.backend, self.db_path) as cursor:
                cursor.execute("SELECT input_hash, metadata FROM RECEIVABLES_ANALYSES WHERE request_key = ?", (request_key,))
                saved = cursor.fetchone()
                if saved and saved[0] == digest:
                    return decode_json(saved[1])
            raise

    def get(self, analysis_id):
        """Return immutable analysis metadata, or raise LookupError for an absent ID."""
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("SELECT metadata FROM RECEIVABLES_ANALYSES WHERE analysis_id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row is None:
                raise LookupError("Analysis not found")
            return decode_json(row[0])

    def original_content(self, analysis_id):
        """Read the original uploaded bytes for an authorized export or audit."""
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("SELECT content FROM RECEIVABLES_ANALYSES WHERE analysis_id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row is None:
                raise LookupError("Analysis not found")
            return bytes(row[0].read() if hasattr(row[0], "read") else row[0])

    def list_analyses(self, limit=20, offset=0):
        """Return newest saved analyses with bounded SQL pagination."""
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("SELECT COUNT(*) FROM RECEIVABLES_ANALYSES")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT metadata FROM RECEIVABLES_ANALYSES ORDER BY created_at DESC, analysis_id "
                           "LIMIT ? OFFSET ?", (limit, offset))
            return {"items": [decode_json(row[0]) for row in cursor.fetchall()], "total": total}

    def all_rows(self, analysis_id):
        """Load only the specified analysis's canonical rows in physical source order."""
        self.get(analysis_id)
        with transaction(self.backend, self.db_path) as cursor:
            if not self.backend.is_hana:
                cursor.execute("SELECT payload FROM RECEIVABLES_SOURCE_ROWS WHERE analysis_id = ? ORDER BY row_number", (analysis_id,))
                return [decode_json(row[0]) for row in cursor.fetchall()]
            # Small NCLOBs otherwise incur a separate network read for every invoice.
            # Read a bounded scalar prefix in bulk; retrieve full oversized values below.
            cursor.execute("SELECT TO_NVARCHAR(SUBSTRING(payload,1,5000)), LENGTH(payload), row_id "
                           "FROM RECEIVABLES_SOURCE_ROWS WHERE analysis_id = ? ORDER BY row_number", (analysis_id,))
            records = cursor.fetchall()
            result = []
            for text, length, row_id in records:
                if length > 5000:
                    cursor.execute("SELECT payload FROM RECEIVABLES_SOURCE_ROWS WHERE analysis_id = ? AND row_id = ?", (analysis_id,row_id))
                    text = cursor.fetchone()[0]
                result.append(decode_json(text))
            return result

    def list_rows(self, analysis_id, filters, limit=100, offset=0):
        """Return a visible page and its full filtered count, preserving stable row IDs."""
        rows = filter_rows(self.all_rows(analysis_id), filters)
        return {"items": rows[offset:offset + limit], "total": len(rows)}
