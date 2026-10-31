"""Immutable dataset staging, source retention, publication, and catalog review."""

from __future__ import annotations

import collections
import hashlib
import json
import uuid
from datetime import datetime, timezone

from .discovery import filter_records
from .models import INPUT_VIEWS


def now():
    """Return a timezone-aware UTC timestamp for durable application events."""
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    """Hash a deterministic identity/configuration payload."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


class DatasetService:
    """Dataset operations using the repository supplied by WorkspaceService."""

    def register_dataset(self, name, extracted, sources, *, reuse_content=False):
        """Stage extracted tables and (filename, bytes) sources; return review metadata.

        Explicit uploads create new identities. Only the legacy migration opts
        into content reuse to preserve its existing deterministic run lineage.
        """
        metadata = dict(extracted["metadata"])
        tables = dict(extracted["tables"])
        issues = list(extracted["issues"])
        unknown = set(tables) - INPUT_VIEWS
        if unknown:
            raise ValueError(f"unregistered extracted tables: {sorted(unknown)}")
        identity = {
            "hashes": metadata.get("input_hashes") or digest(tables),
            "settings": metadata.get("settings", {}),
            "parser": metadata.get("parser_version", "legacy"),
        }
        # Each explicit import owns its review/publication lifecycle, even when
        # the bytes match a published reference. Keep content identity as evidence.
        fingerprint = digest(identity)
        dataset_id = fingerprint[:32] if reuse_content else uuid.uuid4().hex
        if reuse_content:
            try:
                return self.inspect_dataset(dataset_id)
            except KeyError:
                pass
        metadata["content_fingerprint"] = fingerprint
        tables["validation_issues"] = issues
        if not tables.get("field_dictionary"):
            tables["field_dictionary"] = [
                {
                    "table_name": view,
                    "field_name": key,
                    "description": key.replace("_", " "),
                    "source": "canonical extraction",
                }
                for view, rows in tables.items()
                for key in sorted({k for row in rows for k in row})
            ]
        metadata["counts"] = {view: len(rows) for view, rows in tables.items()}
        metadata["source_files"] = [name for name, _ in sources]
        counts = dict(
            collections.Counter(
                r.get("model_status") for r in tables.get("fini_master", [])
            )
        )
        metadata["admission_counts"] = counts
        metadata["block_count"] = len(
            {
                (r.get("plant"), r.get("sefi"))
                for r in tables.get("fini_master", [])
                if r.get("model_status") == "modeled"
            }
        )
        valid = counts.get("modeled", 0) > 0 and not any(
            i.get("severity") == "error" for i in issues
        )
        value = {
            "dataset_id": dataset_id,
            "name": name[:255] or "Dataset",
            "status": "review" if valid else "invalid",
            "revision": 1,
            "metadata": metadata,
            "issues": issues,
            "created_at": now(),
        }
        with self.repo.transaction():
            self.repo.replace_tables(dataset_id, tables)
            for filename, content in sources:
                self.repo.put_artifact(dataset_id, filename, content)
            self.repo.insert("datasets", dataset_id, value)
        return value

    def inspect_dataset(self, dataset_id):
        """Read the immutable source version, quality findings and summary counts."""
        value = self.repo.get("datasets", dataset_id)
        value["summary"] = value["metadata"].get("admission_counts", {})
        return value

    def list_datasets(self, filters=None, include_removed=False):
        """Discover datasets by status/plant/name and inclusive ISO creation bounds."""
        return filter_records(
            [
                row
                for row in self.repo.list("datasets")
                if include_removed or not row.get("removed_at")
            ],
            filters,
            "datasets",
        )

    def publish_dataset(self, dataset_id):
        """Publish a reviewed version without mutating any canonical row."""
        value = self.repo.get("datasets", dataset_id)
        if value.get("removed_at"):
            raise ValueError("This snapshot was previously removed; import its workbook as a new snapshot")
        if value["status"] == "published":
            return value
        if value["status"] != "review":
            raise ValueError("dataset has blocking errors or no modeled FINIs")
        return self.repo.cas(
            "datasets",
            dataset_id,
            value["revision"],
            {"status": "published", "published_at": now()},
        )

    def input_tables(self, dataset_id, views=None):
        """Load canonical tables from HANA for one immutable snapshot."""
        with self.repo.transaction():
            metadata = self.repo.get("datasets", dataset_id)["metadata"]
            selected = views if views is not None else metadata.get("counts", {})
            return {
                view: self.repo.rows(dataset_id, view)
                for view in selected
                if view in INPUT_VIEWS
            }
