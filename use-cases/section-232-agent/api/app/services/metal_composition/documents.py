"""Document upload validation for UI-driven classification flows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from app.models.metal_composition import DocumentReference
from .ui_state import StoredDocumentReference


class DocumentValidationError(ValueError):
    """Raised when a persisted document path is invalid."""


class MetalCompositionDocumentStore:
    """Validate and store uploaded PDFs under the configured upload root."""

    def __init__(self, uploaded_root: Path) -> None:
        self.uploaded_root = uploaded_root.resolve()
        self.uploaded_root.mkdir(parents=True, exist_ok=True)

    def validate_paths(self, document_paths: Iterable[str]) -> List[DocumentReference]:
        references: List[DocumentReference] = []
        for raw_path in document_paths:
            path = Path(raw_path).expanduser().resolve()
            if path.suffix.lower() != ".pdf":
                raise DocumentValidationError(f"{path} is not a PDF file.")
            if not path.exists() or not path.is_file():
                raise DocumentValidationError(f"{path} does not exist.")
            if path.is_relative_to(self.uploaded_root):
                references.append(self._build_reference(path, source="uploaded"))
                continue
            raise DocumentValidationError(f"{path} is outside the configured upload document root.")
        return references

    def references_from_paths(self, document_paths: Iterable[str], *, source: str) -> List[DocumentReference]:
        del source
        references: List[DocumentReference] = []
        for raw_path in document_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                continue
            if not self._is_known_root(path):
                continue
            references.append(self._build_reference(path, source="uploaded"))
        return references

    def references_from_stored(self, document_refs: Iterable[StoredDocumentReference]) -> List[DocumentReference]:
        references: List[DocumentReference] = []
        for document_ref in document_refs:
            path = self.resolve_stored_path(document_ref)
            if path is None or not path.exists() or not path.is_file():
                continue
            references.append(self._build_reference(path, source=document_ref.source))
        return references

    def to_stored_reference(self, reference: DocumentReference) -> StoredDocumentReference:
        path = Path(reference.path).expanduser().resolve()
        if path.is_relative_to(self.uploaded_root):
            return StoredDocumentReference(source="uploaded", relative_path=reference.relative_path)
        raise DocumentValidationError(f"{path} is outside the configured upload document root.")

    def resolve_stored_path(self, document_ref: StoredDocumentReference) -> Optional[Path]:
        if document_ref.source != "uploaded":
            return None
        relative_path = Path(document_ref.relative_path)
        path = (self.uploaded_root / relative_path).resolve()
        if path.is_relative_to(self.uploaded_root):
            return path
        return None

    def save_uploaded_pdf(
        self,
        *,
        dataset_scope: str,
        item_id: str,
        filename: str,
        content: bytes,
    ) -> DocumentReference:
        sanitized_name = self._sanitize_filename(filename)
        target_dir = (self.uploaded_root / dataset_scope / item_id).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = (target_dir / sanitized_name).resolve()
        if not target_path.is_relative_to(self.uploaded_root):
            raise DocumentValidationError("Resolved upload target is outside the upload document root.")
        target_path.write_bytes(content)
        return self._build_reference(target_path, source="uploaded")

    def _is_known_root(self, path: Path) -> bool:
        return path.is_relative_to(self.uploaded_root)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        candidate = Path(filename or "").name.strip() or "uploaded.pdf"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(candidate).stem).strip("._") or "uploaded"
        suffix = Path(candidate).suffix.lower()
        if suffix != ".pdf":
            suffix = ".pdf"
        return f"{stem}{suffix}"

    def _build_reference(self, path: Path, *, source: str) -> DocumentReference:
        if source != "uploaded":
            raise DocumentValidationError("Only uploaded PDFs can be assigned to items.")
        stat = path.stat()
        if path.is_relative_to(self.uploaded_root):
            relative_path = path.relative_to(self.uploaded_root).as_posix()
        else:
            raise DocumentValidationError(f"{path} is outside the configured upload document root.")
        return DocumentReference(
            path=str(path),
            relative_path=relative_path,
            file_name=path.name,
            source=source,
            size_bytes=int(stat.st_size),
        )
