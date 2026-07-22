"""Document discovery and lightweight metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .settings import AppPaths


@dataclass(frozen=True)
class DocumentMetadata:
    """Local document metadata shown in the UI."""

    name: str
    path: str
    size_bytes: int
    suffix: str
    text_length: int | None = None
    pdf_type: str = "unknown"

    def to_json_dict(self) -> dict:
        return asdict(self)


def extract_pdf_text_length(path: Path) -> int | None:
    """Return extractable text length when pypdf is installed."""

    try:
        import pypdf  # type: ignore
    except Exception:
        return None

    try:
        parts: list[str] = []
        with path.open("rb") as handle:
            reader = pypdf.PdfReader(handle)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return len("\n".join(parts).strip())
    except Exception:
        return None


def classify_pdf_type(text_length: int | None) -> str:
    """Classify a PDF by text-layer availability."""

    if text_length is None:
        return "unknown"
    return "searchable" if text_length >= 50 else "scanned_or_image_only"


def list_documents(paths: AppPaths | None = None) -> list[DocumentMetadata]:
    """List supported documents from the data directory."""

    paths = paths or AppPaths()
    if not paths.data_dir.exists():
        return []
    docs: list[DocumentMetadata] = []
    for path in sorted(paths.data_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        text_length = extract_pdf_text_length(path) if path.suffix.lower() == ".pdf" else None
        docs.append(
            DocumentMetadata(
                name=path.name,
                path=str(path),
                size_bytes=path.stat().st_size,
                suffix=path.suffix.lower(),
                text_length=text_length,
                pdf_type=classify_pdf_type(text_length),
            )
        )
    return docs
