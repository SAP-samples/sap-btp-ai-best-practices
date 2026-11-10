"""Expose caller-authorized PDF, Word, and Excel rule documents to the agent.

The caller binds an explicit source-file allowlist for one asynchronous run.
The two LangChain tools in this module can then list those sources and read
their text in bounded chunks.  Model-provided paths are never resolved against
the filesystem, so the tools cannot read files outside the caller's allowlist.

DOCX extraction uses Python's ZIP/XML standard library. XLSX extraction uses
the project's existing ``openpyxl`` dependency. PDF extraction uses the existing
``pypdf`` dependency (digital text only -- scanned/image PDFs have no OCR).
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from langchain_core.tools import BaseTool, ToolException, tool
from openpyxl import load_workbook
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class RuleSourceError(ValueError):
    """Report an invalid, unauthorized, unsupported, or unreadable rule source."""


_SUPPORTED_SUFFIXES = {".docx", ".xlsx", ".pdf"}
_MAX_TOOL_CHARACTERS = 50_000
_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_WORD_NAMESPACE}}}"
_BOUND_SOURCES: ContextVar[dict[str, Path] | None] = ContextVar(
    "deduction_rule_sources",
    default=None,
)


def bind_rule_sources(paths: Iterable[str | Path]) -> Token[dict[str, Path] | None]:
    """Bind explicit DOCX/XLSX paths for the current asynchronous execution.

    Args:
        paths: Existing Word or Excel files authorized by the caller.

    Returns:
        Context token that must be passed to :func:`reset_rule_sources`.

    Raises:
        RuleSourceError: If a path is missing, unsupported, temporary, or has a
            duplicate file name within the bound set.
    """

    sources: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise RuleSourceError(f"Rule source does not exist or is not a file: {path}")
        if path.name.startswith("~$"):
            raise RuleSourceError(f"Temporary Office files cannot be rule sources: {path.name}")
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise RuleSourceError(
                f"Only PDF, DOCX and XLSX rule sources are supported, got {path.suffix or '<none>'}"
            )
        if path.name in sources:
            raise RuleSourceError(f"Rule source names must be unique: {path.name}")
        sources[path.name] = path
    if not sources:
        raise RuleSourceError("At least one PDF, DOCX or XLSX rule source is required")
    return _BOUND_SOURCES.set(dict(sorted(sources.items())))


def reset_rule_sources(token: Token[dict[str, Path] | None]) -> None:
    """Restore the rule-source binding represented by ``token``.

    Args:
        token: Token returned by :func:`bind_rule_sources`.
    """

    _BOUND_SOURCES.reset(token)


def get_bound_source_path(source_name: str) -> Path:
    """Return the local path of one uploaded source bound for this turn (e.g. a schema sample).

    Args:
        source_name: File name as listed by :func:`list_rule_sources`.

    Raises:
        RuleSourceError: When no files are bound or the name is not among them.
    """

    sources = _require_sources()
    if source_name not in sources:
        raise RuleSourceError(f"Unknown uploaded file {source_name!r}; available: {', '.join(sources)}")
    return sources[source_name]


def list_rule_sources() -> list[dict[str, str | int]]:
    """Return safe metadata for every rule source bound by the caller.

    Returns:
        Alphabetically ordered source names, formats, and byte sizes. Absolute
        local paths are deliberately omitted from the model-visible result.
    """

    sources = _require_sources()
    return [
        {
            "source_name": name,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sources.items()
    ]


def read_rule_source(
    source_name: str,
    cursor: int = 0,
    max_characters: int = 20_000,
) -> dict[str, str | int | bool | None]:
    """Read one authorized rule source as traceable, paginated plain text.

    Args:
        source_name: Exact file name returned by :func:`list_rule_sources`.
        cursor: Character offset at which to start this chunk.
        max_characters: Maximum characters to return, capped at 50,000.

    Returns:
        Source name, content chunk, cursor metadata, and total character count.

    Raises:
        RuleSourceError: If the source is not bound or pagination is invalid.
    """

    sources = _require_sources()
    path = sources.get(source_name)
    if path is None:
        raise RuleSourceError(f"Rule source {source_name!r} is not bound by the caller")
    if cursor < 0:
        raise RuleSourceError("cursor must be zero or greater")
    if not 1 <= max_characters <= _MAX_TOOL_CHARACTERS:
        raise RuleSourceError(
            f"max_characters must be between 1 and {_MAX_TOOL_CHARACTERS}"
        )

    # ponytail: re-extracting keeps the binding stateless; add a per-run cache only
    # if multi-megabyte authoring workbooks make pagination measurably slow.
    content = _extract_source_text(path)
    if cursor > len(content):
        raise RuleSourceError(
            f"cursor {cursor} exceeds source length {len(content)} for {source_name}"
        )
    end = min(len(content), cursor + max_characters)
    has_more = end < len(content)
    return {
        "source_name": source_name,
        "content": content[cursor:end],
        "cursor": cursor,
        "next_cursor": end if has_more else None,
        "has_more": has_more,
        "total_characters": len(content),
    }


def build_rule_source_tools() -> list[BaseTool]:
    """Return the two LangChain tools used during deduction-skill authoring.

    Returns:
        ``list_rule_sources`` and ``read_rule_source`` tools backed by the
        execution-local source allowlist.
    """

    @tool
    def list_rule_sources_tool() -> list[dict[str, str | int]]:
        """List caller-authorized PDF/Word/Excel sources available for skill authoring."""

        try:
            return list_rule_sources()
        except RuleSourceError as exc:
            raise ToolException(str(exc)) from exc

    @tool
    def read_rule_source_tool(
        source_name: str,
        cursor: int = 0,
        max_characters: int = 20_000,
    ) -> dict[str, str | int | bool | None]:
        """Read an authorized PDF/Word/Excel source chunk with page/paragraph/cell locations."""

        try:
            return read_rule_source(source_name, cursor, max_characters)
        except RuleSourceError as exc:
            raise ToolException(str(exc)) from exc

    list_rule_sources_tool.name = "list_rule_sources"
    read_rule_source_tool.name = "read_rule_source"
    return [list_rule_sources_tool, read_rule_source_tool]


def _require_sources() -> dict[str, Path]:
    """Return the current source allowlist or reject calls outside an authoring run."""

    sources = _BOUND_SOURCES.get()
    if sources is None:
        raise RuleSourceError("No rule sources are bound for this execution")
    return sources


def _extract_source_text(path: Path) -> str:
    """Dispatch to the format-specific extractor based on the file suffix.

    The suffix is already validated against ``_SUPPORTED_SUFFIXES`` in
    :func:`bind_rule_sources`, so only ``.docx``/``.xlsx``/``.pdf`` reach here.
    """

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".xlsx":
        return _extract_xlsx(path)
    return _extract_pdf(path)


def _extract_pdf(path: Path) -> str:
    """Return PDF page text with per-page locators.

    ponytail: digital-text PDFs only. A scanned/image PDF yields no extractable
    text (no OCR); rather than hand the agent an empty string, raise a clear
    error so the caller learns the file is not machine-readable. Add an OCR
    fallback only if scanned rule documents become a real input.
    """

    try:
        reader = PdfReader(str(path))
        pages = list(reader.pages)
    except (OSError, ValueError, PdfReadError) as exc:
        raise RuleSourceError(f"Could not read PDF source {path.name}: {exc}") from exc

    lines: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            lines.append(f"[Page {page_number}]\n{text}")
    if not lines:
        raise RuleSourceError(
            f"No extractable text in PDF source {path.name} "
            "(scanned/image PDF? OCR is not supported)"
        )
    return "\n".join(lines)


def _extract_docx(path: Path) -> str:
    """Return DOCX body paragraphs and tables with stable human-readable locators."""

    try:
        with ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
    except (BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise RuleSourceError(f"Could not read DOCX source {path.name}: {exc}") from exc

    body = root.find(f"{_W}body")
    if body is None:
        raise RuleSourceError(f"DOCX source has no document body: {path.name}")

    lines: list[str] = []
    paragraph_number = 0
    table_number = 0
    for child in body:
        if child.tag == f"{_W}p":
            text = _word_text(child)
            if text:
                paragraph_number += 1
                lines.append(f"[Paragraph {paragraph_number}] {text}")
        elif child.tag == f"{_W}tbl":
            table_number += 1
            for row_number, row in enumerate(child.findall(f"{_W}tr"), start=1):
                cells = [_word_text(cell) for cell in row.findall(f"{_W}tc")]
                if any(cells):
                    lines.append(
                        f"[Table {table_number} row {row_number}] " + " | ".join(cells)
                    )
    return "\n".join(lines)


def _word_text(element: ElementTree.Element) -> str:
    """Join current Word text nodes while excluding deleted tracked-change text."""

    return " ".join(
        "".join(node.text or "" for node in paragraph.iter(f"{_W}t")).strip()
        for paragraph in element.iter(f"{_W}p")
        if "".join(node.text or "" for node in paragraph.iter(f"{_W}t")).strip()
    )


def _extract_xlsx(path: Path) -> str:
    """Return non-empty workbook cells with sheet names and cell coordinates."""

    try:
        workbook = load_workbook(
            filename=path,
            read_only=True,
            data_only=False,
            keep_links=False,
        )
    except (OSError, ValueError, BadZipFile) as exc:
        raise RuleSourceError(f"Could not read XLSX source {path.name}: {exc}") from exc

    lines: list[str] = []
    try:
        for sheet in workbook.worksheets:
            lines.append(f"[Sheet: {sheet.title}]")
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        lines.append(f"{cell.coordinate} = {cell.value}")
    finally:
        workbook.close()
    return "\n".join(lines)
