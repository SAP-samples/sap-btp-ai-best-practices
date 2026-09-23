"""Small public data models shared by the runtime and CLI."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, field_validator


class Attachment(BaseModel):
    """Describe one local image or PDF supplied with a user message.

    Attributes:
        path: Existing local file to send to the configured model.
        mime_type: Optional explicit MIME type; otherwise inferred from the name.
    """

    path: Path
    mime_type: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        """Return a resolved file path or reject a missing/non-file attachment."""

        path = value.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Attachment does not exist or is not a file: {path}")
        return path

    def resolved_mime_type(self) -> str:
        """Return and validate the image or PDF MIME type for this attachment."""

        mime_type = self.mime_type or mimetypes.guess_type(self.path.name)[0]
        if not mime_type or not (
            mime_type.startswith("image/") or mime_type == "application/pdf"
        ):
            raise ValueError(
                f"Only images and PDFs are supported, got {mime_type!r} for {self.path.name}"
            )
        return mime_type

    def to_content_block(
        self,
        provider: Literal["openai", "gemini", "claude"] | None = None,
    ) -> dict[str, Any]:
        """Return a LangChain standard image or file content block.

        Args:
            provider: Optional configured provider. Claude receives a Bedrock-
                compatible PDF document name; other providers retain the file name.

        Returns:
            A standard block that the provider's LangChain integration translates
            to its native request format.
        """

        mime_type = self.resolved_mime_type()
        encoded = base64.b64encode(self.path.read_bytes()).decode("ascii")
        if mime_type.startswith("image/"):
            return {"type": "image", "base64": encoded, "mime_type": mime_type}
        filename = (
            _bedrock_document_name(self.path.stem)
            if provider == "claude"
            else self.path.name
        )
        return {
            "type": "file",
            "base64": encoded,
            "mime_type": mime_type,
            "filename": filename,
        }


def _bedrock_document_name(value: str) -> str:
    """Return a non-empty PDF name accepted by Bedrock Converse."""

    normalized = re.sub(r"[^A-Za-z0-9()\[\] -]", "-", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "document"


class AgentResult(BaseModel):
    """Contain the final text, optional structured value, and invocation trace."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_text: str
    output_parsed: Any | None = None
    messages: list[BaseMessage]
