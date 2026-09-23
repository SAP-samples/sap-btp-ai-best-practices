"""Verify attachment blocks and portable YAML configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from template_agent.config import load_config
from template_agent.models import Attachment


def test_attachment_builds_standard_image_and_pdf_blocks(tmp_path: Path) -> None:
    """Encode supported files using LangChain standard content blocks."""

    image = tmp_path / "tiny.png"
    image.write_bytes(b"png")
    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    image_block = Attachment(path=image).to_content_block()
    pdf_block = Attachment(path=pdf).to_content_block()
    claude_pdf_block = Attachment(path=pdf).to_content_block("claude")

    assert image_block["type"] == "image"
    assert image_block["mime_type"] == "image/png"
    assert pdf_block["type"] == "file"
    assert pdf_block["filename"] == "tiny.pdf"
    assert claude_pdf_block["filename"] == "tiny"


def test_attachment_rejects_unsupported_file(tmp_path: Path) -> None:
    """Reject arbitrary files before model invocation."""

    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Only images and PDFs"):
        Attachment(path=path).to_content_block()


def test_config_resolves_paths_and_allows_disabled_missing_secret(tmp_path: Path) -> None:
    """Resolve local paths while leaving disabled optional secrets untouched."""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    config = config_dir / "agent.yaml"
    config.write_text(
        """
base_prompt: Test
model:
  provider: openai
  name: gpt-test
skills:
  directory: ../skills
mcp:
  servers:
    optional:
      enabled: false
      transport: sse
      url: https://example.test/sse
      headers:
        x-api-key: ${NOT_DEFINED_FOR_TEST}
memory:
  enabled: false
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.skills.directory == skill_dir.resolve()
    assert loaded.mcp.servers["optional"].headers["x-api-key"] == "${NOT_DEFINED_FOR_TEST}"
    assert loaded.a2a.host == "127.0.0.1"


def test_config_loads_and_normalizes_enabled_a2a_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve the public A2A URL while preserving explicit server metadata."""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / "skills").mkdir()
    monkeypatch.setenv("A2A_PUBLIC_URL", "https://agent.example.test/root/")
    config = config_dir / "agent.yaml"
    config.write_text(
        """
base_prompt: Test
model:
  provider: openai
  name: gpt-test
skills:
  directory: ../skills
a2a:
  enabled: true
  name: Test Agent
  description: Test description
  version: 2.3.4
  public_url: ${A2A_PUBLIC_URL}
  host: 127.0.0.1
  port: 9100
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.a2a.enabled is True
    assert loaded.a2a.public_url == "https://agent.example.test/root"
    assert loaded.a2a.name == "Test Agent"
    assert loaded.a2a.version == "2.3.4"
    assert loaded.a2a.host == "127.0.0.1"
    assert loaded.a2a.port == 9100


@pytest.mark.parametrize(
    ("a2a_yaml", "message"),
    [
        ("enabled: true", "public_url"),
        (
            "enabled: true\n  public_url: ftp://agent.example.test",
            "http:// or https://",
        ),
        (
            "enabled: true\n  public_url: https://agent.example.test\n  port: 70000",
            "less than or equal to 65535",
        ),
    ],
)
def test_config_rejects_invalid_enabled_a2a_settings(
    tmp_path: Path, a2a_yaml: str, message: str
) -> None:
    """Reject incomplete, unsafe, or out-of-range A2A server settings."""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / "skills").mkdir()
    config = config_dir / "agent.yaml"
    config.write_text(
        f"""
base_prompt: Test
model:
  provider: openai
  name: gpt-test
skills:
  directory: ../skills
a2a:
  {a2a_yaml}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config)
