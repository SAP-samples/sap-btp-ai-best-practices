"""Verify attachment blocks and portable YAML configuration.

Converted from the scaffold's pytest-based test to stdlib unittest so it runs
under ``unittest discover`` without requiring pytest.
"""

import tempfile
import unittest
from pathlib import Path

from app.deduction_agent.template_agent.config import load_config
from app.deduction_agent.template_agent.models import Attachment


class TestAttachmentBlocks(unittest.TestCase):
    """Test attachment content-block building for images and PDFs."""

    def test_attachment_builds_standard_image_and_pdf_blocks(self) -> None:
        """Encode supported files using LangChain standard content blocks."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image = tmp_path / "tiny.png"
            image.write_bytes(b"png")
            pdf = tmp_path / "tiny.pdf"
            pdf.write_bytes(b"%PDF-1.4")

            image_block = Attachment(path=image).to_content_block()
            pdf_block = Attachment(path=pdf).to_content_block()
            claude_pdf_block = Attachment(path=pdf).to_content_block("claude")

            self.assertEqual(image_block["type"], "image")
            self.assertEqual(image_block["mime_type"], "image/png")
            self.assertEqual(pdf_block["type"], "file")
            self.assertEqual(pdf_block["filename"], "tiny.pdf")
            self.assertEqual(claude_pdf_block["filename"], "tiny")

    def test_attachment_rejects_unsupported_file(self) -> None:
        """Reject arbitrary files before model invocation."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "note.txt"
            path.write_text("hello", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Only images and PDFs"):
                Attachment(path=path).to_content_block()


class TestAgentConfig(unittest.TestCase):
    """Test YAML config loading and path resolution."""

    def test_config_resolves_paths_and_allows_disabled_missing_secret(self) -> None:
        """Resolve local paths while leaving disabled optional secrets untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
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

            self.assertEqual(loaded.skills.directory, skill_dir.resolve())
            self.assertEqual(
                loaded.mcp.servers["optional"].headers["x-api-key"],
                "${NOT_DEFINED_FOR_TEST}",
            )


if __name__ == "__main__":
    unittest.main()
