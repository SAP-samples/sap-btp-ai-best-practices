"""Verify deterministic, safe, multi-skill progressive disclosure.

Converted from the scaffold's pytest-based test to stdlib unittest so it runs
under ``unittest discover`` without requiring pytest.
"""

import tempfile
import unittest
from pathlib import Path

from app.deduction_agent.template_agent.skills import SkillError, SkillLoader, catalogue_prompt


def _skill(root: Path, name: str, description: str = "Useful skill") -> Path:
    """Create one valid test skill and return its directory."""
    directory = root / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return directory


class TestSkillLoader(unittest.TestCase):
    """Test SkillLoader discovery, ordering, and safety invariants."""

    def test_loads_multiple_skills_and_recursive_text_in_order(self) -> None:
        """Load ordered unique skills with SKILL.md before nested references."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            alpha = _skill(tmp_path, "alpha")
            references = alpha / "references"
            references.mkdir()
            (references / "z.md").write_text("nested-z", encoding="utf-8")
            (references / "a.md").write_text("nested-a", encoding="utf-8")
            (alpha / "asset.bin").write_bytes(b"\x00\xff")
            _skill(tmp_path, "beta")

            result = SkillLoader(tmp_path).load(["beta", "alpha", "beta"])

            self.assertLess(result.index("SKILL: beta"), result.index("SKILL: alpha"))
            self.assertEqual(result.count("SKILL: beta"), 1)
            self.assertLess(result.index("FILE: SKILL.md"), result.index("FILE: references/a.md"))
            self.assertLess(result.index("nested-a"), result.index("nested-z"))
            self.assertIn("BINARY FILES OMITTED", result)
            self.assertIn("asset.bin", result)

    def test_missing_skill_is_atomic(self) -> None:
        """Reject the complete request rather than returning a partial skill set."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _skill(tmp_path, "present")
            with self.assertRaisesRegex(SkillError, "Unknown skill.*missing"):
                SkillLoader(tmp_path).load(["present", "missing"])

    def test_rejects_folder_metadata_mismatch(self) -> None:
        """Require the folder and metadata name to be identical."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            directory = _skill(tmp_path, "wrong")
            (directory / "SKILL.md").write_text(
                "---\nname: other\ndescription: mismatch\n---\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillError, "must match"):
                SkillLoader(tmp_path).scan()

    def test_rejects_malformed_metadata_and_traversal_names(self) -> None:
        """Reject malformed SKILL.md files and path-like requested names."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            malformed = tmp_path / "malformed"
            malformed.mkdir()
            (malformed / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
            with self.assertRaisesRegex(SkillError, "Missing YAML frontmatter"):
                SkillLoader(tmp_path).scan()

            (malformed / "SKILL.md").write_text(
                "---\nname: malformed\ndescription: valid now\n---\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SkillError, r"Unknown skill.*\.\./malformed"):
                SkillLoader(tmp_path).load(["../malformed"])

    def test_rejects_symlinks_and_aggregate_overflow(self) -> None:
        """Block path escapes and oversized prompt injection."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            directory = _skill(tmp_path, "safe")
            target = tmp_path / "outside.txt"
            target.write_text("outside", encoding="utf-8")
            (directory / "linked.txt").symlink_to(target)
            with self.assertRaisesRegex(SkillError, "cannot be symlinks"):
                SkillLoader(tmp_path).load(["safe"])

            (directory / "linked.txt").unlink()
            with self.assertRaisesRegex(SkillError, "limit is 10"):
                SkillLoader(tmp_path, max_loaded_characters=10).load(["safe"])

    def test_catalogue_prompt_advertises_batch_loading(self) -> None:
        """Tell the model about available skills and the ordered-list tool syntax."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _skill(tmp_path, "alpha", "Alpha description")
            prompt = catalogue_prompt("Base", SkillLoader(tmp_path).scan())
            self.assertIn("alpha: Alpha description", prompt)
            self.assertIn('load_skill(skill_names=["skill-a", "skill-b"])', prompt)


if __name__ == "__main__":
    unittest.main()
