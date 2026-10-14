"""Verify deterministic, safe, multi-skill progressive disclosure."""

from pathlib import Path

import pytest

from template_agent.skills import SkillError, SkillLoader, catalogue_prompt


def _skill(root: Path, name: str, description: str = "Useful skill") -> Path:
    """Create one valid test skill and return its directory."""

    directory = root / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return directory


def test_loads_multiple_skills_and_recursive_text_in_order(tmp_path: Path) -> None:
    """Load ordered unique skills with SKILL.md before nested references."""

    alpha = _skill(tmp_path, "alpha")
    references = alpha / "references"
    references.mkdir()
    (references / "z.md").write_text("nested-z", encoding="utf-8")
    (references / "a.md").write_text("nested-a", encoding="utf-8")
    (alpha / "asset.bin").write_bytes(b"\x00\xff")
    _skill(tmp_path, "beta")

    result = SkillLoader(tmp_path).load(["beta", "alpha", "beta"])

    assert result.index("SKILL: beta") < result.index("SKILL: alpha")
    assert result.count("SKILL: beta") == 1
    assert result.index("FILE: SKILL.md") < result.index("FILE: references/a.md")
    assert result.index("nested-a") < result.index("nested-z")
    assert "BINARY FILES OMITTED" in result
    assert "asset.bin" in result


def test_missing_skill_is_atomic(tmp_path: Path) -> None:
    """Reject the complete request rather than returning a partial skill set."""

    _skill(tmp_path, "present")
    with pytest.raises(SkillError, match="Unknown skill.*missing"):
        SkillLoader(tmp_path).load(["present", "missing"])


def test_rejects_folder_metadata_mismatch(tmp_path: Path) -> None:
    """Require the folder and metadata name to be identical."""

    directory = _skill(tmp_path, "wrong")
    (directory / "SKILL.md").write_text(
        "---\nname: other\ndescription: mismatch\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillError, match="must match"):
        SkillLoader(tmp_path).scan()


def test_rejects_malformed_metadata_and_traversal_names(tmp_path: Path) -> None:
    """Reject malformed SKILL.md files and path-like requested names."""

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    with pytest.raises(SkillError, match="Missing YAML frontmatter"):
        SkillLoader(tmp_path).scan()

    (malformed / "SKILL.md").write_text(
        "---\nname: malformed\ndescription: valid now\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillError, match=r"Unknown skill.*\.\./malformed"):
        SkillLoader(tmp_path).load(["../malformed"])


def test_rejects_symlinks_and_aggregate_overflow(tmp_path: Path) -> None:
    """Block path escapes and oversized prompt injection."""

    directory = _skill(tmp_path, "safe")
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    (directory / "linked.txt").symlink_to(target)
    with pytest.raises(SkillError, match="cannot be symlinks"):
        SkillLoader(tmp_path).load(["safe"])

    (directory / "linked.txt").unlink()
    with pytest.raises(SkillError, match="limit is 10"):
        SkillLoader(tmp_path, max_loaded_characters=10).load(["safe"])


def test_catalogue_prompt_advertises_batch_loading(tmp_path: Path) -> None:
    """Tell the model about available skills and the ordered-list tool syntax."""

    _skill(tmp_path, "alpha", "Alpha description")
    prompt = catalogue_prompt("Base", SkillLoader(tmp_path).scan())
    assert "alpha: Alpha description" in prompt
    assert 'load_skill(skill_names=["skill-a", "skill-b"])' in prompt
