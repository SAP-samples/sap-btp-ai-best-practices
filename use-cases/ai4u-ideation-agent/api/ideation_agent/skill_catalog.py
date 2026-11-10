"""Skill catalog utilities for dynamic skill discovery and loading.

This module scans the local ``skills/`` directory for ``SKILL.md`` files and
extracts lightweight summaries from YAML-style frontmatter. It also provides
helpers to load a full skill bundle for tool-based retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


_MODULE_DIR = Path(__file__).resolve().parent
_SKILLS_DIR = _MODULE_DIR / "skills"
_SKILL_FILE_NAME = "SKILL.md"
_COMPANION_SUFFIXES = {".md", ".txt"}


def _split_frontmatter_and_body(text: str) -> Tuple[Dict[str, str], str]:
    """Split Markdown content into simple frontmatter key-value pairs and body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    frontmatter: Dict[str, str] = {}
    for raw_line in lines[1:end_index]:
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if (
            len(value) >= 2
            and ((value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")))
        ):
            value = value[1:-1]
        frontmatter[key.strip()] = value

    body = "\n".join(lines[end_index + 1 :])
    return frontmatter, body


def _first_meaningful_body_line(text: str) -> str:
    """Return the first non-empty non-heading line from Markdown body text."""
    in_code_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not line or line.startswith("#"):
            continue
        return line
    return ""


def scan_skill_summaries() -> List[Dict[str, str]]:
    """Scan skill folders and return sorted skill summaries.

    Returns:
        List of dictionaries containing ``id``, ``name``, and ``description``.
    """
    if not _SKILLS_DIR.exists():
        return []

    summaries: List[Dict[str, str]] = []
    for skill_dir in sorted((path for path in _SKILLS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name):
        skill_path = skill_dir / _SKILL_FILE_NAME
        if not skill_path.is_file():
            continue

        text = skill_path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter_and_body(text)
        skill_id = skill_dir.name
        name = frontmatter.get("name", "").strip() or skill_id
        description = frontmatter.get("description", "").strip()
        if not description:
            description = _first_meaningful_body_line(body) or f"Instructions for {name}."

        summaries.append(
            {
                "id": skill_id,
                "name": name,
                "description": " ".join(description.split()),
            }
        )

    return summaries


def _available_skill_ids() -> List[str]:
    """Return all available skill IDs in deterministic order."""
    return [entry["id"] for entry in scan_skill_summaries()]


def load_skill_bundle(skill_id: str) -> Dict[str, object]:
    """Load one skill's full instruction bundle.

    The bundle always includes ``SKILL.md`` first. Then any top-level sibling
    ``.md``/``.txt`` files are appended in alphabetical order.
    """
    normalized_id = str(skill_id).strip()
    if not normalized_id:
        raise ValueError("load_skill: 'skill_name' is required")

    skill_dir = _SKILLS_DIR / normalized_id
    skill_path = skill_dir / _SKILL_FILE_NAME
    if not skill_dir.is_dir() or not skill_path.is_file():
        available = _available_skill_ids()
        if available:
            raise ValueError(
                f"Unknown skill '{normalized_id}'. Available skills: {', '.join(available)}"
            )
        raise ValueError(f"Unknown skill '{normalized_id}'. No skills are currently available")

    file_paths = [skill_path]
    for child in sorted(skill_dir.iterdir(), key=lambda path: path.name):
        if not child.is_file():
            continue
        if child.name == _SKILL_FILE_NAME:
            continue
        if child.suffix.lower() not in _COMPANION_SUFFIXES:
            continue
        file_paths.append(child)

    files: List[str] = []
    parts: List[str] = []
    for file_path in file_paths:
        rel_path = file_path.relative_to(_MODULE_DIR).as_posix()
        files.append(rel_path)
        text = file_path.read_text(encoding="utf-8").strip()
        parts.append(f"### {rel_path}\n{text}")

    return {
        "skill_id": normalized_id,
        "files": files,
        "content": "\n\n".join(parts).strip(),
    }
