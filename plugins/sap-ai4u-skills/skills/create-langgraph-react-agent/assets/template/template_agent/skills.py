"""Discover agent-local skills and expose their complete text through one tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field


class SkillError(ValueError):
    """Report an invalid, unsafe, missing, or oversized skill."""


@dataclass(frozen=True, slots=True)
class SkillSummary:
    """Store the catalogue metadata required by the system prompt."""

    name: str
    description: str
    directory: Path


class LoadSkillInput(BaseModel):
    """Validate the ordered skill names requested by the model."""

    skill_names: list[str] = Field(min_length=1)


class SkillLoader:
    """Scan one safe directory and concatenate selected skill text files."""

    def __init__(self, root: Path, max_loaded_characters: int = 200_000) -> None:
        """Initialize the loader.

        Args:
            root: Directory containing one folder per skill.
            max_loaded_characters: Maximum combined tool result size.
        """

        self.root = root.expanduser().resolve()
        self.max_loaded_characters = max_loaded_characters

    def scan(self) -> list[SkillSummary]:
        """Return validated skill metadata sorted by name."""

        if not self.root.exists():
            return []
        if not self.root.is_dir():
            raise SkillError(f"Skill root is not a directory: {self.root}")
        summaries: list[SkillSummary] = []
        for directory in sorted(self.root.iterdir(), key=lambda path: path.name):
            if directory.name.startswith("."):
                continue
            if directory.is_symlink():
                raise SkillError(f"Skill directories cannot be symlinks: {directory}")
            if not directory.is_dir():
                continue
            skill_file = directory / "SKILL.md"
            if not skill_file.is_file() or skill_file.is_symlink():
                raise SkillError(f"Skill {directory.name!r} must contain a regular SKILL.md")
            metadata = _frontmatter(skill_file)
            name = metadata.get("name")
            description = metadata.get("description")
            if name != directory.name:
                raise SkillError(
                    f"Skill folder {directory.name!r} must match metadata name {name!r}"
                )
            if not isinstance(description, str) or not description.strip():
                raise SkillError(f"Skill {name!r} must define a non-empty description")
            summaries.append(SkillSummary(name, description.strip(), directory))
        return summaries

    def load(self, skill_names: list[str]) -> str:
        """Concatenate one or more complete skills in requested order.

        Args:
            skill_names: Names discovered in the configured skill root.

        Returns:
            Text with explicit skill and file boundaries. Binary files are named
            but omitted from prompt content.
        """

        requested = list(dict.fromkeys(skill_names))
        if not requested:
            raise SkillError("At least one skill name is required")
        available = {summary.name: summary for summary in self.scan()}
        missing = [name for name in requested if name not in available]
        if missing:
            raise SkillError(f"Unknown skill(s): {', '.join(missing)}")

        sections = [self._load_one(available[name]) for name in requested]
        result = "\n\n".join(sections)
        if len(result) > self.max_loaded_characters:
            raise SkillError(
                f"Requested skills contain {len(result)} characters; "
                f"limit is {self.max_loaded_characters}"
            )
        return result

    def tool(self) -> BaseTool:
        """Return a structured LangChain tool backed by this loader."""

        def invoke(skill_names: list[str]) -> str:
            """Load several available skills, including all nested text files."""

            try:
                return self.load(skill_names)
            except SkillError as exc:
                raise ToolException(str(exc)) from exc

        return StructuredTool.from_function(
            func=invoke,
            name="load_skill",
            description=(
                "Load one or more available skills in a single call. Pass an ordered "
                "list of names; each result includes SKILL.md and every nested text file."
            ),
            args_schema=LoadSkillInput,
            handle_tool_error=True,
        )

    def _load_one(self, summary: SkillSummary) -> str:
        """Return one validated skill with deterministic recursive file ordering."""

        files: list[Path] = []
        for path in summary.directory.rglob("*"):
            relative = path.relative_to(summary.directory)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink():
                raise SkillError(f"Skill files cannot be symlinks: {path}")
            if path.is_file():
                files.append(path)
        skill_file = summary.directory / "SKILL.md"
        ordered = [skill_file, *sorted((path for path in files if path != skill_file), key=lambda path: path.relative_to(summary.directory).as_posix())]
        sections = [f"===== SKILL: {summary.name} ====="]
        omitted: list[str] = []
        for path in ordered:
            relative = path.relative_to(summary.directory).as_posix()
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                omitted.append(relative)
                continue
            sections.append(f"----- FILE: {relative} -----\n{content.rstrip()}")
        if omitted:
            sections.append(
                "----- BINARY FILES OMITTED -----\n" + "\n".join(f"- {path}" for path in omitted)
            )
        return "\n\n".join(sections)


def catalogue_prompt(base_prompt: str, summaries: list[SkillSummary]) -> str:
    """Append the current skill catalogue and loading instructions to a prompt."""

    if not summaries:
        catalogue = "No agent-local skills are currently available."
    else:
        lines = [f"- {item.name}: {item.description}" for item in summaries]
        catalogue = "\n".join(lines)
    return (
        f"{base_prompt.rstrip()}\n\n"
        "Available agent-local skills:\n"
        f"{catalogue}\n\n"
        "When detailed skill guidance is useful, call "
        '`load_skill(skill_names=["skill-a", "skill-b"])`. '
        "Load all needed skills together when possible."
    )


def _frontmatter(path: Path) -> dict[str, object]:
    """Parse and return YAML frontmatter from one SKILL.md."""

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SkillError(f"Missing YAML frontmatter in {path}")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SkillError(f"Unterminated YAML frontmatter in {path}")
    metadata = yaml.safe_load(text[4:end])
    if not isinstance(metadata, dict):
        raise SkillError(f"Frontmatter must be a mapping in {path}")
    return metadata
