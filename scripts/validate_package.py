"""Validate every portable plugin package and both marketplace adapters.

Examples:
    python scripts/validate_package.py
    python scripts/validate_package.py --repository-root /path/to/marketplace

The command is read-only. It prints a progress bar while checking skills and
returns a non-zero exit code when the package is unsafe or inconsistent.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import yaml

PORTABLE_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
PORTABLE_FIELDS = {
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
}
SKILL_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
PLUGIN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,62}[a-z0-9]$|^[a-z0-9]$")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
FORBIDDEN_FILE_NAMES = {".DS_Store", "__pycache__"}
FORBIDDEN_TEXT = {
    "/Users/": "user-specific absolute path",
    "ghp_": "GitHub personal access token prefix",
    "github_pat_": "GitHub fine-grained token prefix",
    "[TODO:": "unfinished placeholder",
}
IGNORED_SCAN_PARTS = {".git", ".venv", ".pytest_cache", ".mypy_cache"}


def _load_json(path: Path, errors: list[str]) -> dict[str, object] | None:
    """Load one JSON object from ``path`` and append parse errors to ``errors``.

    The parsed mapping is returned on success. Missing, invalid, or non-object
    documents return ``None`` so callers can continue reporting other issues.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path}: cannot load JSON: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: root value must be a JSON object")
        return None
    return value


def _validate_plugin_name(name: object, label: str, errors: list[str]) -> str | None:
    """Validate a portable plugin identifier and return its string value.

    ``name`` may be any parsed JSON value. Invalid identifiers append a
    labelled error and return ``None``; valid identifiers are returned.
    """

    if not isinstance(name, str) or not PLUGIN_NAME_PATTERN.fullmatch(name):
        errors.append(f"{label}: invalid plugin name {name!r}")
        return None
    if "--" in name or ".." in name:
        errors.append(f"{label}: plugin name cannot contain '--' or '..'")
        return None
    return name


def _validate_semver(version: object, label: str, errors: list[str]) -> str | None:
    """Validate a strict semantic version and return it when valid.

    ``version`` is the parsed manifest value. Invalid values are recorded in
    ``errors`` and result in ``None``.
    """

    if not isinstance(version, str) or not SEMVER_PATTERN.fullmatch(version):
        errors.append(f"{label}: version must use semantic versioning")
        return None
    return version


def _validate_portable_manifest(
    manifest: dict[str, object], path: Path, errors: list[str]
) -> None:
    """Validate Agent Plugins 1.0 fields in the portable manifest.

    ``manifest`` is the parsed root object and ``path`` identifies it in error
    messages. All violations are appended to ``errors``; the function returns
    no value.
    """

    unknown_fields = sorted(set(manifest) - PORTABLE_FIELDS)
    if unknown_fields:
        errors.append(f"{path}: unsupported portable fields {unknown_fields}")
    if manifest.get("$schema") != PORTABLE_SCHEMA:
        errors.append(f"{path}: $schema must be {PORTABLE_SCHEMA}")
    _validate_plugin_name(manifest.get("name"), str(path), errors)
    _validate_semver(manifest.get("version"), str(path), errors)
    description = manifest.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{path}: description must be a non-empty string")
    author = manifest.get("author")
    if (
        not isinstance(author, dict)
        or not isinstance(author.get("name"), str)
        or not author["name"].strip()
    ):
        errors.append(f"{path}: author.name must be a non-empty string")
    elif any(
        key not in {"name", "email", "url"} or not isinstance(value, str)
        for key, value in author.items()
    ):
        errors.append(f"{path}: author may contain only string name, email, and url fields")
    for field in ("homepage", "repository", "license"):
        value = manifest.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{path}: {field} must be a non-empty string")
    keywords = manifest.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list)
        or not all(isinstance(value, str) and value.strip() for value in keywords)
    ):
        errors.append(f"{path}: keywords must be an array of non-empty strings")
    extensions = manifest.get("extensions")
    if extensions is not None and not isinstance(extensions, dict):
        errors.append(f"{path}: extensions must be an object")


def _validate_manifest_consistency(
    plugin_root: Path,
    manifests: Sequence[tuple[Path, dict[str, object]]],
    errors: list[str],
) -> str | None:
    """Check shared identity fields across portable and native manifests.

    ``plugin_root`` supplies the required folder name. ``manifests`` contains
    path/document pairs. The common plugin name is returned when available;
    mismatches are appended to ``errors``.
    """

    shared_fields = ("name", "version", "description")
    for field in shared_fields:
        values = [document.get(field) for _, document in manifests]
        if any(value != values[0] for value in values[1:]):
            details = ", ".join(f"{path}: {document.get(field)!r}" for path, document in manifests)
            errors.append(f"manifest {field} mismatch: {details}")
    author_names = [
        document.get("author", {}).get("name")
        if isinstance(document.get("author"), dict)
        else None
        for _, document in manifests
    ]
    if any(value != author_names[0] for value in author_names[1:]):
        errors.append("manifest author.name values must match")
    plugin_name = manifests[0][1].get("name") if manifests else None
    if isinstance(plugin_name, str) and plugin_root.name != plugin_name:
        errors.append(
            f"plugin folder {plugin_root.name!r} must match manifest name {plugin_name!r}"
        )
    return plugin_name if isinstance(plugin_name, str) else None


def _validate_marketplace_path(
    value: object, expected: str, label: str, errors: list[str]
) -> None:
    """Validate one marketplace-relative plugin source path.

    ``value`` is the parsed path, ``expected`` is the exact package location,
    and ``label`` identifies the client. Violations are appended to ``errors``.
    """

    if not isinstance(value, str) or not value.startswith("./"):
        errors.append(f"{label}: source path must begin with './'")
        return
    if ".." in PurePosixPath(value).parts:
        errors.append(f"{label}: source path cannot traverse outside the marketplace")
    if value != expected:
        errors.append(f"{label}: source path must be {expected!r}, got {value!r}")


def _validate_codex_entry(entry: dict[str, object], errors: list[str]) -> None:
    """Validate the source, policy, and category of one Codex marketplace entry.

    ``entry`` is a parsed plugin entry whose name is already a string. All
    violations are appended to ``errors``; the function returns no value.
    """

    expected_path = f"./plugins/{entry['name']}"
    label = f"Codex marketplace ({entry['name']})"
    source = entry.get("source")
    if not isinstance(source, dict) or source.get("source") != "local":
        errors.append(f"{label}: source must use the local type")
    else:
        _validate_marketplace_path(source.get("path"), expected_path, label, errors)
    policy = entry.get("policy")
    if not isinstance(policy, dict):
        errors.append(f"{label}: policy must be an object")
    else:
        if policy.get("installation") != "AVAILABLE":
            errors.append(f"{label}: installation policy must be AVAILABLE")
        if policy.get("authentication") != "ON_INSTALL":
            errors.append(f"{label}: authentication policy must be ON_INSTALL")
    if not isinstance(entry.get("category"), str) or not entry["category"].strip():
        errors.append(f"{label}: category must be non-empty")


def _marketplace_plugin_names(
    document: dict[str, object], label: str, errors: list[str]
) -> list[str]:
    """Return the plugin entry names of one marketplace in listed order.

    ``document`` is the parsed marketplace and ``label`` identifies the client
    in messages. Malformed, empty, or duplicate entries are appended to
    ``errors`` and skipped.
    """

    entries = document.get("plugins")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label} must contain at least one plugin entry")
        return []
    names: list[str] = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            errors.append(f"{label}: every plugin entry needs a string name")
        elif name in names:
            errors.append(f"{label}: duplicate plugin entry {name!r}")
        else:
            names.append(name)
    return names


def _validate_marketplaces(
    codex_document: dict[str, object],
    claude_document: dict[str, object],
    errors: list[str],
) -> list[str]:
    """Validate Codex and Claude marketplace identity, sources, and policy.

    Both parsed marketplace documents must list the same plugins, each sourced
    from ``./plugins/<name>``. Problems are appended to ``errors``. The plugin
    names listed by the Claude marketplace are returned for package checks.
    """

    marketplace_name = codex_document.get("name")
    if not isinstance(marketplace_name, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", marketplace_name
    ):
        errors.append("marketplace name must contain only letters, digits, underscores, or hyphens")
    if marketplace_name != claude_document.get("name"):
        errors.append("Codex and Claude marketplace names must match")
    codex_names = _marketplace_plugin_names(codex_document, "Codex marketplace", errors)
    claude_names = _marketplace_plugin_names(claude_document, "Claude marketplace", errors)
    if set(codex_names) != set(claude_names):
        errors.append(
            f"Codex {sorted(codex_names)} and Claude {sorted(claude_names)} "
            "marketplaces must list the same plugins"
        )
    for entry in codex_document.get("plugins") or []:
        if isinstance(entry, dict) and entry.get("name") in codex_names:
            _validate_codex_entry(entry, errors)
    for entry in claude_document.get("plugins") or []:
        if isinstance(entry, dict) and entry.get("name") in claude_names:
            _validate_marketplace_path(
                entry.get("source"),
                f"./plugins/{entry['name']}",
                f"Claude marketplace ({entry['name']})",
                errors,
            )
    return claude_names


def _read_frontmatter(path: Path, errors: list[str]) -> dict[str, object] | None:
    """Parse YAML frontmatter from one SKILL.md file.

    ``path`` is read as UTF-8. Parse and structure errors are appended to
    ``errors``. The YAML mapping is returned on success, otherwise ``None``.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"{path}: cannot read skill: {error}")
        return None
    if not lines or lines[0] != "---":
        errors.append(f"{path}: must start with YAML frontmatter")
        return None
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        errors.append(f"{path}: frontmatter is not closed")
        return None
    try:
        value = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        errors.append(f"{path}: invalid YAML frontmatter: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: frontmatter must be a mapping")
        return None
    if not any(line.strip() for line in lines[closing_index + 1 :]):
        errors.append(f"{path}: skill instructions cannot be empty")
    return value


def _render_progress(index: int, total: int, label: str) -> None:
    """Print one stable progress-bar line for the current skill.

    ``index`` is one-based, ``total`` is the number of skills, and ``label``
    is the skill name. The function writes to standard output and returns no
    value.
    """

    width = 20
    completed = width if total == 0 else round(width * index / total)
    bar = "#" * completed + "-" * (width - completed)
    print(f"[{bar}] {index}/{total} {label}")


def _validate_skills(
    skills_root: Path, errors: list[str], warnings: list[str]
) -> int:
    """Validate every immediate Agent Skill and return the discovered count.

    ``skills_root`` must contain one directory per skill. Structural failures
    are appended to ``errors``; non-fatal size recommendations go to
    ``warnings``. A visible progress bar is printed during the scan.
    """

    if not skills_root.is_dir():
        errors.append(f"{skills_root}: skills directory is missing")
        return 0
    skill_roots = sorted(
        path for path in skills_root.iterdir() if path.is_dir() and not path.is_symlink()
    )
    if not skill_roots:
        errors.append(f"{skills_root}: no skills found")
        return 0
    for index, skill_root in enumerate(skill_roots, start=1):
        _render_progress(index, len(skill_roots), skill_root.name)
        skill_path = skill_root / "SKILL.md"
        frontmatter = _read_frontmatter(skill_path, errors)
        if frontmatter is None:
            continue
        unknown_fields = sorted(set(frontmatter) - SKILL_FIELDS)
        if unknown_fields:
            errors.append(f"{skill_path}: unsupported frontmatter fields {unknown_fields}")
        name = frontmatter.get("name")
        if not isinstance(name, str) or not SKILL_NAME_PATTERN.fullmatch(name):
            errors.append(f"{skill_path}: invalid skill name {name!r}")
        elif name != skill_root.name:
            errors.append(
                f"{skill_path}: name {name!r} must match directory {skill_root.name!r}"
            )
        description = frontmatter.get("description")
        if not isinstance(description, str) or not (1 <= len(description) <= 1024):
            errors.append(f"{skill_path}: description must contain 1-1024 characters")
        compatibility = frontmatter.get("compatibility")
        if compatibility is not None and (
            not isinstance(compatibility, str) or len(compatibility) > 500
        ):
            errors.append(f"{skill_path}: compatibility must contain at most 500 characters")
        line_count = len(skill_path.read_text(encoding="utf-8").splitlines())
        if line_count > 500:
            warnings.append(
                f"{skill_path}: {line_count} lines; Agent Skills recommends at most 500"
            )
    return len(skill_roots)


def _scan_package_files(package_root: Path, errors: list[str]) -> None:
    """Scan repository files for unsafe paths, artifacts, symlinks, and tokens.

    ``package_root`` bounds the distributed plugin scan. Findings are appended to ``errors``;
    binary files are ignored after filename and symlink checks.
    """

    for path in package_root.rglob("*"):
        relative_parts = path.relative_to(package_root).parts
        if any(part in IGNORED_SCAN_PARTS for part in relative_parts):
            continue
        if path.name in FORBIDDEN_FILE_NAMES or path.suffix == ".pyc":
            errors.append(f"{path}: generated local artifact must not be packaged")
            continue
        if path.is_symlink():
            errors.append(f"{path}: symlinks are not allowed in the package repository")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as error:
            errors.append(f"{path}: cannot scan file: {error}")
            continue
        for marker, reason in FORBIDDEN_TEXT.items():
            if marker in text:
                errors.append(f"{path}: contains {reason}")


def _validate_plugin(
    plugin_root: Path, errors: list[str], warnings: list[str]
) -> int:
    """Validate one plugin package: manifests, skills, and packaged files.

    ``plugin_root`` is ``plugins/<name>``. Errors and warnings are appended to
    the given lists. The number of discovered skills is returned.
    """

    print(f"Plugin {plugin_root.name}")
    portable_path = plugin_root / "plugin.json"
    codex_path = plugin_root / ".codex-plugin" / "plugin.json"
    claude_path = plugin_root / ".claude-plugin" / "plugin.json"
    portable = _load_json(portable_path, errors)
    codex = _load_json(codex_path, errors)
    claude = _load_json(claude_path, errors)
    if portable is not None:
        _validate_portable_manifest(portable, portable_path, errors)
    manifests = [
        (path, document)
        for path, document in (
            (portable_path, portable),
            (codex_path, codex),
            (claude_path, claude),
        )
        if document is not None
    ]
    if len(manifests) == 3:
        _validate_manifest_consistency(plugin_root, manifests, errors)
    skill_count = _validate_skills(plugin_root / "skills", errors, warnings)
    _scan_package_files(plugin_root, errors)
    return skill_count


def validate_repository(repository_root: Path) -> tuple[list[str], list[str], int]:
    """Validate the complete marketplace repository without modifying it.

    ``repository_root`` identifies the checkout. Every plugin listed in the
    marketplaces is validated, and every folder under ``plugins/`` must be
    listed. The return value contains errors, warnings, and the total number
    of discovered skills in that order.
    """

    errors: list[str] = []
    warnings: list[str] = []
    plugins_root = repository_root / "plugins"
    codex_marketplace_path = repository_root / ".agents" / "plugins" / "marketplace.json"
    claude_marketplace_path = repository_root / ".claude-plugin" / "marketplace.json"
    codex_marketplace = _load_json(codex_marketplace_path, errors)
    claude_marketplace = _load_json(claude_marketplace_path, errors)
    if codex_marketplace is None or claude_marketplace is None:
        return errors, warnings, 0

    plugin_names = _validate_marketplaces(codex_marketplace, claude_marketplace, errors)
    folders = {path.name for path in plugins_root.iterdir() if path.is_dir()}
    for unlisted in sorted(folders - set(plugin_names)):
        errors.append(f"{plugins_root / unlisted}: plugin folder is not listed in the marketplaces")
    skill_count = 0
    for name in plugin_names:
        if name not in folders:
            errors.append(f"{plugins_root / name}: listed plugin folder is missing")
            continue
        skill_count += _validate_plugin(plugins_root / name, errors, warnings)
    return errors, warnings, skill_count


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments from ``argv`` and return the argparse namespace.

    When ``argv`` is ``None``, arguments are read from the current process.
    """

    parser = argparse.ArgumentParser(
        description="Validate portable, Codex, and Claude plugin packaging."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Marketplace repository root (defaults to the script's parent repository).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run repository validation and return a process-compatible exit code.

    ``argv`` optionally supplies CLI arguments for tests. The function prints
    warnings and errors, returning zero only when no errors are found.
    """

    args = _parse_args(argv)
    repository_root = args.repository_root.resolve()
    errors, warnings, skill_count = validate_repository(repository_root)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(
            f"Validation failed: {len(errors)} error(s), "
            f"{len(warnings)} warning(s), {skill_count} skill(s)"
        )
        return 1
    print(
        f"Validation passed: {skill_count} skill(s), "
        f"{len(warnings)} recommendation warning(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
