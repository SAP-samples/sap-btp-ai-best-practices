from pathlib import Path
from typing import Dict, List, TypedDict
import os


class PromptConfig(TypedDict):
    label: str
    filename: str


PROMPTS_DIR = Path(__file__).parent / "prompts"


def _scan_registry() -> Dict[str, PromptConfig]:
    """Dynamically scan the prompts directory."""
    registry = {}
    if not PROMPTS_DIR.exists():
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in PROMPTS_DIR.glob("*.md"):
        key = file_path.stem
        # Basic label generation: nc-rates -> NC Rates
        # You could enhance this to look for a frontmatter title if you wanted
        label = key.replace("-", " ").title()

        registry[key] = {"label": label, "filename": file_path.name}
    return registry


def get_available_types() -> List[Dict[str, str]]:
    """Return list of available file types for UI."""
    registry = _scan_registry()
    # Sort by label
    items = [{"key": k, "label": v["label"]} for k, v in registry.items()]
    return sorted(items, key=lambda x: x["label"])


def get_prompt_content(key: str) -> str:
    """Load prompt content by key."""
    registry = _scan_registry()

    filename = None
    if key in registry:
        filename = registry[key]["filename"]
    elif (PROMPTS_DIR / key).exists():
        filename = key
    elif (PROMPTS_DIR / f"{key}.md").exists():
        filename = f"{key}.md"

    if not filename:
        raise ValueError(f"Unknown prompt type: {key}")

    file_path = PROMPTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {filename}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_prompt_content(filename: str, content: str) -> None:
    """Save prompt content to file. Filename should include extension or be key."""
    if not filename.endswith(".md"):
        filename += ".md"

    # Simple security check to prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename")

    file_path = PROMPTS_DIR / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def delete_prompt(filename: str) -> None:
    """Delete prompt file."""
    if not filename.endswith(".md"):
        filename += ".md"

    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename")

    file_path = PROMPTS_DIR / filename
    if file_path.exists():
        os.remove(file_path)
    else:
        raise FileNotFoundError(f"Prompt file not found: {filename}")
