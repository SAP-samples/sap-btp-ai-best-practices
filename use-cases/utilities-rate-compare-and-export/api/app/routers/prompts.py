import logging
import re
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict

from ..security import get_api_key
from ..utils.prompt_registry import (
    get_available_types,
    get_prompt_content,
    save_prompt_content,
    delete_prompt,
    _scan_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# --- Pydantic Models ---


class MapRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str = ""
    key: str = ""
    desc: str = ""
    class_: str = Field("", alias="class", serialization_alias="class")
    unit: str = ""


class MapGroup(BaseModel):
    name: str
    rows: List[MapRow]
    notes: str = ""


class PromptStructure(BaseModel):
    meta: str
    map_data: List[MapGroup] = []
    execution: str
    is_structured: bool = True
    raw_content: Optional[str] = None  # For fallback if not structured


class PromptMetadata(BaseModel):
    key: str
    label: str
    filename: str


class CreatePromptRequest(BaseModel):
    filename: str
    structure: PromptStructure


class UpdatePromptRequest(BaseModel):
    structure: PromptStructure


# --- Helper Functions ---


def parse_markdown_to_structure(content: str) -> PromptStructure:
    """Parses the MD content into meta, map_data (groups/rows), and execution."""

    # 1. Split into main sections
    # Pattern looks for "## **3. Master Extraction Map**"
    # We use flexible regex for spaces
    parts = re.split(
        r"##\s+\*\*3\.\s+Master Extraction Map\*\*", content, flags=re.IGNORECASE
    )

    if len(parts) < 2:
        return PromptStructure(
            meta=content,
            map_data=[],
            execution="",
            is_structured=False,
            raw_content=content,
        )

    meta = parts[0]
    rest = parts[1]

    # Split map from execution
    parts2 = re.split(r"##\s+\*\*4\.\s+Execution\*\*", rest, flags=re.IGNORECASE)
    map_text = parts2[0].strip()
    execution = "## **4. Execution**\n\n" + parts2[1].strip() if len(parts2) > 1 else ""
    if len(parts2) > 1 and parts2[1].strip() == "":
        # Handle case where execution header exists but is empty
        execution = "## **4. Execution**"

    # 2. Parse Map Structure
    groups: List[MapGroup] = []
    current_group: Optional[MapGroup] = None
    current_notes: List[str] = []

    lines = map_text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Horizontal rule separators should not be treated as group notes
        if re.match(r"^-{3,}$", line):
            continue

        # Group Header: ### **Group Name**
        group_match = re.match(r"###\s+\*\*(.*?)\*\*", line)
        if group_match:
            # flush any notes collected for previous group
            if current_group is not None:
                current_group.notes = "\n".join(current_notes).strip()
            current_notes = []
            current_group = MapGroup(name=group_match.group(1), rows=[])
            groups.append(current_group)
            continue

        # Row: - **Name** ... -> `Key` | ...
        if "->" in line:
            left, right = line.split("->", 1)

            # Extract Name
            # Remove leading '-', '*', spaces
            name = left.strip().lstrip("-").strip()
            # Remove bold formatting **...**
            name = name.replace("**", "").strip()

            # Extract Columns
            # Expected: `Key` | `Desc` | `Class` | `Unit`
            segments = [s.strip() for s in right.split("|")]
            # Clean backticks - remove surrounding backticks and whitespace
            segments = [s.strip().strip("`").strip() for s in segments]

            # We need at least 4 segments. If less, pad or handle error?
            # We'll just take what we can get.
            if len(segments) >= 4:
                row = MapRow(
                    name=name,
                    key=segments[0],
                    desc=segments[1],
                    class_=segments[2],
                    unit=segments[3],
                )
                if current_group:
                    current_group.rows.append(row)
                elif not groups:
                    # Fallback if row appears before any group
                    current_group = MapGroup(name="General", rows=[])
                    groups.append(current_group)
                    current_group.rows.append(row)
            else:
                # Not a standard mapping row; preserve as notes
                if current_group is not None:
                    current_notes.append(line)
                else:
                    # ignore or could store global notes in future
                    pass
        else:
            # Preserve any non-row content inside a group (e.g. CRITICAL INSTRUCTION blocks)
            if current_group is not None:
                current_notes.append(line)

    # flush last group notes
    if current_group is not None:
        current_group.notes = "\n".join(current_notes).strip()

    return PromptStructure(
        meta=meta.strip(),
        map_data=groups,
        execution=execution.strip(),
        is_structured=True,
    )


def build_markdown_from_structure(structure: PromptStructure) -> str:
    if not structure.is_structured:
        return structure.raw_content or ""

    lines = []
    lines.append(structure.meta.strip())
    lines.append("")
    lines.append("## **3. Master Extraction Map**")
    lines.append("")

    for group in structure.map_data:
        lines.append(f"### **{group.name}**")
        lines.append("")
        if getattr(group, "notes", ""):
            lines.append(group.notes.strip())
            lines.append("")
        for row in group.rows:
            # Reconstruct row
            line = f"- **{row.name}** -> `{row.key}` | `{row.desc}` | `{row.class_}` | `{row.unit}`"
            lines.append(line)
        lines.append("")

    # Keep separator between map and execution (matches existing prompt style)
    if structure.execution.strip():
        lines.append("---")
        lines.append("")
        lines.append(structure.execution.strip())
    lines.append("")  # End with newline

    return "\n".join(lines)


# --- Endpoints ---


@router.get("/", response_model=List[PromptMetadata])
def list_prompts():
    """List all available prompts."""
    # We use _scan_registry directly to get filenames
    registry = _scan_registry()
    results = []
    for key, config in registry.items():
        results.append(
            PromptMetadata(key=key, label=config["label"], filename=config["filename"])
        )
    return sorted(results, key=lambda x: x.label)


@router.get("/{filename}", response_model=PromptStructure)
def get_prompt(filename: str):
    """Get parsed prompt content."""
    # filename might be passed as "nc-rates" (key) or "nc-rates.md"
    try:
        content = get_prompt_content(filename)
        return parse_markdown_to_structure(content)
    except Exception as e:
        logger.error(f"Error reading prompt {filename}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{filename}/download")
def download_prompt(filename: str):
    """Download the raw prompt markdown file."""
    try:
        content = get_prompt_content(filename)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Error downloading prompt {filename}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/", response_model=PromptMetadata)
def create_prompt(request: CreatePromptRequest):
    """Create a new prompt file."""
    try:
        filename = request.filename
        if not filename.endswith(".md"):
            filename += ".md"

        content = build_markdown_from_structure(request.structure)
        save_prompt_content(filename, content)

        # Return metadata
        key = filename.replace(".md", "")
        return PromptMetadata(
            key=key, label=key.replace("-", " ").title(), filename=filename
        )
    except Exception as e:
        logger.exception("Failed to create prompt")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{filename}")
def update_prompt(filename: str, request: UpdatePromptRequest):
    """Update an existing prompt."""
    try:
        # Verify it exists first (optional, save_prompt_content overwrites anyway)
        # But good to check if we want to ensure update vs create logic
        # get_prompt_content(filename) # Will raise if not found

        content = build_markdown_from_structure(request.structure)
        save_prompt_content(filename, content)
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.exception(f"Failed to update prompt {filename}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename}")
def delete_prompt_endpoint(filename: str):
    """Delete a prompt file."""
    try:
        delete_prompt(filename)
        return {"status": "success", "filename": filename}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.exception(f"Failed to delete prompt {filename}")
        raise HTTPException(status_code=500, detail=str(e))
