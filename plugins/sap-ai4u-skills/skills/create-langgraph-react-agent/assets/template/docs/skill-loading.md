# Dynamic agent skills

## Purpose

`load_skill_node` scans `agent_skills/` before every request and injects only
skill names and descriptions into the system prompt. The model retrieves full
instructions only when needed through `load_skill`.

## Folder contract

Each folder must match the `name` in its `SKILL.md` frontmatter:

```text
agent_skills/
└── example/
    ├── SKILL.md
    └── references/
        └── example.md
```

The tool accepts an ordered list:

```json
{"skill_names": ["skill-a", "skill-b"]}
```

It de-duplicates names, validates the complete request atomically, and then
concatenates each `SKILL.md` followed by all recursive UTF-8 text files in
lexical order. Explicit skill/file boundaries prevent content from blending.
Binary files are listed but not injected. Symlinks and traversal are rejected,
and the combined result is capped by `max_loaded_characters`.

## Related files

- `template_agent/skills.py`: discovery, concatenation, and LangChain tool.
- `agent_skills/example/`: runnable multi-file example.
- `tests/test_skills.py`: ordering, batch, safety, and size checks.

## Test

```bash
.venv/bin/python -m template_agent skills-list
.venv/bin/python -m pytest -q tests/test_skills.py
```
