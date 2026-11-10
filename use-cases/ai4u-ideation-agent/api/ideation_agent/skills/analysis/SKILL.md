---
name: analysis
description: "Auditable analysis workflow for file-based investigations with script-first execution."
---

# Analysis Skill

Use this when a task involves structured data analysis and report generation.

## Guardrails

- Keep all reads/writes inside the provided working directory.
- Save every execution as a script under `scripts/`.
- Do not use inline Python execution (`python -c`, heredoc, REPL snippets).
- Do not delete generated scripts.

## Workflow

1. Inspect inputs and assumptions.
2. Create `analysis_progress.md` and log each step.
3. Test one hypothesis per script execution.
4. Copy actual outputs to progress log.
5. Create `analysis_report.md` summarizing findings and limitations.
