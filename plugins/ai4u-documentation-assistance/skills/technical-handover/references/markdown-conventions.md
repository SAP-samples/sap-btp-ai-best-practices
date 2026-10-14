# Markdown conventions for build_docx.py

Write `handover.md` in CommonMark + pipe tables. The builder turns it into an SAP-styled Word document (cover, header/footer with classification and page numbers, numbered chapters, TOC, styled tables and callouts).

### Front matter (drives the cover, header and footer)
```yaml
---
title: Production Wheel Optimizer
subtitle: Technical handover for the implementation team
customer: ACME Coatings
project: Production Wheel Optimization PoC
version: "1.0"
date: 25 September 2026
status: Final draft
classification: Confidential
authors: [Jane Doe (SAP Business AI Platform AI CoE)]
audience: Partner implementation team
header: Wheel Optimizer          # optional: page-header text; defaults to project
cover_note: Describes the proof-of-concept as delivered. Not a statement of production readiness.
---
```

- **`project` appears in every page header** (`project  |  Technical Handover`), so keep it short: "Wheel Optimization PoC", not "…, Phase 1 proof of concept (reference plant)". If the cover needs the long name, set `header` to the short one.
- **`repository` and `commit` are optional and left out by default.** The code is usually delivered in a new repository, so the PoC repository and its commit mean nothing to the reader. Add them only when the user asks; they then appear on the cover as "Source repository" and "Source commit".

### Structure
| Markdown | Result |
|---|---|
| `# About this document {.unnumbered}` | Chapter without number |
| `[[TOC]]` | Table of contents (Word field, pre-filled with links; Word adds page numbers on open) |
| `# Part I: Solution and usage {.part tagline="What it does and how to use it"}` | Part banner at the top of a new page; the next chapter starts on the same page (not numbered, not in TOC) |
| `# Solution at a glance` | Chapter "1.  Solution at a glance" (auto-numbered) |
| `## Scope` | Section "1.1  Scope" |
| `### Detail` | Unnumbered sub-heading |
| `# Appendix A. Glossary` | Never numbered |
| `{.newpage}` on a heading | Page break before it |
| `\newpage` on its own line | Page break |
| `<!-- landscape -->` … `<!-- portrait -->` | Landscape pages in between (for `platform` architecture diagrams and very wide tables). Place the markers around the figure *after* the chapter's introductory text, never right after a Part heading, so no heading or banner is left alone on a page |

### Content
| Markdown | Result |
|---|---|
| `![Current PoC architecture](figures/arch_current.png)` alone in a paragraph | Full-width figure + "Figure N. caption" |
| `![Caption](figures/x.png "width=0.6")` | Figure at 60 % of text width |
| `Table: Endpoint catalog — datasets` on the line before a table | "Table N. …" caption |
| pipe table | Navy header row, banded rows, column widths from content; header row repeats across pages |
| `` `code` `` | Monospace, tinted |
| fenced code block with language | Grey block with blue left rule and language label |
| `[text](#heading-slug)` | Internal link to that heading (GitHub-style slug) |
| `[text](https://…)` | External hyperlink |
| `<br>` inside a table cell | Line break |

### Cross-references
Give figures and tables a label, then refer to it with `@label`; the builder replaces it with "Figure N" / "Table N".
```markdown
![Current PoC architecture](figures/arch_current.png "id=fig-arch-current")
Table: Production gap register {#tbl-gaps}
…as shown in @fig-arch-current, the gaps in @tbl-gaps…
```
Labels must start with `fig-` or `tbl-`. `width=0.7` and `id=…` can be combined in the image title: `"width=0.7 id=fig-x"`.

### Not supported
- LaTeX / `$…$` math (printed literally) — use a table of symbols or a code block for formulas.
- Raw HTML other than `<br>` in cells and the layout comments above.
- Footnotes and nested tables.

### Callouts
```markdown
> [!GAP] Shared API key in the browser
> `VITE_API_KEY` is compiled into the UI bundle; anyone with the URL can call every endpoint.
```
First line after the marker becomes the title when followed by a line break; otherwise the label prefixes the text.

| Marker | Label | Use for |
|---|---|---|
| `[!NOTE]` | Note | context |
| `[!TIP]` | Tip | practical advice for users/devs |
| `[!IMPORTANT]` | Important | must-know behaviour |
| `[!WARNING]` | Warning | traps, irreversible actions |
| `[!CAUTION]` | Security | security-relevant facts |
| `[!GAP]` | Production gap | inline pointer to a Part VI gap row |
| `[!OPEN]` | Open question | unresolved items (also list them in Part VI) |
| `[!EVIDENCE]` | Evidence | how a statement was verified |

### Tips
- Keep table cells short (≤ 25 words); move long explanations to text below the table.
- Tables with 5+ columns get cramped on A4 portrait — split them or drop a column.
- Paths, env var names, endpoints and identifiers always in backticks.
- Don't hand-number headings; the builder does it.
