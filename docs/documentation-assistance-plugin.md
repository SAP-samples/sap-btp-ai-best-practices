# AI4U Documentation Assistance plugin

## What it is

`ai4u-documentation-assistance` is the second plugin in the `sap-ai4u`
marketplace. It groups agent skills that produce delivery documentation for
SAP AI applications and proofs of concept, separate from the development
skills in `sap-ai4u-skills`. Teams install it on its own when they need
documentation help without the full development catalog.

Version 0.3.0 contains one skill: `technical-handover`.

The display name is "AI4U Documentation Assistance". The identifier is
lowercase (`ai4u-documentation-assistance`) because Claude Code, Codex, and
`scripts/validate_package.py` all require kebab-case plugin names.

## The `technical-handover` skill

### Purpose

It produces an SAP-styled Word handbook (.docx) of roughly 40-60 pages. The
handbook lets an implementation team that has never seen the code understand,
run, extend, and productionise the application.

### Inputs

- The application or PoC repository (local path or a remote shell).
- One interview round with the user for facts the code cannot answer.
- Optional: the Business Process Document (BPD, .docx) of the scenario.
- Python 3 with `python-docx`, `markdown-it-py`, `pyyaml`, and `pillow`.
- A headless Chromium (Playwright or a local Chrome/Edge) for diagrams.
  LibreOffice is optional and only used for PDF previews.

### Outputs

Inside a `handover/` folder in the target project:

- `handover.md`: the source text of the handbook.
- `figures/`: diagram PNGs and their JSON specs, plus screenshot placeholders.
- `inventory.json` and `inventory.md`: routes, environment variable names, AI
  stacks, deployment files, and secret-hygiene findings.
- The final `.docx` (optionally a PDF preview).

The handbook has six parts: solution and usage, architecture, backend and
APIs, AI capabilities, security/configuration/operations, and the path to
production on SAP Business AI Platform.

### Path to production scope

The Part VI gap register is deliberately narrow. PoCs are built outside the
customer's landscape on exports (usually Excel workbooks), so only two areas
are certain to change before productive use:

- **Security**: user authentication, API keys in the browser bundle,
  authorization on destructive and administrative routes, secrets in
  manifests or deploy scripts, dangerous exposed surfaces, identity towards
  the back end.
- **Integration**: inbound extraction from and write-back to the customer's
  systems (ECC, S/4HANA, APO/IBP or named non-SAP systems), with Cloud
  Connector and destinations.

With a BPD, each requirement the PoC does not meet adds one "Functional scope"
row. Operational topics (high availability, CI/CD, observability, UI hosting,
tests, dependencies, model choice, retention, audit logging, optimizer
backlog) never become gaps, roadmap items or open questions; the current state
is still described in Part V. A typical register has 5-10 rows, and the
roadmap has two phases (security hardening, integration), plus functional
scope with a BPD. The checklist and the "Do not raise" list live in
`references/path-to-production.md`.

### Skill layout and tools

The skill is a folder. `SKILL.md` holds the principles, house rules, workflow
and chapter blueprint; the rest is loaded only when a step needs it:

```
skills/technical-handover/
  SKILL.md
  references/   path-to-production.md, diagrams.md, markdown-conventions.md, rom-cost.md, bpd-merge.md
  scripts/      the tools below
  assets/       brand.json, examples/platform_target_architecture.json
```

Step 0 copies `scripts/` and `assets/` into `handover/.tools/` of the target
project. The copies are meant to be adapted per project (another framework, a
different brand, extra checker rules); the files in the skill folder are never
edited during a handover. The scripts resolve `assets/brand.json` relative to
their own location, so the two folders must stay side by side.

| File | Role |
| --- | --- |
| `scripts/repo_inventory.py` | Scans the repository for routes, env vars, AI signals, deployment files, secrets |
| `scripts/render_diagram.py` | Renders `platform`, `graph`, `steps`, and `sequence` diagrams from JSON |
| `scripts/render_png.py` | Headless-browser screenshot helper for the diagram renderer |
| `scripts/make_placeholder.py` | Creates WIP screenshot placeholders |
| `scripts/check_handover.py` | Checks chapters, figures, API/env coverage, secrets, house rules, and BPD coverage (`--bpd`) |
| `scripts/build_docx.py` | Builds the SAP-styled .docx from Markdown |
| `scripts/bpd_extract.py` | Extracts a BPD into Markdown/JSON, saves its images and cuts process flows into panels |
| `assets/brand.json` | Colours, fonts, page size, and callout styles |
| `assets/examples/platform_target_architecture.json` | Example `platform` diagram spec |

## Business Process Document (BPD) input

### What it is for

A BPD (SAP Business Scenario Design Document) describes the scenario a PoC
was built for: current process, pain points, requirements, KPIs, roles, data,
process scope, process flow, and the designed technical architecture. When the
user supplies one, the handover:

- opens Part I with a chapter 1 **Business scenario** holding the business
  content (current process and challenges, proposed solution and benefits,
  pain points, KPIs, process scope, process-flow figures);
- merges the BPD's objectives and requirements tables (which repeat each
  other) into one requirements table with `BR-xx` IDs that mirror the BPD
  numbering (BPD requirement 5 is `BR-05`; gaps in the numbering are kept),
  without a PoC status column;
- turns each BPD requirement the PoC does not meet into one gap-register row
  (Area "Functional scope"), citing the BR ID; several shortfalls for the same
  requirement share that row;
- draws the Part VI target architecture from the BPD's technical architecture,
  with every element the PoC lacks marked `planned` (a gap row only through the
  requirement it serves);
- moves the other design sections to their chapters: roles to *Users, roles
  and access*; the data model to the Part II data model (source data lineage)
  and Part IV; data sources to the Part II current data intake and the Part VI
  integration plan; out-of-scope processes to *Known limits* and the roadmap.

The handover **absorbs** the BPD rather than citing it: the word "BPD" and
the phrase "Business Scenario Design" never appear in the handover (text,
table headers, glossary, figure file names or labels). Only diagrams and
relevant images are reused, never the cover photo. Before turning technology
stack differences into gaps, the user is asked whether the designed stack is
binding. Every BPD table's destination, and every merged or dropped column, is
recorded in `notes/bpd-mapping.md`, and `handover/bpd/` (the raw customer
document) should be added to `.gitignore`.

The full mapping is in `skills/technical-handover/references/bpd-merge.md`.

### Inputs and outputs

`bpd_extract.py <bpd.docx> --out handover/bpd` writes:

| Output | Content |
| --- | --- |
| `bpd.md` | Headings numbered like the BPD's TOC, paragraphs, bullets and cleaned tables in document order |
| `bpd.json` | Sections, table blocks (caption, `kind`, header, rows), images, `requirements` (BPD IDs), `out_of_scope` processes |
| `media/img_NN.<ext>` | Every embedded image (EMF/WMF converted with LibreOffice when present) |
| `media/img_NN_panel_K.png` | Panels of a framed multi-process figure (e.g. a wide process flow) |

`check_handover.py ... --bpd handover/bpd/bpd.json` then reports:

- errors: "BPD" (also in file names such as `bpd_flow.png` and labels such as
  `fig-bpd-1`) or "Business Scenario Design" anywhere in the handover; no
  *Business scenario* chapter; no table with `BR-xx` IDs;
- warnings: a BPD requirement without its `BR-xx` row; a BPD out-of-scope
  process not named in Part V or VI.

### Key code

Panel detection finds framed rectangles: two vertical rules with the same span
joined by a horizontal rule. White space inside a panel never causes a cut:

```python
for i, (xa, ya0, ya1) in enumerate(vert):
    for xb, yb0, yb1 in vert[i + 1:]:
        if xb - xa < min_frac * w * 0.5 or abs(ya0 - yb0) > ty or abs(ya1 - yb1) > ty:
            continue
        y0, y1 = min(ya0, yb0), max(ya1, yb1)
        if has_edge(y0, xa, xb) or has_edge(y1, xa, xb):
            boxes.append((xa, y0, xb, y1))
```

The branding rule flags "BTP" unless a capitalised, non-generic word follows
(a service name):

```python
(r"\bBTP\b(?!,?[ \t]+(?!(?:" + BTP_GENERIC + r")\b)[A-Z])|(?i:business technology platform)|\bBAIP\b", ...)
```

"SAP BTP Destination Service" and "SAP BTP Audit Log" pass; "hosted on SAP
BTP" and "the BTP application layer" are errors.

### How to test

```bash
S=plugins/ai4u-documentation-assistance/skills/technical-handover/scripts
python3 $S/bpd_extract.py --self-test
python3 $S/check_handover.py --self-test
python3 $S/bpd_extract.py <bpd.docx> --out /tmp/bpd
python3 $S/check_handover.py handover/handover.md --bpd /tmp/bpd/bpd.json
```

On a sample BPD the extractor found 13 sections, 12 table blocks with the
right kinds, requirements 1, 2, 3, 5, 6, 7, 8, four out-of-scope processes,
and cut the 13,480 px process flow into its two panels (ECN process and New
product process) with their titles. On a test handover the checker reported
a dropped requirement and the unnamed "Global CIB" process.

## Related files

| Path | Role |
| --- | --- |
| `plugins/ai4u-documentation-assistance/plugin.json` | Agent Plugins 1.0 manifest |
| `plugins/ai4u-documentation-assistance/.claude-plugin/plugin.json` | Claude Code manifest |
| `plugins/ai4u-documentation-assistance/.codex-plugin/plugin.json` | Codex manifest with display name and default prompts |
| `plugins/ai4u-documentation-assistance/skills/technical-handover/SKILL.md` | Workflow, house rules, chapter blueprint |
| `plugins/ai4u-documentation-assistance/skills/technical-handover/references/` | Checklist, diagram, Markdown, ROM cost and BPD merge references |
| `plugins/ai4u-documentation-assistance/skills/technical-handover/scripts/` | The helper tools |
| `plugins/ai4u-documentation-assistance/skills/technical-handover/assets/` | Brand file and example diagram spec |
| `.claude-plugin/marketplace.json` | Claude marketplace entry |
| `.agents/plugins/marketplace.json` | Codex marketplace entry |
| `scripts/validate_package.py` | Multi-plugin package validation |

## Multi-plugin validation

Before this plugin, `scripts/validate_package.py` validated one hardcoded
plugin and required each marketplace to list exactly one entry. It now:

1. Reads both marketplaces and requires them to list the same plugin names,
   without duplicates, each sourced from `./plugins/<name>`.
2. Checks the Codex-only fields (local source type, `AVAILABLE` /
   `ON_INSTALL` policy, category) for every Codex entry.
3. Reports any folder under `plugins/` that is not listed, and any listed
   plugin whose folder is missing.
4. Runs the existing per-plugin checks for every listed plugin: the three
   manifests, their consistency with the folder name, every skill's
   frontmatter, and the packaged-file scan for secrets and local artifacts.

Key snippet:

```python
plugin_names = _validate_marketplaces(codex_marketplace, claude_marketplace, errors)
folders = {path.name for path in plugins_root.iterdir() if path.is_dir()}
for unlisted in sorted(folders - set(plugin_names)):
    errors.append(f"{plugins_root / unlisted}: plugin folder is not listed in the marketplaces")
for name in plugin_names:
    ...
    skill_count += _validate_plugin(plugins_root / name, errors, warnings)
```

## How to test

Validate the repository:

```bash
python3 -m pip install -r requirements-dev.txt
python3 scripts/validate_package.py
```

Expected result: `Validation passed: 35 skill(s), 3 recommendation warning(s)`.

Install and try the plugin in Claude Code:

```bash
claude plugin marketplace update sap-ai4u
claude plugin install ai4u-documentation-assistance@sap-ai4u
```

Then, in a project, ask: "Create a technical handover document for this
repository."
