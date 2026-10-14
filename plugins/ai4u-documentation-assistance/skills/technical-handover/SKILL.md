---
name: "technical-handover"
description: "Create an SAP-styled technical handover Word document (.docx) for an application or PoC codebase so an implementation team can take it over: usage, SAP Business AI Platform architecture diagrams, API catalog, AI capabilities, path to production. Optionally merges a Business Process Document (BPD): business scenario, requirement traceability, BPD target architecture."
---

# Technical handover document

Produce a Word handbook that lets an implementation team who has never seen the code understand, run, extend and productionise the application. The depth benchmark is an application handbook of roughly 40–60 pages: every endpoint listed, every component's responsibility stated, every AI capability explained, and a focused gap register for production (security and integration).

The workflow is **install tools → inventory → (extract BPD) → interview → read → draw → write → check → build → look → deliver**. The BPD step only applies when the user supplies a Business Process Document. You write the handover in Markdown, and the bundled tools turn it into an SAP-styled .docx and render the diagrams and screenshot placeholders.

## Principles

- **Evidence over assumption.** Every statement should be traceable to code, config or the user. When the code can't answer something, write it as an open question. A confident wrong sentence costs the implementation team days.
- **Cite code, not repository docs.** Reference source and config paths in backticks (`api/app/routers/workspace.py`). You may use what the repo's `docs/` folder says, but never name, cite or link its files in the document. Where a doc contradicts the code (a different default model, a removed endpoint), the code wins, and you tell the user which docs are stale.
- **No commits, branches or source-repository links by default.** The code is usually delivered to the customer in a new repository, so commit hashes, branch names and the PoC repository's URL mean nothing to the reader. Refer to code by path only; say "the delivered proof-of-concept code" and report test results as "1,004 passed", not "at commit …". Add the optional `repository`/`commit` front-matter keys only when the user asks for them.
- **Never leak secrets.** Document variable *names* and where their values come from, never the values themselves.
- **Honest about PoC status, narrow about gaps.** Shared API keys, manual uploads, single instances, missing tests: say so plainly in the current-state chapters. Only security and integration findings (plus unmet BPD requirements) become gap rows in Part VI; operational choices we can't know from outside the customer's landscape are not gaps ([references/path-to-production.md](references/path-to-production.md)).
- **SAP Business AI Platform as the frame of reference.** Architecture uses the SAP Business AI Platform reference-architecture layout. Production recommendations use SAP Business AI Platform services (XSUAA/IAS, Destination, Connectivity + Cloud Connector, Credential Store, Integration Suite).

## House rules (the checker enforces them)
1. Never mention the **SAP AI Ethics self-assessment**, nor the **data privacy of prompts** (what business data reaches LLM providers, masking prompt data). This applies everywhere: findings, gaps, questions and recommendations.
2. **No time or effort estimates** in the path to production: no durations, sprints, person-days, dates or S/M/L effort columns. Prioritise with Must / Should / Could only.
3. **IDs point home.** When a gap, finding or question ID (G-01, SEC-3, Q-07) is used outside its own table, say where it is defined: "(G-01, see @tbl-gaps in Path to production)".
4. **Don't cite `docs/` files** (see above).
5. **Screens are placeholders.** Use WIP screenshot placeholders that the author replaces with real screenshots. Don't draw mockups and don't capture live systems.
6. **Sequence diagrams close every loop.** Each request has its response drawn back to the caller, all the way to whoever started the flow (e.g. model → agent runtime → chat router → UI).
7. **Branding.** The platform is **SAP Business AI Platform** ("Business AI Platform" after first mention), in text, tables, figure labels and metadata. Never name the platform itself with the old name ("hosted on SAP BTP", "the BTP application layer", "Business Technology Platform") and never abbreviate the new one (no "BAIP"). Service names that carry the old prefix are allowed: "SAP BTP Destination Service", "SAP BTP Audit Log".

## Step 0: install the tools (once per project)

The tools live next to this file: `scripts/`, `assets/` and `references/` (`<skill-dir>` is the base directory shown when this skill loads). Copy `scripts/` and `assets/` into the project's handover folder; never run or edit them in `<skill-dir>`.

```bash
mkdir -p handover/.tools && TOOLS="$(pwd)/handover/.tools"
cp -R "<skill-dir>/scripts" "<skill-dir>/assets" "$TOOLS/"
pip install python-docx markdown-it-py pyyaml pillow   # add --break-system-packages if pip refuses
```

**Adapt the copies to the project.** The tools are a starting point, not a fixed product: a framework the inventory can't parse, a BPD template laid out differently, an extra checker rule or another brand are all fixed by editing the files in `$TOOLS`. The scripts find `assets/brand.json` relative to themselves, so keep `scripts/` and `assets/` side by side. When a change is generic rather than project-specific, point it out to the user as a candidate update for this skill.

Diagrams need a headless Chromium: Playwright (`pip install playwright && python -m playwright install chromium`) or a local Chrome/Edge, which is detected automatically. LibreOffice (`soffice`) is optional; it's used for PDF previews during QA.

| Tool | Purpose |
|---|---|
| `scripts/repo_inventory.py <repo> --out inventory.json --md inventory.md` | Routes, env-var names, AI/ML/optimisation signals, deployment files, secret hygiene, tests |
| `scripts/render_diagram.py spec.json out.png` | `platform` / `graph` / `steps` / `sequence` diagrams from JSON |
| `scripts/make_placeholder.py out.png --title … --capture … [--id SCR-01]` | WIP screenshot placeholder (Pillow only) |
| `scripts/check_handover.py handover.md --inventory inventory.json --repo <repo> [--forbid <term> --docx <file> --bpd bpd.json]` | Required chapters, figures, API/env coverage, secrets, house rules, ROM cost table, forbidden terms, BPD coverage |
| `scripts/bpd_extract.py bpd.docx --out handover/bpd` | Optional BPD input: clean Markdown/JSON tables, requirements, out-of-scope processes, images cut into process-flow panels |
| `scripts/build_docx.py handover.md -o Handover.docx [--pdf]` | SAP-styled Word document: cover, TOC, numbered chapters, tables, callouts, landscape pages |
| `scripts/render_png.py page.html out.png` | Headless-browser screenshot helper (used by the diagram renderer) |
| `assets/brand.json` | Colours, fonts, page size and callout styles. Edit it to restyle everything |
| `assets/examples/platform_target_architecture.json` | Worked `platform` diagram spec to copy from |

References, read when the workflow step needs them:

| File | Read it for |
|---|---|
| [references/path-to-production.md](references/path-to-production.md) | The Part VI gap-register checklist (security and integration) and what never becomes a gap |
| [references/diagrams.md](references/diagrams.md) | Diagram JSON specs (`platform`, `graph`, `steps`, `sequence`) and screenshot placeholders |
| [references/markdown-conventions.md](references/markdown-conventions.md) | What `build_docx.py` understands: front matter, parts, figures, tables, callouts, cross-references |
| [references/rom-cost.md](references/rom-cost.md) | The ROM cost calculation section |
| [references/bpd-merge.md](references/bpd-merge.md) | Only with a BPD: where each BPD section goes, the requirements table, gaps against the BPD |

If the code sits on a machine you only reach through a remote shell, read it there (copy `repo_inventory.py` over if needed). Render and build wherever Python and a browser are available, then copy the finished document next to the code.

## Workflow

### 1. Set up the working folder
Inside `handover/` keep `handover.md`, `figures/` (PNGs plus their JSON sources), `notes/`, `inventory.json` and the `.tools/` from Step 0. If you changed the tools, deliver `.tools/` together with the handover so the team rebuilds with the same versions; if they are unchanged, it can be left out.

### 2. Inventory the repository
```bash
python "$TOOLS/scripts/repo_inventory.py" <repo> --out handover/inventory.json --md handover/inventory.md
```
Read `inventory.md`. It gives you:
- the route list (the API catalog must cover every method+path);
- env-var names;
- which AI stacks are present, which decides the Part IV chapters;
- deployment descriptors;
- any literal secrets committed in config. That's a Part VI finding: report the file and key name, never the value.

Extraction is heuristic, so confirm route prefixes against the app's router mounting. For FastAPI, importing the app and listing `app.openapi()` in a throwaway venv is authoritative.

### 2b. Extract the BPD (only if the user supplied one)
```bash
python "$TOOLS/scripts/bpd_extract.py" <bpd.docx> --out handover/bpd
```
Read `handover/bpd/bpd.md` in full and open every image and panel in `handover/bpd/media/`. Then follow [references/bpd-merge.md](references/bpd-merge.md): it maps each BPD section to its chapter. Key rules:
- **The handover absorbs the BPD and never names it**: no "BPD" or "Business Scenario Design" in text, table headers, glossary, file names or labels (the checker fails on them). Point to the handover's own chapters and `BR-xx` rows instead.
- Only diagrams and relevant images are reused, copied into `figures/` under neutral names; never the cover photo or decorative pictures.
- Record where every BPD table went, and every merged or dropped column, in `notes/bpd-mapping.md`.
- `handover/bpd/` holds the raw customer document: suggest adding it to `.gitignore`.

The BPD answers several interview questions (integration targets, roles, scope); ask only what it leaves open, plus whether its **technology stack is binding** before any stack difference becomes a gap row or open question.

### 3. Interview the user (one round, only what the code can't tell you)
Skip anything the request or the BPD already answers. Ask, ideally in one structured question set:
- Whether a **Business Process Document (BPD)** exists for the scenario, if the user hasn't mentioned one. It is optional; if one arrives, run step 2b.
- **Integrations in the final implementation**, which you always ask about: which systems the data must come *from* and go *to* once this is productive (e.g. SAP ECC, S/4HANA, APO, IBP, Datasphere, non-SAP), in which direction, and which business objects. Today's PoC often uses manual uploads/exports; the target systems drive the target architecture, the integration plan and much of the gap register. Don't assume S/4HANA when the customer runs ECC.
- Customer and project names as they may appear (some customers must be anonymised), document classification, and author names/roles.
- Audience (partner SI, customer IT/security board, SAP Services), which sets the tone and how much SAP Business AI Platform background to explain.
- Target landscape and runtime (CF/Kyma, subaccount/landscape setup), and the identity provider.
- Known issues, decisions already taken, PoC results worth quoting, and anything out of scope.
- The document language (default English).
- **Production model**: which model(s) the productive solution will use. Before writing, compare the answer with the model configured in `manifest.yaml`/env and the code defaults, and with the model the cost estimate assumes. Report any mismatch to the user and get it resolved first; an unresolved mismatch otherwise ends up as an open question and a gap row that disappear again once the code is fixed.
- **Cost estimate** for the ROM cost calculation: the PDF export of the solution's services estimate from the SAP Discovery Center (see [references/rom-cost.md](references/rom-cost.md)). Also ask for the two inputs that size the AI Core line: capacity units per transaction and the expected monthly volume. If no estimate is provided, write the section as described in [references/rom-cost.md](references/rom-cost.md) and add an open question.

If nobody can answer, proceed with sensible defaults (partner SI audience, English, the customer named only if the request names it) and record every assumption in the open-questions table.

**If the customer must not be named**, treat it as a hard requirement for every artifact: the text, the front-matter `project` field, figure sources, placeholder titles, file and folder names, and the .docx metadata. Codes or names that identify the customer (plants, people, systems) count too; replace them with neutral labels and record the substitution. Run the checker with `--forbid <name>` (repeat it for variants) and `--docx`, and look at every figure yourself, because the scan can't read text inside images.

### 4. Read the code deeply
Order: README and `docs/` (as background only) → entry points (`main.py`, server bootstrap, `manifest.yml`/`mta.yaml`) → routers/controllers → services and domain modules → persistence (DDL, repositories) → AI code (agent graph, tools, prompts/skills, model config; ML training/inference; optimisation model) → background workers → UI routes, pages and services → tests → deploy scripts.

Keep notes per area in `notes/`, with file:line evidence. If you can spawn subagents, split big repos across them in parallel (frontend, API + data, AI/agents, optimisation/ML, deployment + security) and ask each for structured notes with evidence, then read the critical files yourself. Without subagents, go area by area and write each area's notes before moving on.

If the environment allows, run the test suite and list the OpenAPI routes. Both are cheap, strong evidence. Record the results in the evidence callout. Never modify the user's repository to do this; use a copy or a throwaway venv.

Repository docs (`docs/`, READMEs, feature notes) often lag behind the code. Whenever one contradicts the code, write what the code does and note the doc (file, claim, what the code says instead) in `notes/stale-docs.md` for the delivery message.

### 5. Plan the document
Map the chapter blueprint (below) onto the application: which usage chapters (one per real workflow), which Part IV chapters apply, which runtime flows deserve diagrams, and which screens need placeholders. With a BPD, add chapter 1 *Business scenario* and place every BPD section as mapped in [references/bpd-merge.md](references/bpd-merge.md). Write the outline into `handover.md` as headings first.

### 6. Produce the figures
The minimum set (details in [references/diagrams.md](references/diagrams.md)):
- **Current PoC architecture**: `type: platform` (Part II).
- **Target production architecture**: `type: platform`, with the systems from the interview (e.g. ECC via Cloud Connector) and the identity and connectivity components marked `"status": "planned"` (Part VI). With a BPD, draw it from the BPD's technical architecture.
- **Business process flow (as designed)**: only with a BPD, the extracted panels on a landscape page (chapter 1).
- **End-to-end journey**: `type: steps` (Part I).
- **2–4 runtime flows**: `sequence` or `steps`, e.g. a chat turn or a durable optimisation run (Part II). Every request loop is closed; the renderer warns about unanswered requests.
- **4–8 screenshot placeholders**: `make_placeholder.py` (Part I).

Open every PNG and look at it. Fix overlaps and clipped text before moving on; expect 2–3 render passes for a `platform` diagram. Placing connected nodes next to each other fixes most crossings. Put `platform` diagrams on landscape pages (`<!-- landscape -->` / `<!-- portrait -->`).

### 7. Write `handover.md`
Follow the blueprint (below) and the Markdown conventions ([references/markdown-conventions.md](references/markdown-conventions.md)). Use tables for anything enumerable, numbered procedures for anything a person does, and callouts (`[!TIP]`, `[!WARNING]`, `[!CAUTION]`, `[!GAP]`, `[!OPEN]`, `[!EVIDENCE]`) where they help a reader skim. Refer to figures and tables by label (`@fig-arch-current`, `@tbl-gaps`), not hard-coded numbers. LaTeX is not rendered; describe formulas with a symbol table or a short code block.

Style: plain, precise, present tense, third person ("The worker claims the oldest queued job…"). Explain *why* a design choice was made when the code tells you. Define domain terms on first use and collect them in the glossary. Leave out marketing language, filler and invented metrics.

Length: if you're heading past ~60 pages, cut repetition before substance. Merge usage chapters that share a screen, and move exhaustive tables (full env-var list, all agent tools, benchmark detail) to appendices. Never cut endpoints, gaps or open questions.

### 8. Check
```bash
python "$TOOLS/scripts/check_handover.py" handover/handover.md --inventory handover/inventory.json --repo <repo> [--forbid "<customer>" --docx <built.docx>] [--bpd handover/bpd/bpd.json]
```
Fix every error: missing chapters, missing figures, secrets, house-rule violations (docs/ citations, banned topics, old platform branding, time estimates in Part VI, IDs without a location), ROM cost table errors (unclassified rows, total not matching the rows), forbidden terms and, with `--bpd`, a missing *Business scenario* chapter or requirements table. Resolve its BPD warnings too: a dropped requirement, or an out-of-scope process not named in Part V/VI. Aim for 100 % of method+path operations, each with method and path on the same table row, and full env-var coverage.

### 9. Build and look at the result
```bash
python "$TOOLS/scripts/build_docx.py" handover/handover.md -o handover/<Project>_Technical_Handover_v1.0.docx --pdf
```
Render a few PDF pages to PNG (e.g. `pdftoppm -r 60 -png file.pdf pg`) and inspect the cover, contents, a dense table page, the landscape architecture page and a sequence diagram. Fix cramped tables (split them or drop a column), stranded headings (move the landscape marker) and tiny diagrams (simplify them or go landscape). Word fills in the TOC page numbers when the file is opened; it asks to update fields.

### 10. Deliver
Give the user the .docx and say where `handover.md` and `figures/` live, so the team can rebuild it when the code changes. Include:
- a short summary: page and figure count, API coverage, the Must gaps, and the ROM total;
- **the list of screenshot placeholders** (ID, screen, what to capture), so the author can take them and replace the images in Word;
- **stale repository docs** from `notes/stale-docs.md` (file, what it claims, what the code does), so the team can fix them;
- if the AI Core quantity was sized from volume and the estimate PDF is shared with the customer, a reminder to regenerate the PDF with the same quantities.

## Final checklist
- [ ] The production model is confirmed and matches the configured model and the cost estimate; stale repository docs are listed for the user.
- [ ] Integration targets confirmed with the user (or recorded as open questions) and reflected in the target architecture and integration plan.
- [ ] Every method+path from the inventory is in the endpoint catalog, with a real purpose.
- [ ] Current and target SAP Business AI Platform architecture diagrams are present; the component table includes "does not own".
- [ ] Every sequence diagram closes its loops back to the initiator.
- [ ] Each AI capability is explained: where it lives in code, models/solvers, guardrails.
- [ ] Usage chapters have numbered procedures and screenshot placeholders.
- [ ] The security chapter states the current state honestly; the configuration table lists names, not values.
- [ ] Part VI has a ROM cost calculation built from the user's estimate ([references/rom-cost.md](references/rom-cost.md)): every line classified Shared / Solution-specific, AI Core sized from CU per transaction × monthly volume with the same volume in the note and the open question, total recomputed from the rows, no source tool or date named. Without an estimate, the one-line fallback and an open question.
- [ ] Part VI has an evidenced gap register limited to security, integration and (with a BPD) unmet requirements, with Must/Should/Could priorities (no estimates); nothing from the "Do not raise" list in [references/path-to-production.md](references/path-to-production.md) appears as a gap, roadmap item or open question; an integration plan, a roadmap without durations, and open questions.
- [ ] No `docs/` citations, no AI Ethics self-assessment, no prompt data privacy, no old name for the platform itself (service names like "SAP BTP Destination Service" are fine) and no "BAIP"; every ID used elsewhere points to its table.
- [ ] With a BPD: the word "BPD" appears nowhere; `notes/bpd-mapping.md` accounts for every BPD table and dropped column; chapter 1 *Business scenario* is present; every BPD requirement has its `BR-xx` row; every requirement the PoC fails has one gap row; the target architecture follows the BPD; every out-of-scope process is in Known limits and the roadmap.
- [ ] `check_handover.py` passes; the rendered pages were inspected.


---

# Chapter blueprint

The document is layered so different readers can stop at different depths: Part I for anyone, Parts II–IV for developers, Part V for operators and architects, Part VI for whoever plans the production project. Keep the order; drop a section only when it truly doesn't apply, and say so in one line rather than omitting it silently (e.g. "The solution trains no ML models; all predictive behaviour comes from the optimizer in Chapter 12").

Depth target: a full-scope handover for a real PoC is typically 40–60 pages (the reference handbook is ~50). Every chapter should let an engineer who has never seen the repo act without asking the original team. Prefer tables for anything enumerable (components, endpoints, env vars, gaps) and short numbered procedures for anything a person does.

### House rules (apply everywhere)
- **Cite code, not repository docs.** Point to source and config paths (`api/app/routers/workspace.py`, `manifest.yaml`). Use what the repo's `docs/` folder says, but never name or link its files. The implementation team works from the code, and those notes are internal working documents.
- **Topics to leave out entirely:** the SAP AI Ethics self-assessment, and data privacy of prompts (what business data is sent to LLM providers, masking of prompt data). Don't raise them as findings, gaps, open questions or recommendations.
- **No time or effort estimates** in the path to production: no durations, weeks, sprints, person-days, dates or S/M/L effort columns. Prioritise with Must / Should / Could only.
- **Every ID points home.** When a gap, finding or question ID (G-01, SEC-3, Q-07) appears outside its own table, say where it is defined, e.g. "(G-01, see @tbl-gaps in Path to production)".

### Contents
- Front matter
- Part I — Solution and usage (opens with *Business scenario* when a BPD is supplied)
- Part II — Architecture
- Part III — Backend and APIs
- Part IV — AI capabilities
- Part V — Security, configuration and operations
- Part VI — Path to production
- Appendices

---

### Front matter (unnumbered)

**Cover**: generated from the YAML front matter (title, subtitle, customer, project, version, date, status, classification, authors, audience, cover_note; optional `header`, `repository`, `commit`). Keep `project` short, or set `header`: it appears on every page.

**About this document** `{.unnumbered}`
- Purpose in two sentences: what was built, who receives it, and what they should be able to do after reading.
- Evidence statement in a `> [!EVIDENCE]` callout: that it describes the delivered proof-of-concept code, what was verified by reading code vs. running it (tests, e.g. "1,004 passed"; OpenAPI import), and what was *not* verified (live tenant, credentials, performance at scale). No commit hashes or branch names.
- Reader paths table: Reader | Start here | Then use (business/product owner, implementation developer, platform operator, security reviewer, data scientist).
- Document history table: version, date, author, change.
- A note that screenshots are placeholders to be replaced (when they are).

`[[TOC]]` goes after this chapter.

---

### Part I — Solution and usage `{.part}`

#### Business scenario (only with a BPD)
Chapter 1 when the user supplied a Business Process Document; omit it otherwise (the next chapter becomes chapter 1). It keeps the business content together: current process and challenges, proposed solution and expected benefits, pain points, the **requirements table** (`{#tbl-requirements}`: `BR-xx` IDs mirroring the BPD numbers, no PoC status column), KPIs, process scope, and the business process flow panels on a landscape page. Details in [references/bpd-merge.md](references/bpd-merge.md).

#### 1. Solution at a glance
- One paragraph: the business problem, what the application does and the primary outcome. Use the customer's own vocabulary for their process.
- **Scope and non-goals**: bullet list of what it deliberately does not do (e.g. "no write-back to ECC", "not a scheduler"). Implementation teams most often break PoCs by assuming scope that was never there.
- **Workspace/module table**: Area | Primary job | Durable result.
- **End-to-end journey**: 6–10 numbered steps from sign-in to final outcome, each naming the screen and what gets persisted. A BPD process-flow narrative can seed it, but write it from the code.
- Steps diagram (`type: steps`) of the journey.
- **PoC results / status**, if the repo or the user gives them (benchmarks, acceptance runs), with the date and dataset they came from.

#### 2. Users, roles and access
- Who uses it (personas) and how they get in today (login flow, API key, or no authentication).
- Role/permission model as implemented vs. as needed (with a BPD: its personas next to the implemented roles). Missing authentication or authorization is a security gap in Part VI.

#### 3–N. How to use the application (one chapter per major workflow)
For each workflow or screen group:
- A **screenshot placeholder** figure per important screen/state, made with `make_placeholder.py`. The title names the screen; the `--capture` bullets say exactly what the real screenshot must show. The author swaps in real screenshots later.
- Numbered procedure: what the user clicks or types, what the system does, what they get back.
- Inputs expected (file formats, required columns/sheets, sizes); use a table when there are more than three.
- What is persisted and where.
- `> [!TIP]` for practical advice; `> [!WARNING]` for traps (irreversible deletes, long runs, stale state).
- If there is a conversational agent: a **prompt cookbook** table (Goal | Example prompt | What the agent does / which tool it calls).

---

### Part II — Architecture `{.part}`

#### Architecture overview
- Short narrative first (portrait): the request path from user to data and models, where state lives, and what runs asynchronously.
- **Current data intake**: how data enters the application today (uploads, flat-file extracts, APIs), with source and format. With a BPD, start from its data-source table's PoC interfaces.
- Then the **current (PoC) architecture diagram** on a landscape page: `type: platform` in the SAP Business AI Platform reference-architecture layout, showing only what exists today. External systems the solution doesn't touch yet are omitted or shown as `planned`.
- **Component responsibility table**: Component | Technology/runtime | Responsibility | What it does *not* own. The last column prevents wrong assumptions (e.g. "UI holds no credentials", "worker does not serve HTTP").
- Deployment topology table: app/service name | type (CF app, service instance, AI Core deployment…) | memory/instances | routes | bound services.

#### Component deep-dives
One `##` section per significant component (frontend, API, workers/jobs, agent runtime, data layer, integration adapters). For each: purpose, entry points (files), internal modules table (Module/path | Responsibility), key design decisions and why, and extension points.

#### Data model and persistence
- Entity/table table: Entity | Purpose | Key columns / lifecycle.
- Lifecycle rules: immutability, archival, deletion cascades, retention.
- Data volumes seen during the PoC, if known.
- With a BPD: a **Source data lineage** subsection, mapping the BPD's data categories and source tables (e.g. MARA/MARC, STKO/STPO) to the entities above. Add the source tables to the glossary.

#### Key runtime flows
2–4 `sequence` or `steps` diagrams for the flows an implementer must understand (e.g. upload→extract→publish, chat→agent→tool→answer, the job worker loop, a durable optimisation run). Each gets a numbered explanation and its failure behaviour (timeouts, retries, orphaned jobs). **Every request in a sequence diagram must be answered**: draw the response path back hop by hop to whoever started it (e.g. model → agent runtime → chat router → UI).

---

### Part III — Backend and APIs `{.part}`

#### API conventions
Base URL(s) and prefixes, authentication per boundary (table: Boundary | Credential | Principal/authorization), content types, pagination, async pattern (202 + polling? streaming?), error format, idempotency, versioning (usually none in a PoC; say so), and where OpenAPI/Swagger is served.

#### Endpoint catalog
Every route from the inventory, grouped by domain (one `##` per group). Table columns: Method | Path | Purpose | Auth. Paths go in backticks, with method and path on the same row. Write the purpose from reading the handler, not from its name. `check_handover.py` reports missing method+path operations.

#### Representative request/response contracts
4–8 of the most important calls, with realistic (synthetic) JSON request and response bodies taken from the Pydantic/TS models. Mark required vs. optional fields.

#### Statuses and errors
A table of lifecycle statuses (job/run/document states) and their transitions, and a table of error codes → meaning → action.

#### Background processing
Workers, schedulers, queues: how jobs are claimed, concurrency, timeouts, what happens on restart, and scaling limits.

---

### Part IV — AI capabilities `{.part}`

Start with an **AI capability map** table: Capability | Type (GenAI / Agent / ML / Optimization / Rules) | Where in code | Model / solver | Inputs → outputs | Guardrails / human-in-the-loop. Then write one chapter per type that exists. With a BPD, compare its agent input/output fields with the real schemas here; a difference becomes a gap only through the requirement it fails (one row per requirement).

#### Generative AI and agents
- Models and providers (SAP AI Core / Generative AI Hub deployments, model names, orchestration usage), and how the model is selected and configured at runtime.
- Agent architecture: framework (LangGraph, DeepAgents, CAP LLM plugin…), graph/loop shape, a tool table (Tool | What it does | Side effects | Guardrail), memory/state, and context-budget handling.
- Prompts and skills: where they live, how they are versioned, and how to change them safely.
- Grounding / RAG: sources, chunking, embeddings, vector store (HANA Vector Engine?), retrieval parameters.
- Structured outputs and validation.
- Guardrails: tool permission checks, iteration and time limits, output validation, refusal paths, human approval gates.
- Usage and cost telemetry (tokens, caching), and latency.
- Known failure modes and how the PoC mitigates them.

#### Machine learning (if present)
Problem framing, features, training data and split, algorithms, evaluation metrics and results, model persistence/registry, inference path, retraining story, and drift/monitoring (usually absent → Part VI).

#### Optimization (if present)
Write this for an engineer who must maintain the model:
- The business decision being optimised, in plain language.
- Formulation: sets/indices, decision variables, objective(s) (and trade-off/Pareto handling), and constraints in a table (Constraint | Meaning | Where in code). LaTeX is not rendered: describe formulas with a symbol table plus a short code block (e.g. `J_CH = Σ_g Σ_(i,j∈g) w_ij`).
- Solver and modelling stack (Pyomo/HiGHS, OR-Tools, Gurobi…), decomposition/parallelism, time limits, optimality gaps, and what "runtime-limited" means for results.
- Pre- and post-processing, and validation of solutions.
- Performance numbers from the PoC and scaling limits.

#### AI quality
How quality was evaluated (test sets, acceptance runs, evals), known limitations, and the human-in-the-loop points. With a BPD, report the PoC results against its AI-quality KPIs (e.g. a recall target) and link to the KPI table in *Business scenario*.

---

### Part V — Security, configuration and operations `{.part}`

#### Security (current state)
Control map table: Control area | Implemented control | Boundary / residual risk. Cover authentication, authorization, secrets, CORS, isolation between users, input validation, generated-code execution, logging of sensitive values, and dependency risk. Be honest: a PoC with a shared API key should say exactly that. When a weakness has a gap row, reference it with its location: "(G-01, see @tbl-gaps)".

#### Configuration reference
Table: Category | Variables | Meaning | Secret? | Supplied today via. Names only, never values.

#### Local development
Numbered setup steps with exact commands in code blocks: clone, venv, install, env files, run backend, run UI, run tests.

#### Deployment
The current procedure (e.g. `deploy.sh` → `cf push`), a manifest walkthrough, the service instances required, and post-deploy checks.

#### Logging, monitoring and troubleshooting
A signals table (Signal | Where | What it proves) and a troubleshooting table (Symptom | Likely cause | Action) with at least 6 rows drawn from code paths that raise errors.

#### Known limits and non-goals
A numbered list of the hard limits of the current implementation. With a BPD, name every process its scope table marks out of scope, using the BPD's name for it (the checker looks for each one).

---

### Part VI — Path to production `{.part}`

This is the part the implementation team will plan from. Keep it to what certainly has to change: security and integration back into the customer's systems. Build it with the checklist in [references/path-to-production.md](references/path-to-production.md) and the integration answers from the interview, and respect its "Do not raise" list. Don't include any time or effort estimate.

#### Production-readiness assessment
Gap register (`Table: … {#tbl-gaps}`): ID (G-01…) | Area | Current PoC state | Production target | Recommended action (SAP Business AI Platform service / pattern) | Priority (Must/Should/Could). Area is Security, Integration or (with a BPD) Functional scope; order the rows in that sequence. Every row must be traceable to evidence in code/config or to an explicit statement ("not present in repository"). Expect 5–10 rows without a BPD; an empty area is fine. With a BPD, add one row per requirement the PoC does not meet, citing it ("BR-06, see @tbl-requirements in Business scenario"); fold several shortfalls of the same requirement into that one row.

#### Target architecture
A second `type: platform` diagram showing the production target: the current components, plus approuter + XSUAA/IAS, Credential Store, and Destination + Connectivity/Cloud Connector to the systems named in the interview (e.g. ECC, S/4HANA, APO, IBP). Those new components get `status: planned`; don't add operational services (logging, alerting, CI/CD) as planned nodes. With a BPD, this is the production design: draw the diagram from the BPD's technical architecture and mark designed elements the PoC lacks as `planned`; an element gets a gap row only through the requirement it serves (stack differences only if the user said the designed stack is binding). Word it as the handover's own design, e.g. subtitle "Production design of the business scenario".

#### ROM cost calculation
A short section (half a page) with the rough order of magnitude of the monthly platform cost, built from the cost estimate the user provides. Follow [references/rom-cost.md](references/rom-cost.md) exactly: a two-paragraph introduction, the cost breakdown table (`{#tbl-rom-cost}`) classifying every line as **Shared** or **Solution-specific**, and one `[!NOTE]` on consumption-based lines. Nothing else: no customer scenarios, no annual figures, no commentary on the numbers.

#### Integration plan
One subsection per system the final implementation must read from or write to (from the interview or the BPD's data-source table, e.g. ECC as source and target). For each: direction (inbound/outbound), business objects and the fields they feed, protocol/API (OData, RFC/BAPI, IDoc, CDS extraction, SAP Integration Suite flow), connectivity (Cloud Connector, Destination), authentication (principal propagation vs. technical user), frequency/trigger, error handling and reconciliation, and what replaces today's manual steps (e.g. workbook upload, Excel export).

#### Phased roadmap
Phase 1 security hardening → Phase 2 integration (→ Phase 3 functional scope, only with a BPD). Table: Phase | Goal | Work items (gap IDs) | Exit criteria. No durations or dates, and no "scale & operate" phase. With a BPD, add one closing row listing each out-of-scope process as a later extension, or state that it stays out of scope.

#### Open questions and decisions
Table (`{#tbl-questions}`): ID (Q-01…) | Question | Why it matters | Owner (customer / SAP / partner). Only questions that block a gap row, the integration plan or the ROM (identity provider and roles, interfaces and fields per system, principal propagation vs. technical user, write-back approval, volumes), plus assumptions made because the interview was incomplete. No questions on topics from the "Do not raise" list.

---

### Appendices
- **Appendix A. Glossary**: business and technical terms (required).
- **Appendix B. Repository source map**: Path | Responsibility for every significant code directory/file (not `docs/`).
- **Appendix C. Dependency inventory**: key libraries with versions from requirements/package.json, plus licence notes where relevant (e.g. solvers).
- **Appendix D. Test inventory**: how to run the tests, what they cover, and known gaps.
- Optional: data dictionaries, full agent tool/prompt listings, benchmark tables.
