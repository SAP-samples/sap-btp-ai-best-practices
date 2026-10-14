# Merging a Business Process Document (BPD)

A BPD (the SAP Business Scenario Design Document) describes the business scenario the PoC was built for: the current process, pain points, requirements, KPIs, roles, data, scope, process flow and the designed architecture. It is an **optional** input. When the user supplies one, the handover gains a business chapter, traces every requirement, and uses the BPD's design as the production target. Without a BPD, skip this file entirely.

The BPD says what was *designed*; the code says what was *built*. Never copy a BPD statement into a current-state chapter without checking it against the code. Every requirement the PoC does not meet becomes one row in the gap register (section 6).

## 1. Absorb it, don't cite it

The handover **absorbs** the BPD: its content becomes part of the handover, written in the handover's own voice. The reader never needs the BPD, so the handover never names it. Never write "BPD" or "Business Scenario Design" anywhere in the handover, figures or file names. `check_handover.py --bpd` fails on both.

| Don't write | Write instead |
|---|---|
| "This chapter summarises the BPD." | Start with the content: "Engineering releases a change notice in the PLM system, which…" |
| Table headers "BPD role", "BPD data category", "BPD data element" | "Role", "Data category", "Data element" |
| "the nine BPD fields" | "the nine mandatory audit fields (BR-06, see @tbl-requirements in Business scenario)" |
| "follows the BPD technical architecture" | "follows the production design in @fig-arch-target" |
| "Phase 1: … BPD completion" | "Phase 1: … all business requirements met (@tbl-requirements)" |
| A glossary entry for BPD | No entry; the term never appears |
| `figures/bpd_flow_1.png`, `fig-bpd-flow-1` | `figures/process_flow_1.png`, `fig-process-flow-1` |
| Diagram subtitle "Business process design applied to the application" | "Production design of the business scenario" |

Point to the handover's own chapters and `BR-xx` rows instead of the source document.

## 2. Extract

```bash
python "$TOOLS/scripts/bpd_extract.py" <bpd.docx> --out handover/bpd
```

- `handover/bpd/bpd.md`: the BPD as clean Markdown, with section numbers matching its table of contents. Read it in full.
- `handover/bpd/bpd.json`: tables with a `kind` (requirements, objectives, pain_points, kpis, roles, data_model, data_sources, process_scope, architecture), plus `requirements` and `out_of_scope` lists. The checker reads it.
- `handover/bpd/media/`: images. Wide process flows made of framed panels are also cut into `img_NN_panel_K.png`. Open every image and panel and name each panel from its title.

**Only diagrams and images with real content go into the handover**: process flows, architecture sketches, data-model or screen drawings. The cover photo (flagged `"cover": true` in `bpd.json`), logos and decorative pictures never do. When unsure whether an image adds information, leave it out.

The extractor targets the standard template. If the BPD is laid out differently (other headings, a requirement ID column with another name, no "In Scope" column), adapt `KINDS`, `ID_HEADER` or `IN_SCOPE_HEADER` in your copy of the script rather than fixing its output by hand.

**Keep the extract out of Git.** `handover/bpd/` holds the raw customer document; suggest adding `handover/bpd/` to the project's `.gitignore`. It is source material, not a deliverable: `check_handover.py --bpd` also skips it in the `--forbid` scan. Figures the handover uses are **copied** into `figures/` under neutral names (`figures/process_flow_1.png`), never linked from `handover/bpd/`. When the customer must not be named, the anonymisation rules apply to everything you carry over.

## 3. Questions for the user before merging

Ask these in the interview round (step 3 of the skill), after reading the BPD:

- **Is the designed technology stack binding?** A BPD often names a stack (e.g. CAP with a Fiori UI5 front end) that differs from what the PoC uses (e.g. FastAPI with UI5 Web Components). Ask before turning the difference into gap rows or open questions. If the implemented stack is accepted, the handover does not mention the designed stack at all: the target architecture uses the implemented runtimes, and no gap row or question refers to the difference. Only if the designed stack is binding does it become the target, with gap rows for the migration.
- Anything the BPD leaves open: integration targets, roles, scope.

## 4. Where each BPD section goes

| BPD section | Destination in the handover | Treatment |
|---|---|---|
| 1 Scenario description: current process, challenges, proposed solution, expected benefits | New chapter 1 **Business scenario** | Kept together, rewritten in the handover's voice (platform naming, tense, customer names if anonymising) |
| 2.2 Pain points | Chapter 1 | Kept as a table with the source's columns (e.g. Pain point \| Impact on business) |
| 2.1 Business objectives + 2.3 Requirements | Chapter 1, **requirements table** `{#tbl-requirements}` | The two tables repeat each other: merge them into one (section 5) |
| 2.4 KPIs | Chapter 1 KPI table with the source's own fields (e.g. KPI \| Calculation); link from Part IV *AI quality* | KPIs that measure the AI (precision, recall, accuracy) are reported against their target in *AI quality*, with the PoC results. Don't add a column the source doesn't have (e.g. "Target") |
| 3.1 Roles | *Users, roles and access* | Designed personas next to the roles actually implemented; missing roles feed the security gap on authorization |
| 3.2 Data model: agent input / output fields | Part IV (capability map inputs → outputs, structured outputs) | Compared with the real schemas and models; a missing field counts toward the requirement it serves |
| 3.2 Data model: data categories, source tables (e.g. MARA/MARC, STKO/STPO) | Part II *Data model and persistence*, "Source data lineage" subsection; the Glossary | Which source table feeds which entity, and how it gets there today |
| 3.3 Data sources: PoC / current interface | Part II *Architecture overview*, "Current data intake" | How data enters today (flat files, uploads, extracts) |
| 3.3 Data sources: production interface | Part VI *Integration plan* (one subsection per system) and the target architecture diagram | Answers the interview's integration question; skip it unless the BPD is silent or unclear |
| 3.4 Process scope: in-scope rows | Chapter 1 scope table (Process \| Description \| Solution as designed) | The "Solution" column is checked against the code; a mismatch counts toward the requirement it serves |
| 3.4 Process scope: out-of-scope rows | Part V *Known limits and non-goals* and Part VI *Phased roadmap* | Name each process as the source does; the checker looks for every one |
| 3.4.1 Process flow: narrative | Seeds the *End-to-end journey* in *Solution at a glance* | The journey is rewritten from the code, not copied |
| 3.4.1 Process flow: image | Chapter 1, landscape page, one figure per panel, copied to `figures/process_flow_<n>.png` | Caption "Business process flow (as designed): <panel title>" |
| 3.5 Technical architecture | Part VI *Target architecture* | The target `platform` diagram is drawn from it (section 7). Every designed element the PoC lacks is `"status": "planned"`; it gets a gap row only through the requirement it serves, and stack differences only if binding (section 3) |
| Cover page and cover photo, logos, decorative images, table of contents, template instruction text | Not reused | Never copied into `figures/` |

## 5. The requirements table

The objectives (2.1) and requirements (2.3) tables usually carry the same items with the same numbers, and the objectives table repeats one KPI and one target benefit on every row. Merge them into a single table:

- **ID**: `BR-` plus the source's own number, two digits: requirement 5 is `BR-05`. Keep the gaps; if the source skips 4, so does the table. Never renumber.
- **Requirement**: the requirement description, lightly edited.
- **Expectation**: the acceptance expectation.
- Keep another source column only if it carries information not already in the table. A column whose value is the same on every row (e.g. "Business Case KPI (financial)", "Target Benefit") is dropped from the table; state it once in the KPI part of the chapter if it adds anything, and record the drop in the mapping (section 8).
- No PoC status column. What the PoC does not meet is a gap row.

```markdown
Table: Business requirements {#tbl-requirements}

| ID | Requirement | Expectation |
|---|---|---|
| BR-01 | Automated ingestion and analysis trigger | A new source file starts the analysis without a manual step |
| BR-02 | Structure traversal and impact classification | Every affected object is identified with its classification and evidence path |
| BR-05 | Human approval before any downstream action | Reviewers can approve, override or escalate each item |
```

## 6. Gaps against the design

Check each requirement in `@tbl-requirements` against the code. Each requirement the PoC does not meet becomes **one** row in `@tbl-gaps`, Area "Functional scope", in the register's normal format, citing the requirement: "Production target: an audit record per decision with the nine mandatory fields (BR-06, see @tbl-requirements in Business scenario)". When several roles, data fields, in-scope solutions or designed elements fall short for the same requirement, list them in that one row's current state; don't split them into separate rows. A shortfall that serves no requirement is not a gap. Designed-but-missing architecture elements (an audit log, an autonomy-control layer, notifications) appear as planned nodes in the target diagram. Technology-stack differences follow the user's answer from section 3.

## 7. Target architecture wording

The Part VI target architecture is the production design of the scenario. Word it as the handover's own design, not as a document it was copied from: title "Target production architecture", subtitle like "Production design of the business scenario", and text such as "The target architecture adds…". Avoid subtitles that spell out a design document ("Business process design applied to the application").

## 8. Mapping note (`notes/bpd-mapping.md`)

Save a mapping from every source table to its place in the handover, so the author (and the user, if asked) can check that nothing disappeared:

```markdown
| Source table (section) | Ended up in | Merged / dropped |
|---|---|---|
| Business objectives (2.1) | @tbl-requirements | Merged with 2.3 (same IDs). Dropped "Business Case KPI (financial)" and "Target Benefit": same value on every row; stated once in the KPI section |
| High-level requirements (2.3) | @tbl-requirements | Kept: ID → BR-xx, description, expectation |
| KPIs (2.4) | Business scenario, KPI table; AI quality | Kept as KPI \| Calculation (no added columns) |
| Process scope (3.4) | Scope table; Known limits; Phased roadmap | In-scope rows → scope table; out-of-scope rows → limits and roadmap |
```

- Every source table gets a row, including the ones that were not reused ("Not reused: template instructions").
- A dropped column is always named with the reason, so nothing disappears silently.
- Columns the source doesn't have are never added in the handover (e.g. no "Target" column in the KPI table if the source has none).

## 9. The Business scenario chapter

Chapter 1, first in Part I, only when a BPD is supplied:

1. **Current process and challenges**
2. **Proposed solution and expected benefits**
3. **Pain points** (table)
4. **Business requirements** (`@tbl-requirements`)
5. **KPIs** (table)
6. **Process scope** (in-scope table; one line pointing to *Known limits and non-goals* for what is out of scope)
7. **Business process flow (as designed)**: the panel figures on a landscape page

Write it in the handover's own voice: open with the business content itself, present tense for what the scenario designs, and no claims about what the PoC does (those belong in the usage and architecture chapters).

## 10. Platform naming in carried-over text

The source text uses the old platform name throughout. House rule 7 applies to everything carried over:

| Source text | In the handover |
|---|---|
| "hosted on SAP Business Technology Platform (BTP)" | "hosted on SAP Business AI Platform" |
| "the BTP application layer" | "the application layer" / "the CAP application" |
| "hosted on SAP BTP" | "hosted on SAP Business AI Platform" |
| "SAP BTP Destination Service", "SAP BTP Audit Log" | allowed as written (service names) |

## 11. Check

```bash
python "$TOOLS/scripts/check_handover.py" handover/handover.md ... --bpd handover/bpd/bpd.json
```

- **Error**: "BPD" (in any case, including `bpd_` file names and `fig-bpd-` labels) or "Business Scenario Design" anywhere in the handover.
- **Error**: no *Business scenario* chapter, or no table whose first column holds `BR-xx` IDs.
- **Warning**: a source requirement without its `BR-xx` row (dropped requirement).
- **Warning**: a source out-of-scope process not named between Part V and the appendices. The checker looks for its first two words (e.g. "Global CIB"), its first and last word, or its parenthetical expansion.
