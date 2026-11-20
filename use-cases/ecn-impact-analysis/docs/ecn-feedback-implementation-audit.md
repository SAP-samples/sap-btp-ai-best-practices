# ECN feedback implementation audit — 24 September 2026

## Result and scope

The initial branch already contained the publisher, BOM change engine, plant
projection, v4 prompts and four ECN pages (commits `2e0d3d9` through `69e9d1b`).
This audit checked those implementations against the supplied design, repaired
correctness and completeness gaps, and exercised the local application against
development HANA. No Cloud Foundry deployment or remote push was performed.

## Implementation status

| Design area | Existing implementation | Completed in this audit |
| --- | --- | --- |
| Reference publishing | Manifest, CLI progress and contextual uploads | Verified matching real publications already in HANA; reused them instead of creating duplicate datasets. Extended the real-workbook acceptance suite to master, purchasing and stock records. |
| BOM changes | MBOM history, pairing and phase flags | Preserve alternatives/repeated parts; handle engineering-only changes; merge both engineering sides into manufacturing pairs; expose BOM-only removals in part inspection; retain uncertainty for missing dates. |
| Plant impact | Green/amber rows and stock/purchasing columns | Include quantity-change parts in cross-plant checks; require the correct dates before green status; display absent stock units explicitly. |
| Change summary | Expandable lists and program dropdown | Complete source membership with types/revisions, no first-100 truncation; one Readiness count derived from exactly the displayed findings, including grouped source warnings. |
| AI language | Three-question executive summary, shared v4 style | Shorter summary guidance; 550-character factor cards; separate engineering/manufacturing dates; bounded BOM/plant examples with full totals; provider schema enforces permitted citations and required gaps. |
| Pages and chat | Overview, Impact, attention and references routes; shell chat | Fixed sidebar clicks that reloaded the app and lost chat drafts. References contain exactly three complete, paged lists with file/sheet/row/position/date/type/state. Source tools and review stay collapsed on Overview; the original description is collapsed in the shared header. |
| Documentation and acceptance | Design existed; real fixture was untracked | Feature guides updated, real fixture completed and committed, full test suites and live checks recorded here. |

## Real-data acceptance

Data comes from the Q900V BOM workbook dated 4 September 2026 and the material,
purchasing, stock and ECN exports dated 16 September. Existing HANA publications
use data group `real-q900v`, program Q900V and plant PL10 Plant A.

**ECN345274:** four replacements across Q923VV2526401 and Q923VV2526404:
QKX4638810 → QKX463881A at position 0150, and QKX8914160 → QKX891416A at
position 0160. Quantity stays 2. Engineering release is 16 January 2026;
old manufacturing rows end 29 June and new rows start 30 June. New/fully-phased-out
flags match the supplied coverage. The program dropdown shows Q900V's 6 EBOM and
8 MBOM rows; the five other declared programs explicitly have no uploaded BOM.
PL10 is the only linked plant. No material master, purchasing or stock records
for these parts are inferred from their absence.

**ECN346863:** three removals in Q900VV0080402 end 18 May 2026; QKX7702260
starts 19 May, separately because the three-to-one relationship is ambiguous.
Quantities for 8241814T1 and 33380541 change from 1 to 2. The remaining additions,
including QKW8860810 quantity 4 in both new assemblies, match the workbook.
The overview has 15 change rows including four new assemblies. PL10 is green;
PK01 is amber because it has purchasing/stock links without a confirmed BOM
change. QKX2833570 has stock quantity 5 at PL10, with the missing unit stated.
Its removed-part dialog opens the original MBOM row 7495 and expiry date.
3388708T1 is correctly still used in another assembly; reused incoming parts
are distinguished from genuinely new ones.

**Original synthetic issue:** SIM-ECN-001 still has its historical upload
warning count of zero, but that value no longer drives the detail card. The
current snapshot exposes four attention items and four matching Readiness groups:
missing BOM location, phase-out still used, manufacturing introduction, and
missing stock fields.

## Verification

Commands were run locally against the final code changes:

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  -m 'not hana_live and not llm_live and not development_fixture'
DEVELOPMENT_FIXTURE_DIR=data PYTHONPATH=api .venv/bin/python -m pytest -q \
  -m development_fixture tests/integration/test_real_bom_changes_fixture.py
npm --prefix ui test
npm --prefix ui run test:e2e
```

- Backend: **964 passed**, 18 explicitly excluded live/development tests; four
  existing dependency/DB-API deprecation warnings.
- Real-file acceptance: **4 passed**.
- UI unit tests: **72 passed**.
- Browser tests: **9 passed**, including navigation without reload/draft loss
  and responsive containment. The production build passed; its bundle-size
  advisory remains.
- Browser acceptance: both real overviews, program filtering, missing-program
  messages, separate impact page, collapsed/expanded Readiness, three reference
  lists and removed-part source dialog inspected against live HANA.

## Live AI failures found and repaired

The first ECN345274 regeneration failed because the model invented a citation
ID. The schema now restricts each stage/section to its allowed IDs, and runtime
validation remains strict. A subsequent ECN346863 summary marked a section with
a required source gap as sufficient. The schema now fixes that section's status
and requires a missing-fact item. Both defects have failing-first regression
tests. Factor text was additionally capped after live inspection showed long
lists despite the plain-language prompt.

Final live regeneration completed for **both** ECN345274 and ECN346863 with
`ecn-snapshot-v4` and no job error. Manufacturing/procurement/configuration cards
were 408/353/437 characters for ECN345274 and 452/449/376 for ECN346863.
The original SIM-ECN-001 example also completed in v4, with 317/440/389-character
factor cards and explicit unresolved phase-out and missing-placement statements.

A live chat question exposed a separate retrieval problem: the assistant searched
ECN JSON dates and missed dates already available in the BOM. Chat now has a
paged `list_bom_changes` tool, and its local skills route replacement/cut-over
questions there. The overview also excludes complete UI reference arrays, which
could otherwise exceed the tool budget and discard all useful context. Both
behaviors have failing-first regressions.

The corrected live chat then returned both replacement pairs, PL10 Plant A,
engineering release 16 January, manufacturing cut-over 30 June and old-part end
29 June in one tool call. Final factor inspection also found one reversed
replacement sentence; a narrowly scoped deterministic prose guard now follows
known directed BOM pairs, leaving hypothetical and ambiguous pairs alone.

The prose guard's first live run exposed a strict-enum round-trip issue. The
normalizer now changes only ordinary strings, preserving validated status and
severity enums. A dedicated strict-model regression covers this boundary.

The original synthetic ECN revealed contradictory intent and BOM rows:
SIM1100/SIM3300 are declared phase-out, but stamped BOM rows record starts without
ends. Those rows now have `DECLARED_PHASE_OUT`, outgoing parts are marked
`PHASE_OUT_NOT_CONFIRMED`, and plant status is amber. They are not counted as new
parts. Incoming ECN parts with no BOM placement remain in the summary's intended
changes instead of disappearing behind the observed BOM rows.

## Review decisions and boundaries

All five independent-review findings were reproduced and fixed with failing-first
tests: excessive synthesis input, duplicated engineering removals, hidden mixed
dates, suppressed future introductions and incomplete paired plant dates.
There are no deferred review findings. The reviewer did not judge live HANA,
provider behavior or browser rendering; those checks were performed separately.

Decisions made during the audit:

1. Continue the existing `dev` checkout and preserve unrelated `.claude/` files.
   This keeps the user's in-progress branch intact; rollback is by the detailed
   milestone commits rather than a separate worktree.
2. Build reference lists from the complete assessed snapshot, with a legacy API
   fallback. This includes BOM-only removals and exact source coordinates; it
   increases snapshot response size for very large assessments.
3. Keep Engineering review, source browsing and Data used collapsed on Overview.
   References stays limited to the requested three lists; those tools require
   expanding their Overview sections.
4. Bound only the AI examples, preserving full totals and omission counts. Very
   large analyses summarize representative rows; the UI/API retain all records.

Known data boundaries remain explicit: only Q900V has an uploaded BOM, PK01 has
no supplied friendly name, stock units are missing, and BOM validity alone does
not prove a configured production variant. Existing saved analyses are retained;
Regenerate summary creates a fresh v4 result. Deployment remains a manual user
step.

Final verification on the running local API confirmed all three jobs are
`COMPLETED`, all use `ecn-snapshot-v4`, and every factor card is within its
550-character limit. Snapshot attention counts equal displayed question counts
for all three (0, 1 and 4 respectively). ECN345274's final prose names both
replacement directions correctly. Live chat returns the separate EBOM/MBOM
dates correctly in one tool call. The local API and UI were left running for
review; only the user's existing `.claude/` directory remains untracked.

## Rollback milestones

| Commit | Scope |
| --- | --- |
| `6c382d4` | Complete BOM facts, inspectable source membership and shared attention counts |
| `06a4d7a` | Four simple pages, complete reference lists and navigation preserving chat |
| `e167dd2` | Provider citation allowlists and shorter executive prompts |
| `c5fa253` | Five independent-review correctness fixes and bounded AI context |
| `cad4960` | Required-gap schema, short factor cards and collapsed source description |
| `8fac29f` | BOM-aware chat retrieval and directed replacement prose |
| `d47a676` | Strict enum preservation in replacement prose processing |
| `ecced0e` | Explicit phase-out/BOM conflict handling for the synthetic example |
