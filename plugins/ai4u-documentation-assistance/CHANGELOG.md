# Changelog

All notable changes to `ai4u-documentation-assistance` are recorded here.

## 0.3.0 - 2026-10-02

- `technical-handover` path to production narrowed to security and
  integration. `references/path-to-production.md` keeps two checklist areas
  (security: authentication, browser API keys, authorization, secrets,
  exposed surfaces, back-end identity; integration: inbound extraction,
  write-back, connectivity, named non-SAP sources) and adds a "Do not raise"
  list (HA, CI/CD, observability, UI hosting, tests, dependencies, model
  choice, retention, audit logging, optimizer backlog) that never becomes a
  gap, roadmap item or open question.
- Gap register: expected size 5-10 rows instead of 15-30; areas ordered
  Security, Integration, Functional scope.
- With a BPD: one gap row per unmet requirement instead of one per
  difference; roles, fields and designed elements count toward the
  requirement they serve.
- Target architecture adds only identity, Credential Store and connectivity
  as planned nodes (plus BPD-designed elements); roadmap has two phases
  (security hardening, integration), plus functional scope with a BPD, and no
  "scale & operate" phase; open questions limited to what blocks a gap row,
  the integration plan or the ROM.

## 0.2.0 - 2026-09-30

- `technical-handover` is now a skill folder: `SKILL.md` (workflow, house
  rules, chapter blueprint), `references/`, `scripts/` and `assets/`. Step 0
  copies the tools into `handover/.tools/`, where they may be adapted per
  project.
- Optional Business Process Document (BPD) input: new `scripts/bpd_extract.py`
  (clean tables, requirements, out-of-scope processes, images, process flows
  cut into panels), a "Business scenario" chapter, a requirements table with
  `BR-xx` IDs mirroring the BPD numbering, BPD gaps in the gap register and
  the BPD architecture as the Part VI target (`references/bpd-merge.md`).
- `check_handover.py --bpd`: errors on a missing Business scenario chapter or
  requirements table; warns on dropped requirements and on out-of-scope
  processes not named in Part V/VI.
- Commits, branches and the source repository are left out by default (the
  code is delivered in a new repository); `repository`/`commit` are optional
  front-matter keys. New `header` key keeps the page header short.
- Interview asks for the production model (checked against the configured
  model and the cost estimate); repository docs that contradict the code are
  reported to the user.
- The handover absorbs the BPD and never names it (`check_handover.py --bpd`
  fails on "BPD" / "Business Scenario Design"); only relevant BPD images are
  reused; the designed technology stack is only a target if it is binding;
  `notes/bpd-mapping.md` records where every BPD table went.
- ROM cost: the AI Core line is sized as CU per transaction x monthly
  volume, and the user is reminded to regenerate a shared estimate PDF.
- `repo_inventory.py` finds FastAPI decorators whose path is on the next
  line; `render_diagram.py` keeps service labels inside their frame;
  sequence `sections[].at` is documented.
- Branding rule: "BTP" is allowed inside service names ("SAP BTP Destination
  Service") and only flagged when it names the platform itself.

## 0.1.0 - 2026-09-28

- Initial release with the `technical-handover` skill: SAP-styled technical
  handover Word handbooks for application and PoC codebases, with embedded
  tools for repository inventory, diagram rendering, screenshot placeholders,
  house-rule checks, and .docx building.
