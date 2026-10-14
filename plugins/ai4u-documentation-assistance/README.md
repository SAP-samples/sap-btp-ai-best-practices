# AI4U Documentation Assistance

`ai4u-documentation-assistance` packages agent skills that produce delivery
documentation for SAP AI applications and proofs of concept.

## Skills

- `technical-handover`: builds an SAP-styled Word handbook (.docx) that lets an
  implementation team understand, run, extend, and productionise a codebase.
  It covers usage, SAP Business AI Platform architecture diagrams, the API
  catalog, AI capabilities, and the path to production.
  It can also merge a Business Process Document (BPD) into a business
  scenario chapter with requirement traceability.

## Compatibility

The plugin contains one shared `skills/` tree and three manifests:

- `plugin.json` for Agent Plugins 1.0 compatible clients.
- `.codex-plugin/plugin.json` for Codex.
- `.claude-plugin/plugin.json` for Claude Code.

Installation is documented in the repository README. Plugin details, inputs,
outputs, and testing are documented in
`../../docs/documentation-assistance-plugin.md`.

## Tool packaging

`technical-handover` is a skill folder:

- `SKILL.md`: principles, house rules, workflow and the chapter blueprint.
- `references/`: detail read on demand (path-to-production checklist, diagrams,
  Markdown conventions, ROM cost, BPD merge).
- `scripts/`: the helper tools (inventory, diagram renderer, placeholder
  generator, checker, Word builder, BPD extractor).
- `assets/`: `brand.json` and an example architecture diagram spec.

Step 0 of the skill copies `scripts/` and `assets/` into the target project's
`handover/.tools/` folder, so each project can adapt its own copy of the tools.
