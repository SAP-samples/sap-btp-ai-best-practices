---
name: dataset-import
description: Propose extraction settings (production frequencies, demand days, productive weeks, canonical factor) for the workbook the user is importing on the Datasets page.
---

# Dataset import settings

Use this skill when the user is on the Datasets page and talks about which FINIs or
frequencies to import, demand days, productive weeks or the canonical factor of a
new snapshot. These settings only affect the next "Upload and extract"; they never
change a snapshot that already exists.

## Always start from the workbook

1. Call `get_pending_import` first. It returns the chosen workbook, the production
   frequency codes it contains with FINI row counts, values without a weekly code,
   the overrides already applied and the defaults.
2. If `pending_import` is null, ask the user to choose the production workbook in
   "Import a snapshot". Do not propose settings yet.
3. Never ask the user to type the list of frequency codes. The codes come from the
   workbook. When you list them, use the counts from `get_pending_import`.

## Mapping requests to `propose_extraction_settings`

- "Use all frequencies", "do not filter on 01W and 02W", "include everything":
  `all_available_frequencies=true`. The upload then admits every weekly code found
  in the workbook.
- "Also model 04W", "drop 16W": `modeled_frequencies` with the complete resulting
  list. Start from the current overrides (or the 01W / 02W default).
- "Use 240 demand days": `demand_days=240`. Explain that, when not overridden, the
  import takes the demand days from the workbook's own source horizon when it has
  one, and 250 otherwise.
- "Back to default" for one setting: put its name in `reset_to_default`.
- Pass only what changes. Earlier applied overrides are kept.
- `summary` must say in plain words what the user approves, for example
  "Model all 7 frequency codes in the workbook (1,180 of 1,200 FINI rows)."

## Values without a weekly code

Rows such as "999.Not possible to determine" have no weekly frequency. They stay
out of scope with every setting. After extraction they are not repeated as a
warning: the review lists them, with the codes left out by the settings, under
the collapsed "Scope set by the extraction settings" notes.
Say how many rows this affects. To model them, the source frequency must be
corrected in the workbook.

## After proposing

The proposal appears as an Apply card in the chat. Nothing changes until the user
clicks Apply. Then the "Extraction settings" panel shows the applied values, and the
user clicks "Upload and extract". If the user picks another workbook, older cards
can no longer be applied: call `get_pending_import` again and propose anew.
