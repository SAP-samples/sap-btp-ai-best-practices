---
name: dataset-review
description: Explain the extraction findings and scope notes of a dataset review (why products were excluded, what a warning means, how it affects optimization runs and how to fix the workbook).
---

# Dataset review findings

Use this skill when the user asks about a finding or warning on the Datasets
review, for example "why do these products not have enough information?", "what
does this warning mean?", or a question sent by an "Ask the Assistant" button
that names a check (`rule_id`) and a reason.

## Always ground the answer in the snapshot

1. Call `explain_dataset_finding(rule_id, reason)`. The dataset defaults to the
   one selected on the page. If no dataset is selected, ask the user to select it.
2. If the user did not name a check, call `inspect_dataset(dataset_id)` and pick
   the check from its `issues`; ask when several could match.
3. Never guess workbook values. Quote the affected rows the tool returns
   (material, SEFI, Selected PV, forecast, lines, lot sizes). When
   `truncated` is true, say how many rows exist in total and that you show a part.

## How to answer

Answer in four short parts, in the user's language:

- **What happens**: the cause from `help.cause` (and `help.reason.cause`), with
  the workbook source columns from `help.reason.source`.
- **Evidence**: a short table of affected products with the fields that prove the
  cause. For production-version reasons, add the SEFI's production versions and
  lot sizes from `production_versions_by_sefi` (an empty list means the SEFI has
  no production version in the workbook; a null lot size means the cell is empty).
- **Effect on runs**: use `pv_mode_outcome`:
  - `excluded_in_both`: no run includes them, with FIXED or OPTIMIZED.
  - `modeled_only_with_optimized`: FIXED (the default of a new draft) leaves them
    out; OPTIMIZED includes them. Name these products.
  - `out_of_scope`: the extraction settings left them out of this snapshot.
  Say whether the finding blocks publishing: only errors do; warnings and scope
  notes do not.
- **How to fix**: `help.fix` and `help.reason.fix`. Workbook fixes need a new
  import ("Upload and extract"); a published snapshot never changes.

## Production version: FIXED or OPTIMIZED

`FINI_OPTIMIZED_PV_ONLY` lists products with no usable "Selected PV" whose SEFI
still has a production version with one positive lot size. The planner chooses
per run in the Optimizer workspace:

- FIXED keeps each product's selected production version; these products are
  left out.
- OPTIMIZED lets the optimizer choose one production version (lot size) per
  group from the SEFI's catalog; these products are included. Batch sizes can
  then differ from the current ones.

Snapshots imported before this check existed report these products under
`FINI_DATA_QUALITY` with reason `missing_fixed_pv`; `pv_mode_outcome` still shows
which of them OPTIMIZED includes. Say so when it is not zero.

## Scope notes

`PRODUCTION_FREQUENCY_UNDETERMINED` (values such as "999.Not possible to
determine") and `FREQUENCY_NOT_SELECTED` (weekly codes not chosen in the
extraction settings) are scope decisions, not data errors. The user already saw
them in the extraction settings. Explain that no setting can include non-weekly
values, and that unselected codes need a new import with those codes selected
(load dataset-import to propose them for a new upload).
