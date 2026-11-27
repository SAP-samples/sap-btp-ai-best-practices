---
name: sap-settings
description: Decision scenarios and SAP settings after a run - what each SAP tab is, every SAP settings rule with its default, why a tab is blocked and how to fix it, how to select a scenario, generate SAP settings and propose rule changes.
---

# SAP settings after a run

After an optimization run completes, people choose ONE decision scenario and turn
it into SAP settings: the same tabs as the customer's "Wheel settings for SAP"
template, downloaded as an XLSX that planners copy into the SAP upload template.
Automatic upload to SAP is production-phase work; in the POC the output is the
XLSX.

Tools:
- `get_decision_scenarios`: Balanced (B, knee of the curve), CO1..CO3
  (changeover-oriented: fewer changeovers, more stock), INV1..INV3
  (inventory-oriented: less stock, more changeovers), plus the stored selection.
- `select_decision_scenario`: store the chosen point. Direct write.
- `get_sap_settings_status`: per tab, what a generation would give, each gap with
  its fix, the plant profile and revision whose rules apply, every rule with its
  value and source, suggestions learnt from an earlier wheel, and whether
  generated settings are stale.
- `generate_sap_settings`: generate for the stored selection. Direct write.
- `propose_profile_settings` with `settings_patch={"sap_settings": {...}}`:
  change a rule. Approval card; nothing is saved until the user approves.

## When to act

- Select a scenario or generate only when the user asks ("generate the SAP
  settings for the balanced one", "use CO1"). Otherwise offer.
- A new selection deletes settings generated for the previous one. If
  `get_sap_settings_status` shows a `last_summary`, tell the user and confirm
  before `select_decision_scenario`.
- After reporting a completed run (wait_for_run returns `sap_settings_offer`),
  end the report with one offer: "Do you want me to select scenario B and
  generate the SAP settings?" Name any blocked tab and its fix in the same
  sentence or the next one. Do not generate unprompted.
- Default planning period: the current quarter (e.g. 26Q4). Ask only when the
  user mentions another period.
- After generating, say which tabs are ready, partial or blocked, and that the
  XLSX is downloaded from the SAP settings card.

## Where the rules live

The rules are a section of the **plant profile** (Settings > SAP settings). They
do not change optimization results. Generation reads the profile's **current**
revision, so a missing code can be fixed after the run, without re-running.
Settings generated before a profile change are `stale`: offer to regenerate.

Always use the `profile_id` that `get_sap_settings_status` returns (the run's
profile). It can differ from the profile selected in the conversation.
`profile.origin`:
- `live`: editable.
- `frozen`: the profile was removed. The copy saved with the run is used and
  cannot be edited.
- `none`: the run has no profile. Defaults apply, and plant codes cannot be set.

## Tabs

| Tab key | Name | Needs |
|---|---|---|
| `pv_check` | Check PV in subgroup | Lists subgroups mixing production versions (partial when any). |
| `kpis` | RoC and change overs | Wheel RoC, batches and changeovers per week (cross-checked with Changeover). |
| `cycle_stock` | Cycle stock | Dataset DC network and pallet tables (inputs workbook). The ABC table completes materials outside the wheel. |
| `planning_calendar` | Planning calendar | Odd- and even-week lot-size profiles in the profile. |
| `recipe_combination` | Recipe combination | Recipe characters: the dataset's RecipeCharacters table, else the profile fallback. |
| `apo_lot_size` | APO lot size | Recipe combination ready, plus the production-version KG table. |
| `central_dc_lot_size` | Central DC lot size | Only for plants with an intermediate DC (`multi_tier_dc`). Not ported: always blocked there. |
| `runner_audit` | Runner RoC audit | Evidence only. |

## Rules (profile `sap_settings`)

| Planner may say | Key | Default | Meaning |
|---|---|---|---|
| "odd/even week profile", "planning calendar codes", "week 1 / week 2 calendar" | `calendar_odd_profile`, `calendar_even_profile` | empty (tab blocked) | Plant customizing codes `<profile>-<calendar>`. The calendar uploaded is the text after `-`. They must differ. The SEFI with the largest forecast always gets the odd-week profile. |
| "two-weekly frequency" | `calendar_frequency_code` | `02W` | Frequency that gets a planning calendar. Other FINIs get a reset row. |
| "calendar lot size", "PK" | `calendar_lot_size` | `PK` | Lot-size procedure for calendar-planned FINIs. |
| "reset lot size", "EX" | `calendar_reset_lot_size` | `EX` | Lot-size procedure for FINIs leaving the calendar. |
| "reset text" | `calendar_reset_text` | `Reset via ticket - to be blank` | Text in the calendar/profile columns of reset rows. |
| "recipe characters", "characters allowed in BW" | `recipe_characters` | empty | Fallback only, when the dataset has no RecipeCharacters table. One character per subgroup of a SEFI: digits, then letters, then symbols. |
| "APO default PV" | `apo_default_pv` | `PV - 1` | Version APO already uses. Its rows are not uploaded. |
| "changeovers per filling line" | `kpi_line_breakdown` | false | Adds per-line KPI rows. |
| "intermediate DC", "central DC", "multi-tier" | `multi_tier_dc` | false | Central-DC tab applies (reported blocked). |
| "minimum lot at DC", "at least 300 litres for C items" | `dc_min_lot` | all classes: `pallet` | Per ABC class: `pallet`, `half_pallet`, `one_piece`, `litres` (with `number`), or `none` (row goes to "without pallet"). |
| "rounding value", "round to a layer" | `dc_rounding` | all classes: `layer_at_max_days` | Per ABC class: `layer_at_max_days` (a layer at the maximum days, else a pallet), `pallet`, `layer`, `half_pallet`, `one_piece`. |
| "cycle days for A/B/C items outside the wheel" | `abc_cycle_days` | A 5, B 10, C 20 | Days for materials not in the wheel. `unclassified` is for materials without a class (empty skips them). |

Per-class shape: `{"all_classes": {"rule": "pallet"}, "C": {"rule": "litres", "number": 300}}`.
A class set to `null` means "same as all classes". Patches are merged key by key,
so send only what changes, e.g.
`{"sap_settings": {"calendar_odd_profile": "ODD-W1", "calendar_even_profile": "EVEN-W2"}}`.

Runner thresholds and the cycle-stock strategy (`runner_thresholds`,
`cycle_stock_policy`) are profile settings too. They were frozen with the run,
because they shaped the optimization. Changing them needs a new run.

## Blocked tabs and fixes

Each gap in `get_sap_settings_status` has a `code`:
- `profile_setting`: a profile rule is empty (usually the calendar codes).
  - If `suggestions` holds a `suggested` entry, show its values, the dataset it
    came from and its message (an earlier wheel of the plant used them). Then,
    if the user agrees, call `propose_profile_settings` with its `patch`.
  - Without a suggestion, ask the user for the codes from the plant's SAP
    customizing. Never invent or guess codes.
- `dataset_view`: the dataset lacks a table. Example: RecipeCharacters is in the
  newer inputs (enrichment) workbook. The fix is to recreate the dataset with
  that workbook. It cannot be fixed in the profile, except the
  recipe-character fallback if the user gives the list.
- `depends_on`: fix the tab it depends on first (APO waits for recipe combination).
- `not_ported`: production-phase work (intermediate-DC settings).

Missing inputs never become invented values. A blocked tab keeps its headers
in the XLSX with a red BLOCKED line. Generating with blocked tabs is allowed:
the other tabs are still useful.

## Manual changes to groupings

The POC exports results as Excel in the customer's production-wheel format. A
planner can edit groupings there, but the export contains values, not formulas.
After moving a FINI, they copy Subgroup and Selected PV into their own
product-groups workbook so RoC and pallets recalculate, then use their own
SAP-settings template. The app cannot read an edited wheel back in yet.
