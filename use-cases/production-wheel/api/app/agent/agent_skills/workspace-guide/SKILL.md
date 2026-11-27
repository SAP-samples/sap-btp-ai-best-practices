---
name: workspace-guide
description: Explain application settings, BASE_GROUP, production versions, budgets, view groups, view members, exports, history removal and conversation reset.
---

# Workspace application guide

Use this skill whenever users ask what a control, configuration value, column or view means. Read get_optimizer_capabilities for live schemas and the selected context before discussing actual current values. These definitions come from workspace models, solver schemas and reporting code. Never infer the user's selected value from a default. Use settings-vocabulary for plain-language names of settings and metrics; always say Changeover for the internal `j_ch` metric.

## Published dataset

The immutable uploaded snapshot used for this run.
It defines the plant, materials, source eligibility, demand, production versions and calendar defaults.
Only published, non-removed snapshots can start new runs.


## Production version

Sets how each group's production version is chosen, and which finished products the run includes.

FIXED (default): each FINI keeps its source production version ("Selected PV" in the production workbook). Group members must share a feasible fixed version. FINIs without a usable "Selected PV" are left out.
OPTIMIZED: the optimizer chooses a feasible version from the source catalog for each group. FINIs without a "Selected PV" are included when their SEFI has a production version with one positive lot size; batch sizes can differ from the current ones.

Validate lists the FINIs the chosen setting leaves out but the other one would include. The dataset review shows them as "only usable with the OPTIMIZED production version".

A production version determines nominal lot litres.
Effective batch also applies the canonical factor.


## Coverage objective

PARETO is the workspace's supported objective.
It samples trade-offs while minimizing both demand-weighted coverage days and Changeover.

Operations anchor: prioritizes Changeover.
Coverage-lean end: prioritizes inventory coverage.

Requested point count is a sampling budget.
It does not guarantee that many distinct nondominated points, or a complete frontier.


## Coverage basis

Sets how group coverage is calculated.

Every basis starts from the wheel RoC R: demand_days × effective batch litres / group demand, rounded to whole days like the workbook (at least 1 day).
Each product's wheel RoC is R × its runner multiple (R for Single RoC). Batches per week (days per week / R), changeovers and runner multiples use these values and never pallets.
Each product's inventory RoC raises its wheel RoC to cover one full pallet (workbook fxAdjustedRoC: below 0.9 pallets per batch, floor(RoC / pallets)).

BASE_GROUP: demand-weighted product wheel RoC, without pallets.
ADJUSTED_GROUP: demand-weighted product inventory RoC, with pallets.
WORST_FINI: the highest product inventory RoC in each group.

Demand-weighted mean uses group demand as weights.
BASE_GROUP means batch-based group coverage. It does not mean the historical baseline grouping.

When explaining a product's RoC, give the wheel RoC (fini_wheel_roc_days, always a whole multiple of the group's wheel_roc_days) and, separately, the inventory RoC with pallets (fini_inventory_roc_days). Never present the inventory RoC as the product's wheel RoC.


## Compatibility matrix

The UI shows readable mode names; stored values are unchanged. Sets how discouraged (AVOID) volume pairs are treated.

Avoid allowed (FLEXIBLE): AVOID pairs are allowed and reported, without trying to minimize them.
Prefer compatible (DIAGNOSTIC): AVOID pairs are allowed, but their number is minimized before coverage.
Strict (HARD): discouraged and forbidden pairs are never grouped.
Ignore table (OFF): no volume preference. Without a plant profile, matrix assessment is skipped.

With a plant profile, forbidden (N) pairs stay blocked in every mode.
Common-line eligibility and group caps always apply.
With a plant profile, the profile's mode is the default for a new scenario.
The scenario may change the mode; the volume pairs stay fixed by the profile.

Volume matching (profile setting) decides which matrix row a product uses.
exact: the matrix lists every package volume; a product whose volume is missing stops the run.
nearest_class: the matrix lists reference volumes (stops, default <1, 1, 2.5, 5, 10, 15, >15). Below the smallest stop a product uses the <n class, above the largest the >n class, otherwise the nearest stop (on a tie the smaller one): 1.1 L uses 1, 2.2 L uses 2.5, 4 L uses 5, 20 L uses >15.

### Matrix coverage and generating a matrix

Read `volume_matching` from `get_plant_profile_details` first: it changes what the
matrix rows are.

**Exact matrix** (`exact`). Every package volume of the modeled products must appear
in the profile matrix. Validation blocks a draft when volumes are missing and
returns `matrix_coverage` with each missing volume and its product count.
`get_dataset_volume_coverage` compares a dataset with a profile before any draft.

**Class matrix** (`nearest_class`, the default of new profiles and of the
customer's profiles). Rows are reference volumes ("stops") plus optional `<n` and
`>n` classes; each product takes the row of its nearest stop, so a new package
volume never stops the run. `<n` must name the smallest stop and `>n` the largest.
A run stops only when a product falls in a tail the matrix lacks (for example a
20 L product without a `>15` class). `get_dataset_volume_coverage` returns each
volume's `matrix_class` and `class_product_counts`.

To build or extend a matrix from planner wording, call `propose_profile_settings`
with `volume_compatibility`; the user approves the card before anything is saved.
- Exact matrix, add a dataset's missing volumes without touching current pairs:
  `dataset_id`, `keep_existing_pairs: true`, a `default_status` and rules.
  (`dataset_id` is refused for a class matrix: volumes are classified, not rows.)
- Replace the whole matrix: list `volumes` (or `dataset_id` /
  `include_profile_volumes`) without `keep_existing_pairs`.
- Wording to rules (applied in order, later wins; overrides apply last):
  "X-Y L go together and exclude the others" -> `range_group` min X, max Y,
  within Y, outside N. "Only volumes at most D litres apart" ->
  `max_difference_litres` litres D, within Y, outside N. "Only volumes within P %"
  -> `max_difference_percent` percent P (of the smaller volume), within Y,
  outside N. "Discouraged" means AVOID instead of N. Families -> `compatible_groups`.
  Single pairs -> `overrides`. Rules only compare numbers: pairs with `<n` / `>n`
  come from `default_status`, `inherit_from`, `compatible_groups` or `overrides`.

#### Changing the stops of a class matrix

A planner may want finer control ("add a 2 L stop", "treat 12.5 L on its own") or
a coarser one ("drop the 2.5 stop"). Workflow:
1. Preview the effect: `get_dataset_volume_coverage(proposed_stops=[...])` with the
   whole new axis. Report `reclassified_volumes` (volume, products, from, to).
2. Ask how the new stop behaves unless the wording says it: like a neighbour
   (`inherit_from`) or with explicit statuses (`overrides`). Never guess.
3. Propose with `keep_existing_pairs: true` so every other pair is unchanged:
   - add a stop: `volumes: ["2"]`, `inherit_from: {"2": "2.5"}` (2 behaves like
     2.5 and 2-2.5 is Y), `default_status` for anything left open;
   - remove a stop: `remove_volumes: ["2.5"]` (its products move to the nearest
     remaining stop; say which);
   - move a tail: adding a stop beyond `>15` means renaming it, for example
     `volumes: ["20", ">20"]`, `remove_volumes: [">15"]`,
     `inherit_from: {"20": "15", ">20": ">15"}`;
   - switch an exact profile to classes: `settings_patch: {"volume_matching":
     "nearest_class"}` together with the class matrix in one card.
4. Describe the returned `matrix_summary`: added/removed stops, pairs per status,
   changed pairs, and with a dataset in the conversation `reclassified_volumes`
   and `unclassified_volumes` (a run would stop on those).

Ask for the status of pairs the wording leaves open instead of guessing; then
describe the returned `matrix_summary` (added volumes, pairs per status, changed
pairs) in plain language.

### Outdated drafts

A draft freezes the plant profile revision it was created with. When the profile
is edited later (for example after approving a matrix proposal), validation
returns `profile_status.outdated` and a warning: the draft still uses the old
settings and matrix. Offer `recreate_draft_with_current_profile`; it keeps the
scenario's own choices and reports any skipped values. Validate the new draft.

### Failed runs

`get_run_failure_diagnostics` returns the solve process's own report
(`worker_failure`: failure type, error, stage, traceback) and the last lines it
printed (`worker_log_tail`). `profile_matrix_incomplete` means missing matrix
volumes: list them and offer to propose them. `out_of_memory` or a log tail saying
the process was killed by SIGKILL points to memory: suggest fewer block workers.
`solver_inconsistent` in `failure_types`, or block failures with error type
`SolverResultInconsistentError`, mean the HiGHS solver library returned a result
that contradicts the model (for example a false objective or a false
"infeasible"). This is not infeasibility of the plant rules or data: do not
suggest relaxing rules or longer runtimes. Say the installed solver version must
match the pinned one in the API requirements and the worker must be redeployed by
the user before rerunning.
When neither report nor log was saved (older runs), `recheck_run_inputs` re-runs
the input checks on the run's frozen configuration.


## Global member cap

Maximum FINIs in one group.
The quick control uses the governed base of 7.

7: selects HARD.
Above 7: selects BOUNDED_RELAXATION with max_excess = cap − 7.

Plant/SEFI group_size_overrides or typed max_group_size constraints define other local caps.
An empty extra-constraints list does not disable existing structural rules.


## Plant / SEFI scope

Selects which plant/SEFI blocks the run covers.

Empty array []: all eligible plant/SEFI blocks in the chosen snapshot.
Otherwise: explicit objects such as {"plant":"P1","sefi":"123"}. IDs are strings.

Whole population means the modeled population under this run's production-version mode.
Excluded, external or out-of-scope source rows remain evidence.


## Typed constraints

[] means no additional user constraints.

Supported kinds:
max_group_size: maximum.
group_size_relaxation: base_limit and max_excess.
must_link: materials together.
cannot_link: material pair apart.
fini_disposition: admission or exclusion.
fixed_pv, allowed_pvs, required_lines.
group_rule: conditional group/member predicates.
selection_bound: bounds a linear sum over selected groups across one block or a plant.

Each requires constraint_id and a governed approval_status.
scope limits application to plant/SEFI.
draft, approved and rejected are review states, not solver proof.
coverage_bound, freeze_assignment and volume_compatibility_override are currently deferred and rejected.
Get the live constraint schema before constructing JSON.
Unknown required inputs fail closed.


## Execution budget

frontier_points: requested global samples (2 to 100, default 30).
min_frontier_points: retained-point target (2 to 100, default 30, clamped to frontier_points). When duplicates or dominated portfolios leave fewer points, extra solves split the widest gaps with the same low-Changeover bias. The run reports frontier_shortfall if the curve has fewer distinct trade-offs.
block_options_per_block: retained candidate partitions explored per block (2 to 100, default 5). Not the number of individual candidate groups.
per_block_total_seconds: total solver time per block (default 120 s), shared across its solves. Not a guaranteed whole-run duration.
block_workers: concurrent blocks (1 to 16, default 2). More workers increase CPU and memory use.
global_epsilon_exponent: spacing bias (1 to 10, default 2). 1 is uniform; above 1 emphasizes low-Changeover points.
wall_time_seconds: overall job deadline (10 to 172800, default 3600).

Example: 33 points, 10 block options, 120 seconds, 4 workers, exponent 2.
More time cannot recover candidate groups omitted from a restricted pool.
For coupled business rules, per_block_total_seconds bounds joint solver stages for the requested frontier. Preprocessing is additional, and wall_time_seconds remains the overall job limit.


## Complete optimizer configuration

Apply advanced config: writes the entire config object to the shared draft.
Save draft: updates quick controls.
Validate the latest saved revision before launch.
Configuration identifiers are technical JSON keys; display labels use Changeover.
Nested fields are explained below.

The nested field definitions, with their plain-language names, planner phrasing, ranges and defaults, are in settings-vocabulary.

## Dataset and result views

groups: one proposed group per frontier point, with member count, PV, demand, coverage, common lines and group Changeover contribution.
members: one source FINI row per point, including excluded/out-of-scope rows. Selected fields are blank when unassigned. Repeated group metrics in members must not be summed as group totals.
solutions: one frontier point with whole-solution KPIs.
validation: structural validation findings.
block_audits/global_audits: solver attempts, bounds, gaps and proof scope.
matrix_pairs: package-volume compatibility evidence.
fini_master: canonical source material records.
production_versions: source PV lot catalog.
fini_line_eligibility: plant-scoped material/line eligibility.
line_corrections: FixedLine source overrides.
validation_issues: extraction quality issues.
pallet_resolution: source/enrichment evidence for resolved pallet litres.

Search filters only the selected view.
Pagination is not the full population. Use server aggregates for totals.


## History and conversation

Delete permanently removes a terminal run and its evidence. Deleting a snapshot also deletes its sources, extracted data, drafts and all associated runs/results. The UI asks for confirmation and there is no restore. Active work must finish or be cancelled first. Each workbook import creates a separate review snapshot requiring explicit publication. Clicking a selected snapshot or run again, or Deselect, clears its selection; deselecting does not cancel optimizer jobs. New conversation clears page chat history and stops the current chat turn; submitted optimizer jobs continue independently. To stop a job use Cancel selected run. A recursion failure affects the chat, not proof of job failure; rediscover the existing run before considering another launch.

## Export and line notation

Export production wheel XLSX uses the selected point and uploaded column order. Original Filling line text is retained verbatim. Effective eligible lines includes source corrections; Common eligible lines shows shared alternatives in the same notation. Examples: `- 2  3   -   -   -` means strings 2 and 3, `11 / 12 / 18 / 23` means those four IDs, `L1 / L2` means those named IDs. Never convert L1 into 1 or infer a scheduled assignment. Inspect Export notes for source versus proposal fields. Export current view CSV is the technical evidence view.

## Reports after a run

When the user requests a completion report, call wait_for_run once. Waiting streams job progress without repeated model calls. If wait_for_run returns wait_status=queued_not_started, explain that no worker has claimed the existing run and end the turn without waiting again or resubmitting. Read get_run_results after completion, then report actual representative points and the closest singleton count requested. Do not fabricate a full frontier or assume 33 requested samples means 33 distinct solutions. Historical numbers supplied by a user are comparison markers until matching population/formula evidence is established. Both coverage and Changeover are minimized. Business acceptance and global proof remain separate from structural validation.

End the report with the SAP settings offer from wait_for_run's `sap_settings_offer`: offer to select a scenario (Balanced by default) and generate SAP settings, naming blocked inputs and their fix; see the sap-settings skill. Do not select or generate unless asked.

Use the computed baseline_comparison counts and point_indices from get_run_results for whole-frontier claims. Its baseline metrics were computed from the run snapshot and configuration; distinguish this evidence from a user-only comparison marker. Do not count rows mentally or confuse group_count with singleton_group_count. Detailed solver evidence remains available through query_optimizer_data.


## Plant profiles

Discover `list_plant_profiles`, then `select_plant_profile` before creating a new
draft. `update_run_draft` creation accepts `patch.plant_profile_id` and optional
`patch.title`. Supply a concise scenario title describing the requested strategy;
when the user has not supplied one, generate it from the requested settings.
An existing draft permits title updates; its complete profile is
frozen. Fixed settings and approved profile rules cannot be replaced by scenario
constraints or advanced JSON. Additional scenario constraints are additive.
Profile horizon is demand days; source horizon mismatch requires a changed
profile or source upload. Existing runs retain their profile revision even after
profile editing or removal. Name, plant and numeric settings are edited in
Settings; the Excel matrix is uploaded there, or proposed from this chat (see
"Matrix coverage and generating a matrix"). Plant constraints are added,
changed or removed in this chat: propose them with `propose_profile_constraint`
or `propose_profile_settings` and the user approves the card. Approved
constraints appear in the Settings "Plant constraints" list, where they can also
be removed.
