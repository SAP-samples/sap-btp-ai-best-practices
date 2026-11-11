---
name: workspace-guide
description: Explain application settings, BASE_GROUP, production versions, budgets, view groups, view members, exports, history removal and conversation reset.
---

# Workspace application guide

Use this skill whenever users ask what a control, configuration value, column or view means. Read get_optimizer_capabilities for live schemas and the selected context before discussing actual current values. These definitions come from workspace models, solver schemas and reporting code. Never infer the user's selected value from a default. Use settings-vocabulary for plain-language names of settings and metrics; always say Changeover for the internal `j_ch` metric.

## Published dataset

The immutable uploaded snapshot used for this run. It defines the plant, materials, source eligibility, demand, production versions and calendar defaults. Only published, non-removed snapshots can start new runs.


## Production version

FIXED keeps each FINI's source production version; group members must share a feasible fixed version. OPTIMIZED lets the optimizer choose a feasible version from the source catalog for each group. A production version determines nominal lot litres; effective batch also applies the canonical factor.


## Coverage objective

PARETO is the workspace's supported objective: sample trade-offs while minimizing both demand-weighted coverage days and Changeover. The operations anchor prioritizes Changeover; the coverage-lean end prioritizes inventory coverage. Requested point count is a sampling budget, not a guarantee of that many distinct nondominated points or a complete frontier.


## Coverage basis

BASE_GROUP = demand_days × effective batch litres / group annual demand. ADJUSTED_GROUP replaces effective batch with the sum of pallet-adjusted FINI allocations. WORST_FINI uses the highest individual pallet-adjusted coverage in each group. Demand-weighted mean uses group demand as weights. BASE_GROUP means batch-based group coverage; it does not mean the historical baseline grouping.


## Compatibility matrix

HARD rejects AVOID exceptions as well as prohibitions. FLEXIBLE allows and reports AVOID pairs; DIAGNOSTIC additionally prioritizes reducing matrix exceptions before coverage. With a plant profile, N is forbidden in all modes including OFF. Without a profile, OFF skips matrix assessment. Common-line eligibility and group caps always apply.


## Global member cap

Maximum FINIs in one group. The quick control uses the governed base of 7: 7 selects HARD; values above 7 select BOUNDED_RELAXATION with max_excess = cap − 7. Plant/SEFI group_size_overrides or typed max_group_size constraints define other local caps. An empty extra-constraints list does not disable existing structural rules.


## Plant / SEFI scope

An empty array [] selects all eligible plant/SEFI blocks in the chosen snapshot. Otherwise use explicit objects such as {"plant":"P1","sefi":"123"}. IDs are strings. Whole population means the modeled population under this run's production-version mode; excluded, external or out-of-scope source rows remain evidence.


## Typed constraints

[] means no additional user constraints. Supported kinds: max_group_size (maximum), group_size_relaxation (base_limit/max_excess), must_link (materials together), cannot_link (material pair apart), fini_disposition (admission/exclusion), fixed_pv, allowed_pvs, required_lines. Each requires constraint_id and governed approval_status; scope limits application to plant/SEFI. draft/approved/rejected are review states, not solver proof. coverage_bound, freeze_assignment and volume_compatibility_override are currently deferred and rejected. Get the live constraint schema before constructing JSON. group_rule supports conditional group/member predicates; selection_bound bounds a linear sum over selected groups across one block or a plant. Unknown required inputs fail closed.


## Execution budget

frontier_points: requested global samples (2–100, default 17). block_options_per_block: retained candidate partitions explored per block (2–100, default 5), not the number of individual candidate groups. per_block_total_seconds: total solver time per block (default 120 s), shared across its solves, not a guaranteed whole-run duration. block_workers: concurrent blocks (1–16, default 2), increasing CPU/memory use. global_epsilon_exponent: spacing bias (1–10, default 2); 1 is uniform and above 1 emphasizes low-Changeover points. wall_time_seconds: overall job deadline (10–172800, default 3600). Example: 33 points, 10 block options, 120 seconds, 4 workers, exponent 2. More time cannot recover candidate groups omitted from a restricted pool. For coupled business rules, per_block_total_seconds bounds joint solver stages for the requested frontier; preprocessing is additional and wall_time_seconds remains the overall job limit.


## Complete optimizer configuration

Apply advanced config writes the entire config object to the shared draft; Save draft updates quick controls. Validate the latest saved revision before launch. Nested fields are explained below. Configuration identifiers are technical JSON keys; display labels use Changeover.

The nested field definitions, with their plain-language names, planner phrasing, ranges and defaults, are in settings-vocabulary.

## Dataset and result views

groups: one proposed group per frontier point, with member count, PV, demand, coverage, common lines and group Changeover contribution. members: one source FINI row per point, including excluded/out-of-scope rows; selected fields are blank when unassigned. Repeated group metrics in members must not be summed as group totals. solutions: one frontier point with whole-solution KPIs. validation: structural validation findings. block_audits/global_audits: solver attempts, bounds, gaps and proof scope. matrix_pairs: package-volume compatibility evidence. fini_master: canonical source material records. production_versions: source PV lot catalog. fini_line_eligibility: plant-scoped material/line eligibility. line_corrections: FixedLine source overrides. validation_issues: extraction quality issues. pallet_resolution: source/enrichment evidence for resolved pallet litres. Search filters only the selected view; pagination is not the full population. Use server aggregates for totals.


## History and conversation

Delete permanently removes a terminal run and its evidence. Deleting a snapshot also deletes its sources, extracted data, drafts and all associated runs/results. The UI asks for confirmation and there is no restore. Active work must finish or be cancelled first. Each workbook import creates a separate review snapshot requiring explicit publication. Clicking a selected snapshot or run again, or Deselect, clears its selection; deselecting does not cancel optimizer jobs. New conversation clears page chat history and stops the current chat turn; submitted optimizer jobs continue independently. To stop a job use Cancel selected run. A recursion failure affects the chat, not proof of job failure; rediscover the existing run before considering another launch.

## Export and line notation

Export production wheel XLSX uses the selected point and uploaded column order. Original Filling line text is retained verbatim. Effective eligible lines includes source corrections; Common eligible lines shows shared alternatives in the same notation. Examples: `- 2  3   -   -   -` means strings 2 and 3, `11 / 12 / 18 / 23` means those four IDs, `L1 / L2` means those named IDs. Never convert L1 into 1 or infer a scheduled assignment. Inspect Export notes for source versus proposal fields. Export current view CSV is the technical evidence view.

## Reports after a run

When the user requests a completion report, call wait_for_run once. Waiting streams job progress without repeated model calls. If wait_for_run returns wait_status=queued_not_started, explain that no worker has claimed the existing run and end the turn without waiting again or resubmitting. Read get_run_results after completion, then report actual representative points and the closest singleton count requested. Do not fabricate a full frontier or assume 33 requested samples means 33 distinct solutions. Historical numbers supplied by a user are comparison markers until matching population/formula evidence is established. Both coverage and Changeover are minimized. Business acceptance and global proof remain separate from structural validation.

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
profile editing or removal. Profile editing is done through reviewed Settings.
