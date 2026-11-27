---
name: settings-vocabulary
description: Translate planner wording into optimizer draft settings and translate internal setting names, metrics and labels back into plain planner language.
---

# Settings vocabulary

Planners describe what they want in business words. The draft stores JSON keys.
Use this guide in both directions: map a request onto the correct key, and explain
every setting, metric or status in plain language when replying.

## Reply language

- **Customer-approved terms** may be used as-is: FINI (finished product), SEFI,
  PV (production version), runner (high/medium/low runner), plant, filling line.
- **Every other internal name** must be replaced by its plain phrase from the
  glossary below. A JSON key appears in a reply only inside a JSON snippet, or
  in parentheses after the plain phrase when the user used the key first or
  asks which setting changed: "Changeover (`j_ch`)".
- The internal metric `j_ch` / `J_CH` is always called **Changeover** in prose,
  tables and column headers, even when the user wrote J_CH.
- When confirming a draft change, name the setting in plain words, the old and
  new value, and the unit: "Candidate splits per SEFI block: 5 -> 10".

## Glossary of internal terms

| Internal term | Say instead | Meaning |
|---|---|---|
| `j_ch`, J_CH | Changeover | Weekly within-group product switches: for each group, runs per week times (members - 1), summed. Singletons add zero. Lower is better. Excludes first setups, switches between groups and cleaning. |
| `demand_weighted_mean_coverage_days` | Average coverage (days) | Days of demand one batch covers, averaged with each group's demand as weight. Lower means less stock. |
| block, plant/SEFI block | SEFI block | All FINIs of one SEFI in one plant. Groups never mix blocks; each block is optimized separately and then combined. |
| group, subgroup | Production group | FINIs produced together from one batch of one PV. |
| singleton | Single-product group | A group with one FINI; it adds no Changeover. |
| frontier, Pareto frontier | Trade-off curve | The set of solutions where coverage cannot improve without more Changeover, and vice versa. |
| point, frontier point | Solution (option N) | One complete grouping on the trade-off curve. |
| operations anchor | Lowest-Changeover solution | The end of the curve with the fewest switches (usually higher coverage). |
| coverage anchor, coverage-lean end | Lowest-coverage solution | The end of the curve with the least stock (usually more Changeover). |
| `BASE_GROUP` | Batch-based group coverage | Wheel RoC: effective batch / group daily demand rounded to whole days (at least 1), times each product's runner multiple, demand-weighted. No pallets. |
| `ADJUSTED_GROUP` | Pallet-adjusted group coverage | Demand-weighted product inventory RoC: each product's wheel RoC raised to cover one full pallet (workbook fxAdjustedRoC). |
| `WORST_FINI` | Worst-product coverage | The highest single-FINI coverage inside each group. |
| effective batch | Effective batch size | Nominal PV lot litres times the usable-batch factor (default 0.90). |
| greenfield | Clean-sheet grouping | Groups designed from scratch; historical groups are only used for comparison. |
| baseline | Current (historical) grouping | The existing wheel recalculated with the same data and formulas. |
| candidate pool, candidate library | Group options considered | The possible groups the solver may choose from in a block. |
| restricted pool | Shortlisted group options | Too many possible groups to list, so only a ranked shortlist was considered; the best result is proven only within that shortlist. |
| exhaustive | All group options considered | Every possible group in the block was listed. |
| incumbent, feasible incumbent | Best solution found so far | Valid, but the time limit stopped the search before optimality was proven. |
| runtime-limited | Stopped by the time limit | The search ended on time, not because it proved it was finished. |
| `mip_gap`, gap | Optimality tolerance | How far (as a percentage) a result may be from the proven best. |
| `VALID_PARETO_POINT` | Passes the structural checks | Every FINI is in exactly one group and all hard rules hold. Not a business approval. |
| proof scope | What has been proven | Which subproblem, if any, was solved to proven optimality. |
| epsilon, epsilon budget | Changeover cap for one solution | The Changeover limit used to generate one point of the curve. |
| matrix, compatibility matrix | Package-volume compatibility table | Which can volumes may share a group: Y allowed, AVOID discouraged, N forbidden. |
| AVOID pair | Discouraged volume pair | Allowed only when the compatibility mode permits it; always reported. |
| common lines | Shared filling lines | Lines every member of the group is allowed to use. |
| fingerprint, configuration ID | Settings identifier | A hash proving two runs used identical settings. |

## Planner phrases to settings

Values below are schema defaults. The selected plant profile or dataset may set
other values, and profile-fixed settings cannot be overridden by the draft
(see "Profile-fixed settings"). Always read the current draft before quoting a value.

### Execution budget (`budget`)

| Planner may say | Key | Plain name | Range / default |
|---|---|---|---|
| "N solutions", "N points", "N options on the curve", "trace N trade-offs" | `frontier_points` | Solutions requested on the trade-off curve | 2-100, default 30. A request, not a guarantee: duplicates are dropped. |
| "at least N solutions", "make sure I get N points", "minimum N options" | `min_frontier_points` | Minimum solutions kept on the curve | 2-100, default 30 (clamped to `frontier_points`). Extra solves fill the widest gaps, keeping the low-Changeover bias. Assume the default (30 or more, biased to low Changeover) when the user does not mention it. Fewer survive only when the curve has fewer distinct trade-offs; the run then reports a shortfall. |
| "try N ways to split each SEFI", "explore N partitions per block", "more alternatives per SEFI" | `block_options_per_block` | Candidate splits per SEFI block | 2-100, default 5. Complete alternative groupings of one block, not individual groups. |
| "N minutes per SEFI", "solver time per block", "think longer on each SEFI" | `per_block_total_seconds` | Solver time per SEFI block | Seconds, > 0, default 120. Convert minutes to seconds. Shared by all solves of that block. |
| "run N at once", "parallel", "N workers", "use more CPU" | `block_workers` | SEFI blocks solved in parallel | 1-16, default 2. More workers use more CPU and memory. |
| "focus on low changeover", "more points near few switches", "spread evenly" | `global_epsilon_exponent` | Changeover focus of the curve | 1-10, default 2. 1 spreads points evenly; higher packs more points near the lowest-Changeover end. |
| "stop after N hours", "overall time limit", "deadline" | `wall_time_seconds` | Overall run time limit | 10-172800 s, default 3600. The whole job stops at this limit. |

### Scope and population (`request.scope`)

| Planner may say | Key | Plain name | Notes |
|---|---|---|---|
| "all SEFIs", "whole plant", "whole population" | `scope` = `[]` | Which SEFI blocks to optimize | Empty list means every eligible SEFI block in the dataset. |
| "only SEFI 123", "just plant P1" | `scope` = `[{"plant": "P1", "sefi": "123"}]` | | IDs are strings; omit `sefi` to take every SEFI of that plant. |

### Objective and formulas (`request.config`)

| Planner may say | Key | Plain name | Values |
|---|---|---|---|
| "trade-off", "compare coverage and changeovers" | `coverage_mode` | Optimization goal | Workspace accepts only `PARETO` (trade-off curve). |
| "coverage by batch", "by pallet", "worst product" | `coverage_basis` | Coverage calculation | `BASE_GROUP`, `ADJUSTED_GROUP`, `WORST_FINI` (see glossary). |
| "round to full pallets", "minimum pallet" | `pallet_formula` | Pallet rounding rule | Only the reported physical pallet allocation of one batch. `MINIMUM_ONLY`: at least one pallet per FINI. `WHOLE_PALLET_ROUNDING`: round up to whole pallets. Inventory coverage always uses the workbook's one-pallet rule; batches per week and changeovers never use pallets. |
| "keep current PVs", "let the tool choose PVs", "change batch sizes" | `pv_mode` | PV choice | `FIXED` keeps each FINI's current PV. `OPTIMIZED` lets the solver pick an allowed PV per group. |
| "respect compatibility strictly", "allow AVOID pairs", "ignore compatibility", "prefer similar volumes", "group similar can volumes, soft" | `matrix_mode` | Compatibility mode | Say the label to users; send the stored value in tool calls. `FLEXIBLE` = **Avoid allowed**: AVOID allowed and reported, not avoided. `DIAGNOSTIC` = **Prefer compatible**: AVOID allowed but minimized before coverage. `HARD` = **Strict**: AVOID and N rejected. `OFF` = **Ignore table**: no AVOID preference (N still forbidden when a profile is selected). Never call DIAGNOSTIC "diagnostic" to users. |
| "target coverage between X and Y days" | `target_band` (`lower_days`, `upper_days`) | Coverage reporting band | Default 5-365. Only flags points outside the band; does not constrain the solver. For a hard limit, explain it is not supported (`coverage_bound` is deferred). |
| "annual working days", "demand days" | `demand_days` | Demand days per year | Default 250. Must match the profile horizon. |
| "production weeks per year" | `productive_weeks` | Productive weeks per year | Default 50. Used to turn annual demand into weekly runs and Changeover. |
| "usable batch", "batch efficiency", "fill factor" | `canonical_factor` | Usable-batch factor | Between 0 and 1, default 0.90. |
| "high runner below X days", "low runner above Y days", "medium runners", "runner thresholds" | `runner_thresholds` (`high_max_days`, `low_min_days`, `high_max_ratio`, `low_min_ratio`) | Runner classes | Fixed by the plant profile; change it with a profile settings proposal, never in a scenario. Defaults 10 / 80 days and 3 / 10 times the SEFI's fastest product. Days view: coverage <= `high_max_days` is high, >= `low_min_days` is low. Ratio view: coverage / fastest coverage of the same SEFI <= `high_max_ratio` is high, >= `low_min_ratio` is low. In between is medium; the faster view wins, so the fastest FINI of every SEFI is high. The same cut-offs classify runners in the run's SAP settings. |
| "managed complexity", "adjusted RoC", "fill slow runners every other batch", "single RoC", "simple and efficient site", "runner multiples" | `cycle_stock_policy` (`mode`, `high_min_days`, `medium_min_days`, `low_min_days`, `medium_batch_factor`, `max_days`, `mixed_groups_only`) | Cycle stock strategy | Fixed by the plant profile; change it with a profile settings proposal, never in a scenario. `mode`: `single` = **Single RoC** (every FINI of a subgroup gets the subgroup's RoC; default, today's behaviour); `runner_multiple` = **Runner multiples** (only the fastest FINI keeps the RoC, every other FINI is filled every 2nd, 3rd... batch, so its RoC is a whole multiple); `optimizer_choice` = **Optimizer chooses** per subgroup. `high_min_days` / `medium_min_days` / `low_min_days` (defaults 0 / 20 / 40): minimum multiplied RoC of a non-fastest high / medium / low runner. `medium_batch_factor` (0.5): a medium runner covers at most this share of one full batch. `max_days` (200): cap of cycle-stock days, also the SAP DC lot-size cap. `mixed_groups_only` (true): with `optimizer_choice`, only subgroups mixing runner classes get the multiple option. Coverage and Changeover of a subgroup follow its strategy. |
| "classify runners by current lot" / "by the chosen PV" | `runner_basis` | Runner classification basis | `reference` uses the source "RoC 1 full SEFI batch" (lot size considered / daily demand), the value SAP settings use; `candidate_pv` uses each candidate PV's effective batch (optimizer only). The ratio view is the same under both. |
| "odd/even week profile", "planning calendar codes", "minimum lot at DC", "rounding value", "recipe characters", "intermediate DC" | `sap_settings.*` | SAP settings rules | Plant profile section used only when SAP settings are generated after a run; read live, so no re-run is needed. Change with a profile settings proposal (`settings_patch={"sap_settings": {...}}`). Load the sap-settings skill for every key, default and fix. |
| "assign filling lines", "which line produces each group" | `assign_filling_lines` | Assign a filling line per group | Boolean, default false. Switched on automatically by a preferred line or a rule on the selected line. |
| "prefer line 6", "put as much as possible on line 6" | `preferred_line` | Preferred filling line | Line ID as a string. After the main goal, the solver maximizes litres assigned to this line. It never forces a group onto it. |
| "a label for this scenario" | `scenario_id` | Scenario label | Letters, digits, `.`, `_`, `-`. Not the run title. |

### Group size

| Planner may say | Key | Plain name | Values |
|---|---|---|---|
| "max 7 products per group", "allow up to 9" | `group_size` (`mode`, `base_limit`, `max_excess`) | Maximum FINIs per group (whole plant) | `base_limit` is fixed at 7. Cap 7: `mode` `HARD`, `max_excess` 0. Cap above 7: `mode` `BOUNDED_RELAXATION`, `max_excess` = cap - 7. |
| "SEFI 123 at most 3 per group" | `group_size_overrides` (`plant`, `sefi`, `maximum`) | Per-SEFI group-size cap | Replaces the global cap for that one block. One entry per block. |

### Rules (`request.constraints` and `business_rules`)

| Planner may say | Kind | Plain name |
|---|---|---|
| "no more than N per group" (scoped) | `max_group_size` | Group-size limit |
| "allow N extra above 7" | `group_size_relaxation` | Group-size allowance |
| "A and B always together" | `must_link` | Keep together |
| "A and B never together" | `cannot_link` | Keep apart |
| "leave FINI X out", "force X in" | `fini_disposition` | Include / exclude a FINI |
| "use PV P for these" | `fixed_pv` | Fixed PV |
| "only PVs P1 or P2" | `allowed_pvs` | Allowed PVs |
| "must run on line 6" | `required_lines` | Required filling lines |
| "groups on line 6 need a high runner", conditional rules | `group_rule` (in `business_rules`) | Conditional group rule |
| "at most N groups in this plant", totals across groups | `selection_bound` (in `business_rules`) | Limit across selected groups |
| "volumes 1L and 5L are compatible" | `matrix_pairs` (`volume_a`, `volume_b`, `status`) | Volume compatibility pairs |
| "classify by nearest volume", "use reference volumes", "volume classes / bands / tramos" | `volume_matching` (`exact` or `nearest_class`, profile setting) | Volume matching |
| "which class did 2.2 L fall in", "volume band of a product" | `matrix_volume_classes` (run, per product `matrix_volume_class`) | Volume class of each package volume |
| "add a 2 L stop", "reference volume", "below 1 L", "above 15 L" | class matrix rows: stops (`"2"`) and tails (`"<1"`, `">15"`); `volume_compatibility.inherit_from`, `remove_volumes` | Stops and classes of the volume matrix |
| "0.5-1 L together, exclude the rest", "at most 1 L apart", "within 25 %" | `volume_compatibility.rules` (`range_group`, `max_difference_litres`, `max_difference_percent`) | Volume compatibility rules (profile matrix proposal) |

### Rule expression fields (`group_rule`, `selection_bound`)

| Planner may say | Field / operator | Plain name |
|---|---|---|
| "products per group", "FINIs in a group" | `group.size` | FINIs in the group |
| "lines every product can use" | `group.common_lines` | Shared filling lines |
| "when produced on line 6", "assigned to line" | `group.selected_line` | Assigned filling line |
| "production version", "PV" | `group.pv_id` | Chosen PV |
| "batch size", "SEFI batch size", "disperser size", "selected PV lot" | `group.pv_lot_litres` | SEFI batch size (PV lot, litres) |
| "usable batch", "effective batch" | `group.batch_litres` | Effective batch size (PV lot x usable-batch factor) |
| "number of batches", "SEFI batches", "share of batches" | `group.batch_count` | SEFI batches over the horizon (demand / effective batch, fractional) |
| "group demand", "litres of the group" | `group.demand_litres` | Group demand (litres over the horizon) |
| "lots per day", "batches per day", "tank load" | `group.batches_per_day` | Lots the group makes per production day (1 / wheel RoC) |
| "batches per week" | `group.batches_per_week` | Batches per week (days per week / wheel RoC) |
| "wheel RoC", "rounded RoC" | `group.wheel_roc_days` | Whole-day wheel RoC R of the group |
| "tank size", "tank" | `group.pv_lot_litres` (ranges confirmed with the planner) | Tank of the selected PV lot |
| "group coverage" | `group.coverage_days` | Group coverage (days); with runner multiples the demand-weighted coverage of its products |
| "changeovers of the group" | `group.j_ch` | Changeover of the group; with runner multiples batches per week x (products filled per batch - 1) |
| "same PCK" | `group.equal_pck` | All members share one PCK |
| "single RoC group", "group with runner multiples", "adjusted RoC group" | `group.cycle_stock_strategy` | Cycle stock strategy of the group: `single` or `runner_multiple`; other values are rejected |
| "product", "material" | `member.fini_id` | FINI |
| "can size", "package volume" | `member.volume` | Package volume (litres) |
| "PCK", "packaging code" | `member.pck` | PCK |
| "high runner", "medium runner", "low runner" | `member.runner` | Runner class (`high`/`medium`/`low`); other labels are rejected |
| "filled every second batch", "runner multiple" | `member.runner_multiple` | Batches one fill of the product covers (1 = every batch; always 1 in a single-RoC group) |
| "times slower than the fastest product of the SEFI" | `member.sefi_ratio` | Coverage divided by the SEFI's fastest coverage (one decimal) |
| "coverage of the product" | `member.reference_coverage_days` | Product reference coverage (days) |
| "product demand" | `member.demand_litres` | FINI demand (litres) |
| "X/Y/Z distribution centres" | `member.xyz_x_dc_count`, `member.xyz_y_dc_count`, `member.xyz_z_dc_count` | DCs per XYZ class |
| "primary / secondary DCs" | `member.primary_dc_count`, `member.secondary_dc_count` | DC counts by role |
| "sales scenario" | `member.sales_scenario`, `member.sales_network_scenario` | Sales scenario labels |
| "if ... then count ... otherwise ..." | `if` (condition, then, else) | Conditional value; turns a yes/no test into a number for totals and shares |

Operators: `and`, `or`, `not`, `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`,
`contains`, `add`, `sub`, `mul`, `div`; member aggregates `count`, `sum`,
`min`, `max`, `distinct`, `any`, `all`. Member fields are only valid inside
exactly one aggregate. A share such as "at least half of the batches" is one
`selection_bound`: `sum(batch_count x (share condition ? 0.5 : -0.5)) >= 0`.

Deferred and rejected: `coverage_bound`, `freeze_assignment`,
`volume_compatibility_override`. Each rule needs `constraint_id`, `scope`
(`plant`, `sefi`), `enforcement`, `source_text`, `description` and
`approval_status`. Use contraint_translation before building rule JSON.

### Advanced solver settings

Change these only on an explicit request; they rarely need tuning.

| Key | Plain name | Meaning |
|---|---|---|
| `pool_limits.exhaustive_subset_limit` | Full-listing limit (product combinations) | Above this many combinations, group options are shortlisted. |
| `pool_limits.exhaustive_configuration_limit` | Full-listing limit (combinations with PVs) | Same, counting PV choices. |
| `pool_limits.candidate_ceiling_per_block` | Shortlist size per SEFI block | Maximum group options kept when shortlisting. |
| `pool_limits.beam_width_per_size_scorer` | Shortlist breadth per group size | Product sets kept per group size and ranking rule. |
| `pool_limits.forced_exact_configuration_ceiling` | Safety ceiling for full listing | Upper limit even when a block is forced to full listing. |
| `exhaustive_blocks` | SEFI blocks forced to full listing | `[plant, sefi]` pairs; still bounded by the safety ceiling. |
| `solver_limits.time_limit_seconds` | Time limit per solver call | Seconds. |
| `solver_limits.suite_time_limit_seconds` | Time limit for a solver series | Seconds. |
| `solver_limits.mip_gap` | Optimality tolerance | 0.01 = within 1% of proven best. |
| `solver_limits.threads` | Solver threads | Per solver call. |
| `solver_limits.random_seed` | Random seed | Same seed, same result. |
| `solver_limits.presolve_reduction_limit` | Preprocessing limit | -1 = solver default. |
| `versions.matrix_version`, `versions.ruleset_version`, `versions.schema_version` | Rule/table versions | Audit identifiers; use only values from the live catalog. |
| `baseline_guardrails.j_ch_relative_tolerance`, `baseline_guardrails.j_ch_absolute_tolerance` | Allowed Changeover increase vs current grouping | Legacy comparison modes only; no effect on trade-off runs. |

## Profile-fixed settings

When a plant profile is selected, its fixed values (for example horizon, runner
threshold and basis, compatibility table and mode, group-size cap, approved
rules) cannot be replaced by the draft. Scenario rules can only add
restrictions. If a request conflicts with a fixed value, say which value is
fixed and that it belongs to the plant profile, not the draft. You can propose a
profile change with `propose_profile_settings` or `propose_profile_constraint`;
it applies to new drafts only after the user approves the card.

## Ambiguous wording: ask once

Ask one short clarifying question instead of guessing when the wording could
mean more than one setting:

- **"N groups"**: number of groups in the result (not directly settable; can
  only be limited with `selection_bound`) or maximum FINIs per group?
- **"N minutes"/"N hours"** without "per SEFI" or "overall": solver time per
  SEFI block or overall run time limit?
- **"N options"/"N solutions"/"N alternatives"**: solutions on the trade-off
  curve or candidate splits per SEFI block?
- **"cap"/"limit"** without a unit: group size, time or number of solutions?
- **"faster"/"more thorough"/"better"**: suggest concrete values for the time,
  splits and parallel settings and ask for confirmation; never pick silently.
- **"low changeover" as a hard limit** ("no more than X changeovers"): there is no
  Changeover cap setting; explain that the trade-off curve already lists
  lowest-Changeover solutions and ask whether the Changeover focus setting is
  what they want.

When the request is clear, apply it without asking and state the mapping in
the reply: "Solver time per SEFI block: 3 minutes (180 s)".
