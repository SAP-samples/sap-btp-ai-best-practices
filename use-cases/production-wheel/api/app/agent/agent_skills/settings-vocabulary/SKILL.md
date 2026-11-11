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
  PV (production version), runner (high/low runner), plant, filling line.
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
| `BASE_GROUP` | Batch-based group coverage | Coverage from the full effective batch divided by the group's daily demand. |
| `ADJUSTED_GROUP` | Pallet-adjusted group coverage | Coverage from the sum of member quantities after pallet rounding. |
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
| "N solutions", "N points", "N options on the curve", "trace N trade-offs" | `frontier_points` | Solutions requested on the trade-off curve | 2-100, default 17. A request, not a guarantee: duplicates are dropped. |
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
| "round to full pallets", "minimum pallet" | `pallet_formula` | Pallet rounding rule | `MINIMUM_ONLY`: at least one pallet per FINI. `WHOLE_PALLET_ROUNDING`: round up to whole pallets. |
| "keep current PVs", "let the tool choose PVs", "change batch sizes" | `pv_mode` | PV choice | `FIXED` keeps each FINI's current PV. `OPTIMIZED` lets the solver pick an allowed PV per group. |
| "respect compatibility strictly", "allow discouraged pairs", "ignore compatibility" | `matrix_mode` | Volume compatibility strictness | `HARD` rejects AVOID and N. `FLEXIBLE` allows AVOID, reports it. `DIAGNOSTIC` allows AVOID but minimizes it first. `OFF` skips the table (N still forbidden when a profile is selected). |
| "target coverage between X and Y days" | `target_band` (`lower_days`, `upper_days`) | Coverage reporting band | Default 5-365. Only flags points outside the band; does not constrain the solver. For a hard limit, explain it is not supported (`coverage_bound` is deferred). |
| "annual working days", "demand days" | `demand_days` | Demand days per year | Default 250. Must match the profile horizon. |
| "production weeks per year" | `productive_weeks` | Productive weeks per year | Default 50. Used to turn annual demand into weekly runs and Changeover. |
| "usable batch", "batch efficiency", "fill factor" | `canonical_factor` | Usable-batch factor | Between 0 and 1, default 0.90. |
| "high runner below X days" | `high_runner_threshold_days` | High-runner threshold | Default 15. Coverage <= X is high runner, > X is low runner. |
| "classify runners by current lot" / "by the chosen PV" | `runner_basis` | Runner classification basis | `reference` uses the source lot size; `candidate_pv` uses each candidate PV's effective batch. |
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

Deferred and rejected: `coverage_bound`, `freeze_assignment`,
`volume_compatibility_override`. Each rule needs `constraint_id`, `scope`
(`plant`, `sefi`), `enforcement`, `source_text` and `approval_status`. Use
contraint_translation before building rule JSON.

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
fixed and that it is changed in Settings, not in the draft.

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
