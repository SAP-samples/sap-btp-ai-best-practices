---
name: contraint_translation
description: Translate planner instructions into supported production-wheel constraints in the chat, ask for missing values, and propose plant-profile constraints for user approval; preserve OR, scope, eligibility and assignment, and refuse contradictory or unsupported requests.
---

# Constraint translation

Apply this guide before interpreting fixed plant rules or changing scenario
constraints. Use optimization_explanation for the model's physical meaning.
The name `contraint_translation` is the application's configured skill identifier.

## Account for the entire request

Identify each intent as a hard rule, objective preference, matrix policy,
alternative scenario, ambiguity, or contradiction. Preserve the plant/SEFI scope,
numeric boundaries and AND/OR structure. Every emitted rule needs source_text
(the planner's own wording), description (plain-language meaning), an identifier
and the correct scope. Never silently remove a clause, relax a prohibition,
invent a threshold, choose an alternative, or approve your own proposal.
Do not propose constraints that contradict each other or the known physical
problem; explain the specific conflict and the minimal choice needed to resolve it.
Missing information is not proof of infeasibility. Do not reject a clear
supported rule merely because its solution quality or full-dataset feasibility
has not yet been tested.

## Authoring plant-profile constraints in the chat

Fixed plant constraints are created conversationally and saved only when the
user clicks **Approve** on the card your proposal produces.

1. Resolve the profile: `get_optimizer_capabilities` shows the selected
   `plant_profile_id`; otherwise `list_plant_profiles` then `select_plant_profile`.
   Read it with `get_plant_profile_details` (settings, revision, approved rules)
   so you do not duplicate an existing rule or a fixed setting.
2. Restate the rule in plain words and classify every clause. Ask for each
   missing value, one short question at a time: thresholds, line IDs, strict
   versus inclusive bounds, per-group versus total meaning. Never fill a gap
   with a guess or a "reasonable default".
3. Ground identifiers with data: query the selected plant's dataset
   (`list_datasets`, `query_optimizer_data` on `fini_line_eligibility`,
   `production_versions`, `fini_master`) to confirm line IDs exist, see the
   PV lot sizes around a threshold, and count how many FINIs/SEFIs are affected.
4. Before proposing, show the expression in words with one example group that
   passes and one that fails, including the exact boundary value.
5. Call `propose_profile_constraint` (add/replace/remove) or
   `propose_profile_settings` (fixed settings, volume compatibility). Fix and
   retry when the tool returns validation errors. The tool result means "card
   shown", never "saved": tell the user to review and Approve it.
6. Optionally test a rule on the selected dataset before proposing: put it in a
   draft's `request.constraints` (`update_run_draft`), validate and launch only
   when the user asks for a test run. Never claim feasibility without that run.
7. When a later user message says "[Approved in UI]" or "[Rejected in UI]",
   treat that as the user's decision; re-read the profile before further changes.

A rule the grammar cannot express, or that needs data the dataset does not
have, is reported as unsupported with the reason (grammar gap or data gap) and
what would be needed. Propose nothing for it.

## Logic and scope examples

- **Inclusive per-group OR:** "Each subgroup must have identical volume OR the
  same nonempty PCK; either is acceptable for each subgroup." Emit
  `or(eq(distinct(member.volume), 1), group.equal_pck)`. Same-volume/different-PCK
  passes; different-volume/same-PCK passes; both true passes; both false fails.
  This is supported. Do not call these mutually exclusive strategies or use AND.
- **Strategy alternative:** "Compare grouping by volume with grouping by PCK"
  requests separate runs. Do not combine their constraints. "Volume or PCK" with
  unclear per-group/global scope may need one scope question, not a false conflict.
- **Runner mixture:** "Every subgroup must contain at least one high and one low
  runner" means `any(high) AND any(low)` and excludes singletons. Medium runners
  satisfy neither side; "at least one fast product" may mean `high`, or `high`
  or `medium` (`in`); ask when it is ambiguous. But "mix runners
  OR keep similar products together" does not authorize emitting only the mixture.
  Similarity needs a defined field/predicate. It does not inherit a preceding
  volume/PCK instruction. Different runner classes CAN share volume or PCK.
- **Scoped cap:** "For SEFI S only, every subgroup has at most two products"
  uses scope.sefi="S" within the selected plant; do not constrain other blocks.
  "At most two groups in this plant" is a selection total, not group.size <= 2.
- **Eligibility:** "Any subgroup eligible for line 6 must have at most two
  products, even when assigned elsewhere" means
  `contains(group.common_lines, '6') => group.size <= 2`.
- **Assignment:** "When assigned to line 6, include a high runner" means
  `group.selected_line == '6' => any(member.runner == 'high')`. Eligibility alone
  must not trigger it. Line identifiers are strings; use grounded identifiers.
- **Preference:** A line-6 eligibility cap does NOT prefer assigning line 6.
  Prefer line 6 requires an objective choice, not an invented hard routing rule.
  Profile constraints are hard rules only, so a preference is never a profile
  constraint. A workspace draft can use the supported preferred_line objective
  after its priority is explicit. The one profile-level preference is the
  compatibility mode (see Soft volume similarity).
  Priority means objective ordering here, not an invented numeric weight.
  Permission to use another line is not a requirement to use it: forcing line 6
  would narrow a preference without authorization, not logically contradict eligibility.
- **Cutoff:** With cutoff 15, individual coverage 15 is high, 15.01 is low.
  The inclusive setting is not identical to a strict `<15` request. Report the
  discrepancy; do not suggest 14.99 as an exact replacement for a strict bound.
  Changing the number on an inclusive cutoff cannot implement a strict real-valued
  bound exactly. A strict predicate needs a supported numeric field with the right
  coverage basis; otherwise describe the representation gap instead of recommending
  an allegedly precise threshold adjustment.
- **Soft volume similarity:** "Combine preferably products with similar can
  volume, as a soft constraint" is not a rule. The volume compatibility table
  already defines similar volumes as families (Y within, AVOID outside), and
  `matrix_mode` `DIAGNOSTIC` (label **Prefer compatible**) makes the solver
  minimize AVOID pairs before coverage while still allowing them. Check with
  `get_plant_profile_details` that the table has AVOID pairs (otherwise the mode
  changes nothing), then propose the mode change with `propose_profile_settings`
  or, for one scenario, `update_run_draft`. State the cost: this preference ranks
  above coverage, so coverage can get worse to remove one discouraged pair. Ask
  for a new definition of "similar" only if the user rejects the existing families.
- **Soft wording:** "Try to avoid frequent changeovers" supplies no numeric
  hard bound. Explain the proxy/objective and ask for the intended objective or
  bound. Never invent a maximum or promise sequencing optimization.

## Customer rule shapes

- **Batch share on large dispersers:** "High-volume dispersers must be used in
  more than 50% of SEFI batches; high volume means a batch of at least T litres."
  The SEFI batch is the selected PV lot (`group.pv_lot_litres`); batches over the
  horizon are `group.batch_count` (group demand / effective batch, fractional).
  A share over selected groups is linear: one `selection_bound` with measure
  `mul(group.batch_count, if(gte(group.pv_lot_litres, T), 0.5, -0.5))` and
  `lower: 0` means high batches >= 50% of all batches. Ask for T, ask whether
  the threshold is on the nominal PV lot or the effective batch
  (`group.batch_litres`), and whether exactly 50% is acceptable. A strict
  "more than" has no exact linear form: ask for the margin the planner accepts
  (for example `if(..., 0.49, -0.51)` means at least 51%) instead of inventing one.
  Scope it to the profile plant unless the planner names a SEFI subset.
- **Cap on an assigned line:** "When using line 6, group at most two FINIs" is an
  assignment rule: `group.selected_line == '6' => group.size <= 2`. Confirm the
  line identifier exists in the plant's line eligibility first.
- **"No more than N FINIs":** ambiguous. Per group, it is the profile's fixed
  maximum group size: propose `propose_profile_settings` with
  `maximum_group_size`, or a tighter scoped `group_rule` when only some SEFIs or
  lines are meant. As a total across the wheel it is not a real choice: every
  modeled FINI is covered exactly once, so the total is fixed by the data. Ask
  which meaning applies; ask for N if missing.
- **Tank capacity (lots per day):** "The plant has N tanks of S litres and can
  make at most M lots per day in them." The data has no tank attribute: a group's
  tank is its selected PV lot (`group.pv_lot_litres`). A group makes one batch
  every wheel RoC, so its lots per production day are `group.batches_per_day`
  (1 / R; `group.batches_per_week` = days per week / R; `group.wheel_roc_days` = R).
  Model one `selection_bound` per tank size, scoped to the profile plant:
  measure `if(and(gt(group.pv_lot_litres, LO), lte(group.pv_lot_litres, HI)),
  group.batches_per_day, 0)`, `upper` = lots per day allowed in that size. Before
  proposing, ask (never assume):
  - whether M is per tank or for all tanks of that size (per tank: upper = N x M);
  - which PV lots go in each tank. Offer the default "a lot uses the smallest
    tank at least as large", list the plant's PV lots with `query_optimizer_data`
    on `production_versions` (grouped by `lot_size_litres`) and get the ranges
    (LO, HI] confirmed.
  Tell the planner: the bound is a weekly average of lots per production day, not
  a daily schedule; it only moves batches to another tank when the draft uses
  `pv_mode` `OPTIMIZED` (with `FIXED` each product keeps its PV and a bound below
  today's load makes the run infeasible); and plant-wide bounds are solved as one
  joint model, so raise the draft budget `per_block_total_seconds` (for example
  600) when testing. To test before the profile change, put the rules in a draft
  (`update_run_draft` `request.constraints`, `pv_mode` `OPTIMIZED`), validate,
  `launch_optimization`, `wait_for_run`, then check each tank size with
  `query_optimizer_data` on the `groups` view (sum of `frequency_per_week` / days
  per week, grouped by `nominal_lot_litres`). Report a run with no points as
  "these capacities cannot cover the demand with the available PVs".
- **Preferred line per product:** "Group products as per each product's
  preferred line" needs a per-FINI preferred filling line. The dataset has only
  line eligibility, and deriving a preference needs an upstream extraction
  change (active FINI production versions plus selected blocked ones). Report
  this data gap and propose nothing; a single plant-wide preferred line is a
  separate workspace objective (`preferred_line`), not a profile constraint.

## Contradictions versus evidence gaps

Immediate contradictions include: the same nonempty population must have every
group both size >=3 and size <=2; every group must be singleton AND contain both
runner classes; exact cover of five products with at most two groups of at most
two products. Explain the arithmetic or incompatible predicates, not just
"infeasible." Do not emit either side as an accepted partial solution.

A conflict is conditional when scopes/when predicates only sometimes overlap.
"Line 6 requires a high runner" plus evidence that one required product has
only line 6 and no possible high-runner partner under its block/PV/compatibility
rules is an infeasibility witness. The rule alone is not such a witness. A group
that violates one rule does not prove two rules cannot hold together. Distinguish
logical impossibility, demonstrated data conflict, missing evidence and solver
timeout. Query only authorized source data when a factual conclusion needs it.

## Expression contract

Use only fields/operators in the supplied live schema (see settings-vocabulary
"Rule expression fields"). GroupRule is
`when => assertion` for each candidate; SelectionBound bounds a linear sum over
selected groups and requires a numeric lower or upper. Member fields must occur
inside exactly one aggregate: any/all/count/sum/min/max/distinct take one member
expression, not a collection argument; nested aggregates are invalid. Literal
and field have no args. Comparisons have two. Runner values are high, medium
and low only.

The actual evaluator is eager: AND/OR evaluate every branch. `if` takes
(condition, then, else) and evaluates only the chosen branch; use it to turn a
predicate into a number inside a selection-bound measure. Missing required
volume raises even if the PCK branch is true. group.equal_pck is true only for
the same nonempty PCK on every member. Missing required facts never become zero
and never silently skip an applicable constraint.

Volume compatibility policies enumerate the supplied volumes unless a replacement
list is explicit. Supply compatible families, an explicit remaining-pair status,
and exceptions. Do not infer N or AVOID for unspecified pairs. Diagonals are Y.
Homogeneous grouping is not a substitute for a compatibility matrix. A matrix
change replaces the whole table: propose it with `propose_profile_settings`
(`volume_compatibility`) and let the user Approve the card.

Before returning, check positive and negative examples, predicate scope, exact
boundary values and whether prose describes the actual expression. Do not claim
solver feasibility, rule enforcement, preference implementation or publication
without the relevant tool evidence. A structurally valid expression can still
misrepresent the user's intent.
