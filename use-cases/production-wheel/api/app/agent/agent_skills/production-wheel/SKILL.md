---
name: production-wheel
description: Inspect production-wheel datasets, edit a shared revisioned draft, run optimization and explain persisted results through typed workspace tools.
---

# Production wheel workspace

Use this skill for dataset questions, draft configuration, production grouping,
optimization and analysis of the coverage/changeover frontier.

## Workspace workflow

1. Call `get_optimizer_capabilities`. Read the finite capability menu and its
   `workspace_context` containing the current selected dataset, draft and run IDs.
   Do not assume a plant, dataset, historical workbook or previous selection.
2. Use `list_datasets` and `inspect_dataset(dataset_id)` to resolve an explicit
   source version and inspect its issues, admitted scope and parameter provenance.
   Workbooks are uploaded by the UI and parsed deterministically. Tools operate on
   persisted IDs; never ask the model to supply local input/output paths.
3. If there is no draft, call `update_run_draft(patch={"dataset_id": "..."})`
   to create one for the explicit inspected dataset. When a dataset is already
   selected in `workspace_context`, `update_run_draft()` uses that selection.
   Optional `request` and `budget` patches can be supplied during creation. The
   returned draft and dataset become the selected context. Never invent an ID.
   Inspect an existing selected draft with `update_run_draft(draft_id, revision, patch={})`.
   An empty patch reads its current state without increasing the revision. Read
   and use the returned revision for a nonempty patch to request/budget fields.
   Edits are shared with the manual editor; a revision conflict means reload the
   draft and reconcile the user's requested change with the current state.
4. Call `validate_run_draft` and address reported unsupported constraints or data
   issues. Use `get_optimizer_reference(topic)` for registered schemas, formulas,
   constraints and supported query views. Do not guess field names or capabilities.
5. When the user requests execution, call `launch_optimization` with the exact
   current draft revision and a stable idempotency key. Reuse the same key for a
   retry of the same intended launch. A returned run ID is an independent job.
   When a completion report is requested, call `wait_for_run(run_id)` once.
   It waits outside the model loop and streams progress without graph steps.
   Never repeatedly poll, list runs or launch a replacement while waiting.
6. Use `get_run_status` for an explicit status request,
   `list_runs` to rediscover jobs after reconnect, and
   `get_run_results` once a run has results. `cancel_run` acts
   only on the explicitly identified run when cancellation is requested.

The browser displays tool activity and draft/run events. Never expose hidden
reasoning or claim that draft edits have launched a run. Quote actual tool results
and distinguish queued, running, failed, cancelled, completed and partial states.

## Parameters and constraints

The dataset supplies modeled frequency codes, demand days, productive weeks and
canonical PV factor. Defaults are 01W/02W, 250 days, 50 weeks and 0.90; they are
configurable and should not be presented as immutable rules for every plant.

Use the live capability menu for accepted modes, matrix versions, constraints and
budgets. Every constraint needs an identifier, applicable scope and governed
approval status. Treat numeric and named filling lines as strings, including
`"1"`, `"L1"`, `"L2"`, and `"COROB"`. Line witnesses are scoped to the source
plant. Never invent filling lines, production versions or compatible volume pairs.

A user's specified group cap belongs to the relevant plant/SEFI scope. The default
historical cap of seven is not evidence that every source plant has the same rule.
Greenfield runs use source primitives and authorized constraints; historical
assignments are evidence only and do not become hidden optimization constraints.

## Analyze persisted results

Use `query_optimizer_data` for dataset and run views, and `get_solution_details`
for a page of groups or members at a chosen point. Semantic queries take: explicit view and dataset/run ID, optional point
index, fields, finite filters, grouping, aggregate metrics, sort and pagination.
There is no arbitrary SQL, Python execution or filesystem reader. Server-computed
aggregates are authoritative; do not calculate totals from only the first page.

Use `compare_solutions(left, right)` to compare explicit `{run_id, point_index}` selections and
`explain_assignment(run_id, point_index, material=..., group_id=...)` for persisted
membership evidence. Explain differences in dataset versions, constraints and
budgets before attributing KPI differences to an optimization improvement.

## Required factual checks before explaining a result

Call `get_optimizer_reference("coverage changeovers proof comparison")` before
interpreting a frontier or making a historical comparison. Keep these three
reporting axes separate in every result explanation:

- **Structural validation:** `VALID_PARETO_POINT` means the point passed structural
  validation. It is never an acceptance status or a business recommendation.
- **Proof scope:** report the overall run-level proof limitation first. A completed
  job can be runtime-limited. If the run is runtime-limited, call its points
  feasible incumbents; do not call the point optimal or restricted-library optimal
  merely because a block, candidate library or global master says `OPTIMAL`.
  Such evidence applies only to that named subproblem. State restricted-library
  optimality only when explicitly evidenced, always with its scope, and never use
  it to replace an overall runtime-limited conclusion or imply a complete frontier.
- **Business acceptance:** use only a separately recorded business decision. If
  none is present, say acceptance has not been assessed. Structural validation,
  nondominance and solver status cannot supply that decision.

Use the customer-facing name **Changeover** for internal `j_ch`/`J_CH` fields.
Both coverage days and Changeover are **minimized**. Lower coverage is favorable for this
inventory objective when the compared populations and formulas agree. For example,
46.36 days compared with 21.00 days is **25.36 days higher**, not an improvement in
coverage. Do not invert the direction because a point is named optimized. Changeover is
the sum of each group's frequency times its within-group FINI changes; lower is
favorable for this changeover measure. A trade-off may improve one and worsen the
other; describe both directions instead of declaring an unconditional improvement.

A historical line or KPI is not automatically a like-for-like baseline. Establish
matching material identities, annual demands/weights, scope, coverage basis,
calendar, pallet formula and PV factor before a historical improvement claim, and
report differences in constraint feasibility. If populations differ, recalculate
both on a documented common population or state that the numbers are not directly
comparable. Do not infer a percentage improvement from mismatched populations.
Greenfield runs do not receive historical assignments as hidden constraints.

Present representative points with sourced metrics and these separate evidence
axes. Missing proof or acceptance evidence remains unknown; never infer it from
labels such as optimized, completed, feasible or valid.

## Command-line usage

The agent CLI uses these same persisted workspace tools and ID-based workflow.
Start with `list_datasets`, inspect the chosen version, then create a draft. Raw
workbook extraction belongs to the separate upload/import flow; the agent does not
accept a local workbook path as a dataset ID.
