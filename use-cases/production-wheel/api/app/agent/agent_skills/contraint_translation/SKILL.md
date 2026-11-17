---
name: contraint_translation
description: Translate planner instructions into supported production-wheel constraints; preserve OR, scope, eligibility and assignment, and block contradictory, ambiguous or unsupported requests.
---

# Constraint translation

Apply this guide before interpreting fixed plant rules or changing scenario
constraints. Use optimization_explanation for the model's physical meaning.
The name `contraint_translation` is the application's configured skill identifier.

## Account for the entire request

Identify each intent as a hard rule, objective preference, matrix policy,
alternative scenario, ambiguity, or contradiction. Preserve the plant/SEFI scope,
numeric boundaries and AND/OR structure. Every emitted rule needs source_text,
an identifier and the correct scope. Profile preview produces draft rules only.
Do not create, save, apply or launch constraints that contradict each other or
the known physical problem. Explain the specific conflict and the minimal choice
needed to resolve it. Never silently remove a clause, relax a prohibition, invent
a threshold, choose an alternative, or approve your own draft.

In Settings, any unresolved intent blocks the WHOLE proposal: return
clarification_required=true, list the issues in unresolved_intents, rules=[],
volume_compatibility=null. In workspace chat, explain first and make no draft
mutation for the contradictory request. Missing information is not proof of
infeasibility. Do not reject a clear supported rule merely because its solution
quality or full-dataset feasibility has not yet been tested.

Stop early when an explicit unsupported intent or logical contradiction already
blocks the proposal. For example, a line preference in Settings needs an objective
choice regardless of source line counts. Explain that immediately; do not query
datasets or validate the remaining clauses just to produce the same blocked
answer. The required guides are already loaded; do not reload them with a tool.

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
  runner" means `any(high) AND any(low)` and excludes singletons. But "mix runners
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
  Settings has no objective patch; mark that intent unresolved. Workspace can
  expose the supported preferred_line objective after its priority is explicit.
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
- **Soft wording:** "Try to avoid frequent changeovers" supplies no numeric
  hard bound. Explain the proxy/objective and ask for the intended objective or
  bound. Never invent a maximum or promise sequencing optimization.

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

Use only fields/operators in the supplied live schema. GroupRule is
`when => assertion` for each candidate; SelectionBound bounds a linear sum over
selected groups and requires a numeric lower or upper. Member fields must occur
inside exactly one aggregate: any/all/count/sum/min/max/distinct take one member
expression, not a collection argument; nested aggregates are invalid. Literal
and field have no args. Comparisons have two. Runner values are high/low only.

The actual evaluator is eager: AND/OR evaluate every branch. Missing required
volume raises even if the PCK branch is true. group.equal_pck is true only for
the same nonempty PCK on every member. Missing required facts never become zero
and never silently skip an applicable constraint.

Volume compatibility policies enumerate the supplied volumes unless a replacement
list is explicit. Supply compatible families, an explicit remaining-pair status,
and exceptions. Do not infer N or AVOID for unspecified pairs. Diagonals are Y.
Homogeneous grouping is not a substitute for a compatibility matrix. Matrix
proposals need the user's Accept and Save steps.

Before returning, check positive and negative examples, predicate scope, exact
boundary values and whether prose describes the actual expression. Do not claim
solver feasibility, rule enforcement, preference implementation or publication
without the relevant tool evidence. A structurally valid expression can still
misrepresent the user's intent.
