---
name: optimization_explanation
description: Explain the production-wheel grouping problem, variables, objectives, physical constraints and proof limits; identify obvious impossible requests without inventing infeasibility.
---

# Production-wheel optimization context

The application chooses feasible production subgroups and production versions
(PVs), optionally with filling-line assignments, to explore the trade-off between
coverage and a within-group changeover proxy. This is a grouping/lot-sizing
optimization, not a time-indexed production schedule or sequencing simulation.
Use actual capabilities, selected source/profile and frozen run settings as
authority; defaults and examples below are not universal plant rules.

## Population, candidates and decision variables

- A FINI is a finished product. Its relevant data includes forecast litres,
  package volume, PCK, eligible filling lines, pallet quantities and optional
  demand-network descriptors. Preserve missing optional evidence as missing.
- Candidates stay within a plant/SEFI block. A candidate c specifies a subset
  M(c) of FINIs, one allowed PV/batch and, when enabled, one selected common
  filling line. Prefer explicit names in explanations.
- Binary decision x(c) chooses a candidate. Exact cover requires
  `sum(x(c) for c containing product i) = 1` for every admitted FINI i.
  Products cannot disappear or appear in two selected groups to satisfy a rule.
- Shared eligibility is the intersection of member line permissions; assignment
  chooses a line from that intersection. Volume compatibility never grants line
  permission. Group size and explicit prohibitions constrain admissible candidates.

## Quantities and objectives

Let H be source demand days, W productive weeks, D(c) the group's forecast litres,
and B(c) the PV's effective batch litres (nominal PV batch times canonical factor).
Daily group demand is D(c)/H. BASE_GROUP coverage is `C(c)=B(c)*H/D(c)`.
Demand-weighted coverage is `sum(D(c)*C(c)*x(c))/sum_i demand_i`.
Other coverage bases may use member allocations/pallet rounding; inspect the
selected basis before recomputing or comparing values.

The changeover contribution is `D(c)/(W*B(c)) * (group_size(c)-1)`; Changeover (internal
field `j_ch`) is its sum over selected candidates. Singletons contribute zero. Initial setups,
between-group changes, actual sequence and cleaning setups are excluded.
Therefore zero Changeover is not zero physical changeovers. Reducing coverage may
increase this proxy; a Pareto point is a trade-off, not an automatic recommendation.

Use the configured coverage/operations/Pareto mode and named priorities. The
preferred_line objective is an explicit assignment preference in supported
workspace runs, not an implied effect of a size cap. Do not invent weights,
capacity limits, a monetary inventory model or an objective patch unsupported
by the calling interface. Fixed-profile interpretation only emits supported hard
rules and matrix policies; it cannot silently install an objective preference.

## Runner and source semantics

High means individual coverage <= the profile threshold; low means greater.
There is no medium. At the default threshold, exactly 15 days is high.
Reference basis uses the source lot size considered divided by individual daily
demand, without applying a second canonical factor. Candidate-PV basis uses
that PV's effective batch divided by the individual's daily demand, as if it
alone received the batch. It does NOT use the whole group's combined demand.
A candidate-PV runner class may differ between PVs.

Daily demand and coverage depend on the uploaded forecast's demand-day horizon.
A profile must match its source horizon; changing 80 to 250 is not a harmless
display conversion. Never rescale source demand without an explicit supported
operation. XYZ values describe counts of DCs in each demand class, not runner
classes. They are optional; missing counts are not evidence of zero. Use parsed
canonical counts, never guess from ambiguous raw delimiters.

Profile volume matrices have Y (compatible), AVOID (discouraged), N (forbidden).
N is forbidden in every mode; HARD also excludes AVOID. Unknown-volume pairs
need an explicit compatible profile. PCK equality and numeric volume comparisons
are separate concepts from matrix compatibility and shared line eligibility.

## Quick feasibility reasoning

Hard rules narrow the feasible set; adding a hard rule cannot make a previously
impossible exact problem feasible. A candidate preference changes selection
priority rather than granting physical permission. Some small contradictions can
be proved without optimization: incompatible bounds on the same required groups,
or insufficient total group capacity to cover the admitted population. Explain
those and do not apply the proposed constraints; use contraint_translation.

Do not infer a global contradiction from one rejected candidate, a missing fact,
or different runner labels. High/low mixture and volume/PCK homogeneity can
coexist. A line-specific contradiction needs evidence that affected required
products cannot be grouped or routed any other allowed way. Check scope, possible
PVs and alternatives before declaring impossibility. Examples from a meeting
are independent scenarios unless the planner explicitly combines them.

## Solver and evidence boundaries

HiGHS solves the binary model through Pyomo. Group-local predicates filter
candidates; scalable block solves generate options for an overall frontier.
Coupled selection bounds and supported assignment priorities use a joint master
where necessary. Candidate generation may be restricted; time limits may leave
only a feasible incumbent. More runtime cannot recover omitted candidates.

Distinguish structural/canonical validation, proof scope, and business acceptance.
An optimal restricted library is not full-problem optimality. Infeasibility in a
restricted library is not proven infeasibility of all possible candidates. A
timeout without a solution proves neither feasibility nor infeasibility. A
singleton-only incumbent may indicate poor search, not restrictive business rules.
Use persisted run evidence and registered optimizer references before explaining
actual results. Never describe generated prose as an independent mathematical audit.
