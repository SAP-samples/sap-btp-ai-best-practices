---
name: pipes-guideline-ps4
description: Evaluate a proposed PS4 pipe design only against the documented pipe-material, geometry, end, auto-crimping, robot-pickup, and crimp requirements in Pipes Guideline PS4.
version: 2.1
source: ../Pipes Guideline PS4.pdf
rubric: ps4-pipe-evaluation-rubric.md
---

# PS4 Pipe Design Feasibility Evaluation

Evaluate a proposed pipe design only against the requirements in `../Pipes Guideline PS4.pdf` (last update: 10/07/2024). Determine whether the design is feasible **within the PS4 guideline's stated scope**; do not add engineering rules, assumptions, tolerances, or alternatives that are not in that document.

## Shared rubric input

Use `ps4-pipe-evaluation-rubric.md` as the canonical input contract. For a raw drawing, first use the `pipe-drawing-to-ps4-specification` skill to produce one `ps4PipeSpecification` XML document with `contract="ps4-pipe-specification"` and `schemaVersion="2.1"`, then evaluate that document. Retain its source evidence in the final XML result.

Interpret rubric states consistently:

- The input is always a complete part/configuration drawing. For PS4 prohibited-feature fields only, `finding="absent_by_drawing_omission"` with `state="reviewed_omission"` means the complete reviewed drawing scope contains no callout or depiction; treat it as **Pass** for prohibited-feature absence. LEFT/RIGHT automatic-crimp callouts are retained as evidence but do not determine the automatic-crimp objective, which is derived from calculator-assessed geometry.
- `state="not_stated"` and `state="unreadable"` mean **Cannot assess** for non-prohibited-feature inputs and for any prohibited feature that lacks the required reviewed-omission evidence.
- Every `state="alternative"` or `finding="stated_option"` must have source-backed evidence. Create a separate `<candidate>` only for a distinct complete pipe configuration; alternatives in one end callout belong to that candidate's `end_configurations` set and are evaluated for limited compatibility. Do not select an option globally or report **Cannot assess** solely because alternatives exist.
- Apply the common drawing facts plus one candidate's `<override>` elements for that candidate's evaluation. A candidate with `completeness="partial"` is **Cannot assess** only for its own missing required inputs.
- A `<field name="ps4RawMaterialDesignationCandidate">` is a comparison key derived from stated diameter and thickness. It is not an approval by itself.
- A bender-schedule `STRAIGHT PART` value can be treated as PS4 L1, an L2 segment, or L4 only when the intermediate XML records a successful `map-standard-bender-schedule` result with `rule="standard_ordered_bender_schedule/v1"`, mapped values, and row evidence. Otherwise it is not auto-crimp geometry.
- Robot pickup is always assessed separately when its `<robotPickup>` `l1`/`l2` inputs permit calculation. Its condition changes the technical-feasibility result only when `<field name="requirementOrApplicability">` explicitly establishes that robot pickup is required. A non-gating failure must be stated as a disclaimer when the technical-feasibility result is feasible.

## Required rubric fields

Collect the following rubric inputs before evaluating. Keep a check as **Cannot assess** when its required field is unavailable or not reliably linked to the guideline's labelled dimension.

- `<pipe>` `outerDiameter`, `wallThickness`, and `totalLength` measurements, plus `<field name="ps4RawMaterialDesignationCandidate">`
- `<ends>` for Point 1 and the last point
- `<candidateConfigurations>` when the drawing states alternatives
- `<bendSchedule>` `bendAngle` measurements
- `<prohibitedFeatures>`
- `<autoCrimping>`, including explicitly labelled or source-mapped `L1`, `L2`, `L3`, `L4`, `θ1`, and `θ4`
- `<robotPickup>` — always assess this separately when possible; it is a feasibility-gating input only when its applicability explicitly states that robot pickup is required.
- `<crimps>`, including `<automaticCrimp side="left">` and `<automaticCrimp side="right">`; retain these source facts as supporting evidence only. The automatic-crimp objective is determined by calculator-assessed geometry.

## Mandatory deterministic calculation tool

Resolve the calculator relative to this skill as `tools/ps4_measurement_calculator.py`. The LLM must **never** perform PS4 arithmetic, trigonometry, table lookup, formatting normalization, rounding, or numeric comparison itself—not even as a fallback. Before issuing a numeric criterion result, call the matching operation with only directly evidenced XML values:

| Criterion | Required operation | Direct source-backed arguments |
|---|---|---|
| Raw material | `raw-material` | `--designation` |
| Tube length | `tube-length` | `--diameter-mm`, `--wall-thickness-mm`, `--total-length-mm` |
| Bend angles | `bend-angles` | one repeated `--angle-deg` per stated angle |
| Standard schedule mapping | `map-standard-bender-schedule` | all copied ordered schedule rows: `point`, `bendAngleDeg`, `bendRadiusMm`, and `straightPartMm` |
| Auto crimping | `auto-crimp` | `--diameter-mm`, mapped/labelled `--l1-mm`, repeated `--l2-mm` once for every L2 segment, `--l3-mm`, `--l4-mm`, `--theta1-deg`, and `--theta4-deg` values |
| Robot pickup | `robot-pickup` | `--diameter-mm` and any directly evidenced `--l1-mm` / `--l2-mm` values |

Use this invocation pattern, substituting only source-backed XML values:

```bash
python3 tools/ps4_measurement_calculator.py <operation> <arguments>
```

Copy the tool's JSON result into the relevant XML criterion result; do not recalculate, reinterpret, or round a numeric result. If required inputs are unavailable, record **Cannot assess** without calling a manual substitute. If the tool returns a JSON error or cannot execute, record **Cannot assess**, preserve the tool error as an evaluation limitation, and do not calculate manually. The calculator does not decide evidence, candidate configuration, or applicability. Its only approved geometry mapping is `map-standard-bender-schedule`, which validates the ordered endpoint/bend-row pattern, maps source-stated schedule values to L1/L2/L4/θ1/θ4, and derives L3. Its full operation contract is in `tools/README.md`.

### Standard-schedule recovery before an auto-crimp result

Before reporting **Cannot assess** for auto-crimp geometry, inspect `<bendSchedule>`. If it contains the complete ordered schedule pattern but `<autoCrimping>` does not yet contain a successful `standard_ordered_bender_schedule/v1` mapping, the evaluator **must** call `map-standard-bender-schedule` itself using the copied source-backed row values. This recovery is required because a missing literal `L1`/`L2`/`L3`/`L4` label is not a reason to discard a valid standard schedule.

Use the returned mapped L1, every L2 segment, L4, θ1, θ4, and derived L3 for the current evaluation. Retain the original `<bendSchedule><row>` evidence and record the operation, rule, mapping status, and source points in the final auto-crimp criterion. Only report **Cannot assess** when the mapping operation returns `mappingStatus="Cannot assess"`, its source rows are incomplete or unreadable, or another required auto-crimp input is absent. Do not select schedule rows manually.

## Evaluation method

1. Build the evaluation set:
   - When there are no `<candidate>` elements in `<candidateConfigurations>`, evaluate the common drawing specification once.
   - When candidates exist, evaluate every candidate independently using the common drawing specification plus that candidate's source-backed `<override>` elements.
   - Do not create combinations across variants unless the drawing explicitly associates them.
2. Evaluate every feasibility-gating criterion for each evaluation in the set as **Pass**, **Fail**, or **Cannot assess**. Always determine the robot-pickup condition separately as **Met**, **Not met**, or **Cannot assess** when the available inputs permit it.
3. Quote the measured/design value alongside the applicable requirement.
4. A **Fail** in a feasibility-gating PS4 requirement means that candidate is not feasible under the stated PS4 requirement. A **Cannot assess** in a feasibility-gating requirement means that candidate's guideline-based feasibility conclusion is incomplete, not that the candidate passes. A robot-pickup condition that is not explicitly required is non-gating: neither **Not met** nor **Cannot assess** changes the technical-feasibility result.
5. State the result:
   - For one evaluation: **Feasible against the assessed PS4 requirements**, **Not feasible against the PS4 requirements**, or **Inconclusive**, using the original definitions below.
   - For multiple evaluations: **Configuration-dependent**. List every candidate ID and its individual result. Do not reduce all candidates to one pass/fail result when the drawing has no globally selected configuration.
   - A candidate is **Feasible against the assessed PS4 requirements** when no feasibility-gating requirement fails and no feasibility-gating required input is unassessed. The crimp objective and a non-gating robot-pickup result are reported separately and do not change this result.
   - A candidate is **Not feasible against the PS4 requirements** when one or more feasibility-gating requirements fail.
   - A candidate is **Inconclusive** when no feasibility-gating requirement fails, but one or more feasibility-gating required inputs cannot be assessed.
6. Do not use a criterion that the source document does not state. List that gap under **Guideline limitations**.

## Criteria

### 1. Listed raw material

Call `raw-material` with the existing tool-generated `<field name="ps4RawMaterialDesignationCandidate">` value. Derive that value from source-backed outer diameter and wall thickness for every lookup. Preserve a drawing's material-grade text or note as evidence, but do **not** substitute it for the PS4 comparison key. Use the tool's `guidelineCoverage` result, rather than manually comparing strings, with the following PS4 rough-steel list (as written in the source, in mm):

| Designation listed in source |
|---|
| Ø08TH1 |
| Ø10TH1 |
| Ø10TH1,5 |
| Ø12TH1 |
| Ø12TH1,5 |
| Ø15TH1,5 |
| Ø15TH2 |
| Ø16TH2 |
| Ø18TH1,5 |
| Ø18TH2 |
| Ø20TH2,5 |
| Ø22TH1,5 |
| Ø22TH2 |

If the designation is not in this list, report it as **Not listed in the guideline**. The source identifies the list as rough steels used at PS4, but does not explicitly state the disposition of an unlisted designation; do not infer a pass or fail from the list alone. Treat an unlisted or unavailable designation as **Inconclusive** overall because the guideline does not establish that the material is covered.

### 2. Tube length

Call `tube-length` with the directly evidenced outer diameter, wall thickness, and total length. Copy its status and selected limits into the criterion; do not manually select or compare the range.

| Design condition | Requirement |
|---|---|
| Diameter Ø8 | Tube length must be between 200 mm and 1500 mm. |
| Diameter Ø10 with wall thickness 1 mm | Tube length must be between 200 mm and 1500 mm. |
| Any other listed diameter/thickness combination | Tube length must be between 200 mm and 3000 mm. |

### 3. End configuration

**Pass only if** the ends are free ends or B4 ends (soft seal). The guideline states: “Only the free ends or with B4 ends (soft seal) are possible.” Preserve every source-backed option in the candidate's `end_configurations` list.

When one end callout lists **both** a free/B4 option and an EO-2 or metal-seal option, report **Warning** rather than **Fail**: at least one compatible option exists. The warning must name the source-stated `Compatible option(s)` and `Non-compatible option(s)`. This warning does not gate the candidate's feasibility. Report **Fail** only when no stated end option is free or B4. Do not mark the check **Cannot assess** merely because the drawing shows alternatives.

### 4. Bend angle between two bends

For each source-backed bend angle, call `bend-angles`, passing the directly stated angle as a repeated `--angle-deg` argument. Copy its per-angle and overall status. The guideline condition evaluated by the tool is `0 < θ < 180` degrees.

### 5. Prohibited design features

Mark **Fail** when the candidate being evaluated records `finding: present` for any of these items. Mark **Pass** only when that candidate records either `finding: explicitly_absent` or `finding: absent_by_drawing_omission` with state `reviewed_omission` for every applicable item. An EO-2 or metal-seal finding that is explicitly one alternative in the same source-backed `end_configurations` set as a free/B4 option is **Warning**, not **Fail**; require selection of the compatible option. A `stated_option` must be evaluated through its corresponding candidate override; it is **Cannot assess** only when no source-backed candidate exists. A `not_determined`, `not_stated`, `unreadable`, or missing reviewed-omission evidence is **Cannot assess**. These are identified by the guideline as specificities outside PS4 specifications:

- Banjo screw
- Beads (form B)
- Change in diameter on the tube
- Chamfers or machining at the ends
- Combined tube (tube + flexible or hose)
- Soldering (transplanting, etc.)
- EO-2 (soft seal)
- Metal seal

### 6. PS4 auto-crimp geometry definitions

Use the `L1`, `L2`, `L3`, and `L4` labels only for the geometry defined in `ps4-pipe-evaluation-rubric.md`:

- **L1:** the straight end-leg length from the left-hand free end to the tangent point of its adjacent bend. In a successful `standard_ordered_bender_schedule/v1` mapping, it is the `STRAIGHT PART` on the start endpoint row.
- **L2:** the straight length between the tangent points of two adjacent bends. In that mapping, every internal bend-to-bend `STRAIGHT PART` is a separate L2 segment; all segments must pass the L2 minimum.
- **L3:** the end-side length identified by the curved `L3` dimension line: from the free end of the right-hand end leg, around the `θ4` bend, to the junction with the incoming diagonal leg. It includes the `L4` end leg and the `θ4` bend as drawn; it is not interchangeable with `L4`. When the intermediate XML records the successful `standard_ordered_bender_schedule/v1` mapping and calculator provenance, use its derived L3 value; do not calculate it manually or substitute another schedule segment.
- **L4:** the straight end-leg length from the right-hand free end to the tangent point of its adjacent `θ4` bend. In that mapping, it is the `STRAIGHT PART` on the final bend row immediately before the terminal endpoint.

`θ1` is the `ANGLE` on the first bend row after the start endpoint; `θ4` is the `ANGLE` on the final bend row before the terminal endpoint. An unlabelled bender-table `Straight Part` is not any of these dimensions unless the source XML includes a successful approved mapping.

### 7. Minimum straight lengths for auto crimping

Use the following table for the design diameter. All values are in mm. `R cintrage` is the bending radius shown in the source diagram.

| Diameter | R cintrage | L2 minimum between 2 bends | L1 minimum (`θ1 ≤ 90`) | L4 minimum (`θ4 ≤ 120`) | L3 minimum (`θ4 ≤ 40`) | L3 minimum (`40 < θ4 ≤ 85`) | L3 minimum (`85 < θ4 < 180`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Ø8 | 15,5 | 12 | 29,4 | 30,4 | 46,9 | 62,4 | 77,9 |
| Ø10 | 20,5 | 15 | 32,9 | 33,9 | 55,4 | 75,9 | 96,4 |
| Ø12 | 25,8 | 18 | 36,0 | 37,0 | 63,8 | 89,6 | 115,4 |
| Ø15 | 30,8 | 24 | 42,6 | 39,6 | 71,4 | 102,2 | 133,0 |
| Ø16 | 33 | 24 | 45,7 | 47,7 | 81,7 | 114,7 | 147,7 |
| Ø18 | 36 | 27 | 46,2 | 45,2 | 82,2 | 118,2 | 154,2 |
| Ø20 | 41 | 30 | 55,0 | 56,0 | 98,0 | 139,0 | 180,0 |
| Ø22 | 49,2 | 33 | 52,6 | 50,6 | 100,8 | 150,0 | 199,2 |

Call `auto-crimp` with the direct diameter and only labelled or successfully source-mapped `L1`, every `L2` segment, `L3`, `L4`, `θ1`, and `θ4` values. Use one repeated `--l2-mm` argument for every `<l2Segments><measurement name="l2">` value. A source-mapped schedule geometry set is permitted only when the intermediate XML records `rule="standard_ordered_bender_schedule/v1"`, `operation="map-standard-bender-schedule"`, mapped source-row evidence, and derived-L3 calculator provenance. The tool selects the diameter-specific table row, selects the applicable L3 range, calculates any L1/L4 adjusted minimum, and compares every supplied L2 segment. The LLM must not use the table, formulas, trigonometry, rounding, or comparisons itself.

The guideline table used by the tool is retained below as the source reference. For each applicable dimension, copy the tool's `designValueMm`, `requiredMinimumMm`, `formulaOrLookup`, and `status` into the evaluation XML. For L2, copy the overall L2 result plus every returned `segments` result; all mapped segments must pass. If the source XML does not contain a successful standard-schedule mapping or direct labelled geometry, record **Cannot assess**; do not pass unvalidated schedule rows as L1–L4 tool arguments.

### 8. Robot pickup — separately assessed; conditionally feasibility-gating

Call `robot-pickup` with the direct diameter and any source-backed labelled or successfully source-mapped L1/L2 values. For a mapped schedule, use `<robotPickup><measurement name="l2">`, populated from the mapping operation's `robotPickupL2` (the first bend-to-bend L2 segment). Do not add, compare, or otherwise evaluate the branches manually. Use the tool's overall `status` as the robot-pickup condition and its `l1Branch` / `l2Branch` details as evidence of the computed thresholds.

The guideline condition evaluated by the tool is:

`L2 ≥ 34` **or** `L1 ≥ L1(θ1 ≤ 90) mini + 34`

The source wording is: “Pour la prise robots il faut au maximum un L2 ≥ 34 ou L1 ≥ L1(θ1 ≤ 90) mini +34.”

Read the `robot-pickup` tool's overall `status` without recalculating it:

- **Met** when the tool reports `Met`.
- **Not met** when the tool reports `Not met`.
- **Cannot assess** when the tool reports `Cannot assess`, required direct inputs are unavailable, or the tool reports an error.

Then apply this fixed skill policy:

- When the drawing explicitly states that robot pickup is **required**, the condition is feasibility-gating: **Met** is a pass, **Not met** is a fail, and **Cannot assess** makes the technical-feasibility result inconclusive.
- When robot pickup is not explicitly required, including when it is explicitly stated not to apply, the condition is non-gating. Its status does not cause a technical-feasibility fail or inconclusive result.
- When the technical-feasibility result is **Feasible against the assessed PS4 requirements** and a non-gating robot-pickup condition is **Not met**, include this exact disclaimer: **“Robot pickup is not explicitly required by the drawing, so this does not change the technical-feasibility result. However, the stated robot-pickup condition is not met; if robot pickup becomes required, this design would not meet that condition.”**

### 9. Automatic-crimp objective

Derive the automatic-crimp objective from the calculator-backed automatic-crimp geometry; do not require an explicit LEFT or RIGHT automatic-crimp callout. The objective remains non-gating for technical feasibility.

- **Pass** when the `auto-crimp` calculator reports that every required L1, L2, L3, and L4 value meets its diameter- and angle-dependent minimum.
- **Fail** when the calculator reports one or more failing automatic-crimp dimensions.
- **Warning** when the schedule cannot be mapped, a required calculator input is unavailable, or the calculation cannot be evaluated. Do not report **Cannot assess** or **Not met** for this customer-facing objective.

LEFT and RIGHT automatic-crimp callouts are retained as drawing evidence, but their omission does not downgrade a passing geometry result. The closed-world treatment of an automatic-crimp omission does not create an additional PS4 feasibility requirement.

## Required evaluation output

Return **only** one well-formed XML document: no Markdown fence, heading, commentary, or prose outside the root element. Use the shared XML conventions in `ps4-pipe-evaluation-rubric.md`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ps4PipeFeasibilityEvaluation
    contract="ps4-pipe-feasibility-evaluation"
    schemaVersion="2.1">
  <inputSpecification
      contract="ps4-pipe-specification"
      schemaVersion="2.1"
      sourceFile="" />
  <assessmentAssumption
      drawingScope="complete_part_configuration_definition"
      prohibitedFeatureOmission="absent_by_drawing_omission"
      automaticCrimpOmission="absent_by_drawing_omission"
      robotPickupFeasibilityGating="only_when_explicitly_required" />

  <overallResult status="">
    <!-- Feasible against the assessed PS4 requirements |
         Not feasible against the PS4 requirements |
         Inconclusive | Configuration-dependent -->
  </overallResult>

  <evaluationVariants>
    <evaluationVariant id="baseline" completeness="complete" result="">
      <sourceOption rawText="">
        <evidence page="" sourceLabel="" rawText="" />
      </sourceOption>

      <criteria>
        <criterion id="raw-material" status="">
          <evaluatedValue value="" rawText="" />
          <designEvidence />
          <requirement>Source-list status: Listed / Not listed in the guideline / Cannot assess</requirement>
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="tube-length" status="">
          <evaluatedValue value="" unit="mm" rawText="" />
          <designEvidence />
          <requirement />
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="end-configuration" status="">
          <designEvidence />
          <requirement>Only free ends or B4 ends (soft seal).</requirement>
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="bend-angles" status="">
          <designEvidence />
          <requirement>0 &lt; θ &lt; 180 for each angle between two bends.</requirement>
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="prohibited-features" status="">
          <designEvidence />
          <requirement>None of the prohibited PS4 features may be present.</requirement>
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="auto-crimping-straight-lengths" status="">
          <designEvidence />
          <scheduleMapping
              operation="map-standard-bender-schedule"
              rule="standard_ordered_bender_schedule/v1"
              mappingStatus="">
            <mappedValue name="l1" sourcePoint="" endPoint="" valueMm="" />
            <mappedValue name="l2" segmentId="" sourcePoint="" endPoint="" valueMm="" />
            <mappedValue name="l4" sourcePoint="" endPoint="" valueMm="" />
            <mappedValue name="theta1" sourcePoint="" valueDeg="" />
            <mappedValue name="theta4" sourcePoint="" valueDeg="" />
            <derivedValue name="l3" terminalBendPoint="" terminalEndpointPoint="" valueMm="" />
          </scheduleMapping>
          <calculatorResult operation="auto-crimp" overallStatus="">
            <dimension name="l1" designValueMm="" requiredMinimumMm="" formulaOrLookup="" status="" />
            <l2Segments status="" requiredMinimumMm="" formulaOrLookup="">
              <segment segmentIndex="" sourcePoint="" endPoint="" designValueMm="" requiredMinimumMm="" formulaOrLookup="" status="" />
            </l2Segments>
            <dimension name="l3" designValueMm="" requiredMinimumMm="" formulaOrLookup="" status="" />
            <dimension name="l4" designValueMm="" requiredMinimumMm="" formulaOrLookup="" status="" />
          </calculatorResult>
          <requirement>Requirement returned by the deterministic calculator from the applicable PS4 table/formula.</requirement>
          <reason />
          <missingInformation />
        </criterion>
        <criterion id="robot-pickup" status="">
          <designEvidence />
          <requirement>Condition evaluated by the deterministic calculator: L2 ≥ 34 or L1 ≥ L1(θ1 ≤ 90) mini + 34.</requirement>
          <reason>Gates feasibility only if the drawing explicitly requires robot pickup.</reason>
          <missingInformation />
        </criterion>
        <criterion id="crimps-objective" status="">
          <designEvidence />
          <requirement>Calculator-derived automatic-crimp geometry: every required L1, L2, L3, and L4 must meet its PS4 minimum.</requirement>
          <reason>Pass, Fail, or Warning is derived from the calculator-backed geometry; LEFT/RIGHT callout omission does not determine this objective.</reason>
          <missingInformation />
        </criterion>
      </criteria>

      <robotPickupAssessment
          applicability=""
          condition=""
          technicalFeasibilityImpact="">
        <applicabilityEvidence />
        <conditionEvidence />
        <disclaimer />
      </robotPickupAssessment>

      <failedCriteria>
        <criterionRef id="" />
      </failedCriteria>
      <cannotAssessItems>
        <item criterionId="" detail="" />
      </cannotAssessItems>
    </evaluationVariant>
  </evaluationVariants>

  <guidelineLimitations>
    <limitation>This result evaluates only the requirements stated in Pipes Guideline PS4.pdf.</limitation>
    <limitation>The guideline provides no criteria for any design aspect not listed in this evaluation.</limitation>
    <limitation>A raw-material designation that is not in the source list is not given an explicit pass/fail disposition by the guideline.</limitation>
    <limitation>The appendix is a dimensional illustration. It does not add written acceptance values beyond the stated table, formulas, robot-pickup condition, and crimp objective.</limitation>
    <limitation>For PS4 prohibited features and automatic-crimp fields, a reviewed omission is treated as absence; other missing inputs remain unavailable.</limitation>
    <limitation>Robot-pickup gating and closed-world automatic-crimp reporting are skill assessment policies, not additional PS4 guideline requirements.</limitation>
  </guidelineLimitations>
</ps4PipeFeasibilityEvaluation>
```

Populate one `<evaluationVariant>` for the baseline when no candidate exists, or one per candidate when the intermediate XML contains `<candidate>` elements. Copy the intermediate `<assessmentAssumption>` values unchanged. For multiple variants, set `<overallResult status="Configuration-dependent">`; do not reduce unselected candidates to a single pass/fail result. Each populated `evaluatedValue`, `designEvidence`, `sourceOption`, `applicabilityEvidence`, and `conditionEvidence` must retain the relevant `<evidence page="…" sourceLabel="…" rawText="…" />` elements from the input.

Use these controlled status values:

- Feasibility-gating criteria: `Pass`, `Fail`, or `Cannot assess`.
- Robot pickup and crimp objective: `Met`, `Not met`, or `Cannot assess`.
- Variant result: `Feasible against the assessed PS4 requirements`, `Not feasible against the PS4 requirements`, or `Inconclusive`.
- Robot-pickup `technicalFeasibilityImpact`: `Gating because explicitly required` or `Non-gating because not explicitly required`.

When a non-gating robot-pickup condition is `Not met` and the variant is feasible, populate `<disclaimer>` with this exact text: `Robot pickup is not explicitly required by the drawing, so this does not change the technical-feasibility result. However, the stated robot-pickup condition is not met; if robot pickup becomes required, this design would not meet that condition.`

## Restrictions and limitations to call out in every result

- The guideline lists 13 rough-steel diameter/thickness designations used at PS4; it does not state a pass/fail disposition for an unlisted designation.
- Tube length is restricted to 200–3000 mm, except Ø8 and Ø10 with 1 mm wall thickness, which are restricted to 200–1500 mm.
- Only free ends or B4 ends (soft seal) are possible.
- Every bend angle between two bends is strictly greater than 0° and strictly less than 180°.
- Banjo screws, beads (form B), diameter changes, end chamfers/machining, combined tube/flexible/hose designs, soldering/transplanting, EO-2 soft seals, and metal seals are outside PS4 specifications.
- Auto-crimping straight lengths must meet the diameter- and angle-dependent minima in this skill.
- The robot-pickup condition is always reported when it can be assessed. Under this skill's assessment policy, it affects technical feasibility only if the drawing explicitly requires robot pickup; a non-gating failure is communicated as a disclaimer.
- PS4 has an automatic-crimp geometry objective. The calculator assesses the required L1, every L2, L3, and L4 values; the customer-facing status is **Pass**, **Fail**, or **Warning**. LEFT/RIGHT callout omission is retained as evidence but does not change a passing geometry result.
- The guideline does not provide evaluation criteria beyond the subjects documented here.
