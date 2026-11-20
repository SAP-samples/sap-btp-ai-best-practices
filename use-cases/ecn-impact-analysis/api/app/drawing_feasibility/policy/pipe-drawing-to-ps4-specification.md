---
name: pipe-drawing-to-ps4-specification
description: Transform a complete pipe engineering drawing into the evidence-preserving ps4-pipe-specification XML contract used by the PS4 guideline evaluator.
version: 2.1
rubric: ../pipes-guideline-ps4/ps4-pipe-evaluation-rubric.md
---

# Pipe Drawing to PS4 Specification

Transform a complete raw pipe engineering drawing into one well-formed `ps4PipeSpecification` XML document using `../pipes-guideline-ps4/ps4-pipe-evaluation-rubric.md` as the canonical contract.

This skill extracts drawing evidence; it does **not** determine whether the design passes PS4. Treat the drawing as a complete part/configuration definition: for PS4 prohibited-feature and automatic-crimp fields only, a feature not called out or depicted in the reviewed drawing scope is absent by drawing omission. Do not add engineering information, globally select a drawing alternative, or create combinations of alternatives that the drawing does not state.

## Inputs

- A pipe engineering drawing PDF or image.
- The shared rubric at `../pipes-guideline-ps4/ps4-pipe-evaluation-rubric.md`.

## Output contract

Return **only** one well-formed XML document: no Markdown fence, heading, commentary, or PS4 pass/fail verdict outside the root element.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ps4PipeSpecification contract="ps4-pipe-specification" schemaVersion="2.1">
  <!-- Use the complete canonical structure in ../pipes-guideline-ps4/ps4-pipe-evaluation-rubric.md. -->
</ps4PipeSpecification>
```

Use the rubric's lower-camel-case elements and attributes. Every populated `field`, `measurement`, `feature`, `automaticCrimp`, `row`, `option`, `mapping`, or `sourceOption` must include one or more `<evidence page="…" sourceLabel="…" rawText="…" />` children, except that a `row` may use its own direct evidence. Escape XML-reserved characters in attribute values. Source wording remains exact after XML decoding.

## Mandatory calculation tool

The LLM must **never** perform arithmetic, formatting normalization, rounding, table lookup, or numeric comparison itself. This transformation skill has one quantitative operation: forming `<field name="ps4RawMaterialDesignationCandidate">`. When both directly evidenced `<measurement name="outerDiameter">` and `<measurement name="wallThickness">` are available, call:

```bash
python3 ../pipes-guideline-ps4/tools/ps4_measurement_calculator.py \
  material-designation \
  --diameter-mm "<source-backed outerDiameter>" \
  --wall-thickness-mm "<source-backed wallThickness>"
```

Copy the JSON `candidate` value into the XML field and retain evidence for both input measurements. If either input is unavailable or the tool returns an error, keep the field `state="not_stated"` and report the extraction limitation. Do not manually format or calculate a fallback value.

Do not call a calculation tool to reconstruct missing drawing geometry except for the approved standard bender-schedule mapping described below. This skill must still extract total length and every bend-schedule row directly from the drawing. The evaluator uses the remaining calculator operations later. See `../pipes-guideline-ps4/tools/README.md` for the shared tool contract.

### Approved standard bender-schedule mapping

For an ordered standard bender schedule, call `map-standard-bender-schedule` when all source-backed rows satisfy the following pattern:

1. The first row is a start endpoint with no bend angle and no bend radius.
2. The final row is a terminal endpoint with no bend angle and no bend radius.
3. Every row between those endpoints is a bend row with both `ANGLE` and `BEND RADIUS`.
4. Every row other than the terminal endpoint has a `STRAIGHT PART` value for the run to the next row.

Pass the copied schedule rows to the deterministic tool. Do not select rows, map dimensions, calculate the final bend arc, or validate the pattern manually:

```bash
python3 ../pipes-guideline-ps4/tools/ps4_measurement_calculator.py \
  map-standard-bender-schedule \
  --schedule-json '<copied ordered schedule-row JSON>'
```

The row objects use `point`, `bendAngleDeg`, `bendRadiusMm`, and `straightPartMm`; use `null` for a blank angle, radius, or straight-part cell.

#### Finding the row roles

Use the copied table order, not the visual pipe orientation or the numeric magnitude of a length:

1. The first schedule row is the **start endpoint**. Its `STRAIGHT PART` is L1.
2. The first following bend row supplies θ1 from `ANGLE`.
3. For every bend row followed by another bend row, the current row's `STRAIGHT PART` is one L2 segment ending at the next row. Keep every such segment; do not select only the shortest or longest one.
4. The final bend row is the bend row immediately before the terminal endpoint. Its `STRAIGHT PART` is L4 and its `ANGLE` is θ4.
5. The terminal endpoint is the final schedule row. It may have a blank `STRAIGHT PART`; that blank is valid and must not invalidate the mapping.
6. The tool derives L3 from the final bend row's L4, bend radius, and θ4.

Do not report auto-crimp geometry as unavailable merely because the drawing does not use the PS4 labels. If the copied rows match this pattern, the tool is the required source-backed mapping authority. Copy the tool output as follows:

- `l1` → `<measurement name="l1">`, `state="mapped"`.
- `theta1` → `<measurement name="theta1">`, `state="mapped"`.
- Every `l2Segments` item → one `<l2Segments><measurement name="l2">`, each with its `segmentId`, `sourcePoint`, and `endPoint`, all `state="mapped"`.
- `l4` → `<measurement name="l4">`, `state="mapped"`.
- `theta4` → `<measurement name="theta4">`, `state="mapped"`.
- `l3` → `<measurement name="l3">`, `state="derived"`, with `mappingRule="standard_ordered_bender_schedule/v1"` and calculator provenance.
- `robotPickupL2` → `<robotPickup><measurement name="l2">` when present; it is the first mapped bend-to-bend L2 segment.
- A successful result → `<mapping state="mapped" rule="standard_ordered_bender_schedule/v1" operation="map-standard-bender-schedule" mappingStatus="Mapped">`.

Attach source evidence for the relevant start endpoint, first bend, every mapped L2 segment, final bend, terminal endpoint, and copied calculation inputs. When the tool returns `mappingStatus="Cannot assess"`, retain every schedule row but leave the PS4 auto-crimp geometry unavailable, record its reasons as extraction limitations, and do not select alternative rows or derive geometry manually.

## Extraction procedure

1. **Identify the drawing and units.** Extract the source file name, page numbers, drawing number, version, lifecycle state, description, and stated linear-dimension unit. Preserve title-block wording exactly.
2. **Extract the pipe definition.** Capture stated diameter, thickness, total length, material designation, and cross-measure. Retain both the source text and normalized numeric values with units.
3. **Normalize the PS4 material candidate only when possible.** When both diameter and thickness are explicitly stated, call the mandatory `material-designation` calculator operation above and store its JSON `candidate` in `<field name="ps4RawMaterialDesignationCandidate">` with evidence for both input values. This is a comparison key, not an approval or a feasibility result. Never compose this value manually.
4. **Copy the bender schedule row by row.** When the drawing provides columns such as `POINT`, `X`, `Y`, `Z`, `ANGLE`, `BEND RADIUS`, `SEGMENT`, and `STRAIGHT PART`, copy each source row into `<bendSchedule><row>`. Do not omit rows with blank angle or radius cells.
5. **Extract endpoint information and candidate configurations.** For Point 1 and the last point, capture connection designations, seals, and all text alternatives. Record B4 soft-seal, EO-2 soft-seal, and metal-seal mentions as `<feature>` elements with the rubric's `finding` value. When a single end callout lists alternatives, populate that pipe candidate's `end_configurations` with **one `state="present"` entry per exact source option**; do not collapse the callout into one label, mark it `not_stated`, or invent a cross-product. For example, `Parker EO2 or CAST B4` plus `Parker EO or CAST B3` must yield the four source-backed options `Parker EO2`, `CAST B4`, `Parker EO`, and `CAST B3`. A set that includes a free/B4 option and an EO-2/metal-seal option is evaluated as limited compatibility: it must retain all evidence so the evaluator can issue a warning that names the compatible and non-compatible source options. Create separately evaluable `<candidate>` entries only for distinct complete pipe configurations that the drawing explicitly associates; do not select an option globally.
6. **Capture PS4-relevant features and their candidates.** For banjo screw, beads/form B, diameter changes, end chamfers or machining, combined tube/flexible/hose, soldering/transplanting, EO-2 soft seals, and metal seals, use `finding="present"` only for source-backed presence and `finding="explicitly_absent"` only for an explicit source statement of absence. When the complete reviewed drawing scope has no callout or depiction for one of these prohibited features, record `finding="absent_by_drawing_omission"` with `state="reviewed_omission"` and evidence naming the reviewed page(s), views, callouts, and notes. Retain EO-2 and metal-seal alternatives with their matching `end_configurations` evidence; they are warnings rather than failures only when that same option set includes a source-backed free/B4 alternative. Do not treat an omitted prohibited feature as unavailable.
7. **Map auto-crimp geometry, then capture robot-pickup and crimp evidence.** Apply the `L1`, `L2`, `L3`, and `L4` definitions in `../pipes-guideline-ps4/ps4-pipe-evaluation-rubric.md`. After copying the complete ordered bender schedule, call the approved `map-standard-bender-schedule` operation whenever its documented source-row pattern is present. Copy all mapped values, L2 segments, provenance, and row evidence exactly as specified above. The tool is the only authority for selecting the schedule rows and deriving L3. If the tool cannot map the schedule, preserve the raw schedule and its tool reasons, but do not populate L1, L2, L3, L4, θ1, or θ4 from schedule rows. For robot pickup, populate `<field name="requirementOrApplicability">` only when the drawing explicitly says it is required or does not apply; otherwise retain `state="not_stated"`. When present, copy the tool's `robotPickupL2` to `<robotPickup><measurement name="l2">`; also copy the mapped L1. Preserve any generic crimp statement in `<field name="automationStatement">`. For `<automaticCrimp side="left">` and `<automaticCrimp side="right">`, use `finding="present"` only when source evidence identifies an automatic crimp for that side; a source statement that expressly identifies automatic crimps for both left and right sides is evidence for both elements. Use `finding="explicitly_absent"` only for an explicit source statement that automatic crimping is absent. If the complete reviewed drawing scope contains no automatic-crimp callout or depiction for a side, use `finding="absent_by_drawing_omission"` with `state="reviewed_omission"` and reviewed-scope evidence.
8. **Attach evidence to every conclusion.** Every populated field needs page, source label, and raw text. Add unreadable or unavailable information as `<limitation>` children in `<extractionLimitations>`.
9. **Return the canonical XML only.** It must declare `contract="ps4-pipe-specification"`, `schemaVersion="2.1"`, and the fixed complete-drawing, automatic-crimp, and robot-pickup feasibility-gating assessment assumptions. Do not append a PS4 pass/fail verdict. The guideline-evaluation skill consumes this XML later.

## Output quality gate

Before returning the XML, verify all of the following:

- The document is well-formed XML with root `<ps4PipeSpecification contract="ps4-pipe-specification" schemaVersion="2.1">`.
- `<assessmentAssumption>` declares `drawingScope="complete_part_configuration_definition"`, `prohibitedFeatureOmission="absent_by_drawing_omission"`, `automaticCrimpOmission="absent_by_drawing_omission"`, and `robotPickupFeasibilityGating="only_when_explicitly_required"`.
- Each extracted number retains its unit and source text.
- Every populated extracted value has source evidence with `page`, `sourceLabel`, and `rawText`.
- If the copied bender schedule has the complete start-endpoint/bend-row/terminal-endpoint pattern, `map-standard-bender-schedule` has been called. When it returns `mappingStatus="Mapped"`, L1, every L2 segment, L4, θ1, θ4, and derived L3 are populated with the stated mapping states; they must not be reported as unavailable because the drawing lacks literal PS4 labels.
- Every prohibited feature with no callout or depiction has `finding="absent_by_drawing_omission"`, `state="reviewed_omission"`, and evidence of the reviewed drawing scope.
- Every automatic-crimp side with no callout or depiction has `finding="absent_by_drawing_omission"`, `state="reviewed_omission"`, and evidence of the reviewed drawing scope. This is retained as drawing evidence only; the evaluator derives the automatic-crimp objective from calculator-backed geometry.
- Each `or`-separated end callout becomes one `end_configurations` set with one `state="present"` entry per exact source option. Do not create separate `<candidate>` entries for those alternatives and do not mark legible options `not_stated`; no globally selected configuration is asserted.
- The standard bender-schedule mapping is the only permitted mapping of unlabelled schedule `STRAIGHT PART` fields to PS4 auto-crimp geometry. A successful `map-standard-bender-schedule` result must retain row evidence, mapped L1/L2/L4/θ1/θ4 values, derived L3, and tool provenance. If the operation returns `Cannot assess`, keep all PS4 auto-crimp geometry unavailable; do not select rows manually.
- `finding="not_determined"` with `state="not_stated"` is used only where the fixed closed-world rule does not apply; an omitted PS4 prohibited feature or automatic-crimp side uses `finding="absent_by_drawing_omission"` instead.
- The robot-pickup applicability field records only an explicit drawing statement. Absence of such a statement remains `state="not_stated"`; it does not suppress extraction of mapped L1 or the mapped first L2 segment when the standard schedule operation succeeds.
- The XML contains no PS4 feasibility verdict.
- The fixed robot-pickup feasibility-gating policy is an evaluator policy, not extracted drawing evidence.

## Limitations to report in `<extractionLimitations>`

- The output is a structured transcription of drawing evidence, not a manufacturing-feasibility decision.
- A standard bender-schedule `STRAIGHT PART` can become PS4 L1, an L2 segment, or L4 only when `map-standard-bender-schedule` validates the complete ordered-schedule pattern. The operation also maps θ1/θ4 and derives L3. If it cannot validate the pattern, the PS4 auto-crimp geometry remains unavailable; manual row selection is not permitted.
- A drawing option is preserved with source evidence, not selected as a production configuration. A `or`-separated end callout remains one `end_configurations` set; if it includes free/B4 plus EO-2/metal alternatives, retain all options so evaluation can issue a non-gating limited-compatibility warning that names the compatible and non-compatible options. A partial candidate remains incomplete only for its own missing inputs, not because other alternatives exist.
- The complete-drawing assumption applies only to PS4 prohibited features and automatic-crimp sides. A required value for any other criterion remains unavailable when it is not stated or readable.
- Robot-pickup applicability is not inferred from omission; the evaluator separately applies the fixed feasibility-gating policy.
