# PS4 Pipe Evaluation Rubric

**Rubric ID:** `ps4-pipe-specification/xml/v2.1`

**Authoritative guideline source:** `Pipes Guideline PS4.pdf`

**Purpose:** A shared, evidence-preserving XML contract between a drawing-transformation agent and the PS4 guideline-evaluation agent.

This rubric separates **facts extracted from an engineering drawing** from **the PS4 feasibility determination**. It creates no additional engineering requirements. The guideline-evaluation skill remains the only place that assigns a PS4 pass, fail, or inconclusive result. When a drawing states multiple configurations, the rubric preserves every source-backed alternative as a separate candidate for independent evaluation; it does not select a production configuration.

**Fixed drawing and assessment policy:** Every drawing evaluated through this rubric is a complete part/configuration definition. The drawing is therefore closed-world for PS4 prohibited-feature fields; an automatic-crimp callout that is not depicted is still recorded as absent by drawing omission, but that fact is supporting evidence only. Automatic-crimp objective status is derived from calculator-assessed geometry. Robot pickup is always assessed separately when its condition can be evaluated, but it changes the technical-feasibility result only when the drawing explicitly requires robot pickup. A non-gating robot-pickup failure must be communicated as a disclaimer when the technical-feasibility result passes.

## XML conventions

Both pipe skills return one well-formed XML document and no Markdown fence or prose outside that document.

- The intermediate root is `<ps4PipeSpecification contract="ps4-pipe-specification" schemaVersion="2.1">`.
- The final-evaluation root is `<ps4PipeFeasibilityEvaluation contract="ps4-pipe-feasibility-evaluation" schemaVersion="2.1">`.
- Store structured status in the element's `state`, `finding`, or `status` attribute; do not encode a result only in prose.
- Preserve drawing values in `value`, `unit`, and `rawText` attributes where applicable.
- Every populated extracted value must contain one or more `<evidence>` children. Each evidence item has `page`, `sourceLabel`, and `rawText` attributes.
- Escape XML-reserved characters in every attribute value: `&` as `&amp;`, `<` as `&lt;`, `>` as `&gt;`, `"` as `&quot;`, and `'` as `&apos;` where required. Preserve the source wording after XML decoding.
- Use the element names and lower-camel-case field names in the canonical template. Do not add unstructured narrative sections.

## Evidence rules

Every populated value must include enough source evidence to find it again in the drawing:

- `page`: drawing page number
- `sourceLabel`: the drawing label, table column, note, or callout
- `rawText`: the exact source text, or `not text; drawing geometry` when an unambiguous visual feature is used

Use one of these field states:

| State | Meaning | Evaluation consequence |
|---|---|---|
| `stated` | The drawing explicitly states the value or feature. | May be evaluated. |
| `depicted` | The feature is unambiguously visible in the drawing geometry. | Preserve the evidence; the evaluator decides whether the guideline check can use it. |
| `alternative` | The drawing presents more than one source-backed option. | Create and evaluate each option as a separate candidate configuration; do not select one globally. |
| `mapped` | An approved mapping selects an exact source-stated schedule value for a PS4 field. | May be evaluated when the mapping rule and source evidence are retained. |
| `derived` | A deterministic tool has calculated a value using an approved mapping and source-backed inputs. | May be evaluated when the mapping rule, tool result, and source evidence are retained. |
| `reviewed_omission` | A prohibited feature or automatic-crimp field is not called out or depicted after the complete drawing scope has been reviewed. | Record `finding="absent_by_drawing_omission"`; it is accepted as absent for the prohibited-feature check. An automatic-crimp omission remains supporting evidence and does not decide the geometry-derived crimp objective. |
| `not_stated` | A field outside the closed-world prohibited-feature and automatic-crimp fields has no source evidence. | Cannot assess. |
| `unreadable` | The relevant source area cannot be read reliably. | Cannot assess. |

For `<feature>` and `<automaticCrimp>` fields, use `finding="present"`, `finding="explicitly_absent"`, `finding="absent_by_drawing_omission"`, `finding="stated_option"`, or `finding="not_determined"`. `explicitly_absent` requires an explicit source statement. For every PS4 prohibited-feature field and automatic-crimp field, use `finding="absent_by_drawing_omission"` with `state="reviewed_omission"` when the complete drawing scope contains no callout or depiction. Each `stated_option` must be copied into a source-backed candidate configuration for independent evaluation; it is not the globally selected configuration.

Do not relabel an omission as `explicitly_absent`, add tolerances, infer manufacturing capability, or form combinations of alternatives unless the drawing explicitly associates those choices.

## PS4 auto-crimp geometry definitions

Use the labels `L1`, `L2`, `L3`, and `L4` only for the corresponding geometry in the PS4 guideline diagram. They are not generic bender-schedule segment names.

- **L1:** the straight end-leg length from the left-hand free end to the tangent point of its adjacent bend. In an approved standard ordered bender schedule, map it from the `STRAIGHT PART` on the first endpoint row.
- **L2:** the straight length between the tangent points of two adjacent bends. In an approved standard ordered bender schedule, map every internal bend-to-bend `STRAIGHT PART` as a separate L2 segment; each segment must be evaluated.
- **L3:** the end-side length identified by the curved `L3` dimension line: from the free end of the right-hand end leg, around the `θ4` bend, to the junction with the incoming diagonal leg. It includes the `L4` end leg and the `θ4` bend as drawn; it is not interchangeable with `L4`. In an approved standard ordered bender schedule, calculate it as `L3 = terminal straight part + final bend arc` using the final bend row immediately before the terminal endpoint row.
- **L4:** the straight end-leg length from the right-hand free end to the tangent point of its adjacent `θ4` bend. In an approved standard ordered bender schedule, map it from the `STRAIGHT PART` on the final bend row immediately before the terminal endpoint.

In the same approved schedule mapping, `θ1` is the `ANGLE` on the first bend row immediately after the start endpoint, and `θ4` is the `ANGLE` on the final bend row immediately before the terminal endpoint. Call `map-standard-bender-schedule` to perform the row-role validation, mappings, and L3 calculation; do not select rows or calculate values manually. It treats the current row's `STRAIGHT PART` as the run to the next schedule row: start endpoint → first bend is L1, each bend → next bend is one L2 segment, and final bend → terminal endpoint is L4. The terminal endpoint may have a blank `STRAIGHT PART`. An unlabelled schedule segment remains unavailable unless the full approved mapping succeeds.

## Deterministic calculation boundary

The LLM must never perform PS4 arithmetic, trigonometry, table lookup, formatting normalization, rounding, or numeric comparison itself. It must call `pipes-guideline-ps4/tools/ps4_measurement_calculator.py` with directly evidenced drawing values and use the returned JSON result. The tool documentation is `pipes-guideline-ps4/tools/README.md`.

The calculator is required for:

- forming `<field name="ps4RawMaterialDesignationCandidate">` from stated diameter and wall thickness;
- mapping `L1`, every internal `L2` segment, `L4`, `θ1`, `θ4`, and derived `L3` from an ordered standard schedule using the approved `standard_ordered_bender_schedule/v1` rule;
- raw-material-list coverage, tube-length, and bend-angle comparisons;
- auto-crimp table lookup, applicable L1/L4 formula calculation, and auto-crimp comparison; and
- robot-pickup threshold calculation and comparison.

The calculator must not reconstruct missing geometry. The only approved geometry exception is `map-standard-bender-schedule`, which validates an ordered schedule with a start endpoint, bend rows, and a terminal endpoint; then maps its source-stated `STRAIGHT PART` and `ANGLE` values to L1, every L2 segment, L4, θ1, and θ4, and derives L3 through the approved terminal-bend-arc rule. Retain the relevant schedule-row evidence and tool result. When this mapping succeeds, use `state="mapped"` for L1, L2, L4, θ1, and θ4, and `state="derived"` for L3. If the tool cannot map the schedule, report `Cannot assess`; do not choose rows or calculate manually.

## Canonical intermediate specification XML

Produce one XML document using this structure. Keep numbers and source wording as shown in the drawing; preserve the raw representation as well as the normalized value where both are available. Empty attributes in this template are placeholders; omit an optional attribute only when the element is present solely to record an unavailable state.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ps4PipeSpecification contract="ps4-pipe-specification" schemaVersion="2.1">
  <assessmentAssumption
      drawingScope="complete_part_configuration_definition"
      prohibitedFeatureOmission="absent_by_drawing_omission"
      automaticCrimpOmission="absent_by_drawing_omission"
      robotPickupFeasibilityGating="only_when_explicitly_required" />

  <sourceDrawing file="">
    <pages>
      <page number="" />
    </pages>
    <field name="drawingNumber" value="" state="not_stated" />
    <field name="version" value="" state="not_stated" />
    <field name="lifecycleState" value="" state="not_stated" />
    <field name="description" value="" state="not_stated" />
    <field name="linearDimensionUnit" value="" state="not_stated" />
  </sourceDrawing>

  <pipe>
    <measurement name="outerDiameter" value="" unit="mm" rawText="" state="not_stated" />
    <measurement name="wallThickness" value="" unit="mm" rawText="" state="not_stated" />
    <measurement name="totalLength" value="" unit="mm" rawText="" state="not_stated" />
    <field name="materialDesignation" value="" rawText="" state="not_stated" />
    <measurement name="crossMeasure" value="" unit="mm" rawText="" state="not_stated" />
    <field name="ps4RawMaterialDesignationCandidate" value="" state="not_stated"
        calculationTool="ps4_measurement_calculator.py material-designation"
        derivation="Use the deterministic material-designation tool with stated diameter and thickness." />
  </pipe>

  <bendSchedule state="not_stated">
    <sourceTableLabels>
      <label value="" />
    </sourceTableLabels>
    <row point="">
      <coordinate axis="x" value="" unit="mm" />
      <coordinate axis="y" value="" unit="mm" />
      <coordinate axis="z" value="" unit="mm" />
      <measurement name="bendAngle" value="" unit="deg" />
      <measurement name="bendRadius" value="" unit="mm" />
      <field name="segment" value="" />
      <measurement name="straightPart" value="" unit="mm" />
      <evidence page="" sourceLabel="" rawText="" />
    </row>
  </bendSchedule>

  <ps4CriteriaInputs>
    <ends>
      <end id="point1">
        <field name="connectionDesignation" value="" state="not_stated" />
        <statedOptions />
        <ps4RelevantMentions>
          <feature name="freeEnd" finding="not_determined" state="not_stated" />
          <feature name="b4SoftSeal" finding="not_determined" state="not_stated" />
          <feature name="eo2SoftSeal" finding="not_determined" state="not_stated" />
          <feature name="metalSeal" finding="not_determined" state="not_stated" />
        </ps4RelevantMentions>
      </end>
      <end id="lastPoint">
        <field name="connectionDesignation" value="" state="not_stated" />
        <statedOptions />
        <ps4RelevantMentions>
          <feature name="freeEnd" finding="not_determined" state="not_stated" />
          <feature name="b4SoftSeal" finding="not_determined" state="not_stated" />
          <feature name="eo2SoftSeal" finding="not_determined" state="not_stated" />
          <feature name="metalSeal" finding="not_determined" state="not_stated" />
        </ps4RelevantMentions>
      </end>
    </ends>

    <prohibitedFeatures>
      <feature name="banjoScrew" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="beadsFormB" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="diameterChangeOnTube" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="endChamferOrMachining" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="combinedTubeFlexibleOrHose" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="solderingOrTransplanting" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="eo2SoftSeal" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <feature name="metalSeal" finding="absent_by_drawing_omission" state="reviewed_omission" />
    </prohibitedFeatures>

    <autoCrimping>
      <labelledDimensions>
        <measurement name="l1" value="" unit="mm" state="not_stated" />
        <l2Segments state="not_stated">
          <measurement name="l2" value="" unit="mm" state="not_stated"
              segmentId="" sourcePoint="" endPoint="" />
        </l2Segments>
        <measurement name="l3" value="" unit="mm" state="not_stated"
            mappingRule="standard_ordered_bender_schedule/v1">
          <calculation tool="ps4_measurement_calculator.py"
              operation="map-standard-bender-schedule"
              formula="L3 = terminal straight part + terminal bend radius × θ4(rad)" />
          <evidence page="" sourceLabel="terminal schedule rows" rawText="" />
        </measurement>
        <measurement name="l4" value="" unit="mm" state="not_stated" />
      </labelledDimensions>
      <labelledAngles>
        <measurement name="theta1" value="" unit="deg" state="not_stated" />
        <measurement name="theta4" value="" unit="deg" state="not_stated" />
      </labelledAngles>
      <mapping state="not_stated" rule="" operation="" mappingStatus="" />
    </autoCrimping>

    <robotPickup>
      <field name="requirementOrApplicability" value="" state="not_stated" />
      <measurement name="l1" value="" unit="mm" state="not_stated" />
      <measurement name="l2" value="" unit="mm" state="not_stated" />
    </robotPickup>

    <crimps>
      <field name="automationStatement" value="" state="not_stated" />
      <automaticCrimp side="left" finding="absent_by_drawing_omission" state="reviewed_omission" />
      <automaticCrimp side="right" finding="absent_by_drawing_omission" state="reviewed_omission" />
    </crimps>
  </ps4CriteriaInputs>

  <candidateConfigurations state="not_stated">
    <candidate id="" completeness="" description="">
      <sourceOption rawText="">
        <evidence page="" sourceLabel="" rawText="" />
      </sourceOption>
      <overrides>
        <override target="">
          <!-- Exactly one source-backed field, feature, measurement, or end element. -->
        </override>
      </overrides>
    </candidate>
  </candidateConfigurations>

  <extractionLimitations>
    <limitation />
  </extractionLimitations>
</ps4PipeSpecification>
```

## Candidate-configuration rules

- Common extracted facts apply to every candidate configuration.
- A disjunctive list of seals under one end callout (for example, `Parker EO2 or CAST B4`) is one source-backed end-option set, not separate complete pipe configurations. Preserve every option and its evidence in the current candidate so the evaluator can determine whether at least one free/B4 option is available.
- Create a separate `<candidate>` only for a distinct complete pipe configuration explicitly associated by the drawing. An override's `target` is the lower-camel-case XML path to the replaced field, feature, measurement, or end. Each override replaces that target only for its candidate's evaluation.
- Mark a candidate `completeness="complete"` only when the drawing associates all relevant choices for that configuration. Mark it `completeness="partial"` when the drawing states an option for only part of the design.
- Do not create a cross-product from alternatives at different endpoints or features unless the drawing explicitly states those combinations.
- The evaluation agent evaluates each candidate independently. A candidate is **Cannot assess** only for information missing from that candidate, not because another candidate exists.

## Criterion crosswalk

| PS4 guideline subject | XML input(s) | Handling rule |
|---|---|---|
| Rough-steel list | `<pipe>` measurements and `ps4RawMaterialDesignationCandidate` | Use the calculator's `material-designation` operation to form the candidate and `raw-material` operation to report list coverage. The candidate is a formatting normalization of drawing values, not a material approval. |
| Tube length | `<measurement name="totalLength">` | Preserve the drawing value and its unit. |
| Ends and seals | `<ends>`, `<candidateConfigurations>` | Preserve each endpoint's designation and every stated option. A free/B4 option alongside EO-2 or metal-seal alternatives is limited compatibility: retain all options and report a warning that names the compatible and non-compatible source options rather than failing the candidate. Fail only if no stated end option is free or B4. |
| Bend-angle range | `<bendSchedule><row><measurement name="bendAngle">` | Evaluate every stated bend angle. |
| Prohibited features | `<prohibitedFeatures>` | Because the input drawing is a complete part/configuration definition, a feature not called out or depicted in the reviewed scope is `finding="absent_by_drawing_omission"` with `state="reviewed_omission"` and passes this check. `stated_option` does not establish that an option is selected. |
| Auto-crimping minima | `<pipe>`, `<bendSchedule>`, `<autoCrimping>` | When the approved `map-standard-bender-schedule` operation validates the ordered schedule, map the first endpoint-row `STRAIGHT PART` to L1, every bend-to-bend `STRAIGHT PART` to an L2 segment, and the final bend-row `STRAIGHT PART` to L4; map θ1/θ4 from the first/final bend-row angles and derive L3. Preserve source-row evidence, `standard_ordered_bender_schedule/v1`, and tool provenance. Otherwise mark the mapping `not_stated`; do not choose schedule rows manually. |
| Robot pickup | `<robotPickup>` | Record only source-stated applicability and explicitly labelled `L1`/`L2` values. The evaluator always assesses the condition where the available inputs permit it. A robot-pickup result changes technical feasibility only when the drawing explicitly requires robot pickup; otherwise a condition failure is a non-gating disclaimer. |
| Crimp objective | `<pipe>`, `<bendSchedule>`, `<autoCrimping>`, `<crimps>` | Preserve generic/left/right automation statements as drawing evidence. Derive the customer-facing **Pass**, **Fail**, or **Warning** objective from the calculator-backed automatic-crimp geometry: every required L1, L2, L3, and L4 must pass; a mapping/input/calculator gap is Warning. Left/right callout omission does not downgrade passing geometry. |

## What the rubric intentionally does not decide

- It does not decide whether a drawing passes PS4.
- It does not choose a production configuration; it preserves every source-backed end option in the source-associated candidate and warns when only some listed options are PS4-compatible.
- It does not rename an unlabelled drawing dimension as `L1`, `L2`, or `L4` unless the deterministic `map-standard-bender-schedule` operation validates the documented ordered-schedule pattern; it derives L3 only through that same operation.
- It applies the fixed complete-drawing assumption to PS4 prohibited-feature fields. Automatic-crimp callout omissions remain extracted evidence; the evaluator derives the automatic-crimp objective from calculator-backed geometry.
- It applies the robot-pickup feasibility-gating policy only during evaluation: robot-pickup evidence remains an extracted drawing fact.
- It does not evaluate any property for which the PS4 guideline provides no criterion.
