# PS4 deterministic measurement calculator

`ps4_measurement_calculator.py` is the only component that performs PS4 numeric normalization, lookup, arithmetic, or numeric comparison. It uses only the Python standard library and writes one JSON document to standard output.

## Invocation

Run it with Python from the relevant skill folder:

```bash
python3 tools/ps4_measurement_calculator.py <operation> <arguments>
```

The drawing-transformation skill resolves the same tool as:

```bash
python3 ../pipes-guideline-ps4/tools/ps4_measurement_calculator.py <operation> <arguments>
```

Supply only values that have already been directly evidenced in the drawing. The tool accepts decimal points or decimal commas and an optional `Ø` diameter symbol, so the LLM does not need to reformat source numbers. The tool does not reconstruct missing geometry. Its sole approved geometry exception is `map-standard-bender-schedule`, which validates a complete ordered schedule before mapping its source-stated values to PS4 geometry.

## Standard bender-schedule mapping procedure

Use `map-standard-bender-schedule` when the drawing has an ordered `POINT` schedule with `ANGLE`, `BEND RADIUS`, and `STRAIGHT PART` columns.

1. Transcribe **every** row in drawing order. Preserve blank numeric cells as JSON `null`; do not omit the endpoint rows.
2. The first row must be the start endpoint: blank `ANGLE` and `BEND RADIUS`; its `STRAIGHT PART` maps to L1.
3. The first bend row supplies θ1 from `ANGLE`.
4. Each bend row followed by another bend row contributes one L2 segment: its `STRAIGHT PART` is the run to that next bend row. Pass every returned segment to `auto-crimp` as a repeated `--l2-mm` argument.
5. The bend row directly before the final endpoint maps its `STRAIGHT PART` to L4 and its `ANGLE` to θ4.
6. The final row must be the terminal endpoint: blank `ANGLE` and `BEND RADIUS`. Its `STRAIGHT PART` may be blank.
7. The operation derives L3 from the final bend row; it returns `robotPickupL2` as the first bend-to-bend L2 segment.

If the operation returns `mappingStatus: "Mapped"`, do not report the PS4 geometry as unavailable merely because the drawing contains no literal L1/L2/L3/L4 labels. If it returns `mappingStatus: "Cannot assess"`, preserve its reasons and do not select substitute rows manually.

## Operations

| Operation | Mandatory use | Arguments |
|---|---|---|
| `material-designation` | Format the PS4 raw-material comparison candidate. | `--diameter-mm`, `--wall-thickness-mm` |
| `raw-material` | Determine whether an already formatted candidate appears in the PS4 material list. | `--designation` |
| `tube-length` | Select the diameter- and wall-thickness-specific tube-length range and compare the directly stated total length. | `--diameter-mm`, `--wall-thickness-mm`, `--total-length-mm` |
| `bend-angles` | Compare each directly stated bend angle with the guideline range. Repeat `--angle-deg` for every angle. | `--angle-deg` |
| `map-standard-bender-schedule` | Validate the approved complete ordered-schedule pattern; map L1, every L2 segment, L4, θ1, θ4; and derive L3. | Exactly one of `--schedule-json` or `--schedule-file`, containing row objects with `point`, `bendAngleDeg`, `bendRadiusMm`, and `straightPartMm` |
| `l3-from-terminal-schedule` | Lower-level L3 calculation used by the schedule mapper; retained for direct deterministic validation, but pipe skills must use `map-standard-bender-schedule`. | `--terminal-straight-part-mm`, `--terminal-bend-radius-mm`, `--terminal-bend-angle-deg` |
| `auto-crimp` | Look up and, where required, calculate auto-crimp minima; compare only directly evidenced labelled dimensions or values produced by an approved mapping operation. Repeat `--l2-mm` for every L2 segment. | `--diameter-mm`; optional `--l1-mm`, repeated `--l2-mm`, `--l3-mm`, `--l4-mm`, `--theta1-deg`, `--theta4-deg` |
| `robot-pickup` | Calculate and evaluate the robot-pickup branches. | optional `--diameter-mm`, `--l1-mm`, `--l2-mm` |

A result of `Cannot assess` means the required source-backed input or covered guideline requirement is unavailable. A command error means the input cannot be evaluated. In both cases, do not calculate or compare manually.

## Non-negotiable boundary

The LLM must **never** manually perform arithmetic, trigonometry, table lookups, formatting normalization, rounding, numeric comparisons, or schedule-row selection for PS4. It must call the relevant operation and copy the JSON result into the skill's XML result. The approved `map-standard-bender-schedule` operation is the only geometry-mapping exception: it validates the complete start-endpoint/bend-row/terminal-endpoint pattern, maps L1, every L2 segment, L4, θ1, and θ4 from source-stated schedule fields, then derives L3 through the terminal-bend-arc rule. This tool does not decide any other drawing evidence, candidate configurations, end/seal eligibility, prohibited-feature evidence, or feasibility-gating policy.

## Test

```bash
python3 -m unittest tools/test_ps4_measurement_calculator.py -v
```
