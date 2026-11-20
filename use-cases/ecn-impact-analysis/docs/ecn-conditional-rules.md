# Conditional-rule Rosetta stone

This feature imports the engineering conditional-rule workbook as one immutable,
versioned HANA source artifact. It preserves source text and coordinates while
providing a reversible deterministic AST for later evidence derivation. Parsing
does not make a rule executable: runtime activation policy is absent in MVP v1.

## Inputs and outputs

The operator supplies one XLSX workbook containing a sheet named `Default`.
These ten headers are required exactly once, in any order: `Statut général`,
`Statut de modification`, `Indicateur de type d'objet`, `Numéro`, `Fusionné`,
`Version`, `Etat`, `Définition de règle`, `Description`, and `Modifié le`.

Normalization produces an `ecn-rules-normalized-v1` JSON representation and a
`CONDITIONAL_RULE_WORKBOOK` `SourceArtifact`. Both the original workbook bytes
and normalized content are written atomically to `ECN_IMPACT_SOURCE_ARTIFACT`
through `ArtifactRepository.publish_rule_artifact`. `Numéro` remains business
metadata; the internal identity is SHA-256 over the artifact ID and source row.
A workbook-level failure publishes nothing. Row-level `PARSED`, `UNSUPPORTED`,
and `AMBIGUOUS` results are retained as valid normalized output.

## Token and AST contract

| Source token | Deterministic meaning | Canonical AST | Reverse rendering |
|---|---|---|---|
| `A = B` | exact membership comparison | `EQUALS(A,B)` | `"A" = "B"` |
| `X + Y` | conjunction at one nesting level | `AND(X,Y)` | `(X + Y)` |
| `X / Y` | whitespace-delimited disjunction | `OR(X,Y)` | `(X / Y)` |
| `-(X)` | three-valued negation | `NOT(X)` | `-(X)` |
| `--> Inclure (A = B)` | ordered include assignment | `INCLUDE(A,B)` | `--> Inclure ("A" = "B")` |

Five representative bidirectional examples are:

| Source | Canonical rendering | Meaning |
|---|---|---|
| `A = B --> Inclure (S = V)` | `"A" = "B" --> Inclure ("S" = "V")` | One exact condition and one ordered assignment |
| `(A = B / A = C) + D = ET --> Inclure (S = V)` | `(("A" = "B" / "A" = "C") + "D" = "ET") --> Inclure ("S" = "V")` | Explicitly grouped OR combined with AND |
| `-(A = B / A = C) --> Inclure (S = V)` | `-("A" = "B" / "A" = "C") --> Inclure ("S" = "V")` | Three-valued negation over an explicit group |
| `A = E_GATEWAY_W/O --> Inclure (S = V)` | `"A" = "E_GATEWAY_W/O" --> Inclure ("S" = "V")` | Embedded slash remains part of a scalar |
| `'A = ""N.A."" --> Inclure (S = ""64348300"", X = Y)` | `"A" = "N.A." --> Inclure ("S" = "64348300", "X" = "Y")` | Export apostrophe and doubled quotes normalize while source text remains intact |

`ET` is a scalar value, not a logical keyword. `/` is an operator only as its
own whitespace-delimited token, so `E_GATEWAY_W/O` and `E_AUTOGUIDE_W/` are
scalar values. An unparenthesized mixture of `+` and `/` is `AMBIGUOUS`; the
parser never invents precedence. Unsupported actions such as `Exclure` are
`UNSUPPORTED`. Every failure retains a reason code and source offset.

Original cell text remains available for exact reconstruction, while canonical
rendering guarantees semantic AST round-trip rather than original spacing. A
leading export apostrophe and suspicious OR-of-negations are preserved as
warnings. An action with more than 100 assignments is structurally classified
as `LARGE_MULTI_ASSIGNMENT_ACTION`; that label is a safety quarantine, not a
factory-default interpretation.

## Evaluation truth table and activation boundary

| Expression | `TRUE` | `FALSE` | `UNKNOWN` |
|---|---|---|---|
| `AND` | all operands true | any operand false | otherwise |
| `OR` | any operand true | all operands false | otherwise |
| `NOT` | child false | child true | child unknown |

For equality, a present matching value is true. A nonmatching value is false
only when that characteristic's fact set is complete; otherwise it is unknown.
A missing characteristic is unknown. Runtime activation policy is absent in
MVP v1, so relevant rules are stored as
`NOT_EVALUATED_ACTIVATION_POLICY_MISSING` and no Include assignment changes
configuration facts. Suspicious OR-of-negations and actions with more than 100
assignments require separate policy opt-ins even after lifecycle policy exists.

## Plant assignment

Rules are not evaluated to decide plants. Instead, the brands a VC-group rule
requires (`GEN_E_BRAND = E_BRAND_JB`) are mapped to plants through the Settings
Plant Code Mapping, and a VC group belongs to every mapped plant of its
admitted rules. See [Rule-based plant assignment](rule-based-plant-assignment.md).

## Source-observed acceptance facts

The current ignored development workbook contains 4,575 rule definitions, all of which structurally parse and round-trip as `PARSED`; none are currently `UNSUPPORTED` or `AMBIGUOUS`. Their Include actions contain 6,081 assignments. Of the definitions, 117 contain at least one negation node, 23 carry `SUSPICIOUS_OR_OF_NEGATIONS`, and two carry `LEADING_EXPORT_APOSTROPHE`.

Three source rows have more than 100 Include assignments and are therefore quarantined as `LARGE_MULTI_ASSIGNMENT_ACTION`:

| Rule number | Include assignments |
|---|---:|
| `RL0000105591` | 486 |
| `RL0000105552` | 524 |
| `RL0000105532` | 499 |

These counts are acceptance facts for the currently supplied workbook, not universal rule-system limits. Structural parse success does not establish runtime executability: lifecycle admission and activation policy remain unconfirmed and absent in MVP v1, so production evaluation does not apply these actions.

## Operator commands

Validate the workbook contract and rule counts without publication:

```bash
PYTHONPATH=api python scripts/import_conditional_rules.py \
  --file /absolute/path/to/rules.xlsx --validate-only
```

Validate and append the immutable artifact to the current HANA schema:

```bash
PYTHONPATH=api python scripts/import_conditional_rules.py \
  --file /absolute/path/to/rules.xlsx
```

Both modes initialize or validate the seven-table HANA schema before processing,
and both show row progress. Application request logic never opens local files;
the CLI is the explicit operator-only source boundary.

## Related files and tests

- `api/app/ecn/rules/models.py`: immutable AST, parse, evaluation, and import records.
- `api/app/ecn/rules/parser.py`: lexer, parser, canonical renderer, and strict AST codecs.
- `api/app/ecn/rules/evaluator.py`: activation-gated three-valued evaluation.
- `api/app/ecn/rules/importer.py`: workbook contract, normalization, reload, and publication.
- `scripts/import_conditional_rules.py`: HANA-backed operator entry point.
- `tests/unit/test_rule_parser.py`, `test_rule_evaluator.py`, and
  `test_rule_importer.py`: focused synthetic contract tests.

Run the focused verification with:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_rule_parser.py \
  tests/unit/test_rule_evaluator.py \
  tests/unit/test_rule_importer.py
```
