---
name: confidence-and-fixes
description: Assign calibrated confidences, decide which automatic fixes are allowed, and react to submit_order issues.
---

# Confidence and fixes

## Confidence rubric (per field)
| Confidence | When |
|---|---|
| 0.95 - 1.00 | Printed verbatim and unambiguous, or resolved by an exact identifier |
| 0.80 - 0.95 | Normalised or translated from clear text; resolved by a strong but not exact match |
| 0.50 - 0.80 | Inferred from context, partially legible, or chosen between close candidates |
| below 0.50  | A guess - prefer `value: null` |

Below 0.80 a field needs human review (requirement). Do not inflate confidences to get an order
posted, and do not deflate them to be safe: for sold-to and ship-to the system also recomputes
match confidence from master data and keeps the lower of the two; for materials it applies the
checks described under "Material evidence" in master-data-matching.

## Allowed automatic fixes (always explain in `note`)
- normalising formats (numbers, dates, currency symbols, Incoterms codes);
- translating Vietnamese free text to English (other languages stay as printed) and mapping
  printed units to SAP units;
- completing a truncated or misspelt buyer / ship-to name from master data when the identifier match is strong;
- converting quantities with convert_quantities;
- choosing the only ship-to partner, or the one matching the stated delivery address;
- applying a reviewer's earlier correction for the same customer and printed value.

## Never
- invent PO numbers, customer or material numbers, quantities or prices;
- rewrite printed free text: no expanded abbreviations, reordered words, dropped codes or
  master-data descriptions in `value` (only the Vietnamese translation is allowed);
- silently override what the document says with master data - when they disagree (currency,
  incoterms, address, pack), keep the document value and add a review flag;
- drop lines you could not resolve - submit them with `material.value = null`.

## Not review reasons (do not flag)
Review flags are for doubts that could change what is posted. Do not flag routine situations:
- PO or delivery dates in the past - orders are often entered later; scheduling is handled in ECC;
- a quantity printed without a unit when the line clearly counts packages (kit, can, pail,
  "number" column) - it is pieces; say so in `note` with confidence 0.9 or higher;
- a document whose customer-PO field is explicitly empty ("none", "keine", "-") but which prints the
  buyer's own order number (e.g. "Bestellung", "Order No.") - use that number as the PO number;
- a date in an unambiguous local format for the document's country (dd/mm/yyyy in Europe and Asia);
- values the system fills from master data (payment terms code, plant, division, order type);
- a line whose material you left empty (`material.value = null`) - the system already raises
  "Material could not be determined"; explain what you searched in `material.note` instead;
- terms or notes that do not change the order (shelf-life clauses, generic disclaimers).

## Reacting to submit_order
The result lists `issues` (ERROR, REVIEW, WARNING), `system_fills` and `ignored_fields`.
- Fix and resubmit only when the documents or master data give you the answer (wrong field name,
  a missed conversion, an obviously wrong pick).
- Do not try to make REVIEW issues disappear by changing confidences; reviewers need them.
- `system_fills` are applied automatically (dates, currency, terms, plant); do not repeat them.
- After at most three submissions, stop and write the reviewer summary.
