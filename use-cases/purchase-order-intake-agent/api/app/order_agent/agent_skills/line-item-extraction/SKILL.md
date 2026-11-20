---
name: line-item-extraction
description: Extract order lines (codes, descriptions, quantities, units, prices, colours), separate charges and notes, and convert quantities to sales units.
---

# Line-item extraction

Read `references/field-definitions.md` for the definition of every line type and line field.
Report lines in document order as `{line_type, fields}`; each field is a FieldValue (`value`,
`raw`, `confidence`, `source`, `note`). Omit fields that are not on the line.

## What is a line
- A line has its own quantity (or its own price). Heading rows ("North", a brand or section
  name), column headers, free-text notes and kit components without their own quantity are not
  lines.
- Charges count as CHARGE lines even when they appear only in the totals block.
- Notes that change the order (e.g. "thinner to be supplied is X", "order product Y for colour Z")
  are not lines: put them in `header_note` and use them when resolving materials.

## Codes
Pass every code printed on the line to search_materials (`codes`), including codes embedded in the
description and the buyer's article code. A code is a hint, not an answer: many documents print
codes of the buyer's own system or of another manufacturer (e.g. a dealer's product column
"BrandX 5211985", "BrandY 589826"), which are not ECC material numbers - see master-data-matching.
Forms seen on real orders:
- 6-7 digit material numbers ("SAP code", "Order code", "Item"): try them first;
- sales codes with pack and market suffix, e.g. `ABC123/20L/XA`, `ABC123/4AP`, `ABC123/A4LT`;
- sales codes printed without their leading letter (`G1421` for `OG1421`) - the search tolerates it;
- a brand word before the number ("Brand 455991") - pass the whole token, the search splits it;
- tariff / HS codes (e.g. "320890..") are never material codes.

## Units, packs and quantities
- When the quantity counts packages (cans, pails, kits, "number"), the unit is pieces.
- When the quantity is a volume (litres, gallons), call convert_quantities with that volume unit;
  it divides by the material's fill volume (a "20 L" pail may hold 17.6 L). Use the returned
  `sales_qty` / `sales_uom` as `quantity.value` / `uom.value` and explain the conversion in `note`.
- Vietnamese: "Lít" = litre, "thùng" / "t" = pail, "bộ" = set/kit, "lon" = can. German: "Stück"/"St"
  = piece. A note such as "250 t" next to 4 400 Lít means 250 pails: if the converted quantity and
  such a note disagree, keep the converted value and add a review flag for that line.
- If `exact` is false, or the unit column contradicts the pack in the description (e.g. "1QL"
  pack but "Gallon" column), decide from the description/price, explain in `note`, lower the
  confidence and add a review flag.
- If the code's pack (e.g. `/1LT`) contradicts the line's volume (e.g. "4 Ltr") flag the line
  instead of choosing silently.

## Colours
A colour that is not a valid `colour_code` stays in the description and is still passed to
search_materials as `colour`.
