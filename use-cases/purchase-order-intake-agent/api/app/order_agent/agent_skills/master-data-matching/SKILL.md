---
name: master-data-matching
description: Resolve sold-to, ship-to and materials against ECC master data with the search tools, and decide when a match is good enough.
---

# Master-data matching

## Sold-to
1. Call search_customers with everything known about the buyer: name, address, country, sender
   e-mail, buyer tax id, buyer phone. Strong identifiers (tax id 1.00, e-mail domain or phone 0.97)
   outrank name similarity. Ignore the supplier's own name and addresses.
2. Take the top candidate only if its score is clearly the best (`ambiguous` is false). Otherwise
   submit the best candidate with a low confidence and a review flag naming the alternatives.
3. Call get_customer_profile for the chosen customer before anything else that depends on it.
   Its `recent_corrections` are decisions reviewers made for this customer before: follow them
   for the same printed values.

## Sales organisation
Use the inbox sales organisation if given; otherwise the profile's only sales area. If the customer
has several, omit `sales_org`: the system derives it from the materials or the reviewer chooses.

## Ship-to (requirement: single ship-to is used automatically)
- Choose only among the profile's partners with function `ship-to`.
- If the document states a delivery address, pick the ship-to whose name/address matches it.
  If none matches, leave `ship_to.value` null, keep the address in `ship_to_address`, and flag it.
- If the document states no delivery address, omit `ship_to`: the system uses the only or the
  default ship-to.

## Materials (requirement order: description first, material number second, colour, then /XA)
1. Call search_materials once for all lines with `sold_to` and `sales_org`. Per line pass the
   description as printed, every printed code, the colour, the pack and the printed unit price.
   The candidates' `score` only ranks them - read each candidate description yourself.
2. **Description first.** Choose the candidate whose description is the same product as the line:
   same brand / product line, grade or grit, size, pack and colour. Any number-bearing difference
   (P600 vs P400, "MM 400" vs "MM 600", 150MM vs 75MM, 1L vs 5L, C356 vs E000) means a different
   product - never pick it, however similar the rest of the text is. Abbreviations may differ
   ("PKIT" = "PELKIT", "STD" = "STANDARD", "HARD" = "HARDENER", a shortened brand name).
3. **Material number second.** A printed code that equals the candidate's ECC material number,
   sales code or the customer's own code confirms the choice. Codes from the buyer's own system or
   another manufacturer (brand + article number in a dealer's product column, e.g. "BrandX 5211985")
   are not ECC numbers: never choose a material because of them, and never put them in
   `material.value`.
4. **Colour** next: when several candidates fit, the colour (name or code) decides.
5. **/XA code**: if several still fit, prefer the candidate whose sales code carries the `/XA`
   suffix of the line (or of the sales organisation's local range). It may not reduce the list to
   one.
6. Still more than one: submit the best one with a confidence below 0.80 and a review flag
   naming the alternatives. Pack-only differences (quart vs gallon, 5 L vs 20 L): the pack on the
   document decides; if the document gives no pack, flag the line.
7. Read `notes`: a pack conflict between code and line must become a review flag.
8. **Retry in English.** If no candidate fits a line, call search_materials again for those lines
   with an English translation and expanded abbreviations of the description (e.g. German
   "Gitternetz Blau 70mmx10m 240" -> "mesh sheet blue 70mm x 10m 240"). The retry text is a query
   only; `description.value` keeps the printed text.
9. If still no candidate fits, leave `material.value` null, keep the printed code in
   `material.raw`, and say in `material.note` what was searched - the reviewer will decide.

## Material evidence (required for every line with `material.value`)
Submit `material_evidence` on the line:
- `matched`: each attribute the line and the chosen material share, as "attribute: value"
  (product line, grade / grit, size, pack, colour, manufacturer part number, price).
- `conflicts`: anything that disagrees or that the line states and the material does not confirm.
  Never hide a conflict to raise the confidence; an empty list means everything agrees.
- `alternatives`: material numbers of other candidates that fit equally well.
`material.confidence` is how sure you are that it is the same product (0.95+ when every printed
attribute agrees and nothing else fits). The system then lowers it for conflicts, a printed price
that differs from the customer price (`price_check`), or a weak text resemblance, and sends
alternatives to review.

## Quantities
Call convert_quantities once for all resolved lines (see line-item-extraction for units).
