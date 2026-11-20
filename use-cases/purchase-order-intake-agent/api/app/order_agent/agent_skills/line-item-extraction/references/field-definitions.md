# Line field definitions

## Line type

| line_type | Definition |
|---|---|
| MATERIAL | A product line: its own quantity (or its own price) and goods to deliver. Default. |
| CHARGE | Freight, transport, packing, handling, surcharges and similar fees, whether printed as a line or only in the totals block. Keep the amount; never map it to a material. |
| TEXT | A note that must be kept as its own line. Notes that change the order go to `header_note` instead (see SKILL.md). |

## Fields

| Field | Definition |
|---|---|
| customer_item_no | The buyer's own line / position number as printed ("Pos.", "Item", "No.", "STT"). Not a running row counter you create, not a material code. Omit it if the document has no line numbering. |
| material | `value` = ECC material number chosen with search_materials (never invented; null with low confidence if none fits); `raw` = product code as printed on the line (from the code column, or embedded in the description). Requires `material_evidence` whenever `value` is set. |
| customer_material | The buyer's own article code for the product, as printed (it is checked against the customer-material info records). Not the ECC material number, not another manufacturer's code, not a tariff / HS code. |
| description | The description column exactly as printed, multi-line cells joined with one space. Keep codes, abbreviations, grit / size / pack tokens and word order; never the master-data description. The product-code column is not part of it. Vietnamese: `value` = English translation, `raw` = Vietnamese; other languages as printed. |
| quantity | `value` = quantity in the unit of `uom.value` (after convert_quantities when the line orders a volume); `raw` = quantity and unit as printed. |
| uom | `value` = SAP sales unit (from convert_quantities, or PC when the quantity counts packages); `raw` = unit as printed. A pack-size column ("4 Ltr", "20L", "946ML") is the package, not the unit of the quantity. |
| unit_price | Price per unit as printed, number format converted to a plain number. Never converted to another unit or currency. Optional. |
| line_amount | Line total as printed (after line discounts if the document shows them), number format converted to a plain number. Optional. |
| colour_code | Only a RAL code ("RAL 9010"), an NCS code ("S 1050-Y90R") or the manufacturer's colour / tint code of a paint or coating ("C356", "F236"). Never a colour name ("blue", "Blau") and never the colour of non-coating goods (abrasives, tapes, sealants, consumables). Omit otherwise. |
