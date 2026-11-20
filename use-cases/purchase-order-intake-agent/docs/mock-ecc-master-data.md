# Mock ECC master data, matching and seed CLI

## What it is

The application validates and enriches extracted purchase orders against SAP ECC master
data. Because no ECC system is available, the relevant ECC tables are mirrored in HANA
(`*_POAGENT`, see [hana-data-model.md](hana-data-model.md)) and filled from a curated seed file.
Three pieces work together:

| Piece | File | Role |
|---|---|---|
| Master-data access | `api/app/ecc/master_data.py` | Reads customers, partners, sales areas, contacts, materials, customer-material codes, customer prices (A305) and unit aliases; rows keyed by SAP field names |
| Matching | `api/app/orders/matching.py` | Pure scoring of customers and materials against what a document says; SAP ALPHA format helpers |
| Seed CLI | `api/app/cli/seed_master_data.py` | Validates and loads `data/seed/master_data.json` into HANA |

## What it is for

* The order agent's tools (`search_customers`, `get_customer_profile`, `search_materials`)
  and the deterministic validator use these functions to resolve the sold-to, ship-to and
  materials of an order, exactly as the requirements prescribe (exact match first, then fuzzy,
  ship-to from partner function `WE`, colour and the `/XA` code to narrow materials).
* In production only `ecc/master_data.py` changes (SAP Integration Suite iFlows or replicated
  master data, see [ecc-integration.md](ecc-integration.md)); matching and callers stay the same.

## Inputs and outputs

**Seed file** (`data/seed/master_data.json`):

```json
{"_meta": {"note": "..."}, "KNA1_POAGENT": [{"KUNNR": "759842", "NAME1": "...", "LAND1": "LK"}], "MARA_POAGENT": [...]}
```

Data notice: a few customer and material numbers are printed on the sample documents and
price lists and were reused; **all other customer, partner, contact and material numbers,
addresses, plants and sales areas are invented mock values.** In particular, for DE10:

* every product of the six German test orders exists (74 materials). Material numbers
  (71xxxxx) are invented so they never equal a printed code; descriptions are written in ECC
  style (full product names, e.g. `KESTREN AUTOFLUX MM 400 EU 1L`, `1Z PELKIT DISC 325U 150MM P600 100PC`)
  and some abbreviation expansions are assumptions;
* `A305_POAGENT` prices are **copied from the unit prices printed on those orders** (one per
  customer and product). They are not real ECC prices. One German test order (every price
  1,00 EUR) has no price records.

Seed files in `data/seed/`:
`de_order_lines.json` (lines parsed from the German PDFs: printed code, description, price),
`master_data.json` (generated seed) and `expected_lines.json` (generated acceptance list: each
test-order line with the material it must resolve to). Contacts use generic role
mailboxes on the customers' e-mail domains (domain matching works, no personal data).

The seed is designed so that the samples exercise both outcomes:

| Situation in the samples | What the seed contains | Expected effect |
|---|---|---|
| Ship-to differs from sold-to | sold-to with two `WE` partners | Ship-to resolved from the document |
| Similar company names | a second customer with a near-identical name, different VAT | VAT / e-mail domain disambiguates |
| Blocked customer | `AUFSD = 01` on a similar-name customer | Validation error if chosen |
| Quart / gallon variants of one code | two materials per sales code | Pack size selects the right one |
| Local vs export variant | same base code with and without `/XA` | `/XA` narrows the result |
| Code printed without letter prefix | sales codes stored with prefix | Tolerated with a small penalty |
| Customer's own article codes | `KNMT_POAGENT` records | Matched with score 0.99 |
| Pails filled below nominal size | `VOLUM` = fill volume (e.g. 17.6 L in a 20 L pail) | Litres converted to pieces exactly |
| Dealer orders printing their own product column ("Brand 5211985") | DE materials under invented numbers 71xxxxx, ECC-style descriptions | Material found by description, not by the printed code |
| Traded goods of other manufacturers | `MARA.MFRPN` = manufacturer part number printed in the description | Part number confirms the description match |
| Several variants with similar text (grit, size, pack) | e.g. five 150 mm discs P80-P600 | Only the variant with the same numbers fits |
| Customer-specific prices | `A305_POAGENT` per customer and material | Printed price confirms (or questions) the choice |
| Ship-to only in the e-mail body, unknown site | no matching `WE` partner | Order goes to review |

**Matching output** (`rank_customers`, `rank_materials`): candidates sorted by `SCORE` with
`MATCHED_ON` (signals used), `NOTES` (conflicts such as "line states 4.0 L but material pack
is 1.0 L") and `AMBIGUOUS` (top-2 scores closer than 0.05).

## Key code

Score scale used everywhere:

```text
1.00  exact identifier (material number, sales code, VAT / tax id)
0.99  customer-material info record (KNMT) code for this customer
0.95  manufacturer part number (MARA.MFRPN) printed as a code or inside the description
0.97  sender e-mail domain or phone number of a contact / customer
0.80-0.95 sales code with tolerated variations, or very close name + address
<0.80 description / name similarity only  -> below the auto-post confidence threshold
```

Sales codes are split into base, pack and suffix so one rule set covers the variants seen on
real documents:

```python
parse_code("IPJ715/20L/XA")  # ("IPJ715", 20.0, {"XA"})
pack_litres("A4LT")           # 4.0      pack_litres("1GL") -> 3.785
rank_materials(materials, codes=["G1421/1GL"])   # -> OG1421/GL 0.92, OG1421/QT 0.57
```

ALPHA conversion (ECC stores numeric numbers zero-padded):

```python
alpha_in("759842", 10)    # "0000759842"
alpha_out("0000759842")   # "759842"
```

## How to run

```bash
cd api
../.venv/bin/python -m app.cli.seed_master_data --dry-run              # validate names and lengths
../.venv/bin/python -m app.cli.seed_master_data                        # (re)load all master tables
../.venv/bin/python -m app.cli.seed_master_data --tables KNMT_POAGENT  # one table
../.venv/bin/python -m app.cli.seed_master_data --reset-transactions   # also clear orders + mock sales documents
```

The CLI checks every column name and `NVARCHAR` length against the schema catalogue before
touching HANA (ECC limits apply, e.g. `NAME1` is 35 characters, so long names continue in
`NAME2`), creates missing tables, then empties and reloads each table in one transaction.

## How to test

```bash
cd api
../.venv/bin/python -m unittest discover -s ../tests/unit -t ..   # includes tests/unit/test_matching.py
```

`test_matching.py` uses invented generic fixtures and covers ALPHA conversion, accent folding,
pack parsing, tax-id / e-mail-domain / phone priority over names, missing letter prefixes,
`/XA` narrowing, missing suffixes, pack conflicts, brand prefixes, customer-material codes and
the cap on description-only matches. `test_material_evidence.py` covers the manufacturer part
number, the customer price check and the evidence-based material confidence.

**Acceptance check (needs HANA and the local seed):** every material line of the test orders
must have its expected material among the `search_materials` candidates, because the agent can
only choose what the search returns.

```bash
cd api
../.venv/bin/python -m app.cli.check_material_matching              # misses only + summary, exit 1 on a miss
../.venv/bin/python -m app.cli.check_material_matching --show-all   # rank of every line
```

Input: `data/seed/expected_lines.json` (generated with the seed: German lines parsed from
the PDFs, SG / VN lines listed in the builder; `search_query` = the English retry the agent is
told to use). Result on the current seed: 139/139 lines found among 8 candidates, 137 ranked first. End to end with the agent (all 15 sample documents, 2026-11-06): 139/139 expected materials chosen.
