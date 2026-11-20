# Versioned reference datasets

## Purpose

Reusable engineering and procurement sources are uploaded once in Settings and
published as immutable HANA snapshots. ECN uploads later pin the exact active
snapshot for every available dataset type. An invalid workbook is rejected
before the publication transaction and cannot replace the current source.

## Inputs and outputs

`POST /api/reference-datasets` accepts a multipart `dataset_type`, one type-validated
`source_file`, and a `context_json` (model range, data group, environment,
extraction date, coverage, comparison baseline). No plant is declared: plants
are derived from the conditional rules and the Settings Plant Code Mapping (see
[Rule-based plant assignment](rule-based-plant-assignment.md)). The six
maintained types are BOM, conditional rules, material master, purchase info
records, purchase info prices, and stock.

The response contains the immutable dataset ID, source filename and SHA-256,
scope, row/warning counts, parser version, upload time, and active flag. It does
not expose the workbook BLOB. `GET /api/reference-datasets` returns all versions;
`GET /api/reference-datasets/{dataset_id}` adds at most 20 normalized sample
rows for operator verification.

Settings lists every uploaded version with an X control beside its filename.
After confirmation, `DELETE /api/reference-datasets/{dataset_id}` removes the
publication, normalized rows, and original workbook in one HANA transaction.
The next newest version in the same scope becomes active automatically. A
missing ID returns 404. Deletion returns 409 when an ECN upload pins the
version, an upload directly uses its artifact, or another version names it as
its comparison baseline. Remove the dependent history or version first. The
page keeps the file visible and displays that reason when deletion is blocked.

## Persistence and activation

- `ECN_IMPACT_REFERENCE_DATASET` stores publication metadata.
- `ECN_IMPACT_SOURCE_ARTIFACT` stores original bytes and a compact manifest.
- Source-specific row tables store normalized tabular data.
- `ECN_IMPACT_UPLOAD_DATASET` stores later ECN-to-source snapshot links.
- The greatest successful `PUBLICATION_ORDER` is active for each exact
  `(dataset_type, model_range, data_group, environment)` scope.
  `SCOPE_CODE`/`SCOPE_NAME` remain in the table but are no longer filled.
- Settings presents replacement as choose supported file, describe scope, then
  validate and atomically activate. It shows only safe publication metadata:
  active source in scope, rows, validation-warning count, source date, scope,
  and secondary traceability hash. A failed validation leaves the active
  publication unchanged; successful activation applies to future ECN imports
  in that scope while existing ECNs retain their pinned versions.

## Purchase Info Prices

`PURCHASE_INFO_PRICES` is a versioned CSV price overlay. It preserves its Purchase Info identity (`Info Rec.`, vendor, supplier, material, plant, and purchasing group) plus `Net Price`, currency, price quantity, and order-price unit. It does not replace or delete `PURCHASE_INFO_RECORDS`.

The source parser accepts UTF-8 CSV with the documented columns, normalizes comma-grouped numeric values, preserves zero prices as a `PURCHASE_INFO_PRICE_ZERO` warning, and rejects ambiguous locale formats rather than guessing a price. A price overlay is source evidence only: it does not establish a current/as-of price, supplier approval, availability, or a cost-based acceptance outcome until validity and business rules are explicitly defined.

For a material dialog's **Procurement Permission** table, the backend resolves the active same-scope Price Overlay using the existing assessment's model range, data group, and environment, then matches it to the displayed Material + Plant row. This display-only source-price lookup intentionally does not require reassessment of historic ECNs and does not change their immutable procurement-permission assessment. The browser only renders the returned state; it does not join records or choose a price. A single positive source row renders `currency net price / price per order-price unit`. Missing, zero, and multiple positive source rows render `Not Supplied`, `Zero Source Price — Review`, and `Multiple Source Prices — Review` respectively. The adjacent **Lead Time** field is currently returned as `Not Supplied`; no duration, unit, or delivery commitment is inferred until a source contract is available. A source price is shown only when external procurement is permitted, and it never changes the procurement-permission result. The section is rendered for every material: if its assessment does not contain a source-backed Material Master procurement row, it explicitly states that procurement permission is unavailable rather than inventing a plant, permission, or price.

## Targeted Purchase Info successor preview

For the controlled transition from a historical `PURCHASE_INFO_RECORDS` baseline, run the local preview builder before publishing anything:

```bash
PYTHONPATH=api api/.venv/bin/python scripts/build_purchase_info_successor_preview.py \
  --price-csv 'SAP AI4U - Purchase Info Record3+Price.csv' \
  --output /tmp/purchase-info-successor-preview.xlsx
```

The command is read-only against HANA. It reads the one active Purchase Info Record baseline and derives the exact Material + Plant rows shown in the current **Procurement Permission** view from stored ECN snapshots, without updating the list-page plant cache. It writes a local **not published** workbook. It carries every historical baseline row forward and adds only CSV rows for displayed Material + Plant pairs not already covered by the baseline. Its console summary also classifies displayed price-overlay coverage without selecting a price: one positive record, zero-price review, multiple-source-price review, or not supplied.

The first worksheet uses the established supplier-name Purchase Info layout for normal validation. The `Reconciliation Manifest` worksheet records the source dataset ID, price-file SHA-256, aggregate coverage counts, and per-row origin/source coordinates. The builder refuses to create a workbook if populated legacy fields (`category`, `tax_code`, or `fixed_firm_date`) would be lost by that layout. Review the output and its uncovered-pair count before separately approving publication and activation.

## Make/Buy procurement status

Make/Buy is not a separate ingestion source. It is a direct attribute of the
material master: the `ProcType` column stores the SAP code (`E`, `F`, `X`) for
each material/plant row. The snapshot enriches each material-master reference
record with a `procurement_type_label` derived from the code without inference:

| Code   | Label                                        |
|--------|----------------------------------------------|
| `E`    | In-house production                          |
| `F`    | External procurement                         |
| `X`    | In-house and external procurement allowed    |
| other  | Unknown                                      |

Raw codes are always preserved in storage. Unknown codes remain unknown and are
not inferred from other sources. The historical Make/Buy history workbook and
Make/Buy rules workbook are retired and no longer accepted as input types.

## Related files

- `api/app/ecn/reference_data/`: contracts, parsers, service, and HTTP DTOs.
- `api/app/persistence/reference_dataset_repository.py`: atomic HANA writes and
  metadata-only reads.
- `api/app/routers/reference_datasets.py`: authenticated Settings endpoints.
- `ui/src/pages/settings/settings.js` and `settings.css`: version list, delete
  control, confirmation, and result message.
- `ui/src/services/ecn-api.js`: browser API client for publication deletion.
- `api/app/persistence/table_contracts.py`: automatic table creation contracts.

## Tests

```bash
PYTHONPATH=api .venv/bin/python -m pytest \
  tests/unit/test_reference_parsers.py \
  tests/unit/test_reference_dataset_service.py \
  tests/unit/test_reference_dataset_repository.py \
  tests/unit/test_hana_table_contracts.py \
  tests/unit/test_hana_schema.py -q

cd ui && npm test -- --run tests/settings-page.test.js
```

## Publishing a dataset set from the command line

`scripts/publish_reference_datasets.py` publishes every workbook listed in a manifest through
`POST /api/reference-datasets` (same validation and HANA tables as Settings), then imports selected
ECN numbers from a JSON export through `POST /api/uploads`. It shows a `tqdm` progress bar and reads
the API key from `API_KEY` or `api/.env` (parsed by python-dotenv, never shell-sourced).

```bash
.venv/bin/python scripts/publish_reference_datasets.py --manifest scripts/manifests/q900v-2026-09.json
```

Manifest `scripts/manifests/q900v-2026-09.json` holds the real Q900V extraction:

| Dataset | File | Context |
| --- | --- | --- |
| BOM (enriched EBOM, 130% EBOM, MBOM sheets) | `data/EBOM MBOM_Q900V_04.09.2026.xlsx` | Q900V, PRODUCTION, 2026-09-04, coverage complete |
| Conditional rules | `data/Règlesconditionnelleslocales~...xlsx` | Q900V, 2026-09-04 |
| Material master, purchase info records, stock | `data/new_16-09/SAP AI4U - *.XLSX` (rows carry PL10 / PK01) | 2026-09-16 |
| ECNs | `data/new_16-09/ECN_full.json` | ECN345274, ECN346863 |

No workbook declares a plant. BOM rows get their plants from the rules of their VC group; SAP
master-data rows keep their own plant code and count for a plant only when the rules place the
material there. Enter the Plant Code Mapping in Settings (JB and IS to PL10 Plant A, JD to PK01
Plant B) **before** importing ECNs: it is frozen into each assessment at import.

Everything goes into data group `real-q900v`. When importing ECNs in the UI for this data, type
`real-q900v` in the data-group field.

## Real Q900V acceptance manifest

`scripts/manifests/q900v-2026-09.json` publishes the September BOM for model range Q900V, plus conditional rules, material master, purchase info records and stock, in the `real-q900v` data group. Plants come from the rules and the Plant Code Mapping.

```bash
.venv/bin/python scripts/publish_reference_datasets.py --manifest scripts/manifests/q900v-2026-09.json --assessment-date 2026-09-24
# Reuse published datasets and import only the selected ECNs:
.venv/bin/python scripts/publish_reference_datasets.py --manifest scripts/manifests/q900v-2026-09.json --skip-datasets
```

The publisher uses the authenticated Settings and upload APIs, shows progress, and imports only ECN345274 and ECN346863. It reads credentials with python-dotenv; do not shell-source credential files. Runtime data remains in HANA. Inspect existing publications before repeating the command: a new ECN import creates a new assessment. The September 24 feedback audit reused the already published datasets and assessments.

See [BOM change engine](ecn-bom-changes.md) for interpretation and the real-file acceptance command.
