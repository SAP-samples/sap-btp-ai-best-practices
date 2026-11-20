# HANA data model and schema bootstrap

## What it is

All persistent data of the purchase-order intake app lives in SAP HANA Cloud. The tables
fall into two families:

* **Mock ECC tables** (`<SAP table>_POAGENT`): the customer master, material master and
  sales-document tables of SAP ECC, with SAP-native table and field names. They stand in for
  the real ECC system, which in production is reached through SAP Integration Suite (CPI) and
  the Cloud Connector (see [ecc-integration.md](ecc-integration.md)). The validation and posting
  logic that reads and writes them is real; only the data is mocked.
* **App tables** (`POAGENT_<NAME>`): inbound e-mails, attachments, extracted orders, per-field
  confidence and issue metadata, agent runs, and reviewer corrections.

## What it is for

* The order agent and the deterministic validator read master data (customers, partners,
  sales areas, materials, units) from the mock ECC tables, exactly as they would read it from
  ECC through CPI.
* Posting an order writes `VBAK`/`VBKD`/`VBPA`/`VBAP` rows with a number from
  `VBELN_SEQ_POAGENT`, mirroring what `BAPI_SALESORDER_CREATEFROMDAT2` stores in ECC.
* The UI reads and edits orders through the app tables.

## Inputs and outputs

| Input | Purpose |
|---|---|
| `HANA_ADDRESS`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD` | Connection (required) |
| `HANA_ENCRYPT`, `HANA_SSL_VALIDATE_CERTIFICATE` | TLS flags, default `true` |
| `HANA_SCHEMA` | Optional dedicated schema; created on first connection if missing (needs `CREATE SCHEMA`), then set on every pooled connection |

Output: every table and sequence below exists in the current schema. Start-up never drops or
alters anything; columns missing from an existing table are logged as warnings.

### Schema override (`HANA_SCHEMA`)

HANA resolves unqualified table names against `CURRENT_SCHEMA`, which defaults to the login
user's own schema. `HANA_SCHEMA` decouples the two: the app can log in as one user (for example a
technical admin user) and work in another schema. Empty means the login user's own schema.

Every connection site applies it:

| Connection site | How |
|---|---|
| `api/app/db/engine.py` `get_engine()` | `SET SCHEMA` on every new pooled connection (creates the schema if missing) |
| `api/app/agent_core/memory.py` `_hana_connection()` | `currentSchema` connection property for the agent conversation-memory table |

Required privileges on the target schema: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, plus
`CREATE ANY`, `ALTER` and `DROP` because tables are created on start-up.

Check: if the app connects but shows an empty workspace after a credentials change, the
connection is in the wrong schema. Run `SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY` through
the app and compare with `HANA_SCHEMA`.

## Related files

| File | Role |
|---|---|
| `api/app/db/engine.py` | Process-wide pooled SQLAlchemy engine (`get_engine()`), optional `SET SCHEMA` |
| `api/app/db/schema.py` | Table catalogue (`ECC_TABLES`, `APP_TABLES`, `SEQUENCES`), `ensure_schema()` |
| `api/app/main.py` | `lifespan` runs `ensure_schema` in a worker thread at start-up |
| `tests/unit/test_schema_naming.py` | Naming, primary-key and DDL rendering rules |

## Key code

Tables are declared as data, so one definition produces the DDL and the drift check:

```python
"KNVP_POAGENT": _t(
    "Customer master: partner functions (KNVP); PARVW AG/WE/RE/RG", ("KUNNR", "VKORG", "VTWEG", "SPART", "PARVW", "PARZA"),
    ("KUNNR", f"{KUNNR} NOT NULL"), ("VKORG", f"{VKORG} NOT NULL"), ...
),
```

```python
summary = ensure_schema(get_engine())
# {'created_tables': [...], 'created_sequences': [...], 'added_columns': [...], 'missing_columns': [...]}
```

Missing tables are created. Missing **nullable** columns of existing tables are added with
`ALTER TABLE ... ADD` (so a new optional column such as `MARA_POAGENT.MFRPN` migrates itself in an
existing landscape); missing `NOT NULL` columns are only reported and logged.

## Conventions and deliberate deviations from ECC

* **ALPHA format**: numeric customer (`KUNNR`) and material (`MATNR`) numbers are stored
  zero-padded to 10 / 18 characters, as ECC stores them internally. Conversion happens in one
  helper (`alpha_in` / `alpha_out`); the UI shows the short form.
* **No `MANDT`**: the mock holds a single client, so the client column is omitted.
* **Contact e-mail** (`SMTP_ADDR`) is kept in `KNVK_POAGENT`; in ECC it lives in `ADR6`
  through the contact's address number.
* **Addresses** are kept in `KNA1` fields (`STRAS`, `PSTLZ`, `ORT01`, `LAND1`), which ECC
  also maintains; the central address table `ADRC` is not mocked.
* **Sales code** (customer-facing code with pack size and suffix, e.g. `BASE/20L/XA`) is
  stored in `MARA.BISMT` (old material number), a common place for such codes.
* **Unit aliases** (`ZUOM_ALIAS_POAGENT`) is a custom Z-table mapping free-text units found on
  purchase orders (e.g. "ltrs", "gallon") to SAP units; ECC's `T006A` only holds official texts.
* **Colour code** on items uses `VBAP.ZZCOLOR`, following the customer-append (`ZZ`) convention.
* **Sales-document numbers** come from `VBELN_SEQ_POAGENT` starting at 9100000001, outside real
  number ranges, so mock documents can never be mistaken for real ones.
* **Schedule lines** (`VBEP`) are not mocked; the requested delivery date is kept in `VBAK.VDATU`.

## Tables

### Mock ECC

#### `KNA1_POAGENT`

Customer master: general data (KNA1). Primary key: `KUNNR`.

| Column | Type |
|---|---|
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `KTOKD` | `NVARCHAR(4)` |
| `NAME1` | `NVARCHAR(35) NOT NULL` |
| `NAME2` | `NVARCHAR(35)` |
| `STRAS` | `NVARCHAR(60)` |
| `PSTLZ` | `NVARCHAR(10)` |
| `ORT01` | `NVARCHAR(35)` |
| `REGIO` | `NVARCHAR(3)` |
| `LAND1` | `NVARCHAR(3) NOT NULL` |
| `SPRAS` | `NVARCHAR(1)` |
| `TELF1` | `NVARCHAR(16)` |
| `STCEG` | `NVARCHAR(20)` |
| `AUFSD` | `NVARCHAR(2)` |
| `LOEVM` | `NVARCHAR(1)` |

#### `KNVV_POAGENT`

Customer master: sales area data (KNVV). Primary key: `KUNNR, VKORG, VTWEG, SPART`.

| Column | Type |
|---|---|
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `VKORG` | `NVARCHAR(4) NOT NULL` |
| `VTWEG` | `NVARCHAR(2) NOT NULL` |
| `SPART` | `NVARCHAR(2) NOT NULL` |
| `WAERS` | `NVARCHAR(5)` |
| `ZTERM` | `NVARCHAR(4)` |
| `INCO1` | `NVARCHAR(3)` |
| `INCO2` | `NVARCHAR(28)` |
| `VWERK` | `NVARCHAR(4)` |
| `AUFSD` | `NVARCHAR(2)` |
| `LOEVM` | `NVARCHAR(1)` |

#### `KNVP_POAGENT`

Customer master: partner functions (KNVP); PARVW AG/WE/RE/RG. Primary key: `KUNNR, VKORG, VTWEG, SPART, PARVW, PARZA`.

| Column | Type |
|---|---|
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `VKORG` | `NVARCHAR(4) NOT NULL` |
| `VTWEG` | `NVARCHAR(2) NOT NULL` |
| `SPART` | `NVARCHAR(2) NOT NULL` |
| `PARVW` | `NVARCHAR(2) NOT NULL` |
| `PARZA` | `NVARCHAR(3) NOT NULL` |
| `KUNN2` | `NVARCHAR(10) NOT NULL` |
| `DEFPA` | `NVARCHAR(1)` |

#### `KNVK_POAGENT`

Customer master: contact persons (KNVK, e-mail from ADR6 folded in). Primary key: `PARNR`.

| Column | Type |
|---|---|
| `PARNR` | `NVARCHAR(10) NOT NULL` |
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `NAMEV` | `NVARCHAR(35)` |
| `NAME1` | `NVARCHAR(35)` |
| `TELF1` | `NVARCHAR(16)` |
| `SMTP_ADDR` | `NVARCHAR(241)` |

#### `MARA_POAGENT`

Material master: general data (MARA); BISMT carries the sales code, MFRPN the manufacturer part number of traded goods. Primary key: `MATNR`.

| Column | Type |
|---|---|
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `MTART` | `NVARCHAR(4)` |
| `MATKL` | `NVARCHAR(9)` |
| `SPART` | `NVARCHAR(2)` |
| `MEINS` | `NVARCHAR(3) NOT NULL` |
| `BISMT` | `NVARCHAR(18)` |
| `VOLUM` | `DECIMAL(13,3)` |
| `VOLEH` | `NVARCHAR(3)` |
| `BRGEW` | `DECIMAL(13,3)` |
| `GEWEI` | `NVARCHAR(3)` |
| `MSTAE` | `NVARCHAR(2)` |
| `LVORM` | `NVARCHAR(1)` |
| `MFRPN` | `NVARCHAR(40)` |

#### `MAKT_POAGENT`

Material descriptions per language (MAKT). Primary key: `MATNR, SPRAS`.

| Column | Type |
|---|---|
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `SPRAS` | `NVARCHAR(1) NOT NULL` |
| `MAKTX` | `NVARCHAR(40) NOT NULL` |

#### `MARM_POAGENT`

Alternative units of measure (MARM): 1 MEINH = UMREZ/UMREN base units. Primary key: `MATNR, MEINH`.

| Column | Type |
|---|---|
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `MEINH` | `NVARCHAR(3) NOT NULL` |
| `UMREZ` | `DECIMAL(5,0) NOT NULL` |
| `UMREN` | `DECIMAL(5,0) NOT NULL` |

#### `MVKE_POAGENT`

Material sales data per sales org / channel (MVKE). Primary key: `MATNR, VKORG, VTWEG`.

| Column | Type |
|---|---|
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `VKORG` | `NVARCHAR(4) NOT NULL` |
| `VTWEG` | `NVARCHAR(2) NOT NULL` |
| `VMSTA` | `NVARCHAR(2)` |
| `DWERK` | `NVARCHAR(4)` |
| `VRKME` | `NVARCHAR(3)` |

#### `KNMT_POAGENT`

Customer-material info records (KNMT): customer's own material codes. Primary key: `VKORG, VTWEG, KUNNR, MATNR`.

| Column | Type |
|---|---|
| `VKORG` | `NVARCHAR(4) NOT NULL` |
| `VTWEG` | `NVARCHAR(2) NOT NULL` |
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `KDMAT` | `NVARCHAR(35)` |
| `POSTX` | `NVARCHAR(40)` |
| `WERKS` | `NVARCHAR(4)` |

#### `A305_POAGENT`

Customer-specific material price: condition table A305 (sales org / channel / customer / material)
with its KONP rate folded in; `KBETR` per `KPEIN` `KMEIN` in `KONWA`. Primary key: `VKORG, VTWEG, KUNNR, MATNR`.

| Column | Type |
|---|---|
| `VKORG` | `NVARCHAR(4) NOT NULL` |
| `VTWEG` | `NVARCHAR(2) NOT NULL` |
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `KBETR` | `DECIMAL(15,2) NOT NULL` |
| `KONWA` | `NVARCHAR(5) NOT NULL` |
| `KPEIN` | `DECIMAL(5,0)` |
| `KMEIN` | `NVARCHAR(3)` |

#### `ZUOM_ALIAS_POAGENT`

Custom Z-table: free-text unit aliases (lower-case) to SAP units. Primary key: `ALIAS`.

| Column | Type |
|---|---|
| `ALIAS` | `NVARCHAR(30) NOT NULL` |
| `MSEHI` | `NVARCHAR(3) NOT NULL` |

#### `VBAK_POAGENT`

Sales document header (VBAK). Primary key: `VBELN`.

| Column | Type |
|---|---|
| `VBELN` | `NVARCHAR(10) NOT NULL` |
| `AUART` | `NVARCHAR(4) NOT NULL` |
| `VKORG` | `NVARCHAR(4)` |
| `VTWEG` | `NVARCHAR(2)` |
| `SPART` | `NVARCHAR(2)` |
| `KUNNR` | `NVARCHAR(10) NOT NULL` |
| `BSTNK` | `NVARCHAR(20)` |
| `BSTDK` | `DATE` |
| `BSARK` | `NVARCHAR(4)` |
| `AUDAT` | `DATE` |
| `VDATU` | `DATE` |
| `WAERK` | `NVARCHAR(5)` |
| `NETWR` | `DECIMAL(15,2)` |
| `ERDAT` | `DATE` |
| `ERZET` | `TIME` |
| `ERNAM` | `NVARCHAR(12)` |

#### `VBKD_POAGENT`

Sales document business data (VBKD); POSNR 000000 = header. Primary key: `VBELN, POSNR`.

| Column | Type |
|---|---|
| `VBELN` | `NVARCHAR(10) NOT NULL` |
| `POSNR` | `NVARCHAR(6) NOT NULL` |
| `BSTKD` | `NVARCHAR(35)` |
| `BSTDK` | `DATE` |
| `BSARK` | `NVARCHAR(4)` |
| `ZTERM` | `NVARCHAR(4)` |
| `INCO1` | `NVARCHAR(3)` |
| `INCO2` | `NVARCHAR(28)` |

#### `VBAP_POAGENT`

Sales document items (VBAP); ZZCOLOR follows the customer-append convention. Primary key: `VBELN, POSNR`.

| Column | Type |
|---|---|
| `VBELN` | `NVARCHAR(10) NOT NULL` |
| `POSNR` | `NVARCHAR(6) NOT NULL` |
| `MATNR` | `NVARCHAR(18) NOT NULL` |
| `ARKTX` | `NVARCHAR(40)` |
| `KDMAT` | `NVARCHAR(35)` |
| `KWMENG` | `DECIMAL(15,3) NOT NULL` |
| `VRKME` | `NVARCHAR(3) NOT NULL` |
| `WERKS` | `NVARCHAR(4)` |
| `NETPR` | `DECIMAL(11,2)` |
| `WAERK` | `NVARCHAR(5)` |
| `ZZCOLOR` | `NVARCHAR(20)` |

#### `VBPA_POAGENT`

Sales document partners (VBPA); POSNR 000000 = header. Primary key: `VBELN, POSNR, PARVW`.

| Column | Type |
|---|---|
| `VBELN` | `NVARCHAR(10) NOT NULL` |
| `POSNR` | `NVARCHAR(6) NOT NULL` |
| `PARVW` | `NVARCHAR(2) NOT NULL` |
| `KUNNR` | `NVARCHAR(10) NOT NULL` |

### Application

#### `POAGENT_INBOUND_MESSAGE`

One inbound e-mail or upload batch. Primary key: `MESSAGE_ID`.

| Column | Type |
|---|---|
| `MESSAGE_ID` | `NVARCHAR(36) NOT NULL` |
| `SOURCE_TYPE` | `NVARCHAR(10) NOT NULL` |
| `SENDER` | `NVARCHAR(320)` |
| `RECIPIENTS` | `NVARCHAR(2000)` |
| `SUBJECT` | `NVARCHAR(1000)` |
| `SENT_AT` | `TIMESTAMP` |
| `RECEIVED_AT` | `TIMESTAMP NOT NULL` |
| `BODY_TEXT` | `NCLOB` |
| `SALES_ORG_HINT` | `NVARCHAR(4)` |

#### `POAGENT_ATTACHMENT`

Files of a message (attachments, inline images, forwarded e-mails). Primary key: `ATTACHMENT_ID`.

| Column | Type |
|---|---|
| `ATTACHMENT_ID` | `NVARCHAR(36) NOT NULL` |
| `MESSAGE_ID` | `NVARCHAR(36) NOT NULL` |
| `PARENT_ID` | `NVARCHAR(36)` |
| `SEQ` | `INTEGER NOT NULL` |
| `FILE_NAME` | `NVARCHAR(500) NOT NULL` |
| `MIME_TYPE` | `NVARCHAR(200)` |
| `SIZE_BYTES` | `BIGINT` |
| `ROLE` | `NVARCHAR(20) NOT NULL` |
| `MODEL_INPUT` | `NVARCHAR(10) NOT NULL` |
| `MODEL_INPUT_REASON` | `NVARCHAR(200)` |
| `CONTENT` | `BLOB` |
| `EXTRACTED_TEXT` | `NCLOB` |

#### `POAGENT_ORDER_HEADER`

Extracted / validated purchase order and its processing state. Primary key: `ORDER_ID`.

| Column | Type |
|---|---|
| `ORDER_ID` | `NVARCHAR(36) NOT NULL` |
| `MESSAGE_ID` | `NVARCHAR(36) NOT NULL` |
| `STATUS` | `NVARCHAR(20) NOT NULL` |
| `CLASSIFICATION` | `NVARCHAR(20)` |
| `CLASSIFICATION_REASON` | `NVARCHAR(1000)` |
| `STATUS_MESSAGE` | `NVARCHAR(2000)` |
| `ATTEMPTS` | `INTEGER DEFAULT 0 NOT NULL` |
| `LOCKED_BY` | `NVARCHAR(64)` |
| `LOCKED_AT` | `TIMESTAMP` |
| `REVIEWER_NOTE` | `NVARCHAR(2000)` |
| `PO_NUMBER` | `NVARCHAR(35)` |
| `PO_DATE` | `DATE` |
| `REQ_DELIVERY_DATE` | `DATE` |
| `SOLD_TO` | `NVARCHAR(10)` |
| `SOLD_TO_NAME` | `NVARCHAR(200)` |
| `SOLD_TO_ADDRESS` | `NVARCHAR(500)` |
| `SHIP_TO` | `NVARCHAR(10)` |
| `SHIP_TO_NAME` | `NVARCHAR(200)` |
| `SHIP_TO_ADDRESS` | `NVARCHAR(500)` |
| `SALES_ORG` | `NVARCHAR(4)` |
| `DISTR_CHANNEL` | `NVARCHAR(2)` |
| `DIVISION` | `NVARCHAR(2)` |
| `ORDER_TYPE` | `NVARCHAR(4)` |
| `PO_TYPE` | `NVARCHAR(4)` |
| `CURRENCY` | `NVARCHAR(5)` |
| `NET_AMOUNT` | `DECIMAL(15,2)` |
| `GROSS_AMOUNT` | `DECIMAL(15,2)` |
| `PAYMENT_TERMS_TEXT` | `NVARCHAR(200)` |
| `PAYMENT_TERMS` | `NVARCHAR(4)` |
| `INCOTERMS` | `NVARCHAR(3)` |
| `INCOTERMS_PLACE` | `NVARCHAR(28)` |
| `HEADER_NOTE` | `NVARCHAR(2000)` |
| `AGENT_SUMMARY` | `NCLOB` |
| `SALES_DOC` | `NVARCHAR(10)` |
| `POSTING_PAYLOAD` | `NCLOB` |
| `POSTED_AT` | `TIMESTAMP` |
| `CREATED_AT` | `TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL` |
| `UPDATED_AT` | `TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL` |

#### `POAGENT_ORDER_ITEM`

Order lines (MATERIAL / CHARGE / TEXT); ITEM_NO 1, 2, ... in document order. Primary key: `ORDER_ID, ITEM_NO`.

| Column | Type |
|---|---|
| `ORDER_ID` | `NVARCHAR(36) NOT NULL` |
| `ITEM_NO` | `INTEGER NOT NULL` |
| `LINE_TYPE` | `NVARCHAR(10) NOT NULL` |
| `CUSTOMER_ITEM_NO` | `NVARCHAR(20)` |
| `MATERIAL` | `NVARCHAR(18)` |
| `CUSTOMER_MATERIAL` | `NVARCHAR(35)` |
| `DESCRIPTION` | `NVARCHAR(200)` |
| `QUANTITY` | `DECIMAL(15,3)` |
| `UOM` | `NVARCHAR(3)` |
| `UNIT_PRICE` | `DECIMAL(15,4)` |
| `LINE_AMOUNT` | `DECIMAL(15,2)` |
| `COLOUR_CODE` | `NVARCHAR(20)` |
| `PLANT` | `NVARCHAR(4)` |

#### `POAGENT_FIELD_STATE`

Per-field metadata: raw value, confidences, fixes, issues (ITEM_NO 0 = header). Primary key: `ORDER_ID, ITEM_NO, FIELD_NAME`.

| Column | Type |
|---|---|
| `ORDER_ID` | `NVARCHAR(36) NOT NULL` |
| `ITEM_NO` | `INTEGER NOT NULL` |
| `FIELD_NAME` | `NVARCHAR(40) NOT NULL` |
| `RAW_VALUE` | `NVARCHAR(2000)` |
| `SOURCE_REF` | `NVARCHAR(200)` |
| `EXTRACT_CONFIDENCE` | `DECIMAL(4,3)` |
| `MATCH_CONFIDENCE` | `DECIMAL(4,3)` |
| `CONFIDENCE` | `DECIMAL(4,3)` |
| `CHANGED_BY` | `NVARCHAR(10) NOT NULL` |
| `FIX_NOTE` | `NVARCHAR(1000)` |
| `ISSUE_SEVERITY` | `NVARCHAR(10)` |
| `ISSUE_CODE` | `NVARCHAR(40)` |
| `ISSUE_TEXT` | `NVARCHAR(2000)` |
| `UPDATED_AT` | `TIMESTAMP` |
| `EVIDENCE` | `NVARCHAR(2000)` |

#### `POAGENT_CORRECTION_LOG`

Reviewer corrections (long-term agent memory). Primary key: `CORRECTION_ID`.

| Column | Type |
|---|---|
| `CORRECTION_ID` | `NVARCHAR(36) NOT NULL` |
| `ORDER_ID` | `NVARCHAR(36) NOT NULL` |
| `ITEM_NO` | `INTEGER NOT NULL` |
| `FIELD_NAME` | `NVARCHAR(40) NOT NULL` |
| `SOLD_TO` | `NVARCHAR(10)` |
| `RAW_VALUE` | `NVARCHAR(2000)` |
| `AGENT_VALUE` | `NVARCHAR(500)` |
| `USER_VALUE` | `NVARCHAR(500)` |
| `CREATED_AT` | `TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL` |

#### `POAGENT_AGENT_RUN`

One agent execution per processing attempt, with token usage and trace. Primary key: `RUN_ID`.

| Column | Type |
|---|---|
| `RUN_ID` | `NVARCHAR(36) NOT NULL` |
| `ORDER_ID` | `NVARCHAR(36) NOT NULL` |
| `STARTED_AT` | `TIMESTAMP NOT NULL` |
| `FINISHED_AT` | `TIMESTAMP` |
| `OUTCOME` | `NVARCHAR(20)` |
| `MODEL` | `NVARCHAR(60)` |
| `LLM_CALLS` | `INTEGER` |
| `INPUT_TOKENS` | `BIGINT` |
| `CACHED_INPUT_TOKENS` | `BIGINT` |
| `OUTPUT_TOKENS` | `BIGINT` |
| `ERROR_TEXT` | `NVARCHAR(2000)` |
| `REVIEWER_NOTE` | `NVARCHAR(2000)` |
| `TRACE` | `NCLOB` |

### Sequences

* `VBELN_SEQ_POAGENT`: `START WITH 9100000001 INCREMENT BY 1 NO CYCLE`

## How to test

```bash
cd api
../.venv/bin/python -m unittest discover -s ../tests/unit -t ..
```

Live check (needs `api/.env` with HANA credentials): start the API twice; the first start logs
`HANA schema check: 21 tables created, 1 sequences created`, the second `0 tables created`.
`GET /api/health` returns `"status": "healthy"` when the bootstrap succeeded and `"degraded"`
otherwise.
