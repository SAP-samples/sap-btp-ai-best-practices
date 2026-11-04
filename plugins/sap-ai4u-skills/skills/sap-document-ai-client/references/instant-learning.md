# Instant Learning: Ground Truth and Document Confirmation

Read this before implementing corrections, feedback loops, or "teach the model" features on top of SAP Document AI.

## What instant learning is

Instant learning lets SAP Document AI reuse corrected extraction results as examples for later documents. A user
(or an application) corrects the values of an extracted document job, saves them as ground truth, and confirms the
document. Later uploads that use the same schema and have a similar layout are then extracted with those corrections
as context.

Key facts from the SAP documentation:

- In the premium (generative AI) edition, confirmed documents work like few-shot prompt examples. Nothing is retrained.
  The service learns the layout and the corrections and applies them to documents with a similar layout.
- The feedback loop is only active for schema fields with setup type `auto` and without default extractors. Custom
  schemas created with this client's field contract (`setupType: static`, `setup: {type: auto, priority: 1}`,
  `defaultExtractor: {}`) qualify.
- The effect lasts as long as the confirmed document is stored. Deleting the confirmed job (`DELETE /document/jobs`)
  removes its influence. The tenant's document retention time (`documentRetentionTimeDays`) also applies.
- The service may also use confirmed documents to choose the best model per field and as samples for automatically
  created templates. SAP gives no guarantee of improvement.
- Ground truth cannot be saved for jobs extracted with `SAP_OCROnly_schema`.
- After ground truth is saved, the confidence of every header and line-item field is set to `1.0`.

Sources:

- SAP Community, "Instantly improve document information extraction":
  https://community.sap.com/t5/technology-blog-posts-by-sap/instantly-improve-document-information-extraction/ba-p/14028842
- SAP Help, "Save Ground Truth":
  https://help.sap.com/docs/DOCUMENT_INFORMATION_EXTRACTION/5fa7265b9ff64d73bac7cec61ee55ae6/4839cb3130d94ec6a93ee687350164a2.html
- SAP Help, "Add Document and Edit Extraction Results for Instant Learning":
  https://help.sap.com/docs/document-information-extraction/document-information-extraction/add-document-and-edit-extraction-results-for-instant-learning
- Service OpenAPI spec: `GET <url>/document-information-extraction/v1/swagger.json` (bearer token required).

## Endpoints

All paths are relative to `<url>/document-information-extraction/v1`.

| Purpose | Method and path | Body / query | Success |
|---|---|---|---|
| Read configuration | `GET /configuration` or `GET /configuration/<key>` | `scope` = `active` (default), `tenant`, `instance`, `client`; `clientId` for client scope | `200 {"results": {"<key>": {"value": "...", "scope": "..."}}}` |
| Set configuration | `POST /configuration` | `{"value": {"enableInstantLearning": "true"}}`; `scope` = `tenant` (default), `instance`, `client`; `clientId` for client scope | `201 {"inserted": n, "modified": n}` |
| Delete configuration | `DELETE /configuration` | `{"value": ["<key>"]}` (empty list deletes all) | `200 {"deleted": n}` |
| Save ground truth | `POST /document/jobs/<id>` | `{"extraction": {"headerFields": [...], "lineItems": [[...]]}}`, optional `languages`, `country`, `enrichment` | `201 {"status": "DONE", "message": "Ground truth / corrected values uploaded successfully"}` |
| Confirm document | `POST /document/jobs/<id>/confirm` | query `dataForRetraining=true/false` | `200 {"status": "CONFIRMED", "message": "Document confirmed successfully."}` |
| Export document and learning data | `GET /document/jobs/<id>/export` | query `clientId` (required) | `200` |

Configuration values are always strings (`"true"`, not `true`). Valid keys: `dataFeedbackCollection`,
`disableProductSimilarityMatching`, `performPIICheck`, `activateDocumentNotifications`,
`enrichmentConfidenceThreshold`, `documentRetentionTimeDays`, `clientSegregation`, `manualDataActivation`,
`coordinateFormat`, `enableInstantLearning`.

`GET /configuration` with the default `active` scope returns the effective value and the scope it comes from, for
example `{"results": {"enableInstantLearning": {"value": "true", "scope": "tenant"}}}`.

## Ground truth payload

Only the document fields (header fields and line items) and enrichment data are part of the ground truth. Per field
the relevant attributes are:

| Attribute | Required | Notes |
|---|---|---|
| `name` | yes | Schema field name |
| `value` | yes | Normalized value: ISO date `YYYY-MM-DD`, numbers as JSON numbers, currency as ISO code, units as the codes the extraction returns (UN/CEFACT for `purchaseOrder`) |
| `rawValue` | no | Text as printed |
| `page` | no | 1-based page |
| `coordinates` | no | `{"x", "y", "w", "h"}` relative to the page, same format as the extraction |

`lineItems` is a list of lines; each line is a list of field objects. Enrichment ground truth uses
`{"enrichment": {"sender": [{"id": "..."}]}}` (max one entry).

```json
{
  "extraction": {
    "headerFields": [
      {"name": "documentNumber", "value": "PO-1001", "rawValue": "PO-1001", "page": 1,
       "coordinates": {"x": 0.70, "y": 0.08, "w": 0.08, "h": 0.01}},
      {"name": "grossAmount", "value": 200.0}
    ],
    "lineItems": [
      [
        {"name": "description", "value": "Professional Services"},
        {"name": "quantity", "value": 2.0},
        {"name": "netAmount", "value": 200.0}
      ]
    ]
  }
}
```

Rules:

- Send the complete corrected extraction: every header field and every line, not only the changed values. Remove lines
  the extraction invented and add lines it missed.
- Start from the job's own extraction (`GET /document/jobs/<id>`) and only change values. Keep `page` and
  `coordinates` for values that were correct; drop them (or set the correct ones) for values you change, so the
  example does not point at the wrong text.
- Use the same value normalization the service returns. A unit sent as `Litre` while the service returns `LTR`
  teaches an inconsistent example.

## Confirmation

- Confirming is irreversible for that job: no further ground truth can be saved. Ask the user before confirming.
- `dataForRetraining=true` additionally permits SAP to use the document to retrain its models. Instant learning does
  not need it. Default to `false` and only set `true` with the user's explicit consent, because it shares customer
  documents with SAP for training.
- The confirmed job's status becomes `CONFIRMED`; `wait_for_result` already treats it as terminal.

## End-to-end flow

```python
client = SapDoxClient.from_service_key("service_key.json")

# 1. Instant learning must be enabled (tenant, instance or client scope).
config = client.get_configuration("enableInstantLearning")
if config.get("results", {}).get("enableInstantLearning", {}).get("value") != "true":
    client.set_configuration({"enableInstantLearning": "true"}, scope="client", client_id="c_00")

# 2. Extract with a custom schema whose fields use setup type "auto".
job = client.upload_document("order.pdf", client_id="c_00", schema_id=schema_id, document_type="purchaseOrder")
job = client.wait_for_result(job["id"])

# 3. Correct the values and save them as ground truth.
extraction = SapDoxClient.ground_truth_from_job(job)
for field in extraction["headerFields"]:
    if field["name"] == "documentNumber":
        field.update(value="PO-1001", rawValue="PO-1001")
        field.pop("coordinates", None)
client.save_ground_truth(job["id"], extraction)

# 4. Confirm (irreversible). The corrections now apply to similar documents.
client.confirm_document(job["id"], data_for_retraining=False)
```

## Measuring the effect

Instant learning is layout-based, so test it per layout:

1. Pick documents with known extraction errors and known correct values.
2. Split them within each layout: teach (ground truth + confirm) one or more documents, re-extract the others of the
   same layout. A random split across layouts mostly tests layouts that received no example.
3. Re-extract documents from layouts that received no example as a control group. Changes there measure run-to-run
   variance of the extraction, not learning.
4. Optionally re-upload a taught file as a new job (positive control): if even that does not improve, the feedback
   loop is not active (check the configuration and the field setup type).
5. Compare field by field against the same correct values: fixed, broken, unchanged. Do not compare against the
   confirmed job itself; its confidence is forced to `1.0` and its values are the ground truth.
