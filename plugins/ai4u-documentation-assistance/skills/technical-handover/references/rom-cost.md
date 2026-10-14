# ROM cost calculation

The ROM (rough order of magnitude) cost section sits in Part VI after *Target architecture*. Its message is that most of the platform cost comes from shared services the customer may already own; only the workload-driven consumption is truly incremental. Keep it short.

### Input
The user provides a PDF cost estimate exported from the SAP Discovery Center. Each service block gives: service name, service plan, region/provider, one or more metrics with quantity, unit price per month and cost. Read every metric row; some services (e.g. SAP Document AI Premium Edition) list several metrics in one block, and each becomes its own table row.

Watch for these quirks of the export:
- **Truncated totals.** Headers and some line costs are shortened with a glyph that looks like `l` (`EUR 2,30l`, `EUR 1,02l` meaning thousands). Never copy them: take each line's full cost (quantity × unit price, cross-checked against the "Cost" row) and recompute the total by summing the table rows.
- **Number format.** The export uses European notation (`1.020,00`). Write amounts in the document's language convention (English: `1,020.00`), always with two decimals.
- **Old platform name.** The export still says "SAP BTP, Cloud Foundry Runtime" and "SAP BTPEA". Write "Cloud Foundry Runtime" and don't mention the commercial model; the shorter service name keeps the table consistent with the rest of the document (house rule 7 allows the old prefix only inside service names).

### Sizing the AI Core line from volume
The estimate usually carries SAP AI Core at a placeholder quantity. Size it from the expected workload instead; ask the user for both inputs in the interview:
- **Capacity units (CU) per transaction**: what one business transaction (e.g. one request or one analysed document) consumes.
- **Expected monthly volume** of those transactions in production.

Then:
1. Capacity units = CU per transaction × monthly volume (e.g. 0.02 × 5,000 = 100).
2. Cost = capacity units × unit price (100 × 1.04 = 104.00).
3. Recompute the total from all table rows.

In the document, the note states only the volume: "based on an estimated volume of N <transactions> per month; the final value can differ depending on actual usage". Leave the per-transaction arithmetic out. The matching open question states the same N. If either input is missing, keep the estimate's quantity, say it is a minimum quantity, and ask for the volume in the open question.

**Keep the estimate PDF in step.** Once the AI Core quantity differs from the exported estimate, a shared PDF no longer matches the handover. Remind the user in the delivery message to regenerate the estimate with the same quantities whenever the volume changes, if the PDF is shared with the customer.

### What the section must not contain
- The name of the tool that produced the estimate (SAP Discovery Center) or the date it was generated.
- Scenario or conclusion blocks ("what this means for the customer", greenfield vs. existing-entitlement totals, annual figures, percentages).
- Unit names that reveal user roles beyond the export's metric name (write "Users", not "Admin users").

### Classifying each line
| Service / metric | Cost type | Why |
|---|---|---|
| SAP AI Core, Capacity units | Solution-specific | Capacity units act as credits for the solution's own Generative AI Hub token consumption |
| SAP AI Launchpad, Tenants | Shared | One tenant manages all AI Core usage in the account |
| SAP HANA Cloud, Capacity units | Shared | The solution can use a schema/HDI container in an existing instance |
| SAP Document AI, Users | Shared | Account-level user licences |
| SAP Document AI, Requests | Solution-specific | Driven by the solution's API calls |
| SAP Document AI, Blocks of 100 documents | Solution-specific | Driven by the solution's document volume |
| Cloud Foundry Runtime, GB memory | Shared, if spare memory quota exists | Reusable only when the customer's quota has room for the apps |
| SAP Cloud Logging, Capacity units | Shared | An existing instance can ingest the solution's logs |
| SAP Credential Store, Records | Shared | An existing instance can hold the solution's namespace |

For a service not listed here: account-level licences, tenants and instances the solution only occupies a part of are **Shared**; anything that scales with the solution's own calls, documents or tokens is **Solution-specific**. If unsure, classify it and add an open question.

### Template
Adapt the second paragraph only if the solution-specific services differ (name those actually present). Keep the table columns exactly as below, one row per metric, in the order of the estimate, and end with a bold total row.

```markdown
# ROM cost calculation

This section gives a rough order of magnitude (ROM) for the monthly platform cost of running the solution with the services used in the PoC. The figures are list prices for the Europe (Frankfurt) region, before any customer-specific discount, and they are not a quote.

Most of this cost comes from **shared platform services**. The customer may already hold entitlements for these in its SAP Business AI Platform global account. If so, the solution reuses existing capacity instead of adding new cost. Only the consumption driven by the solution's own workload is truly incremental: AI Core capacity units for model token usage, and Document AI requests and document blocks.

Table: ROM cost breakdown (EUR, list price per month) {#tbl-rom-cost}

| Service | Plan | Metric | Qty | Unit price | Cost | Cost type |
|---|---|---|---:|---:|---:|---|
| SAP AI Core | Standard | Capacity units | 100 | 1.04 | 104.00 | Solution-specific (token consumption) |
| SAP AI Launchpad | Standard | Tenants | 1 | 583.00 | 583.00 | Shared |
| SAP HANA Cloud | SAP HANA Cloud | Capacity units | 885 | 0.75 | 663.75 | Shared |
| SAP Document AI | Premium Edition | Users | 1 | 700.00 | 700.00 | Shared |
| SAP Document AI | Premium Edition | Requests | 1 | 0.55 | 0.55 | Solution-specific |
| SAP Document AI | Premium Edition | Blocks of 100 documents | 1 | 290.00 | 290.00 | Solution-specific |
| Cloud Foundry Runtime | Standard | GB memory | 12 | 85.00 | 1,020.00 | Shared, if spare memory quota exists |
| SAP Cloud Logging | Overall | Capacity units | 1 | 1.00 | 1.00 | Shared |
| SAP Credential Store | standard | Records | 1 | 10.00 | 10.00 | Shared |
| **Total** | | | | | **3,372.30** | |

> [!NOTE] Consumption-based lines
> AI Core capacity units work as credits for Generative AI Hub token consumption; they are based on an estimated volume of 5,000 requests per month, and the final value can differ depending on actual usage. Document AI requests and document blocks are entered at minimum quantities and grow with the number of documents processed; size them from the expected production volumes before budgeting (Q-05, see @tbl-questions in Open questions and decisions).
```

Use the region from the estimate in the first paragraph. Add the matching open question (Q-xx) to `@tbl-questions` and point the note to it. It states the same volume as the note, e.g. "Is 5,000 requests per month the right production volume for AI Core, and what document volume should Document AI be sized for?" Write amounts only as "per month" in the caption; never "N months" or annualised figures (the checker's time-estimate rule scans Part VI).

### No estimate provided
Keep the heading and write one sentence: the platform cost has not been estimated yet and should be produced for the services in the target architecture (@fig-arch-target or the equivalent label), distinguishing shared services from solution-specific consumption. Add an open question for it.
