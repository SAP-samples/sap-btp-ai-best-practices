# Access To SAP RPT Model

![Access To SAP RPT Model](https://media.akamai.odsp.cdn.office.net/westeurope1-mediap.svc.ms/transform/thumbnail?provider=url&inputFormat=jpg&docid=https%3A%2F%2Fcdn.hubblecontent.osi.office.net%2Fm365content%2Fpublish%2F235d262e-6785-4a03-9953-65b69541c1ab%2F901627692.jpg&w=1600&lqip=B46R%7BIDN00p3Mtbf)

Predictive AI

## Steps
1. [Overview](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx#1.-overview)
2. [Pre-requisites](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx#2.-pre-requisites)
3. [Key Choices and Guidelines](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx#3.-key-choices-and-guidelines)
4. [Implementation](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx#4.-implementation)
[Webinar (PDF Presentation)](https://sap.sharepoint.com/sites/210313/Shared%20Documents/Best%20Practices/BTP%20AI%20Best%20Practices%20-%20Access%20to%20SAP%20RPT-1%20model.pdf?web=1)

## 1. Overview

### Description

[SAP-RPT-1](https://www.sap.com/products/artificial-intelligence/sap-rpt.html) (Relational Pretrained Transformer) is a table-native foundation model designed specifically for tabular and relational data. It performs **classification** and **regression** tasks without any training or fine-tuning through **in-context learning**. Unlike traditional ML approaches, RPT-1 learns patterns from example rows provided directly in the API request and predicts values for query rows.

### Expected Outcome

Provide instant, high-quality predictions on structured business data without the need for training infrastructure, datasets, or model fine-tuning cycles. Applications can leverage RPT-1 to automate classification and regression tasks on enterprise tabular data.

### Benefits

- **Zero Training Required:** No need to collect training datasets, manage compute infrastructure, or wait for model training cycles. Deploy immediately with production-ready accuracy.
- **In-Context Learning:** Accepts representative context examples during inference to return instant predictions without setup or customization for specific use cases.
- **Fast Adaptation:** Adapts to the context data provided in requests without additional training steps or deployments.
- **Enterprise-Grade Quality:** Table-native architecture delivers prediction quality ahead of state-of-the-art narrow AI models and LLMs for tabular tasks.
- **Low maintenance cost:** No need to update, retrain and maintain several versions of the prediction models.

## 2. Pre-requisites

### Commercial

- SAP AI Core with the “Extended” tier on SAP BTP
- SAP AI Launchpad (Optional but recommended)

You can find pricing details for AI Core “Extended” in the [Discovery Center – AI Core](https://discovery-center.cloud.sap/serviceCatalog/sap-ai-core?tab=service_plan&region=all&service_plan=extended&commercialModel=btpea).

### Technical

- Setup SAP Business Technology Platform (SAP BTP) subaccount ([Setup Guide](https://btp-ai-bp.docs.sap/docs/technology/sap-business-technology-platform#setup-guide))
- Create an instance of SAP AI Core ([Setup Guide](https://btp-ai-bp.docs.sap/docs/technology/sap-ai-core#setup-guide))
- Subscribe to SAP AI Launchpad ([Setup Guide](https://btp-ai-bp.docs.sap/docs/technology/sap-ai-launchpad))

### High-level Reference Architecture​

_[Image not exported from SharePoint (blob URL)]_

#### SAP Business Technology Platform (SAP BTP)

[SAP Business Technology Platform (BTP)](https://btp-ai-bp.docs.sap/docs/technology/sap-business-technology-platform) is an integrated suite of cloud services, databases, AI, and development tools that enable businesses to build, extend, and integrate SAP and non-SAP applications efficiently.

#### SAP AI Core

[SAP AI Core](https://btp-ai-bp.docs.sap/docs/technology/sap-ai-core) is a managed AI runtime that enables scalable execution of AI models and pipelines, integrating seamlessly with SAP applications and data on SAP BTP. RPT-1 is deployed through the `foundation-models` scenario using the `aicore-sap` executable ID.

## 3. Key Choices and Guidelines

### Model Version Selection

Two versions of SAP-RPT-1 are available, each optimized for different scenarios:

| Specification | sap-rpt-1-small | spa-rpt-1-large |
| --- | --- | --- |
| Max Context Length | 2048 rows | 65536 rows |
| Max Columns | 100 columns | 256 columns |
| Target Calsses (Classification) | 256* | 1024* |
| Recommended Context Length | 500-2000 rows | 4000-8000 rows |

*The number of target classes is a recommendation for best prediction quality, not a hard limit

**Both versions share**:

- Max 128 simultaneous prediction rows per request
- Max 10 simultaneous prediction columns per request
- supported task types: `classification` and `regression`

#### When to use which version?

**sap-rpt-1-small** – Use when:

- The task fits within 2048 context rows and 100 columns
- Prediction scenarios are of medium complexity
- Low latency and high-prediction throughput are the primary objective

**sap-rpt-1-large** – Use when:

- The task benefits from larger context windows, for example, use cases that deal with transactional data or regression tasks that improve with thousands of context rows
- Prediction scenarios are complex
- Best prediction quality and lowest error rates are the primary objective

Benchmarks show that when both models receive the same context budget and configuration, their prediction quality is effectively identical. The large model's advantage comes from its ability to consume far more context rows. Choose the model based on the context capacity your task requires

### Tabular In-Context Learning

RPT-1 produces predictions through tabular in-context learning. Prompts are given in table form containing:

1. Context rows: Example data with known target values (the model learns patterns from these)
2. Query rows: Rows where the target column contains a placeholder (
\[PREDICT\]
)

**Example prompt (as a table):**

| PRODUCT | PRICE | ORDERDATE | ID | COSTCENTER |
| --- | --- | --- | --- | --- |
| Couch | 999,99 | 28-11-2025 | ID | \[PREDICT\] |
| Office Chair | 150,80 | 02-11-2025 | 44 | Office Furniture |
| Server Rack | 2200,00 | 01-11-2025 | 104 | Data Infrastructure |

The model analyzes context rows, learns the relationships, and predicts values for rows containing `[PREDICT]`.

### Context Budget: The Primary Quality Lever

The number of context rows provided to RPT-1 is the single most impactful factor for prediction quality.

- **Classification tasks**: For binary or low-cardinality targets, a moderate context budget often suffice to reach peak accuracy. In many cases, 256-512 well selected context rows are enough, and additional rows usually provide diminishing returns. The optimal budget depends more on class separability, data quality and representativeness than on the number of classes alone
- **Regression tasks**: Benefit substantially from larger context budgets. Continuous targets improve steadily as context grows from hundreds to thousands of rows, because more examples improve interpolation across complex feature interactions

#### Practical Guidance

| Task Type | Starting Budget | When to Increase |
| --- | --- | --- |
| Binary / low-cardinality classification | 256–512 rows | Only if accuracy is below expectations |
| Multi-class classification (10+ classes) | 500–1,000 rows | When balanced accuracy is insufficient |
| Regression | 1,000–2,000 rows (small) or 4,000–8,000 rows (large) | Whenever more context rows are available and quality has not plateaued |

If your regression task has access to more than 2,048 good context rows, prefer `sap-rpt-1-large` to take full advantage of the extended context window.

### Context Selection Strategy

How you select context rows matters. Not all selection strategies are equally effective:

| Strategy | When to Use | Notes |
| --- | --- | --- |
| Random | Default starting point for any task | Surprisingly effective on well-structured tabular data. Provides broad feature-space coverage. |
| Diversity | Regression tasks with continuous features | Covers the feature distribution and avoids redundant examples. Best overall strategy for regression in benchmarks. |
| Stratified | Classification with class imbalance | Ensures all target classes are represented proportionally. Comparable to random on balanced datasets. |
| Minority | Classification with strong class imbalance | Oversamples underrepresented classes. Falls back to diversity for regression. |

**Recommendation:** Start with `random` sampling as your baseline. For regression tasks, switch to `diversity` sampling. Later versions of RPT will come with context retrieval algorithms

### Query Batch Size

Batch size (the number of query rows per request) has minimal impact on prediction quality but significantly affects operational throughput:

| Batch Size | Use Case |
| --- | --- |
| 16 | Debugging, small datasets, per-row analysis |
| 64 | Safe default for most production scenarios, especially with wider schemas |
| 128 | Throughput-optimized for production sweeps when payload size allows |

For immediate predictions, such as live user interaction or very time-critical transactions, consider using smaller batch sizes.

### Data Quality Best Practices

To achieve the best possible prediction quality:

- Use clean data with no or minimum missing values
- Use descriptive column names (under 100 characters)
- Limit descriptions in text columns to 500 characters
- Keep data types consistent within each column
- Remove columns that are not relevant to the prediction task (reduces noise and payload size)
- Provide `data_schema` and `task_type` parameters for best results
- **Remove direct leakage columns:** any column that trivially encodes the target (e.g., a derived flag or lookup code) inflates apparent quality and does not reflect real-world inference conditions
- Encode potential rules as features
- Prefer meaningful date-derived features over raw date values. For example, use features such as `days_since_last_purchase`, `months_since_contract_start`, `days_until_due_date`, or `days_between_order_and_delivery`, depending on the prediction task. Ensure these features only use information that is available at prediction time to avoid target leakage.
- **Preserve categorical semantics:** if numeric-looking values are actually categories, such as company codes, product IDs, region codes, or status codes, provide them as strings or define them explicitly as categorical fields in the schema so they are not interpreted as continuous numbers

**Note:** The accuracy and fairness of predictions depend on the quality and balance of the data provided. Any bias present in the input data can result in biased or skewed outputs.

### Comparison with Classical ML Approaches

RPT-1 operates in a fundamentally different mode than traditional ML:

| Aspect | Classical ML (e.g., LightGBM, Random Forest) | RPT-1 |
| --- | --- | --- |
| Training | Requires explicit training step | No training; learns from context at inference time |
| Data efficiency | Needs large training sets for best results | Achieves competitive quality with hundreds of context rows |
| Deployment | Train, export, deploy model artifacts | Single API call with context + query rows |
| Adaptation | Retrain for new data distributions | Update context rows in the request |
| Maintenance | Model drift requires retraining cycles | Context rows are always fresh |

RPT-1 is particularly well-suited when:

- Training infrastructure or ML pipeline maintenance is not feasible
- Rapid adaptation to new data distributions is required
- The use case involves moderate-sized datasets (hundreds to low-thousands of rows)
- Multiple prediction tasks need to be served without managing individual model artifacts

## 4. Implementation

Below you can find the necessary information to develop a project in your preferred language using the official SAP SDKs. For application code, prefer the SDKs over manually calling OAuth and `/predict` endpoints. The SDKs handle authentication, SAP AI Core integration, deployment resolution, request serialization, and response parsing.

### Programming Model Selection Guidelines

When developing RPT-1 applications within the SAP ecosystem, consider the following recommendations:

- **Python data science and AI prototypes:** Use the SAP Cloud SDK for AI Python package when you need notebooks, Pandas workflows, rapid experimentation, or backend services in Python.
- **JavaScript/TypeScript services:** Use the SAP AI SDK for JavaScript/TypeScript when building Node.js services, full-stack applications, or CAP-adjacent services that already use the JavaScript ecosystem.
- **Java enterprise applications:** Use the SAP AI SDK for Java when building Spring Boot services or Java enterprise applications that need typed SAP AI SDK integration.

Regardless of the language, the implementation pattern is the same:

1. Configure SAP AI Core credentials using the SDK-supported environment or destination mechanism.
2. Ensure that an RPT-1 deployment exists for `sap-rpt-1-small` or `sap-rpt-1-large`.
3. Build one prediction request containing context rows with known target values and query rows where the target value is marked with a prediction placeholder.
4. Declare target columns with the correct task type: `classification` or `regression`.
5. Provide an `index_column` so predictions can be mapped back to source records.
6. Provide an explicit schema for numeric or date-heavy tables, or when automatic type inference could be ambiguous.

### Common RPT-1 Request Concepts

| Concept | Description |
| --- | --- |
| Context rows | Example rows with known target values. RPT-1 learns from these rows during inference. |
| Query rows | Rows to score. Their target cells contain the prediction placeholder. |
| Prediction placeholder | Value used to mark cells that should be predicted, for example \[PREDICT\] or a configured numeric placeholder. |
| Target column | Column to predict, configured with task type classification or regression. |
| Index column | Stable row identifier returned with predictions so application code can map results back to source data. |
| Data schema | Optional but recommended column type declaration using string, numeric, or date. |

## Python

### Recommendation

Starting on version 6.5.0, [SAP Cloud SDK for AI Python](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/gen_ai_hub.html#sap-rpt-1-models) package (`sap-ai-sdk-gen`) has native compatibility with the SAP RPT-1 model. The client exposes it through `gen_ai_hub.proxy.native.sap`. This is the recommended path for Python applications because `RPTClient` handles authentication, deployment lookup, prediction calls, and response parsing.

### SDKs

The Python SDK exposes the main RPT-1 request and response models:

- RPTClient
- RPTRequest
- PredictionConfig
- TargetColumn

Use `client.predict(...)` for synchronous workflows and `await client.apredict(...)` for asynchronous workflows. Prefer model-based resolution with `model_name="sap-rpt-1-small"` or `model_name="sap-rpt-1-large"` and, when needed, `model_version="1"` or `model_version="latest"`.

```python
from gen_ai_hub.proxy.native.sap import (
    PredictionConfig,
    RPTClient,
    RPTRequest,
    TargetColumn,
)

client = RPTClient()

body = RPTRequest(
    prediction_config=PredictionConfig(
        target_columns=[
            TargetColumn(name="COSTCENTER", task_type="classification")
        ]
    ),
    rows=[
        {"PRODUCT": "Couch", "PRICE": 999.99, "ID": "35", "COSTCENTER": "[PREDICT]"},
        {"PRODUCT": "Office Chair", "PRICE": 150.80, "ID": "44", "COSTCENTER": "Office Furniture"},
        {"PRODUCT": "Server Rack", "PRICE": 2200.00, "ID": "104", "COSTCENTER": "Data Infrastructure"},
    ],
    index_column="ID",
)

response = client.predict(body=body, model_name="sap-rpt-1-small", model_version="latest")
print(response.predictions)
```

### Configuration

Use the standard SAP AI SDK / Generative AI Hub environment configuration:

| Variable | Description |
| --- | --- |
| AICORE_AUTH_URL | OAuth2 token endpoint for the SAP AI Core instance. |
| AICORE_CLIENT_ID | OAuth2 client ID from the service key. |
| AICORE_CLIENT_SECRET | OAuth2 client secret from the service key. |
| AICORE_BASE_URL | SAP AI Core API base URL. |
| AICORE_RESOURCE_GROUP | SAP AI Core resource group, often default. |

### Tutorials and Learning Journeys

- [SAP Cloud SDK for AI - SAP RPT-1 Models](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_reference/gen_ai_hub.html#sap-rpt-1-models)
- [SAP Help: SAP-RPT-1](https://help.sap.com/docs/sap-ai-core/generative-ai/sap-rpt-1?locale=en-US)
- [Example Payloads for Inferencing: sap-rpt-1](https://help.sap.com/docs/sap-ai-core/generative-ai/example-payloads-for-inferencing-sap-rpt-1)
- [SAP Learning: Introduction to SAP-RPT-1](https://learning.sap.com/courses/genai-lpad-gnd/bt-sap-rpt-1)
- [SAP's AI Golden Path: Predictive & Tabular AI](https://architecture.learning.sap.com/docs/golden-path/ai-golden-path/build-and-deliver/predictive-tabular-ai)

### Reference Code

#### Recommended

- [SAP BTP AI Best Practices - RPT-1 Python](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/narrow-ai/RPT-1/python)

**Other Sources**

- [SAP-samples Python examples](https://github.com/SAP-samples/sap-rpt-samples/tree/main/sap-rpt-1/code_samples/python): The official sample repo includes `predict.py` and `predict_parquet.py`. The Python example shows OAuth token retrieval, the SAP AI Core endpoint, `AI-Resource-Group` header usage, request payload construction, and optional GZIP compression

---

## JavaScript/TypeScript

### Recommendation

Starting on version 2.6.0, the [SAP AI SDK for JavaScript/TypeScript](https://sap.github.io/ai-sdk/docs/js/rpt) package `@sap-ai-sdk/rpt` for Node.js and TypeScript applications is compatible with RPT-1 model. This is the recommended path for TypeScript services that need a typed SDK client for RPT-1 inference.

### SDKs

The TypeScript SDK exposes `RptClient` and prediction methods for schema-free and schema-based requests:

- predictWithoutSchema(...)
for straightforward tabular requests where automatic type inference is sufficient.
- predictWithSchema(...)
for requests where the application provides an explicit schema.
- predictParquet(...)
for Parquet-based tabular prediction workflows.

```ts
import { RptClient } from "@sap-ai-sdk/rpt";

const client = new RptClient({ deploymentId: process.env.RPT1_DEPLOYMENT_ID });

const prediction = await client.predictWithoutSchema({
  prediction_config: {
    target_columns: [
      {
        name: "COSTCENTER",
        prediction_placeholder: "[PREDICT]",
        task_type: "classification",
      },
    ],
  },
  index_column: "ID",
  rows: [
    { PRODUCT: "Couch", PRICE: 999.99, ID: "35", COSTCENTER: "[PREDICT]" },
    { PRODUCT: "Office Chair", PRICE: 150.8, ID: "44", COSTCENTER: "Office Furniture" },
    { PRODUCT: "Server Rack", PRICE: 2200.0, ID: "104", COSTCENTER: "Data Infrastructure" },
  ],
});

console.log(prediction);
```

### Configuration

For local development, the sample code uses:

| Variable | Description |
| --- | --- |
| AICORE_SERVICE_KEY | JSON service key for the SAP AI Core instance. |
| RPT1_DEPLOYMENT_ID | Deployment ID of the RPT-1 model on SAP AI Core. |

### Tutorials and Learning Journeys

- [SAP Cloud SDK for AI JavaScript](https://sap.github.io/ai-sdk/docs/js/rpt)
- **SAP Help:** SAP-RPT-1
- [Example Payloads for Inferencing: sap-rpt-1](https://help.sap.com/docs/sap-ai-core/generative-ai/example-payloads-for-inferencing-sap-rpt-1)
- [SAP Learning: Introduction to SAP-RPT-1](https://learning.sap.com/courses/genai-lpad-gnd/bt-sap-rpt-1)
- [SAP's AI Golden Path: Predictive & Tabular AI](https://architecture.learning.sap.com/docs/golden-path/ai-golden-path/build-and-deliver/predictive-tabular-ai)

### Reference Code

#### Recommended

- [SAP BTP AI Best Practices - RPT-1 TypeScript](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/narrow-ai/RPT-1/typescript)

**SAP Samples**

- [SAP CodeJam Sample: Code-based Agents with Generative AI Hub](https://github.com/SAP-samples/codejam-code-based-agents): This official SAP-samples repo includes Python and JavaScript exercises and explicitly includes SAP-RPT-1 as part of the training material
- [SAP CodeJam JS exercise: Add your first tool to the agent](https://github.com/SAP-samples/codejam-code-based-agents/blob/main/exercises/JavaScript/03-add-your-first-tool.md): It shows an agent calling SAP-RPT-1 directly, using `@sap-ai-sdk/rpt`, and returning predictions plus confidence scores
- [SAP-samples reference implementation](https://github.com/SAP-samples/btp-joule-a2a-pro-code-agent): Sales inquiry optimization with SAP-RPT-1: It uses SAP-RPT-1 for forecasting/simulation, with SAP BTP, Cloud Foundry, generative AI hub, CAP, LangGraph JS, and SAP Cloud SDK for AI

---

## Java

### Recommendation

Starting on version 1.16.0, the [SAP AI SDK for Java](https://sap.github.io/ai-sdk/docs/java/foundation-models/sap-rpt/table-completion) is compatible with SAP RPT-1 models. The Java SDK provides an RPT client for table completion against deployed SAP RPT models in SAP AI Core.

### SDKs

You will need to first add the SAP RPT module to your Java project:

```xml
<dependency>
    <groupId>com.sap.ai.sdk.foundationmodels</groupId>
    <artifactId>sap-rpt</artifactId>
    <version>${ai-sdk.version}</version>
</dependency>
```

The Java SDK uses `RptClient` with generated request and response model classes. Use `RptClient.forModel(...)` to resolve the model from SAP AI Core configuration and call `tableCompletion(...)` for RPT-1 prediction.

```java
var client = RptClient.forModel(config.model());
var input = PredictRequestPayload.create()
    .predictionConfig(PredictionConfig.create().targetColumns(targetColumns))
    .indexColumn("ID")
    .rows(rows);

var response = client.tableCompletion(input);
System.out.println(response.getPredictions());
```

### Configuration

For local development, the sample code uses:

| Variable | Description |
| --- | --- |
| AICORE_SERVICE_KEY | JSON service key for the SAP AI Core instance. |
| RPT1_MODEL_NAME | RPT-1 model name, for example sap-rpt-1-small. |
| RPT1_MODEL_VERSION | Optional RPT-1 model version. |

### Tutorials and Learning Journeys

- [SAP Cloud SDK for AI Java: SAP RPT table completion](https://sap.github.io/ai-sdk/docs/java/foundation-models/sap-rpt/table-completion)
- **SAP Help:** SAP-RPT-1
- [Example Payloads for Inferencing: sap-rpt-1](https://help.sap.com/docs/sap-ai-core/generative-ai/example-payloads-for-inferencing-sap-rpt-1)
- [SAP Learning: Introduction to SAP-RPT-1](https://learning.sap.com/courses/genai-lpad-gnd/bt-sap-rpt-1)
- [SAP's AI Golden Path: Predictive & Tabular AI](https://architecture.learning.sap.com/docs/golden-path/ai-golden-path/build-and-deliver/predictive-tabular-ai)

### Reference Code

#### Recommended

- [SAP BTP AI Best Practices - RPT-1 Java](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/narrow-ai/RPT-1/Java)

**SAP Samples**

- [SAP-samples Java examples](https://github.com/SAP-samples/sap-rpt-samples/tree/main/sap-rpt-1/code_samples/java): The official sample repo includes `Predict.java` and `PredictParquet.java`

---

## Related Best Practices

[Access to Generative AI Models](https://sap.sharepoint.com/sites/210313/SitePages/GenAI%20-%20Plain%20-%20Direct%20-%20Inference%20Request.aspx)

[Classification](https://sap.sharepoint.com/sites/210313/SitePages/Narrow%20AI%20-%20Classification.aspx)

[Regression](https://sap.sharepoint.com/sites/210313/SitePages/Narrow%20AI%20-%20Linear%20Regression.aspx)

## Related Functional Patterns

[Information Classification in Categories](https://sap.sharepoint.com/sites/210313/SitePages/Functional%20Patterns%20-%20Information%20Analysis%20%26%20Processing%20-%20Information%20Classification%20in%20Categories.aspx)

### Additional Materials

## Contributors

_[Image not exported from SharePoint (embedded data URL)]_

Robledo, Francisco

CSS _BTP Hub_AI

---

Source: https://sap.sharepoint.com/sites/210313/SitePages/Predictive%20AI%20-%20Access%20To%20SAP%20RPT-1%20Model.aspx?isSPOFile=1&xsdata=MDV8MDJ8fGRjNzg1ZjgxYmZiYzQ0ODc0ZmIzMDhkZTgzNDZjNTkyfDQyZjc2NzZjZjQ1NTQyM2M4MmY2ZGMyZDk5NzkxYWY3fDB8MHw2MzkwOTI1MzcxMTg4NTc5MjR8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDJOb1lYUnpMekU1T2preFpHTTNaRGhqTW1aaE5EUmxaams1TmpneU4yWmlZakJoWlRBNVpXRmlRSFJvY21WaFpDNTJNaTl0WlhOellXZGxjeTh4Tnpjek5qVTJPVEE1TmprM3xiYjIxNjUzOTViODU0MDVmYzA1NDA4ZGU4MzQ2YzU5MXw4YzVkNDI4NjMwYjM0MDQ0OWJmYWEzNjY3ZmU5MGY2Zg%3D%3D&sdata=VEliZ3gzZWJLWlFZSDZNV1RZT3JDaUc2NjRZQTFqYStrNzJpYmY1N1RNST0%3D&ovuser=42f7676c-f455-423c-82f6-dc2d99791af7%2Cl.marques%40sap.com
