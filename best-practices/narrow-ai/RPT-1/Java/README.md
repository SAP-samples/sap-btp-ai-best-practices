# SAP RPT-1 SDK Demo (Java)

This example shows how to call SAP RPT-1 from Java using the SAP AI SDK.

It includes two runnable examples:

- `without-schema`: schema-free table completion predicting `COSTCENTER`
- `with-schema`: structured table completion predicting `DISCOUNT_RATE`

## Project Structure

```text
Java
├── README.md
└── sample
    ├── .env.example
    ├── .gitignore
    ├── pom.xml
    └── src
        ├── main
        │   └── java/com/sap/ai/bestpractices/rpt1
        └── test
            └── java/com/sap/ai/bestpractices/rpt1
```

## Prerequisites

- Java 17
- Maven 3.9+
- An SAP AI Core service key with access to SAP RPT

## Configuration

The sample expects these environment variables:

```bash
export AICORE_SERVICE_KEY='{"clientid":"...","clientsecret":"...","url":"...","serviceurls":{"AI_API_URL":"https://..."}}'
export RPT1_MODEL_NAME='sap-rpt-1-small'
```

Optional:

```bash
export RPT1_MODEL_VERSION='<optional-model-version>'
```

The current Java SDK resolves the RPT deployment through the model identifier and AI Core configuration rather than an explicit deployment ID. `AICORE_SERVICE_KEY` is the primary local-development setup. If your landscape uses SAP Cloud SDK destinations or another supported SAP AI SDK connection mechanism, you can adapt the runtime environment accordingly, but this sample documents and validates the service-key path first.

## Install And Verify

```bash
cd best-practices/narrow-ai/RPT-1/Java/sample
mvn test
```

## Run

Run one example:

```bash
mvn spring-boot:run -Dspring-boot.run.arguments=without-schema
mvn spring-boot:run -Dspring-boot.run.arguments=with-schema
```

Run both:

```bash
mvn spring-boot:run -Dspring-boot.run.arguments=all
```

If no argument is provided, the sample defaults to `without-schema`.
