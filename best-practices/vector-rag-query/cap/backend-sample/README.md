# SAP BTP AI Best Practice Demo - Vector Query (RAG) - CAP (backend)

Large Language Models (LLMs) may struggle to answer questions about topics not present in their training data. To overcome this limitation, external information can be provided during inference to enhance the model’s response. This technique is known as Retrieval-Augmented Generation (RAG).

To fetch the relevant context for a given query, a vector store is used to store documents in vectorized form. Using similarity metrics, the store retrieves the most semantically relevant documents, which are then passed as context to the LLM.

This project demonstrates how to retrieve top-matching documents from the SAP HANA Vector Store and use them as context for the LLM during inference.

## Prerequisites

- SAP Business Technology Platform account
- Access to SAP AI Core service
- Node.js LTS version
- Cloud Foundry CLI

## Configuration

The application requires proper configuration to connect to the SAP AI Core service. This is handled through CDS bindings in both local and remote deployments.

## Project Structure

```
backend-sample/
├── db
│   └── schema.cds           # Entity definitions used for HANA database schema
├── srv/                     # Service layer containing CAP services
│   ├── types.ts             # Type definitions for use in service
│   ├── services
│   │    └── ai-sdk.ts       # Logic for interacting with the AI SDK
│   ├── utils
│   │    └── csv-file.ts     # CSV File parsing utility
│   ├── rag-service.cds      # CAP Service Definition
│   └── rag-service.ts       # CAP Service Implementation
├── package.json             # Project dependencies and scripts
├── tsconfig.json            # TypeScript configuration
├── manifest.yml             # CF deployment configuration
└── README.md                # Project documentation
```

## Local Deployment

1. Install dependencies using `npm install`.

2. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`.

3. Bind the application to your AI Core instance:

   ```bash
   cds bind -2 AI_CORE_INSTANCE_NAME:AI_CORE_INSTANCE_SERVICE_KEY_NAME
   ```

4. Build the HANA / HDI Container

   ```bash
   npm run build:db
   ```

5. Run the application with the binding:

   ```bash
   npm run watch
   ```

## Remote Deployment

1. Install dependencies using `npm install`.
2. In the `mta.yml`, under the `resources` section on the `best-practices-aicore`, modify the `service-name` from `best-practices-aicore` to the name of your AI Core Service instance.
3. Transpile the CAP application using `npm run build`.
4. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`.
5. Deploy the application using `npm run deploy`.

## Usage

The application will serve the following APIs:

 - POST `/uploadScienceData` upload and created embeddings for the science data from a CSV file.
 - GET `/scienceData` view the uploaded science data
 - GET `/queryScienceData` query the uploaded science data for similar records using vector query

For local deployment, set `SAMPLE_CAP_HOST` as `localhost:4004`. For remote deployment, set `SAMPLE_CAP_HOST` as the value returned from the deployment step (and you will need to change it to https).

> [!WARNING]  
> All CDS services are marked with `@requires: 'any'` and are publicly accessible in order to simplify the testing and deployment process.
> Apply proper authentication mechanisms to avoid unauthorized access.

### Upload csv file

```bash
curl --request PUT --url http://SAMPLE_CAP_HOST/odata/v4/rag/ScienceDataUpload/content \
  --header 'Content-Type: text/csv' \
  --data-binary '@../../sample_files/science-data-sample.csv'
```

### View Science Data

```bash
curl --request GET --url http://SAMPLE_CAP_HOST/odata/v4/rag/ScienceData
```

#### Query Science Data

```bash
curl --request GET \
  --url 'http://localhost:4004/odata/v4/rag/queryScienceData(query=%27how%2520to%2520test%2520for%2520fat%2520in%2520foods%27)'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.