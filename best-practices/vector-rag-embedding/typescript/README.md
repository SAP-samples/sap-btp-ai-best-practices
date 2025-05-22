# SAP BTP AI Best Practice Demo - Vector Embedding (RAG) - Typescript

This project demonstrates best practices for accessing Embeddings models using the SAP GenAI Hub and storing them in SAP HANA Vector Store. It provides examples of different ways to create and store embeddings in the vector store.

## Project Structure

```
typescript
├── db
│   └── src                     # HANA database source files, table definitions, etc.
├── src
│   ├── server.ts               # Entry point of the application
│   ├── types.ts                # Type definitions for use in service
│   ├── services
│   │   └── ai-sdk.ts           # Logic for interacting with the AI SDK
│   │   └── hana-db.ts          # Logic for interacting with the HANA database
│   └── utils
│       ├── csv-file.ts         # CSV File parsing utility
│       └── logger.ts           # Logger utility for logging messages
├── .env.example                # Template for environment variables
├── .gitignore                  # Specifies files to ignore in Git
├── mta.yaml                    # CF deployment configuration file
├── package.json                # NPM configuration file
├── tsconfig.json               # TypeScript configuration file
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SAP-samples/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices/best-practices/vector-rag-embedding/typescript
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`.

4. **Build and deploy to create the HDI container:**
   ```bash
   npm run build
   npm run deploy
   ```

5. **Configure environment variables:**

   Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   
   Populate the `.env` file with the required values.
   ```bash
   VCAP_SERVICES='{
      < contents of VCAP_SERVICES >
   }'
   ```

   To view the environment (VCAP_SERVICES) of the deployed service, you can use the BTP cockpit or use this command:
   ```
   cf env ai-rag-embedding-typescript-demo-srv
   ```
  
6. **Run the application locally:**
   ```bash
   npm run watch
   ```

## Usage Example

The application will serve the following APIs:
 - POST `/uploadScienceData` upload and created embeddings for the science data from a CSV file.
 - GET `/scienceData` view the uploaded science data

For local deployment, set `$SAMPLE_HOST` as `http://localhost:3000`. For remote deployment, set `$SAMPLE_HOST` as the value returned from the deployment step.

#### Upload Science Data

```bash
curl --request POST --url http://$SAMPLE_HOST/uploadScienceData \
  --header 'Content-Type: multipart/form-data' \
  --form 'csvFile=@../sample_files/science-data-sample.csv' 
```

#### View Science Data

```bash
curl --request GET --url http://$SAMPLE_HOST/scienceData
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
