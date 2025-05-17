# SAP BTP AI Best Practice Demo - Vector Query (RAG) - Typescript

This project demonstrates best practices for using Retrieval Augmented Generation with generative AI models using the SAP AI SDK. It provides a simple example of how to query with a language model to obtain responses based on user prompts.

## Project Structure

```
typescript
├── db
│   └── src                     # Contains HANA Database schema tables, etc.
├── src
│   ├── server.ts               # Entry point of the application
│   ├── services
│   │   └── ai-sdk.ts           # Contains the logic for AI model access using the AI SDK
│   │   └── hana-db.ts          # Contains the logic fo HANA Database access and query
│   └── utils
│       └── csv-file.ts         # Utility for parsing the CSV file
│       └── logger.ts           # Logger utility for logging messages
├── .env.example                # Template for environment variables
├── .gitignore                  # Specifies files to ignore in Git
├── package.json                # NPM configuration file
├── tsconfig.json               # TypeScript configuration file
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices/best-practices/vector-rag-query/typescript
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
   cf env ai-rag-query-typescript-demo-srv
   ```
  
6. **Run the application locally:**

   ```bash
   npm run watch
   ```

## Usage Example

The application will serve the following APIs:

 - POST `/uploadScienceData` upload and created embeddings for the science data from a CSV file.
 - GET `/scienceData` view the uploaded science data
 - GET `/queryScienceData` query the uploaded science data for similar records using vector query

For local deployment, set `$SAMPLE_HOST` as `localhost:3000`. For remote deployment, set `$SAMPLE_HOST` as the value returned from the deployment step (and you will need to change it to https).

#### Upload Science Data

```bash
curl --request POST --url http://$SAMPLE_HOST/uploadScienceData \
  --header 'Content-Type: multipart/form-data' \
  --form 'csvFile=@../../vector-rag-embedding/sample_files/science-data-sample.csv'
```

#### View Science Data

```bash
curl --request GET --url http://$SAMPLE_HOST/scienceData
```

#### Query Science Data for closest matches

```bash
curl --request GET -G --url http://$SAMPLE_HOST/queryScienceData \
   --data-urlencode 'query=how to test for fat in foods'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
