# SAP BTP AI Best Practice Demo - Content Filtering - Typescript

This project demonstrates best practices for using content filtering with generative AI models using the SAP AI SDK. Content filtering is a critical process that ensures the quality and relevance of input data. It involves evaluating and selecting data to prevent low-quality or irrelevant inputs from affecting model performance. This process is essential for maintaining the accuracy and reliability of LLMs, especially when they are used for real-time applications or sensitive tasks.

## Project Structure

```
typescript
├── src
│   ├── server.ts               # Entry point of the application
│   ├── services
│   │   └── aiOrchestration.ts # Contains the orchestration logic for AI model access
│   └── utils
│       └── logger.ts          # Logger utility for logging messages
├── .env.example                # Template for environment variables
├── .gitignore                  # Specifies files to ignore in Git
├── manifest.yml                # Deployment configuration file
├── package.json                # NPM configuration file
├── tsconfig.json               # TypeScript configuration file
└── README.md                   # Project documentation
```
## Configuration

The application requires proper configuration to connect to the SAP AI Core service. This is handled through CDS bindings in both local and remote deployments.

## Local Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices/best-practices/content-filtering/typescript
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Configure environment variables:**

   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Populate the `.env` file with the required values.

4. **Run the application:**
   ```bash
   npm run watch
   ```

## Remote Deployment

1. Install dependencies using `npm install`.
2. Transpile the CAP application using `npm run build`.
3. Modify `services` in `manifest.yml`, rename `best-practices-aicore` to match the service instance in your space.
4. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`.
5. Deploy the application using `npm run deploy`.
   

## Usage Examples

The application will serve the `/chatWithSupport` (filter input) and `/generateParaphrase` (filter output) APIs, which uses content filtering, then sends a prompt to the AI model and logs the response. 

For local deployment, set `SAMPLE_HOST` as `http://localhost:3000`. For remote deployment, set `SAMPLE_HOST` as the value returned from the deployment step.

#### Chat with Agent

**Filtered**
```bash
curl --request POST \
  --url http://$SAMPLE_HOST$/chatWithSupport \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": true
}'
```

**Unfiltered**
```bash
curl --request POST \
  --url http://$SAMPLE_HOST$/chatWithSupport \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": false
}'
```

#### Generate Paraphrase

**Filtered**
```bash
curl --request POST \
  --url http://$SAMPLE_HOST$/generateParaphrase \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterOutput": true
}'
```

**Unfiltered**
```bash
curl --request POST \
  --url http://$SAMPLE_HOST$/generateParaphrase \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterOutput": false
}'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
