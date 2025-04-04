# SAP BTP AI Best Practice Demo - Content Filtering - CAP (backend)

This project demonstrates best practices for accessing generative AI models through a CAP (Cloud Application Programming model) application using the SAP Cloud SDK for AI. It provides examples of how to interact with language models to obtain responses based on user prompts within a CAP service context.

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
├── srv/                     # Service layer containing CAP services
│   ├── orchestration.cds    # CDS service definitions for AI orchestration
│   └── orchestration.ts     # Service implementation with AI SDK integration
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

4. Run the application with the binding:

   ```bash
   npm run watch
   ```

## Remote Deployment

1. Install dependencies using `npm install`.
2. In the `mta.yml`, under the `resources` section on the `best-practices-aicore`, modify the `service-name` from `best-practices-aicore` to the name of your AI Core Service instance.
3. Transpile the CAP application using `npm run build`.
4. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`.
5. Deploy the application using `npm run deploy`.


## Usage Examples
> [!WARNING]  
> All CDS services are marked with `@requires: 'any'` and are publicly accessible in order to simplify the testing and deployment process.
> Apply proper authentication mechanisms to avoid unauthorized access.
>
> 

The application will serve the `/chatWithAgent` (filter input) and `/generateParaphrase` (filter output) APIs, which uses content filtering, then sends a prompt to the AI model and logs the response. 

For local deployment, set `SAMPLE_CAP_HOST` as `http://localhost:4004`. For remote deployment, set `SAMPLE_CAP_HOST` as the value returned from the deployment step.

#### Chat with Agent (Filter Input)

**Filtered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/chatWithAgent \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": true
}'
```

**Unfiltered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/chatWithAgent \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": false
}'
```

#### Generate Paraphrase (Filter Output)

**Filtered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/generateParaphrase \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterOutput": true
}'
```

**Unfiltered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/generateParaphrase \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterOutput": false
}'
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.