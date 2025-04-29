# SAP BTP AI Best Practice Demo - Content Filtering - CAP (full stack)

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
fullstack-sample/
├── app/                     # UI5 frontend applications
│   └── ask-agent/         # TypeScript UI5 application 
│       ├── webapp/          # UI5 application source code
│       ├── package.json     # Frontend dependencies
│       ├── ui5.yaml         # UI5 tooling configuration
│       └── README.md        # Frontend documentation
├── srv/                     # Service layer containing CAP services
│   ├── orchestration.cds    # CDS service definitions for AI orchestration
│   └── orchestration.js     # Service implementation with AI SDK integration
├── package.json             # Project dependencies and scripts
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

4. Start the backend with AI Core binding:

   ```bash
   npm run watch
   ```

5. In a separate terminal, start the UI5 application:

   ```bash
   npm run start:ui5
   ```

   The UI5 application will automatically connect to the running CAP backend.

## Remote Deployment

> [!WARNING]  
> All CDS services are marked with `@requires: 'any'` and are publicly accessible in order to simplify the deployment process.
> Apply proper authentication mechanisms to avoid unauthorized access.

1. Install dependencies using `npm install`.
2. In the `mta.yml`, under the `resources` section on the `fullstack-sample-test-aicore`, modify the `service-name` from `best-practices-aicore` to the name of your AI Core Service instance.
3. Transpile the CAP application using `npm run build`.
4. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`
5. Deploy the application using `npm run deploy`
6. Open URL of the `ai-content-filteirng-cap-fullstack-demo` in the browser to access the application.

## Usage

The application will serve the `/chatWithSuport` (filter input) and `/generateParaphrase` (filter output) APIs, which uses content filtering, then sends a prompt to the AI model and logs the response. 

For local deployment, the backend services are available at `http://localhost:4004` and the UI5 application at `http://localhost:8080`.

For remote deployment, access the application at the url returned during the deployment or found in the BTP cockpit.

### Chat with Agent (Filter Input)
#### Sample UI - Request
<img src="readme-image-1.png" alt="image1" width="500"/>

#### Sample UI - Response - With Filtering
<img src="readme-image-2.png" alt="image1" width="500"/>

#### Sample UI - Response - Without Filtering
<img src="readme-image-3.png" alt="image1" width="500"/>

#### Backend Services (OData v4)
**Filtered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/chatWithSuport \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": true
}'
```

**Unfiltered**
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST$/odata/v4/orchestration/chatWithSupport \
  --header "Content-Type: application/json" \
  --data '{
  "input": "I hate you!", "filterInput": false
}'
```

### Generate Paraphrase (Filter Output)
#### Sample UI - Request
<img src="readme-image-4.png" alt="image1" width="500"/>

#### Sample UI - Response - With Filtering
<img src="readme-image-5.png" alt="image1" width="500"/>

#### Sample UI - Response - Without Filtering
<img src="readme-image-6.png" alt="image1" width="500"/>

#### Backend Services (OData v4)
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
