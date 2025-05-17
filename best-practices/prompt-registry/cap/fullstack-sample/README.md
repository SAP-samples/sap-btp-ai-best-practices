# SAP BTP AI Best Practice Demo - Prompt Registry - CAP (full stack)

This project demonstrates best practices for using prompt registry with generative AI models using the SAP AI SDK. It provides a simple example of how to interact with a prefined template to interact with a language model to obtain responses based on user prompts.

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
├── app/                                  # UI5 frontend applications
│   └── ask-capital/                      # TypeScript UI5 application for querying capitals
│       ├── webapp/                       # UI5 application source code
│       ├── package.json                  # Frontend dependencies
│       ├── ui5.yaml                      # UI5 tooling configuration
│       └── README.md                     # Frontend documentation
├── srv/                                  # Service layer containing CAP services
|   └── orchestration/                     
│       ├── orchestration-service.cds     # CDS service definitions for AI orchestration
│       └── orchestration-service.ts      # Service implementation with AI SDK integration
|   └── prompt-registry/                    
│       ├── prompt-registry-service.cds   # CDS service definitions for AI prompt registry
│       └── prompt-registry-service.ts    # Service implementation with AI prompt registry
├── package.json                          # Project dependencies and scripts
├── manifest.yml                          # CF deployment configuration
└── README.md                             # Project documentation
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
2. In the `mta.yml`, under the `resources` section on the `best-practices-aicore`, modify the `service-name` from `best-practices-aicore` to the name of your AI Core Service instance.
3. Transpile the CAP application using `npm run build`.
4. Login using `cf login -a API_ENDPOINT -o ORG -s SPACE`
5. Deploy the application using `npm run deploy`
6. Open URL of the `ai-prompt-registry-cap-fullstack-demo` in the browser to access the application.

## Usage
The application will serve the `/askCapitalOfCountry` API, which uses prompt templating, then sends a prompt to the AI model and logs the response. 

> [!IMPORTANT] 
> In order to run the fiori application, make sure to execute the [Create Prompt Registry Template](#create-prompt-registry-template) request first.

For local deployment, set `SAMPLE_CAP_HOST` as `http://localhost:4004`. For remote deployment, set `SAMPLE_CAP_HOST` as the value returned from the deployment step.

### Sample UI - Request
![alt text](readme-image-1.png)

### Sample UI - Response
![alt text](readme-image-2.png)

### Backend Services (OData v4)
#### Create Prompt Registry Template
This is a one-time setup step, and could also be created manually in the AI Launchpad.

*NOTE: Save the ID from this step, so you can use it in the last step to delete the prompt from the registry.
```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST/odata/v4/prompt-registry/createPromptTemplate \
  --header "Content-Type: application/json" \
  --data '{
   "name": "askCapitalOfCountry",
   "scenario": "ai-best-practices",
   "content": "What is the capital of {{?country}}?"
}'
```

#### Ask for Capital Of Country

```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST/odata/v4/orchestration/askCapitalOfCountry \
  --header "Content-Type: application/json" \
  --data '{
  "country": "United States"
}'
```

#### Delete Prompt Registry Template
This is just to show it can be deleted. It can also be deleted manually in the AI Launchpad.

```bash
curl --request POST \
  --url http://$SAMPLE_CAP_HOST/odata/v4/prompt-registry/deletePromptTemplate \
  --header "Content-Type: application/json" \
  --data '{
   "id": "$PROMPT_ID$"
}'
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
