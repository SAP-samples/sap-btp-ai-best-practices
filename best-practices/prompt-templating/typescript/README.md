# SAP BTP AI Best Practice Demo - Prompt Templating - Typescript

This project demonstrates best practices for using prompt templating with generative AI models using the SAP AI SDK. It provides a simple example of how to interact with a language model to obtain responses based on user prompts.

## Project Structure

```
typescript
├── src
│   ├── server.ts               # Entry point of the application
│   ├── services
│   │   └── aiOrchestration.ts  # Contains the orchestration logic for AI model access
│   └── utils
│       └── logger.ts           # Logger utility for logging messages
├── .env.example                # Template for environment variables
├── .gitignore                  # Specifies files to ignore in Git
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
   cd sap-btp-ai-best-practices/best-practices/prompt-templating/typescript
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
   

## Usage Example

The application will serve the `/askCapitalOfCountry` API, which uses prompt templating, then sends a prompt to the AI model and logs the response. 

For local deployment, set `SAMPLE_HOST` as `http://localhost:3000`. For remote deployment, set `SAMPLE_HOST` as the value returned from the deployment step.

#### Ask for Capital Of Country

```bash
curl --request POST \
  --url http://$SAMPLE_HOST/askCapitalOfCountry \
  --header "Content-Type: application/json" \
  --data '{
  "country": "United States"
}'
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
