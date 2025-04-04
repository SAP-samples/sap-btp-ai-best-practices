# SAP BTP AI Best Practice Demo - Data Masking - Typescript

This project demonstrates best practices for using data masking with generative AI models using the SAP AI SDK. It provides a simple example of how to interact with a language model to obtain responses based on user prompts.

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
├── package.json                # NPM configuration file
├── tsconfig.json               # TypeScript configuration file
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices/best-practices/data-masking/typescript
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

## Usage Example

The application will serve the `/generateEmail` API, which triggers the masking model, then sends a prompt to the AI model and logs the response. 

For local deployment, set `$SAMPLE_HOST` as `http://localhost:4004`. For remote deployment, set `SAMPLE_CAP_HOST` as the value returned from the deployment step.

#### Generate Email with masked data

```bash
curl --request POST \
  --url http://$SAMPLE_HOST/generateEmail \
  --header "Content-Type: application/json" \
  --data '{
  "prompt": "Please write an email to John Doe (john.doe@sap.com), informing them about the amazing capabilities of generative AI! Be brief and concise, write at most 6 sentences.", 
  "anonymize": true
}'
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
