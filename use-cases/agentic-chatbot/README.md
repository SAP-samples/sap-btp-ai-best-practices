# UI5 Web Components + FastAPI Template for SAP BTP

This project is a template for building full-stack applications using a modern frontend and a powerful backend, designed for easy deployment to SAP BTP, Cloud Foundry.

- **Frontend:** [UI5 Web Components](https://sap.github.io/ui5-webcomponents/) (with plain JavaScript and Vite)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)

## Project Structure

```
template-ui5-web-components-fastapi/
├── api/          # FastAPI backend application
│   ├── app/
│   ├── requirements.txt
│   └── ...
├── ui/           # UI5 Web Components frontend application
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── ...
└── manifest.yaml # Deployment manifest for Cloud Foundry
```

## Prerequisites

Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (which includes npm)
- [Python](https://www.python.org/) (3.8 or higher)
- [Cloud Foundry CLI](https://github.com/cloudfoundry/cli/releases)

## Local Development Setup

Before running the application locally, you need to configure your environment variables. This project uses `.env.example` files in the `api` and `ui` directories as templates.

### 1. Create Your `.env` Files

Copy the example files to create your own local configuration:

```bash
# For the backend API
cp api/.env.example api/.env

# For the frontend UI
cp ui/.env.example ui/.env
```

### 2. Configure Your Secrets

After creating the files, open them and update the placeholder values:

- **`api/.env`**: Fill in your `AICORE_*` credentials and set a secure `API_KEY`.
- **`ui/.env`**: Ensure `VITE_API_KEY` matches the `API_KEY` from `api/.env`.

## Local Development

To run the application on your local machine, you will need to run the frontend and backend servers separately.

### 1. Run the Backend (API)

```bash
# Navigate to the API directory
cd api

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server (it will automatically load variables from api/.env)
uvicorn app.main:app --reload
```

Once the server is running, you can explore the interactive API documentation (Swagger UI) at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Run the Frontend (UI)

Open a new terminal window for the frontend.

```bash
# Navigate to the UI directory
cd ui

# Install dependencies
npm install

# Run the Vite development server (it will automatically load variables from ui/.env)
npm run dev
```

## Configuration for Deployment

Before deploying, you need to update the `manifest.yaml` file with the unique names for your application. The template uses `template-ui5-web-components-fastapi` as a placeholder.

Perform a "find and replace" in `manifest.yaml` for the following string:

- **Find:** `template-ui5-web-components-fastapi`
- **Replace with:** `my-awesome-app` (or your chosen app name)

## Deployment to SAP BTP

This project includes an automated deployment script that handles API key generation and simplifies the deployment process.

### Automated Deployment (Recommended)

1.  **Login to Cloud Foundry**

    Before running the script, make sure you are logged in to your Cloud Foundry endpoint.

    ```bash
    # Replace <API_ENDPOINT> with your specific endpoint
    cf login -a <API_ENDPOINT> --sso

    # Example:
    cf login -a https://api.cf.eu10-004.hana.ondemand.com -o btp-ai-sandbox -s Dev
    ```

2.  **Run the Deployment Script**

    From the root directory of the project, run the `deploy.sh` script:

    ```bash
    # Make the script executable
    chmod +x deploy.sh

    # Run the deployment script
    ./deploy.sh
    ```

    The script will:

    1.  Generate a secure, temporary API key for the deployment.
    2.  Deploy both the UI and API applications with the new key.

### Manual Deployment

If you prefer to deploy manually or need to provide a specific, long-lived API key, you can use the standard `cf push` command.

<details>
<summary>Click to view manual deployment instructions</summary>

1.  **Generate a Secure API key**

    If you don't already have one, generate a secure key:

    ```bash
    openssl rand -hex 32
    ```

2.  **Login to Cloud Foundry**

    If you are not already logged in, open your terminal and connect to your Cloud Foundry endpoint.

    ```bash
    # Replace <API_ENDPOINT> with your specific endpoint
    cf login -a <API_ENDPOINT> --sso

    # Example:
    cf login -a https://api.cf.eu10-004.hana.ondemand.com -o btp-ai-sandbox -s Dev
    ```

3.  **Deploy the Application**

        From the root directory, run the `push` command, providing your API key as a variable.
        ```bash
        cf push --var api_key="your-secure-api-key-goes-here"
        ```

    </details>

Once the deployment is complete, the URLs for your live application will be displayed in the terminal.

## Data Anonymization (seed files)

Use the helper script to anonymize CSV seed data before sharing the project.

```bash
# Generate anonymized copies next to the originals
python3 scripts/anonymize_data.py

# Overwrite originals or set a custom suffix
python3 scripts/anonymize_data.py --inplace
python3 scripts/anonymize_data.py --suffix .anon
```

What it changes (deterministic per material):

- Pseudonymizes plant/material IDs
- Replaces product names with generic labels
- Shifts dates 0–45 days into the past (format preserved, never future)
- Scales numeric quantities; blanks preserved
