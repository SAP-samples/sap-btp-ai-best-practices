# AI-PDF-Extraction: Document Processing System

This project is a document processing system that uses AI to extract information from PDF documents. It's designed for deployment to SAP BTP, Cloud Foundry.

- **Frontend:** [Streamlit](https://streamlit.io/) (Python-based web framework)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI Services:** OpenAI GPT-4 Vision, SAP AI SDK for document processing
## Project Structure

```
AI-PDF-Extraction/
├── api/                    # FastAPI backend application
│   ├── extraction/        # Document extraction modules
│   │   ├── common.py      # LLM factory helpers
│   │   └── simple_pdf_extractor.py
│   ├── routers/           # API route handlers
│   │   ├── extraction.py  # Document extraction endpoints
│   │   └── (removed)
│   ├── api_server.py      # Main FastAPI application
│   ├── models.py          # Pydantic data models
│   ├── requirements.txt   # Backend dependencies
│   └── README_api.md      # API documentation
├── ui/                     # Streamlit frontend application
│   ├── src/               # UI source code
│   │   ├── pages/         # Streamlit pages
│   │   │   ├── 1_Extraction_Config.py
│   │   │   ├── 2_Document_Processing.py
│   │   │   └── 3_Results_Dashboard.py
│   │   ├── api_client.py  # API communication
│   │   ├── Home.py        # Main UI page
│   │   └── utils.py       # Utility functions
│   ├── static/            # Static assets (fonts, images, styles)
│   ├── requirements.txt   # Frontend dependencies
│   └── streamlit_app.py   # Streamlit entry point
├── manifest.yaml          # Cloud Foundry deployment manifest
├── deploy.sh              # Automated deployment script
└── README.md              # This file
```

## Prerequisites

Before you begin, ensure you have the following installed:

- [Python](https://www.python.org/) (3.9 or higher)
- [Cloud Foundry CLI](https://github.com/cloudfoundry/cli/releases)
- OpenAI API key for GPT-4 Vision model
- SAP AI SDK access and credentials (for SAP AI services)

### API Keys Required

- **OpenAI API Key**: Required for document processing and information extraction
- **SAP AI SDK**: Required for SAP AI services integration (optional, for enhanced processing)

## Local Development Setup

Before running the application locally, you need to configure your environment variables. The application uses environment variables for configuration.

### 1. Configure Environment Variables

Create environment files for both the API and UI components:

```bash
# For the backend API
touch api/.env

# For the frontend UI  
touch ui/.env
```

### 2. Configure Your Secrets

Add the following environment variables to your `.env` files:

**`api/.env`**:
```bash
OPENAI_API_KEY=your_openai_api_key_here
API_KEY=your_secure_api_key_here
```

**`ui/.env`**:
```bash
API_BASE_URL=http://localhost:8000
API_KEY=your_secure_api_key_here
```

**Note**: The `API_KEY` should be the same in both files for authentication between frontend and backend.

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
uvicorn api_server:app --reload
```

Once the server is running, you can explore the interactive API documentation (Swagger UI) at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The API provides the following main endpoints:
- `/api/health` - Health check
- `/api/extraction/single` - Process single PDF document

### 2. Run the Frontend (UI)

Open a new terminal window for the frontend.

```bash
# Navigate to the UI directory
cd ui

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit development server
python streamlit_app.py
```

The Streamlit UI will be available at [http://localhost:8501](http://localhost:8501) and provides:

- **Extraction Configuration**: Configure document types and extraction schemas
- **Document Processing**: Upload and process PDF documents
- **Results Dashboard**: View extraction results and verification outcomes

## Configuration for Deployment

The `manifest.yaml` file is already configured for the AI-PDF-Extraction application with the following applications:

- **API Application**: `Sesajal_DataExtraction_api` (FastAPI backend)
- **UI Application**: `Sesajal_DataExtraction_ui` (Streamlit frontend)

If you need to customize the application names or routes, update the following in `manifest.yaml`:

- **API name**: `Sesajal_DataExtraction_api`
- **UI name**: `Sesajal_DataExtraction_ui`
- **Routes**: Update the route URLs as needed for your Cloud Foundry environment

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

## API Documentation

The application provides comprehensive API endpoints for document processing:

### Extraction API (`/api/extraction`)

- **POST `/api/extraction/single`**: Extract information from a single PDF document

### Supported Document Types

- `conoce_cliente` - Customer identification documents
- `comentarios_vendedor` - Sales comments
- `constancia_fiscal` - Tax certificates
- `ine` - National ID documents
- `custom` - Custom document types with user-defined questions

### Features

- **AI-Powered Extraction**: Uses OpenAI GPT-4 Vision for intelligent document processing
- **Client-Side Batch**: The UI can iterate over multiple files using the single endpoint
- **Flexible Schemas**: Configure custom questions in the UI

For detailed API documentation, visit the Swagger UI at `/docs` when running the API server locally, or check the `api/README_api.md` file.
