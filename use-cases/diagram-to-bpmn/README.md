# Diagram to BPMN Converter

A full-stack application that transforms business process diagram images into BPMN 2.0 XML format compatible with Signavio and SAP Build Process Automation. The application uses Large Language Models (LLMs) via SAP GenAI Hub to analyze diagram images and generate standardized BPMN XML output.

## Overview

This application provides an intuitive web interface for converting process diagrams into BPMN format, making it easy to digitize and standardize business processes. The system supports multiple AI providers and models, ensuring flexibility and optimal results for different types of diagrams.

### Key Features

- **Image Analysis**: Upload process diagrams in various formats (PNG, JPG, JPEG, WEBP, SVG)
- **AI-Powered Conversion**: Leverages state-of-the-art LLMs to understand and interpret diagram content
- **BPMN 2.0 Output**: Generates standardized BPMN XML files ready for import into Signavio
- **Multiple AI Providers**: Support for Anthropic Claude, OpenAI GPT, and Google Vertex AI
- **Real-time Processing**: Fast conversion with progress indicators and error handling
- **Download Ready**: Direct download of generated BPMN XML files

## Architecture

The application consists of two main components:

- **Frontend**: Streamlit-based web interface for user interaction
- **Backend**: FastAPI service handling image processing and LLM integration

```
Diagram_to_BPMN/
├── api/                    # FastAPI backend application
│   ├── app/
│   │   ├── main.py        # FastAPI application entry point
│   │   ├── models/        # Data models and schemas
│   │   ├── routers/       # API route handlers
│   │   ├── services/      # Business logic (BPMN generation)
│   │   ├── prompts/       # LLM prompt templates
│   │   └── utils/         # Utility functions
│   └── requirements.txt   # Backend dependencies
├── ui/                     # Streamlit frontend application
│   ├── src/
│   │   ├── Home.py        # Main UI application
│   │   ├── api_client.py  # API communication
│   │   └── utils.py       # UI utilities
│   ├── static/            # Static assets (CSS, images)
│   ├── streamlit_app.py   # Streamlit entry point
│   └── requirements.txt   # Frontend dependencies
├── diagrams/              # Sample diagram images for testing
├── docs/                  # Documentation
├── manifest.yaml          # Cloud Foundry deployment configuration
└── deploy.sh             # Automated deployment script
```

## Prerequisites

Before running the application, ensure you have:

- **Python 3.10+** (for local development)
- **Cloud Foundry CLI** (for deployment to SAP BTP)
- **SAP GenAI Hub Access** with valid credentials
- **Network Access** to selected LLM providers (OpenAI, Anthropic, Google Vertex)

## Local Development Setup

### 1. Environment Configuration

Create environment files for both frontend and backend:

```bash
# For the backend API
cp api/.env.example api/.env

# For the frontend UI
cp ui/.env.example ui/.env
```

Configure your credentials in the `.env` files:
- **`api/.env`**: Add your `AICORE_*` credentials and set a secure `API_KEY`
- **`ui/.env`**: Ensure `API_KEY` matches the backend configuration

### 2. Backend Setup

```bash
# Navigate to the API directory
cd api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app.main:app --reload
```

The API server will start on `http://localhost:8000`. You can explore the interactive API documentation at `http://localhost:8000/docs`.

### 3. Frontend Setup

Open a new terminal window:

```bash
# Navigate to the UI directory
cd ui

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
python streamlit_app.py
```

The web interface will be available at `http://localhost:8501`.

## Usage

### Web Interface

1. **Upload Diagram**: Use the file uploader to select a process diagram image
2. **Select AI Model**: Choose from available providers and models in the sidebar
3. **Generate BPMN**: Click "Generate BPMN XML" to process the image
4. **Download Result**: Download the generated BPMN XML file for import into Signavio

### Supported File Formats

- PNG (Portable Network Graphics)
- JPG/JPEG (Joint Photographic Experts Group)
- WEBP (Web Picture Format)
- SVG (Scalable Vector Graphics)

### Available AI Models

- **Anthropic**: Claude 4 Sonnet
- **OpenAI**: GPT-4.1, GPT-5 (Thinking)
- **Google Vertex AI**: Gemini 2.5 Pro

## API Reference

### Health Check

```bash
GET /api/health
```

Returns service status and availability.

### Generate BPMN

```bash
POST /api/bpmn/generate
```

**Parameters:**
- `file` (required): Image file upload
- `provider` (optional): AI provider (`anthropic`, `openai`, `gemini`)
- `model` (optional): Specific model name

**Response:**
```json
{
  "bpmn_xml": "<definitions>...</definitions>",
  "provider": "anthropic",
  "model": "anthropic--claude-4-sonnet",
  "success": true,
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 800,
    "total_tokens": 2300
  },
  "error": null
}
```

## Deployment to SAP BTP

### Automated Deployment

1. **Login to Cloud Foundry**:
   ```bash
   cf login -a <API_ENDPOINT> --sso
   ```

2. **Run Deployment Script**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

The script automatically generates a secure API key and deploys both applications.

### Manual Deployment

1. **Generate API Key**:
   ```bash
   openssl rand -hex 32
   ```

2. **Deploy Applications**:
   ```bash
   cf push --var api_key="your-secure-api-key"
   ```

### Configuration Updates

Before deployment, update the `manifest.yaml` file:
- Replace `diagram-to-bpmn` with your chosen application name
- Update route URLs to match your SAP BTP environment

## Technical Details

### BPMN Generation Process

1. **Image Upload**: User uploads a process diagram image
2. **Image Encoding**: Image is base64-encoded for LLM processing
3. **LLM Analysis**: Selected AI model analyzes the diagram structure and content
4. **XML Generation**: LLM generates BPMN 2.0 XML following Signavio compatibility standards
5. **Response Processing**: XML is extracted and validated before returning to user

### Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- LLM service failures
- Network connectivity issues
- Authentication problems
- Malformed responses

### Security

- API key authentication between frontend and backend
- CORS configuration for secure cross-origin requests
- Environment-based configuration management
- Secure credential handling via SAP GenAI Hub

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with both frontend and backend
5. Submit a pull request

## License

This project is designed for use with SAP BTP and follows SAP's development guidelines and best practices.

## Support

For issues and questions:
1. Check the API documentation at `/docs` endpoint
2. Review the application logs for detailed error information
3. Verify your SAP GenAI Hub credentials and access
4. Ensure network connectivity to selected LLM providers