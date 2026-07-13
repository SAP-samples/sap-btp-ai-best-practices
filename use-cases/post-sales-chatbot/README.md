# Post-Sales Chatbot - Apex Automotive Services

An intelligent AI-powered chatbot application for post-sales customer service in the automotive industry. This application enables customers to interact with Apex Automotive Services through a conversational interface to access vehicle information, service history, recommendations, and schedule appointments.

## Overview

The Post-Sales Chatbot is a full-stack application that provides an AI assistant for automotive service customers. It uses advanced language models and agentic AI capabilities to help customers:

- Identify themselves using various identifiers (email, phone, VIN, license plate, name)
- View their vehicle information and details
- Access service history
- Get personalized service recommendations based on mileage
- Discover active promotions
- Schedule service appointments

## Architecture

The application consists of two main components:

### Backend API (`api/`)
- **Framework**: FastAPI (Python)
- **AI Framework**: LangGraph with LangChain
- **LLM Provider**: OpenAI (GPT-4o)
- **Data Management**: Pandas for CSV data processing
- **Session Management**: In-memory session state management

### Frontend UI (`ui/`)
- **Framework**: UI5 Web Components
- **Build Tool**: Vite
- **Features**: Real-time streaming chat interface with markdown support

## Features

### Core Capabilities

1. **Client Identification**
   - Find clients by email, phone number, VIN, license plate, or full name
   - Automatic session state management with client context

2. **Vehicle Management**
   - List all vehicles for an identified client
   - Select and view detailed vehicle information
   - Access vehicle specifications (make, model, year, mileage, license plate)

3. **Service History**
   - Retrieve the last 3 service visits for a vehicle
   - View service dates, operations performed, and prices
   - Access detailed service records

4. **Service Recommendations**
   - Get personalized service recommendations based on current mileage
   - Calculate next service milestone (10k, 20k, 30k, etc.)
   - Display recommended service kits with operations and pricing

5. **Promotions**
   - Find active promotions for specific vehicles
   - View promotion details, valid dates, and included operations

6. **Appointment Scheduling**
   - Request service appointments
   - Specify preferred dates and times
   - Receive confirmation messages

### Technical Features

- **Streaming Responses**: Real-time NDJSON streaming for progressive response updates
- **Tool Execution Visibility**: Users can see when the AI is using tools to fetch information
- **Session Management**: Persistent conversation context across multiple interactions
- **API Security**: API key-based authentication
- **CORS Support**: Configurable CORS for development and production environments

## Project Structure

```
post-sales-chatbot/
├── api/                          # Backend API application
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── security.py          # API key authentication
│   │   ├── models/              # Pydantic models
│   │   │   ├── chat.py          # Chat request/response models
│   │   │   ├── chat_history.py  # Chat history models
│   │   │   ├── session.py       # Session management models
│   │   │   └── common.py        # Common models (health check)
│   │   ├── routers/             # API route handlers
│   │   │   └── apex_chat.py     # Chat endpoints
│   │   ├── services/            # Business logic services
│   │   │   ├── data_handler.py  # CSV data loading and management
│   │   │   └── session_manager.py # Session state management
│   │   └── utils/               # Utility functions
│   │       └── langgraph/      # LangGraph agent configuration
│   │           ├── apex_tools.py # LangChain tools for business operations
│   │           └── common.py    # LLM configuration
│   ├── data/
│   │   └── new_tables/          # CSV data files
│   │       ├── Cliente.csv      # Client data
│   │       ├── Unidad.csv       # Vehicle data
│   │       ├── Servicios.csv    # Service history
│   │       ├── Operacion.csv    # Service operations
│   │       ├── Campanas.csv     # Promotions/campaigns
│   │       ├── Kits.csv          # Service kits
│   │       └── Materiales.csv    # Materials inventory
│   └── requirements.txt         # Python dependencies
├── ui/                          # Frontend UI application
│   ├── src/
│   │   ├── pages/
│   │   │   ├── apex-chat/       # Chat interface page
│   │   │   └── home/            # Home page
│   │   ├── services/
│   │   │   └── api.js           # API client service
│   │   ├── modules/             # Routing and navigation
│   │   └── config/              # Configuration
│   ├── public/                  # Static assets
│   └── package.json             # Node.js dependencies
├── manifest.yaml                # Cloud Foundry deployment manifest
└── deploy.sh                    # Deployment script
```

## Prerequisites

### Backend Requirements
- Python 3.9 or higher
- SAP BTP Cloud Foundry account (for deployment)
- OpenAI API key (for LLM access)
- Access to SAP AI SDK (sap-ai-sdk-gen)

### Frontend Requirements
- Node.js 16 or higher
- npm or yarn package manager

### Data Requirements
- CSV files containing:
  - Client information (Cliente.csv)
  - Vehicle data (Unidad.csv)
  - Service history (Servicios.csv)
  - Service operations (Operacion.csv)
  - Promotions (Campanas.csv)
  - Service kits (Kits.csv)
  - Materials (Materiales.csv)

## Installation

### Backend Setup

1. Navigate to the API directory:
```bash
cd api
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the `api/` directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
API_KEY=your_secure_api_key_here
TABLES_PATH=data/new_tables/
LLM_MODEL_NAME=gpt-4o
APP_ENV=development
```

5. Ensure CSV data files are in the `data/new_tables/` directory

### Frontend Setup

1. Navigate to the UI directory:
```bash
cd ui
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
Create a `.env` file in the `ui/` directory:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_KEY=your_secure_api_key_here
```

## Running Locally

### Backend

1. Activate the virtual environment (if using one):
```bash
cd api
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Run the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

### Frontend

1. Navigate to the UI directory:
```bash
cd ui
```

2. Start the development server:
```bash
npm run dev
```

The UI will be available at `http://localhost:5173` (or the next available port)

## API Endpoints

### Health Check
- `GET /api/health` - Check API health status

### Chat Endpoints
- `POST /api/apex/chat` - Send a chat message and receive a response
- `POST /api/apex/chat/stream` - Send a chat message and receive streaming NDJSON responses
- `POST /api/apex/reset` - Reset the conversation session and get initial greeting

### Authentication
All chat endpoints require an API key in the request header:
```
X-API-Key: your_api_key_here
```

### Session Management
Sessions are managed via the `X-Session-Id` header:
- If not provided, a new session is created
- If provided, the existing session is used
- Session ID is returned in response headers

## Deployment

### Cloud Foundry Deployment

1. Ensure you're logged into Cloud Foundry:
```bash
cf login
```

2. Set the API key environment variable:
```bash
export API_KEY=$(openssl rand -hex 32)
```

3. Run the deployment script:
```bash
./deploy.sh
```

The deployment script will:
- Generate a secure API key if not provided
- Deploy both the API and UI applications to Cloud Foundry
- Configure environment variables
- Set up routes

### Manual Deployment

1. Deploy the API:
```bash
cd api
cf push apex-api --var api_key="$API_KEY"
```

2. Deploy the UI:
```bash
cd ui
cf push postsale-chatbot --var api_key="$API_KEY"
```

### Environment Variables for Production

The `manifest.yaml` file configures the following environment variables:

**API Application:**
- `ALLOWED_ORIGIN`: Frontend application URL
- `API_BASE_URL`: API base URL
- `APP_ENV`: Set to "production"
- `API_KEY`: Secure API key for authentication
- `TABLES_PATH`: Path to CSV data files
- `LLM_MODEL_NAME`: LLM model to use (default: gpt-4o)

**UI Application:**
- `VITE_API_BASE_URL`: Backend API URL
- `VITE_APP_HOST`: Frontend application host
- `VITE_API_KEY`: API key for frontend requests

## Configuration

### LLM Configuration

The application uses OpenAI's GPT-4o model by default. To change the model, update the `LLM_MODEL_NAME` environment variable or modify `app/config.py`.

### Data Configuration

CSV files should be placed in the directory specified by `TABLES_PATH` (default: `data/new_tables/`). The data handler automatically:
- Cleans numeric fields (removes currency symbols, commas)
- Parses date fields
- Removes duplicate records
- Validates data integrity

### Session Configuration

Sessions expire after 60 minutes by default. This can be configured via the `SESSION_EXPIRY_MINUTES` environment variable.

## Usage Examples

### Starting a Conversation

1. Open the chatbot interface
2. The assistant will greet you and ask how it can help
3. Identify yourself using one of the following:
   - "My email is john.doe@example.com"
   - "My phone number is 555-1234"
   - "My name is John Doe"
   - "My VIN is ABC123456789"
   - "My license plate is XYZ-1234"

### Querying Vehicle Information

Once identified, you can ask:
- "What vehicles do I have?"
- "Show me details for my 2020 Toyota Camry"
- "What's the service history for my car?"
- "What service do you recommend for my vehicle?"
- "Are there any promotions available for my car?"
- "I'd like to schedule an oil change appointment"

## Development

### Adding New Tools

To add new capabilities to the chatbot:

1. Create a new tool function in `api/app/utils/langgraph/apex_tools.py`:
```python
@tool("your_tool_name")
def your_tool_function(param1: str, session_id: str = "") -> Dict[str, Any]:
    """Tool description for the LLM."""
    # Implementation
    return {"result": "data"}
```

2. Add the tool to the `APEX_TOOLS` list in the same file

3. Update the system prompt in `api/app/config.py` to mention the new capability

### Modifying the UI

The UI is built with UI5 Web Components. Key files:
- `ui/src/pages/apex-chat/apex-chat.js` - Main chat interface logic
- `ui/src/pages/apex-chat/apex-chat.html` - Chat interface template
- `ui/src/pages/apex-chat/apex-chat.css` - Chat interface styles
- `ui/src/services/api.js` - API client service

## Troubleshooting

### Backend Issues

**Data not loading:**
- Verify CSV files are in the correct directory
- Check file paths in `TABLES_PATH` environment variable
- Review application logs for data loading errors

**API authentication failures:**
- Verify `API_KEY` is set correctly
- Check that the `X-API-Key` header is included in requests

**LLM errors:**
- Verify `OPENAI_API_KEY` is set correctly
- Check OpenAI API quota and rate limits
- Review model name configuration

### Frontend Issues

**API connection errors:**
- Verify `VITE_API_BASE_URL` points to the correct backend URL
- Check CORS configuration in the backend
- Ensure API key is configured correctly

**Streaming not working:**
- Check browser console for errors
- Verify the backend streaming endpoint is accessible
- Review network tab for NDJSON responses

## Logging

The application uses Python's logging module. Log levels can be configured in `app/main.py`. Logs include:
- Application startup/shutdown events
- Data loading status
- API request/response information
- Error details

View logs in Cloud Foundry:
```bash
cf logs apex-api --recent
cf logs postsale-chatbot --recent
```