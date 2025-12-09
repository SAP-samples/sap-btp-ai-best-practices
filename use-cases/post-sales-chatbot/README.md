# Apex Automotive Services Post-Sales Chatbot

An AI-powered customer service chatbot for automotive dealerships that provides vehicle owners with conversational access to their service history, vehicle information, promotions, and appointment scheduling. Built with a ReAct (Reasoning + Acting) agent pattern using LangGraph and OpenAI's GPT-4o model.

## Overview

This application enables automotive customers to interact naturally with their dealership's service center through a conversational interface. Customers can identify themselves, query their vehicles, review service history, get maintenance recommendations, discover active promotions, and request service appointments.

## Architecture

The application follows a two-tier architecture deployed on SAP BTP Cloud Foundry:

```
+-------------------+         +-------------------+
|                   |  HTTP   |                   |
|   Frontend (UI)   +-------->+   Backend (API)   |
|   Node.js/Vite    |         |   Python/FastAPI  |
|   SAP UI5 WebComp |         |   LangGraph Agent |
|                   |         |                   |
+-------------------+         +--------+----------+
                                       |
                                       v
                              +-------------------+
                              |  SAP Gen AI Hub   |
                              |     (GPT-4o)      |
                              +-------------------+
```

### Backend (API)

- **Framework:** FastAPI with Uvicorn ASGI server
- **LLM Integration:** LangChain + LangGraph for ReAct agent orchestration
- **Model Provider:** SAP Generative AI Hub (OpenAI proxy)
- **Data Storage:** CSV files loaded into Pandas DataFrames

### Frontend (UI)

- **Build Tool:** Vite
- **UI Framework:** SAP UI5 Web Components (Core, Fiori, AI)
- **Features:** Real-time streaming, markdown rendering, responsive design

## Project Structure

```
post-sales-chatbot/
├── api/                          # Backend API
│   ├── app/
│   │   ├── main.py               # FastAPI application entry
│   │   ├── config.py             # Configuration and prompts
│   │   ├── security.py           # API key authentication
│   │   ├── routers/
│   │   │   └── apex_chat.py      # Chat endpoints
│   │   ├── models/               # Pydantic models
│   │   │   ├── chat.py
│   │   │   ├── session.py
│   │   │   └── chat_history.py
│   │   ├── services/
│   │   │   ├── data_handler.py   # CSV data loading
│   │   │   └── session_manager.py
│   │   └── utils/
│   │       └── langgraph/
│   │           ├── common.py     # LLM factory
│   │           └── apex_tools.py # Agent tools
│   ├── data/
│   │   └── new_tables/           # CSV data files
│   └── requirements.txt
├── ui/                           # Frontend application
│   ├── src/
│   │   ├── main.js               # App initialization
│   │   ├── config/
│   │   │   └── routes.js
│   │   ├── modules/
│   │   │   ├── router.js
│   │   │   └── navigation.js
│   │   ├── services/
│   │   │   └── api.js            # API client
│   │   └── pages/
│   │       └── apex-chat/        # Chat interface
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── manifest.yaml                 # Cloud Foundry deployment
└── deploy.sh                     # Deployment script
```

## Features

### Conversational Capabilities

The chatbot provides eight specialized tools through its ReAct agent:

| Tool | Description |
|------|-------------|
| `find_client` | Identify customer by email, phone, VIN, license plate, or name |
| `list_client_vehicles` | List all vehicles registered to the customer |
| `select_vehicle` | Select a specific vehicle for subsequent queries |
| `get_vehicle_details` | Retrieve detailed vehicle information (make, model, year, mileage, VIN) |
| `get_service_history` | Get the last 3 service visits with dates, operations, and prices |
| `get_next_service_recommendation` | Recommend service based on mileage (10,000 km milestones) |
| `get_promotions` | Find active promotions for the customer's vehicle |
| `schedule_appointment` | Request appointment scheduling with date/time preferences |

### Real-Time Streaming

The chat interface uses NDJSON streaming to provide real-time feedback:
- Tool execution hints show which operation is in progress
- Responses stream as they are generated
- Markdown formatting is rendered live

### Session Management

- Sessions persist across interactions with automatic cleanup after 60 minutes
- Conversation context is maintained for multi-turn dialogues
- Selected vehicle context carries across queries

## Data Model

The application uses CSV files to store dealership data:

| Table | Description |
|-------|-------------|
| `Cliente.csv` | Customer master data (name, contact info, RFC) |
| `Unidad.csv` | Vehicle records (VIN, make, model, year, mileage) |
| `Servicios.csv` | Service history (order number, dates, service codes) |
| `Operacion.csv` | Service operation descriptions |
| `Kits.csv` | Service packages with pricing |
| `Campanas.csv` | Active promotional campaigns |
| `Materiales.csv` | Parts and materials reference |

## Prerequisites

- Python 3.11+
- Node.js 18+
- Cloud Foundry CLI (for deployment)
- SAP BTP account with:
  - Cloud Foundry environment
  - SAP AI Core with Generative AI Hub access

## Local Development

### Backend Setup

1. Navigate to the API directory:
   ```bash
   cd api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with required environment variables:
   ```env
   AICORE_AUTH_URL=""
   AICORE_CLIENT_ID=""
   AICORE_CLIENT_SECRET=""
   AICORE_BASE_URL=""
   AICORE_RESOURCE_GROUP=""
   API_KEY="<your-api-key>"
   ```

5. Start the development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the UI directory:
   ```bash
   cd ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   VITE_API_KEY="<your-api-key>"
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open http://localhost:5173 in your browser

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/apex/chat` | Send message and receive single response |
| POST | `/api/apex/chat/stream` | Send message with NDJSON streaming response |
| POST | `/api/apex/reset` | Reset conversation and get initial greeting |
| GET | `/api/health` | Health check endpoint |

### Authentication

All API endpoints require an `X-API-Key` header with a valid API key.

### Example Request

```bash
curl -X POST "http://localhost:8000/api/apex/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "I am john.doe@email.com. What vehicles do I have?"}'
```

## Deployment

### Cloud Foundry Deployment

1. Login to Cloud Foundry:
   ```bash
   cf login -a <api-endpoint>
   ```

2. Set required environment variables or use the deployment script:
   ```bash
   export AICORE_AUTH_URL=<your-auth-url>
   export AICORE_CLIENT_ID=<your-client-id>
   export AICORE_CLIENT_SECRET=<your-client-secret>
   export AICORE_RESOURCE_GROUP=<your-resource-group>
   export AICORE_BASE_URL=<your-base-url>
   ```

3. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

   The script will:
   - Generate a random API key if not provided
   - Deploy both API and UI applications
   - Configure CORS and environment variables


## Configuration

### Environment Variables

#### Backend (API)

| Variable | Description | Default |
|----------|-------------|---------|
| `AICORE_AUTH_URL` | SAP AI Core authentication URL | Required |
| `AICORE_CLIENT_ID` | SAP AI Core client ID | Required |
| `AICORE_CLIENT_SECRET` | SAP AI Core client secret | Required |
| `AICORE_RESOURCE_GROUP` | SAP AI Core resource group | Required |
| `AICORE_BASE_URL` | SAP AI Core base URL | Required |
| `LLM_MODEL_NAME` | LLM model name | `gpt-4o` |
| `LLM_TEMPERATURE` | Response randomness (0.0-1.0) | `0.0` |
| `TABLES_PATH` | Path to CSV data files | `data/new_tables` |
| `API_KEY` | API authentication key | Required |
| `APP_ENV` | Environment mode | `development` |
| `ALLOWED_ORIGIN` | CORS allowed origin | `*` (dev) |
| `SESSION_EXPIRY_MINUTES` | Session timeout | `60` |

#### Frontend (UI)

| Variable | Description |
|----------|-------------|
| `VITE_API_BASE_URL` | Backend API URL |
| `VITE_API_KEY` | API authentication key |
| `VITE_APP_HOST` | Application host URL |

## Example Interactions

### Customer Identification
```
User: Hi, I'm john.doe@email.com
Bot: Hello John! I found your account. You have 2 vehicles registered...
```

### Service History Query
```
User: What's the service history for my Ford Explorer?
Bot: Here's the service history for your 2022 Ford Explorer (VIN: 1FMSK8DH...):
     - Dec 1, 2024: Oil Change & Filter - $89.99
     - Sep 15, 2024: Tire Rotation - $45.00
     - Jun 10, 2024: Annual Inspection - $125.00
```

### Service Recommendation
```
User: When should I bring it in for service?
Bot: Based on your current mileage of 47,500 km, I recommend scheduling
     your 50,000 km service soon. This includes...
```

### Appointment Scheduling
```
User: Can I schedule an appointment for next Tuesday at 10am?
Bot: I'd be happy to help schedule your appointment for Tuesday, December 17th
     at 10:00 AM. What service would you like to have performed?
```

## Security Considerations

- API key authentication required for all endpoints
- CORS configured for production origins only
- Session isolation with automatic expiry
- No PII logged or persisted beyond session
- Data access controlled through singleton DataHandler

