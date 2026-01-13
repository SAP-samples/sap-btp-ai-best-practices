# Sales Forecast Dashboard

A modern sales forecast dashboard built with UI5 Web Components and FastAPI, featuring an interactive map, time series charts, and SHAP feature importance visualization.

## Features

- **Interactive Map**: Visualize DMAs and individual stores with Leaflet
- **Dynamic Marker Sizing**: Markers sized based on predicted sales
- **Zoom-dependent Views**: DMA aggregation at low zoom, individual stores at high zoom
- **Time Series Charts**: Plotly charts with p50/p90 confidence bands
- **SHAP Analysis**: Feature importance visualization for sales predictions
- **AI Chatbot**: Mocked AI assistant for future integration
- **SAP Fiori Design**: UI5 Web Components for consistent SAP look and feel

## Project Structure

```
sales-forecast-dashboard/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py        # FastAPI entry point
│   │   ├── security.py    # API key authentication
│   │   ├── models/        # Pydantic models
│   │   ├── routers/       # API endpoints
│   │   ├── services/      # Business logic
│   │   └── data/          # JSON data files
│   └── requirements.txt
├── ui/                     # Vite + UI5 frontend
│   ├── index.html         # Main shell
│   ├── package.json
│   └── src/
│       ├── main.js        # App initialization
│       ├── config/        # Route configuration
│       ├── modules/       # Router and navigation
│       ├── services/      # API service layer
│       └── pages/         # Page components
├── manifest.yaml          # Cloud Foundry deployment
└── deploy.sh              # Deployment script
```

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and set API_KEY

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd ui

# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env and set VITE_API_KEY (must match backend API_KEY)

# Run development server
npm run dev
```

The frontend will be available at http://localhost:5173

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/dma` | GET | Get all DMAs summary |
| `/api/dma/{name}` | GET | Get specific DMA |
| `/api/stores` | GET | Get all stores |
| `/api/stores?dma={name}` | GET | Get stores by DMA |
| `/api/stores/{id}` | GET | Get specific store |
| `/api/timeseries/store/{id}` | GET | Store timeseries + SHAP |
| `/api/timeseries/dma/{name}` | GET | DMA timeseries |

All endpoints except `/api/health` require the `X-API-Key` header.

## Deployment to Cloud Foundry

1. Login to Cloud Foundry:
   ```bash
   cf login -a <api-endpoint>
   ```

2. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

The script will:
- Generate a secure random API key
- Deploy both API and UI applications
- Configure environment variables

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **uvicorn**: ASGI server

### Frontend
- **UI5 Web Components**: SAP Fiori design components
- **Vite**: Build tool and dev server
- **page.js**: Client-side routing
- **Leaflet**: Interactive maps
- **Plotly**: Data visualization

## License

Proprietary - Internal Use Only
