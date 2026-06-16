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

## Generated Dashboard Data

The dashboard JSON files are generated artifacts and should not be committed with customer or sample data. They are intentionally ignored by this use case:

- `api/app/data/stores.json`
- `api/app/data/dma_summary.json`
- `api/app/data/timeseries/*.json`

Generate them from your own SAP HANA tables before running the dashboard with real data.

### Expected HANA Tables

The dashboard data generator requires these HANA tables in the schema configured by `HANA_SCHEMA`:

| Table | Required For | Notes |
|-------|--------------|-------|
| `PREDICTIONS_BM` | Dashboard summaries, B&M store time series, DMA time series | Must contain 2024 and 2025 prediction rows. |
| `PREDICTIONS_WEB` | WEB store time series and WEB AUV fields | Must contain 2024 and 2025 prediction rows. |
| `PROFIT_CENTER` | Store names, locations, metadata, and DMA/store joins | Must include latitude and longitude for mapped stores. |
| `CALENDAR` | Runtime fiscal week/month/quarter labels in API responses | Used by the API when serving time-series data. |

The broader agent and scenario workflows may also use:

| Table | Purpose |
|-------|---------|
| `MODEL_B` | Historical feature matrix for scenario planning and what-if forecasts. |
| `AWARENESS_CONSIDERATION` | Brand awareness and consideration signals. |
| `YOUGOV_DMA_MAP` | Market-to-DMA mapping for awareness data. |
| `BUDGET_MARKETING` | Marketing budget inputs. |
| `GA_DMA` | Geographic/DMA mapping support. |

### Generate JSON Files

1. Configure HANA credentials in `api/.env`:

   ```bash
   cd api
   cp .env.example .env
   ```

   Set these values:

   ```bash
   hana_address=<your-hana-host>
   hana_port=443
   hana_user=<your-user>
   hana_password=<your-password>
   hana_encrypt=true
   HANA_SCHEMA=<schema-containing-the-tables>
   ```

2. Install API dependencies and validate the HANA connection:

   ```bash
   pip install -r requirements.txt
   python -m app.scripts.regenerate_dashboard_data --dry-run
   ```

3. Generate the dashboard files:

   ```bash
   python -m app.scripts.regenerate_dashboard_data --verbose
   ```

The script writes `stores.json`, `dma_summary.json`, and the `timeseries/` JSON files under `api/app/data/`. Use `--skip-timeseries` only when you want to refresh `stores.json` and `dma_summary.json` without rebuilding the per-store and per-DMA time-series files.

If you need to create the HANA tables from local files first, use the companion HANA upload utility from the source project (`hana_tables/upload_tables.py`). Place CSV or Parquet files in `hana_tables/data_anonymized/` for anonymized data, or `hana_tables/data/` for original/private data, using these lowercase file names: `profit_center`, `predictions_bm`, `predictions_web`, `model_b`, `awareness_consideration`, `yougov_dma_map`, `budget_marketing`, `calendar`, and `ga_dma`. Then run the upload utility with the same HANA environment variables, for example `python upload_tables.py --source anonymized --tables profit_center predictions_bm predictions_web calendar`.

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
