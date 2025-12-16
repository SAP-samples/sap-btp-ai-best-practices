# Sales Order Anomaly Detection

A comprehensive machine learning application for detecting anomalies in sales orders, built with FastAPI and SAP UI5 Web Components. This application provides real-time anomaly detection, explainability features, and an intuitive dashboard for analyzing sales order data.

## Overview

This application helps identify anomalous sales orders using machine learning models. It provides:

- **Real-time anomaly detection** using trained ML models
- **Interactive dashboard** with monthly views and calendar visualization
- **Order analysis** with detailed explanations
- **Model fine-tuning** interface for customizing detection parameters
- **Explainability features** using SHAP values and AI-generated explanations
- **Batch processing** capabilities for large datasets

## Architecture

The application consists of two main components:

### Backend (FastAPI)
- **Location**: `backend/`
- **Framework**: FastAPI (Python)
- **ML Models**: scikit-learn (Isolation Forest), with optional SAP HANA integration
- **Features**: RESTful API, model training pipeline, explainability services

### Frontend (UI5 Web Components)
- **Location**: `frontend/`
- **Framework**: SAP UI5 Web Components with Vite
- **Features**: Router-based page system, responsive dashboard, interactive visualizations

## Features

### 1. Dashboard
- Monthly overview of sales orders and anomalies
- Calendar visualization showing daily anomaly counts
- Summary metrics (total orders, anomaly rate, values)
- Filtering by year and month

### 2. Order Analysis
- Detailed view of individual sales orders
- SHAP-based feature importance explanations
- AI-generated natural language explanations
- Binary classification for quick anomaly assessment

### 3. Fine-Tuning
- Configure model parameters (contamination rate, estimators)
- Feature selection interface
- Model training with custom configurations
- Model persistence and versioning

### 4. Explainability
- **SHAP Explanations**: Feature-level contribution analysis
- **AI Explanations**: Natural language descriptions of anomalies
- **Fallback Explanations**: Rule-based explanations when ML explanations are unavailable

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- SAP BTP account (for Cloud Foundry deployment)
- Cloud Foundry CLI (for deployment)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp ../.env.example .env
# Edit .env with your configuration
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Application

### Development Mode

#### Backend

From the `backend` directory:
```bash
# Activate virtual environment if not already active
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the FastAPI server
cd app
python main.py
```

The API will be available at `http://localhost:8000`

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Frontend

From the `frontend` directory:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Production Mode

Build the frontend for production:
```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

## Project Structure

```
anomaly-detection/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── routers/                # API route handlers
│   │   │   ├── dashboard.py        # Dashboard endpoints
│   │   │   ├── orders.py           # Order analysis endpoints
│   │   │   ├── anomaly.py          # Anomaly detection endpoints
│   │   │   └── fine_tuning.py     # Model training endpoints
│   │   ├── services/               # Business logic services
│   │   │   ├── model_service.py   # Model loading and management
│   │   │   ├── data_loader.py     # Dataset loading
│   │   │   ├── shap_explanations.py # SHAP explanation generation
│   │   │   └── ai_classification.py # AI explanation generation
│   │   ├── models/                 # ML model implementations
│   │   │   ├── sklearn_model.py   # scikit-learn Isolation Forest
│   │   │   ├── hana_model.py      # SAP HANA PAL models
│   │   │   └── stratified_model.py # Customer-stratified models
│   │   ├── explainability/         # Explanation generators
│   │   ├── evaluation/             # Model evaluation metrics
│   │   ├── data/                   # Data processing and features
│   │   ├── training_pipeline.py   # Model training orchestration
│   │   └── datasets/               # Training datasets
│   ├── requirements.txt            # Python dependencies
│   └── results/                    # Trained models and results
├── frontend/
│   ├── src/
│   │   ├── pages/                  # Page components
│   │   │   ├── dashboard/         # Dashboard page
│   │   │   ├── orders/            # Order analysis page
│   │   │   └── fine-tuning/       # Fine-tuning page
│   │   ├── services/              # Frontend services
│   │   │   └── api.js             # API client
│   │   ├── modules/               # Core modules
│   │   │   └── router.js          # Client-side routing
│   │   └── config/                # Configuration
│   │       └── routes.js          # Route definitions
│   ├── package.json               # Node.js dependencies
│   └── vite.config.js            # Vite configuration
├── manifest.yaml                  # Cloud Foundry deployment manifest
└── deploy.sh                     # Deployment script
```

## API Endpoints

### Dashboard
- `GET /api/dashboard/summary` - Get summary statistics
- `GET /api/dashboard/calendar` - Get calendar data for visualization

### Orders
- `GET /api/orders/random-anomalous` - Get a random anomalous order
- `GET /api/orders/{doc_number}/{doc_item}` - Get order details
- `POST /api/orders/{doc_number}/{doc_item}/explain` - Generate AI explanation

### Anomaly Detection
- `POST /api/anomaly/predict` - Predict anomaly for a sample
- `POST /api/anomaly/explain-binary` - Get binary classification explanation

### Fine-Tuning
- `POST /api/fine-tuning/train` - Train a new model
- `GET /api/fine-tuning/status` - Get training status
- `GET /api/fine-tuning/models` - List available models

### Health Check
- `GET /api/health` - Health check endpoint

## Model Training

The application supports training models with various configurations:

```bash
cd backend/app
python training_pipeline.py \
    --backend sklearn \
    --contamination 0.05 \
    --shap \
    --n-estimators 400 \
    --max-samples 2048
```

### Training Parameters

- `--backend`: Model backend (`sklearn` or `hana`)
- `--contamination`: Expected proportion of anomalies (0.0-1.0 or `auto`)
- `--shap`: Enable SHAP explanation generation
- `--customer-stratified`: Use customer-stratified models
- `--n-estimators`: Number of trees in Isolation Forest
- `--max-samples`: Maximum samples per tree

### Feature Selection

Features can be selected via environment variable:
```bash
export SELECTED_FEATURES="feature1,feature2,feature3"
python training_pipeline.py
```

## Deployment

### SAP BTP Cloud Foundry

The application is configured for deployment to SAP BTP Cloud Foundry.

1. **Login to Cloud Foundry**:
```bash
cf login -a <api-endpoint> -u <username> -p <password>
```

2. **Deploy using the deployment script**:
```bash
./deploy.sh
```

The script will:
- Generate a secure API key
- Deploy both backend and frontend applications
- Configure routes and environment variables

### Manual Deployment

1. **Deploy backend**:
```bash
cd backend
cf push --var api_key="<your-api-key>"
```

2. **Build and deploy frontend**:
```bash
cd frontend
npm run build
cd dist
cf push
```

### Environment Variables

Configure these environment variables for production:

- `APP_ENV`: Application environment (`production` or `development`)
- `API_KEY`: API key for securing endpoints
- `ALLOWED_ORIGIN`: Allowed CORS origin
- `API_BASE_URL`: Base URL for API endpoints
- `PYTHONPATH`: Python path for imports

## Configuration

### Backend Configuration

Configuration files are located in `backend/app/config/`:
- `settings.py`: Application settings
- `cli.py`: Command-line argument parsing

### Frontend Configuration

- `frontend/src/config/routes.js`: Route definitions
- `frontend/src/services/api.js`: API client configuration

## Data Format

The application expects CSV data with the following structure:
- Sales order information (document number, item, dates)
- Customer information
- Product information
- Financial data (amounts, quantities)
- Derived features for anomaly detection

See `backend/app/datasets/all_data.csv` for an example format.

## Explainability

### SHAP Explanations

SHAP (SHapley Additive exPlanations) provides feature-level contributions to anomaly predictions. The application uses TreeExplainer for Isolation Forest models.

### AI Explanations

AI-generated explanations use natural language to describe why an order is flagged as anomalous, making the results more accessible to business users.

### Fallback Explanations

When ML-based explanations are unavailable, the system falls back to rule-based explanations using business rules.

## Troubleshooting

### Backend Issues

**Model not loading:**
- Check that model files exist in `backend/results/`
- Verify model compatibility with current code version
- Check logs for loading errors

**API errors:**
- Verify environment variables are set correctly
- Check that dataset files are accessible
- Review FastAPI logs for detailed error messages

### Frontend Issues

**Pages not loading:**
- Verify all page files exist (HTML, CSS, JS)
- Check browser console for JavaScript errors
- Ensure routes are properly configured in `routes.js`

**API calls failing:**
- Verify API base URL in `api.js`
- Check CORS configuration in backend
- Ensure backend is running and accessible

### Deployment Issues

**Build failures:**
- Verify all dependencies are in `requirements.txt` and `package.json`
- Check buildpack compatibility
- Review Cloud Foundry logs: `cf logs <app-name> --recent`

**Runtime errors:**
- Check environment variables are set correctly
- Verify file paths are correct for Cloud Foundry
- Review application logs

## Development

### Adding New Features

1. **Backend**: Add routes in `routers/`, services in `services/`
2. **Frontend**: Create new pages in `pages/`, add routes in `config/routes.js`
3. **Models**: Extend base model classes in `models/`

### Code Style

- Python: Follow PEP 8 guidelines
- JavaScript: Use ES6+ syntax, follow UI5 Web Components patterns
- Comments: Add comments explaining complex logic

## Dependencies

### Backend
- FastAPI: Web framework
- scikit-learn: Machine learning models
- pandas: Data manipulation
- SHAP: Explainability
- SAP AI SDK: SAP HANA integration

### Frontend
- SAP UI5 Web Components: UI framework
- Vite: Build tool and dev server
- Chart.js: Data visualization
- page.js: Client-side routing

