# Credit Evaluation Engine

A comprehensive AI-powered credit evaluation system that combines document processing, data validation, and automated credit policy assessment to streamline financial decision-making processes.

## Overview

This system provides end-to-end credit evaluation capabilities, from document extraction to automated policy compliance checking. It processes customer documents, validates cross-document consistency, analyzes payment behavior, and applies configurable credit rules to determine approval recommendations.

## Key Features

### Document Processing & Extraction
- **AI-Powered PDF Processing**: Extract structured data from various document types using SAP Document AI
- **Multi-Document Support**: Handle KYC, CSF, Vendor Comments, CGV, Legal Investigations, and Commercial Investigations
- **Cross-Document Validation**: Automatically verify consistency across multiple documents (RFC, addresses, legal names, etc.)
- **Batch Processing**: Process multiple documents simultaneously for maximum efficiency

### Credit Policy Engine
- **Automated Credit Assessment**: AI-driven credit policy engine with comprehensive rule sets
- **Risk Scoring**: Generate CAL (Credit Assessment Level), C3M (3-month performance), and historical payment scores
- **Multi-Currency Support**: Handle MXN, USD, and EUR credit evaluations
- **Role-Based Authorization**: Different approval limits for analysts and coordinators
- **Use Case Coverage**: Support for new credit, credit updates, and credit exceptions

### Decision Support
- **Approval Recommendations**: Get clear approval/denial recommendations with detailed reasoning
- **Director Escalation**: Automatic identification of cases requiring director-level approval
- **Executive Reporting**: Generate comprehensive credit reports with AI analysis
- **Payment Behavior Analysis**: Historical payment pattern analysis with delinquency tracking

## System Architecture

```
Credit Evaluation Engine/
├── api/                    # FastAPI backend application
│   ├── routers/           # API route handlers
│   │   ├── extraction.py  # Document processing endpoints
│   │   ├── verification.py # Cross-document validation
│   │   ├── credit_policy.py # Credit evaluation engine
│   │   ├── credit_report.py # Report generation
│   │   └── tools/         # Business logic and schemas
│   ├── models.py          # Pydantic data models
│   └── api_server.py      # Main FastAPI application
├── ui/                    # Streamlit frontend application
│   ├── src/              # Frontend source code
│   │   ├── Home.py       # Main dashboard
│   │   ├── pages/        # Application pages
│   │   └── api_client.py # Backend communication
│   └── streamlit_app.py  # Streamlit application entry point
└── docs/                 # Documentation and sample data
```

## API Endpoints

### Document Processing
- `POST /api/extraction/single` - Process single document
- `POST /api/extraction/batch` - Process multiple documents
- `GET /api/extraction/schemas` - Get document schemas

### Verification
- `POST /api/verification/verify` - Cross-document validation

### Credit Evaluation
- `POST /api/credit/evaluate` - Full credit evaluation
- `POST /api/credit/scores` - Payment score calculation
- `POST /api/credit/report` - Generate credit reports

### System
- `GET /api/health` - System health check

## Prerequisites

- Python 3.8 or higher
- SAP Document AI access and credentials
- Cloud Foundry CLI (for deployment)
- Virtual environment support

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
- **`api/.env`**: SAP Document AI credentials and API keys
- **`ui/.env`**: Backend API connection settings

### 2. Backend Setup

```bash
# Navigate to API directory
cd api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn api_server:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000) with interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Frontend Setup

```bash
# Navigate to UI directory (in a new terminal)
cd ui

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
python streamlit_app.py
```

The UI will be available at [http://localhost:8501](http://localhost:8501).

## Usage Workflow

### 1. Credit Creation Process
1. **Document Upload**: Upload required documents (KYC, CSF, Vendor Comments, CGV, Excel files)
2. **AI Extraction**: System extracts key attributes using AI processing
3. **Validation**: Cross-document consistency verification
4. **Data Integration**: Parse Excel files for payment history and credit requests
5. **Credit Evaluation**: Run policy engine with extracted data
6. **Results Review**: Review scores, checks, and recommendations
7. **Report Generation**: Generate executive reports and download results

### 2. Document Processing
- Define custom document types and extraction fields
- Upload and process documents using AI extraction
- View and export structured results

### 3. Results & Reporting
- Review extraction results and validation status
- Download credit evaluation reports as PDF
- Export data in various formats (JSON, CSV)

## Deployment

### SAP BTP Cloud Foundry Deployment

1. **Login to Cloud Foundry**:
```bash
cf login -a <API_ENDPOINT> --sso
```

2. **Update Application Names**:
   - Replace `template-streamlit-fastapi` with your app name in `manifest.yaml`

3. **Deploy**:
```bash
# Make deployment script executable
chmod +x deploy.sh

# Run automated deployment
./deploy.sh
```

The deployment script will:
- Generate secure API keys
- Deploy both UI and API applications
- Configure environment variables
- Provide live application URLs

## Configuration

### Document Schemas
The system supports configurable document schemas for different document types:
- KYC (Know Your Customer)
- CSF (Constancia de Situación Fiscal)
- Vendor Comments
- CGV (Condiciones Generales de Venta)
- Legal Investigations
- Commercial Investigations

### Credit Policy Rules
Credit policy rules are defined in `api/routers/tools/credit_policy_engine.py` and can be customized for:
- Customer group classifications
- Payment scoring thresholds
- Approval limits by role
- Use case-specific requirements

