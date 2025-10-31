# Procurement Assistant API

FastAPI implementation of the Core Optimization Pipeline APIs for the Procurement Assistant application.

## Overview

This API provides programmatic access to:
- Vendor evaluation with tariff impact analysis
- Procurement optimization using linear programming
- Policy comparison between historical and optimized allocations
- Configuration management for costs, tariffs, and metric weights

## Installation

1. Ensure you have the required dependencies installed:
```bash
cd resources
pip install -r requirements.txt
```

## Running the API

### Development Server

```bash
cd resources/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using Python directly:
```bash
cd resources/api
python main.py
```

### Production Server

For production, use multiple workers:
```bash
cd resources/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- Alternative API documentation: http://localhost:8000/redoc
- OpenAPI schema: http://localhost:8000/openapi.json

## Testing

Run the basic test script:
```bash
cd resources/api
python test_api.py
```

## API Endpoints

### Core Optimization Pipeline

- **POST** `/api/optimization/pipeline` - Run complete optimization pipeline
- **POST** `/api/optimization/evaluate-vendors` - Evaluate and rank vendors
- **POST** `/api/optimization/optimize-allocation` - Optimize procurement allocation
- **POST** `/api/optimization/compare-policies` - Compare historical vs optimized policies

### Job Management

- **GET** `/api/jobs/` - List all jobs
- **GET** `/api/jobs/{job_id}` - Get job status
- **GET** `/api/jobs/{job_id}/download` - Download job results
- **DELETE** `/api/jobs/{job_id}` - Cancel a job
- **POST** `/api/jobs/cleanup` - Clean up expired jobs

### Configuration Management

- **GET** `/api/configuration/profiles/{profile_id}/config` - Get all configuration
- **GET** `/api/configuration/profiles/{profile_id}/config/costs` - Get costs configuration
- **PUT** `/api/configuration/profiles/{profile_id}/config/costs` - Update costs configuration
- **GET** `/api/configuration/profiles/{profile_id}/config/tariffs` - Get tariff configuration
- **PUT** `/api/configuration/profiles/{profile_id}/config/tariffs` - Update tariff configuration
- **GET** `/api/configuration/profiles/{profile_id}/config/metric-weights` - Get metric weights
- **PUT** `/api/configuration/profiles/{profile_id}/config/metric-weights` - Update metric weights

### Health Checks

- **GET** `/health` - Main application health check
- **GET** `/api/optimization/health` - Optimization service health
- **GET** `/api/jobs/health` - Jobs service health
- **GET** `/api/configuration/health` - Configuration service health

## Architecture

The API is structured in layers:

```
api/
├── main.py              # FastAPI application entry point
├── config.py            # API configuration and settings
├── models/              # Pydantic models for requests/responses
│   ├── requests.py      # Request models
│   ├── responses.py     # Response models
│   └── jobs.py          # Job tracking models
├── routers/             # API route handlers
│   ├── optimization.py  # Optimization endpoints
│   ├── jobs.py          # Job management endpoints
│   └── configuration.py # Configuration endpoints
├── services/            # Business logic layer
│   ├── vendor_evaluator.py  # Vendor evaluation wrapper
│   ├── optimizer.py         # Optimization wrapper
│   ├── comparator.py        # Comparison wrapper
│   ├── pipeline_runner.py   # Pipeline orchestration
│   └── job_manager.py       # Job lifecycle management
├── utils/               # Utility modules
│   └── file_manager.py  # File operations utilities
└── background/          # Background task workers
```

## Key Features

1. **Asynchronous Processing**: Large operations run asynchronously with job tracking
2. **Profile Support**: All operations are profile-scoped for multi-tenant usage
3. **Flexible Configuration**: Runtime configuration of costs, tariffs, and weights
4. **Job Management**: Track long-running operations with progress updates
5. **Error Handling**: Comprehensive error handling with detailed error responses
6. **CORS Support**: Configurable CORS for web application integration

## Environment Variables

- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CORS_ORIGINS`: Comma-separated list of allowed origins

## Notes

- The API wraps existing Python scripts for compatibility
- Large datasets automatically trigger async processing
- Results are stored temporarily and cleaned up after TTL expiration
- All monetary values are in USD