# RFQx UI - Streamlit Dashboard

This folder contains the Streamlit-based user interface for the RFQx application, providing an interactive dashboard for Request for Quotation (RFQ) analysis and supplier comparison.

## Overview

The UI is built using Streamlit and provides a comprehensive interface for:
- Supplier quotation comparison and analysis
- Interactive chatbot for RFQ queries
- Detailed cost breakdowns and risk assessments
- Part-by-part comparison capabilities
- Real-time data visualization and metrics

## Project Structure

```
ui/
├── streamlit_app.py          # Main application entry point
├── requirements.txt          # Python dependencies
├── dashboard_components.py   # Reusable UI components
├── src/                      # Source code directory
│   ├── Offers_Comparison.py # Main dashboard page
│   ├── api_client.py        # API communication layer
│   ├── data_loader.py       # Data loading utilities
│   ├── utils.py             # General utilities
│   └── pages/               # Additional pages
│       ├── 1_Detailed_Comparison.py
│       ├── 2_RFQ_Chatbot.py
│       ├── 3_Detailed_Dashboard.py
│       ├── 4_Part_Comparison.py
│       ├── 5_Cost_Breakdown.py
│       └── 6_Risk_Assessment.py
└── static/                  # Static assets
    ├── font/                # Custom fonts
    ├── images/              # Images and logos
    └── styles/              # CSS stylesheets
```

## Key Features

### Main Dashboard (`Offers_Comparison.py`)
- **Supplier Comparison**: Side-by-side analysis of supplier quotations
- **Interactive Metrics**: Real-time calculation of key performance indicators
- **Source Traceability**: Links analysis results back to source documents
- **Multi-page Navigation**: Seamless navigation between different analysis views

### Analysis Pages
- **Detailed Comparison**: Comprehensive supplier comparison with metrics
- **Part Comparison**: Component-level analysis and comparison
- **Cost Breakdown**: Detailed cost analysis and visualization
- **Risk Assessment**: Risk evaluation and mitigation strategies

## Technical Architecture

### Frontend Framework
- **Streamlit**: Modern Python web framework for data applications
- **Plotly**: Interactive data visualization
- **Pandas**: Data manipulation and analysis
- **Custom CSS**: SAP-inspired styling with 72 font family

### API Integration
- **RESTful Communication**: HTTP-based API client for backend services
- **Environment Configuration**: Flexible API endpoint configuration
- **Authentication**: API key-based authentication support

### Data Management
- **Caching**: Streamlit caching for performance optimization
- **Session State**: Persistent user session management
- **Data Loading**: Efficient data loading and processing utilities

## Configuration

### Environment Variables
The application supports the following environment variables:

- `API_BASE_URL`: Backend API base URL (default: http://localhost:8000)
- `API_KEY`: Authentication key for API access
- `LLM_MODEL`: Primary LLM model for analysis (default: gemini-2.5-pro)
- `COMPARATOR_MODEL`: Model for comparison tasks (default: gemini-2.5-pro)
- `SUPPLIER1_ID`: Default first supplier ID
- `SUPPLIER2_ID`: Default second supplier ID
- `PORT`: Application port (default: 8501)

### Core Part Categories
The application supports configurable part categories for analysis:
- Clutch Disk
- Clutch Cover
- Releaser

## Installation and Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation Steps

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file in the ui directory with required variables:
   ```env
   API_BASE_URL=http://localhost:8000
   API_KEY=your_api_key_here
   LLM_MODEL=gemini-2.5-pro
   ```

4. **Run the application**:
   ```bash
   python streamlit_app.py
   ```

## Deployment

### Cloud Foundry Deployment
The application is configured for Cloud Foundry deployment with:
- Automatic port binding
- Environment variable handling
- Production-ready configuration

## Dependencies

### Core Dependencies
- `streamlit`: Web application framework
- `requests`: HTTP client for API communication
- `python-dotenv`: Environment variable management
- `plotly`: Interactive data visualization
- `pandas`: Data manipulation and analysis

### Optional Dependencies
- Custom fonts and styling for SAP branding
- Image processing libraries for logo display

## Usage

### Starting the Application
1. Ensure the backend API is running
2. Configure environment variables
3. Run `python streamlit_app.py`
4. Access the application at `http://localhost:8501`

### Navigation
- **Main Dashboard**: Default landing page with supplier comparison
- **Chatbot**: AI-powered query interface
- **Detailed Pages**: Specialized analysis views accessible via sidebar

### Key Interactions
- **Supplier Selection**: Choose suppliers for comparison
- **Metric Analysis**: View calculated KPIs and metrics
- **Source Navigation**: Click on source references to view original documents
- **Chat Interface**: Ask questions about RFQ data and analysis

## Troubleshooting

### Common Issues
1. **API Connection Errors**: Verify `API_BASE_URL` and `API_KEY` configuration
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Port Conflicts**: Change `PORT` environment variable
4. **Font Loading**: Ensure font files are in `static/font/` directory

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```
