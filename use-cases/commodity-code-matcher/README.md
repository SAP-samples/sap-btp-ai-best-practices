# Deployment Guide

This directory contains the deployment configuration and scripts for deploying the Commodity Code Pipeline to SAP Cloud Foundry.

## Overview

The deployment consists of two applications:

1. **API Application** (`commodity-code-pipeline-api`): FastAPI backend service that provides document extraction and reference-code matching functionality
2. **UI Application** (`commodity-code-pipeline-ui`): Streamlit frontend application that provides a user interface for interacting with the API

Both applications are configured to run on SAP Cloud Foundry and communicate with each other using secure API keys.

## Prerequisites

Before deploying, ensure you have:

- SAP Cloud Foundry CLI (`cf`) installed and configured
- Access to the target Cloud Foundry space (eu10-004)
- Appropriate permissions to deploy applications
- Python 3.x installed locally (for local testing)

## Folder Structure

```
anonymized/
├── manifest.yaml          # Cloud Foundry deployment manifest
├── deploy.sh              # Automated deployment script
├── mkdocs.yml            # MkDocs configuration (for documentation)
├── api/                  # FastAPI backend application
│   ├── app/              # Application code
│   │   ├── main.py       # FastAPI application entry point
│   │   ├── models/       # Data models
│   │   ├── routers/      # API route handlers
│   │   └── services/    # Business logic services
│   ├── .cfignore         # Excludes local seed data from the CF artifact
│   ├── outputs/          # Generated output files
│   ├── scripts/          # Synthetic data generation and HANA load tooling
│   └── requirements.txt  # Python dependencies
└── ui/                   # Streamlit frontend application
    ├── streamlit_app.py  # Streamlit application entry point
    ├── src/              # Source code
    ├── static/           # Static assets (fonts, images, styles)
    └── requirements.txt  # Python dependencies
```

## LLM Extraction Features

### Vendor Name Extraction

When using `llm_extraction=True` (default behavior), the system extracts vendor company names with enhanced accuracy:

- **Combined Text + Image Analysis**: The LLM processes both the full document text AND the first page as an image in a single call
- **Logo Recognition**: Vendor information is extracted from logos and letterheads on the first page that may not be captured in text extraction
- **Export Field**: Results are exported to Excel as the `header_vendorName` column
- **Optimized Performance**: Single LLM call per document reduces costs and improves speed

### Extracted Fields

The extraction schema has been optimized to eliminate redundant fields and reduce token usage:

**Header Fields** (9 fields):
- `documentDate` - Document/invoice date
- `deliveryDate` - Delivery or due date
- `senderAddress` - Sender address
- `vendorName` - Vendor/company name (extracted from text + first page image)
- `receiverID` - Receiver identifier
- `shipToName` - Ship-to recipient name
- `shipToAddress` - Ship-to address
- `currencyCode` - Currency code (header-level)
- `netAmount` - Total net amount

**Line Item Fields** (7 fields):
- `description` - Item description
- `netAmount` - Line item net amount
- `quantity` - Item quantity
- `unitPrice` - Unit price
- `materialNumber` - Material/product number
- `itemNumber` - Line item number
- `usageSummary` - Semantic summary of item's business purpose

### Excel Output

The Excel workbook produced by the API includes:
- Document metadata fields (prefixed with `header_`)
- Line item details
- Reference code matches (Top 5 with descriptions)
- LLM verification results (when `llm_verify=True`)

All schema fields are exported to ensure efficient token usage - no fields are extracted without being used in the final output.

### Reference Data Runtime

The deployment runtime no longer reads local Excel/CSV reference files from the Cloud Foundry container.

- Synthetic reference datasets are generated offline with `api/scripts/generate_and_load_reference_data.py`
- The generated datasets are loaded into HANA tables
- The API validates a shared `DATA_VERSION` across the three HANA tables before processing requests
- `.cfignore` prevents the local seed files, generated outputs, and scripts from being shipped with the API app

### Reference Data Layout

This repository retains only anonymized reference assets.

- Synthetic reference datasets live under `generated_reference_data/`
- Original source workbooks and CSV seed files are not included
- Any sample documents and outputs kept in the repo are anonymized examples only

## Deployment Process

### Automated Deployment

The easiest way to deploy is using the provided deployment script:

  ```bash
  cd anonymized
  chmod +x deploy.sh
  ./deploy.sh
  ```

The script will:
1. Generate a secure, random 256-bit API key
2. Deploy both applications to Cloud Foundry using the generated key
3. Configure the applications with the necessary environment variables

### Manual Deployment

If you prefer to deploy manually or need more control:

1. **Generate an API key** (optional, if not using the script):
   ```bash
   API_KEY=$(openssl rand -hex 32)
   ```

2. **Deploy using Cloud Foundry CLI**:
   ```bash
   cf push --var api_key="$API_KEY"
   ```

   Or if you want to use a specific API key:
   ```bash
   cf push --var api_key="your-api-key-here"
   ```

### Deployment Configuration

The `manifest.yaml` file contains the deployment configuration:

- **Memory**: 512M for API application, 256M for UI application
- **Disk Quota**: 1G for each application
- **Buildpack**: Python buildpack
- **Routes**: 
  - API: `commodity-code-pipeline-api.cfapps.eu10-004.hana.ondemand.com`
  - UI: `commodity-code-pipeline.cfapps.eu10-004.hana.ondemand.com`

## Environment Variables

### API Application

The API application uses the following environment variables:

- `PYTHONPATH`: Set to "." for proper module resolution
- `ALLOWED_ORIGIN`: CORS origin for the UI application (automatically set to UI route)
- `APP_ENV`: Set to "production" for production deployments
- `API_KEY`: Secure API key for authentication (generated automatically by deploy.sh)
- `hana_address`, `hana_port`, `hana_user`, `hana_password`, `hana_encrypt`: HANA connectivity for synthetic reference tables
- `HANA_SCHEMA`, `HANA_REFERENCE_DATA_VERSION`, `HANA_*_TABLE`: Optional schema/table overrides for HANA reference data

### UI Application

The UI application uses the following environment variables:

- `PYTHONPATH`: Set to "." for proper module resolution
- `API_BASE_URL`: Base URL of the API application (automatically set to API route)
- `API_KEY`: Secure API key for authentication (must match API application key)

## Post-Deployment

After successful deployment:

1. **Verify API Health**: Visit `https://commodity-code-pipeline-api.cfapps.eu10-004.hana.ondemand.com/api/health` to check if the API is running

2. **Access UI**: Visit `https://commodity-code-pipeline.cfapps.eu10-004.hana.ondemand.com` to access the Streamlit application

3. **Check Logs**: Monitor application logs using:
   ```bash
   cf logs commodity-code-pipeline-api --recent
   cf logs commodity-code-pipeline-ui --recent
   ```

4. **Validate branding cleanup**:
   ```bash
   ./verify_anonymization.sh
   ```

## Troubleshooting

### Common Issues

1. **Deployment Fails with "Not logged in"**
   - Solution: Log in to Cloud Foundry first using `cf login`

2. **API Key Mismatch**
   - Ensure both applications use the same API key
   - The deploy.sh script handles this automatically

3. **CORS Errors**
   - Verify that `ALLOWED_ORIGIN` in the API matches the UI route
   - Check that `APP_ENV` is set to "production"

4. **Module Import Errors**
   - Ensure `PYTHONPATH` is set correctly in the manifest
   - Verify that all dependencies are listed in requirements.txt

5. **Memory or Disk Quota Exceeded**
   - Adjust memory and disk_quota values in manifest.yaml
   - Ensure your Cloud Foundry space has sufficient quota

### Viewing Application Status

Check application status:
```bash
cf apps
```

View detailed information about an application:
```bash
cf app commodity-code-pipeline-api
cf app commodity-code-pipeline-ui
```

## Updating Applications

To update an application after making changes:

1. Make your code changes in the `api/` or `ui/` directories
2. Run the deployment script again:
   ```bash
   ./deploy.sh
   ```

Cloud Foundry will detect changes and redeploy the applications automatically.

## Security Notes

- The API key is generated randomly for each deployment
- Both applications must share the same API key for communication
- The API key is injected during deployment and stored as an environment variable
- Never commit API keys to version control
- The deploy.sh script generates a new key for each deployment

## Additional Resources

- [SAP Cloud Foundry Documentation](https://help.sap.com/docs/btp/sap-business-technology-platform/cloud-foundry-environment)
- [Cloud Foundry CLI Documentation](https://docs.cloudfoundry.org/cf-cli/)
