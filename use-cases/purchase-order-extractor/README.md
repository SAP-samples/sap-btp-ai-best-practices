# Acme Corporation PDF Extractor API

REST API to extract purchase order data from PDF files using SAP Generative AI Hub.

## Features

- Automatic data extraction from purchase order PDFs
- Batch processing of multiple files
- Data export in JSON format
- Integration with SAP Generative AI Hub for intelligent text analysis
- **Interactive Swagger/OpenAPI documentation**

## Extracted Data

- **Customer**: Customer name
- **Date**: Order date
- **Vendor**: Always "Acme Corporation"
- **Products**: List of products with:
  - Product description
  - Quantity
  - Unit price

## API Documentation

### Swagger UI
The API includes interactive Swagger documentation available at:
- **Local**: `http://localhost:5001/swagger/`
- **Cloud Foundry**: `https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/swagger/`

### API Endpoints

#### `GET /api/health/`
API health check verification.

#### `POST /api/extract/purchase-order`
Extract data from an individual PDF file.
- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (PDF file)

#### `POST /api/extract/batch`
Process multiple PDF files.
- **Content-Type**: `multipart/form-data`
- **Parameter**: `files` (multiple PDF files)

#### `POST /api/export/json`
Export purchase order data in structured JSON format.

## Usage Examples from Other Programs

### Python with requests

```python
import requests

# API base URL
BASE_URL = "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api"

# 1. Check API health
def check_health():
    response = requests.get(f"{BASE_URL}/health/")
    return response.json()

# 2. Extract data from a PDF
def extract_pdf(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(f"{BASE_URL}/extract/purchase-order", files=files)
        return response.json()

# 3. Batch processing
def batch_extract(file_paths):
    files = []
    for file_path in file_paths:
        files.append(('files', open(file_path, 'rb')))
    
    response = requests.post(f"{BASE_URL}/extract/batch", files=files)
    
    # Close files
    for _, file in files:
        file.close()
    
    return response.json()

# 4. Export data
def export_data(extracted_data):
    response = requests.post(f"{BASE_URL}/export/json", json=extracted_data)
    return response.json()

# Usage example
if __name__ == "__main__":
    # Check health
    health = check_health()
    print("API Status:", health)
    
    # Extract data from a PDF
    result = extract_pdf("purchase_order.pdf")
    if result['success']:
        print("Customer:", result['data']['customer'])
        print("Date:", result['data']['date'])
        print("Total products:", result['data']['summary']['total_products'])
    else:
        print("Error:", result['message'])
```

### JavaScript/Node.js with axios

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const BASE_URL = 'https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api';

// 1. Check API health
async function checkHealth() {
    try {
        const response = await axios.get(`${BASE_URL}/health/`);
        return response.data;
    } catch (error) {
        console.error('Error checking health:', error.message);
        return null;
    }
}

// 2. Extract data from a PDF
async function extractPdf(filePath) {
    try {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));
        
        const response = await axios.post(`${BASE_URL}/extract/purchase-order`, form, {
            headers: {
                ...form.getHeaders(),
            },
        });
        
        return response.data;
    } catch (error) {
        console.error('Error extracting PDF:', error.message);
        return null;
    }
}

// 3. Batch processing
async function batchExtract(filePaths) {
    try {
        const form = new FormData();
        
        filePaths.forEach(filePath => {
            form.append('files', fs.createReadStream(filePath));
        });
        
        const response = await axios.post(`${BASE_URL}/extract/batch`, form, {
            headers: {
                ...form.getHeaders(),
            },
        });
        
        return response.data;
    } catch (error) {
        console.error('Error in batch extract:', error.message);
        return null;
    }
}

// 4. Export data
async function exportData(extractedData) {
    try {
        const response = await axios.post(`${BASE_URL}/export/json`, extractedData);
        return response.data;
    } catch (error) {
        console.error('Error exporting data:', error.message);
        return null;
    }
}

// Usage example
async function main() {
    // Check health
    const health = await checkHealth();
    console.log('API Status:', health);
    
    // Extract data from a PDF
    const result = await extractPdf('purchase_order.pdf');
    if (result && result.success) {
        console.log('Customer:', result.data.customer);
        console.log('Date:', result.data.date);
        console.log('Total products:', result.data.summary.total_products);
    } else {
        console.log('Error:', result?.message || 'Unknown error');
    }
}

main();
```

### cURL (Command line)

```bash
# 1. Check API health
curl -X GET "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/health/"

# 2. Extract data from a PDF
curl -X POST \
  "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/extract/purchase-order" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@purchase_order.pdf"

# 3. Batch processing
curl -X POST \
  "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/extract/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@order1.pdf" \
  -F "files=@order2.pdf" \
  -F "files=@order3.pdf"

# 4. Export data (example with JSON data)
curl -X POST \
  "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/export/json" \
  -H "Content-Type: application/json" \
  -d '{"customer": "Example Corp", "date": "2024-04-15", "vendor": "Acme Corporation", "products": []}'
```

### PowerShell

```powershell
# 1. Check API health
$healthResponse = Invoke-RestMethod -Uri "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/health/" -Method Get
Write-Host "API Status: $($healthResponse.status)"

# 2. Extract data from a PDF
$filePath = "C:\path\to\purchase_order.pdf"
$uri = "https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/extract/purchase-order"

$form = @{
    file = Get-Item -Path $filePath
}

$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
if ($response.success) {
    Write-Host "Customer: $($response.data.customer)"
    Write-Host "Date: $($response.data.date)"
    Write-Host "Total products: $($response.data.summary.total_products)"
} else {
    Write-Host "Error: $($response.message)"
}
```

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
SAP_AI_CORE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com
SAP_AI_CORE_TOKEN=your_token_here
```

3. Run the application:
```bash
python backend/api_swagger.py
```

The API will be available at:
- **API**: `http://localhost:5001/api/`
- **Swagger**: `http://localhost:5001/swagger/`

## SAP Cloud Foundry Deployment

### Prerequisites

1. **SAP BTP Account** with Cloud Foundry access
2. **CF CLI** installed and configured
3. **Generative AI Hub Service** configured in your Cloud Foundry space

### Deployment Steps

1. **Login to Cloud Foundry:**
```bash
cf login -a https://api.cf.eu01-canary.hana.ondemand.com
```

2. **Select organization and space:**
```bash
cf target -o your-org -s your-space
```

3. **Deploy the application:**
```bash
cf push
```

4. **Create and bind Generative AI Hub service (optional):**
```bash
# Check available services
cf marketplace

# Create service (adjust name according to your marketplace)
cf create-service generative-ai-hub standard generative-ai-hub

# Bind service to application
cf bind-service acme-pdf-extractor generative-ai-hub

# Restart application to apply changes
cf restart acme-pdf-extractor
```

5. **Verify deployment:**
```bash
cf apps
```

### Environment Variables Configuration

After deployment, configure the necessary environment variables:

```bash
cf set-env acme-pdf-extractor SAP_AI_CORE_URL "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com"
cf set-env acme-pdf-extractor SAP_AI_CORE_TOKEN "your_token_here"
cf restage acme-pdf-extractor
```

### Deployment Verification

1. **Check application status:**
```bash
cf app acme-pdf-extractor
```

2. **View logs:**
```bash
cf logs acme-pdf-extractor --recent
```

3. **Test the API:**
```bash
curl https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/api/health/
```

4. **Access Swagger:**
Open in browser: `https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/swagger/`

## Project Structure

```
.
├── backend/
│   ├── api_swagger.py  # Main API with Swagger
│   ├── api.py          # Basic API (without Swagger)
│   ├── app.py          # Alternative application
│   └── pdf.py          # PDF extraction module
├── uploads/            # Temporary directory for files
├── .env               # Environment variables (local)
├── manifest.yml       # Cloud Foundry configuration
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## API Response

### Successful Response Structure

```json
{
  "success": true,
  "message": "Purchase order data extracted successfully",
  "filename": "purchase_order.pdf",
  "processing_timestamp": 1704067200.0,
  "data": {
    "customer": "Dora's Naturals Inc.",
    "date": "15-Mar",
    "vendor": "Acme Corporation",
    "products": [
      {
        "description": "Acme UH Whole 6/64 oz",
        "quantity": 300,
        "unit_price": 25.32
      }
    ],
    "summary": {
      "total_products": 1,
      "total_quantity": 300,
      "total_value": 7596.0
    }
  }
}
```

### Error Response Structure

```json
{
  "success": false,
  "error": "Invalid file type",
  "message": "Only PDF, Word, and text files are allowed"
}
```

## Monitoring and Maintenance

### Cloud Foundry Logs
```bash
# View real-time logs
cf logs acme-pdf-extractor

# View recent logs
cf logs acme-pdf-extractor --recent
```

### Scaling
```bash
# Scale instances
cf scale acme-pdf-extractor -i 2

# Scale memory
cf scale acme-pdf-extractor -m 1G
```

### Updates
```bash
# Update application
cf push

# Restart application
cf restart acme-pdf-extractor
```

## Troubleshooting

### Generative AI Hub Connection Error
- Verify service is bound: `cf services`
- Check environment variables: `cf env acme-pdf-extractor`

### Memory Error
- Increase memory in `manifest.yml`
- Redeploy with `cf push`

### Timeout Error
- Check logs: `cf logs acme-pdf-extractor --recent`
- Consider increasing timeout in configuration

### Swagger not loading
- Verify application is running: `cf app acme-pdf-extractor`
- Access directly: `https://acme-pdf-extractor.cfapps.eu01-canary.hana.ondemand.com/swagger/`

## Support

For technical issues or deployment questions, consult the SAP BTP Cloud Foundry documentation.
