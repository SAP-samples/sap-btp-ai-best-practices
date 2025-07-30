# Document Extraction Service

This repository contains a Python script, `document_extraction_service.py`, which interacts with the SAP Document Information Extraction REST API to extract information from documents such as invoices.

## Prerequisites

1. **Python Environment**:
   - Ensure Python 3.6 or higher is installed.
   - Install the required dependencies using the `requirements.txt` file:
   
     ```bash
     pip install -r requirements.txt
     ```

2. **Environment Variables**:
   - The script uses credentials and configuration stored in a `.env` file. Create a `.env` file in the root directory and populate it with the required variables (see below for an example).

3. **Document Information Extraction Service**:
   - Set up the SAP Document Information Extraction Service and UI. Refer to [SAP Documentation](https://developers.sap.com/tutorials/cp-aibus-dox-service-instance.html) for setup instructions.

## Environment Variables

Create a `.env` file in the root directory with the following content:

```env
CLIENT_ID=service_key['uaa']['clientid']
CLIENT_SECRET=service_key['uaa']['clientsecret']
URL=service_key['url']
UAADOMAIN=service_key['uaa']['url']
```

> **Note**: Replace the placeholder values with your actual SAP Document Information Extraction service credentials.

## Usage Instructions

1. **Run the Script**:

   Execute the `document_extraction_service.py` file to extract information from a document:

   ```bash
   python document_extraction_service.py
   ```

2. **Input Document**:

   - Place the document to be processed in the appropriate directory.
   - Update the `document_path` in the script to point to the correct file.

3. **Output**:

   - The extracted information will be saved in `result.json` in the root directory.


