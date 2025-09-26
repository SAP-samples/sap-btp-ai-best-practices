# Translation Hub Service

This repository contains below Python script

- text_translation_llm.ipynb
- document_translation_aynchronous.ipynb
- document_translation_sychronous.ipynb 

 which interacts with the SAP Translation Hub AI REST API to translate from documents or texts.

## Prerequisites

1. **Python Environment**:
   - Ensure Python 3.6 or higher is installed.
   - Install the required dependencies below commands:
   ```bash
      pip install python-docx python-dotenv requests pypdf
     ```
     Or using the `requirements.txt` file
     ```bash
     pip install -r requirements.txt
     ```

2. **Environment Variables**:
   - The script uses credentials and configuration stored in a `.env` file. Create a `.env` file in the root directory and populate it with the required variables (see below for an example).

3. **Translation Hub AI Service**:
   - Set up the Translation Hub AI Service and UI. Refer to [SAP Documentation](https://help.sap.com/docs/translation-hub/sap-translation-hub/initial-setup) for setup instructions.

## Environment Variables

Create a `.env` file in the root directory with the following content:

```env
CLIENT_ID=service_key['uaa']['clientid']
CLIENT_SECRET=service_key['uaa']['clientsecret']
AUTH_URL=service_key['uaa']['url']
DOCTRANS_BASE=service_key['documenttranslation']['url']
SOFTWARE_BASE =service_key['softwaretranslation']['url']
```

> **Note**: Replace the placeholder values with your actual SAP Translation Hub AI service credentials.

## Usage Instructions

1. **Run the Script**:

   Choose the relevant file depending on your use case:

    - Text Translation (LLM) → text_translation_llm.ipynb
    - Document Translation (Asynchronous) → document_translation_asynchronous.ipynb
    - Document Translation (Synchronous) → document_translation_synchronous.ipynb
    
    for example run below file to translate text
   ```bash
   python text_translation_llm.py
   ```

2. **Input Document**:

    For Document Translation
   - Place the document to be processed in the appropriate directory.
   - Update the `in_path` in the script to point to the correct file.
   
    For Text Translation
    - Directly embed your text to be translated in data 

    ```python
    body = {
   "sourceLanguage": "de-DE",
   "targetLanguage": "en-US",
   "encoding": "plain",
   "model": "llm",
   "data": "Dieser Satz soll übersetzt werden"
    }
    ```

3. **Output**:

   - The translated text will be shown in the notebook cell.


