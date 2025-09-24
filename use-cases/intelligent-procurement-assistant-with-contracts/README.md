# Intelligent Procurement Assistant

This repository provides an interactive web app for automated extraction, validation, and analysis of commodity codes and other structured contract data, tailored for procurement and compliance workflows.

## Features

- **Automated Commodity Code Identification** from unstructured contract documents (PDFs, DOCX, etc.)
- **Streamlit-based Interface** for document upload, review, validation, and reporting
- **Attribute Extraction:** Party names, contract values, service periods, and more
- **Batch Processing & Audit Trails**
- **Integration-ready** for SAP and enterprise platforms
-

## Architecture Diagram

![Intelligent Procurement Assistant Process Flow](IPA%20Process%20Flow%20202507.drawio.png)

_Figure: High-level process flow for the Intelligent Procurement Assistant system_

---

## 1. How is the Commodity Code Determined?

Commodity codes are extracted and validated using a multi-step pipeline:

1. **Text Extraction:**  
   The uploaded document is parsed (using [PyMuPDF](https://pymupdf.readthedocs.io/) for PDFs or [python-docx](https://python-docx.readthedocs.io/) for DOCX) to extract all text content.

2. **Pattern-Based Matching:**  
   The system scans for numeric or alphanumeric codes that match known commodity code patterns (e.g., UNSPSC, CPV, or company-specific schemes).  
   This is performed using regular expressions, dictionary lookups, and (optionally) LLM-based entity recognition.

3. **Contextual Validation:**  
   Detected codes are cross-referenced with internal codebooks or public databases.  
   If the context matches expected commodity/service descriptions, the code is marked as “validated.”  
   For ambiguous cases, the system prompts the user for manual validation in the UI.

4. **User Review & Correction:**  
   All detected codes are shown in the interface for review.  
   The user can approve, edit, or reject the proposed code. Changes are logged for auditability.

5. **Final Output:**  
   Validated commodity codes are saved as part of the structured contract record (CSV/JSON/Database), ready for downstream integration.

**Example Logic (Pseudocode):**

```python
for match in re.finditer(COMMODITY_CODE_PATTERN, extracted_text):
    code = match.group()
    if code in KNOWN_CODES:
        status = "validated"
    else:
        status = "needs_review"
    results.append({"code": code, "status": status})
```

---

## 2. Interface Design & Logic

The app uses [Streamlit](https://streamlit.io/) for a fast, interactive, and auditable workflow.

**Interface Logic:**

- **Step 1: Document Upload**

  - User selects and uploads a contract or invoice file.
  - The app detects the file type and starts extraction.

- **Step 2: Attribute Extraction**

  - Key contract data (including commodity codes) are automatically extracted.
  - Results are displayed as editable fields.

- **Step 3: Review & Validation**

  - Each detected commodity code and attribute is shown in a structured table.
  - Users can:
    - Confirm the extraction
    - Edit incorrect codes or attributes
    - Flag issues for manual review
  - Changes are saved with user/time for traceability.

- **Step 4: Export & Integration**
  - Processed data can be exported as CSV/JSON.
  - Optionally, data can be pushed to an SAP system or other ERP.

**Interface Example:**

```python
import streamlit as st

uploaded_file = st.file_uploader("Upload Contract", type=["pdf", "docx"])
if uploaded_file:
    extracted_data = extract_contract_data(uploaded_file)
    st.write("Extracted Commodity Codes:", extracted_data["commodity_codes"])
    for code in extracted_data["commodity_codes"]:
        new_code = st.text_input(f"Edit code {code['code']}", value=code["code"])
        # ... Save changes, track user actions
    if st.button("Export"):
        export_data(extracted_data)
```

**Design Rationale:**

- **Transparency:** Every automated extraction is visible and editable to the user.
- **Auditability:** All manual changes are tracked.
- **Simplicity:** The workflow minimizes steps and complexity for the user.

---

## Installation

```bash
git clone <repo_url>
 cd intelligent-procurement-assistant
pip install -r requirements.txt
streamlit run app.py
```

Configure endpoint in `config.py`.

---

**For further questions or enterprise integration, contact the maintainer.**
