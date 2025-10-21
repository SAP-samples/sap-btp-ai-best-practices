# SAP RFQx Document Analysis System

A comprehensive system for analyzing and comparing RFQ (Request for Quotation) documents using simplified RAG (Retrieval-Augmented Generation) with direct PDF-to-LLM processing. Available as both a Python API and a Streamlit web application.

This enhanced implementation combines the successful simplified RAG approach with advanced markdown report generation capabilities, eliminating chunking-related information loss while maintaining the original agentic_rfq_comparator's clean formatting.

## Features

### Core Processing Capabilities
- **Direct PDF-to-LLM Processing**: Uses GPT-4.1's large context window to process entire PDFs without chunking
- **No Information Loss**: Eliminates chunking-related information loss by processing complete documents
- **Structure-Aware Processing**: Maintains document structure integrity during extraction
  - Uses PyMuPDF for enhanced text formatting with style preservation
  - Leverages pdfplumber for table detection and extraction
  - Converts tables to structured text format with pipe separators
  - Preserves lists and formatting through direct PDF text extraction
- **Original-Style Markdown Reports**: Clean comparison reports with exact formatting from the original agentic_rfq_comparator

### API Features
- **Multiple Document Comparison**: Compare any number of RFQ documents simultaneously
- **Single Document Extraction**: Extract structured information from individual documents
- **Query Interface**: Answer specific questions about document contents
- **Command Line Interface**: Full CLI support with flexible argument parsing for batch processing

### Web Application Features
- **Interactive Document Upload**: Upload multiple PDF files through web interface
- **Real-time Processing**: Live progress tracking and feature extraction statistics
- **Side-by-side Comparison**: Categorized comparison tables with summary statistics
- **Document Chat**: Natural language Q&A interface for document content
- **Chat History**: Track conversation history with processed documents

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment**
   Ensure you have access to GPT-4.1 through SAP GenAI Hub SDK and appropriate API credentials configured.

## Usage

### API Usage

#### Compare Multiple RFQ Documents
```python
from main import SimplifiedRFQComparator

comparator = SimplifiedRFQComparator()
# Compare two documents
result = comparator.compare_documents(["doc1.pdf", "doc2.pdf"], "comparison_report.md")

# Compare multiple documents
result = comparator.compare_documents(["doc1.pdf", "doc2.pdf", "doc3.pdf"], "comparison_report.md")
```

#### Extract Information from Single Document
```python
extracted_data = comparator.process_single_document("rfq_document.pdf")
```

#### Answer Specific Queries
```python
answer = comparator.answer_query("rfq_document.pdf", "What is the submission deadline?")
```

#### Command Line Interface
```bash
# Compare two documents with output file
python main.py compare doc1.pdf doc2.pdf -o report.md

# Compare multiple documents with output file
python main.py compare doc1.pdf doc2.pdf doc3.pdf doc4.pdf -o report.md

# Compare documents without output file (prints to console)
python main.py compare doc1.pdf doc2.pdf

# Extract from single document
python main.py extract document.pdf

# Answer specific query
python main.py query document.pdf "What is the estimated contract value?"
```

### Web Application Usage

#### Starting the Application

**Option 1: Using the Run Script**
```bash
python run_app.py
```

**Option 2: Direct Streamlit Command**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

#### Using the Web Interface

**Step 1: Process Documents**
1. Go to the "Process Documents" tab
2. Upload one or more PDF files containing RFQ documents
3. Click "Process Documents" to extract features
4. Monitor the progress and extraction statistics

**Step 2: Compare Documents (Optional)**
1. Go to the "Compare Documents" tab (requires at least 2 processed documents)
2. Click "Compare Documents" to see side-by-side comparison
3. Review the categorized comparison tables and summary statistics

**Step 3: Chat with Documents (Optional)**
1. Go to the "Document Chat" tab (requires at least 1 processed document)
2. Enter your question in the text input
3. Click "Ask Question" to get responses from all processed documents
4. Review answers and chat history

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface with interactive document processing
- **Backend**: SimplifiedRFQComparator class with direct PDF-to-LLM processing
- **AI Model**: GPT-4.1 through SAP GenAI Hub SDK
- **Document Processing**: PyMuPDF and pdfplumber for PDF text extraction

### Supported Document Types
- RFQ (Request for Quotation) documents
- PDF format only
- Multi-page documents supported
- Complex structured documents with tables and sections

### Extraction Schema
The system extracts information across 10 major categories:
- Project Information
- Key Dates & Deadlines
- Scope & Technical Requirements
- Supplier Requirements
- Evaluation Criteria
- Pricing & Payment
- Legal & Contractual
- Compliance & Exclusion Grounds
- Sustainability & Social Value
- Contract Management & Reporting

## File Structure

- `main.py`: Core SimplifiedRFQComparator class with original-style report generation
- `app.py`: Main Streamlit web application interface
- `run_app.py`: Launch script for the web application
- `llm_client.py`: GPT-4.1 client interface for direct PDF processing
- `pdf_processor.py`: Enhanced PDF content extraction utilities
- `rfq_schema.py`: Comprehensive RFQ extraction schema definitions
- `test_simplified.py`: Test script for validation
- `demo.py`: Demonstration script
- `requirements.txt`: Python dependencies
- `uploads/`: Directory for uploaded files (created automatically)

## Benefits Over Chunked Approach

1. **No Information Loss**: Processes complete documents without chunking
2. **Context Preservation**: Maintains full document context for better extraction
3. **Structure Integrity**: Preserves table and list structures
4. **Enhanced Reports**: Detailed markdown reports with difference analysis
5. **Proven Success**: Addresses the "Not Found" issues from the original implementation

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Make sure you're running from the correct directory

2. **API Errors**
   - Verify your SAP GenAI Hub SDK credentials are configured
   - Check internet connectivity for API access

3. **Processing Errors**
   - Ensure PDF files are not corrupted or password-protected
   - Check that files contain extractable text (not just images)

4. **Performance Issues**
   - Large PDF files may take longer to process
   - GPT-4.1 API calls can be rate-limited

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all requirements are installed
3. Ensure PDF files are valid and contain text content
4. Check API credentials and connectivity