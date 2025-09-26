# SAP RFQx Document Analysis System

A comprehensive enterprise-grade system for analyzing and comparing Quotation documents from a given RFQ using advanced AI processing with direct document-to-LLM integration. Features multi-format document support, knowledge graph visualization, project management, and country risk analysis capabilities.

This enhanced implementation combines simplified RAG processing with advanced analytics, project management, and interactive visualization tools, providing a complete solution for RFQ document analysis and comparison.

## Features

### Core Processing Capabilities
- **Multi-Format Document Support**: Processes PDFs, Excel files, and CSV documents
- **Direct Document-to-LLM Processing**: Uses GPT-4.1's large context window to process entire documents without chunking
- **No Information Loss**: Eliminates chunking-related information loss by processing complete documents
- **Structure-Aware Processing**: Maintains document structure integrity during extraction
  - Uses PyMuPDF for enhanced text formatting with style preservation
  - Leverages pdfplumber for table detection and extraction
  - Converts tables to structured text format with pipe separators
  - Preserves lists and formatting through direct document text extraction
- **Advanced Report Generation**: Clean comparison reports with detailed analysis and insights

### API Features
- **Multiple Document Comparison**: Compare any number of RFQ documents simultaneously
- **Single Document Extraction**: Extract structured information from individual documents
- **Query Interface**: Answer specific questions about document contents
- **Command Line Interface**: Full CLI support with flexible argument parsing for batch processing

### Web Application Features
- **Interactive Document Upload**: Upload multiple document formats (PDF, Excel, CSV) through web interface
- **Real-time Processing**: Live progress tracking and feature extraction statistics
- **Side-by-side Comparison**: Categorized comparison tables with summary statistics
- **Document Chat**: Natural language Q&A interface for document content
- **Chat History**: Track conversation history with processed documents
- **Project Management**: Save and load project states with document references and analysis results
- **Knowledge Graph Visualization**: Interactive graph exploration of document relationships
- **Country Risk Analysis**: Enhanced analysis with country-specific risk insights
- **Advanced Analytics**: Multi-provider comparison with filtering and search capabilities

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

```bash
streamlit run RFQx.py
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

**Step 4: Advanced Graph Analysis (Optional)**
1. Go to the "Advanced Graph" tab (requires processed documents)
2. Explore interactive knowledge graph visualization
3. Use filtering and search capabilities to analyze relationships
4. Export graph data for further analysis

**Step 5: Project Management**
1. Save your current analysis session using the project management features
2. Load previously saved projects to continue analysis
3. Export analysis results and reports

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface with multi-page navigation and interactive components
- **Backend**: Modular architecture with specialized processors and managers
- **AI Model**: GPT-4.1 through SAP GenAI Hub SDK
- **Document Processing**: Multi-format support with PyMuPDF, pdfplumber, and Excel/CSV processors
- **Knowledge Graph**: NetworkX-based graph processing with interactive visualization
- **Project Management**: JSON-based project state persistence with security validation
- **Risk Analysis**: Country-specific risk data integration for enhanced insights

### Supported Document Types
- RFQ (Request for Quotation) documents
- PDF format with multi-page support
- Excel files (.xlsx, .xls) with structured data
- CSV files with tabular data
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

### Core Application Files
- `RFQx.py`: Main Streamlit web application with multi-page interface
- `main.py`: Core SimplifiedRFQComparator class with enhanced processing capabilities
- `llm_client.py`: GPT-4.1 client interface with country risk integration
- `document_processor.py`: Enhanced multi-format document processing
- `file_processor.py`: Unified file handling for PDF, Excel, and CSV documents
- `rfq_schema.py`: Comprehensive RFQ extraction schema definitions

### Advanced Features
- `graph_processor.py`: Knowledge graph creation and visualization using NetworkX
- `project_manager.py`: Project state management with save/load functionality
- `country_risk_manager.py`: Country risk data integration and analysis
- `ui_components.py`: Reusable Streamlit UI components
- `pdf_generator.py`: Advanced PDF report generation capabilities

### Application Pages
- `pages/3_Advanced_Graph.py`: Interactive knowledge graph explorer with st-link-analysis

### Configuration and Dependencies
- `requirements.txt`: Python dependencies including advanced visualization libraries
- `uploads/`: Directory for uploaded files (created automatically)
- `projects/`: Directory for saved project states
- `extra_docs/`: Additional documentation and data files
- `.streamlit/`: Streamlit configuration directory

## Key Benefits

### Processing Advantages
1. **No Information Loss**: Processes complete documents without chunking
2. **Context Preservation**: Maintains full document context for better extraction
3. **Structure Integrity**: Preserves table and list structures across multiple formats
4. **Enhanced Reports**: Detailed markdown reports with difference analysis
5. **Multi-Format Support**: Unified processing for PDF, Excel, and CSV documents

### Advanced Features
6. **Knowledge Graph Visualization**: Interactive exploration of document relationships
7. **Project Management**: Save and restore analysis sessions with full state persistence
8. **Country Risk Analysis**: Enhanced insights with country-specific risk data
9. **Real-time Collaboration**: Multi-user support with session management
10. **Extensible Architecture**: Modular design for easy feature additions

## Advanced Features

### Knowledge Graph Processing
- **Interactive Visualization**: NetworkX-based graph creation with Plotly integration
- **Relationship Analysis**: Automatic detection of document relationships and patterns
- **Advanced Filtering**: Search and filter capabilities for large document sets
- **Export Capabilities**: Export graph data in multiple formats

### Project Management
- **State Persistence**: Save complete analysis sessions with all document references
- **Security Validation**: Secure project loading with path validation
- **Version Control**: Track project changes and maintain history
- **Collaborative Features**: Multi-user session management

### Country Risk Analysis
- **Risk Data Integration**: Country-specific risk assessment capabilities
- **Enhanced Insights**: Risk-aware document analysis and comparison
- **Data Management**: CSV-based risk data with automatic updates
- **Visualization**: Risk metrics integration in reports and graphs

### Multi-Format Processing
- **Unified Interface**: Single API for PDF, Excel, and CSV processing
- **Format Detection**: Automatic document type recognition
- **Structure Preservation**: Maintains formatting across different document types
- **Batch Processing**: Efficient processing of multiple document formats

## Dependencies

### Core Dependencies
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `generative-ai-hub-sdk`: SAP GenAI Hub integration
- `PyMuPDF`: PDF text extraction
- `pdfplumber`: Advanced PDF processing
- `tiktoken`: Token counting for context management

### Advanced Dependencies
- `networkx`: Knowledge graph processing
- `plotly`: Interactive visualizations
- `st-link-analysis`: Advanced graph visualization
- `openpyxl`: Excel file processing
- `xlrd`: Legacy Excel support
- `reportlab`: PDF generation
- `matplotlib`: Static graph generation

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
   - Knowledge graph processing may be memory-intensive for large document sets

5. **Graph Visualization Issues**
   - Ensure `st-link-analysis` is installed: `pip install st-link-analysis`
   - Large graphs may require filtering for better performance
   - Browser compatibility issues with complex visualizations

6. **Project Management Issues**
   - Ensure write permissions in the projects directory
   - Check file paths for security violations
   - Verify project JSON structure for corrupted files

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all requirements are installed
3. Ensure PDF files are valid and contain text content
4. Check API credentials and connectivity