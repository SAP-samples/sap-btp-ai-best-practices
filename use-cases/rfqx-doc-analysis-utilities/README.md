# SAP RFQx Document Analysis System

A comprehensive multi-page Streamlit application for analyzing and comparing Quotation documents using AI-powered processing. The application provides a complete workflow from project setup to supplier comparison and document chat capabilities.

## Current Application Features

### Multi-Page Interface
- **Project Setup & Management**: Create and manage RFQ analysis projects
- **Process Supplier Documents**: Upload and process supplier RFQ documents with AI extraction
- **Compare Suppliers**: Generate side-by-side comparisons of processed suppliers
- **RFQ Insights & Recommendations**: AI-generated knowledge graphs and narrative reports
- **Supplier Document Chat**: Interactive Q&A with processed documents

### Core Processing Capabilities
- **Multi-Format Support**: Process PDF, Excel, and CSV documents
- **AI-Powered Extraction**: Uses GPT-4.1 for intelligent document analysis
- **Structure-Aware Processing**: Maintains document structure integrity during extraction
- **Project Management**: Save and load analysis projects with session persistence
- **Real-time Processing**: Live progress tracking and extraction statistics

### Advanced Features
- **Knowledge Graph Visualization**: Interactive network graphs showing supplier relationships
- **Country Risk Analysis**: Integrated risk assessment for supplier locations
- **Document Chat**: Natural language Q&A interface with streaming responses
- **Comparative Analysis**: Side-by-side supplier comparison with detailed metrics

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment**
   Ensure you have access to GPT-4.1 through SAP GenAI Hub SDK and appropriate API credentials configured.

## Usage

### Starting the Application

**Using the Streamlit Launcher**
```bash
python streamlit_app.py
```

**Alternative: Direct Streamlit Command**
```bash
streamlit run RFQx.py
```

The application will open in your default web browser at `http://localhost:8501`

### Application Workflow

#### Step 1: Project Setup & Management
1. Navigate to the "Project Setup & Management" page
2. Create a new project or load an existing one
3. Configure project settings and metadata
4. Save your project for future sessions

#### Step 2: Process Supplier Documents
1. Go to the "Process Supplier Documents" page
2. Upload supplier RFQ documents (PDF, Excel, CSV supported)
3. Configure extraction attributes and processing options
4. Launch processing jobs and monitor progress
5. Review extracted features and processing statistics

#### Step 3: Compare Suppliers
1. Navigate to the "Compare Suppliers" page
2. Select processed suppliers for comparison
3. Generate side-by-side comparison reports
4. Review categorized comparison tables and metrics

#### Step 4: RFQ Insights & Recommendations
1. Go to the "RFQ Insights & Recommendations" page
2. Generate knowledge graph visualizations
3. Create AI-powered narrative reports
4. Analyze supplier relationships and recommendations

#### Step 5: Supplier Document Chat
1. Visit the "Supplier Document Chat" page
2. Ask questions about processed documents
3. Get comparative responses across suppliers
4. Review chat history and insights

## Technical Details

### Architecture
- **Frontend**: Multi-page Streamlit application with SAP template styling
- **Backend**: Modular architecture with specialized processors
- **AI Model**: GPT-4.1 through SAP GenAI Hub SDK
- **Document Processing**: Multi-format support (PDF, Excel, CSV)
- **Project Management**: Persistent session state and project storage

### Supported Document Types
- RFQ (Request for Quotation) documents
- PDF, Excel, and CSV formats
- Multi-page documents supported
- Complex structured documents with tables and sections

### Application Structure
- **Multi-Page Layout**: Dedicated pages for each major functionality
- **Session Management**: Persistent state across page navigation
- **Project Persistence**: Save and load analysis projects
- **Real-time Processing**: Live progress tracking and status updates

## File Structure

### Core Application Files
- `streamlit_app.py`: Main application launcher
- `RFQx.py`: Home page and navigation hub
- `app_context.py`: Shared application context and utilities

### Page Components
- `pages/1_Project_Setup.py`: Project management interface
- `pages/2_Process_Documents.py`: Document processing interface
- `pages/3_Compare_Providers.py`: Supplier comparison interface
- `pages/4_Rfq_Recommender.py`: Insights and recommendations
- `pages/5_Supplier_Chat.py`: Document chat interface

### Backend Components
- `main.py`: Core SimplifiedRFQComparator class
- `llm_client.py`: GPT-4.1 client interface
- `document_processor.py`: Document processing utilities
- `file_processor.py`: Multi-format file handling
- `rfq_schema.py`: RFQ extraction schema definitions
- `project_manager.py`: Project persistence management
- `graph_processor.py`: Knowledge graph processing
- `country_risk_manager.py`: Risk assessment utilities
- `ui_components.py`: Reusable UI components

### Configuration
- `requirements.txt`: Python dependencies
- `manifest.yml`: Application deployment configuration
- `uploads/`: Directory for uploaded files (created automatically)
- `projects/`: Directory for saved projects (created automatically)

## Key Features

### Project Management
- Create and manage multiple RFQ analysis projects
- Save and load project sessions with full state persistence
- Track project metadata and supplier information

### Document Processing
- Multi-format document support (PDF, Excel, CSV)
- AI-powered feature extraction using GPT-4.1
- Real-time processing progress tracking
- Structure-aware text extraction with table preservation

### Supplier Analysis
- Side-by-side supplier comparison with detailed metrics
- Knowledge graph visualization of supplier relationships
- Country risk assessment integration
- AI-generated insights and recommendations

### Interactive Chat
- Natural language Q&A with processed documents
- Comparative responses across multiple suppliers
- Chat history tracking and session management
- Streaming response support for real-time interaction

## Troubleshooting

### Common Issues

1. **Application Startup**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're running from the correct directory
   - Verify SAP GenAI Hub SDK credentials are configured

2. **Document Processing**
   - Ensure uploaded files are not corrupted or password-protected
   - Check that files contain extractable text (not just images)
   - Verify file formats are supported (PDF, Excel, CSV)

3. **Session Management**
   - Clear browser cache if experiencing session issues
   - Check that projects directory is writable
   - Restart application if session state becomes corrupted

4. **Performance**
   - Large documents may take longer to process
   - GPT-4.1 API calls can be rate-limited
   - Consider processing documents in smaller batches

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all requirements are installed correctly
3. Ensure API credentials are properly configured
4. Check that uploaded files are valid and contain extractable content