# Development Guide: Building SAP Fiori + AI Core Applications with MCP Servers

This guide provides a comprehensive, step-by-step approach to building full-stack SAP Fiori applications integrated with SAP BTP AI Core using Claude Code and Model Context Protocol (MCP) servers.

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Prerequisites](#prerequisites)
4. [Setting Up MCP Servers](#setting-up-mcp-servers)
5. [Development Workflow](#development-workflow)
6. [Step-by-Step Project Generation](#step-by-step-project-generation)
7. [Deployment Process](#deployment-process)
8. [Testing Strategy](#testing-strategy)
9. [Best Practices](#best-practices)
10. [Common Issues & Solutions](#common-issues--solutions)
11. [Advanced Topics](#advanced-topics)

---

## Overview

This guide demonstrates how to leverage **Claude Code** with **MCP (Model Context Protocol) servers** to rapidly build enterprise-grade SAP Fiori applications that integrate with SAP BTP AI Core for AI-powered capabilities.

### What You'll Build

A full-stack application with:
- **Frontend**: SAP UI5 (Fiori) responsive web application
- **Backend**: FastAPI Python service
- **AI Integration**: SAP BTP AI Core with Google Gemini 2.5 Pro
- **Deployment**: Cloud Foundry (SAP BTP)

### Why Use MCP Servers?

MCP servers provide:
- **Domain expertise**: SAP Fiori/UI5 specific code generation
- **Best practices**: Automated adherence to SAP development standards
- **Faster development**: Scaffolding, component generation, and boilerplate code
- **Error prevention**: Built-in validation and linting

---

## Technology Stack

### Frontend Stack
```
SAP UI5 1.136.7
├── sap.m (Mobile/Responsive controls)
├── sap.ui.layout (Layout containers)
├── sap.ui.unified (File upload, calendar)
└── sap.ushell (Fiori Launchpad)

Theme: SAP Horizon
CDN: https://ui5.sap.com/1.136.7/
```

### Backend Stack
```
Python 3.12
├── FastAPI (Web framework)
├── Uvicorn/Gunicorn (ASGI server)
├── httpx (HTTP client for AI Core)
└── python-dotenv (Environment management)

API Style: OData-inspired RESTful endpoints
```

### Infrastructure Stack
```
SAP BTP Cloud Foundry
├── staticfile_buildpack (Frontend)
├── python_buildpack (Backend)
├── AI Core Service (AI execution)
└── OAuth 2.0 (Authentication)
```

---

## Prerequisites

### Required Tools

1. **Claude Code** (latest version)
   - Download from: https://www.anthropic.com/claude/code
   - Ensure you have access to Claude Code with MCP support

2. **Cloud Foundry CLI**
   ```bash
   # Install via Homebrew (macOS)
   brew install cloudfoundry/tap/cf-cli

   # Or download from: https://github.com/cloudfoundry/cli/releases
   ```

3. **Python 3.12+**
   ```bash
   python --version  # Should be 3.12 or higher
   ```

4. **Node.js 18+** (optional, for UI5 CLI)
   ```bash
   node --version
   npm --version
   ```

### Required Access

1. **SAP BTP Account**
   - Global account with Cloud Foundry enabled
   - Subaccount in your preferred region (e.g., eu10-004)
   - Space for application deployment

2. **SAP AI Core Service**
   - AI Core service instance
   - Service key with credentials
   - Gemini 2.5 Pro (or other model) deployment

3. **GitHub/Git**
   - Git repository for code storage
   - Git installed locally

---

## Setting Up MCP Servers

### Step 1: Install Fiori MCP Server

The `fiori-mcp` server provides SAP Fiori/UI5 development assistance.

#### Installation Methods

**Option A: Via Claude Code Settings**

1. Open Claude Code
2. Go to Settings > MCP Servers
3. Click "Add Server"
4. Enter server details:
   ```json
   {
     "name": "fiori-mcp",
     "command": "npx",
     "args": ["-y", "@sap/fiori-mcp-server"]
   }
   ```

**Option B: Via Configuration File**

Edit your Claude Code configuration file (`.claude/config.json`):

```json
{
  "mcpServers": {
    "fiori-mcp": {
      "command": "npx",
      "args": ["-y", "@sap/fiori-mcp-server"],
      "env": {}
    }
  }
}
```

**Option C: Manual Installation**

```bash
# Install globally
npm install -g @sap/fiori-mcp-server

# Or use via npx (recommended)
npx @sap/fiori-mcp-server --help
```

### Step 2: Verify MCP Server

1. Restart Claude Code
2. In a new chat, type: `/mcp list`
3. Verify `fiori-mcp` appears in the list
4. Test functionality:
   ```
   List available Fiori functionalities
   ```

### Step 3: Configure MCP Server (Optional)

For advanced configuration, create `.fiori-mcp.config.json`:

```json
{
  "ui5Version": "1.136.7",
  "theme": "sap_horizon",
  "namespace": "your-namespace",
  "odataVersion": "v4",
  "typescript": false
}
```

---

## Development Workflow

### High-Level Process

```
1. Project Initialization
   ↓
2. Backend Development (FastAPI + AI Core)
   ↓
3. Frontend Development (SAP UI5)
   ↓
4. Local Testing
   ↓
5. Cloud Foundry Deployment
   ↓
6. Integration Testing
   ↓
7. Production Deployment
```

### Recommended Development Order

1. **Define Requirements** (30 min)
   - Business logic
   - User interface mockups
   - AI capabilities needed

2. **Backend First** (2-3 hours)
   - API endpoints
   - AI Core integration
   - Data models

3. **Frontend Second** (3-4 hours)
   - UI5 components
   - Service integration
   - User experience

4. **Integration & Testing** (1-2 hours)
   - End-to-end testing
   - Error handling
   - Performance optimization

---

## Step-by-Step Project Generation

### Phase 1: Project Initialization

#### Step 1.1: Create Project Structure

**Prompt for Claude Code:**

```
Create a new full-stack SAP Fiori project for video/audio incident analysis with the following requirements:

Technology Stack:
- Frontend: SAP UI5 1.136.7 (Fiori)
- Backend: FastAPI (Python 3.12)
- AI: SAP BTP AI Core with Gemini 2.5 Pro
- Deployment: Cloud Foundry

Project Structure:
- video-incident-monitor/webapp (frontend)
- backend_odata_service.py (backend)
- manifest.yaml (CF deployment)
- requirements.txt (Python deps)

Please create the basic directory structure and essential files.
```

#### Step 1.2: Initialize Git Repository

```bash
cd "AI CORE Video template1"
git init
git add .
git commit -m "Initial project structure"
```

### Phase 2: Backend Development

#### Step 2.1: Generate Backend Service

**Prompt for Claude Code:**

```
Create a FastAPI backend service with the following features:

1. OData-style endpoint: POST /odata/v4/VideoIncidentService/MediaAnalysis
2. File upload handling (video/audio, max 100MB)
3. SAP AI Core integration:
   - OAuth 2.0 authentication
   - Gemini 2.5 Pro API calls
   - Token management
4. CORS configuration for frontend
5. API key authentication
6. Sample media file management (list, download)

Environment variables:
- AICORE_AUTH_URL
- AICORE_CLIENT_ID
- AICORE_CLIENT_SECRET
- AICORE_BASE_URL
- AICORE_RESOURCE_GROUP
- API_KEY
- ALLOWED_ORIGIN

Please generate backend_odata_service.py with all functionality.
```

#### Step 2.2: Create Requirements File

**Prompt for Claude Code:**

```
Create requirements.txt for the backend with:
- FastAPI (latest)
- Uvicorn (ASGI server)
- Gunicorn (production server)
- httpx (HTTP client)
- python-dotenv (environment variables)
- python-multipart (file uploads)

And runtime.txt specifying Python 3.12
```

#### Step 2.3: Create Environment Template

**Prompt for Claude Code:**

```
Create .env.template file with placeholders for:
- AI Core authentication
- API configuration
- CORS settings

Also create .gitignore to exclude .env, __pycache__, etc.
```

### Phase 3: Frontend Development with MCP

#### Step 3.1: List Available Fiori Functionalities

**Prompt for Claude Code:**

```
Using the fiori-mcp server, list all available functionalities for creating a SAP Fiori application in:
/path/to/video-incident-monitor
```

Claude Code will call the MCP server and display available options.

#### Step 3.2: Generate UI5 Application Structure

**Prompt for Claude Code:**

```
Using fiori-mcp, create a SAP UI5 application with:

Functionality: Create new Fiori Elements application (or custom UI5 app)
App Path: video-incident-monitor/webapp
App Name: Video Incident Monitor
Namespace: videoincidentmonitor
UI5 Version: 1.136.7
Theme: sap_horizon

Include:
- manifest.json (app descriptor)
- Component.js (main component)
- index.html (entry point)
- controller/ folder
- view/ folder
- i18n/ folder
```

**Important:** Use MCP server tools in this sequence:
1. `mcp__fiori-mcp__list_functionality` - Get functionality list
2. `mcp__fiori-mcp__get_functionality_details` - Get specific functionality details
3. `mcp__fiori-mcp__execute_functionality` - Execute the functionality

#### Step 3.3: Create Standalone Analysis Page

**Prompt for Claude Code:**

```
Create analyze-standalone.html with the following UI elements using SAP UI5:

Layout:
- Title: "Analyze Video & Audio with AI"
- Description with SAP BTP AI Core mention
- Media type selector (Video/Audio) - SegmentedButton
- File source selector (Server/Upload) - SegmentedButton
- Server file dropdown (populated from backend)
- File uploader (for custom files)
- Analysis configuration panel:
  - Instruction textarea
  - Temperature slider (0.0-2.0)
  - Max tokens input
- Analyze button
- Results panel:
  - Status indicator
  - Progress indicator
  - Results display
  - Download report button
- Link to UI5 examples

Components to use:
- sap.m.Page
- sap.m.Panel
- sap.m.SegmentedButton
- sap.m.FileUploader
- sap.m.TextArea
- sap.m.Slider
- sap.m.Button
- sap.ui.layout.VerticalLayout

Bootstrap from CDN: https://ui5.sap.com/1.136.7/
```

#### Step 3.4: Create Configuration File

**Prompt for Claude Code:**

```
Create config.js that:
1. Detects environment (localhost vs Cloud Foundry)
2. Sets backend base URL accordingly:
   - Local: http://localhost:5000
   - CF: https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com
3. Handles API key from:
   - URL parameter (?api_key=xxx)
   - Meta tag
   - Session storage
4. Exposes window.AI_CORE_VIDEO_CONFIG object
```

#### Step 3.5: Create FLP Sandbox for Testing

**Prompt for Claude Code:**

```
Create test/flpSandbox.html with:
- SAP Fiori Launchpad sandbox
- Tile configuration for the app
- UI5 component examples
- UI5 CDN bootstrap (not local resources)
- Theme: sap_horizon

Include locate-reuse-libs.js for resource loading.
```

### Phase 4: Cloud Foundry Configuration

#### Step 4.1: Create Deployment Manifest

**Prompt for Claude Code:**

```
Create manifest.yaml for Cloud Foundry deployment with two applications:

Application 1 (Frontend):
- name: video-incident-monitor
- buildpack: staticfile_buildpack
- memory: 256M
- disk: 512M
- path: video-incident-monitor/webapp
- route: video-incident-monitor.cfapps.eu10-004.hana.ondemand.com

Application 2 (Backend):
- name: video-incident-monitor-backend
- buildpack: python_buildpack
- stack: cflinuxfs4
- memory: 512M (for video processing)
- disk: 1G
- path: . (root)
- route: video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com
- env: All AICORE_* and API_KEY variables
- health-check-type: port

Use --var for API_KEY (not hardcoded)
```

#### Step 4.2: Create Staticfile Configuration

**Prompt for Claude Code:**

```
Create video-incident-monitor/webapp/Staticfile with:
- root: .
- pushstate: enabled (for client-side routing)

Also create .cfignore to exclude:
- localService/
- .gitignore
- .git/
- node_modules/
- nginx/
But INCLUDE test/ folder for FLP sandbox
```

#### Step 4.3: Create Procfile for Backend

**Prompt for Claude Code:**

```
Create Procfile for Cloud Foundry backend with:
- Gunicorn server
- Uvicorn worker class
- backend_odata_service:app
- Bind to $PORT
- 4 workers
```

### Phase 5: Sample Media Files

#### Step 5.1: Prepare Media Library

**Prompt for Claude Code:**

```
Create directory structure:
- video-incident-monitor/webapp/Video/
- video-incident-monitor/webapp/Audio/

Add placeholder files or instructions to add:
- Video: warehouse-incident.mp4, construction-site.mp4
- Audio: factory-noise.wav, emergency-call.wav

Update backend to serve these via /sample-media endpoints.
```

---

## Deployment Process

### Step 1: Prepare Cloud Foundry Environment

```bash
# Login to Cloud Foundry
cf login -a https://api.cf.eu10-004.hana.ondemand.com

# Target org and space
cf target -o <your-org> -s <your-space>

# Verify connectivity
cf apps
```

### Step 2: Set Environment Variables (Optional)

```bash
# Set AI Core credentials (if not using manifest.yaml)
cf set-env video-incident-monitor-backend AICORE_AUTH_URL "https://..."
cf set-env video-incident-monitor-backend AICORE_CLIENT_ID "sb-..."
cf set-env video-incident-monitor-backend AICORE_CLIENT_SECRET "..."
cf set-env video-incident-monitor-backend AICORE_BASE_URL "https://..."
cf set-env video-incident-monitor-backend AICORE_RESOURCE_GROUP "default"
```

### Step 3: Deploy Applications

**Option A: Deploy Both Apps Together**

```bash
cf push --var api_key="your-secure-api-key-here"
```

**Option B: Deploy Separately**

```bash
# Frontend only
cf push video-incident-monitor

# Backend only
cf push video-incident-monitor-backend --var api_key="your-secure-api-key-here"
```

### Step 4: Verify Deployment

```bash
# Check app status
cf apps

# View logs
cf logs video-incident-monitor-backend --recent
cf logs video-incident-monitor --recent

# Test endpoints
curl https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com/sample-media/video
```

### Step 5: Access Applications

- **Frontend**: https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/
- **UI5 Examples**: https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/test/flpSandbox.html
- **Backend API**: https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com/

---

## Testing Strategy

### Local Testing

#### Backend Testing

```bash
# Start backend
python backend_odata_service.py

# Test with curl
curl -X POST http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis \
  -H "X-API-Key: test-key" \
  -F "file=@sample-video.mp4" \
  -F "instruction=Analyze for safety violations"

# Test sample media endpoint
curl http://localhost:5000/sample-media/video
```

#### Frontend Testing

```bash
# Serve frontend
cd video-incident-monitor/webapp
python -m http.server 8080

# Open in browser
# http://localhost:8080/analyze-standalone.html
```

### Cloud Foundry Testing

#### Integration Testing

1. **Upload Test**
   - Open frontend
   - Upload small test video
   - Verify backend receives and processes

2. **Sample Media Test**
   - Select server file
   - Analyze and verify results

3. **Error Handling Test**
   - Test with invalid API key (should get 401)
   - Test with oversized file (should reject)
   - Test with wrong file type (should validate)

#### Performance Testing

```bash
# Test with larger files
time curl -X POST https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com/odata/v4/VideoIncidentService/MediaAnalysis \
  -H "X-API-Key: your-key" \
  -F "file=@large-video.mp4"
```

---

## Best Practices

### Code Organization

1. **Separation of Concerns**
   ```
   Frontend: Pure UI logic (no business logic)
   Backend: Business logic, AI integration, data processing
   Config: Environment-specific configuration
   ```

2. **Environment Management**
   ```python
   # Use .env for local, CF env vars for production
   from dotenv import load_dotenv
   load_dotenv()  # Only in development

   AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
   ```

3. **Error Handling**
   ```javascript
   // Frontend: User-friendly error messages
   try {
       const response = await fetch(url);
       if (!response.ok) {
           sap.m.MessageBox.error("Failed to analyze media. Please try again.");
       }
   } catch (error) {
       sap.m.MessageBox.error("Network error. Please check your connection.");
   }
   ```

### Security Best Practices

1. **Never Commit Secrets**
   ```gitignore
   # .gitignore
   .env
   *.key
   credentials.json
   ```

2. **Use Environment Variables**
   ```yaml
   # manifest.yaml
   env:
     API_KEY: ((api_key))  # Passed via --var
   ```

3. **Validate All Inputs**
   ```python
   # Backend validation
   if file.size > MAX_FILE_SIZE:
       raise HTTPException(400, "File too large")

   if not file.content_type.startswith("video/"):
       raise HTTPException(400, "Invalid file type")
   ```

4. **CORS Configuration**
   ```python
   # Restrict origins in production
   app.add_middleware(
       CORSMiddleware,
       allow_origins=[os.getenv("ALLOWED_ORIGIN")],  # Not "*"
       allow_credentials=True
   )
   ```

### Performance Optimization

1. **Frontend**
   - Use UI5 CDN for caching
   - Lazy load components
   - Minimize API calls
   - Show loading indicators

2. **Backend**
   - Cache OAuth tokens (don't request on every call)
   - Stream large files
   - Use async operations
   - Set appropriate timeouts

3. **Cloud Foundry**
   - Right-size memory/disk quotas
   - Use health checks
   - Monitor logs and metrics
   - Scale horizontally if needed

---

## Common Issues & Solutions

### Issue 1: CORS Errors in Cloud Foundry

**Symptom:**
```
Access-Control-Allow-Origin header contains multiple values
```

**Root Cause:** Staticfile buildpack adding CORS headers that conflict with backend headers.

**Solution:**
1. Remove custom nginx.conf
2. Use simple Staticfile configuration
3. Ensure backend CORS is properly configured

### Issue 2: UI5 Resources Not Loading

**Symptom:**
```
Uncaught SyntaxError: Unexpected token '<'
sap is not defined
```

**Root Cause:** Pushstate routing redirecting JS file requests to index.html.

**Solution:**
Use absolute CDN URLs in flpSandbox.html:
```html
<script src="https://ui5.sap.com/1.136.7/resources/sap-ui-core.js"></script>
```

### Issue 3: Backend Memory Crashes

**Symptom:**
```
[ERROR] Worker (pid:67) was sent SIGABRT!
```

**Root Cause:** Insufficient memory for video processing.

**Solution:**
Increase memory in manifest.yaml:
```yaml
memory: 512M  # or higher
disk_quota: 1G
```

### Issue 4: API Key Not Working

**Symptom:**
```
401 Unauthorized
```

**Root Cause:** API key not sent with requests.

**Solution:**
Add meta tag to analyze-standalone.html:
```html
<meta name="backend-api-key" content="your-key">
```

Or pass via URL:
```
?api_key=your-key
```

---

## Advanced Topics

### Topic 1: Adding New AI Models

To integrate additional AI models:

1. **Update Backend**
   ```python
   # Support multiple models
   MODEL_CONFIGS = {
       "gemini": {...},
       "gpt4": {...},
       "claude": {...}
   }
   ```

2. **Update Frontend**
   ```javascript
   // Add model selector
   var oModelSelect = new sap.m.Select({
       items: [
           new sap.ui.core.Item({text: "Gemini 2.5 Pro"}),
           new sap.ui.core.Item({text: "GPT-4"})
       ]
   });
   ```

### Topic 2: Implementing OData V4 Properly

For full OData V4 compliance:

1. **Add OData Metadata**
   ```xml
   <!-- metadata.xml -->
   <edmx:Edmx xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx" Version="4.0">
     <edmx:DataServices>
       <Schema xmlns="http://docs.oasis-open.org/odata/ns/edm" Namespace="VideoIncidentService">
         <EntityType Name="MediaAnalysis">
           <Property Name="ID" Type="Edm.Guid" Nullable="false"/>
           <Property Name="incidentType" Type="Edm.String"/>
           ...
         </EntityType>
       </Schema>
     </edmx:DataServices>
   </edmx:Edmx>
   ```

2. **Update Backend Routes**
   ```python
   @app.get("/odata/v4/VideoIncidentService/$metadata")
   async def get_metadata():
       return FileResponse("metadata.xml", media_type="application/xml")
   ```

### Topic 3: Adding Authentication

For user authentication:

1. **Use SAP BTP XSUAA**
   ```yaml
   # manifest.yaml
   services:
     - xsuaa-service
   ```

2. **Validate JWT Tokens**
   ```python
   from fastapi import Security
   from fastapi.security import HTTPBearer

   security = HTTPBearer()

   async def verify_token(token: str = Security(security)):
       # Validate JWT
       ...
   ```

### Topic 4: Monitoring and Logging

#### Application Logging

**Backend:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Processing file: {filename}")
```

**Frontend:**
```javascript
// Use sap.base.Log
sap.ui.require(["sap/base/Log"], function(Log) {
    Log.info("Media analysis started");
    Log.error("Failed to upload file", error);
});
```

#### Cloud Foundry Monitoring

```bash
# Stream logs
cf logs video-incident-monitor-backend

# Get app metrics
cf app video-incident-monitor-backend

# View recent crashes
cf events video-incident-monitor-backend
```

---

## Conclusion

This guide provides a comprehensive approach to building SAP Fiori applications with AI Core integration using Claude Code and MCP servers. Key takeaways:

1. **Use MCP servers** for SAP-specific code generation
2. **Follow the workflow**: Backend first, then frontend
3. **Test locally** before Cloud Foundry deployment
4. **Secure credentials** with environment variables
5. **Monitor and optimize** in production

For questions or issues, refer to:
- [SAP UI5 Documentation](https://ui5.sap.com/)
- [SAP BTP AI Core Docs](https://help.sap.com/docs/sap-ai-core)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Cloud Foundry Docs](https://docs.cloudfoundry.org/)

Happy coding!
