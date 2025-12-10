# AI Core Video Incident Monitoring

A full-stack enterprise application for video and audio incident detection using **SAP BTP AI Core** and **Google Gemini 2.5 Pro**. Built with SAP UI5 (Fiori) frontend and FastAPI backend.

## Overview

This application enables users to analyze video and audio files for safety incidents using advanced AI capabilities. It leverages SAP BTP AI Core for secure, enterprise-grade AI model execution and provides a modern SAP Fiori user interface for seamless user experience.

### Key Features

- **Video & Audio Analysis**: Upload media files or select from server examples
- **AI-Powered Detection**: Uses Google Gemini 2.5 Pro via SAP BTP AI Core for secure and scalable AI model execution
- **Safety Incident Monitoring**: Detects workplace safety violations, hazardous conditions, and security incidents
- **SAP Fiori UI**: Modern, responsive user interface built with SAPUI5 1.136.7
- **Flexible Deployment**: Supports local development and Cloud Foundry deployment
- **Example Gallery**: FLP Sandbox with UI5 component examples
- **PDF Report Generation**: Download detailed analysis reports

## Technology Stack

### Frontend
- **Framework**: SAP UI5 1.136.7 (Fiori)
- **Components**: sap.m, sap.ui.layout, sap.ui.unified, sap.ushell
- **Theme**: SAP Horizon
- **Deployment**: Cloud Foundry with staticfile_buildpack
- **CDN**: UI5 CDN for fast resource loading

### Backend
- **Framework**: FastAPI (Python 3.12)
- **AI Integration**: SAP BTP AI Core + Google Gemini 2.5 Pro
- **Authentication**: OAuth 2.0 with SAP AI Core
- **Media Processing**: Video/Audio file analysis with multipart upload
- **API Style**: OData-inspired endpoints
- **Deployment**: Cloud Foundry with python_buildpack (cflinuxfs4)
- **ASGI Server**: Gunicorn with Uvicorn workers

### Infrastructure
- **Platform**: SAP Business Technology Platform (BTP)
- **Runtime**: Cloud Foundry (eu10-004 region)
- **AI Service**: SAP AI Core
- **Storage**: Temporary file processing with local media library

## Architecture

```
┌─────────────────────────────────────────────────┐
│  SAP Fiori Frontend (SAP UI5 1.136.7)          │
│  ┌──────────────────────────────────────────┐  │
│  │  analyze-standalone.html                 │  │
│  │  - Main application UI                   │  │
│  │  - Media upload/selection                │  │
│  │  - Analysis configuration                │  │
│  │  - Results visualization                 │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  test/flpSandbox.html                    │  │
│  │  - UI5 component examples                │  │
│  │  - FLP tile demos                        │  │
│  └──────────────────────────────────────────┘  │
│  Staticfile buildpack + pushstate routing     │
└────────────────┬────────────────────────────────┘
                 │ HTTPS/REST API + API Key Auth
                 ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Backend (Python)                       │
│  ┌──────────────────────────────────────────┐  │
│  │  /odata/v4/VideoIncidentService/         │  │
│  │    MediaAnalysis                         │  │
│  │  - Media upload & validation             │  │
│  │  - AI Core OAuth token management        │  │
│  │  - Gemini 2.5 Pro API calls              │  │
│  │  - Results processing & formatting       │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Sample Media Management                 │  │
│  │  - Video/Audio file library              │  │
│  │  - File listing & download               │  │
│  └──────────────────────────────────────────┘  │
│  CORS + API Key validation                     │
└────────────────┬────────────────────────────────┘
                 │ OAuth 2.0 (Client Credentials)
                 │ REST API calls
                 ▼
┌─────────────────────────────────────────────────┐
│  SAP BTP AI Core                                │
│  ┌──────────────────────────────────────────┐  │
│  │  Authentication Service                  │  │
│  │  - OAuth 2.0 token endpoint              │  │
│  └──────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │  Deployment Management                   │  │
│  │  - Resource group: default               │  │
│  │  - Model: Google Gemini 2.5 Pro          │  │
│  │  - Secure execution environment          │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
AI CORE Video template1/
├── video-incident-monitor/
│   └── webapp/
│       ├── analyze-standalone.html    # Main application UI
│       ├── index.html                 # Redirects to main app
│       ├── config.js                  # Runtime configuration
│       ├── Component.js               # UI5 component definition
│       ├── manifest.json              # App descriptor
│       ├── Staticfile                 # CF buildpack config
│       ├── .cfignore                  # CF deployment exclusions
│       ├── controller/                # UI5 controllers
│       │   ├── App.controller.js
│       │   ├── MediaAnalysisList.controller.js
│       │   └── MediaAnalysisDetail.controller.js
│       ├── view/                      # UI5 XML views
│       │   ├── App.view.xml
│       │   ├── MediaAnalysisList.view.xml
│       │   └── MediaAnalysisDetail.view.xml
│       ├── test/
│       │   ├── flpSandbox.html       # FLP sandbox (UI5 examples)
│       │   └── locate-reuse-libs.js  # Library loader
│       ├── i18n/                      # Internationalization
│       ├── Audio/                     # Sample audio files
│       │   ├── factory-noise.wav
│       │   └── emergency-call.wav
│       └── Video/                     # Sample video files
│           ├── warehouse-incident.mp4
│           └── construction-site.mp4
├── backend_odata_service.py           # FastAPI backend server
├── manifest.yaml                      # CF deployment manifest
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python version (3.12)
├── Procfile                          # CF process configuration
├── metadata.xml                       # OData metadata
├── README.md                         # This file
└── Prompt.md                         # Development guide with MCP
```

## Prerequisites

### For Local Development
- **Node.js** 18+ and npm (optional, for UI5 tooling)
- **Python** 3.12+
- **SAP BTP** account with AI Core service
- **Cloud Foundry CLI** (for deployment)

### For Deployment
- SAP BTP Cloud Foundry space (eu10-004 or your region)
- AI Core service instance with Gemini 2.5 Pro deployment
- Service key with OAuth credentials

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "AI CORE Video template1"
```

### 2. Backend Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with your AI Core credentials:

```env
AICORE_AUTH_URL=https://dts-ai-core.authentication.eu10.hana.ondemand.com
AICORE_CLIENT_ID=sb-xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx!b217189|aicore!b540
AICORE_CLIENT_SECRET=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx$...
AICORE_BASE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2
AICORE_RESOURCE_GROUP=default
API_KEY=your-secure-api-key
ALLOWED_ORIGIN=http://localhost:8080
```

### 3. Frontend Setup

The frontend uses UI5 CDN (https://ui5.sap.com/1.136.7/), so no npm install is required for runtime dependencies.

For development tooling (optional):

```bash
cd video-incident-monitor
npm install
```

## Running Locally

### Start Backend Server

```bash
python backend_odata_service.py
```

Backend will run on `http://localhost:5000`

### Start Frontend Server

**Option 1**: Using UI5 CLI (recommended if installed):

```bash
cd video-incident-monitor
ui5 serve --port 8080
```

**Option 2**: Using any static file server:

```bash
cd video-incident-monitor/webapp
python -m http.server 8080
```

**Option 3**: Using npx serve:

```bash
cd video-incident-monitor/webapp
npx serve -p 8080
```

Frontend will be available at:
- **Main App**: `http://localhost:8080/analyze-standalone.html`
- **UI5 Examples**: `http://localhost:8080/test/flpSandbox.html`
- **Index** (redirects to main): `http://localhost:8080/`

## Deployment to Cloud Foundry

### 1. Login to Cloud Foundry

```bash
cf login -a https://api.cf.eu10-004.hana.ondemand.com -o <org> -s <space>
```

### 2. Deploy Both Applications

Deploy frontend and backend together:

```bash
cf push --var api_key="your-secure-api-key"
```

This deploys:
- **Frontend** (`video-incident-monitor`): `https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com`
- **Backend** (`video-incident-monitor-backend`): `https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com`

### 3. Deploy Only Frontend

```bash
cf push video-incident-monitor
```

### 4. Deploy Only Backend

```bash
cf push video-incident-monitor-backend --var api_key="your-secure-api-key"
```

### 5. Verify Deployment

```bash
# Check application status
cf apps

# View backend logs
cf logs video-incident-monitor-backend --recent

# View frontend logs
cf logs video-incident-monitor --recent
```

## Configuration

### Backend Configuration (manifest.yaml)

```yaml
applications:
  - name: video-incident-monitor-backend
    buildpacks:
      - python_buildpack
    stack: cflinuxfs4
    memory: 512M          # Increased for video processing
    disk_quota: 1G        # Increased for temporary file storage
    path: .
    env:
      AICORE_AUTH_URL: https://dts-ai-core.authentication.eu10.hana.ondemand.com
      AICORE_CLIENT_ID: sb-xxxxx|aicore!b540
      AICORE_CLIENT_SECRET: xxxxx$...
      AICORE_BASE_URL: https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2
      AICORE_RESOURCE_GROUP: default
      API_KEY: ((api_key))  # Passed via --var
      ALLOWED_ORIGIN: https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com
    health-check-type: port
```

### Frontend Configuration (config.js)

The frontend automatically detects the environment:
- **Local**: Uses `http://localhost:5000`
- **Cloud Foundry**: Uses `https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com`
- **Override**: Add `<meta name="backend-base-url" content="https://custom-backend.example.com">`

API key can be provided via:
1. **URL parameter**: `?api_key=your-key` (highest priority)
2. **Meta tag**: `<meta name="backend-api-key" content="your-key">`
3. **Session storage**: Persists across page reloads

### Staticfile Configuration (webapp/Staticfile)

```
root: .
pushstate: enabled
```

Enables client-side routing with fallback to index.html.

## Usage

### Analyzing Media Files

1. Open the application at `https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/`
2. Select media type: **Video** or **Audio**
3. Choose file source:
   - **Server Files**: Select from pre-loaded examples (warehouse-incident.mp4, factory-noise.wav, etc.)
   - **Upload**: Upload your own media file (max 100MB)
4. Configure analysis parameters:
   - **Prompt**: Custom instructions for AI analysis (default: safety incident detection)
   - **Temperature**: Creativity level 0.0-2.0 (default: 1.0)
   - **Max Tokens**: Response length limit (default: 8192)
5. Click **"Analyze Media"**
6. View results:
   - Incident type and description
   - Severity level (Critical, High, Medium, Low, None)
   - Detected incidents list
   - Token usage statistics
   - Processing time
7. Click **"Download Report"** for PDF export

### Viewing UI5 Component Examples

Click **"View UI5 Component Examples"** at the bottom of the page to access the FLP Sandbox with various UI5 component demonstrations and tile layouts.

## API Endpoints

### Backend API

**Base URL**: `https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com`

#### Analyze Media

```http
POST /odata/v4/VideoIncidentService/MediaAnalysis
Content-Type: multipart/form-data
X-API-Key: your-api-key

Form fields:
- file: Binary file (video/audio)
- instruction: Analysis prompt (optional)
- temperature: Float 0.0-2.0 (optional, default: 1.0)
- maxTokens: Integer (optional, default: 8192)
- autoAnalyze: Boolean (optional, default: true)
```

**Response**:
```json
{
  "incidentType": "Safety Violation",
  "description": "Worker not wearing proper PPE...",
  "severity": "High",
  "incidents": ["Missing hard hat", "Unsafe ladder usage"],
  "inputTokens": 245680,
  "outputTokens": 1542,
  "processingTimeMs": 8234
}
```

#### List Sample Media

```http
GET /sample-media/video
GET /sample-media/audio
```

**Response**:
```json
[
  {
    "id": "warehouse-incident.mp4",
    "name": "warehouse-incident.mp4",
    "type": "video",
    "size": 12456789
  }
]
```

#### Download Sample Media

```http
GET /sample-media/video/warehouse-incident.mp4
GET /sample-media/audio/factory-noise.wav
```

Returns binary file stream.

## Security Considerations

- **API Key Authentication**: Backend requires `X-API-Key` header for all requests
- **CORS Protection**: Configured `ALLOWED_ORIGIN` restricts cross-origin requests
- **Environment Variables**: Sensitive credentials stored in environment, not code
- **OAuth 2.0**: AI Core authentication using client credentials flow
- **File Validation**: Media type and size validation on upload (max 100MB)
- **HTTPS Enforcement**: Cloud Foundry routes use HTTPS by default
- **No Credentials in Code**: All secrets managed via CF environment variables

## Troubleshooting

### CORS Errors in Cloud Foundry

**Problem**: Browser shows CORS error when calling backend API.

**Solution**: Ensure `ALLOWED_ORIGIN` in manifest.yaml matches your frontend URL:
```yaml
env:
  ALLOWED_ORIGIN: https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com
```

### Backend 401 Unauthorized

**Problem**: API calls return 401 error.

**Solution**: Check that API key is properly configured:
1. In `analyze-standalone.html`: `<meta name="backend-api-key" content="74646474849">`
2. Or pass via URL: `?api_key=74646474849`
3. Or store in sessionStorage

### Memory Issues (Backend)

**Problem**: Backend crashes with `SIGABRT` or out-of-memory errors when processing large videos.

**Solution**: Increase memory allocation in manifest.yaml:
```yaml
memory: 512M  # or higher (1G, 2G) for very large files
disk_quota: 1G
```

Then restage:
```bash
cf restage video-incident-monitor-backend
```

### UI5 Resources Not Loading

**Problem**: Blank page or "sap is not defined" errors.

**Solution**: Verify CDN URLs are accessible:
- UI5 Core: `https://ui5.sap.com/1.136.7/resources/sap-ui-core.js`
- UShell Sandbox: `https://ui5.sap.com/1.136.7/test-resources/sap/ushell/bootstrap/sandbox.js`

Check browser console for specific CDN errors.

### FLP Sandbox Shows Errors

**Problem**: `/test/flpSandbox.html` not loading or redirecting incorrectly.

**Solution**:
1. Ensure `test/` folder is not in `.cfignore`
2. Verify `pushstate: enabled` in Staticfile
3. Access directly: `https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/test/flpSandbox.html`

### Redirect Loop on Index

**Problem**: Index page keeps redirecting.

**Solution**: This is expected behavior. Index redirects to `analyze-standalone.html` via:
```html
<meta http-equiv="refresh" content="0; url=analyze-standalone.html">
```

To disable, modify `index.html`.

## Development with MCP Servers

This project was developed using **Claude Code** with **MCP (Model Context Protocol)** servers for enhanced SAP Fiori/UI5 development capabilities.

### Available MCP Servers

- **fiori-mcp**: SAP Fiori/UI5 development assistance
  - Application scaffolding
  - Component generation
  - Best practices guidance
  - Code snippets and examples

See [Prompt.md](Prompt.md) for detailed guide on:
- Setting up MCP servers for SAP development
- Generating similar full-stack applications
- Best practices for Fiori + AI Core integration
- Complete development workflow from design to deployment

## Performance Optimization

### Frontend Optimization
- UI5 resources loaded from CDN (cached globally)
- resourceroots configured for local module loading
- Pushstate routing for SPA-like navigation
- Lazy loading of UI5 libraries

### Backend Optimization
- Gunicorn with Uvicorn workers for async processing
- OAuth token caching to reduce authentication overhead
- Temporary file cleanup after processing
- CORS preflight optimization

## Testing

### Local Testing

1. Test backend endpoints:
```bash
curl -X POST http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis \
  -H "X-API-Key: your-api-key" \
  -F "file=@test-video.mp4" \
  -F "instruction=Detect safety violations"
```

2. Test frontend locally at `http://localhost:8080/analyze-standalone.html`

### Cloud Foundry Testing

1. Test deployed backend:
```bash
curl https://video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com/sample-media/video
```

2. Test frontend at `https://video-incident-monitor.cfapps.eu10-004.hana.ondemand.com/`

## Monitoring

### View Logs

```bash
# Backend logs (real-time)
cf logs video-incident-monitor-backend

# Backend logs (recent)
cf logs video-incident-monitor-backend --recent

# Frontend logs
cf logs video-incident-monitor --recent
```

### Check Application Health

```bash
# View app status
cf app video-incident-monitor-backend

# View app metrics
cf app video-incident-monitor-backend --guid
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Test locally and in Cloud Foundry
5. Commit: `git commit -m "Add new feature"`
6. Push: `git push origin feature/new-feature`
7. Submit a pull request

## License

Internal prototype for AI Core Video Incident Monitoring. Adapt as needed for production usage.

## Support

For issues and questions:
- Create an issue in the repository
- Contact the development team
- Refer to documentation below

## Related Documentation

- [SAP UI5 Documentation](https://ui5.sap.com/)
- [SAP BTP AI Core Documentation](https://help.sap.com/docs/sap-ai-core)
- [Cloud Foundry Documentation](https://docs.cloudfoundry.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MCP Usage Guide (English)](MCP_Usage_Guide_EN.md)
- [MCP Usage Guide (Russian)](MCP_Usage_Guide_RU.md)
- [Development Guide with MCP](Prompt.md)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)

## Changelog

### Version 1.1 (Latest)
- Added SAP BTP AI Core integration
- Implemented API key authentication
- Increased backend memory to 512M
- Added UI5 component examples page
- Fixed CORS issues in Cloud Foundry
- Added automatic redirect from root to main app
- Enhanced error handling and logging
- Updated to SAP UI5 1.136.7

### Version 1.0 (Initial)
- Basic video/audio analysis functionality
- FastAPI backend with Gemini integration
- SAP UI5 frontend
- Cloud Foundry deployment support
