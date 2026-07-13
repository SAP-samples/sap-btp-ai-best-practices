# Video Incident & Safety Monitoring - SAP Fiori Application

SAP Fiori List Report application for monitoring and analyzing video/audio incidents using AI powered by SAP BTP AI Core and Gemini 2.5 Pro.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         SAP Fiori Frontend (SAPUI5)                 │
│           video-incident-monitor/                   │
│  - List Report (table view)                         │
│  - Object Page (detail view)                        │
│  - Upload & Analysis                                │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│     Python OData V4 Service (FastAPI)               │
│        backend_odata_service.py                     │
│  - REST/OData endpoints                             │
│  - File upload & management                         │
│  - AI analysis orchestration                        │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│        SAP AI Core + Gemini 2.5 Pro                 │
│  - Video analysis                                   │
│  - Audio transcription                              │
│  - Incident detection                               │
└─────────────────────────────────────────────────────┘
```

## Features

### Frontend (Fiori)
- **List Report** - Table view of all media analyses with:
  - Filtering by file type, status, severity
  - Sorting by date, name, etc.
  - Status indicators (pending, processing, completed, failed)
  - Severity badges (low, medium, high, critical)

- **Object Page** - Detailed view with:
  - Media preview (video/audio player)
  - Analysis results
  - Technical metrics (tokens, processing time)
  - File information

- **Actions**
  - Upload new media files
  - Trigger analysis
  - Download results
  - Delete entries

### Backend (Python OData)
- **OData V4 Service** compliant with SAP standards
- **Entity: MediaAnalysis** with full CRUD operations
- **AI Integration** with SAP AI Core & Gemini
- **Incident Detection** with severity classification
- **Token Tracking** for cost monitoring

## Quick Start

### Prerequisites

1. **Node.js** (v18+) and npm
2. **Python** (3.9+)
3. **SAP AI Core** credentials

### 1. Configure Environment

Create `.env` file in the root directory:

```env
AICORE_AUTH_URL=https://dts-ai-core.authentication.eu10.hana.ondemand.com
AICORE_CLIENT_ID=your_client_id
AICORE_CLIENT_SECRET=your_client_secret
AICORE_BASE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2
AICORE_RESOURCE_GROUP=default
```

### 2. Start Backend (Terminal 1)

**Windows:**
```bash
start_backend.bat
```

**Manual:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend_requirements.txt
python backend_odata_service.py
```

Backend will start on: `http://localhost:5000`

### 3. Start Frontend (Terminal 2)

**Windows:**
```bash
start_frontend.bat
```

**Manual:**
```bash
cd video-incident-monitor
npm run start
```

Frontend will start on: `http://localhost:8080`

## Usage

### 1. Upload Media File

1. Open the Fiori app: `http://localhost:8080`
2. Click "Upload" button
3. Select video (MP4, AVI, MOV, WEBM) or audio (WAV, MP3, OGG, FLAC)
4. Optionally configure:
   - Custom instruction for analysis
   - Temperature (0.0-2.0)
   - Max tokens
5. Choose "Auto-analyze" to analyze immediately
6. Click "Upload"

### 2. Analyze Existing File

1. In the List Report, select a media entry
2. Navigate to Object Page
3. Click "Analyze" button
4. Wait for analysis to complete (status updates automatically)

### 3. View Results

1. Analysis results appear in the Object Page
2. Check for:
   - **Incident Detected**: Yes/No
   - **Severity**: Low, Medium, High, Critical
   - **Analysis Text**: Full AI response
   - **Metrics**: Tokens used, processing time

### 4. Filter & Search

Use the Smart Filter Bar to:
- Search by file name
- Filter by file type (video/audio)
- Filter by status (pending/completed/failed)
- Filter by severity level
- Filter by incident detected

## OData Endpoints

### Service Root
```
GET http://localhost:5000/odata/v4/VideoIncidentService/
```

### Metadata
```
GET http://localhost:5000/odata/v4/VideoIncidentService/$metadata
```

### Get All Analyses
```
GET http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis
```

### Get Single Analysis
```
GET http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis({id})
```

### Upload & Create
```
POST http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis
Content-Type: multipart/form-data

file: [binary]
instruction: "Analyze this video for safety violations"
temperature: 0.7
maxTokens: 2000
autoAnalyze: true
```

### Trigger Analysis
```
POST http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis({id})/analyze
```

### Delete
```
DELETE http://localhost:5000/odata/v4/VideoIncidentService/MediaAnalysis({id})
```

## Entity Model: MediaAnalysis

```typescript
{
  ID: UUID;

  // File Info
  fileName: string;
  fileType: 'video' | 'audio';
  mimeType: string;
  fileSize: number;
  filePath: string;
  uploadedAt: DateTime;

  // Analysis Parameters
  instruction: string;
  temperature: number;
  maxTokens: number;

  // Results
  status: 'pending' | 'processing' | 'completed' | 'failed';
  analysisResult: string;
  incidentDetected: boolean;
  incidentType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';

  // Metrics
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  processingTime: number;  // seconds

  analyzedAt: DateTime;
  analyzedBy: string;
}
```

## Fiori Elements Used

### Layout
- **sap.f.DynamicPage** - List Report layout
- **sap.uxap.ObjectPageLayout** - Detail page layout
- **sap.ui.layout.Grid** - Responsive grid

### Controls
- **sap.m.Table** - Media analyses table
- **sap.m.ObjectStatus** - Status & severity indicators
- **sap.m.ProgressIndicator** - Processing progress
- **sap.ui.unified.FileUploader** - File upload
- **sap.m.MessageStrip** - Notifications
- **sap.m.IconTabBar** - Tab navigation

### Annotations (in metadata.xml)
- `@UI.HeaderInfo` - Object page header
- `@UI.LineItem` - Table columns
- `@UI.SelectionFields` - Filter fields
- `@UI.FieldGroup` - Field grouping
- `@UI.Facet` - Object page sections
- `@UI.DataPoint` - KPI values

## Development

### Project Structure

```
AI CORE Video template/
│
├── video-incident-monitor/          # Fiori frontend
│   ├── webapp/
│   │   ├── manifest.json            # App descriptor
│   │   ├── Component.js             # Root component
│   │   └── view/                    # Views (auto-generated)
│   ├── package.json
│   └── ui5.yaml
│
├── backend_odata_service.py         # Python OData service
├── backend_requirements.txt         # Python dependencies
├── metadata.xml                     # OData metadata
│
├── App_prototype/                   # Original Streamlit app
│   └── app_multimodal_gemini.py
│
├── Video/                           # Video storage
├── Audio/                           # Audio storage
│
├── start_backend.bat                # Start backend
├── start_frontend.bat               # Start frontend
│
└── .env                             # Configuration
```

### Backend Customization

To modify the OData service:

1. **Add new endpoints**: Edit `backend_odata_service.py`
2. **Change entity model**: Update `MediaAnalysis` class and `metadata.xml`
3. **Modify AI logic**: Edit `call_gemini_with_file()` and incident detection

### Frontend Customization

To modify the Fiori app:

1. **Add fields**: Edit `metadata.xml` annotations
2. **Custom actions**: Use `mcp__fiori-mcp__add-page` functionality
3. **Styling**: Modify `webapp/manifest.json` theme settings

## Deployment

### SAP BTP Cloud Foundry

1. Create `manifest.yml` for multi-target application:

```yaml
applications:
- name: video-incident-backend
  path: .
  buildpacks:
    - python_buildpack
  command: python backend_odata_service.py
  memory: 512M

- name: video-incident-frontend
  path: video-incident-monitor
  buildpacks:
    - nodejs_buildpack
  command: npm run start
  memory: 256M
```

2. Deploy:
```bash
cf push
```

### Docker

Create `Dockerfile` for backend:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY backend_requirements.txt .
RUN pip install -r backend_requirements.txt
COPY . .
CMD ["python", "backend_odata_service.py"]
```

## Troubleshooting

### Backend Issues

**Error: "Missing environment variables"**
- Check `.env` file exists
- Verify all AI Core credentials are set

**Error: "Connection refused to AI Core"**
- Verify AI Core deployment is active
- Check `DEPLOYMENT_ID` in code

**Error: "File not found"**
- Ensure `Video/` and `Audio/` folders exist
- Check file permissions

### Frontend Issues

**Error: "OData service not responding"**
- Start backend first: `start_backend.bat`
- Verify backend is running on port 5000

**Error: "Metadata not found"**
- Ensure `metadata.xml` exists in root directory
- Check metadata endpoint: http://localhost:5000/odata/v4/VideoIncidentService/$metadata

**UI not loading**
- Run `npm install` in `video-incident-monitor/`
- Clear browser cache
- Check console for errors

## Performance Tips

### Video Files
- Keep files under 50MB for faster processing
- Use MP4 (H.264) format
- Resolution: 720p or lower
- Duration: < 5 minutes

### Audio Files
- Use WAV format for best quality
- Sample rate: 16 kHz for speech
- Keep files under 20MB
- Mono channel for transcription

## Security

- **Authentication**: Add OAuth/JWT for production
- **File Upload**: Validate file types and sizes
- **CORS**: Restrict origins in production
- **Secrets**: Use SAP BTP Secret Store or Vault
- **Rate Limiting**: Add throttling for API calls

## Next Steps

1. Add database persistence (PostgreSQL/HANA)
2. Implement user authentication
3. Add batch processing for multiple files
4. Create dashboard with analytics
5. Add export functionality (PDF reports)
6. Integrate with SAP Workflow for incident handling
7. Add real-time notifications (WebSockets)

## Support

For issues or questions:
1. Check logs: Backend console and browser DevTools
2. Review OData metadata: `/odata/v4/VideoIncidentService/$metadata`
3. Test endpoints with Postman/cURL
4. Check SAP AI Core deployment status

## License

This project is based on SAP Fiori Elements and uses SAP BTP AI Core services.
