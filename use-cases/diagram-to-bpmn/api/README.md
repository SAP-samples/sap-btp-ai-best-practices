# Diagram to BPMN API

FastAPI backend that transforms business process diagram images into BPMN 2.0 XML files ready for import into Signavio or SAP Build Process Automation. The service uses a sophisticated three-stage pipeline that leverages Large Language Models (LLMs) via SAP GenAI Hub to understand diagrams and generate compliant BPMN XML.

---

## Architecture Overview

The API implements a **deterministic three-stage pipeline** for robust BPMN generation:

1. **Stage A: Vision → Graph JSON** - LLM analyzes the diagram image and extracts structured graph data
2. **Stage B: Graph JSON → BPMN XML** - LLM converts the graph structure into complete BPMN 2.0 XML with DI
3. **Stage C: BPMN Validation** - Always validates and corrects BPMN against Graph JSON structure

### Complete Workflow

```
Image Upload → API Validation → Stage A (Vision→Graph) → Stage B (Graph→BPMN) → Stage C (Validation) → Response
```

**Stage A Details:**
- Input: Process diagram image (PNG, JPG, JPEG, WEBP, SVG)
- Process: LLM with vision capabilities analyzes the image
- Output: Structured JSON graph containing lanes, nodes, and edges
- Schema: Validates against predefined JSON schema for process elements

**Stage B Details:**
- Input: Graph JSON from Stage A
- Process: LLM converts graph structure to complete BPMN 2.0 XML with DI
- Output: BPMN XML with semantic elements and complete Diagram Interchange
- Focus: Both semantic BPMN elements and visual layout (DI shapes and edges)

**Stage C Details:**
- Input: Graph JSON from Stage A + BPMN XML from Stage B
- Process: Validates BPMN structure against Graph JSON, ensures completeness and correctness
- Output: Validated and corrected BPMN XML ready for Signavio/SAP Build Process Automation
- Trigger: Always runs (deterministic pipeline)
- Validation: Structural integrity, reference validity, DI completeness, XML well-formedness

---

## Prerequisites

- Python 3.10+ (for local development)
- Access credentials for SAP GenAI Hub (`AICORE_*` values)
- Generated API key shared with the Streamlit UI or external clients
- Network access to the selected LLM providers (OpenAI, Anthropic, Google Vertex)

Create a virtual environment and install dependencies:

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

The API reads secrets and runtime configuration from environment variables. Copy the template to `.env` and fill in your values:

```bash
cp .env.example .env
```

**Required Environment Variables:**
- `API_KEY`: Authentication key for API access (required)
- `AICORE`: SAP GenAI Hub credentials for LLM access

---

## Running Locally

```bash
uvicorn app.main:app --reload
```

The server defaults to port `8000`. Swagger UI is available at `http://localhost:8000/docs`.

---

## API Endpoints

### Health Check

`GET /api/health`

Simple readiness probe returning service status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "service": "api"
}
```

**Example:**
```bash
curl http://localhost:8000/api/health
```

### Generate BPMN from Diagram

`POST /api/bpmn/generate`

**Primary endpoint** for the complete BPMN generation workflow. Accepts a process diagram image and optional LLM selection inputs.

**Request Parameters:**
- `file` (required): Image file (`png`, `jpg`, `jpeg`, `webp`, `svg+xml`)
- `provider` (optional): `anthropic` | `openai` | `gemini` (default: `anthropic`)
- `model` (optional): Specific model name for the chosen provider

**Supported Providers and Models:**
- **Anthropic**: `anthropic--claude-4-sonnet` (default)
- **OpenAI**: `gpt-4.1`, `gpt-5` (with reasoning mode)
- **Google Vertex AI**: `gemini-2.5-pro`

**Response Format:**
```json
{
  "bpmn_xml": "<definitions>...</definitions>",
  "provider": "anthropic",
  "model": "anthropic--claude-4-sonnet",
  "success": true,
  "usage": {
    "stage_a": {"prompt_tokens": 1000, "completion_tokens": 500},
    "stage_b": {"prompt_tokens": 800, "completion_tokens": 1200},
    "stage_c": {"prompt_tokens": 600, "completion_tokens": 400}
  },
  "error": null
}
```

**Usage Statistics:**
- `stage_a`: Token usage for image analysis (Vision → Graph)
- `stage_b`: Token usage for BPMN generation (Graph → BPMN with DI)
- `stage_c`: Token usage for validation (always included)

---

## Usage Examples

### 1. Basic cURL Request

```bash
curl -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -H "accept: application/json" \
  -F "file=@diagrams/sample-process.png"
```

### 2. Advanced cURL with Provider Selection

```bash
curl -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -H "accept: application/json" \
  -F "provider=openai" \
  -F "model=gpt-5" \
  -F "file=@diagrams/sample-process.png"
```

### 3. Save Generated BPMN to File

```bash
curl -s -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -F "provider=anthropic" \
  -F "file=@diagrams/sample-process.png" \
  | jq -r '.bpmn_xml' > output/process.bpmn
```

### 4. Python Integration

```python
import requests

API_URL = "http://localhost:8000/api/bpmn/generate"
API_KEY = "your-api-key"

def generate_bpmn(image_path, provider="anthropic", model=None):
    with open(image_path, "rb") as fh:
        files = {"file": (image_path, fh, "image/png")}
        data = {"provider": provider}
        if model:
            data["model"] = model
        headers = {"X-API-Key": API_KEY}

        response = requests.post(
            API_URL, 
            headers=headers, 
            data=data, 
            files=files, 
            timeout=900  # 15 minutes for long-running BPMN generation
        )
        response.raise_for_status()
        payload = response.json()
        
        if payload["success"]:
            return payload["bpmn_xml"], payload["usage"]
        else:
            raise RuntimeError(f"Generation failed: {payload.get('error')}")

# Usage
bpmn_xml, usage = generate_bpmn("diagrams/sample-process.png", "openai", "gpt-5")
print(f"Generated BPMN with {usage['stage_b']['total_tokens']} tokens")
```

### 5. Streamlit UI Integration

The UI automatically handles the complete workflow:

```python
# From ui/src/api_client.py
response = generate_bpmn_from_image(
    file_bytes=uploaded_file.getvalue(),
    filename=uploaded_file.name,
    content_type=uploaded_file.type,
    provider=selected_provider,
    model=selected_model
)
```

---

## Technical Implementation Details

### File Processing Pipeline

1. **File Validation**: Checks MIME type against allowed formats
2. **Image Processing**: Reads file bytes for LLM processing
3. **Provider Resolution**: Normalizes provider and model selection
4. **Multi-Stage Processing**: Executes the three-stage pipeline
5. **Response Assembly**: Formats final response with metadata

### LLM Integration

- **SAP GenAI Hub**: Centralized access to multiple LLM providers
- **Provider Abstraction**: Unified interface for OpenAI, Anthropic, and Google Vertex AI
- **Model Selection**: Automatic fallback to provider defaults
- **Temperature Control**: Non-reasoning models use temperature=0, reasoning models (GPT-5) use reasoning mode

### BPMN Compliance

- **BPMN 2.0 Standard**: Generates XML compliant with BPMN 2.0 specification
- **Signavio Compatibility**: Optimized for Signavio Process Manager import
- **SAP Build Integration**: Compatible with SAP Build Process Automation
- **Diagram Interchange**: Includes BPMN DI elements for visual representation

---

## Deployment Notes

- **Environment Variables**: Ensure all required environment variables are set
- **CORS Configuration**: Configure `ALLOWED_ORIGIN` for production UI access
- **Logging**: Set appropriate log levels via `UVICORN_LOG_LEVEL`
- **Port Configuration**: Cloud Foundry sets `PORT` environment variable
- **Health Checks**: Use `/api/health` endpoint for load balancer health checks

---

## Testing and Validation

### Manual Testing Steps

1. **Health Check**: Verify `/api/health` endpoint responds correctly
2. **Provider Testing**: Test each provider (Anthropic, OpenAI, Gemini) with sample diagrams
3. **Model Testing**: Verify different models work correctly within each provider
4. **BPMN Validation**: Import generated XML into Signavio/SAP Build to verify compatibility
5. **Error Handling**: Test with invalid files, missing API keys, and network failures

### Sample Test Commands

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# Test with each provider
curl -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -F "provider=anthropic" \
  -F "file=@test-diagram.png"

curl -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -F "provider=openai" \
  -F "model=gpt-4.1" \
  -F "file=@test-diagram.png"

curl -X POST "http://localhost:8000/api/bpmn/generate" \
  -H "X-API-Key: $API_KEY" \
  -F "provider=gemini" \
  -F "file=@test-diagram.png"
```

---

## Integration with Streamlit UI

The API is designed to work seamlessly with the Streamlit frontend:

- **Authentication**: Shared API key via `X-API-Key` header
- **File Upload**: Multipart form data handling for image uploads
- **Real-time Processing**: Long-running requests with progress indication
- **Error Display**: Structured error responses for user-friendly display
- **Token Usage**: Detailed usage statistics for cost tracking

The UI automatically handles the complete workflow from image upload to BPMN download, providing a user-friendly interface for the sophisticated three-stage processing pipeline.
