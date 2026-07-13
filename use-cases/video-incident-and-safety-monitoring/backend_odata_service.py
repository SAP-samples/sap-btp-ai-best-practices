"""
Python OData V4 Service for Video Incident Monitoring
Integrates with SAP Fiori frontend and Gemini AI backend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
import os
import json
import requests
import base64
import mimetypes
from pathlib import Path
import traceback

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Configuration
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL")
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
AICORE_BASE_URL = os.getenv("AICORE_BASE_URL")
AICORE_RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP", "default")
DEPLOYMENT_ID = "dc573586f4d0d974"
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
API_KEY = os.getenv("API_KEY")

# Directories
VIDEO_DIR = Path(__file__).parent / "Video"
AUDIO_DIR = Path(__file__).parent / "Audio"
VIDEO_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="Video Incident Monitoring OData Service",
    description="OData V4 service for SAP Fiori Video Incident & Safety Monitoring",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Data Models =================


class MediaAnalysis(BaseModel):
    ID: UUID = Field(default_factory=uuid4)
    fileName: Optional[str] = None
    fileType: Optional[str] = None  # 'video' | 'audio'
    mimeType: Optional[str] = None
    fileSize: Optional[int] = None
    filePath: Optional[str] = None
    uploadedAt: Optional[datetime] = Field(default_factory=datetime.now)

    # Analysis parameters
    instruction: Optional[str] = None
    temperature: Optional[float] = 0.7
    maxTokens: Optional[int] = 2000

    # Analysis results
    status: Optional[str] = "pending"  # pending | processing | completed | failed
    analysisResult: Optional[str] = None
    incidentDetected: Optional[bool] = False
    incidentType: Optional[str] = None
    severity: Optional[str] = None  # low | medium | high | critical

    # Metrics
    promptTokens: Optional[int] = 0
    completionTokens: Optional[int] = 0
    totalTokens: Optional[int] = 0
    processingTime: Optional[int] = 0

    analyzedAt: Optional[datetime] = None
    analyzedBy: Optional[str] = "System"

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


# In-memory storage (replace with database in production)
media_analyses: List[MediaAnalysis] = []


# ================= Helper Functions =================


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple API key check. Set API_KEY env to enable; if not set, allow all."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_token() -> str:
    """Fetches OAuth token from AI Core."""
    url = f"{AICORE_AUTH_URL}/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
    data = {
        "client_id": AICORE_CLIENT_ID,
        "client_secret": AICORE_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]


def call_gemini_with_file(
    token: str,
    file_path: str,
    question: str,
    mime_type: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> dict:
    """Calls Gemini 2.5 Pro with video/audio file."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "AI-Resource-Group": AICORE_RESOURCE_GROUP,
    }

    url = f"{AICORE_BASE_URL}/inference/deployments/{DEPLOYMENT_ID}/models/gemini-2.5-pro:generateContent"

    with open(file_path, "rb") as f:
        file_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "generation_config": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.95,
        },
        "contents": {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": file_data}},
                {"text": question},
            ],
        },
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    resp = requests.post(url, headers=headers, data=body, timeout=120)
    resp.raise_for_status()

    return resp.json()


def extract_response(api_result: dict) -> str:
    """Extract text from Gemini response."""
    try:
        return api_result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        return f"Error extracting response: {e}"


def get_token_usage(api_result: dict) -> dict:
    """Extract token usage metadata."""
    usage = api_result.get("usageMetadata", {})
    return {
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0),
    }


def get_mime_type(file_path: str) -> str:
    """Detect MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(file_path)

    if not mime_type:
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".wav": "audio/wav",
            ".mp3": "audio/mp3",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")

    return mime_type


# ================= OData V4 Endpoints =================


@app.get("/")
async def root():
    return {
        "message": "Video Incident Monitoring OData Service",
        "version": "1.0.0",
        "endpoints": {
            "service_document": "/odata/v4/VideoIncidentService/",
            "metadata": "/odata/v4/VideoIncidentService/$metadata",
            "media_analysis": "/odata/v4/VideoIncidentService/MediaAnalysis",
        },
    }


@app.get("/odata/v4/VideoIncidentService/")
@app.head("/odata/v4/VideoIncidentService/")
async def service_document(api_key_ok: None = Depends(verify_api_key)):
    """OData Service Document"""
    return {
        "@odata.context": "$metadata",
        "value": [
            {"name": "MediaAnalysis", "kind": "EntitySet", "url": "MediaAnalysis"}
        ],
    }


@app.get("/odata/v4/VideoIncidentService/$metadata")
async def metadata(api_key_ok: None = Depends(verify_api_key)):
    """Return OData metadata XML"""
    metadata_path = Path(__file__).parent / "metadata.xml"

    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found")

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    return Response(content=content, media_type="application/xml")


@app.get("/odata/v4/VideoIncidentService/MediaAnalysis")
async def get_media_analyses(api_key_ok: None = Depends(verify_api_key)):
    """Get all media analyses (OData collection)"""
    return {
        "@odata.context": "$metadata#MediaAnalysis",
        "value": [analysis.model_dump() for analysis in media_analyses],
    }


@app.get("/odata/v4/VideoIncidentService/MediaAnalysis({id})")
async def get_media_analysis(id: str, api_key_ok: None = Depends(verify_api_key)):
    """Get single media analysis by ID"""
    try:
        analysis_id = UUID(id)
        for analysis in media_analyses:
            if analysis.ID == analysis_id:
                return {
                    "@odata.context": f"$metadata#MediaAnalysis/$entity",
                    **analysis.model_dump(),
                }
        raise HTTPException(
            status_code=404, detail=f"MediaAnalysis with ID {id} not found"
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")


@app.post("/odata/v4/VideoIncidentService/MediaAnalysis")
async def create_media_analysis(
    file: UploadFile = File(...),
    instruction: str = Form(None),
    temperature: float = Form(0.7),
    maxTokens: int = Form(2000),
    autoAnalyze: bool = Form(False),
    api_key_ok: None = Depends(verify_api_key),
):
    """Upload media file and optionally analyze it"""
    try:
        # Determine file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext in [".mp4", ".avi", ".mov", ".webm"]:
            file_type = "video"
            save_dir = VIDEO_DIR
            default_instruction = "Describe in detail what you see in the video and what is happening. Are there any safety incidents?"
        elif file_ext in [".wav", ".mp3", ".ogg", ".flac"]:
            file_type = "audio"
            save_dir = AUDIO_DIR
            default_instruction = "Recognize and transcribe what is said in this audio file. Write the text verbatim."
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_ext}"
            )

        # Save file
        file_path = save_dir / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Create analysis record
        analysis = MediaAnalysis(
            fileName=file.filename,
            fileType=file_type,
            mimeType=get_mime_type(str(file_path)),
            fileSize=len(content),
            filePath=str(file_path),
            instruction=instruction or default_instruction,
            temperature=temperature,
            maxTokens=maxTokens,
            status="pending",
        )

        # Analyze if requested
        if autoAnalyze:
            analysis.status = "processing"
            media_analyses.append(analysis)

            try:
                start_time = datetime.now()
                token = get_token()
                result = call_gemini_with_file(
                    token=token,
                    file_path=str(file_path),
                    question=analysis.instruction,
                    mime_type=analysis.mimeType,
                    temperature=analysis.temperature,
                    max_tokens=analysis.maxTokens,
                )

                # Extract results
                analysis.analysisResult = extract_response(result)
                usage = get_token_usage(result)
                analysis.promptTokens = usage["prompt_tokens"]
                analysis.completionTokens = usage["completion_tokens"]
                analysis.totalTokens = usage["total_tokens"]
                analysis.processingTime = int(
                    (datetime.now() - start_time).total_seconds()
                )
                analysis.analyzedAt = datetime.now()
                analysis.status = "completed"

                # Simple incident detection (check for keywords)
                incident_keywords = [
                    "incident",
                    "accident",
                    "violation",
                    "unsafe",
                    "missing",
                    "hazard",
                    "danger",
                ]
                analysis.incidentDetected = any(
                    keyword in analysis.analysisResult.lower()
                    for keyword in incident_keywords
                )

                if analysis.incidentDetected:
                    # Simple severity classification
                    critical_words = ["critical", "severe", "fatal", "emergency"]
                    high_words = ["high", "serious", "major", "significant"]
                    medium_words = ["medium", "moderate", "minor"]

                    result_lower = analysis.analysisResult.lower()
                    if any(word in result_lower for word in critical_words):
                        analysis.severity = "critical"
                    elif any(word in result_lower for word in high_words):
                        analysis.severity = "high"
                    elif any(word in result_lower for word in medium_words):
                        analysis.severity = "medium"
                    else:
                        analysis.severity = "low"

            except Exception as e:
                analysis.status = "failed"
                analysis.analysisResult = f"Analysis failed: {str(e)}"
                print(f"Analysis error: {traceback.format_exc()}")
        else:
            media_analyses.append(analysis)

        return {
            "@odata.context": "$metadata#MediaAnalysis/$entity",
            **analysis.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/odata/v4/VideoIncidentService/MediaAnalysis({id})/analyze")
async def analyze_media(id: str, api_key_ok: None = Depends(verify_api_key)):
    """Analyze an existing media file"""
    try:
        analysis_id = UUID(id)
        analysis = None

        for a in media_analyses:
            if a.ID == analysis_id:
                analysis = a
                break

        if not analysis:
            raise HTTPException(
                status_code=404, detail=f"MediaAnalysis with ID {id} not found"
            )

        if not Path(analysis.filePath).exists():
            raise HTTPException(status_code=404, detail="Media file not found on disk")

        # Update status
        analysis.status = "processing"

        try:
            start_time = datetime.now()
            token = get_token()
            result = call_gemini_with_file(
                token=token,
                file_path=analysis.filePath,
                question=analysis.instruction,
                mime_type=analysis.mimeType,
                temperature=analysis.temperature,
                max_tokens=analysis.maxTokens,
            )

            # Extract results
            analysis.analysisResult = extract_response(result)
            usage = get_token_usage(result)
            analysis.promptTokens = usage["prompt_tokens"]
            analysis.completionTokens = usage["completion_tokens"]
            analysis.totalTokens = usage["total_tokens"]
            analysis.processingTime = int((datetime.now() - start_time).total_seconds())
            analysis.analyzedAt = datetime.now()
            analysis.status = "completed"

            # Incident detection
            incident_keywords = [
                "incident",
                "accident",
                "violation",
                "unsafe",
                "missing",
                "hazard",
                "danger",
            ]
            analysis.incidentDetected = any(
                keyword in analysis.analysisResult.lower()
                for keyword in incident_keywords
            )

            if analysis.incidentDetected:
                critical_words = ["critical", "severe", "fatal", "emergency"]
                high_words = ["high", "serious", "major", "significant"]
                medium_words = ["medium", "moderate", "minor"]

                result_lower = analysis.analysisResult.lower()
                if any(word in result_lower for word in critical_words):
                    analysis.severity = "critical"
                elif any(word in result_lower for word in high_words):
                    analysis.severity = "high"
                elif any(word in result_lower for word in medium_words):
                    analysis.severity = "medium"
                else:
                    analysis.severity = "low"

            return {
                "@odata.context": f"$metadata#MediaAnalysis/$entity",
                **analysis.model_dump(),
            }

        except Exception as e:
            analysis.status = "failed"
            analysis.analysisResult = f"Analysis failed: {str(e)}"
            raise HTTPException(status_code=500, detail=str(e))

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")


@app.delete("/odata/v4/VideoIncidentService/MediaAnalysis({id})")
async def delete_media_analysis(id: str, api_key_ok: None = Depends(verify_api_key)):
    """Delete a media analysis"""
    try:
        analysis_id = UUID(id)
        for i, analysis in enumerate(media_analyses):
            if analysis.ID == analysis_id:
                # Delete file if exists
                if analysis.filePath and Path(analysis.filePath).exists():
                    Path(analysis.filePath).unlink()

                # Remove from list
                media_analyses.pop(i)
                return Response(status_code=204)

        raise HTTPException(
            status_code=404, detail=f"MediaAnalysis with ID {id} not found"
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")


# ================= Health & Info Endpoints =================


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Video Incident Monitoring OData Service",
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Video Incident Monitoring OData Service...")
    print(f"OData endpoint: http://localhost:5000/odata/v4/VideoIncidentService/")
    print(f"Metadata: http://localhost:5000/odata/v4/VideoIncidentService/$metadata")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
