"""
Document extraction API router (SAP DOX integration).

This router replaces prior LLM-based extraction with SAP Document Information
Extraction (DOX). It uploads PDFs to DOX using predefined schemas (by ID/version)
and returns a normalized response compatible with the existing UI.

Key document types supported via DOX schemas:
- conoce_cliente (KYC)
- comentarios_vendedor
- constancia_fiscal (CSF)
- cgv

Service key for DOX is read from the environment variable DOX_SERVICE_KEY.
Fallbacks are attempted under api/routers/tools/ (service_key.json or
service_key2.json).
"""

import logging
import asyncio
import tempfile
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse

# Import models
from models import (
    DocumentType,
    ExtractionResponse,
    ExtractionResult,
    BatchExtractionResponse,
    DocumentSchema,
    DocumentSchemasResponse,
    UpdateSchemaRequest,
    TaskStatusResponse
)

# Import SAP DOX client from tools/ only
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTERS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = ROUTERS_DIR / "tools"
import sys as _sys
if str(TOOLS_DIR) not in _sys.path:
    _sys.path.append(str(TOOLS_DIR))
try:
    from sap_dox_client import SapDoxClient, ServiceKey  # type: ignore
except Exception as _e:  # pragma: no cover
    SapDoxClient = None  # Fallback to allow import-time errors to be reported at runtime

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for schemas and tasks (in production, use Redis or database)
DOCUMENT_SCHEMAS: Dict[str, Dict[str, Any]] = {}

# Task storage
TASKS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# DOX client helpers
# -----------------------------

_DOX_CLIENT: Optional[SapDoxClient] = None


def _resolve_service_key_path() -> str:
    """Resolve the DOX service key path.

    Order:
    1) DOX_SERVICE_KEY env var (absolute or relative to project root)
    2) api/routers/tools/service_key.json
    3) api/routers/tools/service_key2.json
    """
    # 1) Environment variable
    env_path = os.getenv("DOX_SERVICE_KEY")
    if env_path:
        # Expand and resolve relative paths against project root
        candidate = Path(env_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return str(candidate)

    # 2) Default locations under tools/
    for name in ("service_key.json", "service_key2.json"):
        candidate = TOOLS_DIR / name
        if candidate.is_file():
            return str(candidate)

    raise RuntimeError(
        "DOX service key file not found. Set DOX_SERVICE_KEY env var or place service_key.json in DOX/."
    )


def get_dox_client() -> SapDoxClient:
    """Return a cached SapDoxClient instance loaded from the service key.

    Raises a clear error if the sap_dox_client module is not available or if
    the service key cannot be located.
    """
    global _DOX_CLIENT
    if _DOX_CLIENT is not None:
        return _DOX_CLIENT
    if SapDoxClient is None:  # pragma: no cover
        raise RuntimeError("sap_dox_client is not available. Ensure DOX/ is present and importable.")

    # Prefer environment variables when provided
    env_uaa_url = os.getenv("DOX_UAA_URL") or os.getenv("DOX_TOKEN_BASE_URL")
    env_client_id = os.getenv("DOX_CLIENT_ID")
    env_client_secret = os.getenv("DOX_CLIENT_SECRET")
    env_base_url = os.getenv("DOX_BASE_URL")
    env_swagger = os.getenv("DOX_SWAGGER_PATH", "/document-information-extraction/v1/")

    if env_uaa_url and env_client_id and env_client_secret and env_base_url:
        logger.info("Initializing SAP DOX client from environment variables")
        service_key = ServiceKey(
            token_base_url=env_uaa_url,
            client_id=env_client_id,
            client_secret=env_client_secret,
            dox_base_url=env_base_url,
            swagger_path=env_swagger,
        )
        _DOX_CLIENT = SapDoxClient(service_key)
        return _DOX_CLIENT

    # Fallback to service key JSON file
    sk_path = _resolve_service_key_path()
    logger.info(f"Initializing SAP DOX client using service key: {sk_path}")
    _DOX_CLIENT = SapDoxClient.from_service_key(sk_path)
    return _DOX_CLIENT


def _load_schema_file(path: Path) -> Dict[str, Any]:
    """Load a DOX schema JSON file and return its dict contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_dox_schema_mappings() -> Dict[str, Dict[str, Any]]:
    """Return mapping of DocumentType.value -> DOX schema metadata.

    Each entry contains:
      - schema_id: str
      - schema_name: str
      - schema_version: str
      - field_display_map: Mapping from DOX field 'name' -> UI display name
    """
    mappings: Dict[str, Dict[str, Any]] = {}

    def _schema_path(filename: str) -> Path:
        """Find schema JSON path in tools/."""
        preferred = TOOLS_DIR / filename
        if preferred.is_file():
            return preferred
        raise FileNotFoundError(f"Schema file not found in tools/: {filename}")

    try:
        # KYC
        kyc_json = _load_schema_file(_schema_path("Conoce_cliente_schema.json"))
        kyc_map = {item["name"]: item.get("label") or item["name"] for item in kyc_json.get("headerFields", [])}
        # Override labels to match UI and verification expectations
        kyc_overrides = {
            "Domicilio": "Domicilio Fiscal (Información Fiscal)",
            "Linea_Credito_Actual": "Línea de Crédito Actual",
            "Plazo_Credito_Actual": "Plazo de Crédito Actual",
            "Linea_credito_solicitada": "Línea de Crédito Solicitada",
            "Plazo_Credito_solicitado": "Plazo de Crédito Solicitado",
            "Productos_vendidos": "Productos Vendidos",
        }
        kyc_map.update(kyc_overrides)
        mappings[DocumentType.CONOCE_CLIENTE.value] = {
            "schema_id": kyc_json.get("id"),
            "schema_name": kyc_json.get("name"),
            "schema_version": str(kyc_json.get("version", "1")),
            "field_display_map": kyc_map,
            "title": "Conoce a tu Cliente",
        }

        # Comentarios del Vendedor
        vend_json = _load_schema_file(_schema_path("Comentarios_vendedor_schema.json"))
        vend_map = {item["name"]: item.get("label") or item["name"] for item in vend_json.get("headerFields", [])}
        vend_overrides = {
            "Client_name": "Nombre del Cliente",
            "id_SAP_number": "ID Cliente SAP",
            "Linea_credito_actual": "Línea de Crédito Actual",
            "plazo_credito_actual": "Plazo de Crédito Actual",
            "linea_credito_solicitado": "Línea de Crédito Solicitada",
            "plazo_credito_solicitado": "Plazo de Crédito Solicitado",
            "Productos_vendidos": "Productos Vendidos",
        }
        vend_map.update(vend_overrides)
        mappings[DocumentType.COMENTARIOS_VENDEDOR.value] = {
            "schema_id": vend_json.get("id"),
            "schema_name": vend_json.get("name"),
            "schema_version": str(vend_json.get("version", "1")),
            "field_display_map": vend_map,
            "title": "Comentarios del Vendedor",
        }

        # CSF (Constancia Fiscal)
        csf_json = _load_schema_file(_schema_path("CSF_document_schema.json"))
        csf_map = {item["name"]: item.get("label") or item["name"] for item in csf_json.get("headerFields", [])}
        csf_overrides = {
            "Razon_fiscal": "Denominación / Razón Social",
            "Domicilio": "Datos del Domicilio registrado",
            "Fecha_emision": "Fecha de emisión",
        }
        csf_map.update(csf_overrides)
        mappings[DocumentType.CONSTANCIA_FISCAL.value] = {
            "schema_id": csf_json.get("id"),
            "schema_name": csf_json.get("name"),
            "schema_version": str(csf_json.get("version", "1")),
            "field_display_map": csf_map,
            "title": "Constancia Situación Fiscal",
            "template_id": csf_json.get("templateId", "56f56727-21f2-452c-b3b0-6fbb747933cc"),  # Custom template ID
        }

        # CGV
        cgv_json = _load_schema_file(_schema_path("CGV_document_schema.json"))
        cgv_map = {item["name"]: item.get("label") or item["name"] for item in cgv_json.get("headerFields", [])}
        cgv_overrides = {
            "Razon_social": "Denominación / Razón Social",
            "Domicilio_completo": "Domicilio Completo",
            "Linea_credito": "Línea de Crédito Otorgado",
            "Plazo_credito": "Plazo de Crédito Otorgado (días)",
            "Fecha_firma": "Fecha de firma",
        }
        cgv_map.update(cgv_overrides)
        mappings[DocumentType.CGV.value] = {
            "schema_id": cgv_json.get("id"),
            "schema_name": cgv_json.get("name"),
            "schema_version": str(cgv_json.get("version", "1")),
            "field_display_map": cgv_map,
            "title": "Formato Condiciones Generales de Venta (CGV)",
        }

        # Investigacion Comercial
        ic_json = _load_schema_file(_schema_path("Investigacion_comercial_schema.json"))
        ic_map = {item["name"]: item.get("label") or item["name"] for item in ic_json.get("headerFields", [])}
        ic_overrides = {
            "Monto_Maximo_recomendado": "Monto Máximo Recomendado",
            "Identificador_Fiscal": "Identificador Fiscal",
            "Domicilio_Comercial": "Domicilio Comercial",
        }
        ic_map.update(ic_overrides)
        mappings[DocumentType.INVESTIGACION_COMERCIAL.value] = {
            "schema_id": ic_json.get("id"),
            "schema_name": ic_json.get("name"),
            "schema_version": str(ic_json.get("version", "1")),
            "field_display_map": ic_map,
            "title": "Investigación Comercial",
        }

        # Investigacion Legal
        il_json = _load_schema_file(_schema_path("Investigacion_legal_schema.json"))
        il_map = {item["name"]: item.get("label") or item["name"] for item in il_json.get("headerFields", [])}
        il_overrides = {
            "Historial_procesos_judiciales_empresa": "Historial de Procesos Judiciales de la Empresa",
            "Factor_riesgo_empresa": "Factor de Riesgo de la Empresa",
            "Historial_procesos_judiciales_persona": "Historial de Procesos Judiciales de la Persona",
            "Factor_riesgo_persona": "Factor de Riesgo de la Persona",
            "Confirmacion_Representacion_social_persona": "Confirmación de Representación Social de la Persona",
            "RFC": "RFC",
            "Domicilio_fiscal": "Domicilio Fiscal"
        }
        il_map.update(il_overrides)
        mappings[DocumentType.INVESTIGACION_LEGAL.value] = {
            "schema_id": il_json.get("id"),
            "schema_name": il_json.get("name"),
            "schema_version": str(il_json.get("version", "1")),
            "field_display_map": il_map,
            "title": "Investigación Legal",
        }
    except Exception as e:
        logger.error(f"Error loading DOX schema mappings: {e}")

    # Also prepare a placeholder for custom
    mappings[DocumentType.CUSTOM.value] = {
        "schema_id": None,
        "schema_name": None,
        "schema_version": None,
        "field_display_map": {},
        "title": "Documento Personalizado",
    }

    return mappings


_DOX_MAPPINGS: Optional[Dict[str, Dict[str, Any]]] = None


def _get_mappings() -> Dict[str, Dict[str, Any]]:
    """Load mappings once and cache them in memory."""
    global _DOX_MAPPINGS
    if _DOX_MAPPINGS is None:
        _DOX_MAPPINGS = _get_dox_schema_mappings()
    return _DOX_MAPPINGS


async def process_document_async(
    pdf_path: str,
    document_type: str,
    questions: List[str],
    fields: List[str],
    temperature: float = 0.1,
    language: str = "es",
    use_simple_extractor: bool = True
) -> ExtractionResponse:
    """
    Process a single PDF using SAP DOX with the appropriate schema mapping.

    For predefined document types, this will use the DOX schema by ID/version.
    For custom types, this will perform ad-hoc extraction using provided field
    names as DOX headerFields (best-effort, formattingType defaults to string).
    """
    start_time = datetime.now()

    try:
        mappings = _get_mappings()
        mapping = mappings.get(document_type)
        if not mapping:
            raise RuntimeError(f"Unsupported document type: {document_type}")

        client = get_dox_client()

        # Build upload options
        upload_kwargs: Dict[str, Any] = {
            "file_path": pdf_path,
            # DOX tenant client ID for schema/job operations (not OAuth client)
            "client_id": os.getenv("DOX_TENANT_CLIENT_ID") or os.getenv("DOX_TENANT_ID") or "default",
        }

        if document_type != DocumentType.CUSTOM.value:
            upload_kwargs["schema_id"] = mapping.get("schema_id")
            # schema_version is optional; include if present
            if mapping.get("schema_version"):
                upload_kwargs["schema_version"] = str(mapping.get("schema_version"))
        else:
            # Custom: define headerFields from provided fields
            # Default all to string; descriptions taken from field names
            header_fields: List[Dict[str, Any]] = []
            for f_name in (fields or []):
                if not isinstance(f_name, str) or not f_name.strip():
                    continue
                header_fields.append({
                    "name": f_name.replace(" ", "_").replace("/", "_").strip(),
                    "description": f_name,
                    "formattingType": "string",
                })
            if not header_fields:
                raise RuntimeError("For custom documents, 'fields' must be provided")
            upload_kwargs["header_fields"] = header_fields

        # Upload and poll result
        job = client.upload_document(**upload_kwargs)
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("DOX did not return a job id")

        result = client.wait_for_result(job_id=job_id, timeout_seconds=180, poll_interval_seconds=3)

        # Extract fields from DOX result
        header_fields = (((result or {}).get("extraction") or {}).get("headerFields") or [])
        dox_name_to_entry: Dict[str, Dict[str, Any]] = {item.get("name"): item for item in header_fields if isinstance(item, dict)}

        extraction_results: List[ExtractionResult] = []

        if document_type == DocumentType.CUSTOM.value:
            # Display using provided fields order
            for f_name in fields or []:
                dox_key = f_name.replace(" ", "_").replace("/", "_").strip()
                entry = dox_name_to_entry.get(dox_key)
                value = entry.get("value") if entry else None
                confidence = entry.get("confidence") if entry else None
                display = f_name
                extraction_results.append(
                    ExtractionResult(
                        question=display,
                        answer=str(value) if value is not None else "No encontrado",
                        field=display,
                        confidence=float(confidence) if isinstance(confidence, (float, int)) else None,
                    )
                )
        else:
            display_map: Dict[str, str] = mapping.get("field_display_map", {})
            # Preserve a stable order by using the mapping's keys
            for dox_field_name, display in display_map.items():
                entry = dox_name_to_entry.get(dox_field_name)
                value = entry.get("value") if entry else None
                confidence = entry.get("confidence") if entry else None
                extraction_results.append(
                    ExtractionResult(
                        question=str(display),
                        answer=str(value) if value is not None else "No encontrado",
                        field=str(display),
                        confidence=float(confidence) if isinstance(confidence, (float, int)) else None,
                    )
                )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ExtractionResponse(
            success=True,
            document_type=document_type,
            results=extraction_results,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error processing document via DOX: {e}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return ExtractionResponse(
            success=False,
            document_type=document_type,
            results=[],
            processing_time_ms=processing_time,
            error=str(e),
        )


@router.post("/single", response_model=ExtractionResponse)
async def extract_single_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(...),
    questions: Optional[str] = Form(None),  # JSON string
    fields: Optional[str] = Form(None),  # JSON string of field names
    temperature: float = Form(0.1),
    language: str = Form("es"),
    use_simple_extractor: bool = Form(True)
) -> ExtractionResponse:
    """
    Extract information from a single PDF document.
    
    Args:
        file: PDF file to process
        document_type: Type of document
        questions: Optional custom questions (JSON array string)
        fields: Optional custom fields (JSON array string)
        temperature: LLM temperature
        language: Response language
        use_simple_extractor: Use simplified image-only extractor (faster)
        
    Returns:
        ExtractionResponse with extraction results
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Resolve fields to request/display
        # For predefined DOX-backed types, we don't use questions; we only need a stable display field list.
        mappings = _get_mappings()
        mapping = mappings.get(document_type.value)
        if not mapping:
            raise HTTPException(status_code=400, detail=f"Unknown document type: {document_type}")

        if document_type == DocumentType.CUSTOM:
            # Custom: fields can be passed by the UI to instruct DOX ad-hoc extraction
            if fields:
                try:
                    fields_list = json.loads(fields)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid fields format for custom document")
            else:
                raise HTTPException(status_code=400, detail="Fields are required for custom document extraction")
            questions_list = [str(f) for f in fields_list]
        else:
            # Predefined: derive display field order from DOX mapping
            display_map = mapping.get("field_display_map", {})
            fields_list = [display for _, display in display_map.items()]
            questions_list = [str(display) for display in fields_list]
        
        # Process document
        response = await process_document_async(
            pdf_path=tmp_path,
            document_type=document_type.value,
            questions=questions_list,
            fields=fields_list,
            temperature=temperature,
            language=language,
            use_simple_extractor=True
        )
        
        return response
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


@router.post("/batch", response_model=BatchExtractionResponse)
async def extract_batch_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_types: str = Form(...),  # JSON array of document types
    max_concurrent: int = Form(10),
    use_simple_extractor: bool = Form(True)
) -> BatchExtractionResponse:
    """
    Process multiple PDF documents in batch.
    
    Args:
        files: List of PDF files
        document_types: JSON array of document types (one per file)
        max_concurrent: Maximum concurrent processing
        use_simple_extractor: Use simplified image-only extractor (faster)
        
    Returns:
        BatchExtractionResponse with task ID for tracking
    """
    # Parse document types
    try:
        types_list = json.loads(document_types)
        if len(types_list) != len(files):
            raise HTTPException(
                status_code=400, 
                detail="Number of document types must match number of files"
            )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid document types format")
    
    # Read all file contents before passing to background task
    # This prevents "I/O operation on closed file" error
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append({
            "filename": file.filename,
            "content": content
        })
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task
    TASKS[task_id] = {
        "status": "processing",
        "total_documents": len(files),
        "processed": 0,
        "results": [],
        "created_at": datetime.now().timestamp()
    }
    
    # Add background task with file data instead of UploadFile objects
    background_tasks.add_task(
        process_batch_background,
        task_id=task_id,
        files_data=files_data,
        document_types=types_list,
        max_concurrent=max_concurrent,
        use_simple_extractor=use_simple_extractor
    )
    
    return BatchExtractionResponse(
        task_id=task_id,
        total_documents=len(files),
        status="processing",
        created_at=TASKS[task_id]["created_at"]
    )


async def process_batch_background(
    task_id: str,
    files_data: List[Dict[str, Any]],
    document_types: List[str],
    max_concurrent: int,
    use_simple_extractor: bool = True
):
    """
    Background task for processing batch documents.
    
    Args:
        task_id: Task identifier
        files_data: List of dicts with 'filename' and 'content' keys
        document_types: List of document types
        max_concurrent: Maximum concurrent processing
    """
    temp_files = []
    
    try:
        logger.info(f"Starting batch processing for task {task_id} with {len(files_data)} files")
        
        # Save all files temporarily
        for idx, file_data in enumerate(files_data):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(file_data["content"])
                    temp_files.append(tmp_file.name)
                    logger.info(f"Saved file {idx+1}/{len(files_data)}: {file_data['filename']} to {tmp_file.name}")
            except Exception as e:
                logger.error(f"Error saving file {file_data['filename']}: {e}")
                raise Exception(f"Failed to save file {file_data['filename']}: {str(e)}")
        
        # Process documents with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(pdf_path, doc_type, filename):
            async with semaphore:
                try:
                    mappings = _get_mappings()
                    mapping = mappings.get(doc_type)
                    if not mapping:
                        logger.warning(f"Schema mapping not found for document type: {doc_type}")
                        return None

                    logger.info(f"Processing {filename} via DOX as {doc_type}")
                    display_fields = list(mapping.get("field_display_map", {}).values())
                    result = await process_document_async(
                        pdf_path=pdf_path,
                        document_type=doc_type,
                        questions=display_fields,  # Unused; kept for response consistency
                        fields=display_fields,
                        use_simple_extractor=True,
                    )
                    result.metadata = {"filename": filename}
                    logger.info(f"Successfully processed {filename}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    # Return a failed result instead of None
                    return ExtractionResponse(
                        success=False,
                        document_type=doc_type,
                        results=[],
                        processing_time_ms=0,
                        error=str(e),
                        metadata={"filename": filename}
                    )
        
        # Create tasks
        tasks = [
            process_with_limit(tmp_path, doc_type, file_data["filename"])
            for tmp_path, doc_type, file_data in zip(temp_files, document_types, files_data)
        ]
        
        # Process all
        results = await asyncio.gather(*tasks)
        
        # Update task status
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["results"] = [r for r in results if r]
        TASKS[task_id]["completed_at"] = datetime.now().timestamp()
        
    except Exception as e:
        import traceback
        error_details = f"Batch processing error: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_details)
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        
    finally:
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                os.unlink(tmp_file)
            except:
                pass


@router.get("/schemas", response_model=DocumentSchemasResponse)
async def get_document_schemas() -> DocumentSchemasResponse:
    """
    Get all document schemas with fields based on DOX mappings.
    
    Returns:
        DocumentSchemasResponse with all schemas
    """
    mappings = _get_mappings()
    schemas: Dict[str, DocumentSchema] = {}

    for doc_type, mapping in mappings.items():
        # Skip custom here; it's dynamic
        title = mapping.get("title", doc_type)
        field_display_map = mapping.get("field_display_map", {})
        fields = list(field_display_map.values()) if field_display_map else []
        # Questions are not used in DOX flow; mirror fields for UI compatibility
        questions = [str(f) for f in fields]
        schemas[doc_type] = DocumentSchema(
            document_type=DocumentType(doc_type),
            title=title,
            questions=questions,
            fields=fields,
        )

    return DocumentSchemasResponse(schemas=schemas)


@router.post("/schemas")
async def update_document_schema(request: UpdateSchemaRequest) -> Dict[str, str]:
    """
    Update questions and fields for a document type.
    
    Args:
        request: Update schema request
        
    Returns:
        Success message
    """
    doc_type = request.document_type.value
    
    if doc_type not in DOCUMENT_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Document type not found: {doc_type}")
    
    # Update schema
    DOCUMENT_SCHEMAS[doc_type]["questions"] = request.questions
    DOCUMENT_SCHEMAS[doc_type]["fields"] = request.fields
    
    return {"message": f"Schema updated for {doc_type}"}


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Check the status of a batch processing task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        TaskStatusResponse with current status
    """
    task = TASKS.get(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    # Calculate progress
    progress = 0
    if task["status"] == "completed":
        progress = 100
    elif task["status"] == "processing" and task["total_documents"] > 0:
        progress = int((task.get("processed", 0) / task["total_documents"]) * 100)
    
    # Return error message if task failed, otherwise show processing status
    message = task.get("error") if task["status"] == "failed" else f"Processing {task['total_documents']} documents"
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=progress,
        message=message,
        result=task.get("results") if task["status"] == "completed" else None
    )