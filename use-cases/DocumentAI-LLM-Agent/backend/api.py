"""
api.py
------
DocAI - FastAPI REST API
Expone todos los módulos del pipeline como endpoints HTTP consumibles
desde un frontend SAP Fiori (u otro cliente).

Iniciar servidor:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Documentación interactiva:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Módulos del proyecto (sin modificar ninguno)
# ---------------------------------------------------------------------------
from modules.auth.get_token import AuthenticationError, get_token
from modules.evaluation.evaluator import EvaluationError, run_evaluation
from modules.genai.compare_results import compare
from modules.genai.multimodal_prompting import extract_multimodal_prompting
from modules.genai.multimodal_structured import extract_multimodal_structured
from modules.invoice.process_invoice import (
    InvoiceProcessor,
    InvoiceProcessingError,
    JobFailedError,
    PollingTimeoutError,
)
from modules.routing.routing_engine import route_invoice
from modules.routing.template_processor import TemplateInvoiceProcessor, TemplateProcessingError
from modules.schemas.get_schema import DocumentAIError as SchemaError
from modules.schemas.get_schema import get_schemas
from modules.templates.get_templates import DocumentAIError as TemplateError
from modules.templates.get_templates import get_templates
from InvoiceProcess.po_invoice.po_detector import detect_po_number
from SalesOrderProcess.document_classifier import classify_document
from SalesOrderProcess.so_extractor import SalesOrderExtractor
from SalesOrderProcess.so_validator import validate_purchase_order
from PaymentAdviceProcess.pa_extractor import PaymentAdviceExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directorios de salida (igual que en los módulos originales)
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path("output")
OUTPUT_GENAI_DIR: Path = OUTPUT_DIR / "genai"

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DocAI API",
    description=(
        "API REST para el pipeline SAP Document AI + Gen AI Hub.\n\n"
        "Permite procesar facturas, ejecutar el pipeline GenAI multimodal, "
        "evaluar resultados y consultar schemas/templates desde SAP BTP."
    ),
    version="1.0.0",
    contact={"name": "DocAI Team"},
)

# CORS – permite consumo desde frontend Fiori / SAP BTP / cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _save_uploaded_file(upload: UploadFile) -> Path:
    """
    Guarda el archivo subido en un directorio temporal y devuelve su Path.
    El llamador es responsable de eliminar el archivo cuando termine.
    """
    suffix = Path(upload.filename).suffix.lower() if upload.filename else ".pdf"
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato de archivo no soportado: '{suffix}'. "
                f"Formatos válidos: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return Path(tmp.name)


def _load_genai_output(filename: str) -> dict:
    """Carga un archivo JSON del directorio output/genai/."""
    path = OUTPUT_GENAI_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Archivo no encontrado: {path}. "
                "Ejecute primero el pipeline GenAI (POST /api/v1/genai/pipeline)."
            ),
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ── Sistema ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Sistema"],
    summary="Health check",
)
def health_check() -> dict:
    """Verifica que la API está activa y responde correctamente."""
    return {"status": "ok", "service": "DocAI API", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# ── Auth ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/auth/token",
    tags=["Auth"],
    summary="Obtener token SAP BTP",
)
def get_auth_token() -> dict:
    """
    Obtiene un Bearer token válido desde SAP BTP (OAuth2 client_credentials).
    El token se cachea automáticamente hasta 60 segundos antes de su expiración.
    """
    try:
        token = get_token()
        return {"access_token": token, "token_type": "Bearer"}
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ── Schemas ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/schemas",
    tags=["Schemas"],
    summary="Listar schemas de SAP Document AI",
)
def list_schemas(
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
) -> dict:
    """
    Recupera todos los schemas disponibles en SAP Document AI
    para el clientId indicado (con paginación automática).
    """
    try:
        return get_schemas(client_id=client_id)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except SchemaError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ── Templates ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/templates",
    tags=["Templates"],
    summary="Listar templates de SAP Document AI",
)
def list_templates(
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
) -> dict:
    """
    Recupera todos los templates disponibles en SAP Document AI
    para el clientId indicado (con paginación automática).
    """
    try:
        return get_templates(client_id=client_id)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except TemplateError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ── Invoice ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/invoice/process",
    tags=["Invoice"],
    summary="Procesar factura con SAP Document AI",
)
async def process_invoice_endpoint(
    file: UploadFile = File(..., description="Archivo de factura (PDF, JPG, PNG, TIFF)"),
    schema_name: str = Query("SAP_invoice_schema", description="Nombre del schema"),
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
    document_type: str = Query("invoice", description="Tipo de documento"),
) -> dict[str, Any]:
    """
    Sube un documento y lo procesa con SAP Document AI.

    Flujo:
    1. Recibe el archivo vía multipart/form-data
    2. Envía el documento al endpoint de jobs de SAP Document AI
    3. Hace polling hasta que el job esté en estado DONE
    4. Guarda el resultado en `output/{JOB_ID}.json`
    5. Devuelve el JSON con los campos extraídos
    """
    tmp_path = _save_uploaded_file(file)

    try:
        processor = InvoiceProcessor()
        job_id = processor.submit_document(
            tmp_path, schema_name, client_id, document_type
        )
        result = processor.poll_until_done(job_id)
        output_path = processor.save_result(job_id, result)

        return {
            "job_id": job_id,
            "output_file": str(output_path),
            "result": result,
        }

    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except (JobFailedError, PollingTimeoutError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except InvoiceProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ── GenAI ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/genai/pipeline",
    tags=["GenAI"],
    summary="Pipeline GenAI multimodal con routing inteligente",
)
async def genai_pipeline_endpoint(
    file: UploadFile = File(..., description="Archivo de factura (PDF, JPG, PNG, TIFF)"),
    schema_name: str = Query("SAP_invoice_schema", description="Nombre del schema SAP"),
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
    document_type: str = Query("invoice", description="Tipo de documento"),
) -> dict[str, Any]:
    """
    Ejecuta el pipeline con routing inteligente:

    **STEP 1** — SAP Document AI (genérico): extracción inicial del proveedor.

    **STEP 2** — Routing Engine:
    - Detecta el nombre del proveedor
    - Busca un template coincidente en SAP Document AI
    - Decide la ruta de procesamiento

    **STEP 3a — Template encontrado:**
    - Reprocesa la factura con el template especializado de SAP
    - Omite completamente el flujo GenAI/LLM

    **STEP 3b — Sin template:**
    - LLM Técnica 1 (free prompting)
    - LLM Técnica 2 (JSON estructurado)
    - Comparación de resultados

    Guarda todos los resultados en `output/genai/` y `output/routing/`.
    """
    tmp_path = _save_uploaded_file(file)

    try:
        OUTPUT_GENAI_DIR.mkdir(parents=True, exist_ok=True)

        # ── STEP 0: Classify document type BEFORE sending to DocAI ─────
        # Uses LLM multimodal vision to decide: "invoice" vs "purchase_order"
        detected_doc_type = classify_document(tmp_path)
        logger.info("Document classified as: %s", detected_doc_type)

        # ── STEP 1: SAP Document AI — choose schema based on doc type ───

        # Payment Advice → SAP_paymentAdvice_schema
        if detected_doc_type == "payment_advice":
            logger.info("Routing to Payment Advice pipeline (SAP_paymentAdvice_schema)")
            extracted_pa = None
            try:
                pa_extractor = PaymentAdviceExtractor()
                extracted_pa = pa_extractor.extract(tmp_path, client_id=client_id)
            except Exception as exc_pa:
                logger.warning("PA extraction failed: %s", exc_pa)

            return {
                "route":            "payment_advice",
                "document_type":    "payment_advice",
                "job_id":           "pa_pipeline",
                "output_dir":       str(OUTPUT_GENAI_DIR),
                "sap_result":       {"extraction": {"headerFields": [{"name": k, "value": v} for k, v in (extracted_pa.raw_sap_fields or {}).items()]}} if extracted_pa else {},
                "po_number":        None,
                "extracted_pa":     extracted_pa.model_dump() if extracted_pa else None,
                "routing_decision":     None,
                "template_result":      None,
                "template_output_file": None,
                "llm_prompting":        None,
                "llm_structured":       None,
                "comparison":           None,
            }

        if detected_doc_type == "purchase_order":
            # Customer PO → SAP_purchaseOrder_schema + LLM enrichment + S4 validation
            logger.info("Routing to Sales Order pipeline (SAP_purchaseOrder_schema)")
            extracted_po = None
            validation   = None
            llm_po       = None
            try:
                # DocAI + LLM — both run, best value per field is used
                so_extractor = SalesOrderExtractor()
                extracted_po, llm_po = so_extractor.extract_with_llm(tmp_path, client_id=client_id)

                validation = validate_purchase_order(extracted_po)
            except Exception as exc_so:
                logger.warning("SO extraction/validation failed: %s", exc_so)

            return {
                "route":            "purchase_order",
                "document_type":    "purchase_order",
                "job_id":           "so_pipeline",
                "output_dir":       str(OUTPUT_GENAI_DIR),
                "sap_result":       {"extraction": {"headerFields": [{"name": k, "value": v} for k, v in (extracted_po.raw_sap_fields or {}).items()]}} if extracted_po else {},
                "po_number":        None,
                "extracted_po":     extracted_po.model_dump() if extracted_po else None,
                "so_validation":    validation.model_dump() if validation else None,
                "routing_decision":     None,
                "template_result":      None,
                "template_output_file": None,
                "llm_prompting":        None,
                "llm_structured":       llm_po,
                "comparison":           None,
            }

        # ── Invoice path: SAP Document AI with invoice schema ───────────
        processor = InvoiceProcessor()
        job_id = processor.submit_document(
            tmp_path, schema_name, client_id, document_type
        )
        sap_result = processor.poll_until_done(job_id)
        processor.save_result(job_id, sap_result)

        sap_out = OUTPUT_GENAI_DIR / "sap_result.json"
        with open(sap_out, "w", encoding="utf-8") as f:
            json.dump(sap_result, f, indent=2, ensure_ascii=False)

        # ── STEP 2 (Invoice): Routing Engine ───────────────────────────
        routing_decision = route_invoice(sap_result, client_id=client_id)
        route = routing_decision.get("route", "genai")

        # ── STEP 3a: Template flow ──────────────────────────────────────
        if route == "template":
            template_match = routing_decision["template_match"]
            template_id = template_match["template_id"]

            logger.info(
                "Reprocessing invoice using schema + template. schema=%s, template=%s",
                schema_name,
                template_id,
            )

            template_processor = TemplateInvoiceProcessor()
            template_result, template_output_path = template_processor.process(
                tmp_path,
                template_id=template_id,
                schema_name=schema_name,
                client_id=client_id,
                document_type=document_type,
            )

            return {
                "route": "template",
                "job_id": job_id,
                "output_dir": str(OUTPUT_GENAI_DIR),
                "routing_decision": {
                    "route": routing_decision["route"],
                    "decision_reason": routing_decision["decision_reason"],
                    "supplier_name": routing_decision["supplier_detection"].get("supplier_name"),
                    "template_name": template_match.get("template_name"),
                    "template_id": template_id,
                    "confidence_pct": template_match.get("confidence_pct"),
                },
                "sap_result": sap_result,
                "template_result": template_result,
                "template_output_file": str(template_output_path),
                "llm_prompting": None,
                "llm_structured": None,
                "comparison": None,
                "po_number": detect_po_number(sap_result=template_result),
            }

        # ── STEP 3b: GenAI fallback flow ────────────────────────────────
        llm_p1 = extract_multimodal_prompting(tmp_path)
        llm_p1_out = OUTPUT_GENAI_DIR / "llm_multimodal_prompting.json"
        with open(llm_p1_out, "w", encoding="utf-8") as f:
            json.dump(llm_p1, f, indent=2, ensure_ascii=False)

        llm_p2 = extract_multimodal_structured(tmp_path)
        llm_p2_out = OUTPUT_GENAI_DIR / "llm_multimodal_structured.json"
        with open(llm_p2_out, "w", encoding="utf-8") as f:
            json.dump(llm_p2, f, indent=2, ensure_ascii=False)

        comparison = compare(sap_result, llm_p1, llm_p2)
        comp_out = OUTPUT_GENAI_DIR / "comparison.json"
        with open(comp_out, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        return {
            "route": "genai",
            "job_id": job_id,
            "output_dir": str(OUTPUT_GENAI_DIR),
            "routing_decision": {
                "route": routing_decision["route"],
                "decision_reason": routing_decision["decision_reason"],
                "supplier_name": routing_decision["supplier_detection"].get("supplier_name"),
                "template_name": None,
                "template_id": None,
                "confidence_pct": None,
            },
            "sap_result": sap_result,
            "template_result": None,
            "template_output_file": None,
            "llm_prompting": llm_p1,
            "llm_structured": llm_p2,
            "comparison": comparison,
            "po_number": detect_po_number(
                sap_result=sap_result,
                llm_structured=llm_p2,
                llm_prompting=llm_p1,
            ),
        }

    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except (JobFailedError, PollingTimeoutError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except (InvoiceProcessingError, TemplateProcessingError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ── Evaluation ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/evaluation/run",
    tags=["Evaluation"],
    summary="Evaluar resultados del pipeline GenAI",
)
def evaluation_run_endpoint() -> dict[str, Any]:
    """
    Ejecuta el pipeline de evaluación sobre los resultados guardados en
    `output/genai/` (generados por el endpoint `/api/v1/genai/pipeline`).

    Pasos:
    1. Carga `sap_result.json`, `llm_multimodal_prompting.json` y
       `llm_multimodal_structured.json`
    2. Analiza campos (completitud, missing, conflictos)
    3. Calcula scores por método
    4. Evaluación inteligente con LLM
    5. Genera reportes y los guarda en `output/evaluation/`

    **Requisito:** ejecutar primero `POST /api/v1/genai/pipeline`.
    """
    try:
        result = run_evaluation()
        # Convertir Path objects a strings para que sean serializables
        if "output_paths" in result:
            result["output_paths"] = {
                k: str(v) for k, v in result["output_paths"].items()
            }
        return result
    except EvaluationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ── Outputs ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/output/genai",
    tags=["Outputs"],
    summary="Listar archivos de salida GenAI",
)
def list_genai_outputs() -> dict:
    """
    Lista los archivos JSON disponibles en `output/genai/`.
    Estos son los resultados del último pipeline GenAI ejecutado.
    """
    if not OUTPUT_GENAI_DIR.exists():
        return {"files": []}

    files = sorted(
        p.name for p in OUTPUT_GENAI_DIR.iterdir() if p.suffix == ".json"
    )
    return {"directory": str(OUTPUT_GENAI_DIR), "files": files}


@app.get(
    "/api/v1/output/genai/{filename}",
    tags=["Outputs"],
    summary="Obtener un archivo de salida GenAI",
)
def get_genai_output(filename: str) -> dict:
    """
    Devuelve el contenido de un archivo JSON específico de `output/genai/`.

    Archivos disponibles típicamente:
    - `sap_result.json`
    - `llm_multimodal_prompting.json`
    - `llm_multimodal_structured.json`
    - `comparison.json`
    """
    if not filename.endswith(".json"):
        filename = filename + ".json"
    return _load_genai_output(filename)


@app.get(
    "/api/v1/output/evaluation",
    tags=["Outputs"],
    summary="Listar archivos de salida de evaluación",
)
def list_evaluation_outputs() -> dict:
    """
    Lista los archivos disponibles en `output/evaluation/`.
    Estos son los reportes del último proceso de evaluación ejecutado.
    """
    eval_dir = OUTPUT_DIR / "evaluation"
    if not eval_dir.exists():
        return {"files": []}

    files = sorted(p.name for p in eval_dir.iterdir() if p.is_file())
    return {"directory": str(eval_dir), "files": files}


# ---------------------------------------------------------------------------
# ── DOC AI NEW ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

from modules.docai_new.pipeline import DocAiNewPipeline
from modules.docai_new.template_training_service import TemplateTrainingService
from modules.docai_new.template_discovery_service import TemplateDiscoveryService


@app.post(
    "/api/v2/docai-new/process",
    tags=["DocAI NEW"],
    summary="DOC AI NEW — Process invoice(s) with Free Prompt pipeline",
)
async def docai_new_process_endpoint(
    files: list[UploadFile] = File(..., description="One or multiple PDF files"),
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
    auto_create_template: bool = Query(True, description="Auto-create template if not found"),
) -> dict[str, Any]:
    """
    DOC AI NEW pipeline:

    1. Validate PDF files (MIME type = application/pdf)
    2. Detect PDF type (searchable vs scanned)
    3. Free Prompt extraction via LLM
    4. Customer detection
    5. Template lookup by customer name
    6. If template exists → use it; if not → auto-create + annotate
    """
    # Validate all files are PDFs
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: '{suffix}' for '{upload.filename}'",
            )

    tmp_paths: list[Path] = []
    try:
        # Save all uploaded files
        for upload in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
                shutil.copyfileobj(upload.file, tmp)
            finally:
                tmp.close()
            tmp_paths.append(Path(tmp.name))

        pipeline = DocAiNewPipeline()

        if len(tmp_paths) == 1:
            result = pipeline.process(
                pdf_path=tmp_paths[0],
                client_id=client_id,
                auto_create_template=auto_create_template,
            )
            # Restore original filename
            result["filename"] = files[0].filename or tmp_paths[0].name
            return {"results": [result], "total": 1}
        else:
            results = pipeline.process_multiple(
                pdf_paths=tmp_paths,
                client_id=client_id,
                auto_create_template=auto_create_template,
            )
            for i, r in enumerate(results):
                r["filename"] = files[i].filename or tmp_paths[i].name
            return {"results": results, "total": len(results)}

    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.post(
    "/api/v2/docai-new/train",
    tags=["DocAI NEW"],
    summary="DOC AI NEW — Train Template with PDF(s)",
)
async def docai_new_train_endpoint(
    files: list[UploadFile] = File(..., description="One or multiple PDF files for training"),
    template_id: str = Query(..., description="SAP Document AI template ID to train"),
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
) -> dict[str, Any]:
    """
    Train an existing SAP Document AI template with one or more PDFs.

    For each PDF:
    1. Run Free Prompt extraction
    2. Generate annotations
    3. Attach document to template
    4. Trigger training API
    """
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: '{suffix}' for '{upload.filename}'",
            )

    tmp_paths: list[Path] = []
    try:
        for upload in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
                shutil.copyfileobj(upload.file, tmp)
            finally:
                tmp.close()
            tmp_paths.append(Path(tmp.name))

        trainer = TemplateTrainingService()
        result = trainer.train_template(
            template_id=template_id,
            pdf_paths=tmp_paths,
            client_id=client_id,
        )
        return result

    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)


@app.get(
    "/api/v2/docai-new/templates",
    tags=["DocAI NEW"],
    summary="DOC AI NEW — List all templates",
)
def docai_new_list_templates_endpoint(
    client_id: str = Query("default", description="Client ID de SAP Document AI"),
) -> dict[str, Any]:
    """Return all available SAP Document AI templates for the Train Template screen."""
    try:
        discovery = TemplateDiscoveryService()
        templates = discovery.list_all_templates(client_id=client_id)
        return {"templates": templates, "total": len(templates)}
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ── Chat ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

import asyncio

from modules.chat.chat_service import stream_chat


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    context: Optional[dict] = None


@app.post(
    "/api/v1/chat/message",
    tags=["Chat"],
    summary="Streaming chat with DocAI Assistant",
)
async def chat_message_endpoint(body: ChatRequest):
    """
    Send a message to the DocAI Assistant and receive a streaming response.

    Returns a stream of NDJSON lines:
    - `{"type": "delta", "content": "..."}` — one per LLM token chunk
    - `{"type": "done"}` — signals the end of the response
    - `{"type": "error", "message": "..."}` — on failure

    Optionally pass `context` with the current extraction result so the
    assistant can answer specific questions about it.
    """
    history = [{"role": t.role, "content": t.content} for t in body.history]

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _sentinel = object()

        def producer():
            try:
                for line in stream_chat(body.message, history, body.context):
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _sentinel)

        loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is _sentinel:
                break
            yield item

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# ── S4 / FI Routers ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

from S4.sap_session_routes import router as sap_session_router
from S4.business_partners.business_partners_routes import router as bp_router
from S4.search_routes import router as search_router
from S4.debug_routes import router as debug_router
from InvoiceProcess.supplier_invoice.supplier_invoice_routes import router as fi_router
from InvoiceProcess.po_invoice.po_invoice_routes import router as po_fi_router
from InvoiceProcess.purchase_orders.purchase_orders_routes import router as po_search_router
from SalesOrderProcess.so_routes import router as so_router
from PaymentAdviceProcess.pa_routes import router as pa_router

app.include_router(sap_session_router)
app.include_router(bp_router)
app.include_router(search_router)
app.include_router(debug_router)
app.include_router(fi_router)
app.include_router(po_fi_router)
app.include_router(po_search_router)
app.include_router(so_router)
app.include_router(pa_router)


# ---------------------------------------------------------------------------
# Punto de entrada para ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
