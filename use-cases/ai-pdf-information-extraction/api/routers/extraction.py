"""
Document extraction API router.

Provides endpoints for processing PDF documents and extracting
structured information using AI.
"""

import logging
import tempfile
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

# Import models
from models import (
    DocumentType,
    ExtractionResponse,
    ExtractionResult,
)

# Import data extraction module
from extraction.simple_pdf_extractor import SimpleImageExtractor


# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

"""
In-memory storage for predefined schemas.
Note: UI now defines questions client-side; these defaults remain for built-in types only.
"""
DOCUMENT_SCHEMAS = {
    "conoce_cliente": {
        "title": "Conoce a tu Cliente",
        # Questions and fields updated to match requested attributes for extraction.
        "questions": [
            "¿Cuál es la razón social del cliente?",
            "¿Cuál es el RFC del cliente?",
            "¿Cuál es la nacionalidad del cliente?",
            "¿Cuál es el domicilio fiscal del cliente en la sección de Información Fiscal?",
            "¿Cuál es la línea de crédito actual?",
            "¿Cuál es el plazo de crédito actual?",
            "¿Cuál es la línea de crédito solicitada?",
            "¿Cuál es el plazo de crédito solicitado?",
            "¿En qué moneda está expresada la operación? (en caso de que sea pesos, diga MXN)",
            "¿Cuáles son los productos que se le están vendiendo al cliente?",
            "¿En qué fecha fue rellenado el formulario?"
        ],
        "fields": [
            "Razón Social",
            "RFC",
            "Nacionalidad",
            "Domicilio Fiscal (Información Fiscal)",
            "Línea de Crédito Actual",
            "Plazo de Crédito Actual",
            "Línea de Crédito Solicitada",
            "Plazo de Crédito Solicitado",
            "Moneda",
            "Productos Vendidos",
            "Fecha"
        ]
    },
    "comentarios_vendedor": {
        "title": "Comentarios del Vendedor",
        "questions": [
            "¿Cuál es el ID de cliente SAP?",
            "¿Cuál es el nombre del cliente?",
            "¿Cuál es la línea de crédito actual?",
            "¿Cuál es el plazo de crédito actual?",
            "¿Cuál es la línea de crédito solicitada?",
            "¿Cuál es el plazo de crédito solicitado?",
            "¿En qué moneda está expresada la operación? (en caso de que sea pesos, diga MXN)",
            "¿Qué producto o productos se le está vendiendo al cliente?",
            "¿Quién es el responsable de cuentas por cobrar?"
            "¿Quién es el responsable de customer service?"
        ],
        "fields": [
            "ID Cliente SAP",
            "Nombre del Cliente",
            "Línea de Crédito Actual",
            "Plazo de Crédito Actual",
            "Línea de Crédito Solicitada",
            "Plazo de Crédito Solicitado",
            "Moneda",
            "Productos Vendidos",
            "Responsable de cuentas por cobrar",
            "Responsable de customer service"
        ]
    },
    "ine": {
        "title": "INE",
        "questions": [
            "¿Cuál es el nombre completo de la persona?",
            "¿Cuál es el domicilio completo?",
            "¿Cuál es el CURP?",
            "¿Cuál es la fecha de nacimiento?",
            "¿Cuál es la sección electoral?",
            "¿Cuál es la vigencia del documento?",
            "¿Cuál es el año de registro?",
            "¿Cuál es el sexo de la persona?",
            "¿Cuál es el código completo del reverso del documento?"
        ],
        "fields": [
            "Nombre Completo",
            "Domicilio",
            "CURP",
            "Fecha de Nacimiento",
            "Sección",
            "Vigencia",
            "Año de Registro",
            "Sexo",
            "Clave de elector (Reverso)"
        ]
    },
    "constancia_fiscal": {
        "title": "Constancia Situación Fiscal",
        "questions": [
            "¿Cuál es el RFC?",
            "¿Cuál es el domicilio fiscal completo?",
            "¿Cuál es la denominación o razón social del cliente?",
            "¿Cuál es la fecha de emisión de la constancia?"
        ],
        "fields": [
            "RFC",
            "Datos del Domicilio registrado",
            "Denominación / Razón Social",
            "Fecha de emisión"
        ]
    },
    "cgv": {
        "title": "Formato Condiciones Generales de Venta (CGV)",
        "questions": [
            "¿Cuál es el nombre o razón social del cliente?",
            "¿Cuál es el RFC del cliente?",
            "¿Cuál es el nombre del representante legal o del solicitante?",
            "¿Cuál es el domicilio completo del cliente?",
            "¿Cuál es la línea de crédito otorgada?",
            "¿Cuál es el plazo de crédito otorgado (en días)?",
            "¿Cuál es la fecha de firma del documento?",
            "¿El documento está firmado por el solicitante o representante legal? Responda Sí o No."
        ],
        "fields": [
            "Denominación / Razón Social",
            "RFC",
            "Nombre del Representante",
            "Domicilio Completo",
            "Línea de Crédito Otorgado",
            "Plazo de Crédito Otorgado (días)",
            "Fecha de firma",
            "Firmado por Solicitante o Representante Legal"
        ]
    },
    "custom": {
        "title": "Documento Personalizado",
        "questions": [],
        "fields": []
    }
}



def get_simple_extractor():
    """Initialize the simple image-only extractor with standard quality settings."""
    return SimpleImageExtractor(
        max_dimension=2048,
        dpi=200,
        provider="openai",
        model_name="gpt-4.1",
        temperature=0.2,
        max_tokens=2048
    )


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
    Process a single document asynchronously.
    
    Args:
        pdf_path: Path to the PDF file
        document_type: Type of document
        questions: List of questions to ask
        fields: List of field names
        temperature: LLM temperature
        language: Response language
        use_simple_extractor: Use the simplified image-only extractor
        
    Returns:
        ExtractionResponse with results
    """
    start_time = datetime.now()
    
    try:
        # Get appropriate extractor
        logger.info("Using simple image-only extractor")
        extractor = get_simple_extractor()

        
        extraction_results = []
        
        if use_simple_extractor:
            # Use simple extractor's batch method
            batch_result = extractor.extract_batch(
                pdf_path=pdf_path,
                questions=questions,
                language=language,
                max_pages=None  # Process all pages
            )
            
            if batch_result["success"]:
                # Process each question result
                for question, field in zip(questions, fields):
                    if question in batch_result["results"]:
                        result_data = batch_result["results"][question]
                        extraction_results.append(
                            ExtractionResult(
                                question=question,
                                answer=result_data["answer"],
                                field=field
                            )
                        )
                    else:
                        extraction_results.append(
                            ExtractionResult(
                                question=question,
                                answer="No encontrado",
                                field=field
                            )
                        )
            else:
                # Handle batch failure
                for question, field in zip(questions, fields):
                    extraction_results.append(
                        ExtractionResult(
                            question=question,
                            answer=f"Error: {batch_result.get('error', 'Unknown error')}",
                            field=field
                        )
                    )
        else:
            # This path is no longer supported - use simple extractor instead
            raise ValueError("LangGraph extractor is not implemented. Use simple extractor instead.")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExtractionResponse(
            success=True,
            document_type=document_type,
            results=extraction_results,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExtractionResponse(
            success=False,
            document_type=document_type,
            results=[],
            processing_time_ms=processing_time,
            error=str(e)
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
        # Get questions and fields
        if questions:
            # Parse custom questions
            try:
                questions_list = json.loads(questions)
                # Parse custom fields if provided, otherwise generate default names
                if fields:
                    try:
                        fields_list = json.loads(fields)
                        # Ensure fields and questions have same length
                        if len(fields_list) != len(questions_list):
                            fields_list = [f"Campo {i+1}" for i in range(len(questions_list))]
                    except json.JSONDecodeError:
                        fields_list = [f"Campo {i+1}" for i in range(len(questions_list))]
                else:
                    fields_list = [f"Campo {i+1}" for i in range(len(questions_list))]
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid questions format")
        else:
            # Use default schema
            schema = DOCUMENT_SCHEMAS.get(document_type.value)
            if not schema:
                raise HTTPException(status_code=400, detail=f"Unknown document type: {document_type}")
            questions_list = schema["questions"]
            fields_list = schema["fields"]
        
        # Process document
        response = await process_document_async(
            pdf_path=tmp_path,
            document_type=document_type.value,
            questions=questions_list,
            fields=fields_list,
            temperature=temperature,
            language=language,
            use_simple_extractor=use_simple_extractor
        )
        
        return response
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


