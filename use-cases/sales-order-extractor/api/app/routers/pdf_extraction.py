"""PDF extraction router for document processing with LLM integration."""

import logging
import base64
import json
import tempfile
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File

# Gen AI Hub Orchestration imports
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService

# Anthropic imports
try:
    from gen_ai_hub.proxy.native.bedrock import Session
except ImportError:
    Session = None

# Gemini imports
try:
    from vertexai.generative_models import GenerativeModel
except ImportError:
    GenerativeModel = None

from dotenv import load_dotenv

load_dotenv() 

# PDF processing
import fitz  # PyMuPDF

from ..models.pdf_extraction import PDFUploadRequest, PDFExtractionResponse, ExtractedData, HeaderData, LineItem
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# Extraction prompt template
EXTRACTION_PROMPT = """
You are a precise document processing assistant specialized in extracting data from purchase orders and invoices. 

CRITICAL RULES - FOLLOW EXACTLY:
1. Extract ONLY text that is clearly visible and readable in the document
2. Do NOT generate, invent, or hallucinate any data
3. Do NOT use placeholder names like "Widget A", "Product X", "Company ABC", etc.
4. If text is unclear, blurry, or unreadable, use null instead of guessing
5. Look for actual company names, product codes, and real numerical values
6. Pay special attention to tables with line items containing products and prices

WHAT TO EXTRACT:

HEADER INFORMATION:
- Client: Look for the customer/client company name (often at the top or in "Ship To"/"Bill To" sections)
- Date: Look for order date, invoice date, or document date (format: MM/DD/YYYY or similar)

LINE ITEMS (from tables or itemized lists):
- Material: Exact product name, description, or SKU code as written
- Quantity: Numerical quantity (look for "Qty", "Quantity" columns)
- Unit Price: Price per unit (look for "Unit Price", "Price Each" columns)

DOCUMENT TYPES TO EXPECT:
- Purchase orders with line items
- Invoices with product listings
- Order confirmations with quantities and prices

RESPONSE FORMAT:
Return ONLY a valid JSON object with this exact structure:
{{
  "header": {{
    "client": "exact company name from document or null",
    "date": "exact date from document or null"
  }},
  "line_items": [
    {{
      "material": "exact product name/description or null",
      "quantity": "exact quantity number or null",
      "unit_price": "exact unit price or null"
    }}
  ]
}}

VALIDATION CHECKLIST:
- Is the client name a real company name from the document?
- Is the date in a valid date format from the document?
- Are the materials actual product names/codes from the document?
- Are quantities and prices real numbers from the document?
- Did I avoid making up any information?

Remember: It's better to return null than to guess or invent data.
"""


async def extract_with_anthropic(prompt: str, pdf_content: bytes, temperature: float, max_tokens: int) -> tuple[str, Dict[str, Any]]:
    """Extract data using Anthropic Claude model with PDF content."""
    bedrock = Session().client(model_name="anthropic--claude-3-5-sonnet-20241022")
    
    # Encode PDF as base64 for multimodal input
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    messages: list[Dict[str, Any]] = [
        {
            "role": "user", 
            "content": [
                {"text": prompt},
                {
                    "document": {
                        "format": "pdf",
                        "name": "document.pdf",
                        "source": {
                            "bytes": pdf_base64
                        }
                    }
                }
            ]
        }
    ]
    
    response = bedrock.converse(
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    
    text: str = response["output"]["message"]["content"][0]["text"]
    usage: Dict[str, Any] = response.get("usage", {})
    
    return text, usage


async def extract_with_openai(prompt: str, pdf_content: bytes, temperature: float, max_tokens: int) -> tuple[str, Dict[str, Any]]:
    """Extract data using GPT-4.1 via SAP Generative AI Hub Orchestration Service with PDF document."""
    try:
        # Save PDF to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text from PDF using PyMuPDF
            doc = fitz.open(temp_file_path)
            logger.info(f"PDF opened successfully. Total pages: {len(doc)}")
            
            # Extract text from all pages
            extracted_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                extracted_text += page_text + "\n"
                logger.info(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
            
            doc.close()
            logger.info(f"Total text extracted: {len(extracted_text)} characters")
            logger.info(f"Sample of extracted text: {extracted_text[:500]}...")
            
            # If no text was extracted, this is likely a scanned PDF (image-based)
            if len(extracted_text.strip()) < 10:
                logger.warning("⚠️  WARNING: Very little text extracted from PDF - likely a scanned document")
                logger.warning("This PDF appears to be image-based and requires OCR processing")
                logger.warning("For now, we'll send the PDF directly to the LLM for image processing")
                
                # For image-based PDFs, we need to use a different approach
                # The LLM can process the PDF directly as a document
                extracted_text = f"[SCANNED_PDF_DOCUMENT] This is a scanned PDF document with {len(doc)} page(s). The document appears to be image-based and contains no extractable text. Please analyze the visual content of this document to extract the required information."
            
            # Use the new extraction function format
            filename = "document.pdf"  # Default filename
            response_content = extract_order_data(extracted_text, filename)
            
            logger.info(f"Response from SAP Gen AI Hub: {len(response_content)} characters")
            logger.info(f"Raw response content: {response_content}")
            
            # Extract usage information (default values since the new function doesn't return usage)
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            
            return response_content, usage
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"PDF processing error with SAP Gen AI Hub: {e}")
        logger.error(f"Error details: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")


def extract_order_data(document_content, filename):
    """Extract header and line item data from purchase order PDF using SAP Generative AI Hub"""
    
    # Use the same optimized prompt as the main EXTRACTION_PROMPT but adapted for the new format
    prompt = """You are a precise document processing assistant specialized in extracting data from purchase orders and invoices. 

CRITICAL RULES - FOLLOW EXACTLY:
1. Extract ONLY text that is clearly visible and readable in the document
2. Do NOT generate, invent, or hallucinate any data
3. Do NOT use placeholder names like "Widget A", "Product X", "Company ABC", etc.
4. If text is unclear, blurry, or unreadable, use null instead of guessing
5. Look for actual company names, product codes, and real numerical values
6. Pay special attention to tables with line items containing products and prices

WHAT TO EXTRACT:

HEADER INFORMATION:
- Customer: Look for the customer/client company name (often at the top or in "Ship To"/"Bill To" sections)
- Date: Look for order date, invoice date, or document date (format: MM/DD/YYYY or similar)

LINE ITEMS (from tables or itemized lists):
- Material: Exact product name, description, or SKU code as written
- Quantity: Numerical quantity (look for "Qty", "Quantity" columns)
- Price: Price per unit (look for "Unit Price", "Price Each" columns)

DOCUMENT TYPES TO EXPECT:
- Purchase orders with line items
- Invoices with product listings
- Order confirmations with quantities and prices

RESPONSE FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
  "header": {
    "customer": "exact company name from document or null",
    "date": "exact date from document or null"
  },
  "positions": [
    {
      "material": "exact product name/description or null",
      "quantity": "exact quantity number or null",
      "price": "exact unit price or null"
    }
  ]
}

VALIDATION CHECKLIST:
- Is the customer name a real company name from the document?
- Is the date in a valid date format from the document?
- Are the materials actual product names/codes from the document?
- Are quantities and prices real numbers from the document?
- Did I avoid making up any information?

Remember: It's better to return null than to guess or invent data.

Document content to analyze:
""" + document_content

    try:
        llm = LLM(name="gpt-4.1", version="latest")
        template = Template(messages=[UserMessage(content=prompt)])
        config = OrchestrationConfig(template=template, llm=llm)
        orchestration_service = OrchestrationService(config=config)

        result = orchestration_service.run(template_values=[])
        return result.orchestration_result.choices[0].message.content.strip()
    except Exception as e:
        return f"Error extracting order data: {str(e)}"


async def extract_with_gemini(prompt: str, pdf_content: bytes, temperature: float, max_tokens: int) -> tuple[str, Dict[str, Any]]:
    """Extract data using Google Gemini model with PDF content."""
    model = GenerativeModel("gemini-1.5-pro")
    
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    # Create document part for Gemini
    import tempfile
    import os
    
    # Save PDF to temporary file for Gemini processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_content)
        temp_file_path = temp_file.name
    
    try:
        # Import Part for document handling
        from vertexai.generative_models import Part
        
        # Create document part
        document_part = Part.from_uri(
            uri=f"file://{temp_file_path}",
            mime_type="application/pdf"
        )
        
        response = model.generate_content(
            contents=[prompt, document_part], 
            generation_config=generation_config
        )
        
        text: str = response.text
        usage: Dict[str, int] = {
            "prompt_tokens": (
                response.usage_metadata.prompt_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
            "completion_tokens": (
                response.usage_metadata.candidates_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
            "total_tokens": (
                response.usage_metadata.total_token_count
                if hasattr(response, "usage_metadata")
                else 0
            ),
        }
        
        return text, usage
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post("/upload", response_model=PDFExtractionResponse)
async def upload_and_extract_pdf(file: UploadFile = File(...)) -> PDFExtractionResponse:
    """Upload PDF file and extract structured data using GPT-4.1 via SAP Generative AI Hub.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        PDFExtractionResponse with extracted data or error information
    """
    try:
        logger.info(f"Processing PDF upload: {file.filename} with GPT-4.1")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                error="Only PDF files are supported"
            )
        
        # Read file content
        pdf_content = await file.read()
        logger.info(f"PDF content size: {len(pdf_content)} bytes")
        
        # Validate PDF content size (max 20MB for LLM processing)
        if len(pdf_content) > 20 * 1024 * 1024:
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                error="PDF file too large. Maximum size is 20MB."
            )
        
        # Extract data using GPT-4.1 with PDF content directly
        response_text, usage = await extract_with_openai(EXTRACTION_PROMPT, pdf_content, 0.1, 2000)
        
        logger.info(f"GPT-4.1 response received: {len(response_text)} characters")
        
        # Parse JSON response with improved validation
        try:
            # Clean response text to extract JSON - handle markdown code blocks
            clean_response = response_text.strip()
            
            # Remove markdown code blocks if present
            if clean_response.startswith('```json'):
                clean_response = clean_response.replace('```json', '').replace('```', '').strip()
            elif clean_response.startswith('```'):
                clean_response = clean_response.replace('```', '').strip()
            
            # Find JSON boundaries
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error(f"No JSON found in response: {response_text[:500]}...")
                raise ValueError("No JSON found in response")
                
            json_text = clean_response[json_start:json_end]
            logger.info(f"Extracted JSON text: {json_text[:200]}...")
            
            extracted_json = json.loads(json_text)
            
            # Validate JSON structure
            if not isinstance(extracted_json, dict):
                raise ValueError("Response is not a JSON object")
            
            # Check for both old and new format
            if "header" not in extracted_json:
                raise ValueError("Missing required field 'header' in JSON response")
            
            # Support both "line_items" (old format) and "positions" (new format)
            line_items_key = "positions" if "positions" in extracted_json else "line_items"
            if line_items_key not in extracted_json:
                raise ValueError(f"Missing required field '{line_items_key}' in JSON response")
            
            # Validate header structure
            header = extracted_json.get("header", {})
            if not isinstance(header, dict):
                logger.warning("Header is not a dictionary, using empty header")
                header = {}
            
            # Validate line_items structure
            line_items_data = extracted_json.get(line_items_key, [])
            if not isinstance(line_items_data, list):
                logger.warning("Line items is not a list, using empty list")
                line_items_data = []
            
            # Convert to structured data with validation
            # Support both "client" (old format) and "customer" (new format)
            client_value = header.get("customer") or header.get("client")
            header_data = HeaderData(
                client=client_value if client_value not in [None, "", "null", "Not available"] else None,
                date=header.get("date") if header.get("date") not in [None, "", "null", "Not available"] else None
            )
            
            line_items = []
            for i, item in enumerate(line_items_data):
                if not isinstance(item, dict):
                    logger.warning(f"Line item {i} is not a dictionary, skipping")
                    continue
                    
                # Clean up null/empty values
                material = item.get("material")
                if material in [None, "", "null", "N/A", "Not available"]:
                    material = None
                    
                quantity = item.get("quantity")
                if quantity in [None, "", "null", "N/A", "Not available"]:
                    quantity = None
                    
                # Support both "unit_price" (old format) and "price" (new format)
                unit_price = item.get("price") or item.get("unit_price")
                if unit_price in [None, "", "null", "N/A", "Not available"]:
                    unit_price = None
                
                line_items.append(LineItem(
                    material=material,
                    quantity=quantity,
                    unit_price=unit_price
                ))
            
            extracted_data = ExtractedData(
                header=header_data,
                line_items=line_items
            )
            
            logger.info(f"Successfully extracted data: {len(line_items)} line items")
            logger.info(f"Client: {header_data.client}, Date: {header_data.date}")
            
            return PDFExtractionResponse(
                success=True,
                extracted_data=extracted_data,
                filename=file.filename,
                model_used="gpt-4.1",
                usage=usage
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text (first 1000 chars): {response_text[:1000]}")
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                usage=usage,
                error=f"Failed to parse extraction results: {str(e)}. Response may contain hallucinated or invalid data."
            )
        
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return PDFExtractionResponse(
            success=False,
            filename=file.filename if file else "unknown",
            model_used="gpt-4.1",
            error=str(e)
        )


@router.post("/extract", response_model=PDFExtractionResponse)
async def extract_from_base64(request: PDFUploadRequest) -> PDFExtractionResponse:
    """Extract data from base64 encoded PDF content.
    
    Args:
        request: PDFUploadRequest with base64 encoded PDF content
        
    Returns:
        PDFExtractionResponse with extracted data or error information
    """
    try:
        logger.info(f"Processing base64 PDF: {request.filename} with model: {request.extraction_model}")
        
        # Decode base64 content
        try:
            pdf_content = base64.b64decode(request.file_content)
        except Exception as e:
            return PDFExtractionResponse(
                success=False,
                filename=request.filename,
                model_used=request.extraction_model,
                error=f"Invalid base64 content: {str(e)}"
            )
        
        logger.info(f"Decoded PDF content size: {len(pdf_content)} bytes")
        
        # Validate PDF content size (max 20MB for LLM processing)
        if len(pdf_content) > 20 * 1024 * 1024:
            return PDFExtractionResponse(
                success=False,
                filename=request.filename,
                model_used=request.extraction_model,
                error="PDF file too large. Maximum size is 20MB."
            )
        
        # Extract data using selected LLM with PDF content directly
        if request.extraction_model == "anthropic":
            response_text, usage = await extract_with_anthropic(EXTRACTION_PROMPT, pdf_content, request.temperature, request.max_tokens)
        elif request.extraction_model == "openai":
            response_text, usage = await extract_with_openai(EXTRACTION_PROMPT, pdf_content, request.temperature, request.max_tokens)
        elif request.extraction_model == "gemini":
            response_text, usage = await extract_with_gemini(EXTRACTION_PROMPT, pdf_content, request.temperature, request.max_tokens)
        else:
            return PDFExtractionResponse(
                success=False,
                filename=request.filename,
                model_used=request.extraction_model,
                error=f"Unsupported extraction model: {request.extraction_model}"
            )
        
        logger.info(f"LLM response received: {len(response_text)} characters")
        
        # Parse JSON response with improved validation
        try:
            # Clean response text to extract JSON - handle markdown code blocks
            clean_response = response_text.strip()
            
            # Remove markdown code blocks if present
            if clean_response.startswith('```json'):
                clean_response = clean_response.replace('```json', '').replace('```', '').strip()
            elif clean_response.startswith('```'):
                clean_response = clean_response.replace('```', '').strip()
            
            # Find JSON boundaries
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error(f"No JSON found in response: {response_text[:500]}...")
                raise ValueError("No JSON found in response")
                
            json_text = clean_response[json_start:json_end]
            logger.info(f"Extracted JSON text: {json_text[:200]}...")
            
            extracted_json = json.loads(json_text)
            
            # Validate JSON structure
            if not isinstance(extracted_json, dict):
                raise ValueError("Response is not a JSON object")
            
            # Check for both old and new format
            if "header" not in extracted_json:
                raise ValueError("Missing required field 'header' in JSON response")
            
            # Support both "line_items" (old format) and "positions" (new format)
            line_items_key = "positions" if "positions" in extracted_json else "line_items"
            if line_items_key not in extracted_json:
                raise ValueError(f"Missing required field '{line_items_key}' in JSON response")
            
            # Validate header structure
            header = extracted_json.get("header", {})
            if not isinstance(header, dict):
                logger.warning("Header is not a dictionary, using empty header")
                header = {}
            
            # Validate line_items structure
            line_items_data = extracted_json.get(line_items_key, [])
            if not isinstance(line_items_data, list):
                logger.warning("Line items is not a list, using empty list")
                line_items_data = []
            
            # Convert to structured data with validation
            # Support both "client" (old format) and "customer" (new format)
            client_value = header.get("customer") or header.get("client")
            header_data = HeaderData(
                client=client_value if client_value not in [None, "", "null", "Not available"] else None,
                date=header.get("date") if header.get("date") not in [None, "", "null", "Not available"] else None
            )
            
            line_items = []
            for i, item in enumerate(line_items_data):
                if not isinstance(item, dict):
                    logger.warning(f"Line item {i} is not a dictionary, skipping")
                    continue
                    
                # Clean up null/empty values
                material = item.get("material")
                if material in [None, "", "null", "N/A", "Not available"]:
                    material = None
                    
                quantity = item.get("quantity")
                if quantity in [None, "", "null", "N/A", "Not available"]:
                    quantity = None
                    
                # Support both "unit_price" (old format) and "price" (new format)
                unit_price = item.get("price") or item.get("unit_price")
                if unit_price in [None, "", "null", "N/A", "Not available"]:
                    unit_price = None
                
                line_items.append(LineItem(
                    material=material,
                    quantity=quantity,
                    unit_price=unit_price
                ))
            
            extracted_data = ExtractedData(
                header=header_data,
                line_items=line_items
            )
            
            logger.info(f"Successfully extracted data: {len(line_items)} line items")
            logger.info(f"Client: {header_data.client}, Date: {header_data.date}")
            
            return PDFExtractionResponse(
                success=True,
                extracted_data=extracted_data,
                filename=request.filename,
                model_used=request.extraction_model,
                usage=usage
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text (first 1000 chars): {response_text[:1000]}")
            return PDFExtractionResponse(
                success=False,
                filename=request.filename,
                model_used=request.extraction_model,
                usage=usage,
                error=f"Failed to parse extraction results: {str(e)}. Response may contain hallucinated or invalid data."
            )
        
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return PDFExtractionResponse(
            success=False,
            filename=request.filename,
            model_used=request.extraction_model,
            error=str(e)
        )
