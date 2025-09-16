"""Excel/CSV extraction router for document processing with LLM integration."""

import logging
import base64
import json
import tempfile
import os
import pandas as pd
import io
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File

# Gen AI Hub Orchestration imports
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService

from dotenv import load_dotenv

load_dotenv() 

from ..models.pdf_extraction import PDFUploadRequest, PDFExtractionResponse, ExtractedData, HeaderData, LineItem
from ..security import get_api_key

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_api_key)])

# Extraction prompt template
EXTRACTION_PROMPT = """
You are a document processing assistant. Extract the following information from this Excel/CSV data:

HEADER DATA:
- Client: The client name or company name
- Date: Any date mentioned in the document

LINE ITEMS DATA:
For each line item, extract:
- Material: Product name, description, or material
- Quantity: The quantity or amount
- Unit Price: The price per unit

IMPORTANT: Only include line items where the quantity is greater than 0 (zero). Ignore any rows with quantity = 0.

Please respond with a JSON object in this exact format:
{{
  "header": {{
    "client": "client name or null",
    "date": "date or null"
  }},
  "line_items": [
    {{
      "material": "material name or null",
      "quantity": "quantity or null", 
      "unit_price": "unit price or null"
    }}
  ]
}}

Only return the JSON object, no additional text or explanation.
"""


async def extract_with_openai_excel(prompt: str, excel_data: str, temperature: float, max_tokens: int) -> tuple[str, Dict[str, Any]]:
    """Extract data using GPT-4.1 via SAP Generative AI Hub Orchestration Service with Excel/CSV data."""
    try:
        # Create the prompt with Excel/CSV data
        full_prompt = f"""
{prompt}

Here is the Excel/CSV data to analyze:

{excel_data}

Please extract the requested information according to the format specified above. Remember to exclude any line items with quantity = 0.
"""
        
        # Use SAP Generative AI Hub Orchestration Service
        llm = LLM(name="gpt-4.1", version="latest")
        template = Template(messages=[UserMessage(content=full_prompt)])
        config = OrchestrationConfig(template=template, llm=llm)
        orchestration_service = OrchestrationService(config=config)
        
        logger.info("Sending Excel/CSV data to SAP Generative AI Hub for processing...")
        result = orchestration_service.run(template_values=[])
        response_content = result.orchestration_result.choices[0].message.content.strip()
        
        logger.info(f"Received response from SAP Gen AI Hub: {len(response_content)} characters")
        
        # Extract usage information if available
        usage = {
            "prompt_tokens": getattr(result.orchestration_result.usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(result.orchestration_result.usage, 'completion_tokens', 0),
            "total_tokens": getattr(result.orchestration_result.usage, 'total_tokens', 0),
        }
        
        return response_content, usage
        
    except Exception as e:
        logger.error(f"Excel/CSV processing error with SAP Gen AI Hub: {e}")
        logger.error(f"Error details: {str(e)}")
        raise Exception(f"Failed to process Excel/CSV: {str(e)}")


def process_excel_csv_file(file_content: bytes, filename: str) -> str:
    """Process Excel or CSV file and return formatted data as string."""
    try:
        # Determine file type
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension in ['xlsx', 'xls']:
            # Process Excel file
            df = pd.read_excel(io.BytesIO(file_content))
        elif file_extension == 'csv':
            # Process CSV file
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Filter out rows where any quantity-like column is 0
        # Look for common quantity column names
        quantity_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['quantity', 'qty', 'amount', 'count', 'units']):
                quantity_columns.append(col)
        
        # If we found quantity columns, filter out zero values
        if quantity_columns:
            for qty_col in quantity_columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
                # Filter out rows where quantity is 0 or NaN
                df = df[(df[qty_col] > 0) & (df[qty_col].notna())]
        
        # Convert DataFrame to string representation
        data_string = f"File: {filename}\n"
        data_string += f"Columns: {', '.join(df.columns.tolist())}\n"
        data_string += f"Number of rows (after filtering): {len(df)}\n\n"
        data_string += "Data:\n"
        data_string += df.to_string(index=False, max_rows=100)  # Limit to 100 rows for LLM processing
        
        logger.info(f"Processed {file_extension.upper()} file: {len(df)} rows after filtering")
        
        return data_string
        
    except Exception as e:
        logger.error(f"Error processing Excel/CSV file: {e}")
        raise Exception(f"Failed to process file: {str(e)}")


@router.post("/upload", response_model=PDFExtractionResponse)
async def upload_and_extract_excel(file: UploadFile = File(...)) -> PDFExtractionResponse:
    """Upload Excel/CSV file and extract structured data using GPT-4.1 via SAP Generative AI Hub.
    
    Args:
        file: Uploaded Excel/CSV file
        
    Returns:
        PDFExtractionResponse with extracted data or error information
    """
    try:
        logger.info(f"Processing Excel/CSV upload: {file.filename} with GPT-4.1")
        
        # Validate file type
        file_extension = file.filename.lower().split('.')[-1] if file.filename else ''
        if file_extension not in ['xlsx', 'xls', 'csv']:
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                error="Only Excel (.xlsx, .xls) and CSV files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        logger.info(f"File content size: {len(file_content)} bytes")
        
        # Validate file content size (max 10MB for Excel/CSV processing)
        if len(file_content) > 10 * 1024 * 1024:
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                error="File too large. Maximum size is 10MB."
            )
        
        # Process Excel/CSV file
        excel_data = process_excel_csv_file(file_content, file.filename)
        
        # Extract data using GPT-4.1 with Excel/CSV data
        response_text, usage = await extract_with_openai_excel(EXTRACTION_PROMPT, excel_data, 0.1, 2000)
        
        logger.info(f"GPT-4.1 response received: {len(response_text)} characters")
        
        # Parse JSON response
        try:
            # Clean response text to extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_text = response_text[json_start:json_end]
            extracted_json = json.loads(json_text)
            
            # Convert to structured data
            header_data = HeaderData(
                client=extracted_json.get("header", {}).get("client"),
                date=extracted_json.get("header", {}).get("date")
            )
            
            line_items = []
            for item in extracted_json.get("line_items", []):
                # Additional filtering: ensure quantity is not "0" or 0
                quantity = item.get("quantity")
                if quantity and str(quantity).strip() != "0" and quantity != 0:
                    line_items.append(LineItem(
                        material=item.get("material"),
                        quantity=item.get("quantity"),
                        unit_price=item.get("unit_price")
                    ))
            
            extracted_data = ExtractedData(
                header=header_data,
                line_items=line_items
            )
            
            logger.info(f"Successfully extracted data: {len(line_items)} line items (after filtering zeros)")
            
            return PDFExtractionResponse(
                success=True,
                extracted_data=extracted_data,
                filename=file.filename,
                model_used="gpt-4.1",
                usage=usage
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response text: {response_text}")
            return PDFExtractionResponse(
                success=False,
                filename=file.filename,
                model_used="gpt-4.1",
                usage=usage,
                error=f"Failed to parse extraction results: {str(e)}"
            )
        
    except Exception as e:
        logger.error(f"Excel/CSV extraction error: {e}")
        return PDFExtractionResponse(
            success=False,
            filename=file.filename if file else "unknown",
            model_used="gpt-4.1",
            error=str(e)
        )
