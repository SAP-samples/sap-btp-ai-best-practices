from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, Response
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService
from dotenv import load_dotenv
import os
import json
import pandas as pd
try:
    from docx import Document
except ImportError:
    print("Warning: python-docx not available, Word document processing disabled")
    Document = None
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import tempfile
import time
import threading
from werkzeug.utils import secure_filename
import re

# Global variable to store progress
progress_data = {}

# Import the GenAI Hub proxy for OpenAI-compatible interface
try:
    from gen_ai_hub.proxy.native.openai import chat as openai_chat
except ImportError as e:
    print(f"Warning: gen_ai_hub module not found: {e}")
    openai_chat = None
except Exception as e:
    print(f"Warning: Error initializing GenAI Hub (this is normal in local development): {e}")
    openai_chat = None

# Load environment variables
load_dotenv()

# Configure SAP AI Core credentials directly
os.environ['AICORE_AUTH_URL'] = ""
os.environ['AICORE_CLIENT_ID'] = ""
os.environ['AICORE_CLIENT_SECRET'] = ""
os.environ['AICORE_BASE_URL'] = ""
os.environ['AICORE_RESOURCE_GROUP'] = "default"

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(session_id, step, total_steps, message):
    """Update progress for a specific session"""
    progress_data[session_id] = {
        'step': step,
        'total_steps': total_steps,
        'message': message,
        'percentage': int((step / total_steps) * 100),
        'timestamp': time.time()
    }

def extract_purchase_order_data(file_path, filename):
    """Extract purchase order data by sending PDF directly to LLM"""
    try:
        # Read PDF file as binary data
        with open(file_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        
        # Create prompt for direct PDF analysis
        prompt = f"""
You are an expert in purchase order analysis. Analyze this purchase order PDF document directly and extract the specific data needed to create a sales order.

DATA TO EXTRACT:
1. Customer: Name of the company making the purchase order (NOT Acme Corporation)
2. Date: Purchase order date
3. Products: For each product line extract:
   - Product description (complete text)
   - Quantity ordered
   - Unit price per item

SPECIFIC INSTRUCTIONS:
- The vendor will always be "Acme Corporation"
- Look for products containing "Acme Corporation", "Acme", "AC" in the description
- For quantity, look for numbers representing cases, units, etc.
- For price, look for cost per case/individual unit
- If there are multiple products, include all in a list
- Keep product names exactly as they appear

OUTPUT FORMAT (JSON):
{{
    "customer": "customer company name",
    "date": "order date",
    "vendor": "Acme Corporation",
    "products": [
        {{
            "description": "complete product description",
            "quantity": number,
            "unit_price": number
        }}
    ]
}}

Document: {filename}
Analyze the PDF directly and extract the data in JSON format:

PDF (base64): {pdf_base64}
"""

        try:
            # Configure LLM with gpt-4.1
            model_name = "gpt-4.1"
            llm = LLM(name=model_name, version="latest")
            template = Template(messages=[UserMessage(content=prompt)])
            config = OrchestrationConfig(template=template, llm=llm)
            
            # Initialize orchestration service
            orchestration_service = OrchestrationService(config=config)
            result = orchestration_service.run(template_values=[])
            llm_response = result.orchestration_result.choices[0].message.content.strip()
            
            # Try to parse JSON from response
            try:
                # Extract JSON from the response (in case there's extra text)
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_data = json.loads(json_str)
                    return parsed_data
                else:
                    return {"error": "Could not extract JSON from response", "raw_response": llm_response}
            except json.JSONDecodeError as e:
                return {"error": f"Error parsing JSON: {str(e)}", "raw_response": llm_response}
                
        except Exception as e:
            return {"error": f"Error processing PDF with LLM: {str(e)}"}
            
    except Exception as e:
        return {"error": f"Error extracting purchase order data: {str(e)}"}

def extract_text_from_pdf_with_llm(file_path, session_id=None):
    """Extract text from PDF using LLM for both text and images with progress tracking"""
    try:
        doc = fitz.open(file_path)
        if len(doc) == 0:
            return "Error: PDF has no pages"
        
        total_pages = len(doc)
        all_content = []
        
        if session_id:
            update_progress(session_id, 0, total_pages + 1, f"Starting PDF processing ({total_pages} pages)")
        
        for page_num in range(total_pages):
            if session_id:
                update_progress(session_id, page_num + 1, total_pages + 1, f"Processing page {page_num + 1} of {total_pages}")
            
            page = doc.load_page(page_num)
            page_content = f"\n--- PAGE {page_num + 1} ---\n"
            
            # Use LLM to extract and interpret the entire page content
            if session_id:
                update_progress(session_id, page_num + 1, total_pages + 1, f"Extracting text with LLM - page {page_num + 1}")
            
            page_text = extract_page_content_with_llm(page, page_num + 1)
            page_content += page_text
            
            all_content.append(page_content)
        
        doc.close()
        
        if session_id:
            update_progress(session_id, total_pages + 1, total_pages + 1, "PDF processing completed")
        
        final_content = '\n'.join(all_content)
        if not final_content.strip():
            return "Error: Could not extract content from PDF"
        
        return final_content
        
    except Exception as e:
        return f"Error processing PDF with LLM: {str(e)}"

def extract_page_content_with_llm(page, page_num):
    """Extract content from a PDF page using basic text extraction"""
    try:
        # Use basic text extraction from PyMuPDF - more reliable
        basic_text = page.get_text()
        
        # Also try to extract text from tables
        tables = page.find_tables()
        table_text = ""
        
        if tables:
            for table in tables:
                try:
                    table_data = table.extract()
                    table_text += f"\n--- TABLE ON PAGE {page_num} ---\n"
                    for row in table_data:
                        if row:  # Skip empty rows
                            table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                except Exception as e:
                    table_text += f"\n[Error extracting table: {str(e)}]\n"
        
        # Combine basic text and table text
        combined_text = basic_text
        if table_text:
            combined_text += table_text
            
        return combined_text if combined_text.strip() else f"[PAGE {page_num} EMPTY]"
            
    except Exception as e:
        return f"[ERROR PROCESSING PAGE {page_num}]: {str(e)}"

def extract_text_from_file(file_path, filename):
    """Extract text content from various file types"""
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if not content.strip():
                    return "Error: Text file is empty"
                return content
        
        elif file_extension in ['docx', 'doc']:
            if Document is None:
                return "Error: Word document processing not available"
            try:
                doc = Document(file_path)
                text = []
                for paragraph in doc.paragraphs:
                    text.append(paragraph.text)
                content = '\n'.join(text)
                if not content.strip():
                    return "Error: Word document contains no extractable text"
                return content
            except Exception as e:
                return f"Error processing Word document: {str(e)}"
        
        elif file_extension == 'pdf':
            # Use LLM-based extraction for PDFs
            return extract_text_from_pdf_with_llm(file_path)
        
        else:
            return "Unsupported file format"
            
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@app.route('/api/extract_purchase_order', methods=['POST'])
def api_extract_purchase_order():
    """API endpoint for extracting purchase order data from PDF documents"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF, Word and text files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract purchase order data
            purchase_order_data = extract_purchase_order_data(file_path, filename)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            if 'error' in purchase_order_data:
                return jsonify({'error': purchase_order_data['error']}), 500
            
            return jsonify({
                'success': True,
                'filename': filename,
                'purchase_order_data': purchase_order_data,
                'message': 'Purchase order data extracted successfully'
            })
        
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
