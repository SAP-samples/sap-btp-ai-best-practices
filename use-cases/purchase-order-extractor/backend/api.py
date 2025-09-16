from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import time
from werkzeug.utils import secure_filename
import pdf

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Acme Corporation Purchase Order API',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

@app.route('/api/extract-purchase-order', methods=['POST'])
def extract_purchase_order():
    """
    Extract purchase order data from uploaded PDF file
    Returns structured JSON with client, date, vendor, and products
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'message': 'Please upload a PDF file'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type',
                'message': 'Only PDF, Word, and text files are allowed',
                'allowed_extensions': list(ALLOWED_EXTENSIONS)
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract purchase order data using pdf module
            extraction_result = pdf.extract_purchase_order_data(file_path, filename)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            # Check if extraction was successful
            if 'error' in extraction_result:
                return jsonify({
                    'success': False,
                    'error': 'Extraction failed',
                    'message': extraction_result['error'],
                    'filename': filename,
                    'raw_response': extraction_result.get('raw_response', '')
                }), 500
            
            # Validate extracted data structure
            required_fields = ['cliente', 'fecha', 'vendor', 'productos']
            missing_fields = [field for field in required_fields if field not in extraction_result]
            
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': 'Incomplete data extraction',
                    'message': f'Missing required fields: {", ".join(missing_fields)}',
                    'filename': filename,
                    'extracted_data': extraction_result
                }), 500
            
            # Calculate summary statistics
            productos = extraction_result.get('productos', [])
            total_productos = len(productos)
            total_cantidad = sum(p.get('cantidad', 0) for p in productos if isinstance(p.get('cantidad'), (int, float)))
            total_valor = sum(
                p.get('cantidad', 0) * p.get('precio_unitario', 0) 
                for p in productos 
                if isinstance(p.get('cantidad'), (int, float)) and isinstance(p.get('precio_unitario'), (int, float))
            )
            
            # Return successful response with structured JSON
            return jsonify({
                'success': True,
                'message': 'Purchase order data extracted successfully',
                'filename': filename,
                'processing_timestamp': time.time(),
                'data': {
                    'cliente': extraction_result['cliente'],
                    'fecha': extraction_result['fecha'],
                    'vendor': extraction_result['vendor'],
                    'productos': productos,
                    'resumen': {
                        'total_productos': total_productos,
                        'total_cantidad': total_cantidad,
                        'total_valor': round(total_valor, 2)
                    }
                }
            })
        
        except Exception as processing_error:
            # Clean up file on processing error
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': False,
                'error': 'Processing error',
                'message': f'Error processing file: {str(processing_error)}',
                'filename': filename
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': f'Unexpected server error: {str(e)}'
        }), 500

@app.route('/api/batch-extract', methods=['POST'])
def batch_extract_purchase_orders():
    """
    Extract purchase order data from multiple uploaded PDF files
    Returns array of results for each file
    """
    try:
        # Check if files are present in request
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided',
                'message': 'Please upload one or more PDF files'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                'success': False,
                'error': 'No files selected',
                'message': 'Please select files to upload'
            }), 400
        
        results = []
        successful_extractions = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                try:
                    # Extract purchase order data
                    extraction_result = pdf.extract_purchase_order_data(file_path, filename)
                    
                    # Clean up file
                    os.remove(file_path)
                    
                    if 'error' in extraction_result:
                        results.append({
                            'filename': filename,
                            'success': False,
                            'error': extraction_result['error'],
                            'message': 'Failed to extract data from this file'
                        })
                    else:
                        # Calculate summary for this file
                        productos = extraction_result.get('productos', [])
                        total_cantidad = sum(p.get('cantidad', 0) for p in productos if isinstance(p.get('cantidad'), (int, float)))
                        total_valor = sum(
                            p.get('cantidad', 0) * p.get('precio_unitario', 0) 
                            for p in productos 
                            if isinstance(p.get('cantidad'), (int, float)) and isinstance(p.get('precio_unitario'), (int, float))
                        )
                        
                        results.append({
                            'filename': filename,
                            'success': True,
                            'data': {
                                'cliente': extraction_result['cliente'],
                                'fecha': extraction_result['fecha'],
                                'vendor': extraction_result['vendor'],
                                'productos': productos,
                                'resumen': {
                                    'total_productos': len(productos),
                                    'total_cantidad': total_cantidad,
                                    'total_valor': round(total_valor, 2)
                                }
                            }
                        })
                        successful_extractions += 1
                
                except Exception as file_error:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': str(file_error),
                        'message': 'Error processing this file'
                    })
            else:
                results.append({
                    'filename': file.filename if file else 'unknown',
                    'success': False,
                    'error': 'Invalid file type',
                    'message': 'File type not allowed'
                })
        
        # Calculate batch summary
        total_files = len(results)
        success_rate = (successful_extractions / total_files * 100) if total_files > 0 else 0
        
        return jsonify({
            'success': True,
            'message': f'Batch processing completed. {successful_extractions}/{total_files} files processed successfully',
            'batch_summary': {
                'total_files': total_files,
                'successful_extractions': successful_extractions,
                'failed_extractions': total_files - successful_extractions,
                'success_rate': round(success_rate, 1)
            },
            'results': results,
            'processing_timestamp': time.time()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Batch processing error',
            'message': f'Error during batch processing: {str(e)}'
        }), 500

@app.route('/api/export-json', methods=['POST'])
def export_to_json():
    """
    Export purchase order data to a downloadable JSON file
    Expects the extracted data in the request body
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'message': 'Please provide purchase order data to export'
            }), 400
        
        # Create export structure
        export_data = {
            'metadata': {
                'export_timestamp': time.time(),
                'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'vendor': 'Acme Corporation',
                'total_orders': len(data) if isinstance(data, list) else 1
            },
            'purchase_orders': data if isinstance(data, list) else [data]
        }
        
        return jsonify({
            'success': True,
            'message': 'Data prepared for export',
            'export_data': export_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Export error',
            'message': f'Error preparing data for export: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': f'File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested API endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred on the server'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Acme Corporation Purchase Order API...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📄 Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📏 Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("🌐 CORS enabled for frontend communication")
    print("=" * 50)
    
    # Use PORT environment variable for Cloud Foundry compatibility
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_ENV', 'development') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
