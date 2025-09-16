from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
from dotenv import load_dotenv
import os
import json
import time
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import pdf

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure Flask-RESTX
api = Api(
    app,
    version='1.0.0',
    title='Acme Corporation PDF Extractor API',
    description='REST API to extract purchase order data from PDF files using SAP Generative AI Hub',
    doc='/swagger/',
    prefix='/api'
)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define namespaces
ns_health = api.namespace('health', description='Health check operations')
ns_extract = api.namespace('extract', description='PDF extraction operations')
ns_export = api.namespace('export', description='Data export operations')

# Define models for Swagger documentation
product_model = api.model('Product', {
    'description': fields.String(required=True, description='Product description'),
    'quantity': fields.Float(required=True, description='Product quantity'),
    'unit_price': fields.Float(required=True, description='Unit price of the product')
})

summary_model = api.model('Summary', {
    'total_products': fields.Integer(description='Total number of products'),
    'total_quantity': fields.Float(description='Total quantity'),
    'total_value': fields.Float(description='Total value')
})

extraction_data_model = api.model('ExtractionData', {
    'customer': fields.String(required=True, description='Customer name'),
    'date': fields.String(required=True, description='Order date'),
    'vendor': fields.String(required=True, description='Vendor (always Acme Corporation)'),
    'products': fields.List(fields.Nested(product_model), description='List of products'),
    'summary': fields.Nested(summary_model, description='Order summary')
})

success_response_model = api.model('SuccessResponse', {
    'success': fields.Boolean(required=True, description='Indicates if the operation was successful'),
    'message': fields.String(required=True, description='Descriptive message'),
    'filename': fields.String(description='Name of the processed file'),
    'processing_timestamp': fields.Float(description='Processing timestamp'),
    'data': fields.Nested(extraction_data_model, description='Extracted data')
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Indicates if the operation was successful'),
    'error': fields.String(required=True, description='Error type'),
    'message': fields.String(required=True, description='Descriptive error message')
})

batch_result_model = api.model('BatchResult', {
    'filename': fields.String(required=True, description='File name'),
    'success': fields.Boolean(required=True, description='Indicates if processing was successful'),
    'data': fields.Nested(extraction_data_model, description='Extracted data (if successful)'),
    'error': fields.String(description='Error message (if failed)')
})

batch_summary_model = api.model('BatchSummary', {
    'total_files': fields.Integer(description='Total files processed'),
    'successful_extractions': fields.Integer(description='Successful extractions'),
    'failed_extractions': fields.Integer(description='Failed extractions'),
    'success_rate': fields.Float(description='Success rate percentage')
})

batch_response_model = api.model('BatchResponse', {
    'success': fields.Boolean(required=True, description='Indicates if the operation was successful'),
    'message': fields.String(required=True, description='Descriptive message'),
    'batch_summary': fields.Nested(batch_summary_model, description='Batch processing summary'),
    'results': fields.List(fields.Nested(batch_result_model), description='Individual results'),
    'processing_timestamp': fields.Float(description='Processing timestamp')
})

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@ns_health.route('/')
class HealthCheck(Resource):
    @api.doc('health_check')
    @api.marshal_with(api.model('HealthResponse', {
        'status': fields.String(description='Service status'),
        'service': fields.String(description='Service name'),
        'timestamp': fields.Float(description='Current timestamp'),
        'version': fields.String(description='API version')
    }))
    def get(self):
        """API health check verification"""
        return {
            'status': 'healthy',
            'service': 'Acme Corporation Purchase Order API',
            'timestamp': time.time(),
            'version': '1.0.0'
        }

# File upload parser
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True, help='PDF file to process')

batch_upload_parser = reqparse.RequestParser()
batch_upload_parser.add_argument('files', location='files', type=FileStorage, action='append', required=True, help='PDF files to process')

@ns_extract.route('/purchase-order')
class ExtractPurchaseOrder(Resource):
    @api.doc('extract_purchase_order')
    @api.expect(upload_parser)
    @api.marshal_with(success_response_model, code=200, description='Successful extraction')
    @api.marshal_with(error_response_model, code=400, description='Request error')
    @api.marshal_with(error_response_model, code=500, description='Server error')
    def post(self):
        """
        Extract purchase order data from a PDF file
        
        Returns structured JSON data with customer, date, vendor and products.
        The vendor will always be "Acme Corporation".
        """
        try:
            args = upload_parser.parse_args()
            file = args['file']
            
            # Check if file is selected
            if not file or file.filename == '':
                return {
                    'success': False,
                    'error': 'No file selected',
                    'message': 'Please select a file to upload'
                }, 400
            
            # Check file type
            if not allowed_file(file.filename):
                return {
                    'success': False,
                    'error': 'Invalid file type',
                    'message': 'Only PDF, Word, and text files are allowed'
                }, 400
            
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
                    return {
                        'success': False,
                        'error': 'Extraction failed',
                        'message': extraction_result['error']
                    }, 500
                
                # Validate extracted data structure
                required_fields = ['customer', 'date', 'vendor', 'products']
                missing_fields = [field for field in required_fields if field not in extraction_result]
                
                if missing_fields:
                    return {
                        'success': False,
                        'error': 'Incomplete data extraction',
                        'message': f'Missing required fields: {", ".join(missing_fields)}'
                    }, 500
                
                # Map the response structure from pdf module to API response
                products = extraction_result.get('products', [])
                
                # Convert products structure if needed
                products_formatted = []
                for product in products:
                    products_formatted.append({
                        'description': product.get('description', ''),
                        'quantity': product.get('quantity', 0),
                        'unit_price': product.get('unit_price', 0)
                    })
                
                # Calculate summary statistics
                total_products = len(products_formatted)
                total_quantity = sum(p.get('quantity', 0) for p in products_formatted if isinstance(p.get('quantity'), (int, float)))
                total_value = sum(
                    p.get('quantity', 0) * p.get('unit_price', 0) 
                    for p in products_formatted 
                    if isinstance(p.get('quantity'), (int, float)) and isinstance(p.get('unit_price'), (int, float))
                )
                
                # Return successful response with structured JSON
                return {
                    'success': True,
                    'message': 'Purchase order data extracted successfully',
                    'filename': filename,
                    'processing_timestamp': time.time(),
                    'data': {
                        'customer': extraction_result.get('customer', ''),
                        'date': extraction_result.get('date', ''),
                        'vendor': extraction_result.get('vendor', 'Acme Corporation'),
                        'products': products_formatted,
                        'summary': {
                            'total_products': total_products,
                            'total_quantity': total_quantity,
                            'total_value': round(total_value, 2)
                        }
                    }
                }
            
            except Exception as processing_error:
                # Clean up file on processing error
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return {
                    'success': False,
                    'error': 'Processing error',
                    'message': f'Error processing file: {str(processing_error)}'
                }, 500
        
        except Exception as e:
            return {
                'success': False,
                'error': 'Server error',
                'message': f'Unexpected server error: {str(e)}'
            }, 500

@ns_extract.route('/batch')
class BatchExtractPurchaseOrders(Resource):
    @api.doc('batch_extract_purchase_orders')
    @api.expect(batch_upload_parser)
    @api.marshal_with(batch_response_model, code=200, description='Batch processing completed')
    @api.marshal_with(error_response_model, code=400, description='Request error')
    def post(self):
        """
        Extract data from multiple purchase order PDF files
        
        Processes multiple files simultaneously and returns an array of results.
        """
        try:
            args = batch_upload_parser.parse_args()
            files = args['files']
            
            if not files:
                return {
                    'success': False,
                    'error': 'No files provided',
                    'message': 'Please upload one or more PDF files'
                }, 400
            
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
                            products = extraction_result.get('products', [])
                            total_quantity = sum(p.get('quantity', 0) for p in products if isinstance(p.get('quantity'), (int, float)))
                            total_value = sum(
                                p.get('quantity', 0) * p.get('unit_price', 0) 
                                for p in products 
                                if isinstance(p.get('quantity'), (int, float)) and isinstance(p.get('unit_price'), (int, float))
                            )
                            
                            results.append({
                                'filename': filename,
                                'success': True,
                                'data': {
                                    'customer': extraction_result['customer'],
                                    'date': extraction_result['date'],
                                    'vendor': extraction_result['vendor'],
                                    'products': products,
                                    'summary': {
                                        'total_products': len(products),
                                        'total_quantity': total_quantity,
                                        'total_value': round(total_value, 2)
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
            
            return {
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
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': 'Batch processing error',
                'message': f'Error during batch processing: {str(e)}'
            }, 500

export_parser = reqparse.RequestParser()
export_parser.add_argument('data', type=dict, required=True, help='Data to export in JSON format')

@ns_export.route('/json')
class ExportToJson(Resource):
    @api.doc('export_to_json')
    @api.expect(api.model('ExportRequest', {
        'data': fields.Raw(required=True, description='Purchase order data to export')
    }))
    def post(self):
        """
        Export purchase order data to structured JSON format
        
        Prepares extracted data for download in JSON format with metadata.
        """
        try:
            data = request.get_json()
            
            if not data:
                return {
                    'success': False,
                    'error': 'No data provided',
                    'message': 'Please provide purchase order data to export'
                }, 400
            
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
            
            return {
                'success': True,
                'message': 'Data prepared for export',
                'export_data': export_data
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': 'Export error',
                'message': f'Error preparing data for export: {str(e)}'
            }, 500

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
    print("🚀 Starting Acme Corporation Purchase Order API with Swagger...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📄 Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"📏 Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print("🌐 CORS enabled for frontend communication")
    print("📚 Swagger documentation available at /swagger/")
    print("=" * 50)
    
    # Use PORT environment variable for Cloud Foundry compatibility
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_ENV', 'development') != 'production'
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
