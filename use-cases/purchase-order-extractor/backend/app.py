from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, Response
from dotenv import load_dotenv
import os
import json
import time
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

# Import PDF processing module
import pdf

# Load environment variables
load_dotenv()

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

# Global variable to store progress and results
progress_data = {}
processed_documents = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(session_id, step, total_steps, message):
    """Update progress for a specific session"""
    if session_id not in progress_data:
        progress_data[session_id] = {}
    
    progress_data[session_id].update({
        'step': step,
        'total_steps': total_steps,
        'message': message,
        'percentage': int((step / total_steps) * 100),
        'timestamp': time.time()
    })

def process_purchase_order_async(file_path, filename, session_id):
    """Process purchase order document asynchronously with progress tracking"""
    try:
        print(f"[DEBUG] Initiating process for PO {filename} with session_id: {session_id}")
        update_progress(session_id, 1, 3, "Extracting Purchase Order data...")
        
        # Extract purchase order data
        result = pdf.extract_purchase_order_data(file_path, filename)
        print(f"[DEBUG] Resultado del procesamiento para {filename}: {result}")
        
        # Determine success
        success = 'error' not in result
        
        # Store results
        document_result = {
            'filename': filename,
            'type': 'purchase_order',
            'type_description': 'Purchase Order - Acme Corporation',
            'result': {
                'success': success,
                'data': result if success else None,
                'error': result.get('error') if not success else None
            },
            'timestamp': time.time(),
            'session_id': session_id
        }
        
        # Add to processed documents list
        processed_documents.append(document_result)
        
        # Store in progress data for retrieval
        if session_id not in progress_data:
            progress_data[session_id] = {}
        progress_data[session_id]['document_result'] = document_result
        
        if success:
            update_progress(session_id, 3, 3, "Process successful")
        else:
            update_progress(session_id, 3, 3, f"Error: {result.get('error', 'Unknown Error')}")
        
        print(f"[DEBUG] Processing complete for {filename}")
        
        # Clean up file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[DEBUG] Archivo temporal eliminado: {file_path}")
        except Exception as cleanup_error:
            print(f"[DEBUG] Error limpiando archivo {file_path}: {cleanup_error}")
        
    except Exception as e:
        print(f"[ERROR] Error procesando {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Store error result
        error_result = {
            'filename': filename,
            'type': 'purchase_order',
            'type_description': 'Orden de Compra - Error',
            'result': {
                'success': False,
                'error': str(e),
                'data': None
            },
            'timestamp': time.time(),
            'session_id': session_id
        }
        
        processed_documents.append(error_result)
        progress_data[session_id]['document_result'] = error_result
        
        update_progress(session_id, 3, 3, f"Error: {str(e)}")
        
        # Clean up file even on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page for purchase order processing"""
    if request.method == 'POST':
        # Handle file upload
        if 'files' in request.files:
            files = request.files.getlist('files')
            
            if not files or all(file.filename == '' for file in files):
                flash('No se seleccionaron archivos', 'error')
                return redirect(request.url)
            
            # Process multiple files
            session_ids = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Generate unique session ID for progress tracking
                    session_id = f"upload_{int(time.time() * 1000)}_{len(session_ids)}"
                    session_ids.append(session_id)
                    
                    # Start processing in background thread
                    thread = threading.Thread(
                        target=process_purchase_order_async,
                        args=(file_path, filename, session_id)
                    )
                    thread.daemon = True
                    thread.start()
            
            # Store session IDs for tracking
            session['processing_sessions'] = session_ids
            return redirect(url_for('processing'))
    
    # Get processed documents for display
    recent_documents = processed_documents[-10:]  # Show last 10 documents
    
    return render_template('index.html', processed_documents=recent_documents)

@app.route('/processing')
def processing():
    """Processing page with progress tracking"""
    session_ids = session.get('processing_sessions', [])
    return render_template('processing.html', session_ids=session_ids)

@app.route('/upload_files', methods=['POST'])
def upload_files():
    """Handle multiple file upload via AJAX"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No se proporcionaron archivos'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No se seleccionaron archivos'}), 400
        
        session_ids = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Generate unique session ID for progress tracking
                session_id = f"upload_{int(time.time() * 1000)}_{len(session_ids)}"
                session_ids.append(session_id)
                
                # Start processing in background thread
                thread = threading.Thread(
                    target=process_purchase_order_async,
                    args=(file_path, filename, session_id)
                )
                thread.daemon = True
                thread.start()
        
        return jsonify({'session_ids': session_ids})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        while True:
            if session_id in progress_data:
                data = progress_data[session_id]
                yield f"data: {json.dumps(data)}\n\n"
                
                # If processing is complete, send final message and break
                if data['step'] >= data['total_steps']:
                    time.sleep(1)  # Give client time to process final update
                    break
            else:
                yield f"data: {json.dumps({'step': 0, 'total_steps': 3, 'message': 'Iniciando...', 'percentage': 0})}\n\n"
            
            time.sleep(0.5)  # Update every 500ms
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/get_result/<session_id>')
def get_result(session_id):
    """Get the final result of document processing"""
    if session_id in progress_data and 'document_result' in progress_data[session_id]:
        result = progress_data[session_id]['document_result']
        
        # Clean up progress data
        del progress_data[session_id]
        
        return jsonify({
            'success': True,
            'document': result
        })
    else:
        return jsonify({'error': 'Resultado no encontrado'}), 404

@app.route('/results')
def results():
    """Results page showing all processed documents"""
    return render_template('results.html', documents=processed_documents)

@app.route('/get_all_documents')
def get_all_documents():
    """Get all processed documents"""
    return jsonify({
        'documents': processed_documents[-20:]  # Return last 20 documents
    })

@app.route('/clear_documents', methods=['POST'])
def clear_documents():
    """Clear all processed documents"""
    global processed_documents
    processed_documents = []
    return jsonify({'success': True})

# API endpoint from pdf.py module
@app.route('/api/extract_purchase_order', methods=['POST'])
def api_extract_purchase_order():
    """API endpoint for extracting purchase order data from PDF documents"""
    return pdf.api_extract_purchase_order()

@app.route('/api/export_purchase_orders', methods=['GET'])
def export_purchase_orders():
    """Export all processed purchase orders in a structured format"""
    try:
        # Filter only successful purchase orders
        purchase_orders = [doc for doc in processed_documents 
                          if doc['type'] == 'purchase_order' and doc['result'].get('success', False)]
        
        export_data = {
            'metadata': {
                'total_orders': len(purchase_orders),
                'export_timestamp': time.time(),
                'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'vendor': 'Acme Corporation'
            },
            'purchase_orders': []
        }
        
        for doc in purchase_orders:
            order_data = doc['result']['data']
            structured_order = {
                'filename': doc['filename'],
                'processing_timestamp': doc['timestamp'],
                'processing_date': datetime.fromtimestamp(doc['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                'cliente': order_data.get('cliente', ''),
                'fecha': order_data.get('fecha', ''),
                'vendor': order_data.get('vendor', 'Acme Corporation'),
                'productos': order_data.get('productos', []),
                'total_productos': len(order_data.get('productos', [])),
                'session_id': doc.get('session_id', 'unknown')
            }
            
            # Calculate totals if possible
            total_cantidad = sum(p.get('cantidad', 0) for p in order_data.get('productos', []))
            total_valor = sum(p.get('cantidad', 0) * p.get('precio_unitario', 0) for p in order_data.get('productos', []))
            
            structured_order['totales'] = {
                'total_cantidad': total_cantidad,
                'total_valor': total_valor
            }
            
            export_data['purchase_orders'].append(structured_order)
        
        return jsonify(export_data)
    
    except Exception as e:
        return jsonify({'error': f'Error exportando órdenes de compra: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
