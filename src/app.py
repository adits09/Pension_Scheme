import os
import logging
import traceback
import time
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

# Simple in-memory storage
documents = {}

@app.route("/")
def health():
    return jsonify({
        "status": "Backend running successfully!", 
        "timestamp": str(time.time()),
        "documents_count": len(documents)
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        return jsonify({
            "status": "healthy",
            "documents_stored": len(documents),
            "timestamp": time.time(),
            "message": "Simple backend without AI - ready to process PDFs"
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    logger.info("=== Starting PDF Upload ===")
    
    try:
        # Step 1: Check if file exists in request
        logger.info("Step 1: Checking file in request")
        if 'file' not in request.files:
            logger.error("No 'file' key in request.files")
            logger.info(f"Available keys: {list(request.files.keys())}")
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        logger.info(f"Step 2: File received - name: {file.filename}, type: {type(file)}")
        
        # Step 2: Validate file
        if not file or file.filename == '':
            logger.error("File is empty or has no name")
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file extension: {file.filename}")
            return jsonify({"error": "Only PDF files allowed"}), 400
        
        # Step 3: Process filename
        filename = secure_filename(file.filename)
        logger.info(f"Step 3: Secure filename: {filename}")
        
        # Step 4: Try to read file content (simple test)
        try:
            file_content = file.read()
            file_size = len(file_content)
            logger.info(f"Step 4: File read successfully - size: {file_size} bytes")
            
            # Reset file pointer for potential future use
            file.seek(0)
            
        except Exception as read_error:
            logger.error(f"Could not read file: {str(read_error)}")
            return jsonify({"error": "Could not read uploaded file"}), 400
        
        # Step 5: For now, just store basic info (no PDF processing)
        try:
            document_info = {
                'filename': filename,
                'size': file_size,
                'upload_time': time.time(),
                'status': 'uploaded_successfully',
                'content_preview': f'PDF file with {file_size} bytes'
            }
            
            documents[filename] = document_info
            logger.info(f"Step 5: Document stored successfully: {filename}")
            
            return jsonify({
                "status": "success",
                "message": f"File '{filename}' uploaded successfully! ({file_size} bytes)",
                "filename": filename,
                "size": file_size,
                "note": "PDF processing will be added in next step"
            })
            
        except Exception as storage_error:
            logger.error(f"Storage failed: {str(storage_error)}")
            return jsonify({"error": f"Could not store file info: {str(storage_error)}"}), 500
            
    except Exception as e:
        logger.error("=== UPLOAD FAILED ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("=== Chat Request ===")
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        question = data.get("message", "").strip()
        if not question:
            return jsonify({"error": "Empty message"}), 400
        
        logger.info(f"Question: {question}")
        logger.info(f"Documents available: {len(documents)}")
        
        # Generate simple response
        if documents:
            doc_list = list(documents.keys())
            response_text = f"""I have {len(documents)} document(s) uploaded: {', '.join(doc_list)}

Your question: "{question}"

Currently, I can only confirm that your documents are uploaded successfully. PDF text processing will be added in the next update.

For now, I can tell you:
- Number of documents: {len(documents)}
- Document names: {', '.join(doc_list)}
- Upload times: {[f"{name}: {time.ctime(info['upload_time'])}" for name, info in documents.items()]}"""
        else:
            response_text = f"""Your question: "{question}"

I don't have any documents uploaded yet. Please upload a PDF file first, then I'll be able to help you find information in it.

Once you upload documents, I'll be able to search through them and provide relevant information about Rajasthan government schemes."""
        
        # Simple HTML formatting
        html_response = f'<div class="ai-response"><p>{response_text.replace(chr(10), "</p><p>")}</p></div>'
        
        return jsonify({
            "response": {
                "raw": response_text,
                "html": html_response,
                "text": response_text
            },
            "documents_available": len(documents)
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Debug endpoint to see what documents are stored"""
    try:
        return jsonify({
            "documents": documents,
            "count": len(documents)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large (max 50MB)"}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting simple backend on port {port}")
    logger.info("No AI dependencies - just basic file upload and storage")
    app.run(host="0.0.0.0", port=port, debug=False)