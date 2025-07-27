import os
import logging
import traceback
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)
load_dotenv()

# Global variables - start with simple storage
chunks_storage = {}
llm = None
conversation = None

@app.route("/")
def health():
    return jsonify({"status": "Backend is running successfully!", "timestamp": str(time.time())})

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        status = {
            "status": "healthy",
            "llm_initialized": llm is not None,
            "document_count": len(chunks_storage),
            "python_version": os.sys.version,
            "timestamp": time.time()
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        logger.info("Upload PDF endpoint called")
        global chunks_storage
        
        # Check if file is in request
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        logger.info(f"Processing file: {filename}")
        
        # Try to import and use PyMuPDF
        try:
            import fitz
        except ImportError as e:
            logger.error(f"PyMuPDF import error: {str(e)}")
            return jsonify({"error": "PDF processing library not available"}), 500
        
        # Create temp directory
        temp_dir = "/tmp" if os.path.exists("/tmp") else "temp_uploads"
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create temp directory: {str(e)}")
            return jsonify({"error": "Server storage error"}), 500
            
        filepath = os.path.join(temp_dir, filename)
        
        # Save file
        try:
            file.save(filepath)
            logger.info(f"File saved to: {filepath}")
        except Exception as e:
            logger.error(f"Could not save file: {str(e)}")
            return jsonify({"error": "Could not save uploaded file"}), 500

        # Extract text from PDF
        try:
            doc = fitz.open(filepath)
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            doc.close()
            logger.info(f"Extracted {len(full_text)} characters from PDF")
        except Exception as pdf_error:
            logger.error(f"PDF processing error: {str(pdf_error)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Could not process PDF: {str(pdf_error)}"}), 400
        
        if not full_text.strip():
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Simple text chunking
        try:
            # Split by sentences using simple regex
            import re
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create chunks of 5 sentences with 2 sentence overlap
            chunks = []
            chunk_size = 5
            overlap = 2
            
            for i in range(0, len(sentences), chunk_size - overlap):
                chunk_text = " ".join(sentences[i:i + chunk_size])
                if chunk_text.strip():
                    chunks.append(chunk_text)
            
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Text chunking error: {str(e)}")
            chunks = [full_text]  # Fallback to single chunk

        # Store chunks in memory
        try:
            chunks_storage[filename] = {
                'chunks': chunks,
                'upload_time': time.time(),
                'full_text': full_text[:1000] + "..." if len(full_text) > 1000 else full_text  # Store summary
            }
            logger.info(f"Stored {len(chunks)} chunks for {filename}")
        except Exception as e:
            logger.error(f"Storage error: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Could not store document data"}), 500

        # Clean up file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {str(e)}")
        
        return jsonify({
            "status": "success",
            "message": f"PDF processed successfully! Created {len(chunks)} chunks.",
            "chunks_created": len(chunks),
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        logger.info("Chat endpoint called")
        global chunks_storage
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        question = data.get("message", "").strip()
        if not question:
            return jsonify({"error": "Empty message"}), 400

        logger.info(f"Question: {question}")
        logger.info(f"Available documents: {len(chunks_storage)}")

        # Get context from stored chunks
        context = ""
        context_count = 0
        
        if chunks_storage:
            # Simple keyword matching
            question_words = set(question.lower().split())
            relevant_chunks = []
            
            for filename, file_data in chunks_storage.items():
                for chunk in file_data['chunks']:
                    chunk_words = set(chunk.lower().split())
                    # Find common words
                    common_words = question_words & chunk_words
                    if len(common_words) > 0:
                        relevance_score = len(common_words)
                        relevant_chunks.append((chunk, relevance_score))
            
            # Sort by relevance and take top 3
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            context_chunks = [chunk for chunk, _ in relevant_chunks[:3]]
            context = "\n\n".join(context_chunks)
            context_count = len(context_chunks)
            
            logger.info(f"Found {context_count} relevant chunks")

        # Generate response
        response_text = ""
        
        if context:
            response_text = f"""Based on the uploaded documents:

{context}

For your question: "{question}"

I found {context_count} relevant sections in the uploaded documents. The information above should help answer your question about Rajasthan government schemes."""
        else:
            if chunks_storage:
                response_text = f"""I couldn't find specific information related to your question "{question}" in the uploaded documents. 

The documents contain information about Rajasthan government schemes, but your question might need to be more specific or the relevant information might not be in the uploaded files.

Try asking about:
- Specific scheme names
- Eligibility criteria
- Application processes
- Required documents

Or upload more relevant documents that might contain the information you're looking for."""
            else:
                response_text = f"""I don't have any documents uploaded yet to answer your question about "{question}".

Please upload PDF documents related to Rajasthan government schemes first, and then I'll be able to help you find specific information.

You can ask questions about:
- Government scheme details
- Eligibility requirements  
- Application procedures
- Required documents
- Contact information"""

        # Format response
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup('<div class="ai-response"></div>', 'html.parser')
            container = soup.find('div')
            
            paragraphs = response_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    p = soup.new_tag('p')
                    p.string = para.strip()
                    container.append(p)
            
            formatted_html = str(soup)
        except Exception as e:
            logger.warning(f"HTML formatting failed: {str(e)}")
            formatted_html = f'<div class="ai-response"><p>{response_text}</p></div>'

        return jsonify({
            "response": {
                "raw": response_text,
                "html": formatted_html,
                "text": response_text
            },
            "context_found": context_count
        })
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large (max 50MB)"}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Available documents: {len(chunks_storage)}")
    app.run(host="0.0.0.0", port=port, debug=False)