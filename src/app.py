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
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        logger.info(f"Step 2: File received - name: {file.filename}")
        
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
        
        # Step 4: Save file temporarily and extract text
        try:
            # Create temp directory
            temp_dir = "/tmp" if os.path.exists("/tmp") else "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            filepath = os.path.join(temp_dir, filename)
            
            # Save file
            file.save(filepath)
            logger.info(f"File saved to: {filepath}")
            
            # Try to import PyMuPDF for PDF processing
            try:
                import fitz
                logger.info("PyMuPDF imported successfully")
            except ImportError:
                # Fallback - just store file info without text extraction
                logger.warning("PyMuPDF not available - storing file without text extraction")
                document_info = {
                    'filename': filename,
                    'upload_time': time.time(),
                    'status': 'uploaded_no_text_extraction',
                    'text': 'PDF text extraction not available',
                    'chunks': ['PDF uploaded but text extraction failed - please install PyMuPDF']
                }
                documents[filename] = document_info
                os.remove(filepath)
                return jsonify({
                    "status": "success",
                    "message": f"File uploaded but text extraction failed. Install PyMuPDF for full functionality.",
                    "filename": filename,
                    "chunks_created": 1
                })
            
            # Extract text from PDF
            doc = fitz.open(filepath)
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += text + "\n"
            doc.close()
            
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            
            # Clean up temp file
            os.remove(filepath)
            
            if not full_text.strip():
                return jsonify({"error": "Could not extract text from PDF - file might be scanned or corrupted"}), 400
            
        except Exception as pdf_error:
            logger.error(f"PDF processing error: {str(pdf_error)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Could not process PDF: {str(pdf_error)}"}), 500
        
        # Step 5: Create text chunks for better searching
        try:
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
            
            # Create overlapping chunks of sentences
            chunks = []
            chunk_size = 3  # sentences per chunk
            overlap = 1     # overlapping sentences
            
            for i in range(0, len(sentences), chunk_size - overlap):
                chunk = " ".join(sentences[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())
            
            # If no good chunks, use paragraphs
            if not chunks:
                paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p) > 20]
                chunks = paragraphs[:20]  # Limit to 20 paragraphs
            
            # If still no chunks, use the full text
            if not chunks:
                chunks = [full_text]
            
            logger.info(f"Created {len(chunks)} text chunks")
            
        except Exception as chunk_error:
            logger.error(f"Text chunking error: {str(chunk_error)}")
            chunks = [full_text]  # Fallback to full text
        
        # Step 6: Store document with text and chunks
        try:
            document_info = {
                'filename': filename,
                'upload_time': time.time(),
                'status': 'processed_successfully',
                'text': full_text,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'character_count': len(full_text)
            }
            
            documents[filename] = document_info
            logger.info(f"Document stored successfully: {filename} with {len(chunks)} chunks")
            
            return jsonify({
                "status": "success",
                "message": f"PDF processed successfully! Extracted text and created {len(chunks)} searchable chunks.",
                "filename": filename,
                "chunks_created": len(chunks),
                "characters_extracted": len(full_text)
            })
            
        except Exception as storage_error:
            logger.error(f"Storage failed: {str(storage_error)}")
            return jsonify({"error": f"Could not store document: {str(storage_error)}"}), 500
            
    except Exception as e:
        logger.error("=== UPLOAD FAILED ===")
        logger.error(f"Error: {str(e)}")
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
        
        # Search through documents
        if documents:
            # Find relevant chunks
            relevant_chunks = []
            question_words = set(question.lower().split())
            
            for filename, doc_info in documents.items():
                if 'chunks' in doc_info and doc_info['chunks']:
                    for i, chunk in enumerate(doc_info['chunks']):
                        # Simple keyword matching
                        chunk_words = set(chunk.lower().split())
                        common_words = question_words & chunk_words
                        
                        if len(common_words) > 0:
                            # Calculate relevance score
                            relevance = len(common_words) / len(question_words)
                            relevant_chunks.append({
                                'text': chunk,
                                'relevance': relevance,
                                'source': filename,
                                'chunk_id': i
                            })
            
            # Sort by relevance and take top results
            relevant_chunks.sort(key=lambda x: x['relevance'], reverse=True)
            top_chunks = relevant_chunks[:3]  # Top 3 most relevant chunks
            
            if top_chunks:
                # Generate response based on relevant content
                response_parts = [f'Based on the document "{top_chunks[0]["source"]}", here\'s what I found:\n']
                
                for i, chunk_info in enumerate(top_chunks, 1):
                    response_parts.append(f"**Relevant Section {i}:**")
                    response_parts.append(chunk_info['text'])
                    response_parts.append("")  # Empty line
                
                response_parts.append(f"This information comes from {len(set(c['source'] for c in top_chunks))} document(s) and addresses your question about: {question}")
                
                response_text = "\n".join(response_parts)
                
            else:
                # No relevant content found
                doc_names = list(documents.keys())
                response_text = f"""I couldn't find specific information about "{question}" in the uploaded documents.

The uploaded documents are: {', '.join(doc_names)}

Try asking more specific questions about:
- Scheme names mentioned in the documents
- Eligibility criteria
- Application procedures
- Required documents
- Contact information

Or try rephrasing your question using different keywords."""
        
        else:
            # No documents uploaded
            response_text = f"""I don't have any documents uploaded yet to answer your question: "{question}"

Please upload PDF documents related to Rajasthan government schemes first, then I'll be able to search through them and provide specific information.

Once you upload documents, you can ask about:
- Specific government schemes
- Eligibility requirements
- Application processes
- Required documents
- Contact details"""
        
        # Format HTML response
        try:
            # Simple HTML formatting
            html_lines = []
            for line in response_text.split('\n'):
                if line.strip():
                    if line.startswith('**') and line.endswith(':**'):
                        # Bold headers
                        clean_line = line.replace('**', '').replace(':', '')
                        html_lines.append(f'<p><strong>{clean_line}:</strong></p>')
                    else:
                        html_lines.append(f'<p>{line}</p>')
            
            html_response = f'<div class="ai-response">{"".join(html_lines)}</div>'
            
        except Exception as html_error:
            logger.warning(f"HTML formatting failed: {str(html_error)}")
            html_response = f'<div class="ai-response"><p>{response_text}</p></div>'
        
        return jsonify({
            "response": {
                "raw": response_text,
                "html": html_response,
                "text": response_text
            },
            "documents_searched": len(documents),
            "relevant_chunks_found": len(top_chunks) if 'top_chunks' in locals() else 0
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