import os
import logging
import traceback
import time
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Enhanced CORS configuration for Vercel + Render
CORS(app, 
     origins=["https://*.vercel.app", "http://localhost:3000", "https://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Initialize AI components
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
embedding_model = None
genai_model = None

# Initialize Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        genai_model = None
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Initialize embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")

# Simple in-memory storage with embeddings
documents = {}
document_embeddings = {}

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route("/")
def health():
    return jsonify({
        "status": "AI-powered backend running successfully on Render!", 
        "timestamp": str(time.time()),
        "documents_count": len(documents),
        "ai_enabled": genai_model is not None,
        "embeddings_enabled": embedding_model is not None,
        "deployment": "render",
        "cors_enabled": True
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        return jsonify({
            "status": "healthy",
            "documents_stored": len(documents),
            "timestamp": time.time(),
            "message": "Gemini AI-powered backend ready for Vercel frontend",
            "ai_features": {
                "gemini_available": genai_model is not None,
                "embeddings_available": embedding_model is not None,
                "api_key_configured": GEMINI_API_KEY is not None
            },
            "deployment_info": {
                "backend": "render",
                "expected_frontend": "vercel",
                "cors_configured": True
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def create_embeddings(text_chunks):
    """Create embeddings for text chunks"""
    if not embedding_model:
        return None
    
    try:
        embeddings = embedding_model.encode(text_chunks)
        return embeddings
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        return None

def find_relevant_chunks_with_embeddings(question, top_k=3):
    """Find relevant chunks using semantic similarity"""
    if not embedding_model or not document_embeddings:
        return find_relevant_chunks_keyword(question, top_k)
    
    try:
        # Create embedding for the question
        question_embedding = embedding_model.encode([question])
        
        relevant_chunks = []
        
        for filename, doc_info in documents.items():
            if filename not in document_embeddings:
                continue
                
            chunks = doc_info.get('chunks', [])
            chunk_embeddings = document_embeddings[filename]
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            
            # Get top chunks for this document
            for i, similarity in enumerate(similarities):
                if similarity > 0.2:  # Lower threshold for better recall
                    relevant_chunks.append({
                        'text': chunks[i],
                        'relevance': float(similarity),
                        'source': filename,
                        'chunk_id': i
                    })
        
        # Sort by relevance and return top k
        relevant_chunks.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_chunks[:top_k]
        
    except Exception as e:
        logger.error(f"Embedding search failed: {e}")
        return find_relevant_chunks_keyword(question, top_k)

def find_relevant_chunks_keyword(question, top_k=3):
    """Fallback keyword-based search"""
    relevant_chunks = []
    question_words = set(question.lower().split())
    
    for filename, doc_info in documents.items():
        if 'chunks' in doc_info and doc_info['chunks']:
            for i, chunk in enumerate(doc_info['chunks']):
                chunk_words = set(chunk.lower().split())
                common_words = question_words & chunk_words
                
                if len(common_words) > 0:
                    relevance = len(common_words) / len(question_words)
                    relevant_chunks.append({
                        'text': chunk,
                        'relevance': relevance,
                        'source': filename,
                        'chunk_id': i
                    })
    
    relevant_chunks.sort(key=lambda x: x['relevance'], reverse=True)
    return relevant_chunks[:top_k]

def generate_gemini_response(question, context_chunks):
    """Generate AI response using Google Gemini"""
    if not genai_model:
        return generate_template_response(question, context_chunks)
    
    try:
        # Prepare context
        context_text = "\n\n".join([f"Document: {chunk['source']}\nContent: {chunk['text']}" for chunk in context_chunks])
        
        # Create comprehensive prompt
        prompt = f"""You are SevaSaathi, an AI assistant specialized in helping users with Rajasthan government schemes and services. You are friendly, helpful, and knowledgeable about government processes.

CONTEXT FROM UPLOADED DOCUMENTS:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a helpful and accurate response based on the context above
- Be conversational and friendly in tone
- Include specific details like eligibility criteria, required documents, procedures when available
- If the context doesn't contain enough information, acknowledge this and suggest what the user can do
- Structure your response clearly with bullet points or sections when appropriate
- Use simple language that everyone can understand
- Always try to guide the user on next steps

Please provide a comprehensive response:"""

        # Generate response with Gemini
        response = genai_model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            logger.warning("Gemini returned empty response")
            return generate_template_response(question, context_chunks)
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return generate_template_response(question, context_chunks)

def generate_template_response(question, context_chunks):
    """Generate template response when AI is not available"""
    if not context_chunks:
        return f"""I don't have specific information about "{question}" in the uploaded documents.

Please try:
- Uploading relevant PDF documents about Rajasthan government schemes
- Asking more specific questions about scheme names, eligibility, or procedures
- Using different keywords related to your query

I'm here to help with information about government schemes once you provide the relevant documents."""

    response_parts = ["Based on the uploaded documents, here's what I found:\n"]
    
    for i, chunk in enumerate(context_chunks, 1):
        response_parts.append(f"**Information {i} (from {chunk['source']}):**")
        response_parts.append(chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text'])
        response_parts.append("")
    
    response_parts.append("This information comes from your uploaded documents. For more specific details, please ask follow-up questions.")
    
    return "\n".join(response_parts)

@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
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
            # Create temp directory - Render uses /tmp
            temp_dir = "/tmp"
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
                logger.warning("PyMuPDF not available - storing file without text extraction")
                document_info = {
                    'filename': filename,
                    'upload_time': time.time(),
                    'status': 'uploaded_no_text_extraction',
                    'text': 'PDF text extraction not available',
                    'chunks': ['PDF uploaded but text extraction failed - please install PyMuPDF']
                }
                documents[filename] = document_info
                if os.path.exists(filepath):
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
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if not full_text.strip():
                return jsonify({"error": "Could not extract text from PDF - file might be scanned or corrupted"}), 400
            
        except Exception as pdf_error:
            logger.error(f"PDF processing error: {str(pdf_error)}")
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Could not process PDF: {str(pdf_error)}"}), 500
        
        # Step 5: Create text chunks for better searching
        try:
            # Split text into meaningful chunks
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 15]
            
            # Create overlapping chunks of sentences
            chunks = []
            chunk_size = 4  # sentences per chunk
            overlap = 1     # overlapping sentences
            
            for i in range(0, len(sentences), chunk_size - overlap):
                chunk = " ".join(sentences[i:i + chunk_size])
                if chunk.strip() and len(chunk) > 50:  # Minimum chunk size
                    chunks.append(chunk.strip())
            
            # If no good chunks, use paragraphs
            if not chunks:
                paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p) > 30]
                chunks = paragraphs[:25]
            
            # If still no chunks, use the full text in smaller pieces
            if not chunks:
                chunk_size = 800
                overlap = 100
                for i in range(0, len(full_text), chunk_size - overlap):
                    chunk = full_text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
            
            logger.info(f"Created {len(chunks)} text chunks")
            
        except Exception as chunk_error:
            logger.error(f"Text chunking error: {str(chunk_error)}")
            chunks = [full_text[:1000]]  # Fallback with size limit
        
        # Step 6: Create embeddings for chunks
        embeddings = None
        if embedding_model and chunks:
            try:
                logger.info("Creating embeddings for text chunks...")
                embeddings = create_embeddings(chunks)
                if embeddings is not None:
                    document_embeddings[filename] = embeddings
                    logger.info(f"Created embeddings for {len(chunks)} chunks")
            except Exception as embed_error:
                logger.error(f"Embedding creation failed: {embed_error}")
        
        # Step 7: Store document with text and chunks
        try:
            document_info = {
                'filename': filename,
                'upload_time': time.time(),
                'status': 'processed_successfully',
                'text': full_text[:5000],  # Store only first 5000 chars to save memory
                'chunks': chunks,
                'chunk_count': len(chunks),
                'character_count': len(full_text),
                'has_embeddings': embeddings is not None
            }
            
            documents[filename] = document_info
            logger.info(f"Document stored successfully: {filename} with {len(chunks)} chunks")
            
            return jsonify({
                "status": "success",
                "message": f"PDF processed successfully! Created {len(chunks)} searchable chunks with {'semantic' if embeddings is not None else 'keyword'} search.",
                "filename": filename,
                "chunks_created": len(chunks),
                "characters_extracted": len(full_text),
                "ai_features_enabled": {
                    "embeddings": embeddings is not None,
                    "gemini": genai_model is not None
                }
            })
            
        except Exception as storage_error:
            logger.error(f"Storage failed: {str(storage_error)}")
            return jsonify({"error": f"Could not store document: {str(storage_error)}"}), 500
            
    except Exception as e:
        logger.error("=== UPLOAD FAILED ===")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
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
        
        # Find relevant chunks using embeddings or keywords
        if documents:
            relevant_chunks = find_relevant_chunks_with_embeddings(question, top_k=3)
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            # Generate AI response using Gemini
            response_text = generate_gemini_response(question, relevant_chunks)
            
        else:
            # No documents uploaded
            if genai_model:
                # Use Gemini for general responses when no docs are available
                try:
                    prompt = f"""You are SevaSaathi, an AI assistant for Rajasthan government schemes. A user asked: "{question}"

Since no documents are uploaded yet, provide a helpful response that:
1. Acknowledges their question
2. Explains that you need them to upload relevant PDF documents first
3. Suggests what types of documents would be helpful
4. Gives general guidance about Rajasthan government schemes if applicable

Be friendly and helpful."""
                    
                    response = genai_model.generate_content(prompt)
                    response_text = response.text if response and response.text else "Hello! Please upload PDF documents so I can help you with specific information about Rajasthan government schemes."
                except:
                    response_text = "Hello! I'm SevaSaathi, your AI assistant for Rajasthan government schemes. Please upload PDF documents so I can provide you with specific information."
            else:
                response_text = f"""Hello! I'm SevaSaathi, your AI assistant for Rajasthan government schemes.

I don't have any documents uploaded yet to answer your question: "{question}"

To get started:
1. Upload PDF documents about government schemes using the attachment button (📎)
2. Once uploaded, I'll analyze the content and provide detailed answers
3. Ask me about eligibility, application processes, required documents, etc.

I'm here to help once you provide the relevant documents!"""
        
        # Format HTML response for better display
        try:
            html_lines = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Handle different formatting
                if line.startswith('**') and line.endswith('**'):
                    # Bold text
                    clean_line = line.replace('**', '')
                    html_lines.append(f'<p><strong>{clean_line}</strong></p>')
                elif line.startswith('*') and line.endswith('*'):
                    # Italic text
                    clean_line = line.replace('*', '')
                    html_lines.append(f'<p><em>{clean_line}</em></p>')
                elif line.startswith('- ') or line.startswith('• '):
                    # Bullet points
                    clean_line = line[2:] if line.startswith('- ') else line[2:]
                    html_lines.append(f'<li>{clean_line}</li>')
                elif re.match(r'^\d+\.', line):
                    # Numbered lists
                    html_lines.append(f'<li>{line}</li>')
                else:
                    # Regular paragraphs
                    html_lines.append(f'<p>{line}</p>')
            
            # Wrap lists in appropriate tags
            html_content = '\n'.join(html_lines)
            html_content = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', html_content, flags=re.DOTALL)
            html_content = re.sub(r'</ul>\s*<ul>', '', html_content)
            
            html_response = f'<div class="ai-response">{html_content}</div>'
            
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
            "relevant_chunks_found": len(relevant_chunks) if 'relevant_chunks' in locals() else 0,
            "ai_powered": genai_model is not None,
            "processing_info": {
                "backend": "render",
                "ai_service": "gemini",
                "embeddings": embedding_model is not None
            }
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """Debug endpoint to see what documents are stored"""
    try:
        doc_info = {}
        for filename, doc_data in documents.items():
            doc_info[filename] = {
                'filename': doc_data.get('filename'),
                'upload_time': doc_data.get('upload_time'),
                'status': doc_data.get('status'),
                'chunk_count': doc_data.get('chunk_count', 0),
                'character_count': doc_data.get('character_count', 0),
                'has_embeddings': doc_data.get('has_embeddings', False)
            }
        
        return jsonify({
            "documents": doc_info,
            "count": len(documents),
            "ai_features": {
                "gemini_available": genai_model is not None,
                "embeddings_available": embedding_model is not None,
                "api_key_configured": GEMINI_API_KEY is not None
            },
            "deployment": {
                "backend": "render",
                "cors_enabled": True,
                "expected_frontend": "vercel"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers with CORS
@app.errorhandler(404)
def not_found(error):
    response = jsonify({"error": "Endpoint not found"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    response = jsonify({"error": "Internal server error"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 500

@app.errorhandler(413)
def too_large(error):
    response = jsonify({"error": "File too large (max 50MB)"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Gemini AI-powered backend on Render (port {port})")
    logger.info(f"Gemini API Key configured: {GEMINI_API_KEY is not None}")
    logger.info(f"Embedding model available: {embedding_model is not None}")
    logger.info("CORS configured for Vercel frontend")
    app.run(host="0.0.0.0", port=port, debug=False)