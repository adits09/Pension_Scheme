import os
import logging
import traceback
import time
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import fitz
import re
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)
load_dotenv()

# Global variables
llm = None
conversation = None
collection = None
sentence_model = None
chunks_storage = {}  # Temporary in-memory storage

def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def format_ai_response(response_text):
    try:
        soup = BeautifulSoup('<div class="ai-response"></div>', 'html.parser')
        container = soup.find('div')
        sections = response_text.split('\n\n')
        for section in sections:
            section = section.strip()
            if not section:
                continue
            p = soup.new_tag('p')
            p.string = section
            container.append(p)
        return {
            'html': str(soup),
            'text': soup.get_text(' ', strip=True),
            'structured': {'sections': sections}
        }
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return {
            'html': f'<div class="ai-response"><p>{response_text}</p></div>',
            'text': response_text,
            'structured': {'sections': [response_text]}
        }

@app.route("/")
def health():
    return jsonify({"status": "Backend is running successfully!"})

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        status = {
            "status": "healthy",
            "llm_initialized": llm is not None,
            "document_count": len(chunks_storage)
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        global chunks_storage
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        
        # Create temp directory
        temp_dir = "/tmp" if os.path.exists("/tmp") else "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        
        # Save and process file
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")

        # Extract text from PDF
        try:
            doc = fitz.open(filepath)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as pdf_error:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Could not process PDF: {str(pdf_error)}"}), 400
        
        if not full_text.strip():
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Create chunks
        sentences = simple_sent_tokenize(full_text)
        chunks = []
        chunk_size = 5
        overlap = 2
        
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_text = " ".join(sentences[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append(chunk_text)

        # Store chunks in memory
        if chunks:
            chunks_storage[filename] = {
                'chunks': chunks,
                'upload_time': time.time(),
                'full_text': full_text
            }
            logger.info(f"Stored {len(chunks)} chunks for {filename}")

        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
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
        global chunks_storage, llm, conversation
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        question = data.get("message", "").strip()
        if not question:
            return jsonify({"error": "Empty message"}), 400

        # Get context from stored chunks
        context = ""
        if chunks_storage:
            # Simple keyword matching for context
            question_lower = question.lower()
            relevant_chunks = []
            
            for filename, file_data in chunks_storage.items():
                for chunk in file_data['chunks']:
                    # Simple relevance check
                    chunk_lower = chunk.lower()
                    common_words = set(question_lower.split()) & set(chunk_lower.split())
                    if len(common_words) > 0:
                        relevant_chunks.append(chunk)
            
            context = "\n".join(relevant_chunks[:5])  # Limit to 5 most relevant chunks
        
        # Generate response
        if llm and conversation:
            try:
                if context:
                    prompt = f"""
You are a helpful assistant for Rajasthan government schemes and documents.

Context from uploaded documents:
{context}

User Question: {question}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain relevant information, let the user know and provide general guidance if possible.
"""
                else:
                    prompt = f"""
You are a helpful assistant for Rajasthan government schemes.

User Question: {question}

I don't have any specific documents uploaded yet. Please provide general guidance about Rajasthan government schemes if possible, or ask the user to upload relevant documents.
"""
                
                raw_response = conversation.predict(input=prompt)
                formatted = format_ai_response(raw_response)
                
                return jsonify({
                    "response": {
                        "raw": raw_response,
                        "html": formatted["html"],
                        "text": formatted["text"]
                    },
                    "context_found": len(context.split('\n')) if context else 0
                })
                
            except Exception as e:
                logger.error(f"LLM error: {str(e)}")
                # Fallback response
                fallback_response = "I'm having trouble generating a response right now. Please try again or upload a PDF document for me to help with specific questions."
                formatted = format_ai_response(fallback_response)
                return jsonify({
                    "response": {
                        "raw": fallback_response,
                        "html": formatted["html"],
                        "text": fallback_response
                    }
                })
        else:
            # No LLM available
            fallback_response = "The AI assistant is not available right now. Please try again later."
            formatted = format_ai_response(fallback_response)
            return jsonify({
                "response": {
                    "raw": fallback_response,
                    "html": formatted["html"],
                    "text": fallback_response
                }
            })
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

def initialize_components():
    global llm, conversation
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found - AI features will be limited")
            return
            
        logger.info("Initializing LLM...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferMemory
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
        logger.info("LLM initialized successfully!")
        
    except Exception as e:
        logger.error(f"LLM initialization error: {str(e)}")
        llm = None
        conversation = None

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

# Initialize components safely
logger.info("Starting initialization...")
try:
    initialize_components()
    logger.info("Initialization completed")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    logger.info("Server will start with limited functionality")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)