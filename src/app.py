import os
import logging
import traceback
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import fitz
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app, origins=["https://your-vercel-app.vercel.app", "http://localhost:3000"])  # Replace with your actual Vercel URL
load_dotenv()

llm = None
conversation = None
collection = None
sentence_model = None

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
    return jsonify({"status": "Backend is running successfully!", "timestamp": str(os.environ.get('RENDER_SERVICE_ID', 'local'))})

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        global collection, llm
        status = {
            "status": "healthy",
            "collection_initialized": collection is not None,
            "llm_initialized": llm is not None,
            "document_count": collection.count() if collection else 0
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        global collection
        
        if collection is None:
            return jsonify({"error": "System not initialized properly"}), 500
            
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        
        # Create temp directory if it doesn't exist
        temp_dir = "/tmp" if os.path.exists("/tmp") else "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        
        # Save and process file
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")

        # Extract text from PDF
        doc = fitz.open(filepath)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
        
        if not full_text.strip():
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

        # Add to collection
        if chunks:
            try:
                collection.add(
                    documents=chunks,
                    ids=[f"{filename}_{idx}" for idx in range(len(chunks))],
                    metadatas=[{"source": filename} for _ in chunks]
                )
                logger.info(f"Added {len(chunks)} chunks to collection")
            except Exception as e:
                logger.error(f"Error adding to collection: {str(e)}")
                os.remove(filepath)
                return jsonify({"error": f"Database error: {str(e)}"}), 500

        # Clean up
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
        global conversation, collection
        
        if conversation is None or collection is None:
            return jsonify({"error": "System not initialized properly"}), 500
            
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
            
        question = data.get("message", "").strip()
        if not question:
            return jsonify({"error": "Empty message"}), 400

        # Query the collection
        try:
            doc_count = collection.count()
            logger.info(f"Collection has {doc_count} documents")
            
            if doc_count == 0:
                return jsonify({
                    "response": {
                        "raw": "I don't have any documents uploaded yet. Please upload a PDF first to ask questions about it.",
                        "html": "<div class='ai-response'><p>I don't have any documents uploaded yet. Please upload a PDF first to ask questions about it.</p></div>",
                        "text": "I don't have any documents uploaded yet. Please upload a PDF first to ask questions about it."
                    }
                })
            
            results = collection.query(
                query_texts=[question],
                n_results=min(5, doc_count),
                include=["documents"]
            )
            
            context = "\n".join(results.get('documents', [[]])[0]) if results.get('documents') else ""
            
        except Exception as e:
            logger.error(f"Collection query error: {str(e)}")
            context = ""

        # Generate response
        try:
            prompt = f"""
You are a helpful assistant for Rajasthan government schemes and documents.

Context from uploaded documents:
{context}

User Question: {question}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain relevant information, let the user know and provide general guidance if possible.
"""
            raw_response = conversation.predict(input=prompt)
            formatted = format_ai_response(raw_response)
            
            return jsonify({
                "response": {
                    "raw": raw_response,
                    "html": formatted["html"],
                    "text": formatted["text"]
                },
                "context_found": len(results.get('documents', [[]])[0]) if results.get('documents') else 0
            })
            
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return jsonify({"error": f"AI response error: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

def initialize_components():
    global llm, conversation, collection, sentence_model
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        logger.info("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        logger.info("Initializing conversation chain...")
        conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
        
        logger.info("Loading sentence transformer...")
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        logger.info("Initializing ChromaDB...")
        # Use /tmp for Render deployment
        chroma_path = "/tmp/chroma_store" if os.path.exists("/tmp") else "./chroma_store"
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        embedder = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedder)
        
        logger.info("All components initialized successfully!")
        logger.info(f"Collection has {collection.count()} existing documents")
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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

# Initialize components
try:
    initialize_components()
except Exception as e:
    logger.error(f"Failed to initialize: {str(e)}")
    # Don't exit, let the app start and show errors in health check

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)