import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import ssl
import re
from bs4 import BeautifulSoup
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate 

# SSL setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def simple_sent_tokenize(text):
    """Simple sentence tokenizer that doesn't require NLTK downloads"""
    # Handle common abbreviations
    abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd)\.'
    text = re.sub(abbreviations, lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    
    # Split on sentence endings followed by whitespace and capital letter or quote
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    
    # Restore abbreviations and clean up
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
    
    return sentences

def format_ai_response(response_text):
    """Format AI response using BeautifulSoup for better structure"""
    soup = BeautifulSoup('<div class="ai-response"></div>', 'html.parser')
    container = soup.find('div')
    
    # Split into sections
    sections = response_text.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if it's a title/header
        if ':' in section and len(section) < 100 and not section.startswith('*'):
            header = soup.new_tag('h3')
            header.string = section
            container.append(header)
        
        # Check if it's a list section
        elif '* ' in section:
            lines = section.split('\n')
            current_ul = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('* '):
                    if current_ul is None:
                        current_ul = soup.new_tag('ul')
                        container.append(current_ul)
                    
                    li = soup.new_tag('li')
                    li_content = line[2:]
                    if '**' in li_content:
                        li_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', li_content)
                        li.append(BeautifulSoup(li_content, 'html.parser'))
                    else:
                        li.string = li_content
                    current_ul.append(li)
                else:
                    if current_ul is not None:
                        current_ul = None
                    if line:
                        p = soup.new_tag('p')
                        if '**' in line:
                            line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                            p.append(BeautifulSoup(line, 'html.parser'))
                        else:
                            p.string = line
                        container.append(p)
        
        # Regular paragraph
        else:
            p = soup.new_tag('p')
            if '**' in section:
                section = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', section)
                p.append(BeautifulSoup(section, 'html.parser'))
            else:
                p.string = section
            container.append(p)
    
    return {
        'html': str(soup),
        'text': soup.get_text(' ', strip=True),
        'structured': extract_structured_data(soup)
    }

def extract_structured_data(soup):
    """Extract structured data from formatted response"""
    structured = {
        'sections': [],
        'lists': [],
        'key_points': []
    }
    
    # Extract all paragraphs
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text(strip=True)
        if text:
            structured['sections'].append(text)
    
    # Extract all lists
    lists = soup.find_all('ul')
    for ul in lists:
        list_items = [li.get_text(strip=True) for li in ul.find_all('li')]
        structured['lists'].append(list_items)
    
    # Extract bold/important points
    strong_tags = soup.find_all('strong')
    for strong in strong_tags:
        structured['key_points'].append(strong.get_text(strip=True))
    
    return structured

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

CORS(app, origins=["*"], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['*'])

load_dotenv()

# Global variables for components
llm = None
conversation = None
collection = None
sentence_model = None

def is_blocked_action(question):
    blocked_phrases = [
        "write a poem", "poem", "compose a poem", "make a poem", "write a story", "story",
        "joke", "draw", "paint", "song", "lyrics", "haiku", "riddle", "acrostic", "limerick"
    ]

    q = question.lower()
    return any(phrase in q for phrase in blocked_phrases)

def is_rajasthan_govt_question(question):
    keywords = [
        "rajasthan", "scheme", "government", "pension", "calendar", "support", "service", "benefit",
        "yatra", "festival", "application", "document", "state", "department", "widow", "divorced",
        "abandoned", "citizen", "eligibility", "income", "apply", "registration", "official", "helpdesk"
    ]
    q = question.lower()
    return any(word in q for word in keywords)



def initialize_components():
    global llm, conversation, collection, sentence_model
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(">>GEMINI_API_KEY not found!")
            return False
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  
            temperature=0.3,
            google_api_key=api_key
        )
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
        print(">>Gemini LLM initialized")
        
        # Initialize sentence transformer
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        print(">>Sentence transformer loaded")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        embedded_fxn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        
        # Delete old pdfs 
        try:
            chroma_client.delete_collection("pdf_chunks")
            print(">>Cleared existing collection")
        except:
            print(">>No existing collection to clear")
        
        # New DB is formed to store pdf chunks
        collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedded_fxn)
        print(">>ChromaDB initialized")
        
        print(">>All components initialized.")
        return True
        
    except Exception as e:
        print(f">>Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    print("Home route accessed")
    return jsonify({
        "message": "Flask backend is running!",
        "status": "healthy",
        "endpoints": ["/api/health", "/api/upload-pdf", "/api/chat"]
    })

@app.route('/api/health', methods=['GET'])
def health():
    print("Health check accessed")
    global llm, collection
    
    status = {
        "status": "healthy",
        "llm_initialized": llm is not None,
        "collection_initialized": collection is not None,
        "document_count": collection.count() if collection else 0
    }
    
    return jsonify(status)

@app.before_request
def handle_preflight():
    print(f"Request: {request.method} {request.path}")
    if request.method == "OPTIONS":
        print("Handling CORS preflight")
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    print(f"\n{'='*50}")
    print(f"PDF UPLOAD REQUEST: {request.method} {request.path}")
    print(f"{'='*50}")
    
    if request.method == 'OPTIONS':
        print("Handling CORS preflight request")
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization")
        response.headers.add('Access-Control-Allow-Methods', "POST, OPTIONS")
        return response, 200
    
    try:
        if not request.files:
            print(">>No files in request")
            return jsonify({"error": "No files provided"}), 400
        
        print(">>Upload route is working! Files detected.")
        file = request.files.get('file')
        if not file or file.filename == '':
            print(">>No file selected")
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join("temp_uploads", filename)
        os.makedirs("temp_uploads", exist_ok=True)
        file.save(filepath)

        doc = fitz.open(filepath)
        full_text = "\n".join(page.get_text() for page in doc)

        sentences = simple_sent_tokenize(full_text)
        
        chunk_size = 5
        overlap = 2
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = sentences[i:i + chunk_size]
            chunks.append(" ".join(chunk))

        for idx, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{idx}"]
            )
        
        print(f">>Processed and stored {len(chunks)} chunks from {filename}")
        return jsonify({
            "message": "PDF uploaded and processed successfully!",
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "success"
        }), 200

    except Exception as e:
        print(f">>Error in upload route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    print(f"\n{'='*30} CHAT REQUEST {'='*30}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        question = data.get('message', '').strip()
        print(f">>Question: {question}")

        if not question:
            return jsonify({"error": "No message provided"}), 400

        # 1. Block creative/poem/story requests
        if is_blocked_action(question):
            refusal = (
                "I am SevaSaathi, your Rajasthan government assistant. "
                "I can only provide factual information about Rajasthan government schemes, services, events, or support—not poems, stories, or creative writing."
            )
            return jsonify({
                "response": {
                    "raw": refusal,
                    "html": f"<div class='message-content'>{refusal}</div>",
                    "text": refusal,
                    "structured": {}
                },
                "question": question
            })

        # 2. Block non-Rajasthan questions
        if not is_rajasthan_govt_question(question):
            refusal = (
                "I am SevaSaathi, your Rajasthan government assistant. "
                "I can only answer queries related to Rajasthan government schemes, services, events, or support."
            )
            return jsonify({
                "response": {
                    "raw": refusal,
                    "html": f"<div class='message-content'>{refusal}</div>",
                    "text": refusal,
                    "structured": {}
                },
                "question": question
            })

        # 3. Ensure components are initialized
        if collection is None or llm is None:
            print(">>Components not initialized")
            return jsonify({"error": "Components not initialized"}), 500

        doc_count = collection.count()
        print(f">>Document count in collection: {doc_count}")

        if doc_count == 0:
            print(">>No documents found in collection")
            return jsonify({
                "response": "I don't have any documents to search through. Please upload a PDF first.",
                "debug_info": {
                    "document_count": doc_count,
                    "collection_exists": collection is not None
                }
            })

        # 4. Query ChromaDB for relevant chunks
        try:
            print(">>Querying ChromaDB...")
            results = collection.query(
                query_texts=[question],
                n_results=min(3, doc_count),
                include=["documents", "metadatas", "distances"]
            )
            relevant_docs = results.get('documents', [[]])[0] if results.get('documents') else []
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            distances = results.get('distances', [[]])[0] if results.get('distances') else []

            print(f">>Found {len(relevant_docs)} relevant chunks")

            if not relevant_docs:
                return jsonify({
                    "response": "I couldn't find any relevant information in the uploaded documents for your question.",
                    "debug_info": {
                        "query_executed": True,
                        "results_returned": bool(results),
                        "document_count": doc_count
                    }
                })
            
            context = "\n\n".join([f"Context {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
            print(f">>Context length: {len(context)} characters")

            enhanced_prompt = f"""Based on the following context from uploaded documents, please answer the user's question. If the answer cannot be found in the context, please say so clearly.

Context from documents:
{context}

User Question: {question}
Please provide a helpful answer based on the context above:"""

            # Get response from LLM
            # Get response from LLM
            # Get response from LLM
            print(">>Getting LLM response...")
            raw_response = conversation.predict(input=enhanced_prompt)
            print(f">>LLM response length: {len(raw_response)} characters")

            # Format with BeautifulSoup
            formatted_response = format_ai_response(raw_response)

            # Prepare source information
            source_info = []
            for i, (doc, meta, dist) in enumerate(zip(relevant_docs, metadatas, distances)):
                source_info.append({
                    "chunk_index": i,
                    "source_file": meta.get('source', 'Unknown') if meta else 'Unknown',
                    "similarity_score": round(float(1 - dist), 3) if dist is not None else 0,
                    "preview": doc[:150] + "..." if len(doc) > 150 else doc
                })

            # Return in the same structure your frontend expects
            return jsonify({
                "response": {
                    "raw": raw_response,
                    "html": formatted_response['html'],
                    "text": formatted_response['text'],
                    "structured": formatted_response['structured'],
                    "source_documents": source_info,
                    "total_chunks_searched": doc_count,
                    "chunks_used": len(relevant_docs)
                },
                "question": question
            })

        except Exception as query_error:
            print(f">>Error querying ChromaDB: {query_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Database query failed: {str(query_error)}"}), 500

    except Exception as e:
        print(f">>Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(">>Starting Flask backend...")
    
    print(">>Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ', '.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f"   {rule.rule:<30} [{methods}]")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(">>GEMINI_API_KEY not found")
    else:
        print(">>GEMINI_API_KEY found")
   
    # Initialize components
    try:
        components_initialized = initialize_components()
        if components_initialized:
            print(">>All components initialized successfully!")
        else:
            print(">>Some components failed to initialize - upload may not work fully")
    except Exception as e:
        print(f">>Component initialization failed: {e}")
        print(">>Server will still start for debugging...")
    
    print("\n" + "="*60)
    print(">>FLASK SERVER STARTING")
    print(">>Server URL: http://localhost:5000")
    print(">>Test routes:")
    print("   - GET  http://localhost:5000/api/health")
    print("   - POST http://localhost:5000/api/chat")
    print("   - POST http://localhost:5000/api/upload-pdf")
    print("="*60 + "\n")

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f">>Failed to start server: {e}")
        import traceback
        traceback.print_exc()

