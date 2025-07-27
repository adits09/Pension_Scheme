import os
import re
import fitz
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# ---------------- Flask App Config ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
CORS(app, resources={r"/*": {"origins": "*"}})
load_dotenv()

# ---------------- Globals ----------------
llm = None
conversation = None
collection = None
sentence_model = None
chroma_client = None
embedder = None

# ---------------- Utility Functions ----------------
def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def format_ai_response(response_text):
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

# ---------------- Routes ----------------
@app.route("/")
def health():
    return "Backend is running successfully!"

# ✅ Upload PDF and reinitialize collection
@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    global collection, chroma_client, embedder

    # ✅ Delete old collection & recreate
    chroma_client.delete_collection(name="pdf_chunks")
    collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedder)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_uploads", filename)
    os.makedirs("temp_uploads", exist_ok=True)
    file.save(filepath)

    try:
        doc = fitz.open(filepath)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()

        sentences = simple_sent_tokenize(full_text)
        chunks = []
        chunk_size = 5
        overlap = 2
        for i in range(0, len(sentences), chunk_size - overlap):
            chunks.append(" ".join(sentences[i:i + chunk_size]))

        for idx, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{idx}"],
                metadatas=[{"source": filename}]
            )

        os.remove(filepath)
        print(f"✅ Uploaded {len(chunks)} chunks for {filename}")
        return jsonify({
            "status": "success",
            "chunks_created": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Chat Endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation, collection
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    doc_count = collection.count()
    if doc_count == 0:
        return jsonify({"error": "No PDF uploaded yet. Please upload a file first."}), 400

    try:
        # Retrieve top documents
        results = collection.query(
            query_texts=[question],
            n_results=min(5, doc_count),
            include=["documents"]
        )
        retrieved_docs = results.get('documents', [[]])[0]
        context = "\n".join(retrieved_docs)

        prompt = f"""
You are a helpful assistant for Rajasthan government schemes.
Use the context to answer the user's question.

Context:
{context}

User Question: {question}
"""
        raw_response = conversation.predict(input=prompt)
        formatted = format_ai_response(raw_response)

        return jsonify({
            "response": {
                "raw": raw_response,
                "html": formatted["html"],
                "text": formatted["text"]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Enable CORS for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ---------------- Initialize Components ----------------
def initialize_components():
    global llm, conversation, collection, sentence_model, chroma_client, embedder
    api_key = os.getenv("GEMINI_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    sentence_model = SentenceTransformer('all-mpnet-base-v2')

    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedder)

initialize_components()

# ---------------- Run App ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
