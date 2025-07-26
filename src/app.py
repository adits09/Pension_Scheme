import os
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)
load_dotenv()

llm = None
conversation = None
collection = None
sentence_model = None

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

@app.route("/")
def health():
    return "Backend is running successfully!"

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    global collection
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_uploads", filename)
    os.makedirs("temp_uploads", exist_ok=True)
    file.save(filepath)

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
    return jsonify({
        "status": "success",
        "chunks_created": len(chunks)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation, collection
    data = request.get_json()
    question = data.get("message", "")
    doc_count = collection.count()
    results = collection.query(
        query_texts=[question],
        n_results=min(5, doc_count),
        include=["documents"]
    )
    context = "\n".join(results.get('documents', [[]])[0])
    prompt = f"""
You are a helpful assistant for Rajasthan government schemes.

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

def initialize_components():
    global llm, conversation, collection, sentence_model
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
