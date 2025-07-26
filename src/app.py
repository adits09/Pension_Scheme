import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import ssl
import re
from bs4 import BeautifulSoup
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app, origins=["*"], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['*'])
load_dotenv()

llm = None
conversation = None
collection = None
sentence_model = None

def simple_sent_tokenize(text):
    abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd)\.'
    text = re.sub(abbreviations, lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'])', text)
    return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]

def format_ai_response(response_text):
    soup = BeautifulSoup('<div class="ai-response"></div>', 'html.parser')
    container = soup.find('div')
    sections = response_text.split('\n\n')
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if ':' in section and len(section) < 100 and not section.startswith('*'):
            header = soup.new_tag('h3')
            header.string = section
            container.append(header)
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
    structured = {'sections': [], 'lists': [], 'key_points': []}
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if text:
            structured['sections'].append(text)
    for ul in soup.find_all('ul'):
        list_items = [li.get_text(strip=True) for li in ul.find_all('li')]
        structured['lists'].append(list_items)
    for strong in soup.find_all('strong'):
        structured['key_points'].append(strong.get_text(strip=True))
    return structured

def initialize_components():
    global llm, conversation, collection, sentence_model
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] GEMINI_API_KEY not found!")
            return False
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        memory = ConversationBufferMemory()
        conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        embedded_fxn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        try:
            chroma_client.delete_collection("pdf_chunks")
        except:
            pass
        collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedded_fxn)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize components: {e}")
        return False

@app.route("/")
def home():
    return "Backend is running successfully!"

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    if not request.files:
        return jsonify({"error": "No files provided"}), 400
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_uploads", filename)
    os.makedirs("temp_uploads", exist_ok=True)
    file.save(filepath)
    doc = fitz.open(filepath)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    sentences = simple_sent_tokenize(full_text)
    chunk_size, overlap = 5, 2
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size - overlap)]
    for idx, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"user_{filename}_{idx}"], metadatas=[{"source": filename}])
    os.remove(filepath)
    return jsonify({
        "message": "PDF uploaded and processed successfully!",
        "filename": filename,
        "chunks_created": len(chunks),
        "status": "success"
    }), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    question = data.get('message', '').strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400
    if collection is None or llm is None:
        return jsonify({"error": "System components not initialized"}), 500
    results = collection.query(query_texts=[question], n_results=3, include=["documents"])
    relevant_docs = results.get('documents', [[]])[0] if results.get('documents') else []
    context = "\n\n".join(relevant_docs)
    prompt = f"Answer this professionally based on Rajasthan govt schemes:\n\n{context}\n\nQ: {question}"
    raw_response = conversation.predict(input=prompt)
    formatted = format_ai_response(raw_response)
    return jsonify({"response": formatted, "question": question})

if __name__ == '__main__':
    print("[INFO] Starting SevaSaathi Server")
    initialize_components()
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
