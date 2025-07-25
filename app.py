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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
CORS(app, origins=["*"], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['*'])
load_dotenv()

llm = None
conversation = None
collection = None
sentence_model = None
default_pdf_loaded = False

def simple_sent_tokenize(text):
    abbreviations = r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd)\.'
    text = re.sub(abbreviations, lambda m: m.group(0).replace('.', '<DOT>'), text, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
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

def is_inappropriate_request(question):
    inappropriate_patterns = [
        r'\b(write|create|compose|make|generate).*(poem|story|song|lyrics|haiku|limerick|acrostic|riddle)\b',
        r'\b(poem|poetry|story|tale|song|lyrics|haiku|limerick|acrostic|riddle)\b',
        r'\b(code|program|script|algorithm|function|class|variable)\b',
        r'\b(python|java|javascript|html|css|sql|api)\b',
        r'\b(joke|funny|humor|entertainment|game|play)\b',
        r'\b(draw|paint|sketch|design|image|picture)\b',
        r'\b(weather|sports|movies|music|food|recipe)\b',
        r'\b(calculate|solve|equation|mathematics|algebra)\b(?!.*(pension|salary|income|benefit|amount))',
    ]
    question_lower = question.lower()
    for pattern in inappropriate_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    return False

def is_rajasthan_government_related(question):
    government_keywords = [
        'rajasthan', 'government', 'govt', 'scheme', 'yojana', 'pension', 'benefit',
        'service', 'department', 'ministry', 'office', 'application', 'form',
        'eligibility', 'document', 'certificate', 'registration', 'helpline',
        'support', 'assistance', 'welfare', 'social', 'public', 'citizen'
    ]
    rajasthan_specific = [
        'jaipur', 'jodhpur', 'udaipur', 'kota', 'ajmer', 'bikaner', 'alwar',
        'bharatpur', 'pali', 'sikar', 'tonk', 'barmer', 'churu', 'hanumangarh',
        'rajasthani', 'marwari', 'mewar', 'marwar', 'shekhawati'
    ]
    scheme_keywords = [
        'pension', 'widow', 'divorcee', 'abandoned', 'disability', 'old age',
        'scholarship', 'education', 'health', 'medical', 'insurance',
        'employment', 'job', 'skill', 'training', 'loan', 'subsidy',
        'ration', 'card', 'bpl', 'apl', 'income', 'certificate',
        'caste', 'domicile', 'birth', 'death', 'marriage'
    ]
    service_keywords = [
        'apply', 'application', 'process', 'procedure', 'how to',
        'eligibility', 'criteria', 'required', 'documents', 'fees',
        'status', 'track', 'check', 'update', 'renewal', 'contact',
        'helpline', 'office', 'address', 'timing', 'holiday'
    ]
    general_keywords = [
        'help', 'information', 'tell', 'about', 'what', 'how', 'where', 'when',
        'different', 'various', 'types', 'available', 'list'
    ]
    question_lower = question.lower()
    all_keywords = government_keywords + rajasthan_specific + scheme_keywords + service_keywords
    keyword_matches = sum(1 for keyword in all_keywords if keyword in question_lower)
    if keyword_matches >= 2:
        return True
    if 'rajasthan' in question_lower and any(word in question_lower for word in government_keywords + scheme_keywords):
        return True
    if any(word in question_lower for word in ['scheme', 'pension', 'benefit', 'yojana', 'government']) and \
       any(word in question_lower for word in service_keywords):
        return True
    government_question_patterns = [
        r'tell.*about.*scheme',
        r'what.*scheme',
        r'how.*apply',
        r'different.*scheme',
        r'various.*scheme',
        r'list.*scheme',
        r'types.*scheme',
        r'available.*scheme'
    ]
    for pattern in government_question_patterns:
        if re.search(pattern, question_lower):
            return True
    return keyword_matches > 0

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
        sentence_model = SentenceTransformer('all-mpnet-base-v2')
        print(">>Sentence transformer loaded")
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        embedded_fxn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        try:
            chroma_client.delete_collection("pdf_chunks")
            print(">>Cleared existing collection")
        except:
            print(">>No existing collection to clear")
        collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedded_fxn)
        print(">>ChromaDB initialized")
        return True
    except Exception as e:
        print(f">>Error initializing components: {e}")
        return False

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
        doc.close()
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
                ids=[f"user_{filename}_{idx}"],
                metadatas=[{"source": filename, "chunk_type": "user_uploaded"}]
            )
        os.remove(filepath)
        print(f">>Processed and stored {len(chunks)} chunks from {filename}")
        return jsonify({
            "message": "PDF uploaded and processed successfully!",
            "filename": filename,
            "chunks_created": len(chunks),
            "total_documents_in_system": collection.count(),
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
        if is_inappropriate_request(question):
            refusal = (
                "I'm SevaSaathi, your professional assistant for Rajasthan government services and schemes. "
                "I specialize in providing accurate information about government benefits, applications, eligibility criteria, "
                "and administrative procedures. How can I assist you with Rajasthan government services today?"
            )
            return jsonify({
                "response": {
                    "raw": refusal,
                    "html": f"<div class='message-content'><p>{refusal}</p></div>",
                    "text": refusal,
                    "structured": {"sections": [refusal]}
                },
                "question": question,
                "response_type": "scope_limitation"
            })
        if not is_rajasthan_government_related(question):
            gentle_redirect = (
                "I'm SevaSaathi, your dedicated assistant for Rajasthan government services. "
                "While I'd love to help with all topics, I specialize in providing information about "
                "Rajasthan government schemes, services, eligibility criteria, application processes, "
                "and administrative procedures. Could you please ask me something related to "
                "Rajasthan government services or schemes? I'm here to help!"
            )
            return jsonify({
                "response": {
                    "raw": gentle_redirect,
                    "html": f"<div class='message-content'><p>{gentle_redirect}</p></div>",
                    "text": gentle_redirect,
                    "structured": {"sections": [gentle_redirect]}
                },
                "question": question,
                "response_type": "topic_redirect"
            })
        if collection is None or llm is None:
            print(">>Components not initialized")
            return jsonify({"error": "System components not initialized"}), 500
        doc_count = collection.count()
        print(f">>Document count in collection: {doc_count}")
        n_results = max(1, min(5, doc_count)) if doc_count > 0 else 1
        print(f">>Query parameters - doc_count: {doc_count}, n_results: {n_results}")
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        relevant_docs = results.get('documents', [[]])[0] if results.get('documents') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        distances = results.get('distances', [[]])[0] if results.get('distances') else []
        system_prompt = """You are SevaSaathi, a professional and helpful assistant specializing in Rajasthan government services and schemes. 

Your personality:
- Professional yet warm and approachable
- Patient and understanding
- Knowledgeable about government procedures
- Always helpful and solution-oriented
- Conversational but focused on government topics

Your capabilities:
- Provide detailed information about Rajasthan government schemes and benefits
- Explain eligibility criteria and application processes
- Guide users through government procedures
- Answer questions about documents, forms, and requirements
- Provide contact information and office details when available

Your response style:
- Use a conversational, helpful tone
- Structure information clearly with headings and bullet points when appropriate
- Provide step-by-step guidance when explaining processes
- Always be encouraging and supportive
- If information is not available in your knowledge base, admit it honestly

Remember: You are here to serve the citizens of Rajasthan with accurate, helpful information about government services."""
        if relevant_docs:
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
            enhanced_prompt = f"""{system_prompt}

Based on the following context from official documents, please answer the user's question in a helpful and conversational manner:

Context from documents:
{context}

User Question: {question}

Please provide a comprehensive, well-structured answer. If the exact information isn't in the context, provide what you can and suggest how the user might get more specific information."""
        else:
            enhanced_prompt = f"""{system_prompt}

User Question: {question}

I don't have specific document information for this query, but please provide a helpful response based on general knowledge about Rajasthan government services. If you cannot provide specific details, guide the user on how they might find this information through official channels."""
        print(">>Getting LLM response...")
        raw_response = conversation.predict(input=enhanced_prompt)
        print(f">>LLM response length: {len(raw_response)} characters")
        formatted_response = format_ai_response(raw_response)
        source_info = []
        for i, (doc, meta, dist) in enumerate(zip(relevant_docs, metadatas, distances)):
            source_info.append({
                "chunk_index": i,
                "source_file": meta.get('source', 'Unknown') if meta else 'Unknown',
                "chunk_type": meta.get('chunk_type', 'unknown') if meta else 'unknown',
                "similarity_score": round(float(1 - dist), 3) if dist is not None else 0,
                "preview": doc[:150] + "..." if len(doc) > 150 else doc
            })
        return jsonify({
            "response": {
                "raw": raw_response,
                "html": formatted_response['html'],
                "text": formatted_response['text'],
                "structured": formatted_response['structured'],
                "source_documents": source_info,
                "total_chunks_searched": doc_count,
                "chunks_used": len(relevant_docs),
                "has_default_knowledge": default_pdf_loaded
            },
            "question": question,
            "response_type": "government_info"
        })
    except Exception as e:
        print(f">>Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(">>Starting SevaSaathi - Rajasthan Government Assistant...")
    print(">>Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ', '.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        print(f"   {rule.rule:<30} [{methods}]")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(">>GEMINI_API_KEY not found")
    else:
        print(">>GEMINI_API_KEY found")
    try:
        components_initialized = initialize_components()
        if components_initialized:
            print(">>All components initialized successfully!")
        else:
            print(">>Some components failed to initialize - functionality may be limited")
    except Exception as e:
        print(f">>Component initialization failed: {e}")
        print(">>Server will still start for debugging...")
    print("\n" + "="*60)
    print(">>SEVASAATHI SERVER STARTING")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

