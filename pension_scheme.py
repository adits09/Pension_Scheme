'''

import fitz
import nltk
import re
import os
from nltk.tokenize import sent_tokenize
from nltk.data import find
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    exit(1)

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

try:
    model = SentenceTransformer('all-mpnet-base-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    embedded_fxn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

except Exception as e:
    print(f"Error initializing ChromaDB or embedding model: {e}")
    exit(1)

try:
    chroma_client.delete_collection("pdf_chunks")
except Exception as e:
    print("No existing collection to clear")

collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedded_fxn)

def extract_chunks_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return []
    
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text() + " "
        doc.close()

        pattern = re.compile(r'start(.*?)end', re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(all_text)
        
        chunks_list = []
        for match in matches:
            chunk = "start" + match + "end"
            chunks_list.append(chunk.strip())
        
        if not chunks_list:
            print(f"No start...end pattern found in {pdf_path}. Using sentence-based chunking.")
            sentences = sent_tokenize(all_text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) <= 500:
                    current_chunk += " " + sentence
                else:
                    if current_chunk.strip():
                        chunks_list.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk.strip():
                chunks_list.append(current_chunk.strip())
        
        return chunks_list
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []

def add_chunks_to_chroma(pdf_path):
    chunks_list = extract_chunks_from_pdf(pdf_path)
    if not chunks_list:
        return
    
    pdf_name = os.path.basename(pdf_path)
    
    try:
        for i, text in enumerate(chunks_list):
            title_match = re.search(r'(Chief Minister .*? Pension (Scheme|Yojana|Plan))', text, re.IGNORECASE)
            title = title_match.group(0).strip() if title_match else f"Chunk {i+1}"
            
            collection.add(
                documents=[text],
                ids=[f"{pdf_name}_chunk_{i+1}"],
                metadatas=[{"chunk": i+1, "pdf": pdf_name, "title": title}]
            )
        
    except Exception as e:
        print(f"Error adding chunks to ChromaDB: {e}")

def chat_with_bot(question, top_k=3):
    try:
        results = collection.query(query_texts=[question], n_results=top_k)
        
        if not results['documents'] or not results['documents'][0]:
            print("No relevant content found in the PDFs.")
            return
        
        retrieved_chunks = results['documents'][0]
        all_titles = [meta['title'] for meta in results['metadatas'][0]]
        
        context = "\n\n".join(retrieved_chunks)
        titles_str = ", ".join(set(all_titles))
        title_header = f"Information sourced from: {titles_str}"
        
        prompt = f"""{title_header}
Use the following context to answer the user's question accurately and comprehensively.

If the answer isn't directly found in the context, say so politely and suggest what information is available.

Context:
{context}

Question: {question}

Answer:"""
        response = conversation.predict(input=prompt)
        
        print(f"\nAnswer:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during chat: {e}")

def validate_pdf_paths(paths):
    valid_paths = []
    for path in paths:
        path = path.strip()
        if not path:
            continue
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        if not path.lower().endswith('.pdf'):
            print(f"Not a PDF file: {path}")
            continue
        valid_paths.append(path)
    return valid_paths

if __name__ == "__main__":
    print("PDF RAG Chatbot Starting...")    
    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not found in environment variables!")
        print("Please set your Gemini API key in a .env file or environment variable.")
        exit(1)
    
   
    pdf_input = input("\nEnter paths of PDF files (separated by commas): ").strip()
    if not pdf_input:
        print("No PDF paths provided!")
        exit(1)
    
    pdf_paths = pdf_input.split(",")
    valid_paths = validate_pdf_paths(pdf_paths)
    
    if not valid_paths:
        print("No valid PDF files found!")
        exit(1)
    
    for path in valid_paths:
        add_chunks_to_chroma(path)
    try:
        count = collection.count()
        if count == 0:
            #print("No chunks were successfully added to the database!")
            exit(1)
    except Exception as e:
        exit(1)

    print("Type 'exit' to quit, 'help' for tips, or ask any question.")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                continue
            
            chat_with_bot(query, top_k=3)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print("Please try again or type 'exit' to quit.")
'''