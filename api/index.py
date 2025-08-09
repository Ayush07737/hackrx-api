import os
import pdfplumber
import pinecone
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
import requests
from typing import List
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
api_token = os.getenv("API_TOKEN")

if not pinecone_api_key or not openai.api_key or not api_token:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY, OPENAI_API_KEY, or API_TOKEN")

# Initialize Pinecone
try:
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
    index_name = "hackrx-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)
    index = pinecone.Index(index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Initialize Sentence Transformer with a lighter model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller model, ~80 MB

# FastAPI app setup
app = FastAPI(title="HackRx 6.0 API")
security = HTTPBearer()

# Request and Response models
class RequestModel(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]  # List of questions to answer

class ResponseModel(BaseModel):
    answers: List[str]  # List of answers to the questions
    success: bool = True  # Added for HackRx compatibility

# Token authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token != api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")
    return token

# Extract text from PDF with table support and retry
def extract_text_from_pdf(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        with open("/tmp/temp.pdf", "wb") as f:
            f.write(response.content)
        with pdfplumber.open("/tmp/temp.pdf") as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
                for table in page.extract_tables():
                    for row in table:
                        text += " ".join(str(cell) for cell in row if cell) + "\n"
        os.remove("/tmp/temp.pdf")
        if not text.strip():
            raise ValueError("No text or table data extracted from PDF")
        return text
    except requests.RequestException as e:
        logger.error(f"Failed to download PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# Chunk text into manageable pieces
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Generate embeddings for text chunks
def generate_embeddings(texts: List[str]) -> list:
    try:
        return model.encode(texts, show_progress_bar=False, batch_size=32)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Upsert embeddings into Pinecone
def upsert_embeddings(chunks: List[str], embeddings: list, namespace: str = "hackrx"):
    try:
        vectors = [(f"chunk_{i}", emb.tolist(), {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
        index.upsert(vectors=vectors, namespace=namespace)
    except Exception as e:
        logger.error(f"Error upserting embeddings to Pinecone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error upserting embeddings: {str(e)}")

# Retrieve relevant chunks for a query
def search_relevant_chunks(query_embedding, top_k: int = 3, namespace: str = "hackrx") -> List[str]:
    try:
        query_vector = query_embedding.tolist()
        results = index.query(queries=[query_vector], top_k=top_k, namespace=namespace, include_metadata=True)
        return [match['metadata']['text'] for match in results['results'][0]['matches']]
    except Exception as e:
        logger.error(f"Error searching Pinecone index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching Pinecone index: {str(e)}")

# Generate answer using OpenAI LLM
def generate_answer(question: str, chunks: List[str]) -> str:
    if not chunks:
        return "Information not found in the provided context."
    context = "\n".join(chunks)
    prompt = f"Answer the question based on the context below. Provide a concise, accurate answer, referencing specific details from the context. If the context is insufficient, state 'Information not found in the provided context.'\n\nQuestion: {question}\n\nContext:\n{context}"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.3,
            timeout=3
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with OpenAI: {str(e)}")
        return f"Error generating answer: {str(e)}"

# Main API endpoint
@app.post("/hackrx/run", response_model=ResponseModel)
async def process_query(request: RequestModel, token: str = Depends(verify_token)):
    start_time = time.time()
    
    try:
        # Step 1: Extract text from PDF
        text = extract_text_from_pdf(request.documents)
        
        # Step 2: Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Step 3: Generate and upsert embeddings
        chunk_embeddings = generate_embeddings(chunks)
        upsert_embeddings(chunks, chunk_embeddings)
        
        # Step 4: Process each question
        answers = []
        for question in request.questions:
            # Generate query embedding
            question_embedding = generate_embeddings([question])[0]
            
            # Retrieve relevant chunks
            relevant_chunks = search_relevant_chunks(question_embedding)
            
            # Generate answer
            answer = generate_answer(question, relevant_chunks)
            answers.append(answer)
            
            # Check time constraint
            if time.time() - start_time > 7:
                raise HTTPException(status_code=408, detail="Processing timed out")
        
        return ResponseModel(answers=answers)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}