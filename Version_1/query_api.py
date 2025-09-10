from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize Qdrant + embedding model
client = QdrantClient(host="localhost", port=6333)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

COLLECTION_NAME = "gradus_transcripts"

class QueryRequest(BaseModel):
    query: str
    video_id: str
    top_k: int = 5

@app.post("/query")
def query_transcript(request: QueryRequest):
    # Embed the query
    query_vector = encoder.encode(request.query).tolist()

    # Search Qdrant for most relevant transcript chunks
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=request.top_k,
        query_filter={"must": [{"key": "video_id", "match": {"value": request.video_id}}]}
    )

    # Collect retrieved chunks
    retrieved_chunks = [
        hit.payload["text"] for hit in search_result
    ]

    # Construct the LLM-ready prompt
    context = "\n".join(retrieved_chunks)
    llm_prompt = f"""
You are a supportive and knowledgeable tutor, like a big brother helping someone learn. 
Your job is to answer questions using both:
- Transcript context (approx. 30% weight) as the primary inspiration
- General knowledge (approx. 70% weight) to expand, clarify, and fill gaps

Transcript context:
{context}

Question:
{request.query}

Instructions:
- Give detailed, step-by-step answers for most questions, but if the user explicitly asks for a summary, respond briefly and concisely.  
- Always explain concepts in a friendly, approachable way, like you are guiding someone through their doubts.  
- If a transcript excerpt directly supports your answer, cite the **timestamp(s)** (e.g., "At 12:34…") instead of quoting text.  
- If transcript information is wrong, confusing, or missing, rely on your broader knowledge while still noting what the transcript attempted to say.  
- If there isn’t enough info, make your best educated guess while being transparent about uncertainty.  

Answer:
"""
    

    return {
        "query": request.query,
        "retrieved_chunks": retrieved_chunks,
        "llm_prompt": llm_prompt
    }
