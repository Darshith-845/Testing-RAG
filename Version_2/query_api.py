import re
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# -------------------------
# Config
# -------------------------
COLLECTION_NAME = "gradus_transcripts"
client = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    video_id: str
    top_k: int = 5

# -------------------------
# Utils
# -------------------------
def extract_timestamp(query: str):
    """
    Detect timestamps like 0:25, 1:05, 12:30, etc.
    Returns seconds if found, else None.
    """
    pattern = re.compile(r"(\d+):(\d+)")
    match = pattern.search(query)
    if match:
        minutes, seconds = match.groups()
        return int(minutes) * 60 + int(seconds)
    return None


def semantic_search(query: str, video_id: str, top_k: int = 5) -> List[str]:
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedder.encode(query).tolist(),
        limit=top_k,
        query_filter={"must": [{"key": "video_id", "match": {"value": video_id}}]}
    )
    return [hit.payload["text"] for hit in search_result]


def timestamp_search(seconds: int, video_id: str, window: int = 10) -> List[str]:
    """
    Retrieve chunks that overlap with [seconds - window, seconds + window].
    """
    search_result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "video_id", "match": {"value": video_id}},
                {"key": "start", "range": {"lte": seconds + window}},
                {"key": "end", "range": {"gte": seconds - window}}
            ]
        },
        limit=5
    )
    return [point.payload["text"] for point in search_result[0]]


# -------------------------
# API Endpoint
# -------------------------
@app.post("/query")
def query_transcript(request: QueryRequest):
    timestamp = extract_timestamp(request.query)
    retrieved_chunks = []

    if timestamp is not None:
        # Timestamp-based retrieval
        retrieved_chunks = timestamp_search(timestamp, request.video_id)
    else:
        # Semantic retrieval
        retrieved_chunks = semantic_search(request.query, request.video_id, request.top_k)

    # Build LLM prompt
    llm_prompt = f"""
You are a supportive and knowledgeable tutor, like a big brother helping someone learn. 
Your job is to answer questions using both:
- Transcript context (approx. 30% weight) as the primary inspiration
- General knowledge (approx. 70% weight) to expand, clarify, and fill gaps

Transcript context:
{chr(10).join(retrieved_chunks)}

Question:
{request.query}

Instructions:
- Give detailed, step-by-step answers for most questions, but if the user explicitly asks for a summary, respond briefly and concisely.
- Always explain concepts in a friendly, approachable way, like you are guiding someone through their doubts.
- If a transcript excerpt directly supports your answer, cite the **timestamp(s)** instead of quoting text.
- If transcript information is wrong, confusing, or missing, rely on your broader knowledge while still noting what the transcript attempted to say.
- If there isn’t enough info, make your best educated guess while being transparent about uncertainty.

Answer:
"""

    return {
        "query": request.query,
        "retrieved_chunks": retrieved_chunks,
        "llm_prompt": llm_prompt
    }
