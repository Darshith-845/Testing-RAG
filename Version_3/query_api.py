import re
import html
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # DeepInfra-compatible client

# -------------------------
# Config
# -------------------------
COLLECTION_NAME = "gradus_transcripts"
client = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# DeepInfra LLaMA client
deepinfra = OpenAI(
    api_key="XubftArOyxYxUH0piTeJ4ZiTTXKQmnQd",
    base_url="https://api.deepinfra.com/v1/openai",
)

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
STOP_PHRASES = [
    "subscribe", "like button", "thanks for watching",
    "welcome to", "be sure to", "future updates", "like", "subscribe", "share",
    "follow", "click", "button"
]

def clean_text(text: str) -> str:
    text = html.unescape(text)
    for phrase in STOP_PHRASES:
        text = re.sub(rf"\b{phrase}\b", "", text, flags=re.IGNORECASE)
    return text.strip()

def extract_timestamp(query: str):
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

def build_llm_prompt(query: str, retrieved_chunks: List[str]) -> str:
    retrieved_chunks = "\n".join(retrieved_chunks)
    prompt = f"""
You are Orion, a friendly tutor who explains concepts in a clear and approachable way. 
Your role is to help the user learn by combining information from the transcript (≈60%) with your general knowledge (≈40%).

Guidelines:
1. Use the transcript chunks first to ground your answer.  
2. If something is missing, expand naturally with your own knowledge.  
3. Keep the tone friendly, like a big brother teaching step by step.  
4. Break explanations into short paragraphs, bullets, or examples.  
5. If timestamps are relevant, give an approximate range (e.g., “around 0:20–0:40”).  
6. Adjust depth based on the question: be concise if simple, detailed if asked, or playful if the user wants a quiz.  

Transcript chunks:
{retrieved_chunks}

Question:
{query}
"""
    return prompt

def llm_response(prompt: str) -> str:
    """
    Call the DeepInfra LLaMA model to generate a response from the prompt.
    """
    response = deepinfra.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )
    # Combine all content deltas
    return "".join([choice.message.content for choice in response.choices])

# -------------------------
# API Endpoint
# -------------------------
@app.post("/query")
def query_transcript(request: QueryRequest):
    timestamp = extract_timestamp(request.query)
    retrieved_chunks = []

    if timestamp is not None:
        retrieved_chunks = timestamp_search(timestamp, request.video_id)
    else:
        retrieved_chunks = semantic_search(request.query, request.video_id, request.top_k)

    # Build LLM prompt
    llm_prompt = build_llm_prompt(request.query, retrieved_chunks)
    
    # Get LLM response
    llm_output = llm_response(llm_prompt)

    return {
        "llm_output": llm_output
    }
