from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from utils import embed_text, parse_timestamp

app = FastAPI()
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "gradus_transcripts"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    video_id: str = None

@app.post("/query")
def query_transcript(req: QueryRequest):
    # Check if query has timestamp
    ts = parse_timestamp(req.query)

    if ts is not None:
        # Direct timestamp lookup
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={
                "must": [
                    {"key": "video_id", "match": {"value": req.video_id}}
                ]
            }
        )[0]

        # Find chunk containing timestamp
        matches = [r.payload for r in results if r.payload["start"] <= ts <= r.payload["end"]]
        if matches:
            return {"mode": "timestamp", "results": matches}

    # Fallback: semantic ANN retrieval
    query_vector = embed_text(req.query)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=req.top_k,
        query_filter={
            "must": [{"key": "video_id", "match": {"value": req.video_id}}]
        } if req.video_id else None
    )
    return {
        "mode": "semantic",
        "results": [
            {"text": r.payload["text"], "start": r.payload["start"], "end": r.payload["end"], "score": r.score}
            for r in search_result
        ]
    }
