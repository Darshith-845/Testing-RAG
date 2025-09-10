#Basically used miniLm for embedding 
#Broke the script into chunk such that there window size is of 8 and overlapping of 3 chunks 
#

import re
import uuid
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# -------------------------
# Config
# -------------------------
COLLECTION_NAME = "gradus_transcripts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small + fast
WINDOW_SIZE = 8      # Number of transcript lines per chunk
WINDOW_OVERLAP = 3   # Number of overlapping lines

# -------------------------
# Qdrant client
# -------------------------
client = QdrantClient(host="localhost", port=6333)

# Embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# -------------------------
# Utils
# -------------------------
def parse_transcript_txt(path: str) -> List[Dict]:
    """
    Parse transcript lines of format:
    [mm:ss] text
    Returns list of dicts: {start, text, end}
    """
    transcript = []
    pattern = re.compile(r"\[(\d+):(\d+)\]\s*(.*)")

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                minutes, seconds, text = match.groups()
                start_time = int(minutes) * 60 + int(seconds)
                transcript.append({
                    "start": start_time,
                    "end": None,  # optional, will fill later
                    "text": text
                })

    # fill end times as next start
    for i in range(len(transcript) - 1):
        transcript[i]["end"] = transcript[i + 1]["start"]
    transcript[-1]["end"] = transcript[-1]["start"] + 5  # fallback
    return transcript


def create_chunks(transcript: List[Dict]) -> List[Dict]:
    """
    Create sliding window chunks for context-aware embeddings
    """
    chunks = []
    i = 0
    while i < len(transcript):
        chunk_lines = transcript[i:i + WINDOW_SIZE]
        chunk_text = " ".join([line["text"] for line in chunk_lines])
        chunk_start = chunk_lines[0]["start"]
        chunk_end = chunk_lines[-1]["end"]
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text
        })
        i += WINDOW_SIZE - WINDOW_OVERLAP  # move window with overlap
    return chunks


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for each chunk
    """
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    for i, c in enumerate(chunks):
        c["embedding"] = embeddings[i].tolist()
    return chunks


def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def ingest_transcript(path: str, video_id: str):
    transcript = parse_transcript_txt(path)
    chunks = create_chunks(transcript)
    chunks = embed_chunks(chunks)

    points = []
    for ch in chunks:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=ch["embedding"],
                payload={
                    "video_id": video_id,
                    "start": ch["start"],
                    "end": ch["end"],
                    "text": ch["text"]
                }
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingested {len(points)} context-aware chunks for {video_id}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    create_collection()
    ingest_transcript("sample_transcript.txt", "video_001")
