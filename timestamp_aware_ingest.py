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
    Returns list of dicts: {start, text}
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
                    "end": None,  # optional, can compute later
                    "text": text
                })
    # fill end times as next start
    for i in range(len(transcript) - 1):
        transcript[i]["end"] = transcript[i + 1]["start"]
    transcript[-1]["end"] = transcript[-1]["start"] + 5  # last chunk fallback
    return transcript


def chunk_and_embed(transcript: List[Dict]) -> List[Dict]:
    """
    Add embeddings for each chunk.
    """
    texts = [t["text"] for t in transcript]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    for i, t in enumerate(transcript):
        t["embedding"] = embeddings[i].tolist()
    return transcript


def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


def ingest_transcript(path: str, video_id: str):
    transcript = parse_transcript_txt(path)
    chunks = chunk_and_embed(transcript)

    points = []
    for idx, ch in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),  # unique ID
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
    print(f"Ingested {len(points)} chunks for {video_id}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    create_collection()
    ingest_transcript("sample_transcript.txt", "video_001")
