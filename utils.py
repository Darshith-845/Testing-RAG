import re
from sentence_transformers import SentenceTransformer

# Load embedding model globally (MiniLM for speed/cost)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    return embedder.encode(text).tolist()

def chunk_transcript(transcript, chunk_size=400, overlap=100):
    """
    transcript: list of dicts [{"start": float, "end": float, "text": str}, ...]
    Returns chunks with timestamps + embeddings
    """
    chunks, buffer, start_time = [], [], None
    token_count = 0

    for entry in transcript:
        tokens = entry["text"].split()
        if start_time is None:
            start_time = entry["start"]

        for word in tokens:
            buffer.append(word)
            token_count += 1
            if token_count >= chunk_size:
                text = " ".join(buffer)
                end_time = entry["end"]
                chunks.append({
                    "text": text,
                    "start": start_time,
                    "end": end_time,
                    "embedding": embed_text(text)
                })
                buffer = buffer[-overlap:]  # keep overlap
                token_count = len(buffer)
                start_time = end_time

    if buffer:
        text = " ".join(buffer)
        chunks.append({
            "text": text,
            "start": start_time,
            "end": transcript[-1]["end"],
            "embedding": embed_text(text)
        })
    return chunks

def parse_timestamp(query: str):
    """
    Detects timestamp like mm:ss or hh:mm:ss in query.
    Returns seconds if found, else None.
    """
    match = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{2})", query)
    if match:
        h = int(match.group(1)) if match.group(1) else 0
        m = int(match.group(2))
        s = int(match.group(3))
        return h * 3600 + m * 60 + s
    return None
