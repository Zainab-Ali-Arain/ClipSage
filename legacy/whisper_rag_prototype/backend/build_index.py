"""
build_index.py
Step 2 of the ClipSage pipeline.

Takes the transcript JSON made by transcribe.py, groups segments into
slightly bigger chunks (so each chunk has enough context to answer a
question), embeds them, and stores them in a local ChromaDB collection.

Usage:
    python build_index.py data/lecture1_transcript.json
"""

import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Anchor paths to this script's location, not the current working directory,
# so it doesn't matter whether you run this from the project root or from
# inside backend/ - the database always ends up in the same place.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")
COLLECTION_NAME = "clipsage_lectures"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly
CHUNK_WINDOW = 4                   # merge every N whisper segments into one chunk


def load_transcript(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_into_chunks(segments, window=CHUNK_WINDOW):
    """
    Whisper segments are short (one sentence-ish). We merge a few together
    so each chunk has enough context for the LLM to answer well, while still
    keeping an accurate start/end timestamp range for jump-to-video.
    """
    chunks = []
    for i in range(0, len(segments), window):
        group = segments[i:i + window]
        text = " ".join(s["text"] for s in group).strip()
        chunks.append({
            "text": text,
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })
    return chunks


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def index_transcript(video_name: str, video_path: str, segments: list) -> int:
    """
    Reusable core of the pipeline: chunk -> embed -> store.
    Used by both the CLI (main() below) and the /upload API endpoint,
    so there's only one place this logic lives.

    Returns the number of chunks stored.
    """
    chunks = group_into_chunks(segments)

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [f"{video_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"video_name": video_name, "video_path": video_path, "start": c["start"], "end": c["end"]}
        for c in chunks
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <path_to_transcript.json>")
        sys.exit(1)

    transcript_path = sys.argv[1]
    data = load_transcript(transcript_path)
    video_name = data["video_name"]
    video_path = data["video_path"]

    print(f"Loaded transcript for '{video_name}' with {len(data['segments'])} raw segments.")
    print(f"Loading embedding model '{EMBED_MODEL}' ...")
    count = index_transcript(video_name, video_path, data["segments"])

    print(f"\nDone. {count} chunks stored in ChromaDB collection '{COLLECTION_NAME}' at ./{CHROMA_PATH}")
    print("You can run this script again for more videos - they all go into the same collection.")


if __name__ == "__main__":
    main()
