"""
rag.py
Step 3 of the ClipSage pipeline - the actual "ask a question, get an answer
plus the exact timestamp" logic. This is imported by main.py (the API),
and can also be run directly from the terminal to test quickly.

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull phi3

Usage (direct test, no server needed):
    python rag.py "what is a linked list?"
"""

import os
import sys
import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Same anchoring approach as build_index.py - must point to the exact same
# folder or the API won't find what build_index.py stored.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")
COLLECTION_NAME = "clipsage_lectures"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 3  # how many chunks to retrieve per question

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def retrieve_chunks(question: str, top_k: int = TOP_K):
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, **meta})
    return chunks


def call_ollama(prompt: str) -> str:
    """Send the prompt to a locally running Ollama model and get the answer back."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(question: str):
    chunks = retrieve_chunks(question)

    if not chunks:
        return {"answer": "No lecture content indexed yet.", "sources": []}

    context = "\n\n".join(f"[Segment {i+1}]: {c['text']}" for i, c in enumerate(chunks))

    prompt = f"""You are ClipSage, a helpful teaching assistant. Answer the student's
question using ONLY the lecture content below. If the answer isn't in the
content, say you don't have that information in the lecture. Keep the answer
short and clear.

Lecture content:
{context}

Question: {question}

Answer:"""

    answer_text = call_ollama(prompt)

    # the "best" source is simply the top-ranked retrieved chunk
    best = chunks[0]
    sources = [
        {"video_name": c["video_name"], "start": c["start"], "end": c["end"]}
        for c in chunks
    ]

    return {
        "answer": answer_text,
        "best_timestamp": {"video_name": best["video_name"], "start": best["start"], "end": best["end"]},
        "sources": sources,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer_question(question)
    print(json.dumps(result, indent=2))
