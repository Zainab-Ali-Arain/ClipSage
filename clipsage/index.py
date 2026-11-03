"""
Stage 3 - Representation and indexing.

Every segment is turned into a vector twice:
  * baseline: TF-IDF (unigrams + bigrams, lecture-specific stop words)
  * improved: Sentence-BERT embedding (all-MiniLM-L6-v2 by default)

The SBERT vectors go into a FAISS inner-product index (cosine similarity on
normalised vectors). If FAISS is not installed, a NumPy/scikit-learn style
brute-force search is used instead - results are identical.

Usage:
    python -m clipsage.index
"""

import pickle
import time
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .embeddings import get_embedder
from .text import STOP_WORDS, TOKEN_PATTERN
from .utils import read_json, read_jsonl, write_json

TFIDF_FILE = "tfidf.pkl"
SBERT_FILE = "sbert_embeddings.npy"
FAISS_FILE = "sbert.faiss"
INFO_FILE = "index_info.json"


def make_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, lowercase=True,
        ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.5,
    )


class TfidfIndex:
    """Baseline: keyword matching with cosine similarity on TF-IDF vectors."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix):
        self.vectorizer = vectorizer
        self.matrix = matrix  # rows are L2-normalised by TfidfVectorizer

    @classmethod
    def build(cls, texts: List[str]) -> "TfidfIndex":
        vectorizer = make_tfidf_vectorizer()
        if len(texts) < 10:  # max_df would remove everything in a tiny corpus
            vectorizer.set_params(max_df=1.0)
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer, matrix)

    def encode(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = (self.matrix @ self.encode([query]).T).toarray().ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def save(self, directory):
        with open(directory / TFIDF_FILE, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, directory) -> "TfidfIndex":
        with open(directory / TFIDF_FILE, "rb") as f:
            data = pickle.load(f)
        return cls(data["vectorizer"], data["matrix"])


class SbertIndex:
    """Improved: semantic search on Sentence-BERT embeddings (FAISS inner product)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)
        self.faiss_index = None
        try:
            import faiss

            self.faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.faiss_index.add(self.vectors)
        except ImportError:
            pass

    @classmethod
    def build(cls, texts: List[str]) -> "SbertIndex":
        vectors = get_embedder().encode(texts, show_progress=True)
        return cls(vectors)

    def encode(self, texts: List[str]) -> np.ndarray:
        return get_embedder().encode(texts)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = self.encode([query])
        k = min(k, len(self.vectors))
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(q, k)
            return idx[0], scores[0]
        scores = self.vectors @ q[0]
        top = np.argsort(-scores)[:k]
        return top, scores[top]

    def save(self, directory):
        np.save(directory / SBERT_FILE, self.vectors)
        if self.faiss_index is not None:
            import faiss

            faiss.write_index(self.faiss_index, str(directory / FAISS_FILE))

    @classmethod
    def load(cls, directory) -> "SbertIndex":
        return cls(np.load(directory / SBERT_FILE))


def load_segments() -> List[dict]:
    if not config.SEGMENTS_FILE.exists():
        raise FileNotFoundError("No segments found - run `python -m clipsage.preprocess` first.")
    return list(read_jsonl(config.SEGMENTS_FILE))


def segment_index_text(seg: dict) -> str:
    """Text that is vectorised for a segment: the video title adds useful context."""
    return f"{seg.get('title', '')}. {seg['text']}"


def build(skip_sbert: bool = False) -> dict:
    segments = load_segments()
    texts = [segment_index_text(s) for s in segments]
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tfidf = TfidfIndex.build(texts)
    tfidf.save(config.INDEX_DIR)
    t_tfidf = time.time() - t0
    print(f"TF-IDF index: {tfidf.matrix.shape[0]} segments x {tfidf.matrix.shape[1]} terms ({t_tfidf:.1f}s)")

    info = {
        "segments": len(segments),
        "tfidf_terms": int(tfidf.matrix.shape[1]),
        "tfidf_build_seconds": round(t_tfidf, 2),
        "sbert_model": None,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not skip_sbert:
        t0 = time.time()
        sbert = SbertIndex.build(texts)
        sbert.save(config.INDEX_DIR)
        t_sbert = time.time() - t0
        backend = "FAISS" if sbert.faiss_index is not None else "NumPy"
        print(f"SBERT index: {sbert.vectors.shape} with {config.SBERT_MODEL} ({backend}, {t_sbert:.1f}s)")
        info.update(sbert_model=config.SBERT_MODEL, sbert_dim=int(sbert.vectors.shape[1]),
                    sbert_build_seconds=round(t_sbert, 2), search_backend=backend)
    write_json(config.INDEX_DIR / INFO_FILE, info)
    print(f"Index saved in {config.INDEX_DIR}")
    return info


def index_info() -> dict:
    path = config.INDEX_DIR / INFO_FILE
    return read_json(path) if path.exists() else {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ClipSage Stage 3 - build TF-IDF and SBERT indexes")
    parser.add_argument("--skip-sbert", action="store_true", help="only build the TF-IDF baseline")
    args = parser.parse_args()
    build(skip_sbert=args.skip_sbert)


if __name__ == "__main__":
    main()
