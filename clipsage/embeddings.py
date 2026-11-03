"""
Sentence-BERT embeddings (the "improved" representation in the proposal).

All vectors are L2-normalised, so a dot product equals cosine similarity.
The model is loaded once and shared by retrieval, summarisation, KeyBERT,
sub-topic clustering and topic suggestions.
"""

from typing import List, Optional

import numpy as np

from . import config

_embedder = None


class SbertEmbedder:
    def __init__(self, model_name: str = config.SBERT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SbertEmbedder()
    return _embedder


def set_embedder(embedder: Optional[object]) -> None:
    """Swap in another embedder (used by the tests)."""
    global _embedder
    _embedder = embedder


def keybert_backend():
    """Expose our shared embedder to KeyBERT so the model is not loaded twice."""
    from keybert.backend import BaseEmbedder

    embedder = get_embedder()

    class _Backend(BaseEmbedder):
        def embed(self, documents, verbose=False):
            return embedder.encode(list(documents))

    return _Backend()
