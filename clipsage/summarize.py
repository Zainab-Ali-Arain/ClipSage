"""
Stage 5 - Explanation generation.

The timestamped sentences of the retrieved segments are ranked and the best
4-6 are kept as the explanation. Because the summary is EXTRACTIVE, every
sentence is copied from a real lecture and keeps its video ID + timestamp.

Methods (proposal section 7):
  * "top_sentences"  - baseline: sentences most similar to the query
  * "textrank"       - TextRank (Mihalcea & Tarau 2004) on a sentence-similarity
                       graph, personalised towards the query
  * "embedding"      - embedding-based ranking: similarity to the query and to
                       the centroid of all retrieved sentences
  * optional abstractive rewrite with a pre-trained BART / T5 model

A definition-style sentence ("X is ...", "X refers to ...") is placed first
when one exists, so the explanation starts with a definition.
"""

import re
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from . import config
from .text import contains_term, normalize_query
from .utils import format_timestamp, youtube_link

SUMMARIZERS = ("top_sentences", "textrank", "embedding")
SUMMARIZER_LABELS = {
    "top_sentences": "Top-ranked sentences (baseline)",
    "textrank": "TextRank",
    "embedding": "Embedding-based ranking",
}

CHATTER = re.compile(
    r"\b(subscribe|like and share|welcome (back )?to|in this video|in today's video|in the next video|"
    r"see you|thanks for watching|thank you for watching|link in the description|patreon|sponsor|"
    r"my name is|hey guys|hi everyone|hello everyone|let's get started|stay tuned)\b",
    re.IGNORECASE,
)
DEFINITION = re.compile(
    r"\b(is|are)\s+(a|an|the|basically|simply|just|called|defined as|known as)\b|"
    r"\b(refers? to|is defined as|are defined as|means that|is when|happens when|occurs when|is used to)\b",
    re.IGNORECASE,
)


def collect_sentences(hits: List[dict]) -> List[dict]:
    """Unique, usable sentences of the retrieved segments with their source."""
    seen, out = set(), []
    for hit in hits:
        for s in hit.get("sentences", []):
            key = (hit["video_id"], round(s["start"], 1))
            text = s["text"].strip()
            n_words = len(text.split())
            if key in seen or n_words < 6 or n_words > 80 or CHATTER.search(text):
                continue
            seen.add(key)
            out.append({
                "text": text, "start": s["start"], "end": s["end"],
                "video_id": hit["video_id"], "title": hit.get("title", ""),
                "channel_title": hit.get("channel_title", ""),
                "segment_rank": hit.get("rank", 1), "segment_score": hit.get("score", 0.0),
            })
    return out


class Encoder:
    """Vectorises sentences with the same representation as the retrieval method."""

    def __init__(self, method: str, engine=None):
        self.method = method
        self.engine = engine

    def __call__(self, texts: List[str]) -> np.ndarray:
        if self.method == "tfidf" and self.engine is not None:
            m = self.engine.index("tfidf").encode(texts).toarray()
        elif self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            m = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
        else:
            from .embeddings import get_embedder

            return get_embedder().encode(texts)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


def _scores(method: str, sent_vecs: np.ndarray, query_vec: np.ndarray, sentences: List[dict]) -> np.ndarray:
    q_sim = np.clip(sent_vecs @ query_vec, 0, None)
    # small prior for sentences from higher-ranked segments
    prior = np.array([1.0 / (1 + 0.1 * (s["segment_rank"] - 1)) for s in sentences])

    if method == "top_sentences":
        return q_sim * prior

    if method == "embedding":
        centroid = sent_vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm else centroid
        c_sim = np.clip(sent_vecs @ centroid, 0, None)
        return (0.6 * q_sim + 0.4 * c_sim) * prior

    if method == "textrank":
        sim = np.clip(sent_vecs @ sent_vecs.T, 0, None)
        np.fill_diagonal(sim, 0)
        graph = nx.from_numpy_array(sim)
        personalization = {i: float(v) + 1e-6 for i, v in enumerate(q_sim * prior)}
        try:
            pr = nx.pagerank(graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=200)
        except nx.PowerIterationFailedConvergence:
            pr = personalization
        ranks = np.array([pr[i] for i in range(len(sentences))])
        ranks = ranks / ranks.max() if ranks.max() > 0 else ranks
        # keep the summary on-topic: TextRank centrality x query relevance
        return ranks * (0.5 + q_sim)

    raise ValueError(f"unknown summarizer {method}")


def extractive_summary(query: str, hits: List[dict], method: str = "textrank",
                       retrieval_method: str = "sbert", engine=None,
                       n_sentences: int = config.SUMMARY_SENTENCES,
                       topic_terms: Optional[List[str]] = None) -> List[dict]:
    """
    Return the explanation as a list of cited sentences.
    `topic_terms` are names of the topic (e.g. 'k-means', 'kmeans') used to spot
    sentences that mention or define it; by default the query itself.
    """
    sentences = collect_sentences(hits)
    if not sentences:
        return []
    topic = normalize_query(query)
    encode = Encoder(retrieval_method, engine)
    vecs = encode([s["text"] for s in sentences] + [topic])
    sent_vecs, query_vec = vecs[:-1], vecs[-1]
    scores = _scores(method, sent_vecs, query_vec, sentences)

    # Mention of the topic itself is a strong signal for a short explanation.
    terms = [t for t in (topic_terms or [topic]) if t]
    mentions = np.array([1.0 if any(contains_term(s["text"], t) for t in terms) else 0.0 for s in sentences])
    scores = scores * (1 + 0.3 * mentions)

    order = list(np.argsort(-scores))
    n_sentences = int(np.clip(n_sentences, 1, config.SUMMARY_MAX_SENTENCES))

    # 1) a definition sentence first, if a good one exists
    chosen: List[int] = []
    if method != "top_sentences":
        top_slice = order[: max(10, n_sentences * 3)]
        for i in top_slice:
            if mentions[i] and DEFINITION.search(sentences[i]["text"]):
                chosen.append(int(i))
                break

    # 2) fill up with the best non-redundant sentences
    for i in order:
        if len(chosen) >= n_sentences:
            break
        if i in chosen:
            continue
        if chosen and max(float(sent_vecs[i] @ sent_vecs[j]) for j in chosen) > config.REDUNDANCY_THRESHOLD:
            continue
        chosen.append(int(i))

    result = []
    for i in chosen:
        s = sentences[i]
        result.append({
            **s,
            "score": round(float(scores[i]), 4),
            "timestamp": format_timestamp(s["start"]),
            "url": youtube_link(s["video_id"], s["start"]),
            "is_definition": bool(mentions[i] and DEFINITION.search(s["text"])),
        })
    return result


# ---------------------------------------------------------------------- #
# Optional abstractive rewrite (BART / T5) - only if time and hardware allow
# ---------------------------------------------------------------------- #
_abstractive_pipe = None


def abstractive_summary(query: str, extractive: List[dict], hits: List[dict],
                        model_name: str = config.ABSTRACTIVE_MODEL) -> Optional[Dict]:
    """
    Rewrites the extractive explanation with a pre-trained summarisation model.
    The abstractive text cannot be traced sentence-by-sentence, so it is shown
    together with the list of sources it was generated from.
    """
    global _abstractive_pipe
    try:
        if _abstractive_pipe is None:
            from transformers import pipeline

            _abstractive_pipe = pipeline("summarization", model=model_name)
    except Exception as e:  # model not downloaded / no internet / not enough RAM
        return {"text": None, "error": f"Abstractive model unavailable: {e}"}

    context = " ".join(s["text"] for s in extractive)
    extra = " ".join(h["text"] for h in hits[:3])
    words = (context + " " + extra).split()[:600]
    prefix = "summarize: " if "t5" in model_name.lower() else ""
    out = _abstractive_pipe(prefix + " ".join(words), max_length=130, min_length=50, do_sample=False)
    sources = []
    for s in extractive:
        ref = {"video_id": s["video_id"], "start": s["start"], "url": s["url"], "title": s["title"]}
        if ref not in sources:
            sources.append(ref)
    return {"text": out[0]["summary_text"].strip(), "sources": sources, "model": model_name}
