"""
Stage 4 - Query processing and retrieval.

The topic typed by the student is vectorised with the same method as the
segments (TF-IDF or SBERT), the most similar segments are fetched by cosine
similarity, and the list is then re-ranked with Maximal Marginal Relevance
(MMR) plus a per-video cap so that the results come from different videos
instead of one video repeating itself.
"""

from difflib import get_close_matches
from typing import Dict, List, Optional

import numpy as np

from . import config
from .index import SbertIndex, TfidfIndex, load_segments
from .text import normalize_query
from .utils import read_json

METHODS = ("tfidf", "sbert")
METHOD_LABELS = {"tfidf": "TF-IDF (baseline)", "sbert": "Sentence-BERT (improved)"}


class SearchEngine:
    def __init__(self):
        self.segments: List[dict] = load_segments()
        self._indexes: Dict[str, object] = {}
        self._topic_vectors = None
        self.topics = self._load_topic_list()

    # ------------------------------------------------------------------ #
    def index(self, method: str):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if method not in self._indexes:
            loader = TfidfIndex if method == "tfidf" else SbertIndex
            idx = loader.load(config.INDEX_DIR)
            n = idx.matrix.shape[0] if method == "tfidf" else len(idx.vectors)
            if n != len(self.segments):
                raise RuntimeError("Index and segments are out of sync - run `python -m clipsage.index` again.")
            self._indexes[method] = idx
        return self._indexes[method]

    def segment_vectors(self, method: str, rows: List[int]) -> np.ndarray:
        """Vectors of some segments, as a dense L2-normalised array."""
        idx = self.index(method)
        if method == "tfidf":
            return idx.matrix[rows].toarray()
        return idx.vectors[rows]

    # ------------------------------------------------------------------ #
    def candidates(self, query: str, method: str, pool: int = config.CANDIDATE_POOL):
        """Top `pool` segments by cosine similarity, before re-ranking."""
        rows, scores = self.index(method).search(normalize_query(query), pool)
        keep = [(int(r), float(s)) for r, s in zip(rows, scores) if r >= 0]
        return keep

    def search(self, query: str, method: str = "sbert", top_k: int = config.TOP_K_SEGMENTS,
               diversify: bool = True, max_per_video: int = config.MAX_SEGMENTS_PER_VIDEO,
               mmr_lambda: float = config.MMR_LAMBDA) -> List[dict]:
        cands = self.candidates(query, method, max(config.CANDIDATE_POOL, top_k))
        if not cands:
            return []
        # filter by similarity score: weak matches would only add noise to the explanation
        cutoff = cands[0][1] * config.MIN_RELATIVE_SCORE
        cands = [c for c in cands if c[1] >= cutoff]
        if diversify:
            order = self._mmr(cands, method, top_k, max_per_video, mmr_lambda)
        else:
            order = cands[:top_k]
        return [
            {**self.segments[row], "row": row, "score": round(score, 4), "rank": rank}
            for rank, (row, score) in enumerate(order, 1)
        ]

    def _mmr(self, cands, method, top_k, max_per_video, lam):
        rows = [r for r, _ in cands]
        rel = np.array([s for _, s in cands])
        vecs = self.segment_vectors(method, rows)
        sim = vecs @ vecs.T
        selected: List[int] = []
        per_video: Dict[str, int] = {}
        remaining = list(range(len(cands)))
        capped = True
        while remaining and len(selected) < top_k:
            best, best_score = None, -1e9
            for i in remaining:
                vid = self.segments[rows[i]]["video_id"]
                if capped and per_video.get(vid, 0) >= max_per_video:
                    continue
                redundancy = max((sim[i, j] for j in selected), default=0.0)
                score = lam * rel[i] - (1 - lam) * redundancy
                if score > best_score:
                    best, best_score = i, score
            if best is None:
                # Every remaining candidate is from an already "full" video. Few
                # relevant videos exist, so relax the cap rather than return less.
                capped = False
                continue
            selected.append(best)
            remaining.remove(best)
            vid = self.segments[rows[best]]["video_id"]
            per_video[vid] = per_video.get(vid, 0) + 1
        return [cands[i] for i in selected]

    # ------------------------------------------------------------------ #
    def rank_videos(self, query: str, method: str, k: int = 10, pool: int = 200) -> List[dict]:
        """
        Video-level ranking used for evaluation (Precision@5, Recall@10, MRR):
        each video is scored by its best-matching segment.
        """
        best: Dict[str, dict] = {}
        for row, score in self.candidates(query, method, pool):
            seg = self.segments[row]
            vid = seg["video_id"]
            if vid not in best or score > best[vid]["score"]:
                best[vid] = {"video_id": vid, "title": seg["title"], "channel_title": seg["channel_title"],
                             "start": seg["start"], "score": score}
        return sorted(best.values(), key=lambda v: -v["score"])[:k]

    # ------------------------------------------------------------------ #
    def is_covered(self, hits: List[dict], method: str) -> bool:
        return bool(hits) and hits[0]["score"] >= config.NOT_COVERED_THRESHOLD[method]

    def _load_topic_list(self) -> List[str]:
        topics = []
        try:
            for info in read_json(config.TOPICS_FILE)["subjects"].values():
                topics.extend(info.get("keywords", []))
        except FileNotFoundError:
            pass
        # only suggest topics that really appear in the collected lectures
        corpus = " ".join(s["text"].lower() for s in self.segments)
        return sorted(t for t in set(topics) if t.lower() in corpus)

    def suggest_topics(self, query: str, method: str, n: int = 5) -> List[str]:
        """Related topics that ARE covered, shown when a query is not covered."""
        if not self.topics:
            return []
        q = normalize_query(query).lower()
        if method == "sbert":
            try:
                if self._topic_vectors is None:
                    self._topic_vectors = self.index("sbert").encode(self.topics)
                qv = self.index("sbert").encode([q])[0]
                order = np.argsort(-(self._topic_vectors @ qv))
                return [self.topics[i] for i in order[:n]]
            except Exception:
                pass
        close = get_close_matches(q, self.topics, n=n, cutoff=0.3)
        return close or self.topics[:n]


_engine: Optional[SearchEngine] = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def reset_engine() -> None:
    global _engine
    _engine = None
