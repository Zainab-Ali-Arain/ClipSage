"""
Topic grouping (extra ML component in proposal section 7).

  * K-Means on segment vectors groups the retrieved segments of one query into
    sub-topics (k chosen with the silhouette score).
  * LDA over the whole corpus finds the main themes of the collected lectures
    (used for the dataset analysis in the report).

Usage (corpus-level LDA report):
    python -m clipsage.topics --lda-topics 10
"""

from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .text import STOP_WORDS, TOKEN_PATTERN


def label_clusters(texts: List[str], labels: np.ndarray, n_terms: int = 3) -> List[str]:
    """Name each cluster by its most distinctive TF-IDF terms."""
    vec = TfidfVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))
    m = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    names = []
    for c in sorted(set(labels)):
        inside = np.asarray(m[labels == c].mean(axis=0)).ravel()
        outside = np.asarray(m[labels != c].mean(axis=0)).ravel() if (labels != c).any() else 0
        top = np.argsort(-(inside - outside))[:n_terms]
        names.append(", ".join(vocab[i] for i in top))
    return names


def subtopics(hits: List[dict], vectors: np.ndarray, max_k: int = 4) -> List[dict]:
    """Group retrieved segments into 2..max_k sub-topics with K-Means."""
    n = len(hits)
    if n < 4:
        return []
    best_k, best_labels, best_score = None, None, -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels, metric="cosine")
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    if best_k is None:
        return []
    names = label_clusters([h["text"] for h in hits], best_labels)
    groups = []
    for c, name in enumerate(names):
        members = [h for h, lab in zip(hits, best_labels) if lab == c]
        groups.append({"label": name, "segments": members, "size": len(members)})
    groups.sort(key=lambda g: min(m["rank"] for m in g["segments"]))
    return groups


def corpus_lda(n_topics: int = 10, n_words: int = 10):
    """LDA over all segments; writes reports/lda_topics.md."""
    from collections import Counter

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    from . import config
    from .index import load_segments

    segments = load_segments()
    texts = [s["text"] for s in segments]
    vec = CountVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, max_df=0.5 if len(texts) >= 20 else 1.0,
                          min_df=3 if len(texts) >= 100 else 1)
    counts = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    doc_topics = lda.fit_transform(counts)
    vocab = vec.get_feature_names_out()

    dominant = doc_topics.argmax(axis=1)
    lines = ["# LDA topics over all lecture segments", "",
             f"{len(texts)} segments, {n_topics} topics, perplexity {lda.perplexity(counts):.1f}", "",
             "| Topic | Top words | Segments | Main subject |", "|---|---|---|---|"]
    for t in range(n_topics):
        words = ", ".join(vocab[i] for i in np.argsort(-lda.components_[t])[:n_words])
        members = [segments[i]["subject"] for i in np.where(dominant == t)[0]]
        main = Counter(members).most_common(1)[0][0] if members else "-"
        lines.append(f"| {t + 1} | {words} | {len(members)} | {main} |")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "lda_topics.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Corpus-level LDA topics")
    parser.add_argument("--lda-topics", type=int, default=10)
    args = parser.parse_args()
    corpus_lda(args.lda_topics)


if __name__ == "__main__":
    main()
