"""
Key-term extraction (Stage 6, and the "Key terms" feature).

  * baseline: TF-IDF scores of the retrieved segments (IDF from the whole corpus)
  * improved: KeyBERT (Grootendorst) using the same Sentence-BERT model
"""

from typing import List, Tuple

import numpy as np

from . import config
from .text import STOP_WORDS, TOKEN_PATTERN, normalize_query

KEYWORD_METHODS = ("keybert", "tfidf")
KEYWORD_LABELS = {"tfidf": "TF-IDF scores (baseline)", "keybert": "KeyBERT (improved)"}


def singular(word: str) -> str:
    """Very small plural -> singular rule set (processes -> process, queries -> query)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and (w[-3] in "sxz" or w.endswith(("ches", "shes"))):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _norm(term: str) -> List[str]:
    return [singular(w) for w in term.split()]


def _dedupe(terms: List[Tuple[str, float]], query: str, n: int) -> List[Tuple[str, float]]:
    """
    Drop the query itself (it is the centre of the concept map), terms that just
    repeat the query word, plural/word-order duplicates and stop-word edges.
    """
    q_words = set(_norm(normalize_query(query)))
    out: List[Tuple[str, float]] = []
    seen: List[List[str]] = []
    for term, score in terms:
        term = term.strip().lower()
        words = _norm(term)
        if not words or words[0] in STOP_WORDS or words[-1] in STOP_WORDS:
            continue
        if set(words) <= q_words or (len(words) > 1 and set(words) & q_words):
            continue
        if any(sorted(words) == sorted(s) or (len(words) == 1 and words[0] in s and len(s) == 1) for s in seen):
            continue
        seen.append(words)
        out.append((term, float(score)))
        if len(out) >= n:
            break
    return out


def tfidf_keywords(texts: List[str], query: str, n: int = config.NUM_KEY_TERMS, engine=None) -> List[Tuple[str, float]]:
    if engine is not None:
        vectorizer = engine.index("tfidf").vectorizer
        matrix = vectorizer.transform(texts)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
    weights = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    order = np.argsort(-weights)[: n * 5]
    terms = [(vocab[i], weights[i]) for i in order if weights[i] > 0]
    top = weights[order[0]] if len(order) and weights[order[0]] > 0 else 1.0
    return [(t, round(s / top, 4)) for t, s in _dedupe(terms, query, n)]


def keybert_keywords(texts: List[str], query: str, n: int = config.NUM_KEY_TERMS) -> List[Tuple[str, float]]:
    from keybert import KeyBERT
    from sklearn.feature_extraction.text import CountVectorizer

    from .embeddings import keybert_backend

    # Each segment is a separate document (so bigrams never cross segment
    # boundaries); a term's score is summed over the segments it was found in,
    # weighted by how high the segment ranked.
    model = KeyBERT(model=keybert_backend())
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN)
    per_doc = model.extract_keywords(texts, vectorizer=vectorizer, top_n=10)
    if texts and isinstance(per_doc[0], tuple):  # a single document returns a flat list
        per_doc = [per_doc]
    totals = {}
    for rank, terms in enumerate(per_doc):
        weight = 1.0 / (1 + 0.15 * rank)
        for term, score in terms:
            totals[term] = totals.get(term, 0.0) + score * weight
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    top = ranked[0][1] if ranked else 1.0
    return [(t, round(s / top, 4)) for t, s in _dedupe(ranked, query, n)]


_keybert_error = None


def keybert_available() -> bool:
    return _keybert_error is None


def extract_keywords(texts: List[str], query: str, method: str = "keybert",
                     n: int = config.NUM_KEY_TERMS, engine=None) -> List[Tuple[str, float]]:
    global _keybert_error
    if not texts:
        return []
    if method == "keybert" and _keybert_error is None:
        try:
            return keybert_keywords(texts, query, n)
        except Exception as e:  # e.g. model not available - fall back to the baseline
            _keybert_error = str(e)
            print(f"KeyBERT unavailable ({e}); using TF-IDF key terms instead.")
    return tfidf_keywords(texts, query, n, engine)
