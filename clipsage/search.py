"""
Query analysis for the search bar.

    analyze_query("Process aur thread mein kya difference hai?")
    -> {"query": ..., "lang": "ru", "topic": <processes_threads lesson>,
        "subject": "Operating Systems", "matches": [...], "focus": "process thread"}

Steps
  1. detect the response language (langdetect: lexicon + Naive Bayes);
  2. strip question words of BOTH languages to get the topic words;
  3. match the topic words against every lesson's title and aliases
     (whole-phrase and token-overlap scores) and, as a fallback, TF-IDF
     similarity with each lesson's English text;
  4. the subject is the subject of the best lesson, or - when no lesson
     matches well - the subject whose lessons are most similar (TF-IDF),
     or the subject named in the query ("... in operating systems").
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .knowledge import load_topics
from .langdetect import RU_MARKERS, detect_with_details, tokens
from .subjects import load_subjects
from .text import STOP_WORDS

QUESTION_WORDS = {
    "what", "is", "are", "how", "why", "explain", "define", "describe", "tell", "me", "about", "the", "a", "an",
    "difference", "differences", "between", "vs", "versus", "and", "or", "of", "in", "on", "does", "do", "work",
    "works", "working", "meaning", "means", "example", "examples", "please", "give", "show", "with", "solve",
    "find", "use", "uses", "used", "concept", "topic", "lecture", "notes", "introduction", "basics", "kindly",
    "tareeqa", "nikalne", "nikalna", "steps", "step", "detail", "detailed", "simple", "words",
}
FILLER = QUESTION_WORDS | RU_MARKERS | set(STOP_WORDS)
MATCH_THRESHOLD = 0.55
SUBJECT_THRESHOLD = 0.05   # below this similarity the query is not linked to any of the six subjects


def topic_words(query: str) -> List[str]:
    """Content words of a query in either language ('paging kya hai' -> ['paging'])."""
    words = [w for w in tokens(query) if w not in FILLER and len(w) > 1]
    return words or tokens(query)


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _name_score(name: str, words: List[str], text: str) -> float:
    name = name.lower()
    name_words = [_singular(w) for w in tokens(name) if w not in FILLER] or [_singular(w) for w in tokens(name)]
    if not name_words:
        return 0.0
    qset = {_singular(w) for w in words}
    if re.search(rf"\b{re.escape(name)}s?\b", text):
        return 1.0 + 0.05 * len(name_words)          # full phrase present: best evidence
    covered = sum(1 for w in name_words if w in qset) / len(name_words)
    precision = sum(1 for w in qset if w in name_words) / max(1, len(qset))
    return 0.7 * covered + 0.3 * precision if covered else 0.0


@lru_cache(maxsize=1)
def _profile_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    topics = list(load_topics().values())
    docs = [" ".join([t["title"]["en"], " ".join(t.get("aliases", [])), t["definition"]["en"], t["intro"]["en"]])
            for t in topics]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return topics, vec, vec.fit_transform(docs)


def match_topics(query: str, subject: Optional[str] = None, n: int = 5) -> List[Dict]:
    """Ranked lessons for a query; optionally only inside one subject."""
    words = topic_words(query)
    text = " ".join(tokens(query))
    topics, vec, matrix = _profile_index()
    qv = vec.transform([" ".join(words)])
    sims = (matrix @ qv.T).toarray().ravel()
    out = []
    for t, sim in zip(topics, sims):
        if subject and t["subject"] != subject:
            continue
        names = [t["title"]["en"], t["id"].replace("_", " ")] + t.get("aliases", [])
        name_score = max(_name_score(nm, words, text) for nm in names)
        score = max(name_score, 0.9 * float(sim))
        if score > 0:
            out.append({"topic": t, "score": round(score, 3), "name_score": round(name_score, 3),
                        "tfidf": round(float(sim), 3)})
    out.sort(key=lambda m: -m["score"])
    return out[:n]


@lru_cache(maxsize=1)
def _subject_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    subjects = load_subjects()
    docs = []
    for s in subjects:
        lessons = [t for t in load_topics().values() if t["subject"] == s["name"]]
        docs.append(" ".join([s["name"], s.get("tagline", ""), s["intro"]["en"]] +
                             [t["title"]["en"] + " " + " ".join(t.get("aliases", [])) + " " + t["definition"]["en"]
                              for t in lessons]))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return subjects, vec, vec.fit_transform(docs)


def classify_subject(query: str) -> Dict:
    """Which of the six subjects a query belongs to (TF-IDF nearest subject profile)."""
    text = " ".join(tokens(query))
    for s in load_subjects():  # an explicit subject name wins
        if s["name"].lower() in text:
            return {"subject": s["name"], "score": 1.0, "method": "named in query"}
    subjects, vec, matrix = _subject_index()
    sims = (matrix @ vec.transform([" ".join(topic_words(query))]).T).toarray().ravel()
    i = int(np.argmax(sims))
    if sims[i] < SUBJECT_THRESHOLD:
        return {"subject": None, "score": float(sims[i]), "method": "no match"}
    return {"subject": subjects[i]["name"], "score": round(float(sims[i]), 3), "method": "tf-idf similarity"}


def analyze_query(query: str, subject: Optional[str] = None) -> Dict:
    query = (query or "").strip()
    detection = detect_with_details(query)
    named = classify_subject(query) if subject is None else {"subject": subject, "method": "selected subject"}
    restrict = subject or (named["subject"] if named.get("method") == "named in query" else None)
    matches = match_topics(query, restrict)
    best = matches[0] if matches and matches[0]["score"] >= MATCH_THRESHOLD else None
    if best:
        subj, subj_method = best["topic"]["subject"], "matched lesson"
    else:
        subj, subj_method = named["subject"], named["method"]
    return {
        "query": query,
        "lang": detection["lang"],
        "detection": detection,
        "focus": " ".join(topic_words(query)),
        "topic": best["topic"] if best else None,
        "matches": matches,
        "subject": subj,
        "subject_method": subj_method,
    }
