"""
Practice questions: grading, hints and questions built from lecture transcripts.

Grading
  * MCQ        - exact match of the chosen option.
  * numerical  - the number is parsed (fractions like 3/4 allowed) and compared
                 with the stored answer within a tolerance.
  * short      - an ML grader: Sentence-BERT cosine similarity between the
                 student's answer and the reference answer, combined with the
                 share of key terms the answer mentions. Falls back to TF-IDF
                 similarity when the SBERT model is not available.

Lecture questions
  cloze_from_lectures() turns real lecture sentences into fill-in-the-blank
  MCQs: the blank is a key term found in the sentence, the distractors are
  other key terms of the same topic. Each question cites its video + time.
"""

import random
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from .text import contains_term
from .utils import format_timestamp, youtube_link

CORRECT, PARTIAL, INCORRECT = "correct", "partial", "incorrect"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def parse_number(text: str) -> Optional[float]:
    text = str(text).strip().replace(",", "").replace(" ", "")
    if not text:
        return None
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def grade_mcq(question: dict, choice: Optional[int]) -> dict:
    ok = choice is not None and int(choice) == int(question["answer"])
    return {"status": CORRECT if ok else INCORRECT, "score": 1.0 if ok else 0.0}


def grade_numerical(question: dict, text: str) -> dict:
    value = parse_number(text)
    if value is None:
        return {"status": INCORRECT, "score": 0.0, "error": "not_a_number"}
    answer = float(question["answer"])
    tol = float(question.get("tolerance", 1e-3))
    ok = abs(value - answer) <= max(tol, 1e-9)
    return {"status": CORRECT if ok else INCORRECT, "score": 1.0 if ok else 0.0, "value": value}


_sbert_failed = False


def _similarity(a: str, b: str) -> Tuple[float, str]:
    """Cosine similarity of two texts with SBERT, or TF-IDF if SBERT is unavailable."""
    global _sbert_failed
    try:
        if _sbert_failed:
            raise RuntimeError("SBERT unavailable")
        from .embeddings import get_embedder

        vecs = get_embedder().encode([a, b])
        return float(vecs[0] @ vecs[1]), "sbert"
    except Exception:
        _sbert_failed = True
        from sklearn.feature_extraction.text import TfidfVectorizer

        m = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([a, b])
        num = float((m[0] @ m[1].T).toarray()[0, 0])
        return num, "tfidf"


def _keyword_hit(keyword: str, answer: str) -> bool:
    keyword = keyword.lower()
    if contains_term(answer, keyword):
        return True
    # accept a different ending (generalise / generalize / generalisation)
    stem = keyword[: max(4, len(keyword) - 3)]
    return re.search(rf"\b{re.escape(stem)}", answer.lower()) is not None


def grade_short(question: dict, answer: str) -> dict:
    answer = (answer or "").strip()
    if len(answer.split()) < 2:
        return {"status": INCORRECT, "score": 0.0, "similarity": 0.0, "coverage": 0.0,
                "matched": [], "missing": question.get("keywords", []), "method": None}
    sim, method = _similarity(answer, question["answer"])
    keywords = question.get("keywords", [])
    matched = [k for k in keywords if _keyword_hit(k, answer)]
    missing = [k for k in keywords if k not in matched]
    need = question.get("min_keywords") or len(keywords)
    coverage = min(1.0, len(matched) / need) if keywords else sim
    score = 0.5 * max(sim, 0.0) + 0.5 * coverage
    if score >= 0.6 and coverage >= 0.5:
        status = CORRECT
    elif score >= 0.35:
        status = PARTIAL
    else:
        status = INCORRECT
    return {"status": status, "score": round(score, 3), "similarity": round(sim, 3),
            "coverage": round(coverage, 3), "matched": matched, "missing": missing, "method": method}


def grade(question: dict, response) -> dict:
    kind = question["type"]
    if kind == "mcq":
        return grade_mcq(question, response)
    if kind == "numerical":
        return grade_numerical(question, response)
    return grade_short(question, response)


# ---------------------------------------------------------------------------
# Questions from the lecture transcripts
# ---------------------------------------------------------------------------
def cloze_from_lectures(hits: List[dict], key_terms: List[Tuple[str, float]], n: int = 3,
                        seed: int = 7) -> List[dict]:
    """Fill-in-the-blank MCQs built from real lecture sentences."""
    terms = [t for t, _ in key_terms]
    if len(terms) < 4:
        return []
    rng = random.Random(seed)
    questions, used_terms, used_sentences = [], set(), set()
    candidates = []
    for h in hits:
        for s in h.get("sentences", []):
            words = len(s["text"].split())
            if 8 <= words <= 40:
                candidates.append((h, s))
    for h, s in candidates:
        if len(questions) >= n:
            break
        text = s["text"]
        present = [t for t in terms if t not in used_terms and contains_term(text, t)]
        if not present or text in used_sentences:
            continue
        answer = max(present, key=len)  # prefer the most specific term
        pattern = re.compile(rf"\b{re.escape(answer)}s?\b", re.IGNORECASE)
        blanked = pattern.sub("_____", text, count=1)
        distractors = [t for t in terms if t != answer and not contains_term(text, t)]
        if len(distractors) < 3:
            continue
        options = [answer] + rng.sample(distractors, 3)
        rng.shuffle(options)
        used_terms.add(answer)
        used_sentences.add(text)
        questions.append({
            "id": f"lec_{h['video_id']}_{int(s['start'])}",
            "type": "mcq", "kind": "lecture",
            "q": {"en": blanked, "ru": blanked},
            "options": [{"en": o, "ru": o} for o in options],
            "answer": options.index(answer),
            "source": {"video_id": h["video_id"], "title": h.get("title", ""),
                       "channel_title": h.get("channel_title", ""), "start": s["start"],
                       "timestamp": format_timestamp(s["start"]), "url": youtube_link(h["video_id"], s["start"])},
            "hints": [{"en": f"Listen to the lecture at {format_timestamp(s['start'])}.",
                       "ru": f"Lecture ko {format_timestamp(s['start'])} par sunein."}],
            "explanation": {"en": f"The lecture sentence reads: “{text}”",
                            "ru": f"Lecture ka asal sentence: “{text}”"},
        })
    return questions


def score_summary(results: Dict[str, dict]) -> Dict[str, float]:
    graded = [r for r in results.values() if r]
    correct = sum(1 for r in graded if r["status"] == CORRECT)
    return {"answered": len(graded), "correct": correct,
            "percent": round(100 * correct / len(graded)) if graded else 0}
