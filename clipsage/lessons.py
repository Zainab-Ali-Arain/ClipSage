"""
Topic learning: combines the hand-written lesson (if one exists) with what the
retrieved lectures actually say about the topic.

    lesson = build_lesson("overfitting")
    lesson["notes"]        -> ClipSage notes (bilingual), or None
    lesson["lecture"]      -> pipeline result (explanation, references, key terms,
                              concept graph, sub-topics) or None when no index
    lesson["refs"]         -> segment references with start-end, excerpt, concepts
    lesson["lecture_questions"] -> fill-in-the-blank questions from transcripts
"""

from typing import Optional

from . import config
from .knowledge import find_topic, search_query_for
from .lectures import segment_references
from .quiz import cloze_from_lectures
from .text import normalize_query


def topic_terms_for(query: str, notes: Optional[dict]) -> list:
    """Names a lecture may use for the topic: the student's words, the title and the aliases."""
    terms = [normalize_query(query).lower()]
    if notes:
        terms += [notes["title"]["en"].lower()] + [a.lower() for a in notes.get("aliases", [])]
    out = []
    for t in terms:
        if t and t not in out:
            out.append(t)
    return out


def index_available() -> bool:
    from .index import index_info

    return bool(index_info()) and config.SEGMENTS_FILE.exists()


def sbert_available() -> bool:
    from .index import index_info

    return bool(index_info().get("sbert_model"))


def build_lesson(query: str, method: str = "sbert", summarizer: str = "textrank",
                 keyword_method: Optional[str] = None, top_k: int = config.TOP_K_SEGMENTS,
                 n_sentences: int = config.SUMMARY_SENTENCES, sage=None) -> dict:
    notes = find_topic(query)
    lesson = {"query": query, "notes": notes, "lecture": None, "refs": [], "lecture_questions": [],
              "index_available": index_available(), "error": None}
    if not lesson["index_available"]:
        return lesson
    if method == "sbert" and not sbert_available():
        method = "tfidf"
    try:
        if sage is None:
            from .pipeline import ClipSage

            sage = ClipSage()
        search = search_query_for(notes) if notes else query
        terms = topic_terms_for(query, notes)
        result = sage.ask(search, method=method, summarizer=summarizer, keyword_method=keyword_method,
                          top_k=top_k, n_sentences=n_sentences, topic_terms=terms)
        result["query"] = query
        lesson["lecture"] = result
        if result["covered"]:
            lesson["refs"] = segment_references(search, result["hits"], sage.engine, method, topic_terms=terms)
            lesson["lecture_questions"] = cloze_from_lectures(result["hits"], result["key_terms"])
    except Exception as e:  # keep the lesson usable even if retrieval fails
        lesson["error"] = f"{type(e).__name__}: {e}"
    return lesson
