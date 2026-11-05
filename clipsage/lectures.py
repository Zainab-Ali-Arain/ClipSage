"""
Lecture references and lecture navigation (built only from real transcripts).

  * segment_references(): for a topic, the best lecture segments with title,
    channel, thumbnail, start-end time, a link that opens YouTube at the
    start, the most relevant transcript excerpt, the concepts discussed
    (key terms of that segment), and data-based reasons why it is useful.
  * video_timeline(): every segment of one lecture with its main terms and
    definition sentences, so students can jump to a section.
  * explain_range(): "What is taught between 10:20 and 15:30?" - collects the
    transcript sentences of that time range and summarises them.
  * ask_lecture(): finds the lecture sentences that best answer a question.

Nothing here is generated: every excerpt, time and term is taken from the
collected captions. When information is missing, the functions say so.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .summarize import DEFINITION, Encoder, extractive_summary
from .text import contains_term, normalize_query
from .utils import format_timestamp, youtube_link

THUMBNAIL = "https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
TIME = r"(\d{1,2}:\d{2}(?::\d{2})?)"


def thumbnail_url(video_id: str) -> str:
    """YouTube's standard thumbnail address for a video (not stored, not re-hosted)."""
    return THUMBNAIL.format(video_id=video_id)


def parse_timestamp(text: str) -> float:
    """'10:20' -> 620, '1:02:05' -> 3725."""
    parts = [int(p) for p in text.strip().split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def parse_time_range(question: str) -> Optional[Tuple[float, float]]:
    """Find two times like 10:20 and 15:30 in a question."""
    found = re.findall(TIME, question)
    if len(found) < 2:
        return None
    a, b = parse_timestamp(found[0]), parse_timestamp(found[1])
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Helpers over the indexed segments
# ---------------------------------------------------------------------------
def segment_terms(engine, row: int, n: int = 6, exclude: Optional[str] = None) -> List[str]:
    """Most characteristic TF-IDF terms of one segment (uses the corpus IDF)."""
    idx = engine.index("tfidf")
    vec = idx.matrix[row]
    if vec.nnz == 0:
        return []
    vocab = idx.vectorizer.get_feature_names_out()
    order = np.argsort(-vec.data)
    terms: List[str] = []
    for j in order:
        term = vocab[vec.indices[j]]
        if exclude and (contains_term(term, exclude) or contains_term(exclude, term)):
            continue
        if any(term in t or t in term for t in terms):
            continue
        terms.append(term)
        if len(terms) >= n:
            break
    return terms


def video_rows(engine, video_id: str) -> List[int]:
    rows = [i for i, s in enumerate(engine.segments) if s["video_id"] == video_id]
    return sorted(rows, key=lambda i: engine.segments[i]["start"])


def video_sentences(engine, video_id: str) -> List[dict]:
    """All unique transcript sentences of a video, in time order."""
    seen, out = set(), []
    for row in video_rows(engine, video_id):
        for s in engine.segments[row]["sentences"]:
            key = round(s["start"], 1)
            if key not in seen:
                seen.add(key)
                out.append(dict(s))
    return sorted(out, key=lambda s: s["start"])


def video_info(engine, video_id: str) -> Optional[dict]:
    rows = video_rows(engine, video_id)
    if not rows:
        return None
    first, last = engine.segments[rows[0]], engine.segments[rows[-1]]
    return {"video_id": video_id, "title": first.get("title", ""), "channel_title": first.get("channel_title", ""),
            "subject": first.get("subject", ""), "transcript_start": first["start"], "transcript_end": last["end"],
            "segments": len(rows), "thumbnail": thumbnail_url(video_id), "url": youtube_link(video_id, 0)}


def list_videos(engine) -> List[dict]:
    ids = []
    for s in engine.segments:
        if s["video_id"] not in ids:
            ids.append(s["video_id"])
    return [video_info(engine, v) for v in ids]


def _best_excerpt(sentences: List[dict], query: str, encode, size: int = 3) -> List[dict]:
    """The `size` consecutive sentences that match the query best."""
    if len(sentences) <= size:
        return sentences
    vecs = encode([s["text"] for s in sentences] + [query])
    sims = vecs[:-1] @ vecs[-1]
    windows = [sims[i:i + size].sum() for i in range(len(sentences) - size + 1)]
    i = int(np.argmax(windows))
    return sentences[i:i + size]


# ---------------------------------------------------------------------------
# 1. References with exact timestamps
# ---------------------------------------------------------------------------
def segment_references(query: str, hits: List[dict], engine, method: str = "sbert",
                       max_refs: int = 6, topic_terms: Optional[List[str]] = None) -> List[dict]:
    """Enrich retrieved segments with everything the student needs to use them."""
    if not hits:
        return []
    terms_for_topic = [t for t in (topic_terms or [normalize_query(query)]) if t]
    topic = terms_for_topic[0]

    def mentions_topic(text: str) -> bool:
        return any(contains_term(text, t) for t in terms_for_topic)

    encode = Encoder(method, engine)
    query_terms = set(segment_terms(engine, hits[0]["row"], n=10, exclude=topic))
    refs = []
    for h in hits[:max_refs]:
        sentences = h.get("sentences", [])
        excerpt = _best_excerpt(sentences, topic, encode)
        terms = segment_terms(engine, h["row"], n=6, exclude=topic)
        definitions = [s for s in sentences if mentions_topic(s["text"]) and DEFINITION.search(s["text"])]
        mentions = sum(1 for s in sentences if mentions_topic(s["text"]))
        shared = [t for t in terms if t in query_terms][:3]

        reasons = [("why_similarity", {"score": h["score"], "rank": h["rank"], "n": len(hits)})]
        if definitions:
            reasons.append(("why_definition", {"topic": topic}))
        if mentions:
            reasons.append(("why_mentions", {"topic": topic, "n": mentions}))
        if shared and h["rank"] > 1:
            reasons.append(("why_concepts", {"terms": ", ".join(shared)}))

        refs.append({
            "video_id": h["video_id"], "title": h.get("title", ""), "channel_title": h.get("channel_title", ""),
            "thumbnail": thumbnail_url(h["video_id"]),
            "start": h["start"], "end": h["end"],
            "start_label": format_timestamp(h["start"]), "end_label": format_timestamp(h["end"]),
            "url": youtube_link(h["video_id"], h["start"]),
            "score": h["score"], "rank": h["rank"],
            "excerpt": excerpt,
            "excerpt_start": format_timestamp(excerpt[0]["start"]) if excerpt else None,
            "segment_text": h.get("text", ""),
            "concepts": terms,
            "definitions": definitions[:2],
            "mentions": mentions,
            "reasons": reasons,
            "transcript_kind": h.get("transcript_kind", ""),
        })
    return refs


# ---------------------------------------------------------------------------
# 2. Lecture timeline and navigation
# ---------------------------------------------------------------------------
def video_timeline(engine, video_id: str, query: Optional[str] = None, method: str = "tfidf") -> List[dict]:
    """Every (non-overlapping-start) section of a lecture with its main terms."""
    rows = video_rows(engine, video_id)
    scores = {}
    if query and rows:
        q = engine.index(method).encode([normalize_query(query)])
        vecs = engine.segment_vectors(method, rows)
        q = q.toarray()[0] if hasattr(q, "toarray") else q[0]
        norm = np.linalg.norm(q) or 1.0
        scores = {r: float(v @ q / norm) for r, v in zip(rows, vecs)}
    sections = []
    for r in rows:
        seg = engine.segments[r]
        defs = [s for s in seg["sentences"] if DEFINITION.search(s["text"]) and len(s["text"].split()) >= 6]
        sections.append({
            "row": r, "start": seg["start"], "end": seg["end"],
            "start_label": format_timestamp(seg["start"]), "end_label": format_timestamp(seg["end"]),
            "url": youtube_link(video_id, seg["start"]),
            "terms": segment_terms(engine, r, n=5),
            "first_sentence": seg["sentences"][0]["text"] if seg["sentences"] else seg["text"][:120],
            "definitions": defs[:2],
            "relevance": scores.get(r),
        })
    return sections


def most_useful_sections(sections: List[dict], n: int = 3) -> List[dict]:
    scored = [s for s in sections if s.get("relevance") is not None]
    return sorted(scored, key=lambda s: -s["relevance"])[:n]


def explain_range(engine, video_id: str, start: float, end: float, method: str = "tfidf",
                  n_sentences: int = 4) -> dict:
    """What is taught between two times of a lecture - from the transcript only."""
    info = video_info(engine, video_id)
    if info is None:
        return {"ok": False, "reason": "unknown_video"}
    if end < info["transcript_start"] or start > info["transcript_end"]:
        return {"ok": False, "reason": "out_of_range", "info": info}
    sentences = [s for s in video_sentences(engine, video_id) if s["end"] >= start and s["start"] <= end]
    if not sentences:
        return {"ok": False, "reason": "no_sentences", "info": info}

    rows = [r for r in video_rows(engine, video_id)
            if engine.segments[r]["end"] >= start and engine.segments[r]["start"] <= end]
    terms: List[str] = []
    for r in rows:
        for t in segment_terms(engine, r, n=4):
            if t not in terms:
                terms.append(t)
    focus = " ".join(terms[:4]) or info["title"]
    pseudo_hit = {"video_id": video_id, "title": info["title"], "channel_title": info["channel_title"],
                  "sentences": sentences, "rank": 1, "score": 1.0, "text": " ".join(s["text"] for s in sentences)}
    summary = extractive_summary(focus, [pseudo_hit], "embedding", method, engine, n_sentences)
    summary.sort(key=lambda s: s["start"])
    definitions = [s for s in sentences if DEFINITION.search(s["text"]) and len(s["text"].split()) >= 6]
    return {
        "ok": True, "info": info,
        "start": max(start, sentences[0]["start"]), "end": min(end, sentences[-1]["end"]),
        "start_label": format_timestamp(max(start, sentences[0]["start"])),
        "end_label": format_timestamp(min(end, sentences[-1]["end"])),
        "requested": (format_timestamp(start), format_timestamp(end)),
        "terms": terms[:8], "summary": summary, "definitions": definitions[:3],
        "sentences": sentences, "url": youtube_link(video_id, sentences[0]["start"]),
    }


def ask_lecture(engine, video_id: str, question: str, method: str = "tfidf", k: int = 3,
                min_score: float = 0.15) -> Dict:
    """Extractive answer: the lecture sentences most similar to the question."""
    sentences = [s for s in video_sentences(engine, video_id) if len(s["text"].split()) >= 5]
    if not sentences:
        return {"ok": False, "answers": []}
    encode = Encoder(method, engine)
    vecs = encode([s["text"] for s in sentences] + [normalize_query(question)])
    sims = vecs[:-1] @ vecs[-1]
    order = np.argsort(-sims)[:k]
    answers = [{**sentences[i], "score": round(float(sims[i]), 3),
                "timestamp": format_timestamp(sentences[i]["start"]),
                "url": youtube_link(video_id, sentences[i]["start"])} for i in order]
    return {"ok": bool(answers) and answers[0]["score"] >= min_score, "answers": answers}
