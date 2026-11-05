"""
Video library: the collected YouTube videos organised by subject and topic,
and the video panel shown beside every explanation.

Every video here was found by the YouTube Data API search for a lesson's
`video_queries` (see collect.collect_library) - nothing is typed in by hand.
Timestamps come only from the video's captions:
  * best_segment   - the lecture segment most similar to the topic;
  * subtopic times - the first caption sentence that mentions each subtopic
                     (e.g. FCFS / SJF / Round Robin) in that video.
If a video has no usable captions, no timestamps are shown and this is said.
"""

import csv
from typing import Dict, List, Optional

from . import config
from .subjects import load_subjects
from .text import contains_term
from .utils import format_timestamp, youtube_link

THUMB = "https://i.ytimg.com/vi/{}/mqdefault.jpg"


def load_library() -> List[dict]:
    """All collected videos that belong to one of the six subjects."""
    if not config.VIDEOS_CSV.exists():
        return []
    names = {s["name"] for s in load_subjects()}
    out = []
    with open(config.VIDEOS_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject") not in names:
                continue
            vid = row["video_id"]
            out.append({
                "video_id": vid, "title": row.get("title", ""), "channel_title": row.get("channel_title", ""),
                "subject": row["subject"], "topic_id": row.get("topic_id") or None,
                "duration": int(float(row.get("duration_seconds") or 0)),
                "has_transcript": (config.TRANSCRIPT_DIR / f"{vid}.json").exists(),
                "url": youtube_link(vid, 0), "thumbnail": THUMB.format(vid),
            })
    return out


def library_stats(videos: Optional[List[dict]] = None) -> Dict:
    videos = videos if videos is not None else load_library()
    per_subject: Dict[str, int] = {s["name"]: 0 for s in load_subjects()}
    per_topic: Dict[str, int] = {}
    for v in videos:
        per_subject[v["subject"]] = per_subject.get(v["subject"], 0) + 1
        if v["topic_id"]:
            per_topic[v["topic_id"]] = per_topic.get(v["topic_id"], 0) + 1
    ids = [v["video_id"] for v in videos]
    return {"total": len(videos), "per_subject": per_subject, "per_topic": per_topic,
            "with_transcript": sum(v["has_transcript"] for v in videos),
            "duplicates": len(ids) - len(set(ids))}


def _video_sentences(engine, video_id: str) -> List[dict]:
    from .lectures import video_sentences

    return video_sentences(engine, video_id) if engine is not None else []


def subtopic_timestamps(engine, video_id: str, topic: dict) -> List[dict]:
    """First caption sentence mentioning each subtopic of the lesson (real times only)."""
    sentences = _video_sentences(engine, video_id)
    out = []
    for sub in topic.get("subtopics", []):
        hit = next((s for s in sentences if any(contains_term(s["text"], term) for term in sub["terms"])), None)
        out.append({"name": sub["name"], "found": hit is not None,
                    "start": hit["start"] if hit else None,
                    "label": format_timestamp(hit["start"]) if hit else None,
                    "url": youtube_link(video_id, hit["start"]) if hit else None,
                    "sentence": hit["text"] if hit else None})
    return out


def is_about_topic(engine, video_id: str, topic: dict) -> bool:
    """
    A retrieved lecture is only shown for a topic if its captions name the topic
    (its title or a multi-word alias) or mention at least two of its subtopics.
    Single generic words ('process', 'matrix') are not enough.
    """
    text = " ".join(s["text"] for s in _video_sentences(engine, video_id)).lower()
    strong = [topic["title"]["en"]] + [a for a in topic.get("aliases", []) if len(a.split()) >= 2]
    if any(contains_term(text, n.lower()) for n in strong):
        return True
    hits = sum(1 for sub in topic.get("subtopics", []) if any(contains_term(text, t) for t in sub["terms"]))
    return hits >= 2


def best_segment(engine, video_id: str, query: str, method: str = "tfidf") -> Optional[dict]:
    """The section of a video that matches the topic best."""
    if engine is None:
        return None
    from .lectures import video_timeline

    sections = video_timeline(engine, video_id, query=query, method=method)
    scored = [s for s in sections if s.get("relevance") is not None]
    return max(scored, key=lambda s: s["relevance"]) if scored else None


def videos_for_topic(topic: dict, engine=None, method: str = "tfidf", limit: int = 3) -> List[dict]:
    """
    Videos for the panel beside a topic's explanation:
      1. library videos collected for exactly this topic;
      2. otherwise lecture segments of the SAME subject retrieved for the topic.
    """
    from .knowledge import search_query_for

    query = search_query_for(topic)
    chosen = [v for v in load_library() if v["topic_id"] == topic["id"]]
    source = "topic library"
    if not chosen and engine is not None:
        source = "retrieved from the subject's lectures"
        seen = set()
        for h in engine.search(query, method=method, top_k=20):
            if h.get("subject") == topic["subject"] and h["video_id"] not in seen \
                    and is_about_topic(engine, h["video_id"], topic):
                seen.add(h["video_id"])
                chosen.append({"video_id": h["video_id"], "title": h["title"], "channel_title": h["channel_title"],
                               "subject": h["subject"], "topic_id": h.get("topic_id") or None,
                               "has_transcript": True, "url": youtube_link(h["video_id"], 0),
                               "thumbnail": THUMB.format(h["video_id"]), "duration": 0})
    panels = []
    for v in chosen[:limit]:
        seg = best_segment(engine, v["video_id"], query, method) if v["has_transcript"] else None
        subs = subtopic_timestamps(engine, v["video_id"], topic) if v["has_transcript"] and engine else []
        panels.append({**v, "source": source,
                       "start": seg["start"] if seg else 0,
                       "segment": seg,
                       "covered": [s for s in subs if s["found"]],
                       "not_found": [s["name"] for s in subs if not s["found"]],
                       "watch_url": youtube_link(v["video_id"], seg["start"] if seg else 0)})
    return panels


def topic_coverage(subject: str) -> Dict[str, int]:
    """Number of library videos per topic of a subject (for the overview charts)."""
    from .subjects import topics_of

    counts = library_stats()["per_topic"]
    return {t["id"]: counts.get(t["id"], 0) for t in topics_of(subject)}

