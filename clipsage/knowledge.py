"""
Hand-written bilingual lessons ("ClipSage notes") in knowledge/*.json.

Each file holds a list of topics. A topic gives a structured lesson in
English and Roman Urdu (intro, definition, explanation, example, concepts,
formulas, algorithm workflow, Python code, applications, exam points, common
mistakes) and a question bank. These notes are written explanations; they
are always shown with the "ClipSage notes" label so they are never confused
with retrieved lecture content.

Lecture retrieval still works for ANY topic; notes only exist for core topics.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from . import config
from .text import normalize_query

REQUIRED = ("id", "title", "subject", "category", "definition", "explanation")
CATEGORIES = {
    "ml_algorithm": {"en": "ML algorithm", "ru": "ML algorithm"},
    "ml_concept": {"en": "ML concept", "ru": "ML concept"},
    "nlp": {"en": "NLP / retrieval", "ru": "NLP / retrieval"},
    "math": {"en": "Mathematics", "ru": "Mathematics"},
    "cs": {"en": "Computer Science", "ru": "Computer Science"},
}

_cache: Optional[Dict[str, dict]] = None


def load_topics(directory=None, refresh: bool = False) -> Dict[str, dict]:
    global _cache
    if _cache is not None and not refresh and directory is None:
        return _cache
    directory = directory or config.KNOWLEDGE_DIR
    topics: Dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            for topic in json.load(f):
                missing = [k for k in REQUIRED if k not in topic]
                if missing:
                    raise ValueError(f"{path.name}: topic {topic.get('id')} misses {missing}")
                if topic["id"] in topics:
                    raise ValueError(f"duplicate topic id {topic['id']}")
                if isinstance(topic.get("code"), list):  # stored line by line for readability
                    topic["code"] = "\n".join(topic["code"])
                for i, q in enumerate(topic.get("questions", [])):
                    q.setdefault("id", f"{topic['id']}_q{i + 1}")
                topics[topic["id"]] = topic
    if directory == config.KNOWLEDGE_DIR:
        _cache = topics
    return topics


def _names(topic: dict) -> List[str]:
    names = [topic["id"].replace("_", " "), topic["title"]["en"]]
    names += topic.get("aliases", [])
    return [n.lower() for n in names]


def find_topic(query: str, topics: Optional[Dict[str, dict]] = None, min_ratio: float = 0.82) -> Optional[dict]:
    """Best matching topic for a free-text query, or None."""
    topics = topics or load_topics()
    q = normalize_query(query).lower().strip()
    if not q:
        return None
    if q.replace(" ", "_") in topics:
        return topics[q.replace(" ", "_")]
    best, best_score = None, 0.0
    for topic in topics.values():
        for name in _names(topic):
            if q == name:
                return topic
            if re.search(rf"\b{re.escape(name)}\b", q) or re.search(rf"\b{re.escape(q)}\b", name):
                score = 0.9 + 0.1 * min(len(q), len(name)) / max(len(q), len(name))
            else:
                score = SequenceMatcher(None, q, name).ratio()
            if score > best_score:
                best, best_score = topic, score
    return best if best_score >= min_ratio else None


def topics_by_subject(topics: Optional[Dict[str, dict]] = None) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for t in (topics or load_topics()).values():
        out.setdefault(t["subject"], []).append(t)
    return out


def search_query_for(topic: dict) -> str:
    """Text used to retrieve lecture segments for a topic."""
    return topic.get("search_query") or topic["title"]["en"]
