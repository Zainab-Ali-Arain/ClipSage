"""The six ClipSage subjects and helpers to keep their content separate."""

from typing import Dict, List, Optional

from . import config
from .knowledge import load_topics
from .utils import read_json

SUBJECTS_FILE = config.CONFIG_DIR / "subjects.json"
_cache: Optional[List[dict]] = None


def load_subjects() -> List[dict]:
    global _cache
    if _cache is None:
        _cache = read_json(SUBJECTS_FILE)["subjects"]
    return _cache


def subject_names() -> List[str]:
    return [s["name"] for s in load_subjects()]


def get_subject(key: str) -> Optional[dict]:
    """By id ('operating_systems') or by name ('Operating Systems')."""
    key = (key or "").strip().lower()
    for s in load_subjects():
        if key in (s["id"], s["name"].lower()):
            return s
    return None


def subject_id(name: str) -> Optional[str]:
    s = get_subject(name)
    return s["id"] if s else None


def topics_of(subject: str) -> List[dict]:
    """Lessons of one subject only (never mixed with other subjects)."""
    s = get_subject(subject)
    if s is None:
        return []
    return [t for t in load_topics().values() if t["subject"] == s["name"]]


def topics_by_category(subject: str) -> Dict[str, List[dict]]:
    s = get_subject(subject)
    out: Dict[str, List[dict]] = {}
    for t in topics_of(subject):
        label = (s or {}).get("categories", {}).get(t["category"], t["category"].replace("_", " ").title())
        out.setdefault(label, []).append(t)
    return out
