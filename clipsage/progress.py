"""
Learning data: temporary by default, saved only when the student clicks Save.

SessionActivity
    Everything the student does during a visit - searches, studied topics,
    viewed lectures, quiz answers, practice results. It lives only in memory
    (Streamlit session state) and disappears when the browser tab is closed.
    It is NEVER written to disk.

SubjectStore
    Explicitly saved items for ONE subject, in data/user/subjects/<subject_id>.json:
      saved_topics      - "Save Topic"
      notes             - "Save Notes"            (one note per topic)
      quiz_results      - "Save Quiz Results"
      progress          - "Save Progress"         (a snapshot of the session for this subject)
      materials         - "Save Learning Material" (an explanation, a solved problem, a simulation)
    The file is created only by a save action. Each subject has its own file,
    so saved data never mixes between subjects.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from . import config

SECTIONS = ("saved_topics", "notes", "quiz_results", "progress", "materials")
MAX_RECENT = 20


def _slug(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")


# ---------------------------------------------------------------------------
# Temporary (session) activity
# ---------------------------------------------------------------------------
class SessionActivity:
    """In-memory record of the current visit. Has no save method on purpose."""

    def __init__(self):
        self.searches: List[dict] = []
        self.studied: Dict[str, List[dict]] = {}      # subject -> [{id, title, ts}]
        self.viewed: List[dict] = []
        self.answers: Dict[str, Dict[str, dict]] = {}  # subject -> {question_id: result}
        self.practice: Dict[str, List[dict]] = {}     # subject -> simulator / maths results

    @staticmethod
    def _push(items: List[dict], item: dict, key: str) -> None:
        items[:] = [x for x in items if x.get(key) != item.get(key)]
        items.insert(0, item)
        del items[MAX_RECENT:]

    def add_search(self, query: str, lang: str, subject: Optional[str]) -> None:
        if query.strip():
            self._push(self.searches, {"q": query.strip(), "lang": lang, "subject": subject, "ts": time.time()}, "q")

    def add_studied(self, subject: str, topic_id: str, title: str) -> None:
        self._push(self.studied.setdefault(subject, []), {"id": topic_id, "title": title, "ts": time.time()}, "id")

    def add_viewed(self, video_id: str, title: str, subject: Optional[str]) -> None:
        self._push(self.viewed, {"video_id": video_id, "title": title, "subject": subject, "ts": time.time()},
                   "video_id")

    def record_answer(self, subject: str, topic_id: str, question_id: str, status: str, score=None) -> None:
        self.answers.setdefault(subject, {})[question_id] = {
            "topic_id": topic_id, "status": status, "correct": status == "correct",
            "score": None if score is None else round(float(score), 3), "ts": time.time()}

    def add_practice(self, subject: str, kind: str, summary: str, correct: Optional[bool] = None) -> None:
        self.practice.setdefault(subject, []).insert(0, {"kind": kind, "summary": summary, "correct": correct,
                                                        "ts": time.time()})

    def subject_summary(self, subject: str) -> Dict:
        answers = self.answers.get(subject, {})
        per_topic: Dict[str, Dict[str, int]] = {}
        for a in answers.values():
            p = per_topic.setdefault(a["topic_id"], {"answered": 0, "correct": 0})
            p["answered"] += 1
            p["correct"] += int(a["correct"])
        correct = sum(int(a["correct"]) for a in answers.values())
        return {"subject": subject, "answered": len(answers), "correct": correct,
                "accuracy": correct / len(answers) if answers else None, "per_topic": per_topic,
                "studied": [s["title"] for s in self.studied.get(subject, [])],
                "practice": len(self.practice.get(subject, []))}

    def topic_answers(self, subject: str, topic_id: str) -> Dict[str, dict]:
        return {q: a for q, a in self.answers.get(subject, {}).items() if a["topic_id"] == topic_id}


# ---------------------------------------------------------------------------
# Explicitly saved data, one file per subject
# ---------------------------------------------------------------------------
class SubjectStore:
    def __init__(self, subject: str, directory: Optional[Path] = None):
        self.subject = subject
        self.directory = Path(directory) if directory else config.USER_DIR / "subjects"
        self.path = self.directory / f"{_slug(subject)}.json"

    # -- low level ----------------------------------------------------------
    def load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data.setdefault("subject", self.subject)
        for s in SECTIONS:
            data.setdefault(s, {} if s == "notes" else [])
        return data

    def _write(self, data: dict) -> None:
        if data.get("subject") != self.subject:  # never write another subject's data here
            raise ValueError("subject mismatch")
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def exists(self) -> bool:
        return self.path.exists()

    # -- save actions (each one is triggered by a Save button) -------------
    def save_topic(self, topic_id: str, title: str) -> None:
        data = self.load()
        if not any(t["id"] == topic_id for t in data["saved_topics"]):
            data["saved_topics"].insert(0, {"id": topic_id, "title": title, "ts": time.time()})
            self._write(data)

    def save_note(self, topic_id: str, title: str, text: str) -> None:
        data = self.load()
        data["notes"][topic_id] = {"title": title, "text": text, "ts": time.time()}
        self._write(data)

    def save_quiz_result(self, topic_id: str, title: str, answered: int, correct: int, details: List[dict]) -> None:
        data = self.load()
        data["quiz_results"].insert(0, {"id": uuid.uuid4().hex[:8], "topic_id": topic_id, "title": title,
                                        "answered": answered, "correct": correct, "details": details,
                                        "ts": time.time()})
        self._write(data)

    def save_progress(self, summary: dict) -> None:
        data = self.load()
        data["progress"].insert(0, {"id": uuid.uuid4().hex[:8], **summary, "ts": time.time()})
        self._write(data)

    def save_material(self, kind: str, title: str, content: str, lang: str = "en") -> None:
        data = self.load()
        data["materials"].insert(0, {"id": uuid.uuid4().hex[:8], "kind": kind, "title": title,
                                     "content": content, "lang": lang, "ts": time.time()})
        self._write(data)

    # -- manage -------------------------------------------------------------
    def remove(self, section: str, key: str) -> None:
        data = self.load()
        if section == "notes":
            data["notes"].pop(key, None)
        elif section == "saved_topics":
            data["saved_topics"] = [x for x in data["saved_topics"] if x["id"] != key]
        else:
            data[section] = [x for x in data[section] if x.get("id") != key]
        self._write(data)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def is_topic_saved(self, topic_id: str) -> bool:
        return any(t["id"] == topic_id for t in self.load()["saved_topics"])

    def counts(self) -> Dict[str, int]:
        data = self.load()
        return {s: len(data[s]) for s in SECTIONS}
