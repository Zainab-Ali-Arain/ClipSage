"""
Local learning-progress store (single user, one JSON file on this computer).

Keeps recent searches, viewed lectures, saved topics, quiz attempts and
playground activity. Nothing is sent anywhere.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config

MAX_RECENT = 20


def _empty() -> dict:
    return {"searches": [], "viewed": [], "saved": [], "studied": [],
            "quiz": {}, "math": {"attempts": 0, "correct": 0}, "playground_runs": 0, "settings": {}}


class ProgressStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else config.USER_DIR / "progress.json"
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return _empty()
        base = _empty()
        base.update(data)
        return base

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def reset(self) -> None:
        settings = self.data.get("settings", {})
        self.data = _empty()
        self.data["settings"] = settings  # keep preferences such as the language
        self.save()

    def get_setting(self, name: str, default=None):
        return self.data.get("settings", {}).get(name, default)

    def set_setting(self, name: str, value) -> None:
        if self.data.setdefault("settings", {}).get(name) != value:
            self.data["settings"][name] = value
            self.save()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _push_recent(items: List[dict], item: dict, key: str) -> None:
        items[:] = [x for x in items if x.get(key) != item.get(key)]
        items.insert(0, item)
        del items[MAX_RECENT:]

    def add_search(self, query: str) -> None:
        query = query.strip()
        if query:
            self._push_recent(self.data["searches"], {"q": query, "ts": time.time()}, "q")
            self.save()

    def add_studied(self, topic_id: str, title: str) -> None:
        self._push_recent(self.data["studied"], {"id": topic_id, "title": title, "ts": time.time()}, "id")
        self.save()

    def add_viewed(self, video_id: str, title: str, channel: str, start: float) -> None:
        self._push_recent(self.data["viewed"], {"video_id": video_id, "title": title, "channel": channel,
                                                "start": start, "ts": time.time()}, "video_id")
        self.save()

    def toggle_saved(self, topic_id: str, title: str) -> bool:
        """Returns True if the topic is saved after the call."""
        saved = self.data["saved"]
        if any(s["id"] == topic_id for s in saved):
            saved[:] = [s for s in saved if s["id"] != topic_id]
            self.save()
            return False
        saved.insert(0, {"id": topic_id, "title": title, "ts": time.time()})
        self.save()
        return True

    def is_saved(self, topic_id: str) -> bool:
        return any(s["id"] == topic_id for s in self.data["saved"])

    def record_answer(self, topic_id: str, question_id: str, correct: bool, score: float = None) -> None:
        q = self.data["quiz"].setdefault(topic_id, {"attempts": 0, "correct": 0, "answered": {}})
        q["attempts"] += 1
        prev = q["answered"].get(question_id)
        if correct and not (prev and prev.get("correct")):
            q["correct"] += 1
        q["answered"][question_id] = {"correct": bool(correct),
                                      "score": None if score is None else round(float(score), 3),
                                      "ts": time.time()}
        self.save()

    def record_math(self, correct: bool) -> None:
        self.data["math"]["attempts"] += 1
        self.data["math"]["correct"] += int(bool(correct))
        self.save()

    def record_playground_run(self) -> None:
        self.data["playground_runs"] += 1
        self.save()

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict:
        quiz = self.data["quiz"]
        answered = sum(len(v["answered"]) for v in quiz.values())
        correct = sum(sum(1 for a in v["answered"].values() if a["correct"]) for v in quiz.values())
        per_topic = {
            tid: {"answered": len(v["answered"]),
                  "correct": sum(1 for a in v["answered"].values() if a["correct"]),
                  "attempts": v["attempts"]}
            for tid, v in quiz.items()
        }
        weak = [tid for tid, v in per_topic.items() if v["answered"] >= 2 and v["correct"] / v["answered"] < 0.6]
        return {
            "topics_studied": len(self.data["studied"]),
            "questions_answered": answered,
            "questions_correct": correct,
            "accuracy": correct / answered if answered else None,
            "per_topic": per_topic,
            "weak_topics": weak,
            "math": self.data["math"],
            "playground_runs": self.data["playground_runs"],
        }
