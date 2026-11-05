import contextlib
import io

import pytest

from clipsage import i18n
from clipsage.knowledge import find_topic, load_topics
from clipsage.lectures import (ask_lecture, explain_range, parse_time_range, segment_references,
                               video_timeline)
from clipsage.lessons import build_lesson
from clipsage.progress import ProgressStore
from clipsage.quiz import cloze_from_lectures, grade, parse_number
from clipsage.retrieve import get_engine


# --------------------------------------------------------------------------- knowledge
def test_all_topics_are_complete_and_bilingual():
    topics = load_topics(refresh=True)
    assert len(topics) >= 20
    for t in topics.values():
        for field in ["title", "intro", "definition", "explanation", "example"]:
            assert t[field]["en"] and t[field]["ru"], (t["id"], field)
        for field in ["applications", "exam_points", "common_mistakes", "key_concepts"]:
            assert t[field], (t["id"], field)
            assert all(x["en"] and x["ru"] for x in t[field]), (t["id"], field)
        assert len(t["questions"]) >= 3, t["id"]
        for q in t["questions"]:
            assert q["q"]["en"] and q["q"]["ru"] and q["hints"] and q["explanation"]["ru"]
            if q["type"] == "mcq":
                assert 0 <= q["answer"] < len(q["options"])


def test_roman_urdu_uses_latin_letters_only():
    topics = load_topics()
    for t in topics.values():
        for field in ["intro", "definition", "explanation", "example"]:
            assert all(ord(c) < 0x0600 or ord(c) > 0x06FF for c in t[field]["ru"]), t["id"]


@pytest.mark.parametrize("tid", [t for t, v in load_topics().items() if not v.get("code_needs_model")])
def test_lesson_code_runs(tid):
    code = load_topics()[tid]["code"]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, {})


def test_find_topic():
    assert find_topic("What is overfitting?")["id"] == "overfitting"
    assert find_topic("k means")["id"] == "kmeans"
    assert find_topic("SVM")["id"] == "svm"
    assert find_topic("photosynthesis in plants") is None


def test_i18n():
    text = {"en": "Hello", "ru": "Salam"}
    assert i18n.pick(text, "en") == "Hello"
    assert i18n.pick(text, "ru") == "Salam"
    assert "Hello" in i18n.pick(text, "both") and "Salam" in i18n.pick(text, "both")
    assert i18n.t("not_covered", "ru", q="x").startswith("“x”")


# --------------------------------------------------------------------------- grading
def test_grading():
    topics = load_topics()
    lr = {q["id"]: q for q in topics["linear_regression"]["questions"]}
    assert grade(lr["linear_regression_q1"], 0)["status"] == "correct"
    assert grade(lr["linear_regression_q1"], 2)["status"] == "incorrect"
    assert grade(lr["linear_regression_q2"], "1.9")["status"] == "correct"
    assert grade(lr["linear_regression_q2"], "19/10")["status"] == "correct"
    assert grade(lr["linear_regression_q2"], "2.5")["status"] == "incorrect"
    assert parse_number("abc") is None and parse_number("50%") == 0.5

    short = lr["linear_regression_q4"]  # R squared 0.9
    good = grade(short, "It means the model explains 90 percent of the variance in y")
    bad = grade(short, "the learning rate is too high")
    assert good["status"] in ("correct", "partial") and good["score"] > bad["score"]
    assert bad["status"] == "incorrect"


def test_progress_store(tmp_path):
    store = ProgressStore(tmp_path / "p.json")
    store.add_search("overfitting")
    store.add_search("kmeans")
    store.add_search("overfitting")
    assert [s["q"] for s in store.data["searches"]] == ["overfitting", "kmeans"]
    assert store.toggle_saved("svm", "SVM") and store.is_saved("svm")
    store.record_answer("svm", "q1", False)
    store.record_answer("svm", "q1", True)
    store.record_answer("svm", "q2", False)
    summary = ProgressStore(tmp_path / "p.json").summary()  # reload from disk
    assert summary["questions_answered"] == 2 and summary["questions_correct"] == 1
    assert not store.toggle_saved("svm", "SVM")


# --------------------------------------------------------------------------- lectures
def test_parse_time_range():
    assert parse_time_range("What is taught between 10:20 and 15:30?") == (620.0, 930.0)
    assert parse_time_range("from 1:02:05 to 0:30") == (30.0, 3725.0)
    assert parse_time_range("no times here") is None


def test_segment_references(built):
    engine = get_engine()
    hits = engine.search("overfitting", method="tfidf")
    refs = segment_references("overfitting", hits, engine, "tfidf")
    top = refs[0]
    assert top["video_id"] == "fake_ml_001"
    assert top["end"] > top["start"] and top["url"].endswith(f"&t={int(top['start'])}s")
    assert top["thumbnail"].startswith("https://i.ytimg.com/vi/fake_ml_001/")
    assert top["excerpt"] and all(s["text"] in top["segment_text"] for s in top["excerpt"])
    assert top["concepts"] and top["reasons"]


def test_timeline_range_and_ask(built):
    engine = get_engine()
    sections = video_timeline(engine, "fake_ml_001", query="regularization", method="tfidf")
    assert len(sections) >= 2 and sections[0]["start"] == 0
    assert all(s["terms"] for s in sections)

    r = explain_range(engine, "fake_ml_001", 30, 60, method="tfidf")
    assert r["ok"] and r["summary"]
    assert all(s["end"] >= 30 and s["start"] <= 60 for s in r["sentences"])
    assert explain_range(engine, "fake_ml_001", 5000, 6000)["reason"] == "out_of_range"

    ans = ask_lecture(engine, "fake_ml_001", "what does dropout do?", method="tfidf")
    assert ans["ok"] and "dropout" in ans["answers"][0]["text"].lower()


def test_lecture_cloze_questions(built):
    engine = get_engine()
    hits = engine.search("deadlock", method="tfidf")
    terms = [("mutual exclusion", 1), ("circular wait", 0.9), ("semaphore", 0.8), ("bankers algorithm", 0.7),
             ("resource allocation graph", 0.6), ("preemption", 0.5)]
    qs = cloze_from_lectures(hits, terms, n=3)
    assert qs
    for q in qs:
        assert "_____" in q["q"]["en"] and len(q["options"]) == 4
        assert q["source"]["url"].startswith("https://www.youtube.com/watch?v=fake_os_001")
        original = q["explanation"]["en"]
        assert q["options"][q["answer"]]["en"].lower() in original.lower()


def test_build_lesson(built):
    lesson = build_lesson("What is deadlock?", method="tfidf")
    assert lesson["notes"]["id"] == "deadlock"
    assert lesson["lecture"]["covered"] and lesson["refs"]
    assert lesson["refs"][0]["video_id"] == "fake_os_001"
    nothing = build_lesson("photosynthesis", method="tfidf")
    assert nothing["notes"] is None and not nothing["lecture"]["covered"]


def test_lesson_explanation_starts_with_definition(built):
    lesson = build_lesson("overfitting", method="tfidf")
    first = lesson["lecture"]["explanation"][0]
    assert first["is_definition"] and first["text"].lower().startswith("overfitting is")
    assert any("definition" in code for code, _ in lesson["refs"][0]["reasons"])
