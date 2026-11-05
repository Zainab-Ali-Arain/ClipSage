"""Subject chat replies and the practice quiz built from the discussed topics."""

import pytest

from clipsage.chat import TOPIC_KINDS, discussed_topics, practice_quiz, respond
from clipsage.knowledge import load_topics
from clipsage.quiz import grade


@pytest.mark.parametrize("query,subject,kind,topic,lang", [
    ("deadlock kya hota hai?", "Operating Systems", "topic", "deadlock", "ru"),
    ("explain round robin scheduling", "Operating Systems", "topic", "cpu_scheduling", "en"),
    ("LL1 table kaise banta hai", "Compiler Construction", "topic", "ll1_parsing", "ru"),
    ("what is gradient descent", "Operating Systems", "other_subject", "gradient_descent", "en"),
    ("salam", "Linear Algebra", "greeting", None, "ru"),
    ("hello", "Linear Algebra", "greeting", None, "en"),
    ("photosynthesis in plants", "Operating Systems", "open", None, "en"),
])
def test_respond(query, subject, kind, topic, lang):
    r = respond(query, subject)
    assert (r["kind"], r["topic_id"], r["lang"]) == (kind, topic, lang)


def test_discussed_topics_only_this_subject_without_repeats():
    msgs = [{"reply": respond(q, "Operating Systems")} for q in
            ["deadlock", "paging kya hai", "deadlock ki conditions", "what is gradient descent"]]
    assert discussed_topics(msgs, "Operating Systems") == ["deadlock", "paging_virtual_memory"]


def test_practice_quiz_covers_every_topic_and_answers_are_correct():
    topics = load_topics()
    assert all(t in topics for t in TOPIC_KINDS)
    qs = practice_quiz(["cpu_scheduling", "deadlock", "cache_memory"], seed=3, n=9)
    assert len(qs) == 9 and len({q["id"] for q in qs}) == 9
    owner = {q["id"]: tid for tid in topics for q in topics[tid]["questions"]}
    covered = {owner.get(q["id"]) for q in qs} | {"cpu_scheduling" for q in qs if q["id"].startswith("gen_os_")}
    assert {"cpu_scheduling", "deadlock", "cache_memory"} <= covered
    for q in qs:
        if q.get("generated"):
            assert grade(q, repr(q["answer"]))["status"] == "correct"


def test_practice_quiz_difficulty_filter():
    qs = practice_quiz(["deadlock", "pipelining"], seed=1, n=20, levels=["easy"], numerical=False)
    assert qs and all(q["difficulty"] == "easy" for q in qs)
