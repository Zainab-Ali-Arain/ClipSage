"""Six subjects, language detection, query analysis and numerical practice generators."""

import pytest

from clipsage import practice_gen as G
from clipsage.langdetect import _training_texts, detect_language, detect_with_details
from clipsage.search import analyze_query, classify_subject
from clipsage.subjects import get_subject, load_subjects, topics_of

SIX = ["Operating Systems", "Machine Learning", "Linear Algebra", "Compiler Construction", "Web Engineering",
       "Computer Architecture"]


def test_exactly_six_subjects_each_with_content():
    assert [s["name"] for s in load_subjects()] == SIX
    for name in SIX:
        topics = topics_of(name)
        assert len(topics) >= 5, name
        assert sum(len(t["questions"]) for t in topics) >= 15, name
        assert all(t["subject"] == name for t in topics)
        assert get_subject(name)["tools"], name
        kinds = {q["type"] for t in topics for q in t["questions"]}
        assert {"mcq", "tf"} <= kinds, name
        assert {q.get("difficulty") for t in topics for q in t["questions"]} <= {"easy", "medium", "hard"}


@pytest.mark.parametrize("query,lang", [
    ("Process aur thread mein kya difference hai?", "ru"),
    ("paging kya hai", "ru"),
    ("deadlock ki conditions batao", "ru"),
    ("overfitting kaise kam karte hain", "ru"),
    ("cache memory samjhao", "ru"),
    ("Explain karo FCFS scheduling", "ru"),
    ("Mujhe binary search tree samjha dein", "ru"),
    ("Explain paging in operating systems", "en"),
    ("what is a deadlock", "en"),
    ("how does gradient descent work", "en"),
    ("difference between process and thread", "en"),
    ("paging", "en"),
    ("round robin scheduling", "en"),
])
def test_language_detection(query, lang):
    assert detect_language(query) == lang, detect_with_details(query)


def test_language_classifier_accuracy():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    X, y = _training_texts()
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
                          MultinomialNB(alpha=0.1))
    assert cross_val_score(model, X, y, cv=5).mean() > 0.9


@pytest.mark.parametrize("query,subject,topic", [
    ("Process aur thread mein kya difference hai?", "Operating Systems", "processes_threads"),
    ("Explain paging in operating systems", "Operating Systems", "paging_virtual_memory"),
    ("round robin scheduling", "Operating Systems", "cpu_scheduling"),
    ("eigenvalues nikalne ka tareeqa", "Linear Algebra", "eigenvalues"),
    ("gaussian elimination", "Linear Algebra", "systems_of_equations"),
    ("LL1 parsing table kaise banta hai", "Compiler Construction", "ll1_parsing"),
    ("first and follow", "Compiler Construction", "ll1_parsing"),
    ("CSS box model", "Web Engineering", "css_layout"),
    ("rest api kya hai", "Web Engineering", "rest_api"),
    ("two's complement", "Computer Architecture", "number_representation"),
    ("cache memory samjhao", "Computer Architecture", "cache_memory"),
    ("k means clustering", "Machine Learning", "kmeans"),
])
def test_query_analysis(query, subject, topic):
    a = analyze_query(query)
    assert a["subject"] == subject and a["topic"]["id"] == topic


def test_unknown_topics_are_not_forced_into_a_subject():
    assert analyze_query("photosynthesis in plants")["topic"] is None
    assert classify_subject("photosynthesis")["subject"] is None


def test_subject_restricted_search():
    a = analyze_query("matrix", subject="Linear Algebra")
    assert all(m["topic"]["subject"] == "Linear Algebra" for m in a["matches"])


@pytest.mark.parametrize("subject", SIX)
def test_numerical_practice_generators(subject):
    for kind in G.SUBJECT_KINDS[subject]:
        for seed in range(5):
            p = G.generate(kind, seed)
            assert p["q"]["en"] and p["q"]["ru"] and p["steps"]
            assert G.check(p, repr(float(p["answer"])))["status"] == "correct"
            assert G.check(p, str(float(p["answer"]) + 7))["status"] == "incorrect"
