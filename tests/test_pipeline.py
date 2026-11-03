import pytest

from clipsage import evaluate
from clipsage.collect import match_subject
from clipsage.pipeline import ClipSage
from clipsage.retrieve import get_engine
from clipsage.utils import read_jsonl
from clipsage import config


def test_preprocess_and_index(built):
    segments = list(read_jsonl(config.SEGMENTS_FILE))
    assert segments
    assert all(s["sentences"] and s["end"] > s["start"] for s in segments)
    assert (config.INDEX_DIR / "tfidf.pkl").exists()
    assert (config.INDEX_DIR / "sbert_embeddings.npy").exists()


@pytest.mark.parametrize("method", ["tfidf", "sbert"])
def test_retrieval_finds_right_video(built, method):
    engine = get_engine()
    for query, expected in [("overfitting", "fake_ml_001"), ("binary search tree", "fake_ds_001"),
                            ("deadlock conditions", "fake_os_001"), ("tcp handshake", "fake_net_001")]:
        hits = engine.search(query, method=method)
        assert hits[0]["video_id"] == expected, (method, query)


def test_diversity_cap(built, monkeypatch):
    monkeypatch.setattr(config, "MIN_RELATIVE_SCORE", 0.0)  # test the re-ranking on its own
    hits = get_engine().search("regularization to avoid overfitting in gradient descent", method="tfidf", top_k=10, max_per_video=1)
    vids = [h["video_id"] for h in hits]
    distinct = len(set(vids))
    assert distinct >= 2
    # different videos come first; a video repeats only once the others are used up
    assert len(set(vids[:distinct])) == distinct


@pytest.mark.parametrize("summarizer", ["textrank", "embedding", "top_sentences"])
def test_answer_has_cited_explanation_and_visuals(built, summarizer):
    sage = ClipSage()
    sage.ask("deadlock", method="sbert")  # warm-up: the first call pays one-off import costs
    r = sage.ask("What is overfitting?", method="sbert", summarizer=summarizer)
    assert r["covered"]
    assert 1 <= len(r["explanation"]) <= config.SUMMARY_MAX_SENTENCES
    for s in r["explanation"]:
        assert s["url"].startswith("https://www.youtube.com/watch?v=") and "&t=" in s["url"]
        assert "subscribe" not in s["text"].lower()
    if summarizer != "top_sentences":
        assert r["explanation"][0]["is_definition"]
    assert r["references"][0]["video_id"] == "fake_ml_001"
    assert r["key_terms"]
    g = r["concept_graph"]
    assert "overfitting" in g and g.number_of_nodes() > 3
    assert r["timings"]["total"] < 5


def test_not_covered_topic_gets_suggestions(built):
    r = ClipSage().ask("photosynthesis in plants", method="tfidf")
    assert not r["covered"]
    assert r["suggestions"]


def test_visual_outputs(built):
    from clipsage.visualize import concept_map_figure, concept_map_html, keyword_cloud, timeline_figure

    r = ClipSage().ask("deadlock", method="sbert", keyword_method="keybert")
    assert "vis-network" in concept_map_html(r["concept_graph"]) or "vis" in concept_map_html(r["concept_graph"])
    assert keyword_cloud(r["key_terms"]) is not None
    assert timeline_figure(r["hits"]).data
    concept_map_figure(r["concept_graph"])


def test_metrics():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert evaluate.precision_at_k(ranked, {"b", "e", "z"}, 5) == pytest.approx(0.4)
    assert evaluate.recall_at_k(ranked, {"b", "e", "z"}, 10) == pytest.approx(2 / 3)
    assert evaluate.reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert evaluate.reciprocal_rank(ranked, {"x"}) == 0


def test_evaluation_commands(built, tmp_path, monkeypatch):
    import json

    queries = {"queries": [
        {"id": 1, "query": "overfitting", "relevant_videos": ["fake_ml_001"]},
        {"id": 2, "query": "binary search tree", "relevant_videos": ["fake_ds_001"]},
        {"id": 3, "query": "deadlock", "relevant_videos": ["fake_os_001"]},
    ]}
    qfile = tmp_path / "queries.json"
    qfile.write_text(json.dumps(queries))
    monkeypatch.setattr(evaluate, "QUERIES_FILE", qfile)
    monkeypatch.setattr(evaluate, "LABELS_FILE", tmp_path / "labels.csv")
    monkeypatch.setattr(evaluate, "RATINGS_FILE", tmp_path / "ratings.csv")
    monkeypatch.setattr(evaluate, "KEYWORD_SHEET", tmp_path / "kw.csv")
    monkeypatch.setattr(evaluate, "SURVEY_FILE", tmp_path / "survey.csv")
    monkeypatch.setattr(evaluate, "OUT", config.REPORTS_DIR)

    evaluate.pool()
    assert (tmp_path / "labels.csv").exists()
    res = evaluate.evaluate_retrieval()
    assert res["methods"]["sbert"]["MRR"] == pytest.approx(1.0)
    assert evaluate.evaluate_sources()["accuracy"] == pytest.approx(1.0)
    assert evaluate.evaluate_summaries()["rouge_f1"]
    evaluate.keyword_sheet()
    assert evaluate.evaluate_speed()["sbert"]["under_5s"] == 1.0
    text = evaluate.report().read_text()
    assert "Precision@5" in text and "ROUGE" in text


def test_subject_matching():
    subjects = {"OS": {"keywords": ["process", "deadlock"]}, "DB": {"keywords": ["sql"]}}
    assert match_subject("Deadlock in operating systems", subjects) == "OS"
    assert match_subject("Image processing tutorial", subjects) is None
    assert match_subject("SQL joins", subjects, allowed=["OS"]) is None
