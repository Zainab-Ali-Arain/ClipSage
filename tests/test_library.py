"""Video library collection with a FAKE YouTube client (no network, no real video ids)."""

import pytest

from clipsage import collect, config
from clipsage.knowledge import load_topics


class FakeQuota:
    used = 0


class FakeCollector:
    """Mimics YouTubeCollector.search_videos / video_details with invented test data."""

    def __init__(self):
        self.quota = FakeQuota()
        self.searches = []

    def search_videos(self, query, max_results=15):
        self.searches.append(query)
        slug = query.split()[0].lower()
        return [
            {"video_id": f"{slug}_good1", "title": f"{query} lecture", "description": ""},
            {"video_id": f"{slug}_nocaps", "title": f"{query} explained", "description": ""},
            {"video_id": f"{slug}_offtopic", "title": "My holiday vlog", "description": "travel"},
            {"video_id": f"{slug}_good2", "title": f"{query} tutorial", "description": ""},
            {"video_id": "shared_video", "title": f"{query} full course", "description": ""},
        ]

    def video_details(self, ids):
        out = {}
        for vid in ids:
            out[vid] = {"video_id": vid, "title": vid, "channel_id": "UC_test", "channel_title": "Test Channel",
                        "published_at": "", "description": "", "duration_seconds": 900,
                        "has_captions": not vid.endswith("nocaps"), "default_audio_language": "en", "view_count": 1}
        return out


@pytest.fixture()
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "VIDEOS_CSV", tmp_path / "raw" / "videos.csv")
    monkeypatch.setattr(config, "TRANSCRIPT_DIR", tmp_path / "raw" / "transcripts")
    return tmp_path


def test_library_keeps_relevant_captioned_unique_videos(raw_dir):
    topics = {k: v for k, v in load_topics().items() if k in ("cpu_scheduling", "page_replacement")}
    fake = FakeCollector()
    videos = collect.collect_library(fake, per_topic=2, topics=topics)
    by_topic = {}
    for v in videos.values():
        by_topic.setdefault(v["topic_id"], []).append(v["video_id"])
    assert set(by_topic) == {"cpu_scheduling", "page_replacement"}
    assert all(len(ids) == 2 for ids in by_topic.values())
    all_ids = [i for ids in by_topic.values() for i in ids]
    assert len(all_ids) == len(set(all_ids))                      # no duplicates across topics
    assert not any(i.endswith(("nocaps", "offtopic")) for i in all_ids)
    assert all(v["subject"] == "Operating Systems" for v in videos.values())


def test_library_search_cache_avoids_repeat_quota(raw_dir):
    topics = {"cpu_scheduling": load_topics()["cpu_scheduling"]}
    first = FakeCollector()
    collect.collect_library(first, per_topic=1, topics=topics)
    assert first.searches
    # clear saved videos but keep the cached search results
    config.VIDEOS_CSV.unlink()
    second = FakeCollector()
    collect.collect_library(second, per_topic=1, topics=topics)
    assert second.searches == []


def test_every_lesson_has_video_queries_and_subtopics():
    for t in load_topics().values():
        assert t.get("video_queries"), t["id"]
        assert t.get("subtopics") is not None, t["id"]


def test_video_panel_uses_real_topic_videos_and_caption_timestamps(built):
    from clipsage.library import is_about_topic, videos_for_topic
    from clipsage.retrieve import get_engine

    topics = load_topics()
    engine = get_engine()
    panels = videos_for_topic(topics["deadlock"], engine, "tfidf")
    assert panels and panels[0]["video_id"] == "fake_os_001" and panels[0]["source"] == "topic library"
    covered = {s["name"]: s for s in panels[0]["covered"]}
    assert "Banker's algorithm" in covered
    sentence = covered["Banker's algorithm"]["sentence"].lower()
    assert "bankers" in sentence or "banker" in sentence
    assert panels[0]["watch_url"].endswith(f"&t={int(panels[0]['start'])}s")
    # a deadlock lecture says 'process' a lot, but it is NOT a video about processes and threads
    assert not is_about_topic(engine, "fake_os_001", topics["processes_threads"])
    assert videos_for_topic(topics["processes_threads"], engine, "tfidf") == []
    assert videos_for_topic(topics["eigenvalues"], engine, "tfidf") == []
