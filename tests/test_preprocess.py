from clipsage import config
from clipsage.preprocess import clean_snippets, clean_text, make_segments, split_sentences
from clipsage.utils import parse_iso_duration


def test_clean_text_removes_noise_fillers_and_repeats():
    assert clean_text("[Music] um so the the model &amp; data") == "so the model & data"
    assert clean_text("uh this is is a test") == "this is a test"
    assert clean_text(">> [Applause]") == ""


def test_clean_snippets_drops_duplicates_and_rolling_repeats():
    snippets = [
        {"text": "hello world", "start": 0, "duration": 2},
        {"text": "hello world", "start": 2, "duration": 2},
        {"text": "hello world this is new", "start": 4, "duration": 2},
    ]
    cleaned = clean_snippets(snippets)
    assert [s["text"] for s in cleaned] == ["hello world", "this is new"]


def test_split_sentences_keeps_timestamps():
    snippets = [
        {"text": "A stack is a", "start": 10.0, "duration": 2.0},
        {"text": "LIFO structure. A queue is FIFO.", "start": 12.0, "duration": 4.0},
    ]
    sents = split_sentences(snippets)
    assert sents[0]["text"] == "A stack is a LIFO structure."
    assert sents[0]["start"] == 10.0
    assert sents[1]["text"] == "A queue is FIFO."
    assert 12.0 < sents[1]["start"] < 16.0


def test_unpunctuated_captions_are_cut_by_time():
    snippets = [{"text": f"word{i} more words here", "start": i * 3.0, "duration": 3.0} for i in range(20)]
    sents = split_sentences(snippets)
    assert len(sents) > 1
    assert all(s["end"] - s["start"] <= config.MAX_SENTENCE_SECONDS + 3 for s in sents)


def test_segments_are_30_to_60_seconds_and_overlap():
    sentences = [{"text": f"sentence {i} with words", "start": i * 6.0, "end": i * 6.0 + 6.0} for i in range(50)]
    windows = make_segments(sentences)
    lengths = [w[-1]["end"] - w[0]["start"] for w in windows]
    assert all(config.SEGMENT_MIN_SECONDS <= L <= config.SEGMENT_MAX_SECONDS + 15 for L in lengths)
    assert all(config.SEGMENT_MIN_SECONDS <= L <= config.SEGMENT_MAX_SECONDS for L in lengths[:-1])
    # consecutive windows share some sentences (overlap)
    for a, b in zip(windows, windows[1:]):
        assert b[0]["start"] < a[-1]["end"]
    # every sentence is covered
    covered = {s["start"] for w in windows for s in w}
    assert covered == {s["start"] for s in sentences}


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H2M10S") == 3730
    assert parse_iso_duration("PT45S") == 45
    assert parse_iso_duration("P1DT1S") == 86401
    assert parse_iso_duration("bad") == 0
