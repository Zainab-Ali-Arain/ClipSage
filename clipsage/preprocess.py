"""
Stage 2 - Preprocessing.

For every collected transcript:
  1. clean the caption text (noise tags such as "[Music]", filler words,
     repeated words/lines, HTML entities, extra spaces);
  2. split it into timestamped sentences;
  3. group sentences into overlapping segments of about 30-60 seconds.

Each segment keeps its video ID, start/end time and the timestamped sentences
it is made of, so every sentence of an explanation can later be traced back
to the exact second of the original video.

Usage:
    python -m clipsage.preprocess
"""

import csv
import html
import re
from typing import Dict, List

from . import config
from .collect import load_existing_videos
from .utils import read_json, write_jsonl

# "[Music]", "[Applause]", "(laughter)", "♪", ">>" speaker markers ...
NOISE_PATTERN = re.compile(r"\[[^\]]*\]|\([^)]*(?:music|applause|laugh|inaudible|silence)[^)]*\)|[♪♫]+|>>+",
                           re.IGNORECASE)
FILLER_PATTERN = re.compile(r"\b(?:um+|uh+m*|erm+|hmm+|mhm+|ah+|you know,|i mean,)(?=\W|$)[,.]?\s*",
                            re.IGNORECASE)
REPEATED_WORDS = re.compile(r"\b(\w+(?:\s+\w+){0,2})(?:\s+\1\b)+", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    text = html.unescape(text).replace("\n", " ")
    text = NOISE_PATTERN.sub(" ", text)
    text = FILLER_PATTERN.sub("", text)
    text = REPEATED_WORDS.sub(r"\1", text)     # "the the the" -> "the"
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_snippets(snippets: List[dict]) -> List[dict]:
    """Clean each caption line and drop empty or duplicated lines."""
    cleaned, previous = [], ""
    for s in snippets:
        text = clean_text(s["text"])
        if not text or text.lower() == previous.lower():
            continue
        # Rolling auto-captions often repeat the end of the previous line.
        if previous and text.lower().startswith(previous.lower()):
            text = text[len(previous):].strip()
            if not text:
                continue
        cleaned.append({"text": text, "start": float(s["start"]), "duration": float(s.get("duration", 0))})
        previous = text
    return cleaned


def split_sentences(snippets: List[dict]) -> List[dict]:
    """
    Build timestamped sentences from caption lines.

    Captions with punctuation are split at . ! ?  Auto-generated captions often
    have no punctuation at all, so a "sentence" is also closed after
    MAX_SENTENCE_SECONDS or MAX_SENTENCE_WORDS.
    """
    sentences: List[dict] = []
    buf: List[str] = []
    buf_start = buf_end = 0.0

    def flush():
        nonlocal buf
        text = " ".join(buf).strip()
        if len(text.split()) >= 3:
            text = text[0].upper() + text[1:]
            sentences.append({"text": text, "start": round(buf_start, 2), "end": round(buf_end, 2)})
        buf = []

    for s in snippets:
        pieces = [p for p in SENTENCE_END.split(s["text"]) if p.strip()]
        total_chars = max(1, sum(len(p) for p in pieces))
        offset = 0
        for piece in pieces:
            # interpolate a timestamp for pieces that start inside a caption line
            piece_start = s["start"] + s["duration"] * offset / total_chars
            offset += len(piece)
            piece_end = s["start"] + s["duration"] * offset / total_chars
            if not buf:
                buf_start = piece_start
            buf.append(piece.strip())
            buf_end = piece_end
            n_words = sum(len(b.split()) for b in buf)
            if (piece.rstrip()[-1:] in ".!?"
                    or buf_end - buf_start >= config.MAX_SENTENCE_SECONDS
                    or n_words >= config.MAX_SENTENCE_WORDS):
                flush()
    if buf:
        flush()
    return sentences


def make_segments(sentences: List[dict]) -> List[List[dict]]:
    """
    Group sentences into overlapping windows of ~SEGMENT_TARGET_SECONDS
    (between SEGMENT_MIN_SECONDS and SEGMENT_MAX_SECONDS). The next window
    starts SEGMENT_OVERLAP_SECONDS before the current one ends.
    """
    windows, i, n = [], 0, len(sentences)
    while i < n:
        start = sentences[i]["start"]
        j = i
        while j + 1 < n:
            duration_if_added = sentences[j + 1]["end"] - start
            if sentences[j]["end"] - start >= config.SEGMENT_TARGET_SECONDS:
                break
            if duration_if_added > config.SEGMENT_MAX_SECONDS:
                break
            j += 1
        window = sentences[i:j + 1]

        # A short tail is merged into the previous window instead of standing alone.
        if (windows and j == n - 1
                and window[-1]["end"] - window[0]["start"] < config.SEGMENT_MIN_SECONDS):
            last = windows[-1]
            seen = {(x["start"], x["text"]) for x in last}
            last.extend(x for x in window if (x["start"], x["text"]) not in seen)
            break

        windows.append(list(window))
        if j == n - 1:
            break
        # next window starts SEGMENT_OVERLAP_SECONDS before this one ends
        next_start_time = window[-1]["end"] - config.SEGMENT_OVERLAP_SECONDS
        k = i + 1
        while k <= j and sentences[k]["start"] < next_start_time:
            k += 1
        i = k if k > i else i + 1
    return windows


def process_video(video: dict, transcript: dict) -> List[dict]:
    sentences = split_sentences(clean_snippets(transcript["snippets"]))
    segments = []
    for k, window in enumerate(make_segments(sentences)):
        segments.append({
            "segment_id": f"{video['video_id']}_{k:04d}",
            "video_id": video["video_id"],
            "title": video.get("title", ""),
            "channel_title": video.get("channel_title", ""),
            "subject": video.get("subject", ""),
            "transcript_kind": transcript.get("kind", ""),
            "start": window[0]["start"],
            "end": window[-1]["end"],
            "text": " ".join(s["text"] for s in window),
            "sentences": window,
        })
    return segments


def run() -> Dict[str, int]:
    videos = load_existing_videos()
    all_segments: List[dict] = []
    all_sentences: List[dict] = []
    used_videos = 0
    for vid, video in videos.items():
        path = config.TRANSCRIPT_DIR / f"{vid}.json"
        if not path.exists():
            continue
        transcript = read_json(path)
        segments = process_video(video, transcript)
        if not segments:
            continue
        used_videos += 1
        all_segments.extend(segments)
        seen = set()
        for seg in segments:
            for s in seg["sentences"]:
                key = (s["start"], s["text"])
                if key not in seen:
                    seen.add(key)
                    all_sentences.append({"video_id": vid, **s})

    write_jsonl(config.SEGMENTS_FILE, all_segments)
    write_jsonl(config.SENTENCES_FILE, all_sentences)
    with open(config.SEGMENTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "video_id", "subject", "title", "start", "end", "text"])
        for s in all_segments:
            writer.writerow([s["segment_id"], s["video_id"], s["subject"], s["title"], s["start"], s["end"], s["text"]])

    stats = {"videos": used_videos, "sentences": len(all_sentences), "segments": len(all_segments)}
    print(f"Preprocessed {stats['videos']} videos -> {stats['sentences']} sentences, "
          f"{stats['segments']} segments")
    print(f"Saved {config.SEGMENTS_FILE} and {config.SEGMENTS_CSV}")
    return stats


def main():
    if not config.VIDEOS_CSV.exists():
        raise SystemExit("No data yet - run `python -m clipsage.collect` first.")
    run()


if __name__ == "__main__":
    main()
