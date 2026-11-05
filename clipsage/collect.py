"""
Stage 1 - Data collection.

Builds ClipSage's own dataset (proposal section 8):

1. Video metadata from the official YouTube Data API v3
   (channels / playlistItems / search / videos endpoints): video ID, title,
   description, channel, duration, ...
2. Timestamped English transcripts for every selected video.

Everything is cached on disk (data/raw/) so the API quota is only spent once
and never at query time. Re-running the script resumes where it stopped.

Videos are NOT downloaded - only metadata and captions are stored.

Usage:
    python -m clipsage.collect                    # topic video library (2 per lesson) + transcripts
    python -m clipsage.collect --per-topic 3      # more videos per topic
    python -m clipsage.collect --subject "Operating Systems"   # one subject only
    python -m clipsage.collect --skip-metadata    # only fetch missing transcripts
    python -m clipsage.collect --channels         # old mode: scan channels in config/sources.json
"""

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional

from . import config
from .utils import parse_iso_duration, read_json, write_json, youtube_link

VIDEO_FIELDS = [
    "video_id", "title", "channel_id", "channel_title", "subject", "topic_id", "source",
    "duration_seconds", "published_at", "has_captions", "default_audio_language",
    "view_count", "description", "url",
]

# Cost of each endpoint in quota units (YouTube Data API v3 documentation)
COST = {"channels": 1, "playlistItems": 1, "videos": 1, "search": 100}


class QuotaExceeded(RuntimeError):
    pass


class QuotaTracker:
    """Counts quota units so that one run can never burn the whole daily quota."""

    def __init__(self, limit: int = config.DAILY_QUOTA - config.QUOTA_SAFETY_MARGIN):
        self.limit = limit
        self.used = 0

    def spend(self, endpoint: str) -> None:
        cost = COST[endpoint]
        if self.used + cost > self.limit:
            raise QuotaExceeded(f"Quota budget reached ({self.used}/{self.limit} units used).")
        self.used += cost


# --------------------------------------------------------------------------
# Subject assignment
# --------------------------------------------------------------------------
def load_subjects() -> Dict[str, dict]:
    return read_json(config.TOPICS_FILE)["subjects"]


def match_subject(text: str, subjects: Dict[str, dict], allowed: Optional[List[str]] = None) -> Optional[str]:
    """Return the subject whose keywords best match the text (title + description)."""
    text = text.lower()
    best, best_hits = None, 0
    for name, info in subjects.items():
        if allowed and name not in allowed:
            continue
        hits = sum(1 for kw in info["keywords"] if re.search(rf"\b{re.escape(kw.lower())}s?\b", text))
        if hits > best_hits:
            best, best_hits = name, hits
    return best


# --------------------------------------------------------------------------
# YouTube Data API wrapper
# --------------------------------------------------------------------------
class YouTubeCollector:
    def __init__(self, api_key: str, quota: Optional[QuotaTracker] = None):
        from googleapiclient.discovery import build  # imported lazily: only Stage 1 needs it

        self.youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        self.quota = quota or QuotaTracker()

    def resolve_channel(self, handle: str) -> Optional[dict]:
        """@handle -> {channel_id, title, uploads_playlist_id}."""
        self.quota.spend("channels")
        resp = self.youtube.channels().list(
            part="snippet,contentDetails", forHandle=handle.lstrip("@")
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None
        item = items[0]
        return {
            "channel_id": item["id"],
            "title": item["snippet"]["title"],
            "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    def playlist_videos(self, playlist_id: str, max_items: int = 2000) -> List[dict]:
        """All (video_id, title, description) in a playlist, 50 per call (1 unit each)."""
        videos, page_token = [], None
        while len(videos) < max_items:
            self.quota.spend("playlistItems")
            resp = self.youtube.playlistItems().list(
                part="snippet,contentDetails", playlistId=playlist_id,
                maxResults=50, pageToken=page_token,
            ).execute()
            for item in resp.get("items", []):
                snip = item["snippet"]
                videos.append({
                    "video_id": item["contentDetails"]["videoId"],
                    "title": snip.get("title", ""),
                    "description": snip.get("description", ""),
                })
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return videos

    def search_videos(self, query: str, max_results: int = 50) -> List[dict]:
        """search.list costs 100 units, so it is used sparingly (see sources.json)."""
        self.quota.spend("search")
        resp = self.youtube.search().list(
            part="snippet", q=query, type="video", maxResults=min(max_results, 50),
            videoCaption="closedCaption", relevanceLanguage="en",
            videoCategoryId="27",  # Education
            safeSearch="strict",
        ).execute()
        return [
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"].get("title", ""),
                "description": item["snippet"].get("description", ""),
            }
            for item in resp.get("items", [])
        ]

    def video_details(self, video_ids: List[str]) -> Dict[str, dict]:
        """videos.list for up to 50 IDs per call (1 unit each)."""
        details = {}
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            self.quota.spend("videos")
            resp = self.youtube.videos().list(
                part="snippet,contentDetails,statistics", id=",".join(batch)
            ).execute()
            for item in resp.get("items", []):
                snip, content = item["snippet"], item["contentDetails"]
                details[item["id"]] = {
                    "video_id": item["id"],
                    "title": snip.get("title", ""),
                    "channel_id": snip.get("channelId", ""),
                    "channel_title": snip.get("channelTitle", ""),
                    "published_at": snip.get("publishedAt", ""),
                    "description": (snip.get("description", "") or "")[:500].replace("\n", " "),
                    "duration_seconds": parse_iso_duration(content.get("duration", "")),
                    "has_captions": content.get("caption") == "true",
                    "default_audio_language": snip.get("defaultAudioLanguage", "") or snip.get("defaultLanguage", ""),
                    "view_count": int(item.get("statistics", {}).get("viewCount", 0) or 0),
                }
        return details


# --------------------------------------------------------------------------
# Metadata collection
# --------------------------------------------------------------------------
def load_existing_videos() -> Dict[str, dict]:
    if not config.VIDEOS_CSV.exists():
        return {}
    with open(config.VIDEOS_CSV, "r", encoding="utf-8", newline="") as f:
        return {row["video_id"]: row for row in csv.DictReader(f)}


def save_videos(videos: Dict[str, dict]) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.VIDEOS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in videos.values():
            writer.writerow(row)


def is_acceptable(details: dict) -> bool:
    duration = int(details.get("duration_seconds") or 0)
    if not (config.MIN_VIDEO_SECONDS <= duration <= config.MAX_VIDEO_SECONDS):
        return False
    lang = (details.get("default_audio_language") or "").lower()
    return lang == "" or lang.startswith("en")


def collect_metadata(collector: YouTubeCollector, max_total: Optional[int] = None) -> Dict[str, dict]:
    sources = read_json(config.SOURCES_FILE)
    subjects = load_subjects()
    max_total = max_total or sources.get("max_total_videos", 500)
    per_channel = sources.get("max_videos_per_channel", 120)

    videos = load_existing_videos()
    print(f"Already have metadata for {len(videos)} videos.")

    # candidate_id -> (subject, source)
    candidates: Dict[str, tuple] = {}

    def add_candidates(items: List[dict], allowed: List[str], source: str, limit: int) -> int:
        added = 0
        for v in items:
            if added >= limit:
                break
            if v["video_id"] in videos or v["video_id"] in candidates:
                continue
            subject = match_subject(f"{v['title']} {v['description'][:300]}", subjects, allowed or None)
            if subject:
                candidates[v["video_id"]] = (subject, source)
                added += 1
        return added

    try:
        # 1) channel uploads (cheap: 1 unit per 50 videos)
        for ch in sources.get("channels", []):
            info = collector.resolve_channel(ch["handle"])
            if not info:
                print(f"  ! channel {ch['handle']} not found - check the handle in config/sources.json")
                continue
            items = collector.playlist_videos(info["uploads_playlist_id"])
            n = add_candidates(items, ch.get("subjects", []), f"channel:{ch['handle']}", per_channel)
            print(f"  {ch['handle']}: scanned {len(items)} uploads, {n} match the subjects")

        # 2) hand-picked playlists (lecture series)
        for pl in sources.get("playlists", []):
            if not pl.get("id"):
                continue
            items = collector.playlist_videos(pl["id"])
            n = add_candidates(items, pl.get("subjects", []), f"playlist:{pl['id']}", per_channel)
            print(f"  playlist {pl['id']}: {len(items)} videos, {n} match")

        # 3) a few topic searches (expensive: 100 units each)
        if sources.get("use_search", False):
            calls = 0
            for name, info in subjects.items():
                for q in info.get("search_queries", []):
                    if calls >= sources.get("max_search_calls", 0):
                        break
                    items = collector.search_videos(q)
                    calls += 1
                    n = add_candidates(items, [name], f"search:{q}", 50)
                    print(f"  search '{q}': {n} new candidates")
    except QuotaExceeded as e:
        print(f"  ! {e} Continuing with what was collected so far.")

    print(f"{len(candidates)} new candidate videos; fetching details ...")
    try:
        details = collector.video_details(list(candidates))
    except QuotaExceeded as e:
        print(f"  ! {e}")
        details = {}

    # Keep the subjects balanced when the total is capped: take videos
    # round-robin across subjects, most-viewed first inside each subject.
    queues: Dict[str, List[dict]] = {}
    for d in details.values():
        if is_acceptable(d):
            queues.setdefault(candidates[d["video_id"]][0], []).append(d)
    for q in queues.values():
        q.sort(key=lambda d: -int(d.get("view_count") or 0))
    budget = max(0, max_total - len(videos))
    accepted = 0
    while accepted < budget and any(queues.values()):
        for subject in list(queues):
            if accepted >= budget or not queues[subject]:
                continue
            d = queues[subject].pop(0)
            d.update(subject=subject, source=candidates[d["video_id"]][1], url=youtube_link(d["video_id"]))
            videos[d["video_id"]] = d
            accepted += 1

    save_videos(videos)
    print(f"Metadata saved: {len(videos)} videos in {config.VIDEOS_CSV} "
          f"(quota used this run: {collector.quota.used} units)")
    return videos


# --------------------------------------------------------------------------
# Topic-based video library (one YouTube search per lesson topic)
# --------------------------------------------------------------------------
LIBRARY_CACHE = "library_searches.json"


def _relevant(video: dict, topic: dict) -> bool:
    """The title or description must mention the topic (title words or an alias)."""
    from .text import contains_term

    text = f"{video.get('title', '')} {video.get('description', '')[:400]}".lower()
    names = [topic["title"]["en"]] + topic.get("aliases", [])
    if any(contains_term(text, n.lower()) for n in names):
        return True
    words = [w for w in re.findall(r"[a-z0-9']+", topic["title"]["en"].lower()) if len(w) > 3]
    return sum(1 for w in words if w in text) >= max(1, len(words) // 2)


def collect_library(collector: "YouTubeCollector", per_topic: int = 2, subjects: Optional[List[str]] = None,
                    topics: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """
    For every lesson topic, search YouTube with its `video_queries` and keep the
    best `per_topic` videos that (1) have captions, (2) are English, (3) are
    3 min - 3 h long, (4) mention the topic, and (5) are not already used by
    another topic. Search results are cached so that a run interrupted by the
    daily quota continues the next day without repeating searches.
    """
    from .knowledge import load_topics

    topics = topics or load_topics()
    videos = load_existing_videos()
    cache_path = config.RAW_DIR / LIBRARY_CACHE
    cache = read_json(cache_path) if cache_path.exists() else {}
    used = set(videos)
    have = {}
    for v in videos.values():
        if v.get("topic_id"):
            have[v["topic_id"]] = have.get(v["topic_id"], 0) + 1

    try:
        for topic in topics.values():
            if subjects and topic["subject"] not in subjects:
                continue
            need = per_topic - have.get(topic["id"], 0)
            for query in topic.get("video_queries", []):
                if need <= 0:
                    break
                if query not in cache:
                    cache[query] = collector.search_videos(query, max_results=15)
                    write_json(cache_path, cache)
                candidates = [c for c in cache[query] if c["video_id"] not in used and _relevant(c, topic)]
                if not candidates:
                    continue
                details = collector.video_details([c["video_id"] for c in candidates])
                for c in candidates:  # keep YouTube's relevance order
                    d = details.get(c["video_id"])
                    if need <= 0 or not d or not d.get("has_captions") or not is_acceptable(d):
                        continue
                    d.update(subject=topic["subject"], topic_id=topic["id"], source=f"library:{query}",
                             url=youtube_link(d["video_id"]))
                    videos[d["video_id"]] = d
                    used.add(d["video_id"])
                    need -= 1
                    have[topic["id"]] = have.get(topic["id"], 0) + 1
            print(f"  {topic['subject']:22} {topic['id']:26} {have.get(topic['id'], 0)} video(s)")
    except QuotaExceeded as e:
        print(f"  ! {e} Run the same command again tomorrow - finished searches are cached.")

    save_videos(videos)
    per_subject: Dict[str, int] = {}
    for v in videos.values():
        if v.get("topic_id"):
            per_subject[v["subject"]] = per_subject.get(v["subject"], 0) + 1
    print(f"Library: {sum(per_subject.values())} topic videos " +
          ", ".join(f"{k}: {n}" for k, n in sorted(per_subject.items())) +
          f" (quota used this run: {collector.quota.used} units)")
    return videos


# --------------------------------------------------------------------------
# Transcript collection
# --------------------------------------------------------------------------
def fetch_transcript(video_id: str, api=None) -> dict:
    """Fetch a timestamped English transcript. Prefers human captions over auto ones."""
    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

    api = api or YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    try:
        transcript = transcript_list.find_manually_created_transcript(config.TRANSCRIPT_LANGUAGES)
        kind = "manual"
    except NoTranscriptFound:
        transcript = transcript_list.find_generated_transcript(config.TRANSCRIPT_LANGUAGES)
        kind = "auto"
    fetched = transcript.fetch()
    snippets = [
        {"text": s.text, "start": round(float(s.start), 2), "duration": round(float(s.duration), 2)}
        for s in fetched
    ]
    return {"video_id": video_id, "language": transcript.language_code, "kind": kind, "snippets": snippets}


def collect_transcripts(videos: Dict[str, dict]) -> None:
    from youtube_transcript_api import (
        IpBlocked, NoTranscriptFound, RequestBlocked, TranscriptsDisabled,
        VideoUnavailable, YouTubeTranscriptApi,
    )

    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    api = YouTubeTranscriptApi()
    log_rows = []
    todo = [vid for vid in videos if not (config.TRANSCRIPT_DIR / f"{vid}.json").exists()]
    print(f"Transcripts: {len(videos) - len(todo)} cached, {len(todo)} to fetch.")

    for i, vid in enumerate(todo, 1):
        status = "ok"
        try:
            data = fetch_transcript(vid, api)
            if len(data["snippets"]) < 10:
                status = "too_short"
            else:
                write_json(config.TRANSCRIPT_DIR / f"{vid}.json", data)
                status = f"ok_{data['kind']}"
        except (TranscriptsDisabled, NoTranscriptFound):
            status = "no_english_transcript"
        except VideoUnavailable:
            status = "video_unavailable"
        except (IpBlocked, RequestBlocked):
            print("\n! YouTube is blocking transcript requests from this network (too many requests,"
                  " or a cloud/VPN IP). Wait a while or run from a home connection, then re-run with"
                  " --skip-metadata to continue.")
            break
        except Exception as e:  # keep going - one broken video must not stop the run
            status = f"error:{type(e).__name__}"
        log_rows.append({"video_id": vid, "status": status})
        print(f"  [{i}/{len(todo)}] {vid}: {status}")
        time.sleep(config.TRANSCRIPT_DELAY_SECONDS)

    if log_rows:
        new_file = not config.COLLECTION_LOG.exists()
        with open(config.COLLECTION_LOG, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "status"])
            if new_file:
                writer.writeheader()
            writer.writerows(log_rows)

    have = sum(1 for vid in videos if (config.TRANSCRIPT_DIR / f"{vid}.json").exists())
    print(f"Done. {have}/{len(videos)} videos have a usable English transcript.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClipSage Stage 1 - collect videos and transcripts")
    parser.add_argument("--max-videos", type=int, default=None, help="cap on total videos")
    parser.add_argument("--skip-metadata", action="store_true", help="only fetch missing transcripts")
    parser.add_argument("--skip-transcripts", action="store_true", help="only collect metadata")
    parser.add_argument("--channels", action="store_true",
                        help="old mode: scan the channels in config/sources.json instead of the topic library")
    parser.add_argument("--per-topic", type=int, default=2, help="library videos per lesson topic (default 2)")
    parser.add_argument("--subject", action="append", help="only this subject (repeatable)")
    args = parser.parse_args(argv)

    if args.skip_metadata:
        videos = load_existing_videos()
        if not videos:
            sys.exit(f"No {config.VIDEOS_CSV} yet - run without --skip-metadata first.")
    else:
        api_key = os.environ.get(config.YOUTUBE_API_KEY_ENV)
        if not api_key:
            sys.exit(f"Set {config.YOUTUBE_API_KEY_ENV} in a .env file in the project folder "
                     f"(see .env.example) or as an environment variable.")
        collector = YouTubeCollector(api_key)
        if args.channels:
            videos = collect_metadata(collector, args.max_videos)
        else:
            videos = collect_library(collector, per_topic=args.per_topic, subjects=args.subject)

    if not args.skip_transcripts:
        collect_transcripts(videos)


if __name__ == "__main__":
    main()
