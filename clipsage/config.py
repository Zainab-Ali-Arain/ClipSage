"""
Central configuration for ClipSage.

Every path and every tunable number lives here so that the team can change
behaviour in one place (and quote these values in the report).
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
# CLIPSAGE_DATA_DIR lets you keep several datasets (or a demo) side by side.
DATA_DIR = Path(os.environ.get("CLIPSAGE_DATA_DIR", ROOT_DIR / "data"))

RAW_DIR = DATA_DIR / "raw"
TRANSCRIPT_DIR = RAW_DIR / "transcripts"          # one JSON per video
VIDEOS_CSV = RAW_DIR / "videos.csv"               # video metadata (Stage 1)
COLLECTION_LOG = RAW_DIR / "collection_log.csv"   # transcript status per video

PROCESSED_DIR = DATA_DIR / "processed"
SENTENCES_FILE = PROCESSED_DIR / "sentences.jsonl"  # cleaned timestamped sentences
SEGMENTS_FILE = PROCESSED_DIR / "segments.jsonl"    # 30-60 s overlapping segments
SEGMENTS_CSV = PROCESSED_DIR / "segments.csv"       # same, CSV copy for inspection

INDEX_DIR = DATA_DIR / "index"
USER_DIR = DATA_DIR / "user"                       # local learning progress
REPORTS_DIR = ROOT_DIR / "reports"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"             # hand-written bilingual lessons

SOURCES_FILE = CONFIG_DIR / "sources.json"
TOPICS_FILE = CONFIG_DIR / "topics.json"


def load_env_file(path: Path = ROOT_DIR / ".env") -> None:
    """Very small .env reader (KEY=value per line) so no extra package is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

# --------------------------------------------------------------------------
# Stage 1 - Data collection (YouTube Data API v3)
# --------------------------------------------------------------------------
YOUTUBE_API_KEY_ENV = "YOUTUBE_API_KEY"
DAILY_QUOTA = 10_000           # free YouTube Data API quota (units / day)
QUOTA_SAFETY_MARGIN = 500      # stop before hitting the hard limit
MIN_VIDEO_SECONDS = 3 * 60     # ignore shorts / trailers
MAX_VIDEO_SECONDS = 3 * 60 * 60
TRANSCRIPT_LANGUAGES = ["en", "en-US", "en-GB"]
TRANSCRIPT_DELAY_SECONDS = 1.0  # polite pause between transcript requests

# --------------------------------------------------------------------------
# Stage 2 - Preprocessing
# --------------------------------------------------------------------------
SEGMENT_TARGET_SECONDS = 45    # aim for ~45 s segments ...
SEGMENT_MIN_SECONDS = 30       # ... never shorter than 30 s (except a video's tail)
SEGMENT_MAX_SECONDS = 60       # ... never longer than 60 s
SEGMENT_OVERLAP_SECONDS = 15   # consecutive segments share ~15 s of speech
MAX_SENTENCE_SECONDS = 15      # unpunctuated auto-captions are cut into ~15 s "sentences"
MAX_SENTENCE_WORDS = 40

# --------------------------------------------------------------------------
# Stage 3/4 - Representation, indexing and retrieval
# --------------------------------------------------------------------------
SBERT_MODEL = os.environ.get("CLIPSAGE_SBERT_MODEL", "all-MiniLM-L6-v2")
TOP_K_SEGMENTS = 10            # segments shown to the explanation step
CANDIDATE_POOL = 50            # segments fetched before diversity re-ranking
MAX_SEGMENTS_PER_VIDEO = 2     # diversity: at most N segments from one video
MMR_LAMBDA = 0.7               # 1.0 = pure relevance, 0.0 = pure diversity
MIN_RELATIVE_SCORE = 0.4       # drop segments scoring below 40% of the best match

# Below these cosine scores the topic is treated as "not covered".
NOT_COVERED_THRESHOLD = {"tfidf": 0.08, "sbert": 0.35}

# --------------------------------------------------------------------------
# Stage 5 - Explanation
# --------------------------------------------------------------------------
SUMMARY_SENTENCES = 5          # 4-6 sentence explanation
SUMMARY_MIN_SENTENCES = 4
SUMMARY_MAX_SENTENCES = 6
REDUNDANCY_THRESHOLD = 0.8     # drop a sentence this similar to one already chosen
ABSTRACTIVE_MODEL = "facebook/bart-large-cnn"

# --------------------------------------------------------------------------
# Stage 6 - Key terms and visuals
# --------------------------------------------------------------------------
NUM_KEY_TERMS = 15
CONCEPT_MAP_TERMS = 12
MIN_COOCCURRENCE = 1
