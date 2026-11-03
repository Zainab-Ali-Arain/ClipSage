"""Text helpers shared by TF-IDF, keyword extraction and the concept map."""

import re
from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words that are very common in spoken lectures but carry no topic meaning.
SPEECH_STOP_WORDS = {
    "okay", "ok", "right", "yeah", "yes", "gonna", "wanna", "gotta", "going", "go", "let", "lets",
    "just", "really", "basically", "actually", "literally", "thing", "things", "stuff", "know",
    "say", "said", "says", "look", "looks", "like", "kind", "sort", "lot", "lots", "want",
    "need", "make", "makes", "use", "used", "using", "way", "ways", "time", "times", "today",
    "now", "guys", "video", "videos", "lecture", "lectures", "course", "channel", "subscribe",
    "welcome", "hello", "hi", "um", "uh", "mean", "means", "got", "get", "gets", "good",
    "great", "able", "maybe", "pretty", "sure", "able", "doing", "does", "did", "doesn",
    "don", "ll", "ve", "re", "s", "t", "d", "m", "bit", "little", "example", "examples",
    "case", "cases", "question", "questions", "point", "talk", "talking", "think", "see",
    "seen", "come", "comes", "called", "different", "simple", "simply", "today", "second",
    "first", "next", "previous", "step", "also", "one", "two", "three", "four", "five",
    "third", "fourth", "fifth", "last", "another", "bam", "double",
    "hooray", "music", "applause", "laughter", "inaudible", "cs50", "statquest", "josh",
}

STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | SPEECH_STOP_WORDS)

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+\-]*[a-zA-Z0-9+]\b"

QUESTION_WORDS = re.compile(
    r"^\s*(what\s+(is|are|does|do)|define|explain|describe|how\s+(does|do|is)|why\s+(is|are|do|does)|"
    r"tell\s+me\s+about|meaning\s+of)\s+(a\s+|an\s+|the\s+)?",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    """'What is a binary search tree?' -> 'binary search tree'."""
    q = QUESTION_WORDS.sub("", query.strip())
    return re.sub(r"[?.!]+$", "", q).strip() or query.strip()


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(TOKEN_PATTERN, text.lower()) if t not in STOP_WORDS]


def contains_term(text: str, term: str) -> bool:
    """Whole-word, case-insensitive match that also accepts a trailing plural 's'."""
    return re.search(rf"\b{re.escape(term.lower())}s?\b", text.lower()) is not None
