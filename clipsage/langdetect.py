"""
Automatic English / Roman Urdu detection for search queries and questions.

Two signals are combined:
  1. a lexicon of very common Roman Urdu function words (hai, kya, mein, aur,
     kaise, ...) and English function words (what, is, the, how, ...);
  2. a small machine-learning model: Multinomial Naive Bayes on character
     n-grams, trained on ClipSage's own hand-written bilingual lessons
     (every lesson has an English and a Roman Urdu version of the same text).

The lexicon decides when it sees clear evidence; the classifier breaks ties.
Single technical words ("paging", "TCP") default to English.

    detect_language("Process aur thread mein kya difference hai?")  -> "ru"
    detect_language("Explain paging in operating systems")          -> "en"
"""

import re
from functools import lru_cache
from typing import Dict, List, Tuple

RU_MARKERS = {
    "hai", "hain", "hy", "hn", "hota", "hoti", "hote", "hua", "hui", "hue", "tha", "thi", "kya", "kia", "kaise",
    "kese", "kaisay", "kesay", "kyun", "kyon", "kyu", "kiun", "mein", "mei", "mai", "ka", "ki", "ke", "ko", "se",
    "aur", "or", "nahi", "nahin", "nhi", "bhi", "yeh", "ye", "woh", "wo", "kis", "kab", "kahan", "kitna", "kitne",
    "kitni", "batao", "bataen", "batain", "batayein", "bataiye", "samjhao", "samjhayein", "samjhaen", "samjha",
    "samajh", "matlab", "farq", "fark", "darmiyan", "liye", "liay", "wala", "wali", "wale", "karta", "karti",
    "karte", "kar", "karo", "karna", "krna", "kro", "krein", "karein", "kartay", "hum", "aap", "ap", "mujhe",
    "mujhay", "tum", "chahiye", "chahie", "sakta", "sakti", "sakte", "raha", "rahi", "rahe", "gaya", "gayi",
    "kuch", "sab", "bohat", "bahut", "zyada", "ziada", "acha", "achha", "phir", "jab", "tab", "agar", "lekin",
    "magar", "sirf", "yaar", "kaun", "kon", "konsa", "kaunsa", "kaunsi", "konsi", "isko", "usko", "iska",
    "uska", "inka", "unka", "iski", "uski", "dono", "misaal", "tareef", "wazahat", "hal", "jawab", "sawal",
    "kijiye", "dijiye", "do", "dein", "den", "zara", "saath", "baad", "pehle", "bare", "baare", "naam",
}
EN_MARKERS = {
    "the", "is", "are", "was", "were", "what", "how", "why", "which", "who", "when", "where",
    "define", "describe", "between", "of", "in", "and", "does", "do", "did", "can", "could", "should", "would",
    "please", "tell", "me", "about", "give", "show", "an", "a", "with", "for", "to", "from", "this", "that",
    "these", "those", "work", "works", "used", "use", "example", "examples", "meaning", "solve", "find",
    "calculate", "compute", "compare", "list",
}
# "explain", "difference", "example" are used unchanged in Roman Urdu, so they are not evidence.
AMBIGUOUS = {"do", "or", "me", "a", "to", "in", "is", "main", "the", "us", "un", "hi"}

TOKEN = re.compile(r"[a-zA-Z']+")


def tokens(text: str) -> List[str]:
    return [t.lower().strip("'") for t in TOKEN.findall(text or "")]


def lexicon_score(text: str) -> Tuple[int, int]:
    toks = tokens(text)
    ru = sum(1 for t in toks if t in RU_MARKERS and t not in AMBIGUOUS)
    en = sum(1 for t in toks if t in EN_MARKERS and t not in AMBIGUOUS)
    # ambiguous words count half (e.g. "do" = "two"/"give" in Urdu, verb in English)
    ru2 = sum(1 for t in toks if t in RU_MARKERS and t in AMBIGUOUS)
    en2 = sum(1 for t in toks if t in EN_MARKERS and t in AMBIGUOUS)
    return ru * 2 + ru2, en * 2 + en2


def _training_texts() -> Tuple[List[str], List[str]]:
    """English and Roman Urdu sentences from the bilingual lessons and UI strings."""
    from .knowledge import load_topics

    texts, labels = [], []

    def add(item):
        if isinstance(item, dict) and item.get("en") and item.get("ru") and item["en"] != item["ru"]:
            for lang in ("en", "ru"):
                for sent in re.split(r"(?<=[.?!])\s+", item[lang]):
                    if len(sent.split()) >= 3:
                        texts.append(sent)
                        labels.append(lang)

    for t in load_topics().values():
        for field in ("intro", "definition", "explanation", "example"):
            add(t.get(field))
        for field in ("key_concepts", "applications", "exam_points", "common_mistakes", "workflow"):
            for x in t.get(field, []):
                add(x)
        for q in t.get("questions", []):
            add(q.get("q"))
            add(q.get("explanation"))
    return texts, labels


@lru_cache(maxsize=1)
def _model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    texts, labels = _training_texts()
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2),
                          MultinomialNB(alpha=0.1))
    model.fit(texts, labels)
    return model


def classifier_probability_ru(text: str) -> float:
    model = _model()
    ru_index = list(model.classes_).index("ru")
    return float(model.predict_proba([text])[0][ru_index])


def detect_language(text: str, default: str = "en") -> str:
    """'en' or 'ru' (Roman Urdu)."""
    return detect_with_details(text, default)["lang"]


def detect_with_details(text: str, default: str = "en") -> Dict:
    text = (text or "").strip()
    if not text:
        return {"lang": default, "ru_score": 0, "en_score": 0, "p_ru": None, "reason": "empty"}
    ru, en = lexicon_score(text)
    p_ru = None
    if ru > en:
        lang, reason = "ru", "Roman Urdu words"
    elif en > ru:
        lang, reason = "en", "English words"
    else:
        if len(tokens(text)) <= 2:  # a bare topic name such as "paging" or "round robin"
            lang, reason = default, "short topic name"
        else:
            p_ru = classifier_probability_ru(text)
            lang, reason = ("ru" if p_ru >= 0.6 else "en"), "character n-gram classifier"
    return {"lang": lang, "ru_score": ru, "en_score": en, "p_ru": p_ru, "reason": reason}
