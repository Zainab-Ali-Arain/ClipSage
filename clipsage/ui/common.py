"""
Shared UI helpers.

Language rule
  * The INTERFACE (navigation, sidebar, subject names, buttons, labels,
    headings, settings) is always English: use ui() / plain strings.
  * The EDUCATIONAL RESPONSE (explanations, answers, examples, quiz
    explanations) follows the language detected in the student's latest
    search or question: use content() / show_content().
"""

import html
from typing import Iterable, Optional

import streamlit as st

from .. import config
from ..i18n import pick, t
from ..progress import SessionActivity, SubjectStore

PRIMARY, LIGHT, BG, CARD, TEXT, ACCENT = "#4A90E2", "#EAF4FF", "#F5F9FF", "#FFFFFF", "#243447", "#7CB9F7"
LANG_NAMES = {"en": "English", "ru": "Roman Urdu"}

CSS = f"""
<style>
  .stApp {{ background: {BG}; color: {TEXT}; }}
  [data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid #DCE9F8; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
  .cs-card {{ background: {CARD}; border: 1px solid #DCE9F8; border-radius: 14px; padding: 16px 18px;
              box-shadow: 0 2px 10px rgba(74, 144, 226, 0.08); margin-bottom: 12px; }}
  .cs-card h4 {{ margin: 0 0 6px 0; font-size: 1.05rem; }}
  .cs-hero {{ background: {LIGHT}; border: 1px solid #CFE3FA; border-radius: 18px; padding: 24px 28px; margin-bottom: 16px; }}
  .cs-hero h1 {{ margin: 0; font-size: 2rem; }}
  .cs-hero p {{ margin: 6px 0 0 0; font-size: 1.05rem; color: #3D5A78; }}
  .cs-muted {{ color: #5B7590; font-size: 0.92rem; }}
  .cs-badge {{ display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
               margin: 0 6px 6px 0; border: 1px solid transparent; }}
  .cs-badge.lecture  {{ background: #E3F1FF; color: #1F5FA8; border-color: #BFDDFB; }}
  .cs-badge.notes    {{ background: #EEF0FF; color: #4B46A6; border-color: #D5D8FB; }}
  .cs-badge.example  {{ background: #FFF6E6; color: #8A5A00; border-color: #F7DFB2; }}
  .cs-badge.computed {{ background: #E8F7EF; color: #1D6B43; border-color: #BFE6D0; }}
  .cs-badge.sample   {{ background: #F3F4F6; color: #4B5563; border-color: #E2E5EA; }}
  .cs-badge.lang     {{ background: #FFFFFF; color: {PRIMARY}; border-color: #BFDDFB; }}
  .cs-chip {{ display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 999px; background: {LIGHT};
              border: 1px solid #CFE3FA; font-size: 0.88rem; color: {TEXT}; }}
  .cs-ts {{ font-weight: 700; color: {PRIMARY}; text-decoration: none; }}
  .cs-quote {{ background: {BG}; border-left: 3px solid {PRIMARY}; padding: 8px 12px; border-radius: 6px;
               margin: 6px 0; font-size: 0.95rem; }}
  .cs-stat {{ background: {CARD}; border: 1px solid #DCE9F8; border-radius: 14px; padding: 14px 16px;
              box-shadow: 0 2px 10px rgba(74, 144, 226, 0.06); }}
  .cs-stat .v {{ font-size: 1.6rem; font-weight: 700; color: {PRIMARY}; }}
  .cs-stat .l {{ font-size: 0.85rem; color: #5B7590; }}
  .cs-thumb {{ width: 100%; border-radius: 10px; border: 1px solid #DCE9F8; }}
  .cs-def {{ border-left: 3px solid #F5A623; padding-left: 10px; }}
  .cs-subject {{ background: {CARD}; border: 1px solid #DCE9F8; border-top: 4px solid {PRIMARY}; border-radius: 14px;
                 padding: 16px 18px; min-height: 150px; box-shadow: 0 2px 10px rgba(74, 144, 226, 0.08); }}
  .cs-subject h3 {{ margin: 0 0 4px 0; font-size: 1.15rem; }}
  .cs-query {{ background: {CARD}; border: 1px solid #CFE3FA; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; }}
  .cs-query .q {{ font-size: 1.15rem; font-weight: 600; }}
  .cs-video {{ background: {CARD}; border: 1px solid #DCE9F8; border-radius: 14px; padding: 12px 14px; }}
  div[data-testid="stExpander"] {{ background: {CARD}; border-radius: 12px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
  .stTabs [data-baseweb="tab"] {{ background: {CARD}; border-radius: 10px 10px 0 0; padding: 6px 14px; }}
  #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {"lang_mode": "auto", "resp_lang": "en", "method": "sbert", "summarizer": "textrank",
            "keyword_method": "auto", "top_k": config.TOP_K_SEGMENTS, "subject": None, "search": None,
            "topic_id": None, "video_id": None, "video_start": 0}


def init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("lesson_cache", {})
    if "activity" not in st.session_state:          # temporary: lives only in this browser session
        st.session_state.activity = SessionActivity()


def activity() -> SessionActivity:
    return st.session_state.activity


def store(subject: str) -> SubjectStore:
    """Explicitly saved data of one subject (written only by Save buttons)."""
    return SubjectStore(subject)


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------
def rlang() -> str:
    """Language of the educational response."""
    mode = st.session_state.get("lang_mode", "auto")
    return st.session_state.get("resp_lang", "en") if mode == "auto" else mode


def ui(key: str, **fmt) -> str:
    """Interface text - always English."""
    return t(key, "en", **fmt)


def content(text) -> str:
    """Educational text in the response language."""
    return pick(text, rlang())


def esc(text) -> str:
    return html.escape(str(text))


def content_html(text, cls: str = "") -> str:
    return f"<p class='{cls}'>{esc(content(text))}</p>"


def show_content(text, cls: str = "") -> None:
    st.markdown(content_html(text, cls), unsafe_allow_html=True)


BADGES = {"lecture": "From lectures (retrieved transcript)", "notes": "ClipSage notes (written explanation)",
          "example": "Additional example", "computed": "Computed result", "sample": "Sample data (synthetic)"}


def badge(kind: str) -> str:
    return f"<span class='cs-badge {kind}'>{esc(BADGES[kind])}</span>"


def badges(*kinds: str) -> None:
    st.markdown("".join(badge(k) for k in kinds), unsafe_allow_html=True)


def lang_badge() -> str:
    return f"<span class='cs-badge lang'>Response language: {LANG_NAMES[rlang()]}</span>"


def card(title: str = "", body_html: str = "", extra_cls: str = "") -> None:
    head = f"<h4>{esc(title)}</h4>" if title else ""
    st.markdown(f"<div class='cs-card {extra_cls}'>{head}{body_html}</div>", unsafe_allow_html=True)


def stat(value, label: str) -> str:
    return f"<div class='cs-stat'><div class='v'>{esc(value)}</div><div class='l'>{esc(label)}</div></div>"


def chips(items: Iterable[str]) -> str:
    return "".join(f"<span class='cs-chip'>{esc(i)}</span>" for i in items)


def bullet_list(items, cls: str = "") -> None:
    lis = "".join(f"<li>{esc(content(it))}</li>" for it in items)
    st.markdown(f"<ul class='{cls}'>{lis}</ul>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_sage():
    from ..pipeline import ClipSage

    return ClipSage()


def get_sage():
    """The retrieval pipeline, or None when the lecture index has not been built."""
    from ..lessons import index_available

    if not index_available():
        return None
    try:
        return _load_sage()
    except Exception as e:  # corrupt / out-of-sync index
        st.session_state["index_error"] = str(e)
        return None


def effective_method() -> str:
    from ..lessons import sbert_available

    m = st.session_state.get("method", "sbert")
    return "tfidf" if m == "sbert" and not sbert_available() else m


def get_lesson(query: str):
    """Lecture-retrieval part of a lesson (cached for this session)."""
    from ..lessons import build_lesson

    kw = st.session_state.get("keyword_method", "auto")
    key = (query.strip().lower(), effective_method(), st.session_state.get("summarizer"),
           st.session_state.get("top_k"), kw)
    cache = st.session_state.setdefault("lesson_cache", {})
    if key not in cache:
        cache[key] = build_lesson(query, method=effective_method(), summarizer=st.session_state.get("summarizer"),
                                  keyword_method=None if kw == "auto" else kw,
                                  top_k=st.session_state.get("top_k"), sage=get_sage())
    return cache[key]


# ---------------------------------------------------------------------------
# Navigation and search
# ---------------------------------------------------------------------------
def goto(page_key: str, **state) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.switch_page(st.session_state["pages"][page_key])


def request_page(page_key: str, **state) -> None:
    """Callback-safe navigation: switch on the next run."""
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["_goto"] = page_key


def handle_pending_navigation() -> None:
    target = st.session_state.pop("_goto", None)
    if target:
        st.switch_page(st.session_state["pages"][target])


def run_search(query: str, subject: Optional[str] = None) -> None:
    """Analyse a query, set the response language from it and open the results page."""
    from ..search import analyze_query

    query = (query or "").strip()
    if not query:
        return
    analysis = analyze_query(query, subject)
    st.session_state["search"] = analysis
    st.session_state["resp_lang"] = analysis["lang"]
    if analysis["subject"]:
        st.session_state["subject"] = analysis["subject"]
    activity().add_search(query, analysis["lang"], analysis["subject"])
    st.session_state["_goto"] = "search"


def open_topic(topic_id: str, subject: str) -> None:
    """Callback: open a lesson inside its subject environment."""
    st.session_state["topic_id"] = topic_id
    st.session_state["subject"] = subject
    st.session_state["subject_section"] = "Topics & Theory"
    st.session_state["_goto"] = "subject"


def search_box(key: str, subject: Optional[str] = None, placeholder: Optional[str] = None) -> None:
    with st.form(key, border=False):
        cols = st.columns([5, 1])
        q = cols[0].text_input("Search", label_visibility="collapsed", key=f"{key}_q",
                               placeholder=placeholder or "Search a topic or ask a question - English or Roman Urdu")
        if cols[1].form_submit_button("Search", type="primary", width="stretch") and q.strip():
            run_search(q, subject)
            st.rerun()


def en(text) -> str:
    """English version of a bilingual label (for interface elements)."""
    return pick(text, "en")
