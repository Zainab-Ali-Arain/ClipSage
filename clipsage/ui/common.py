"""Shared UI helpers: theme CSS, language, bilingual text, cards, cached resources."""

import html
from typing import Iterable, Optional

import streamlit as st

from .. import config
from ..i18n import LANGUAGES, pick, short, t, tl
from ..progress import ProgressStore

PRIMARY, LIGHT, BG, CARD, TEXT, ACCENT = "#4A90E2", "#EAF4FF", "#F5F9FF", "#FFFFFF", "#243447", "#7CB9F7"

CSS = f"""
<style>
  .stApp {{ background: {BG}; color: {TEXT}; }}
  [data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid #DCE9F8; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
  .block-container {{ padding-top: 2rem; max-width: 1200px; }}

  .cs-card {{ background: {CARD}; border: 1px solid #DCE9F8; border-radius: 14px; padding: 18px 20px;
              box-shadow: 0 2px 10px rgba(74, 144, 226, 0.08); margin-bottom: 14px; }}
  .cs-card h4 {{ margin: 0 0 6px 0; font-size: 1.05rem; }}
  .cs-hero {{ background: {LIGHT}; border: 1px solid #CFE3FA; border-radius: 18px; padding: 26px 28px;
              margin-bottom: 18px; }}
  .cs-hero h1 {{ margin: 0; font-size: 2rem; }}
  .cs-hero p {{ margin: 6px 0 0 0; font-size: 1.05rem; color: #3D5A78; }}
  .cs-muted {{ color: #5B7590; font-size: 0.92rem; }}
  .cs-ru {{ color: #3D5A78; font-style: italic; border-left: 3px solid {ACCENT}; padding-left: 10px; margin-top: 4px; }}
  .cs-en {{ margin-bottom: 2px; }}

  .cs-badge {{ display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
               margin: 0 6px 6px 0; border: 1px solid transparent; }}
  .cs-badge.lecture  {{ background: #E3F1FF; color: #1F5FA8; border-color: #BFDDFB; }}
  .cs-badge.notes    {{ background: #EEF0FF; color: #4B46A6; border-color: #D5D8FB; }}
  .cs-badge.example  {{ background: #FFF6E6; color: #8A5A00; border-color: #F7DFB2; }}
  .cs-badge.computed {{ background: #E8F7EF; color: #1D6B43; border-color: #BFE6D0; }}
  .cs-badge.sample   {{ background: #F3F4F6; color: #4B5563; border-color: #E2E5EA; }}

  .cs-chip {{ display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 999px; background: {LIGHT};
              border: 1px solid #CFE3FA; font-size: 0.88rem; color: {TEXT}; }}
  .cs-ts {{ font-weight: 700; color: {PRIMARY}; text-decoration: none; }}
  .cs-quote {{ background: {BG}; border-left: 3px solid {PRIMARY}; padding: 8px 12px; border-radius: 6px;
               margin: 6px 0; font-size: 0.95rem; }}
  .cs-stat {{ background: {CARD}; border: 1px solid #DCE9F8; border-radius: 14px; padding: 14px 16px; text-align: left;
              box-shadow: 0 2px 10px rgba(74, 144, 226, 0.06); }}
  .cs-stat .v {{ font-size: 1.6rem; font-weight: 700; color: {PRIMARY}; }}
  .cs-stat .l {{ font-size: 0.85rem; color: #5B7590; }}
  .cs-thumb {{ width: 100%; border-radius: 10px; border: 1px solid #DCE9F8; }}
  .cs-def {{ border-left: 3px solid #F5A623; padding-left: 10px; }}
  div[data-testid="stExpander"] {{ background: {CARD}; border-radius: 12px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{ background: {CARD}; border-radius: 10px 10px 0 0; padding: 6px 14px; }}
  #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {"lang": "both", "method": "sbert", "summarizer": "textrank", "keyword_method": "auto",
            "top_k": config.TOP_K_SEGMENTS, "topic": "", "video_id": None, "lesson_cache": {}}


def init_state() -> None:
    first_run = "lang" not in st.session_state
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v if not isinstance(v, dict) else {})
    if first_run:  # restore the saved language after a browser refresh
        saved = get_progress().get_setting("lang")
        if saved in LANGUAGES:
            st.session_state["lang"] = saved


def remember_language() -> None:
    get_progress().set_setting("lang", st.session_state.get("lang"))


def lang() -> str:
    return st.session_state.get("lang", "both")


def T(key: str, **fmt) -> str:
    return t(key, lang(), **fmt)


def TL(key: str, **fmt) -> str:
    return tl(key, lang(), **fmt)


def P(text) -> str:
    return pick(text, lang())


def S(text) -> str:
    return short(text, lang())


def esc(text: str) -> str:
    return html.escape(str(text))


def bi_html(text, cls: str = "") -> str:
    """Bilingual paragraph(s) as HTML; with 'both', Roman Urdu is shown below in its own style."""
    if isinstance(text, str):
        return f"<p class='{cls}'>{esc(text)}</p>"
    en, ru = text.get("en", ""), text.get("ru", "")
    current = lang()
    if current == "en" or not ru:
        return f"<p class='{cls}'>{esc(en)}</p>"
    if current == "ru" or not en:
        return f"<p class='{cls}'>{esc(ru)}</p>"
    return f"<p class='cs-en {cls}'>{esc(en)}</p><p class='cs-ru'>{esc(ru)}</p>"


def show(text, cls: str = "") -> None:
    st.markdown(bi_html(text, cls), unsafe_allow_html=True)


def ui_text(key: str, **fmt) -> None:
    """A UI string from i18n rendered bilingually."""
    from ..i18n import UI

    entry = UI.get(key, {"en": key, "ru": key})
    show({k: (v.format(**fmt) if fmt else v) for k, v in entry.items()})


def badge(kind: str) -> str:
    key = {"lecture": "src_lecture", "notes": "src_notes", "example": "src_example",
           "computed": "src_computed", "sample": "src_sample"}[kind]
    return f"<span class='cs-badge {kind}'>{esc(TL(key))}</span>"


def badges(*kinds: str) -> None:
    st.markdown("".join(badge(k) for k in kinds), unsafe_allow_html=True)


def card(title: str = "", body_html: str = "", extra_cls: str = "") -> None:
    head = f"<h4>{esc(title)}</h4>" if title else ""
    st.markdown(f"<div class='cs-card {extra_cls}'>{head}{body_html}</div>", unsafe_allow_html=True)


def stat(value, label: str) -> str:
    return f"<div class='cs-stat'><div class='v'>{esc(value)}</div><div class='l'>{esc(label)}</div></div>"


def chips(items: Iterable[str]) -> str:
    return "".join(f"<span class='cs-chip'>{esc(i)}</span>" for i in items)


def bullet_list(items, cls: str = "") -> None:
    """Bilingual bullet list (each item is {en, ru})."""
    html_items = []
    for it in items:
        html_items.append(f"<li>{bi_html(it)}</li>")
    st.markdown(f"<ul class='{cls}'>{''.join(html_items)}</ul>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_sage():
    from ..pipeline import ClipSage

    return ClipSage()


def get_sage():
    """The retrieval pipeline, or None when the index has not been built."""
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


def get_progress() -> ProgressStore:
    if "progress_store" not in st.session_state:
        st.session_state.progress_store = ProgressStore()
    return st.session_state.progress_store


def get_lesson(query: str):
    """Build (and cache for this session) the lesson for a query."""
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


def goto(page_key: str, **state) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.switch_page(st.session_state["pages"][page_key])


def open_topic(query: str) -> None:
    """Callback: open a topic in Topic Learning and remember the search."""
    st.session_state["topic"] = query
    get_progress().add_search(query)
    st.session_state["_goto"] = "learn"


def topic_search_box(key: str, button_label: Optional[str] = None) -> None:
    with st.form(key, border=False):
        cols = st.columns([4, 1.4])
        q = cols[0].text_input(TL("search"), placeholder=TL("search_placeholder"), label_visibility="collapsed",
                               key=f"{key}_q")
        if cols[1].form_submit_button(button_label or TL("search"), type="primary", width="stretch") and q.strip():
            open_topic(q.strip())
            st.rerun()


def handle_pending_navigation() -> None:
    target = st.session_state.pop("_goto", None)
    if target:
        st.switch_page(st.session_state["pages"][target])
