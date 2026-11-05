"""Dashboard, Explore Topics, Saved Topics, Learning Progress and Settings pages."""

import time

import plotly.graph_objects as go
import streamlit as st

from ..i18n import LANGUAGES, UI
from ..index import index_info
from ..keywords import KEYWORD_LABELS
from ..knowledge import CATEGORIES, load_topics, topics_by_subject
from ..lessons import index_available, sbert_available
from ..retrieve import METHOD_LABELS
from ..summarize import SUMMARIZER_LABELS
from .common import (P, S, T, TL, badges, bi_html, card, chips, esc, get_progress, goto, open_topic, show, stat,
                     topic_search_box, ui_text)


def _topic_card(topic: dict, key: str) -> None:
    cat = CATEGORIES.get(topic["category"], {"en": topic["category"], "ru": topic["category"]})
    body = (f"<div class='cs-muted'>{esc(topic['subject'])} · {esc(S(cat))}</div>"
            f"{bi_html(topic['definition'])}")
    with st.container(border=False):
        card(S(topic["title"]), body)
        st.button(TL("open"), key=key, on_click=open_topic, args=(topic["title"]["en"],), width="stretch")


# ---------------------------------------------------------------------------
def dashboard_page() -> None:
    store = get_progress()
    summary = store.summary()
    st.markdown(f"<div class='cs-hero'><h1>{esc(TL('welcome'))}</h1>{bi_html(UI['tagline'])}</div>",
                unsafe_allow_html=True)
    topic_search_box("dash_search")

    info = index_info()
    acc = summary["accuracy"]
    cols = st.columns(4)
    cols[0].markdown(stat(summary["topics_studied"], TL("topics_studied")), unsafe_allow_html=True)
    cols[1].markdown(stat(summary["questions_answered"], TL("questions_answered")), unsafe_allow_html=True)
    cols[2].markdown(stat(f"{acc:.0%}" if acc is not None else "-", TL("accuracy")), unsafe_allow_html=True)
    cols[3].markdown(stat(f"{info.get('segments', 0):,}", TL("lectures_indexed")), unsafe_allow_html=True)
    if not index_available():
        st.info(T("no_index"))
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    topics = load_topics()
    with left:
        st.subheader(TL("recommended"))
        studied = {s["id"] for s in store.data["studied"]}
        weak = summary["weak_topics"]
        recommended = [topics[t] for t in weak if t in topics]
        recommended += [t for t in topics.values() if t["id"] not in studied and t not in recommended]
        grid = st.columns(2)
        for i, topic in enumerate(recommended[:6]):
            with grid[i % 2]:
                _topic_card(topic, f"rec_{topic['id']}")

    with right:
        st.subheader(TL("recent_searches"))
        searches = store.data["searches"][:8]
        if searches:
            for i, s in enumerate(searches):
                st.button(f"\U0001F50E {s['q']}", key=f"recent_{i}", on_click=open_topic, args=(s["q"],), width="stretch")
        else:
            st.caption(T("nothing_yet"))

        st.subheader(TL("recent_lectures"))
        viewed = store.data["viewed"][:4]
        if viewed:
            for v in viewed:
                c1, c2 = st.columns([1, 2])
                c1.image(f"https://i.ytimg.com/vi/{v['video_id']}/mqdefault.jpg")
                c2.markdown(f"**{esc(v['title'])}**  \n<span class='cs-muted'>{esc(v['channel'])}</span>",
                            unsafe_allow_html=True)
                if c2.button(TL("open"), key=f"viewed_{v['video_id']}"):
                    goto("videos", video_id=v["video_id"])
        else:
            st.caption(T("nothing_yet"))

        st.subheader(TL("nav_saved"))
        saved = store.data["saved"][:5]
        if saved:
            for s in saved:
                st.button(f"⭐ {s['title']}", key=f"dsaved_{s['id']}", on_click=open_topic, args=(s["title"],),
                          width="stretch")
        else:
            st.caption(T("nothing_yet"))

    st.divider()
    q1, q2, q3 = st.columns(3)
    q1.page_link(st.session_state["pages"]["practice"], label=TL("nav_practice"), icon="✍️")
    q2.page_link(st.session_state["pages"]["math"], label=TL("nav_math"), icon="\U0001F9EE")
    q3.page_link(st.session_state["pages"]["ml"], label=TL("nav_ml"), icon="\U0001F9EA")


# ---------------------------------------------------------------------------
def explore_page() -> None:
    st.title(TL("nav_explore"))
    topic_search_box("explore_search")
    groups = topics_by_subject()
    tabs = st.tabs([f"{name} ({len(items)})" for name, items in groups.items()])
    for tab, (name, items) in zip(tabs, groups.items()):
        with tab:
            badges("notes")
            cols = st.columns(3)
            for i, topic in enumerate(items):
                with cols[i % 3]:
                    _topic_card(topic, f"explore_{topic['id']}")
    st.divider()
    show({"en": "You can also search ANY topic covered by the collected lectures (for example 'paging' or 'sql joins'). "
                "Topics without written notes are explained directly from the lecture transcripts.",
          "ru": "Aap jama ki gayi lectures ka KOI bhi topic search kar sakte hain (maslan 'paging' ya 'sql joins'). "
                "Jin topics ke notes nahi, woh seedha lecture transcripts se samjhaye jate hain."})


# ---------------------------------------------------------------------------
def saved_page() -> None:
    st.title(TL("nav_saved"))
    store = get_progress()
    saved = store.data["saved"]
    if not saved:
        st.info(T("nothing_yet"))
        return
    for s in saved:
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.markdown(f"<div class='cs-card'><h4>⭐ {esc(s['title'])}</h4>"
                    f"<span class='cs-muted'>{time.strftime('%Y-%m-%d', time.localtime(s['ts']))}</span></div>",
                    unsafe_allow_html=True)
        c2.button(TL("open"), key=f"saved_open_{s['id']}", on_click=open_topic, args=(s["title"],), width="stretch")
        if c3.button(TL("remove"), key=f"saved_rm_{s['id']}", width="stretch"):
            store.toggle_saved(s["id"], s["title"])
            st.rerun()


# ---------------------------------------------------------------------------
def progress_page() -> None:
    st.title(TL("nav_progress"))
    store = get_progress()
    s = store.summary()
    topics = load_topics()
    cols = st.columns(4)
    cols[0].markdown(stat(s["topics_studied"], TL("topics_studied")), unsafe_allow_html=True)
    cols[1].markdown(stat(s["questions_answered"], TL("questions_answered")), unsafe_allow_html=True)
    cols[2].markdown(stat(f"{s['accuracy']:.0%}" if s["accuracy"] is not None else "-", TL("accuracy")),
                     unsafe_allow_html=True)
    m = s["math"]
    cols[3].markdown(stat(f"{m['correct']}/{m['attempts']}", TL("nav_math")), unsafe_allow_html=True)
    st.write("")

    if s["questions_answered"] == 0:
        ui_text("progress_start")
    elif s["accuracy"] >= 0.7:
        ui_text("progress_great")
    if s["weak_topics"]:
        names = ", ".join(S(topics[t]["title"]) if t in topics else t for t in s["weak_topics"])
        st.warning(T("progress_review", topics=names))

    if s["per_topic"]:
        names = [S(topics[t]["title"]) if t in topics else t for t in s["per_topic"]]
        answered = [v["answered"] for v in s["per_topic"].values()]
        correct = [v["correct"] for v in s["per_topic"].values()]
        fig = go.Figure([go.Bar(y=names, x=answered, orientation="h", name=TL("questions_answered"), marker_color="#B9D7F7"),
                         go.Bar(y=names, x=correct, orientation="h", name=S({"en": "Correct", "ru": "Sahi"}), marker_color="#4A90E2")])
        fig.update_layout(barmode="overlay", template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF",
                          height=120 + 40 * len(names), margin=dict(l=10, r=10, t=30, b=30), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(TL("recent_searches"))
        st.markdown(chips(x["q"] for x in store.data["searches"][:15]) or T("nothing_yet"), unsafe_allow_html=True)
    with c2:
        st.subheader(TL("topics_studied"))
        st.markdown(chips(x["title"] for x in store.data["studied"][:15]) or T("nothing_yet"), unsafe_allow_html=True)
    st.caption(T("progress_note"))
    with st.expander(TL("reset_progress")):
        if st.button(TL("reset_progress"), type="primary"):
            store.reset()
            st.rerun()


# ---------------------------------------------------------------------------
def settings_page() -> None:
    st.title(TL("nav_settings"))
    langs = list(LANGUAGES)

    def _sync_lang():
        st.session_state["lang"] = st.session_state["lang_settings"]
        get_progress().set_setting("lang", st.session_state["lang"])

    st.radio(TL("language"), langs, format_func=LANGUAGES.get, horizontal=True, key="lang_settings",
             index=langs.index(st.session_state.get("lang", "both")), on_change=_sync_lang)
    show({"en": "Roman Urdu is written in English letters. Lecture transcripts always stay in their original English.",
          "ru": "Roman Urdu English huroof mein likhi jati hai. Lecture transcripts hamesha asal English mein rehte hain."})
    st.divider()
    methods = ["sbert", "tfidf"]
    st.radio(TL("retrieval_method"), methods, format_func=METHOD_LABELS.get, key="method", horizontal=True)
    if st.session_state.method == "sbert" and not sbert_available():
        st.caption("Sentence-BERT index not built yet - TF-IDF is used until you run the pipeline.")
    st.selectbox(TL("summary_method"), ["textrank", "embedding", "top_sentences"], format_func=SUMMARIZER_LABELS.get,
                 key="summarizer")
    st.selectbox("Key terms", ["auto", "keybert", "tfidf"],
                 format_func=lambda k: "Auto (KeyBERT with SBERT, TF-IDF otherwise)" if k == "auto" else KEYWORD_LABELS[k],
                 key="keyword_method")
    st.slider(TL("segments_to_use"), 5, 20, key="top_k")
    st.caption(T("settings_saved"))
    st.divider()
    info = index_info()
    st.subheader("Data")
    if info:
        st.json(info)
    else:
        st.info(T("no_index"))
    st.caption("ClipSage links to the original YouTube videos and credits every source. It is a study aid - "
               "please watch the referenced lectures.")
