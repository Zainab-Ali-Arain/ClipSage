"""Dashboard, Search results, Saved data and Settings pages (English interface)."""

import streamlit as st

from ..index import index_info
from ..keywords import KEYWORD_LABELS
from ..library import library_stats
from ..lessons import index_available, sbert_available
from ..retrieve import METHOD_LABELS
from ..subjects import load_subjects, topics_of
from ..summarize import SUMMARIZER_LABELS
from .common import (LANG_NAMES, activity, badges, chips, esc, get_lesson, lang_badge, open_topic, run_search,
                     search_box, store)
from .topic_view import render_lecture_explanation, render_topic


def _open_section(name: str, section: str) -> None:
    st.session_state["subject"] = name
    st.session_state["subject_section"] = section
    st.session_state["_goto"] = "subject"


def _pick_subject(name: str) -> None:
    _open_section(name, "Overview")


# ---------------------------------------------------------------------------
def dashboard_page() -> None:
    st.markdown("<div class='cs-hero'><h1>ClipSage</h1><p>Study six Computer Science subjects with explanations, real lecture "
                "moments, quizzes and hands-on practice. Ask in English or Roman Urdu - the answer follows your language.</p></div>",
                unsafe_allow_html=True)
    search_box("dash_search")
    stats = library_stats()
    act = activity()
    if not index_available():
        st.info("The video library has not been built yet. Lessons, quizzes and practice tools work now; run "
                "run_pipeline.bat to collect videos and timestamps.")
    st.markdown("### Subjects")
    subjects = load_subjects()
    for row in range(0, len(subjects), 3):
        cols = st.columns(3)
        for col, s in zip(cols, subjects[row:row + 3]):
            topics = topics_of(s["name"])
            saved = store(s["name"]).counts()
            summary = act.subject_summary(s["name"])
            n_questions = sum(len(t["questions"]) for t in topics)
            facts = [f"{len(topics)} topics", f"{n_questions} questions",
                     f"{stats['per_subject'].get(s['name'], 0)} videos", f"{sum(saved.values())} saved"]
            with col:
                st.markdown(f"<div class='cs-subject'><h3>{esc(s['name'])}</h3><p class='cs-muted'>{esc(s['tagline'])}</p>"
                            f"{chips(facts)}</div>", unsafe_allow_html=True)
                if summary["answered"]:
                    st.progress(summary["correct"] / summary["answered"],
                                text=f"This session: {summary['correct']}/{summary['answered']} correct")
                st.button(f"Open {s['name']}", key=f"dash_{s['id']}", on_click=_pick_subject, args=(s["name"],),
                          type="primary", width="stretch")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Recent searches (this session)")
        if act.searches:
            for i, s in enumerate(act.searches[:8]):
                st.button(f"{s['q']}  -  {LANG_NAMES[s['lang']]}", key=f"recent_{i}", on_click=run_search, args=(s["q"],),
                          width="stretch")
        else:
            st.caption("No searches yet. Searches are not saved.")
    with c2:
        st.markdown("### Recently viewed lectures (this session)")
        if act.viewed:
            for v in act.viewed[:5]:
                st.markdown(f"- [{esc(v['title'])}](https://www.youtube.com/watch?v={v['video_id']}) "
                            f"<span class='cs-muted'>{esc(v['subject'] or '')}</span>", unsafe_allow_html=True)
        else:
            st.caption("Nothing viewed yet.")
    c3, c4 = st.columns(2)
    c3.page_link(st.session_state["pages"]["math"], label="Mathematics Playground", icon=":material/calculate:")
    c4.page_link(st.session_state["pages"]["ml"], label="ML Algorithm Playground", icon=":material/science:")


# ---------------------------------------------------------------------------
def search_page() -> None:
    search_box("search_page_box")
    a = st.session_state.get("search")
    if not a:
        st.info("Type a topic or a question above, for example 'Explain paging in operating systems' or "
                "'Process aur thread mein kya difference hai?'.")
        return
    det = a["detection"]
    reason = {"Roman Urdu words": "Roman Urdu words in your question", "English words": "English words in your question",
              "short topic name": "short topic name (default English)",
              "character n-gram classifier": "character n-gram classifier"}.get(det["reason"], det["reason"])
    subject = a["subject"] or "Not one of the six subjects"
    topic = a["topic"]
    facts = [f"Subject: {subject}", "Topic: " + (topic["title"]["en"] if topic else "no written lesson"),
             f"Detected language: {LANG_NAMES[a['lang']]} ({reason})"]
    st.markdown(f"<div class='cs-query'><span class='cs-muted'>Search query</span><div class='q'>{esc(a['query'])}</div>"
                f"{chips(facts)}</div>", unsafe_allow_html=True)
    related = [m for m in a["matches"] if not topic or m["topic"]["id"] != topic["id"]][:4]
    if related:
        st.caption("Related topics")
        cols = st.columns(len(related))
        for col, m in zip(cols, related):
            col.button(m["topic"]["title"]["en"], key=f"rel_{m['topic']['id']}", on_click=open_topic,
                       args=(m["topic"]["id"], m["topic"]["subject"]), width="stretch")
    if topic:
        render_topic(topic, key="search", query=a["query"])
        if topic.get("practice") or topic.get("playground"):
            st.markdown("#### Practice exercises")
            st.button(f"Open {topic['subject']} practice tools", on_click=_open_section,
                      args=(topic["subject"], "Algorithm Practice"))
        return
    # no written lesson: answer from the lectures if possible
    lesson = get_lesson(a["focus"] or a["query"]) if index_available() else None
    result = lesson["lecture"] if lesson else None
    if result and result.get("covered"):
        st.markdown(f"### {esc(a['focus'] or a['query'])}")
        st.markdown(lang_badge(), unsafe_allow_html=True)
        if a["lang"] == "ru":
            st.caption("There is no written Roman Urdu lesson for this topic; the explanation below is taken from "
                       "English lecture transcripts.")
        badges("lecture")
        left, right = st.columns([3, 2], gap="large")
        with left:
            render_lecture_explanation(result)
        with right:
            best = result["hits"][0]
            st.markdown("#### Video reference")
            st.video(f"https://www.youtube.com/watch?v={best['video_id']}", start_time=int(best["start"]))
            st.markdown(f"**{esc(best['title'])}** - {esc(best['channel_title'])}")
        return
    st.warning(f"“{a['query']}” is not covered by the six subjects or the collected lectures yet.")
    if a["matches"]:
        st.write("Closest topics:")
        for m in a["matches"][:3]:
            st.button(m["topic"]["title"]["en"], key=f"close_{m['topic']['id']}", on_click=open_topic,
                      args=(m["topic"]["id"], m["topic"]["subject"]))


# ---------------------------------------------------------------------------
def saved_page() -> None:
    st.title("Saved Data")
    st.caption("ClipSage saves nothing automatically. Items appear here only after you click a Save button, and each "
               "subject keeps its own saved data.")
    for s in load_subjects():
        counts = store(s["name"]).counts()
        total = sum(counts.values())
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{esc(s['name'])}** - " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in counts.items()))
        c2.button("Manage", key=f"manage_{s['id']}", on_click=_open_section, args=(s["name"], "Saved"),
                  disabled=total == 0, width="stretch")


# ---------------------------------------------------------------------------
def settings_page() -> None:
    st.title("Settings")
    st.markdown("#### Response language")
    st.radio("Language of explanations and answers", ["auto", "en", "ru"], key="lang_mode", horizontal=True,
             format_func={"auto": "Automatic (detect from my question)", "en": "Always English",
                          "ru": "Always Roman Urdu"}.get)
    st.caption("The interface (menus, buttons, headings) is always English. Automatic mode answers in the language of "
               "your latest search or question.")
    st.markdown("#### Retrieval")
    st.radio("Retrieval method", ["sbert", "tfidf"], format_func=METHOD_LABELS.get, key="method", horizontal=True)
    if st.session_state.method == "sbert" and not sbert_available():
        st.caption("The Sentence-BERT index is not built yet - TF-IDF is used until you run the pipeline.")
    st.selectbox("Lecture explanation method", ["textrank", "embedding", "top_sentences"], format_func=SUMMARIZER_LABELS.get,
                 key="summarizer")
    st.selectbox("Key-term method", ["auto", "keybert", "tfidf"],
                 format_func=lambda k: "Auto (KeyBERT with SBERT, TF-IDF otherwise)" if k == "auto" else KEYWORD_LABELS[k],
                 key="keyword_method")
    st.slider("Lecture segments to use", 5, 20, key="top_k")
    st.markdown("#### Data and privacy")
    st.write("- Searches, quiz answers, notes you type and practice results are kept only in this browser session.\n"
             "- They are written to disk only when you click Save Topic, Save Notes, Save Quiz Results, Save Progress, "
             "Save Learning Material or Save result.\n"
             "- Saved data is stored per subject in data/user/subjects/<subject>.json on this computer and is never uploaded.\n"
             "- Use Saved Data > Manage to view or delete it.")
    info = index_info()
    st.markdown("#### Lecture index")
    if info:
        st.json(info)
    else:
        st.info("Not built yet - run run_pipeline.bat.")
    lib = library_stats()
    st.write(f"Video library: {lib['total']} videos ({lib['with_transcript']} with captions), duplicates: {lib['duplicates']}")
