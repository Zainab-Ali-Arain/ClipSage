"""Video Library: all collected lectures by subject and topic, with timestamp navigation."""

import plotly.graph_objects as go
import streamlit as st

from ..lectures import ask_lecture, explain_range, most_useful_sections, parse_time_range, video_info, video_timeline
from ..library import library_stats, load_library
from ..subjects import load_subjects, topics_of
from ..utils import format_timestamp
from .common import activity, badges, chips, content, effective_method, esc, get_sage, rlang, store

RANGE_SUMMARY = {"en": "Between {start} and {end} the lecture covers: {terms}.",
                 "ru": "{start} se {end} tak lecture mein yeh parhaya gaya: {terms}."}


def _timeline_chart(sections) -> go.Figure:
    fig = go.Figure()
    top = max([s["relevance"] or 0 for s in sections] + [1e-9])
    for s in sections:
        rel = s["relevance"]
        alpha = 1.0 if rel is None else 0.25 + 0.75 * max(0.0, min(1.0, rel / top))
        fig.add_trace(go.Bar(x=[(s["end"] - s["start"]) / 60], base=[s["start"] / 60], y=["lecture"], orientation="h",
                             marker=dict(color=f"rgba(74,144,226,{alpha})", line=dict(color="white", width=1)), showlegend=False,
                             hovertemplate=f"{s['start_label']}-{s['end_label']}<br>{', '.join(s['terms'][:4])}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=150, template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF",
                      margin=dict(l=10, r=10, t=30, b=40), yaxis=dict(visible=False), xaxis_title="minutes",
                      title="Sections of the lecture (darker = more relevant to the selected topic)")
    return fig


def videos_page() -> None:
    st.title("Video Library")
    lib = load_library()
    stats = library_stats(lib)
    cols = st.columns(len(stats["per_subject"]))
    for col, (name, n) in zip(cols, stats["per_subject"].items()):
        col.metric(name, n)
    if not lib:
        st.info("The video library is empty. Run run_pipeline.bat: it searches YouTube (Data API) for every topic of "
                "the six subjects and downloads the captions used for timestamps.")
        return
    st.caption(f"{stats['total']} videos, {stats['with_transcript']} with captions, {stats['duplicates']} duplicates.")

    c1, c2, c3 = st.columns(3)
    subjects = [s["name"] for s in load_subjects()]
    subject = c1.selectbox("Subject", ["All subjects"] + subjects,
                           index=(subjects.index(st.session_state["subject"]) + 1) if st.session_state.get("subject") in subjects else 0)
    topic_opts = {t["id"]: t["title"]["en"] for t in (topics_of(subject) if subject != "All subjects" else [])}
    topic = c2.selectbox("Topic", ["All topics"] + list(topic_opts), format_func=lambda i: topic_opts.get(i, i))
    text = c3.text_input("Filter by title or channel")
    shown = [v for v in lib if (subject == "All subjects" or v["subject"] == subject)
             and (topic == "All topics" or v["topic_id"] == topic)
             and (not text or text.lower() in f"{v['title']} {v['channel_title']}".lower())]
    if not shown:
        st.info("No videos match these filters.")
        return
    ids = [v["video_id"] for v in shown]
    by_id = {v["video_id"]: v for v in shown}
    current = st.session_state.get("video_id")
    vid = st.selectbox("Lecture", ids, index=ids.index(current) if current in ids else 0,
                       format_func=lambda i: f"{by_id[i]['title']} - {by_id[i]['channel_title']} ({by_id[i]['subject']})")
    if vid != current:
        st.session_state["video_id"] = vid
        st.session_state["video_start"] = 0
    v = by_id[vid]
    activity().add_viewed(vid, v["title"], v["subject"])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.video(f"https://www.youtube.com/watch?v={vid}", start_time=int(st.session_state.get("video_start") or 0))
        st.markdown(f"**{esc(v['title'])}**  \n<span class='cs-muted'>{esc(v['channel_title'])} - {esc(v['subject'])}</span>",
                    unsafe_allow_html=True)
        st.link_button("Open on YouTube", v["url"])
    sage = get_sage()
    if sage is None or not v["has_transcript"]:
        right.info("No captions are available for this video, so timestamps and lecture navigation cannot be shown.")
        return
    engine = sage.engine
    info = video_info(engine, vid)
    query = topic_opts.get(v["topic_id"]) or (v["topic_id"] or "").replace("_", " ") or None
    sections = video_timeline(engine, vid, query=query, method=effective_method())
    with right:
        badges("lecture")
        st.caption(f"Transcript {format_timestamp(info['transcript_start'])}-{format_timestamp(info['transcript_end'])}, "
                   f"{info['segments']} sections")
        if query:
            st.markdown(f"**Most useful sections for: {esc(query)}**")
            for s in most_useful_sections(sections):
                c_a, c_b = st.columns([1, 3])
                if c_a.button(s["start_label"], key=f"jump_top_{s['row']}"):
                    st.session_state["video_start"] = s["start"]; st.rerun()
                c_b.markdown(chips(s["terms"][:4]), unsafe_allow_html=True)
    st.plotly_chart(_timeline_chart(sections), width="stretch")

    st.markdown("#### Lecture timeline")
    for s in sections:
        with st.expander(f"{s['start_label']}-{s['end_label']}  -  {', '.join(s['terms'][:4])}"):
            st.markdown(f"<div class='cs-quote'>{esc(s['first_sentence'])} ...</div>", unsafe_allow_html=True)
            for d in s["definitions"]:
                st.markdown(f"<div class='cs-quote cs-def'>{esc(d['text'])} <span class='cs-muted'>({format_timestamp(d['start'])})"
                            f"</span></div>", unsafe_allow_html=True)
            if st.button(f"Play from {s['start_label']}", key=f"jump_{s['row']}"):
                st.session_state["video_start"] = s["start"]; st.rerun()

    st.markdown("#### What is taught between two times?")
    q_range = st.text_input("Time range question", placeholder="What is taught between 0:30 and 1:30?  /  0:30 se 1:30 tak kya parhaya gaya?")
    if q_range:
        from ..langdetect import detect_language

        st.session_state["resp_lang"] = detect_language(q_range)
        rng = parse_time_range(q_range)
        if rng is None:
            st.warning("Please include two times such as 10:20 and 15:30.")
        else:
            res = explain_range(engine, vid, rng[0], rng[1], method=effective_method())
            if not res["ok"]:
                st.warning(f"That time range is outside the transcript ({format_timestamp(info['transcript_start'])}-"
                           f"{format_timestamp(info['transcript_end'])}).")
            else:
                badges("lecture")
                st.write(content({k: t.format(start=res["start_label"], end=res["end_label"], terms=", ".join(res["terms"][:6]))
                                  for k, t in RANGE_SUMMARY.items()}))
                for s in res["summary"]:
                    st.markdown(f"<div class='cs-quote'><a class='cs-ts' href='{s['url']}' target='_blank'>{s['timestamp']}</a> "
                                f"{esc(s['text'])}</div>", unsafe_allow_html=True)
                if rlang() == "ru":
                    st.caption("Transcript sentences are shown in the lecture's original English.")
                with st.expander(f"Full transcript {res['start_label']}-{res['end_label']} ({len(res['sentences'])} sentences)"):
                    for s in res["sentences"]:
                        st.markdown(f"`{format_timestamp(s['start'])}` {esc(s['text'])}")
                if st.button("Save this explanation", key="save_range"):
                    store(v["subject"]).save_material("lecture section", f"{v['title']} {res['start_label']}-{res['end_label']}",
                                                      "\n".join(f"[{s['timestamp']}] {s['text']}" for s in res["summary"]))
                    st.success(f"Saved in {v['subject']} > Saved.")

    st.markdown("#### Ask a question about this lecture")
    q = st.text_input("Question", placeholder="e.g. what does dropout do?  /  dropout kya karta hai?")
    if q:
        ans = ask_lecture(engine, vid, q, method=effective_method())
        badges("lecture")
        if not ans["ok"]:
            st.info("This lecture does not seem to discuss that clearly - the closest sentences are shown.")
        for a in ans["answers"]:
            st.markdown(f"<div class='cs-quote'><a class='cs-ts' href='{a['url']}' target='_blank'>{a['timestamp']}</a> "
                        f"{esc(a['text'])} <span class='cs-muted'>(similarity {a['score']:.2f})</span></div>", unsafe_allow_html=True)
