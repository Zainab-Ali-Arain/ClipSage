"""Video Lectures page: lecture list, timeline navigation, time-range and segment questions."""

import plotly.graph_objects as go
import streamlit as st

from ..i18n import UI
from ..lectures import (ask_lecture, explain_range, list_videos, most_useful_sections, parse_time_range,
                        video_info, video_timeline)
from ..utils import format_timestamp
from .common import T, TL, badges, bi_html, chips, effective_method, esc, get_progress, get_sage, show


@st.cache_data(show_spinner=False)
def _videos(_sage, n_segments: int):
    return list_videos(_sage.engine)


def _timeline_chart(sections, video_title: str) -> go.Figure:
    fig = go.Figure()
    for s in sections:
        rel = s["relevance"]
        color = "#4A90E2" if rel is None else f"rgba(74,144,226,{0.25 + 0.75 * max(0.0, min(1.0, rel / max(1e-9, max(x['relevance'] or 0 for x in sections))))})"
        fig.add_trace(go.Bar(x=[(s["end"] - s["start"]) / 60], base=[s["start"] / 60], y=["lecture"], orientation="h",
                             marker=dict(color=color, line=dict(color="white", width=1)), showlegend=False,
                             hovertemplate=f"{s['start_label']}–{s['end_label']}<br>{', '.join(s['terms'][:4])}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=150, template="plotly_white", paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#F5F9FF", margin=dict(l=10, r=10, t=30, b=40), yaxis=dict(visible=False),
                      xaxis_title="minutes", title="Sections (darker = more relevant to your topic)")
    return fig


def videos_page() -> None:
    st.title(TL("nav_videos"))
    sage = get_sage()
    if sage is None:
        st.info(T("no_index"))
        return
    engine = sage.engine
    videos = _videos(sage, len(engine.segments))
    method = effective_method()

    c1, c2 = st.columns([2, 1])
    search = c1.text_input(TL("search"), placeholder="title or channel", key="video_filter")
    subjects = sorted({v["subject"] for v in videos if v["subject"]})
    subject = c2.selectbox("Subject", ["All"] + subjects)
    shown = [v for v in videos if (subject == "All" or v["subject"] == subject)
             and (not search or search.lower() in (v["title"] + " " + v["channel_title"]).lower())]
    if not shown:
        st.info(T("nothing_yet"))
        return
    ids = [v["video_id"] for v in shown]
    current = st.session_state.get("video_id")
    index = ids.index(current) if current in ids else 0
    by_id = {v["video_id"]: v for v in shown}
    vid = st.selectbox(TL("pick_video"), ids, index=index,
                       format_func=lambda i: f"{by_id[i]['title']} — {by_id[i]['channel_title']}")
    info = video_info(engine, vid)
    if vid != st.session_state.get("video_id"):
        st.session_state["video_id"] = vid
        st.session_state.pop("video_start", None)
        get_progress().add_viewed(vid, info["title"], info["channel_title"], 0)

    topic = st.session_state.get("topic", "")
    sections = video_timeline(engine, vid, query=topic or None, method=method)

    left, right = st.columns([3, 2], gap="large")
    with left:
        start = int(st.session_state.get("video_start", 0))
        st.video(f"https://www.youtube.com/watch?v={vid}", start_time=start)
        st.markdown(f"**{esc(info['title'])}**  \n<span class='cs-muted'>{esc(info['channel_title'])} · "
                    f"{info['segments']} sections · transcript {format_timestamp(info['transcript_start'])}–"
                    f"{format_timestamp(info['transcript_end'])}</span>", unsafe_allow_html=True)
    with right:
        badges("lecture")
        if topic:
            st.markdown(f"#### {esc(TL('most_useful', topic=topic))}")
            for s in most_useful_sections(sections):
                st.markdown(f"<div class='cs-card'><a class='cs-ts' href='{s['url']}' target='_blank'>"
                            f"{s['start_label']}–{s['end_label']}</a> {chips(s['terms'][:4])}</div>",
                            unsafe_allow_html=True)
                if st.button(f"▶ {s['start_label']}", key=f"jump_top_{s['row']}"):
                    st.session_state["video_start"] = s["start"]
                    st.rerun()

    st.plotly_chart(_timeline_chart(sections, info["title"]), width="stretch")

    st.markdown(f"### {esc(TL('video_timeline'))}")
    show(UI["timestamps_exact"], "cs-muted")
    for s in sections:
        with st.expander(f"{s['start_label']}–{s['end_label']}  ·  {', '.join(s['terms'][:4])}"):
            st.markdown(f"<div class='cs-quote'>{esc(s['first_sentence'])} …</div>", unsafe_allow_html=True)
            if s["definitions"]:
                st.markdown(f"**{esc(TL('definitions_here'))}**")
                for d in s["definitions"]:
                    st.markdown(f"<div class='cs-quote cs-def'>{esc(d['text'])} "
                                f"<span class='cs-muted'>({format_timestamp(d['start'])})</span></div>",
                                unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button(f"▶ {TL('watch_segment', ts=s['start_label'])}", key=f"jump_{s['row']}"):
                st.session_state["video_start"] = s["start"]
                st.rerun()
            c2.link_button("YouTube ↗", s["url"])

    st.divider()
    st.markdown(f"#### {esc(TL('ask_range'))}")
    q_range = st.text_input("range", label_visibility="collapsed", placeholder="What is taught between 0:30 and 1:30?",
                            key="range_q")
    if q_range:
        rng = parse_time_range(q_range)
        if rng is None:
            st.warning(T("range_parse"))
        else:
            res = explain_range(engine, vid, rng[0], rng[1], method=method)
            if not res["ok"]:
                info_ = res.get("info") or info
                st.warning(T("range_out", start=format_timestamp(info_["transcript_start"]),
                             end=format_timestamp(info_["transcript_end"])))
            else:
                badges("lecture")
                show({k: v.format(start=res["start_label"], end=res["end_label"], terms=", ".join(res["terms"][:6]))
                      for k, v in UI["range_summary"].items()})
                for s in res["summary"]:
                    st.markdown(f"<div class='cs-quote'><a class='cs-ts' href='{s['url']}' target='_blank'>"
                                f"{s['timestamp']}</a> {esc(s['text'])}</div>", unsafe_allow_html=True)
                if res["definitions"]:
                    st.markdown(f"**{esc(TL('definitions_here'))}**")
                    for d in res["definitions"]:
                        st.markdown(f"<div class='cs-quote cs-def'>{esc(d['text'])}</div>", unsafe_allow_html=True)
                with st.expander(f"Full transcript {res['start_label']}–{res['end_label']} "
                                 f"({len(res['sentences'])} sentences)"):
                    for s in res["sentences"]:
                        st.markdown(f"`{format_timestamp(s['start'])}` {esc(s['text'])}")

    st.markdown(f"#### {esc(TL('ask_segment'))}")
    q = st.text_input("question", label_visibility="collapsed", placeholder="e.g. what does dropout do?", key="lec_q")
    if q:
        ans = ask_lecture(engine, vid, q, method=method)
        badges("lecture")
        show(UI["answer_from_lecture"])
        if not ans["ok"]:
            st.info("This lecture does not seem to discuss that clearly - the closest sentences are shown below. / "
                    "Yeh lecture is sawal par wazeh baat nahi karta - sab se qareebi sentences neeche hain.")
        for a in ans["answers"]:
            st.markdown(f"<div class='cs-quote'><a class='cs-ts' href='{a['url']}' target='_blank'>{a['timestamp']}</a> "
                        f"{esc(a['text'])} <span class='cs-muted'>(similarity {a['score']:.2f})</span></div>",
                        unsafe_allow_html=True)
