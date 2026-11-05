"""Dedicated learning environment for one subject (selected in the sidebar subject dropdown)."""

import random
import time

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..library import library_stats, load_library
from ..subjects import get_subject, load_subjects, topics_by_category, topics_of
from .common import (activity, badges, card, chips, content, content_html, esc, open_topic, request_page,
                     search_box, show_content, stat, store)
from .tools import algorithm_practice, numerical_practice
from .topic_view import render_question_set, render_topic

LAYOUT = dict(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF", margin=dict(l=30, r=20, t=40, b=30),
              font=dict(color="#243447"))
PALETTE = ["#4A90E2", "#F5A623", "#7ED321", "#BD10E0", "#50E3C2", "#D0021B"]


def current_subject():
    name = st.session_state.get("subject")
    s = get_subject(name) if name else None
    return s or load_subjects()[0]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def relationship_figure(subject_name: str):
    """Topic relationship diagram: topics are linked when their lessons share vocabulary (TF-IDF cosine)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    topics = topics_of(subject_name)
    if len(topics) < 2:
        return None
    docs = [" ".join([t["title"]["en"], " ".join(t.get("aliases", [])), t["definition"]["en"], t["explanation"]["en"]])
            for t in topics]
    m = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    sim = (m @ m.T).toarray()
    g = nx.Graph()
    cats = sorted({t["category"] for t in topics})
    for i, t in enumerate(topics):
        g.add_node(i, label=t["title"]["en"], cat=t["category"])
    for i in range(len(topics)):
        for j in np.argsort(-sim[i])[1:3]:          # two strongest links per topic
            if sim[i, j] > 0.05:
                g.add_edge(i, int(j), weight=float(sim[i, j]))
    pos = nx.spring_layout(g, seed=3, k=1.1)
    fig = go.Figure()
    for a, b, d in g.edges(data=True):
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines", hoverinfo="skip",
                                 line=dict(color="#B9D7F7", width=1 + 8 * d["weight"]), showlegend=False))
    for c_i, cat in enumerate(cats):
        nodes = [n for n in g.nodes if g.nodes[n]["cat"] == cat]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                                 text=[g.nodes[n]["label"] for n in nodes], textposition="top center",
                                 marker=dict(size=22, color=PALETTE[c_i % len(PALETTE)], line=dict(color="white", width=2)),
                                 name=get_subject(subject_name)["categories"].get(cat, cat)))
    fig.update_layout(title="Topic relationships (linked when lessons share key concepts)", height=440,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h"), **LAYOUT)
    return fig


def overview_tab(subject: dict) -> None:
    name = subject["name"]
    topics = topics_of(name)
    stats = library_stats()
    saved = store(name).counts()
    summary = activity().subject_summary(name)
    st.markdown(f"<div class='cs-hero'><h1>{esc(name)}</h1>{content_html(subject['intro'])}</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].markdown(stat(len(topics), "Topics"), unsafe_allow_html=True)
    cols[1].markdown(stat(sum(len(t["questions"]) for t in topics), "Quiz questions"), unsafe_allow_html=True)
    cols[2].markdown(stat(stats["per_subject"].get(name, 0), "Videos in library"), unsafe_allow_html=True)
    cols[3].markdown(stat(len(subject["tools"]), "Practice tools"), unsafe_allow_html=True)
    cols[4].markdown(stat(sum(saved.values()), "Saved items"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Topics by category")
        for cat, items in topics_by_category(name).items():
            st.markdown(f"**{esc(cat)}**")
            grid = st.columns(3)
            for i, t in enumerate(items):
                with grid[i % 3]:
                    answered = summary["per_topic"].get(t["id"], {}).get("answered", 0)
                    card(t["title"]["en"], f"<span class='cs-muted'>{len(t['questions'])} questions"
                                           f" - {stats['per_topic'].get(t['id'], 0)} videos</span>")
                    st.progress(min(1.0, answered / max(1, len(t["questions"]))), text=f"{answered} answered this session")
                    st.button("Open", key=f"ov_open_{t['id']}", on_click=open_topic, args=(t["id"], name), width="stretch")
    with right:
        st.markdown("#### Learning progress (this session)")
        acc = summary["accuracy"]
        st.markdown(stat(f"{summary['correct']}/{summary['answered']}" + (f"  ({acc:.0%})" if acc is not None else ""),
                         "Correct answers"), unsafe_allow_html=True)
        st.caption("Progress is temporary until you click Save Progress.")
        if st.button("Save Progress", type="primary", key=f"save_progress_{subject['id']}", disabled=summary["answered"] == 0
                     and not summary["studied"]):
            store(name).save_progress({k: v for k, v in summary.items() if k != "subject"})
            st.success(f"Progress saved in {name} > Saved.")
        fig = go.Figure([
            go.Bar(y=[t["title"]["en"] for t in topics], x=[len(t["questions"]) for t in topics], orientation="h",
                   name="questions available", marker_color="#B9D7F7"),
            go.Bar(y=[t["title"]["en"] for t in topics], x=[stats["per_topic"].get(t["id"], 0) for t in topics],
                   orientation="h", name="videos", marker_color="#4A90E2"),
            go.Bar(y=[t["title"]["en"] for t in topics], x=[summary["per_topic"].get(t["id"], {}).get("correct", 0) for t in topics],
                   orientation="h", name="correct this session", marker_color="#7ED321")])
        fig.update_layout(barmode="group", height=120 + 46 * len(topics), title="Resources and progress per topic",
                          legend=dict(orientation="h"), **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        studied = summary["studied"]
        st.markdown("**Recently studied (this session)**")
        st.markdown(chips(studied[:8]) or "<span class='cs-muted'>Nothing yet.</span>", unsafe_allow_html=True)
        saved_topics = store(name).load()["saved_topics"]
        st.markdown("**Saved topics**")
        st.markdown(chips(t["title"] for t in saved_topics[:8]) or "<span class='cs-muted'>None saved.</span>",
                    unsafe_allow_html=True)
    fig = relationship_figure(name)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Other tabs
# ---------------------------------------------------------------------------
def theory_tab(subject: dict) -> None:
    topics = topics_of(subject["name"])
    ids = [t["id"] for t in topics]
    by_id = {t["id"]: t for t in topics}
    wanted = st.session_state.get("topic_id")
    index = ids.index(wanted) if wanted in ids else 0
    tid = st.selectbox("Topic", ids, index=index, format_func=lambda i: by_id[i]["title"]["en"], key=f"theory_{subject['id']}")
    st.session_state["topic_id"] = tid
    render_topic(by_id[tid], key=f"th_{subject['id']}")


def videos_tab(subject: dict) -> None:
    name = subject["name"]
    videos = [v for v in load_library() if v["subject"] == name]
    topics = {t["id"]: t for t in topics_of(name)}
    if not videos:
        st.info("No videos have been collected for this subject yet. Run run_pipeline.bat - the collector searches "
                "YouTube for every topic of this subject and keeps captioned, relevant videos.")
        return
    st.caption(f"{len(videos)} videos - {sum(v['has_transcript'] for v in videos)} with captions (timestamps available).")
    for tid in list(topics) + [None]:
        group = [v for v in videos if v["topic_id"] == tid]
        if not group:
            continue
        st.markdown(f"#### {esc(topics[tid]['title']['en']) if tid else 'Other lectures in this subject'}")
        grid = st.columns(3)
        for i, v in enumerate(group):
            with grid[i % 3]:
                mins = f"{v['duration'] // 60} min" if v["duration"] else ""
                st.markdown(f"<div class='cs-card'><a href='{v['url']}' target='_blank'><img class='cs-thumb' src='{v['thumbnail']}' "
                            f"alt='thumbnail'></a><b>{esc(v['title'])}</b><br><span class='cs-muted'>{esc(v['channel_title'])} "
                            f"{mins} - {'captions' if v['has_transcript'] else 'no captions'}</span></div>", unsafe_allow_html=True)
                st.button("Open with timestamps", key=f"vt_{subject['id']}_{v['video_id']}", on_click=request_page,
                          args=("videos",), kwargs={"video_id": v["video_id"], "video_start": 0}, width="stretch")


def quiz_tab(subject: dict) -> None:
    name = subject["name"]
    topics = topics_of(name)
    by_id = {t["id"]: t for t in topics}
    c1, c2, c3 = st.columns([2, 1, 1])
    choice = c1.selectbox("Quiz", ["__mixed__"] + list(by_id),
                          format_func=lambda i: "Mixed quiz (whole subject)" if i == "__mixed__" else by_id[i]["title"]["en"],
                          key=f"quiz_topic_{subject['id']}")
    levels = c2.multiselect("Difficulty", ["easy", "medium", "hard"], default=["easy", "medium", "hard"], key=f"ql_{subject['id']}")
    types = c3.multiselect("Type", ["mcq", "tf", "short", "numerical"], default=["mcq", "tf", "short", "numerical"],
                           format_func={"mcq": "Multiple choice", "tf": "True/False", "short": "Short answer",
                                        "numerical": "Numerical"}.get, key=f"qt_{subject['id']}")
    if choice == "__mixed__":
        pool = [q for t in topics for q in t["questions"]]
        seed = st.session_state.setdefault(f"mix_seed_{subject['id']}", random.randint(0, 10 ** 6))
        if st.button("New mixed quiz", key=f"mixnew_{subject['id']}"):
            st.session_state[f"mix_seed_{subject['id']}"] = seed = random.randint(0, 10 ** 6)
        pool = [q for q in pool if q.get("difficulty", "medium") in levels and q["type"] in types]
        questions = random.Random(seed).sample(pool, min(8, len(pool)))
        render_question_set(name, "mixed", f"{name} mixed quiz", questions, f"mix_{subject['id']}_{seed}")
    else:
        t = by_id[choice]
        questions = [q for q in t["questions"] if q.get("difficulty", "medium") in levels and q["type"] in types]
        render_question_set(name, t["id"], t["title"]["en"], questions, f"quiz_{subject['id']}")


def saved_tab(subject: dict) -> None:
    name = subject["name"]
    s = store(name)
    data = s.load()
    st.caption(f"Saved items for {name} only. Stored on this computer in data/user/subjects/{s.path.name} - "
               "written only when you click a Save button.")
    if not s.exists():
        st.info("Nothing saved for this subject yet.")
        return
    st.markdown("#### Saved topics")
    for t in data["saved_topics"]:
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.markdown(f"**{esc(t['title'])}** <span class='cs-muted'>{time.strftime('%Y-%m-%d %H:%M', time.localtime(t['ts']))}</span>",
                    unsafe_allow_html=True)
        c2.button("Open", key=f"sv_open_{t['id']}", on_click=open_topic, args=(t["id"], name))
        if c3.button("Remove", key=f"sv_rm_{t['id']}"):
            s.remove("saved_topics", t["id"]); st.rerun()
    st.markdown("#### Notes")
    for tid, n in data["notes"].items():
        with st.expander(n["title"]):
            st.write(n["text"])
            if st.button("Delete note", key=f"sv_note_{tid}"):
                s.remove("notes", tid); st.rerun()
    st.markdown("#### Quiz results")
    for r in data["quiz_results"]:
        c1, c2 = st.columns([5, 1])
        c1.write(f"{r['title']}: {r['correct']} / {r['answered']} correct - "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))}")
        if c2.button("Delete", key=f"sv_q_{r['id']}"):
            s.remove("quiz_results", r["id"]); st.rerun()
    st.markdown("#### Progress snapshots")
    for p in data["progress"]:
        c1, c2 = st.columns([5, 1])
        acc = f"{p['accuracy']:.0%}" if p.get("accuracy") is not None else "-"
        c1.write(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p['ts']))}: {p['correct']}/{p['answered']} correct ({acc}), "
                 f"studied: {', '.join(p.get('studied', [])) or '-'}")
        if c2.button("Delete", key=f"sv_p_{p['id']}"):
            s.remove("progress", p["id"]); st.rerun()
    st.markdown("#### Learning material and practice results")
    for mtl in data["materials"]:
        with st.expander(f"{mtl['title']} ({mtl['kind']})"):
            st.text(mtl["content"])
            if st.button("Delete", key=f"sv_m_{mtl['id']}"):
                s.remove("materials", mtl["id"]); st.rerun()
    st.divider()
    if st.button(f"Delete all saved data for {name}", key=f"sv_clear_{subject['id']}"):
        s.clear(); st.rerun()


SECTIONS = ["Overview", "Topics & Theory", "Video References", "Quiz Practice", "Algorithm Practice",
            "Numerical Practice", "Saved", "Search"]


def subject_page() -> None:
    subject = current_subject()
    st.session_state["subject"] = subject["name"]
    st.session_state.setdefault("subject_section", "Overview")
    section = st.segmented_control("Section", SECTIONS, key="subject_section", label_visibility="collapsed") or "Overview"
    if section == "Overview":
        overview_tab(subject)
    elif section == "Topics & Theory":
        theory_tab(subject)
    elif section == "Video References":
        videos_tab(subject)
    elif section == "Quiz Practice":
        quiz_tab(subject)
    elif section == "Algorithm Practice":
        algorithm_practice(subject)
    elif section == "Numerical Practice":
        numerical_practice(subject)
    elif section == "Saved":
        saved_tab(subject)
    else:
        st.caption(f"Search only inside {subject['name']}.")
        search_box(f"subj_search_{subject['id']}", subject=subject["name"],
                   placeholder=f"Search {subject['name']} - English or Roman Urdu")
