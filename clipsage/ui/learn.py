"""Topic Learning and Practice & Quizzes pages."""

from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components

from ..i18n import UI
from ..knowledge import find_topic, load_topics
from ..quiz import CORRECT, PARTIAL, grade
from ..utils import format_timestamp
from ..visualize import concept_map_html, keyword_cloud, timeline_figure
from .common import (P, S, T, TL, badges, bi_html, bullet_list, card, chips, esc, get_lesson, get_progress, goto,
                     lang, open_topic, show, topic_search_box, ui_text)

KIND_LABELS = {
    "basic": {"en": "Basic understanding", "ru": "Buniyadi samajh"},
    "conceptual": {"en": "Conceptual", "ru": "Conceptual"},
    "short": {"en": "Short answer", "ru": "Mukhtasar jawab"},
    "exam": {"en": "Exam-style", "ru": "Exam jaisa"},
    "application": {"en": "Application", "ru": "Amli istemal"},
    "problem": {"en": "Problem solving", "ru": "Masla hal karein"},
    "lecture": {"en": "From the lecture", "ru": "Lecture se"},
}
TYPE_LABELS = {"mcq": "MCQ", "numerical": {"en": "Numerical", "ru": "Numerical"}, "short": {"en": "Written", "ru": "Likhit"}}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def render_question(topic_id: str, q: dict, prefix: str, number: int) -> Optional[dict]:
    key = f"{prefix}_{q['id']}"
    state = st.session_state.setdefault(key, {"result": None, "hints": 0, "solution": False, "attempts": 0, "round": 0})
    kind = S(KIND_LABELS.get(q.get("kind"), {"en": q.get("kind", ""), "ru": q.get("kind", "")}))
    qtype = S(TYPE_LABELS.get(q["type"], q["type"]))
    with st.container(border=True):
        st.markdown(f"<span class='cs-chip'>{esc(kind)}</span><span class='cs-chip'>{esc(qtype)}</span>",
                    unsafe_allow_html=True)
        if q.get("source"):
            src = q["source"]
            st.markdown(f"**Q{number}.** {esc(T('fill_blank', title=src['title'], ts=src['timestamp']))}")
            st.markdown(f"<div class='cs-quote'>{esc(q['q']['en'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<a class='cs-ts' href='{src['url']}' target='_blank'>▶ {esc(src['title'])} @ "
                        f"{src['timestamp']}</a>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Q{number}.**", unsafe_allow_html=True)
            show(q["q"])

        wkey = f"{key}_in_{state['round']}"
        if q["type"] == "mcq":
            response = st.radio(TL("your_answer"), list(range(len(q["options"]))), index=None, key=wkey,
                                format_func=lambda i: S(q["options"][i]))
        elif q["type"] == "numerical":
            response = st.text_input(TL("your_answer"), key=wkey, placeholder="e.g. 0.75 or 3/4")
        else:
            response = st.text_area(TL("your_answer"), key=wkey, height=90)

        c1, c2, c3, c4 = st.columns(4)
        if c1.button(TL("submit"), key=f"{key}_submit_{state['round']}", type="primary", width="stretch"):
            if response is None or (isinstance(response, str) and not response.strip()):
                st.warning(TL("your_answer") + " ?")
            else:
                result = grade(q, response)
                state["result"], state["attempts"] = result, state["attempts"] + 1
                get_progress().record_answer(topic_id, q["id"], result["status"] == CORRECT, result.get("score"))
        if c2.button(TL("hint"), key=f"{key}_hint", width="stretch", disabled=state["hints"] >= len(q.get("hints", []))):
            state["hints"] += 1
        if c3.button(TL("show_solution"), key=f"{key}_sol", width="stretch", disabled=state["attempts"] == 0):
            state["solution"] = True
        if c4.button(TL("retry"), key=f"{key}_retry", width="stretch"):
            state.update(result=None, hints=0, solution=False, round=state["round"] + 1)
            st.rerun()

        for h in q.get("hints", [])[:state["hints"]]:
            st.markdown(f"<div class='cs-quote'>\U0001F4A1 {bi_html(h)}</div>", unsafe_allow_html=True)

        result = state["result"]
        if result:
            status = result["status"]
            msg = {CORRECT: "correct", PARTIAL: "partly"}.get(status, "incorrect")
            (st.success if status == CORRECT else st.warning if status == PARTIAL else st.error)(T(msg))
            if q["type"] == "short" and result.get("method"):
                st.caption(f"Similarity ({result['method'].upper()}): {result['similarity']:.2f} · "
                           f"key terms {len(result['matched'])}/{len(result['matched']) + len(result['missing'])}"
                           + (f" · missing: {', '.join(result['missing'])}" if result["missing"] else ""))
                show(UI["short_grading"])
            if q["type"] == "numerical" and result.get("error"):
                st.caption("Please enter a number, e.g. 0.75 or 3/4.")
        if (result and result["status"] == CORRECT) or state["solution"]:
            answer = q["answer"]
            if q["type"] == "mcq":
                answer_text = S(q["options"][answer])
            elif q["type"] == "short":
                answer_text = answer
            else:
                answer_text = f"{answer:g}"
            st.markdown(f"**{TL('final_answer')}:** {esc(answer_text)}")
            show(q["explanation"])
    return state["result"]


def render_question_set(topic_id: str, questions: List[dict], prefix: str) -> None:
    if not questions:
        st.info(T("no_questions"))
        return
    results = [render_question(topic_id, q, prefix, i + 1) for i, q in enumerate(questions)]
    done = [r for r in results if r]
    correct = sum(1 for r in done if r["status"] == CORRECT)
    st.progress(correct / len(questions), text=f"{TL('score')}: {correct} / {len(questions)}")


# ---------------------------------------------------------------------------
# Lecture references
# ---------------------------------------------------------------------------
def _reason_text(code: str, args: dict) -> dict:
    entry = UI[code]
    return {k: v.format(**args) for k, v in entry.items()}


def render_reference(ref: dict, i: int) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"<a href='{ref['url']}' target='_blank'><img class='cs-thumb' src='{ref['thumbnail']}' "
                        f"alt='thumbnail'></a>", unsafe_allow_html=True)
            st.link_button(TL("watch_segment", ts=ref["start_label"]), ref["url"], width="stretch")
            if st.button(TL("nav_videos"), key=f"ref_nav_{i}_{ref['video_id']}", width="stretch"):
                get_progress().add_viewed(ref["video_id"], ref["title"], ref["channel_title"], ref["start"])
                goto("videos", video_id=ref["video_id"], video_start=ref["start"])
        with c2:
            st.markdown(f"#### [{esc(ref['title'])}]({ref['url']})")
            st.markdown(f"<span class='cs-muted'>{esc(ref['channel_title'])}</span> · "
                        f"**{esc(TL('segment'))}: <a class='cs-ts' href='{ref['url']}' target='_blank'>"
                        f"{ref['start_label']}–{ref['end_label']}</a>**", unsafe_allow_html=True)
            if ref["concepts"]:
                st.markdown(f"<span class='cs-muted'>{esc(TL('concepts_here'))}:</span> {chips(ref['concepts'])}",
                            unsafe_allow_html=True)
            if lang() != "en":
                st.markdown(f"<p class='cs-ru'>{esc(UI['segment_summary_ru']['ru'].format(start=ref['start_label'], end=ref['end_label'], terms=', '.join(ref['concepts'][:4]) or '-'))}</p>",
                            unsafe_allow_html=True)
            if ref["excerpt"]:
                text = " ".join(s["text"] for s in ref["excerpt"])
                st.markdown(f"<span class='cs-muted'>{esc(TL('excerpt'))} ({ref['excerpt_start']}):</span>"
                            f"<div class='cs-quote'>“{esc(text)}”</div>", unsafe_allow_html=True)
            for d in ref["definitions"][:1]:
                st.markdown(f"<div class='cs-quote cs-def'>{esc(d['text'])} "
                            f"<span class='cs-muted'>({format_timestamp(d['start'])})</span></div>", unsafe_allow_html=True)
            st.markdown(f"<span class='cs-muted'>{esc(TL('why_useful'))}:</span>", unsafe_allow_html=True)
            bullet_list([_reason_text(code, args) for code, args in ref["reasons"]])
            if ref.get("transcript_kind") == "auto":
                st.caption("Auto-generated captions: wording may contain small transcription errors.")


def render_lecture_explanation(result: dict) -> None:
    ref_no = {r["video_id"]: r["n"] for r in result["references"]}
    parts = []
    for s in result["explanation"]:
        cls = "cs-def" if s.get("is_definition") else ""
        parts.append(f"<p class='{cls}'>{esc(s['text'])} <a class='cs-ts' href='{s['url']}' target='_blank'>"
                     f"[{ref_no[s['video_id']]}] {s['timestamp']}</a></p>")
    card("", "".join(parts))
    for r in result["references"]:
        if r["used_in_explanation"]:
            stamps = " · ".join(f"[{t['label']}]({t['url']})" for t in r["timestamps"])
            st.markdown(f"**[{r['n']}] {esc(r['title'])}** — {esc(r['channel_title'])}  \n▶ {stamps}")


def render_visuals(result: dict) -> None:
    tab_map, tab_cloud, tab_time, tab_sub = st.tabs(["Concept map", "Keyword cloud", "Timeline", "Sub-topics (K-Means)"])
    with tab_map:
        components.html(concept_map_html(result["concept_graph"]), height=540, scrolling=False)
        st.caption("Terms that appear together in the retrieved lecture sentences are linked (co-occurrence network).")
    with tab_cloud:
        image = keyword_cloud(result["key_terms"])
        if image is not None:
            st.image(image, width="stretch")
    with tab_time:
        st.plotly_chart(timeline_figure(result["hits"]), width="stretch")
    with tab_sub:
        if result["subtopics"]:
            for g in result["subtopics"]:
                with st.expander(f"{g['label']}  ({g['size']} segments)"):
                    for h in sorted(g["segments"], key=lambda h: h["rank"]):
                        link = f"https://www.youtube.com/watch?v={h['video_id']}&t={int(h['start'])}s"
                        st.markdown(f"- [{h['title']} @ {format_timestamp(h['start'])}]({link})")
        else:
            st.write("Not enough segments to group into sub-topics.")


# ---------------------------------------------------------------------------
# Topic Learning page
# ---------------------------------------------------------------------------
def _section(title_key: str, *kinds: str) -> None:
    st.markdown(f"### {esc(TL(title_key))}")
    if kinds:
        badges(*kinds)


def learn_page() -> None:
    topic_search_box("learn_search")
    query = st.session_state.get("topic", "").strip()
    if not query:
        st.title(TL("nav_learn"))
        show({"en": "Search a topic above or pick one in Explore Topics.",
              "ru": "Upar koi topic search karein ya Explore Topics se chunein."})
        return

    with st.spinner("..."):
        lesson = get_lesson(query)
    notes, result = lesson["notes"], lesson["lecture"]
    covered = bool(result and result["covered"])
    store = get_progress()
    topic_id = notes["id"] if notes else f"lecture:{query.lower()}"
    title = S(notes["title"]) if notes else query

    head, save_col = st.columns([4, 1.3])
    head.title(title)
    saved = store.is_saved(topic_id)
    if save_col.button(("⭐ " + TL("saved")) if saved else ("☆ " + TL("save_topic")), width="stretch"):
        store.toggle_saved(topic_id, notes["title"]["en"] if notes else query)
        st.rerun()

    if lesson.get("error"):
        st.warning(f"Lecture retrieval failed: {lesson['error']}")
    if not lesson["index_available"]:
        st.info(T("no_index"))

    if not notes and not covered:
        st.warning(T("not_covered", q=query))
        suggestions = (result or {}).get("suggestions") or []
        if suggestions:
            st.write(T("try_related"))
            cols = st.columns(min(5, len(suggestions)))
            for col, s in zip(cols, suggestions):
                col.button(s, key=f"sugg_{s}", on_click=open_topic, args=(s,))
        return

    if st.session_state.get("_last_studied") != topic_id:
        store.add_studied(topic_id, notes["title"]["en"] if notes else query)
        st.session_state["_last_studied"] = topic_id

    tab_names = []
    if notes:
        tab_names.append("\U0001F4D8 " + S({"en": "Lesson", "ru": "Sabaq"}))
    tab_names.append("\U0001F3A5 " + TL("sec_lectures"))
    if covered:
        tab_names.append("\U0001F5FA️ " + S({"en": "Visual overview", "ru": "Visual jaiza"}))
    if notes and notes.get("code"):
        tab_names.append("\U0001F4BB " + S({"en": "Algorithm & code", "ru": "Algorithm aur code"}))
    if notes:
        tab_names.append("\U0001F4DD " + S({"en": "Exam prep", "ru": "Exam ki tayyari"}))
    tab_names.append("✍️ " + TL("sec_practice"))
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    tab_iter = iter(tabs.values())

    if notes:
        with next(tab_iter):
            badges("notes")
            _section("sec_intro")
            show(notes["intro"])
            _section("sec_definition")
            st.markdown(f"<div class='cs-card'>{bi_html(notes['definition'])}</div>", unsafe_allow_html=True)
            _section("sec_explanation")
            show(notes["explanation"])
            _section("sec_example", "example")
            st.markdown(f"<div class='cs-card'>{bi_html(notes['example'])}</div>", unsafe_allow_html=True)
            _section("sec_concepts")
            cols = st.columns(2)
            for i, c in enumerate(notes.get("key_concepts", [])):
                with cols[i % 2]:
                    card(c["term"], bi_html(c))
            if notes.get("formulas"):
                _section("sec_formulas")
                for f in notes["formulas"]:
                    st.latex(f["latex"])
                    show(f, "cs-muted")
            _section("sec_applications")
            bullet_list(notes.get("applications", []))

    with next(tab_iter):
        if covered:
            badges("lecture")
            show(UI["transcript_english"])
            st.markdown(f"### {esc(TL('sec_lecture_explain'))}")
            render_lecture_explanation(result)
            st.markdown(f"### {esc(TL('sec_lectures'))}")
            show(UI["timestamps_exact"], "cs-muted")
            for i, ref in enumerate(lesson["refs"]):
                render_reference(ref, i)
        elif lesson["index_available"]:
            st.info(T("not_covered", q=query))
        else:
            st.info(T("no_index"))

    if covered:
        with next(tab_iter):
            badges("lecture")
            render_visuals(result)

    if notes and notes.get("code"):
        with next(tab_iter):
            badges("notes")
            if notes.get("workflow"):
                _section("sec_workflow")
                items = "".join(f"<li>{bi_html(step)}</li>" for step in notes["workflow"])
                st.markdown(f"<div class='cs-card'><ol>{items}</ol></div>", unsafe_allow_html=True)
            _section("sec_code")
            st.code(notes["code"], language="python")
            if notes.get("code_needs_model"):
                st.caption("This code downloads the Sentence-BERT model (~90 MB) the first time it runs.")
            _section("sec_code_explain")
            bullet_list(notes.get("code_explanation", []))
            if notes.get("playground"):
                st.markdown(f"### {esc(TL('sec_try_playground'))}")
                if st.button("\U0001F9EA " + TL("nav_ml"), type="primary"):
                    goto("ml", ml_algorithm=notes["playground"])
            if notes.get("math_tool"):
                if st.button("\U0001F9EE " + TL("nav_math")):
                    goto("math")

    if notes:
        with next(tab_iter):
            badges("notes")
            _section("sec_exam")
            bullet_list(notes.get("exam_points", []))
            _section("sec_mistakes")
            bullet_list(notes.get("common_mistakes", []))

    with next(tab_iter):
        show(UI["quiz_instructions"])
        if notes:
            render_question_set(topic_id, notes.get("questions", []), "learn")
        if lesson["lecture_questions"]:
            st.markdown(f"### {esc(TL('lecture_questions'))}")
            badges("lecture")
            show(UI["lecture_q_note"], "cs-muted")
            render_question_set(topic_id, lesson["lecture_questions"], "learnlec")


# ---------------------------------------------------------------------------
# Practice & Quizzes page
# ---------------------------------------------------------------------------
def practice_page() -> None:
    st.title(TL("nav_practice"))
    show(UI["quiz_instructions"])
    topics = load_topics()
    ids = list(topics)
    current = find_topic(st.session_state.get("topic", "")) if st.session_state.get("topic") else None
    default = ids.index(current["id"]) if current else 0
    choice = st.selectbox(TL("choose_topic"), ids, index=default, format_func=lambda i: S(topics[i]["title"]))
    topic = topics[choice]
    kinds = sorted({q.get("kind", "basic") for q in topic["questions"]})
    picked = st.multiselect("Question types", kinds, default=kinds, format_func=lambda k: S(KIND_LABELS.get(k, {"en": k, "ru": k})))
    questions = [q for q in topic["questions"] if q.get("kind", "basic") in picked]
    badges("notes")
    render_question_set(topic["id"], questions, "practice")

    lesson = get_lesson(topic["title"]["en"]) if st.session_state.get("pages") else None
    if lesson and lesson["lecture_questions"]:
        st.markdown(f"### {esc(TL('lecture_questions'))}")
        badges("lecture")
        show(UI["lecture_q_note"], "cs-muted")
        render_question_set(topic["id"], lesson["lecture_questions"], "practicelec")
    elif lesson and not lesson["index_available"]:
        st.caption(T("no_index"))
