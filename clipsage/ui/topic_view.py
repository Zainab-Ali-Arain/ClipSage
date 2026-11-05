"""
Topic view used by Search results and by each subject's Theory tab.

Layout (two columns):
  left  - the explanation in the response language, examples, concepts,
          formulas, code, exam points, notes and Save buttons;
  right - the relevant YouTube video (embedded), its title, the topic it
          covers, the best segment and caption-based subtopic timestamps.
Below: practice questions for the topic.
"""

from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components

from ..i18n import pick
from ..library import video_explanation, videos_for_topic
from ..quiz import CORRECT, PARTIAL, grade
from ..utils import format_timestamp, youtube_link
from ..visualize import concept_map_html, keyword_cloud, timeline_figure
from .common import (activity, badges, bullet_list, card, chips, content, content_html, effective_method, esc,
                     get_lesson, get_sage, lang_badge, rlang, show_content, store)

KIND_LABELS = {"basic": "Basic", "conceptual": "Conceptual", "short": "Short answer", "exam": "Exam-style",
               "application": "Application", "problem": "Problem solving", "lecture": "From the lecture"}
TYPE_LABELS = {"mcq": "Multiple choice", "tf": "True / False", "numerical": "Numerical", "short": "Short answer"}
FEEDBACK = {"correct": {"en": "Correct! Well done.", "ru": "Sahi jawab! Shabash."},
            "partly": {"en": "Partly correct - compare with the full answer.", "ru": "Jawab kuch had tak sahi hai - poore jawab se milayein."},
            "incorrect": {"en": "Not correct yet. Try a hint or try again.", "ru": "Abhi sahi nahi. Hint lein ya dobara koshish karein."}}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def render_question(subject: str, topic_id: str, q: dict, prefix: str, number: int) -> Optional[dict]:
    key = f"{prefix}_{q['id']}"
    state = st.session_state.setdefault(key, {"result": None, "hints": 0, "solution": False, "attempts": 0, "round": 0})
    with st.container(border=True):
        tags = [KIND_LABELS.get(q.get("kind"), q.get("kind", "")), TYPE_LABELS.get(q["type"], q["type"]),
                q.get("difficulty", "medium").title()]
        st.markdown(chips(t for t in tags if t), unsafe_allow_html=True)
        if q.get("source"):
            src = q["source"]
            st.markdown(f"**Q{number}.** Fill in the blank (lecture sentence, {esc(src['title'])} at {src['timestamp']}):")
            st.markdown(f"<div class='cs-quote'>{esc(q['q']['en'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<a class='cs-ts' href='{src['url']}' target='_blank'>Open the lecture at {src['timestamp']}</a>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**Q{number}.** {esc(content(q['q']))}")

        wkey = f"{key}_in_{state['round']}"
        if q["type"] == "mcq":
            response = st.radio("Your answer", list(range(len(q["options"]))), index=None, key=wkey,
                                format_func=lambda i: content(q["options"][i]))
        elif q["type"] == "tf":
            response = st.radio("Your answer", [True, False], index=None, key=wkey, horizontal=True,
                                format_func=lambda v: "True" if v else "False")
        elif q["type"] == "numerical":
            response = st.text_input("Your answer", key=wkey, placeholder="e.g. 0.75 or 3/4")
        else:
            response = st.text_area("Your answer", key=wkey, height=90)

        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Submit", key=f"{key}_submit_{state['round']}", type="primary", width="stretch"):
            if response is None or (isinstance(response, str) and not response.strip()):
                st.warning("Please enter an answer first.")
            else:
                result = grade(q, response)
                state["result"], state["attempts"] = result, state["attempts"] + 1
                activity().record_answer(subject, topic_id, q["id"], result["status"], result.get("score"))
        if c2.button("Show hint", key=f"{key}_hint", width="stretch", disabled=state["hints"] >= len(q.get("hints", []))):
            state["hints"] += 1
        if c3.button("Show solution", key=f"{key}_sol", width="stretch", disabled=state["attempts"] == 0):
            state["solution"] = True
        if c4.button("Try again", key=f"{key}_retry", width="stretch"):
            state.update(result=None, hints=0, solution=False, round=state["round"] + 1)
            st.rerun()

        for h in q.get("hints", [])[:state["hints"]]:
            st.markdown(f"<div class='cs-quote'>Hint: {esc(content(h))}</div>", unsafe_allow_html=True)
        result = state["result"]
        if result:
            status = result["status"]
            msg = FEEDBACK[{CORRECT: "correct", PARTIAL: "partly"}.get(status, "incorrect")]
            (st.success if status == CORRECT else st.warning if status == PARTIAL else st.error)(content(msg))
            if q["type"] == "short" and result.get("method"):
                st.caption(f"Graded by meaning ({result['method'].upper()} similarity {result['similarity']:.2f}) "
                           f"and key terms ({len(result['matched'])}/{len(result['matched']) + len(result['missing'])})"
                           + (f" - missing: {', '.join(result['missing'])}" if result["missing"] else ""))
        if (result and result["status"] == CORRECT) or state["solution"]:
            answer = q["answer"]
            if q["type"] == "mcq":
                answer_text = content(q["options"][answer])
            elif q["type"] == "tf":
                answer_text = "True" if answer else "False"
            elif q["type"] == "short":
                answer_text = answer
            else:
                answer_text = f"{answer:g}"
            st.markdown(f"**Correct answer:** {esc(answer_text)}")
            show_content(q["explanation"])
    return state["result"]


def render_question_set(subject: str, topic_id: str, title: str, questions: List[dict], prefix: str,
                        save_label: str = "Save Quiz Results") -> None:
    if not questions:
        st.info("No questions match these filters.")
        return
    results = {q["id"]: render_question(subject, topic_id, q, prefix, i + 1) for i, q in enumerate(questions)}
    done = {k: r for k, r in results.items() if r}
    correct = sum(1 for r in done.values() if r["status"] == CORRECT)
    st.progress(correct / len(questions), text=f"Score: {correct} / {len(questions)} correct ({len(done)} answered)")
    if st.button(save_label, key=f"{prefix}_{topic_id}_save", disabled=not done,
                 help="Quiz answers are temporary until you save them."):
        store(subject).save_quiz_result(topic_id, title, len(done), correct,
                                        [{"question": q, "status": r["status"]} for q, r in done.items()])
        st.success(f"Quiz result saved in {subject} > Saved.")


# ---------------------------------------------------------------------------
# Video panel (right column)
# ---------------------------------------------------------------------------
MOMENTS_TITLE = {"en": "How the lecture explains it", "ru": "Lecture mein yeh kaise samjhaya gaya hai"}
POINT_RANGE = {"en": "becomes clear from {a} to {b} in the video:", "ru": "video mein {a} se {b} tak yeh point clear hota hai:"}
NOT_IN_VIDEO = {"en": "Not mentioned in this video's captions: {names}.",
                "ru": "Is video ke captions mein in ka zikr nahi mila: {names}."}
ENGLISH_NOTE = {"en": "", "ru": "Lecture ke jumle video ki asal zaban (English) mein hain."}


def topic_panels(topic: dict) -> list:
    """Lecture videos for a topic with caption timestamps (empty when none were collected)."""
    sage = get_sage()
    return videos_for_topic(topic, sage.engine if sage else None, effective_method())


def _jump(start_key: str, seconds: float) -> None:
    st.session_state[start_key] = int(seconds or 0)


def lecture_moments(topic: dict, panels: list, lang: str, start_key: str) -> bool:
    """The explanation taken FROM the lecture: for the topic and each sub-topic, the lecturer's own
    sentences and the time range in which that point becomes clear, with a button that plays the
    embedded video from there. Only caption text and caption times are used. False = nothing to show."""
    sage = get_sage()
    if not panels or sage is None or not panels[0]["has_transcript"]:
        return False
    main = panels[0]
    exp = video_explanation(sage.engine, main["video_id"], topic)
    if not exp["points"]:
        return False
    st.markdown(f"<div class='cs-moments-title'>{esc(pick(MOMENTS_TITLE, lang))}</div>"
                f"<span class='cs-muted'>{esc(main['title'])} - {esc(main['channel_title'])}</span>", unsafe_allow_html=True)
    badges("lecture")
    for i, pt in enumerate(exp["points"]):
        with st.container(border=True):
            c1, c2 = st.columns([1, 5], vertical_alignment="center")
            c1.button(pt["start_label"], help=f"Play {pt['start_label']}-{pt['end_label']}", key=f"{start_key}_{i}", icon=":material/play_arrow:",
                      on_click=_jump, args=(start_key, pt["start"]), width="stretch")
            c2.markdown(f"<b>{esc(pt['name'])}</b> - "
                        f"{esc(pick(POINT_RANGE, lang).format(a=pt['start_label'], b=pt['end_label']))}",
                        unsafe_allow_html=True)
            lines = "".join(f"<div><a class='cs-ts' href='{youtube_link(main['video_id'], x['start'])}' target='_blank'>"
                            f"{x['label']}</a> {esc(x['text'])}</div>" for x in pt["sentences"])
            st.markdown(f"<div class='cs-quote'>{lines}</div>", unsafe_allow_html=True)
    if exp["missing"]:
        st.caption(pick(NOT_IN_VIDEO, lang).format(names=", ".join(exp["missing"])))
    if pick(ENGLISH_NOTE, lang):
        st.caption(pick(ENGLISH_NOTE, lang))
    return True


def video_panel(topic: dict, panels: Optional[list] = None, start_key: Optional[str] = None,
                show_stamps: bool = True) -> None:
    st.markdown("#### Video reference")
    if panels is None:
        panels = topic_panels(topic)
    if not panels:
        st.info("No video has been collected for this topic yet. Run run_pipeline.bat to build the video library "
                "(videos are found with the YouTube Data API - none are typed in by hand).")
        return
    main = panels[0]
    act = activity()
    start = st.session_state.get(start_key) if start_key else None
    st.video(f"https://www.youtube.com/watch?v={main['video_id']}", start_time=int(start if start is not None else main["start"] or 0),
             autoplay=start is not None)
    act.add_viewed(main["video_id"], main["title"], topic["subject"])
    st.markdown(f"<div class='cs-video'><b>{esc(main['title'])}</b><br><span class='cs-muted'>{esc(main['channel_title'])}"
                f"</span><br><span class='cs-muted'>Topic covered: {esc(topic['title']['en'])} - {esc(main['source'])}</span>"
                f"</div>", unsafe_allow_html=True)
    if main.get("segment"):
        seg = main["segment"]
        st.markdown(f"**Best matching part:** <a class='cs-ts' href='{seg['url']}' target='_blank'>"
                    f"{seg['start_label']}-{seg['end_label']}</a> {chips(seg['terms'][:4])}", unsafe_allow_html=True)
    if main["covered"] and show_stamps:
        st.markdown("**Timestamps found in the captions**")
        for s in main["covered"]:
            st.markdown(f"<div class='cs-quote'><a class='cs-ts' href='{s['url']}' target='_blank'>{s['label']}</a> "
                        f"<b>{esc(s['name'])}</b> - <span class='cs-muted'>“{esc(s['sentence'][:120])}”</span></div>",
                        unsafe_allow_html=True)
    if main["not_found"] and show_stamps:
        st.caption("No reliable timestamp in this video's captions for: " + ", ".join(main["not_found"]))
    if not main["has_transcript"]:
        st.caption("This video has no downloaded captions, so no timestamps can be shown.")
    st.link_button("Open on YouTube", main["watch_url"], width="stretch")
    if len(panels) > 1:
        st.markdown("**More videos on this topic**")
        for v in panels[1:]:
            stamps = " ".join(f"<a class='cs-ts' href='{s['url']}' target='_blank'>{s['label']} {esc(s['name'])}</a>"
                              for s in v["covered"][:4])
            st.markdown(f"<div class='cs-card'><a href='{v['watch_url']}' target='_blank'><b>{esc(v['title'])}</b></a>"
                        f"<br><span class='cs-muted'>{esc(v['channel_title'])}</span><br>{stamps}</div>",
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Lecture-based explanation (from the retrieval pipeline)
# ---------------------------------------------------------------------------
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
            stamps = " | ".join(f"[{t['label']}]({t['url']})" for t in r["timestamps"])
            st.markdown(f"**[{r['n']}] {esc(r['title'])}** - {esc(r['channel_title'])}  \n{stamps}")


def render_visuals(result: dict) -> None:
    tab_map, tab_cloud, tab_time, tab_sub = st.tabs(["Concept map", "Keyword cloud", "Timeline", "Sub-topics"])
    with tab_map:
        components.html(concept_map_html(result["concept_graph"]), height=520, scrolling=False)
    with tab_cloud:
        image = keyword_cloud(result["key_terms"])
        if image is not None:
            st.image(image, width="stretch")
    with tab_time:
        st.plotly_chart(timeline_figure(result["hits"]), width="stretch")
    with tab_sub:
        for g in result["subtopics"] or []:
            with st.expander(f"{g['label']}  ({g['size']} segments)"):
                for h in sorted(g["segments"], key=lambda h: h["rank"]):
                    st.markdown(f"- [{h['title']} @ {format_timestamp(h['start'])}]"
                                f"(https://www.youtube.com/watch?v={h['video_id']}&t={int(h['start'])}s)")
        if not result["subtopics"]:
            st.write("Not enough segments to group into sub-topics.")


# ---------------------------------------------------------------------------
# The topic view
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    st.markdown(f"##### {title}")


def notes_box(subject: str, topic_id: str, title: str, key: str) -> None:
    saved = store(subject).load()["notes"].get(topic_id)
    text = st.text_area("My notes", value=saved["text"] if saved else "", key=f"notes_{key}_{topic_id}", height=110,
                        placeholder="Write your own notes. They are not stored unless you click Save Notes.")
    if st.button("Save Notes", key=f"save_notes_{key}_{topic_id}", disabled=not text.strip()):
        store(subject).save_note(topic_id, title, text.strip())
        st.success(f"Notes saved in {subject} > Saved.")


def lesson_as_text(topic: dict) -> str:
    parts = [topic["title"]["en"], "", content(topic["definition"]), "", content(topic["explanation"]), "",
             content(topic["example"])]
    parts += ["", "Exam points:"] + ["- " + content(x) for x in topic.get("exam_points", [])]
    return "\n".join(parts)


def render_topic(topic: dict, key: str, query: Optional[str] = None, show_questions: bool = True) -> None:
    """Explanation (left) + video reference (right) for one lesson."""
    subject = topic["subject"]
    act = activity()
    act.add_studied(subject, topic["id"], topic["title"]["en"])

    head, b1, b2 = st.columns([3, 1, 1.4])
    head.markdown(f"### {esc(topic['title']['en'])}")
    head.markdown(f"<span class='cs-muted'>{esc(subject)}</span> {lang_badge()}", unsafe_allow_html=True)
    s = store(subject)
    if b1.button("Saved" if s.is_topic_saved(topic["id"]) else "Save Topic", key=f"savetopic_{key}_{topic['id']}",
                 width="stretch", disabled=s.is_topic_saved(topic["id"])):
        s.save_topic(topic["id"], topic["title"]["en"])
        st.rerun()
    if b2.button("Save Learning Material", key=f"savemat_{key}_{topic['id']}", width="stretch",
                 help="Saves this explanation (in the current response language) to the subject's Saved tab."):
        s.save_material("explanation", topic["title"]["en"], lesson_as_text(topic), rlang())
        st.success(f"Saved in {subject} > Saved.")

    panels = topic_panels(topic)
    left, right = st.columns([3, 2], gap="large")
    with left:
        badges("notes")
        _section("Introduction")
        show_content(topic["intro"])
        _section("Definition")
        st.markdown(f"<div class='cs-card'>{content_html(topic['definition'])}</div>", unsafe_allow_html=True)
        _section("Detailed explanation")
        show_content(topic["explanation"])
        lecture_moments(topic, panels, rlang(), f"vstart_{key}_{topic['id']}")
        _section("Worked example")
        badges("example")
        st.markdown(f"<div class='cs-card'>{content_html(topic['example'])}</div>", unsafe_allow_html=True)
        _section("Important concepts")
        cols = st.columns(2)
        for i, c in enumerate(topic.get("key_concepts", [])):
            with cols[i % 2]:
                card(c["term"], content_html(c))
        if topic.get("formulas"):
            _section("Important formulas")
            for f in topic["formulas"]:
                st.latex(f["latex"])
                show_content(f, "cs-muted")
        if topic.get("workflow"):
            _section("Algorithm steps")
            items = "".join(f"<li>{esc(content(s_))}</li>" for s_ in topic["workflow"])
            st.markdown(f"<div class='cs-card'><ol>{items}</ol></div>", unsafe_allow_html=True)
        if topic.get("code"):
            with st.expander("Implementation code and explanation"):
                st.code(topic["code"], language=topic.get("code_language", "python"))
                bullet_list(topic.get("code_explanation", []))
        _section("Real-world applications")
        bullet_list(topic.get("applications", []))
        _section("Important exam points")
        bullet_list(topic.get("exam_points", []))
        _section("Common mistakes")
        bullet_list(topic.get("common_mistakes", []))
        notes_box(subject, topic["id"], topic["title"]["en"], key)
    with right:
        video_panel(topic, panels, f"vstart_{key}_{topic['id']}", show_stamps=False)

    lesson = get_lesson(topic["title"]["en"]) if get_sage() else None   # transcripts are English
    result = lesson["lecture"] if lesson else None
    if result and result.get("covered"):
        with st.expander("What the lectures say (transcript excerpts, original English)"):
            badges("lecture")
            render_lecture_explanation(result)
        with st.expander("Visual overview from the lectures (concept map, keywords, timeline)"):
            render_visuals(result)
    if show_questions:
        st.markdown("#### Related questions")
        st.caption("Answers are temporary. Use Save Quiz Results to keep them in this subject.")
        render_question_set(subject, topic["id"], topic["title"]["en"], topic.get("questions", []), f"q_{key}")
        if lesson and lesson.get("lecture_questions"):
            st.markdown("#### Questions built from the lecture sentences")
            badges("lecture")
            render_question_set(subject, topic["id"], topic["title"]["en"], lesson["lecture_questions"], f"lq_{key}")
