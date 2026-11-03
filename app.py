"""
Stage 7 - ClipSage web interface (Streamlit).

Run:
    streamlit run app.py
"""

import html

import streamlit as st
import streamlit.components.v1 as components

from clipsage import config
from clipsage.index import index_info
from clipsage.keywords import KEYWORD_LABELS
from clipsage.retrieve import METHOD_LABELS
from clipsage.summarize import SUMMARIZER_LABELS
from clipsage.utils import format_timestamp
from clipsage.visualize import concept_map_html, keyword_cloud, timeline_figure

st.set_page_config(page_title="ClipSage", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
      .cs-explanation p {font-size: 1.05rem; line-height: 1.65; margin-bottom: .55rem;}
      .cs-cite {font-size: .8rem; text-decoration: none; padding: 1px 6px; border-radius: 8px;
                background: rgba(37, 99, 235, .12); margin-left: 4px; white-space: nowrap;}
      .cs-term {display: inline-block; padding: 3px 10px; margin: 3px; border-radius: 14px;
                background: rgba(37, 99, 235, .10); border: 1px solid rgba(37, 99, 235, .25);}
      .cs-def {border-left: 3px solid #c2410c; padding-left: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the lecture index and models ...")
def load_clipsage():
    from clipsage.pipeline import ClipSage

    sage = ClipSage()
    # warm up (load models, imports) so the first real query is fast
    warm_query = sage.engine.segments[0]["title"]
    sage.ask(warm_query, method="tfidf")
    if index_info().get("sbert_model"):
        sage.ask(warm_query, method="sbert")
    return sage


def render_explanation(result):
    ref_no = {r["video_id"]: r["n"] for r in result["references"]}
    parts = []
    for s in result["explanation"]:
        cite = (f'<a class="cs-cite" href="{s["url"]}" target="_blank" '
                f'title="{html.escape(s["title"])}">[{ref_no[s["video_id"]]}] {s["timestamp"]}</a>')
        cls = ' class="cs-def"' if s.get("is_definition") else ""
        parts.append(f"<p{cls}>{html.escape(s['text'])}{cite}</p>")
    st.markdown(f'<div class="cs-explanation">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_references(result):
    for r in result["references"]:
        stamps = " · ".join(f"[{t['label']}]({t['url']})" for t in r["timestamps"])
        used = "" if r["used_in_explanation"] else " _(related)_"
        st.markdown(f"**[{r['n']}] {r['title']}** — {r['channel_title']}{used}  \n▶ {stamps}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
info = index_info()
with st.sidebar:
    st.header("Settings")
    methods = ["sbert", "tfidf"] if info.get("sbert_model") else ["tfidf"]
    method = st.radio("Retrieval method", methods, format_func=METHOD_LABELS.get)
    summarizer = st.selectbox("Explanation method", ["textrank", "embedding", "top_sentences"],
                              format_func=SUMMARIZER_LABELS.get)
    keyword_method = st.selectbox("Key-term method", ["keybert", "tfidf"], format_func=KEYWORD_LABELS.get,
                                  index=0 if method == "sbert" else 1)
    n_sentences = st.slider("Explanation length (sentences)", config.SUMMARY_MIN_SENTENCES,
                            config.SUMMARY_MAX_SENTENCES, config.SUMMARY_SENTENCES)
    top_k = st.slider("Segments to use", 5, 20, config.TOP_K_SEGMENTS)
    abstractive = st.checkbox("Also write an abstractive summary (BART, slow)", value=False)

    st.divider()
    st.caption("Collection")
    if info:
        st.write(f"{info.get('segments', 0):,} lecture segments indexed")
        st.write(f"Model: {info.get('sbert_model') or 'TF-IDF only'}")
        st.write(f"Built: {info.get('built_at', '-')}")
    st.caption("ClipSage only links to the original YouTube videos and credits every source. "
               "It is a study aid — please watch the referenced lectures.")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🎓 ClipSage")
st.write("Type a topic you want to study. ClipSage searches a collection of lecture videos and gives you "
         "a short explanation, the exact video moments it came from, and a visual map of the topic.")

if not info:
    st.error("No index found. Run the data pipeline first: `python -m clipsage.collect`, "
             "`python -m clipsage.preprocess`, `python -m clipsage.index` (see README).")
    st.stop()

st.session_state.setdefault("query", "")


def choose_topic(topic: str) -> None:
    """Callback for the 'related topic' buttons (runs before the page re-renders)."""
    st.session_state.query = topic
    st.session_state.query_input = topic


with st.form("ask"):
    st.text_input("Topic", key="query_input",
                  placeholder="e.g. overfitting, binary search tree, TCP handshake")
    if st.form_submit_button("Explain", type="primary"):
        st.session_state.query = st.session_state.query_input

query = st.session_state.query.strip()
if not query:
    st.stop()

sage = load_clipsage()
with st.spinner("Searching the lectures ..."):
    result = sage.ask(query, method=method, summarizer=summarizer, keyword_method=keyword_method,
                      top_k=top_k, n_sentences=n_sentences, abstractive=abstractive)

if not result["covered"]:
    st.warning(f"**“{query}” is not covered** by the collected lectures yet.")
    if result["suggestions"]:
        st.write("Related topics you can try:")
        cols = st.columns(len(result["suggestions"]))
        for col, topic in zip(cols, result["suggestions"]):
            col.button(topic, on_click=choose_topic, args=(topic,))
    st.stop()

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(f"Explanation: {query}")
    render_explanation(result)
    abs_result = result.get("abstractive")
    if abs_result:
        if abs_result.get("text"):
            with st.expander(f"Abstractive summary ({abs_result['model']})", expanded=True):
                st.write(abs_result["text"])
                st.caption("Generated from the sources [" + "], [".join(
                    str(next(r["n"] for r in result["references"] if r["video_id"] == s["video_id"]))
                    for s in abs_result["sources"]) + "]")
        else:
            st.info(abs_result.get("error", "Abstractive summary unavailable."))

    st.subheader("References")
    render_references(result)

with right:
    best = result["hits"][0]
    st.subheader("Best matching moment")
    st.video(f"https://www.youtube.com/watch?v={best['video_id']}", start_time=int(best["start"]))
    st.caption(f"{best['title']} — {best['channel_title']} @ {format_timestamp(best['start'])}")

    st.subheader("Key terms")
    st.markdown("".join(f'<span class="cs-term">{html.escape(t)}</span>' for t, _ in result["key_terms"]),
                unsafe_allow_html=True)

st.subheader("Visual overview")
tab_map, tab_cloud, tab_time, tab_sub = st.tabs(["Concept map", "Keyword cloud", "Timeline", "Sub-topics"])
with tab_map:
    components.html(concept_map_html(result["concept_graph"]), height=540, scrolling=False)
    st.caption("Terms that appear together in the lectures are linked; thicker links = more often together.")
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

t = result["timings"]
st.caption(f"Answered in {t['total']:.2f} s · retrieval {t['retrieval'] * 1000:.0f} ms · "
           f"{METHOD_LABELS[method]} · {SUMMARIZER_LABELS[summarizer]}")
