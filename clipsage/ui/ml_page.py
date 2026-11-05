"""ML Algorithm Playground page."""

import numpy as np
import pandas as pd
import streamlit as st

from .. import playground as pg
from ..knowledge import load_topics
from .common import S, T, TL, badges, bi_html, esc, get_progress, open_topic, show

LESSON_FOR = {"linear_regression": "linear_regression", "gradient_descent": "gradient_descent",
              "logistic_regression": "logistic_regression", "knn": "knn", "naive_bayes": "naive_bayes",
              "decision_tree": "decision_tree", "svm": "svm", "kmeans": "kmeans", "tfidf": "tfidf",
              "cosine_similarity": "cosine_similarity", "sentence_bert": "sentence_bert"}


def _dataset(task: str, key: str):
    """Let the student choose sample data, typed CSV or an uploaded CSV."""
    source = st.radio(TL("data_source"), ["sample", "text", "upload"], horizontal=True, key=f"{key}_src",
                      format_func=lambda s: {"sample": TL("use_sample"), "text": TL("enter_data"),
                                             "upload": TL("upload_csv")}[s])
    hint = ("CSV with a header; the LAST column is the target." if task != "clustering"
            else "CSV with a header; two numeric columns.")
    if source == "sample":
        data = pg.make_sample(task, seed=st.session_state.get(f"{key}_seed", 0))
        badges("sample")
        st.caption(data["label"])
        if st.button("\U0001F3B2 New sample", key=f"{key}_reseed"):
            st.session_state[f"{key}_seed"] = st.session_state.get(f"{key}_seed", 0) + 1
            st.rerun()
        return data
    if source == "text":
        text = st.text_area(hint, pg.sample_csv(task), height=180, key=f"{key}_csv")
    else:
        up = st.file_uploader(hint, type=["csv"], key=f"{key}_up")
        if up is None:
            return None
        text = up.getvalue().decode("utf-8", errors="replace")
    return pg.parse_csv(text, task)


def _show_output(out: dict, key: str) -> None:
    badges("computed")
    st.markdown(f"### {esc(TL('results'))}")
    metrics = {k: v for k, v in out.get("metrics", {}).items() if not isinstance(v, dict)}
    if metrics:
        cols = st.columns(min(4, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % len(cols)].metric(k, v if not isinstance(v, float) else f"{v:g}")
    for k, v in out.get("metrics", {}).items():
        if isinstance(v, dict):
            st.markdown(f"**{esc(k)}:** " + ", ".join(f"{a}: {b}" for a, b in v.items()))
    for name, fig in out.get("figures", {}).items():
        st.plotly_chart(fig, width="stretch", key=f"{key}_fig_{name}")
    for name, fig in out.get("matplotlib", {}).items():
        st.markdown(f"**{esc(name)}**")
        st.pyplot(fig)
    if out.get("text"):
        st.code(out["text"], language="text")
    st.markdown(f"### {esc(TL('understand'))}")
    for note in out.get("notes", []):
        st.markdown(f"<div class='cs-card'>{bi_html(note)}</div>", unsafe_allow_html=True)
    st.markdown(f"### {esc(TL('calculations'))}")
    for name, rows in out.get("tables", {}).items():
        if rows:
            st.markdown(f"**{esc(name)}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _predict_box(out: dict, n_features: int, key: str) -> None:
    if not out.get("predict"):
        return
    st.markdown(f"### {esc(TL('predict_point'))}")
    cols = st.columns(n_features + 1)
    point = [cols[i].number_input(f"x{i + 1}", value=3.0, key=f"{key}_p{i}") for i in range(n_features)]
    if cols[-1].button(TL("run"), key=f"{key}_predict"):
        st.json({k: (str(v) if not isinstance(v, (int, float, dict)) else v) for k, v in out["predict"](point).items()})


def ml_page() -> None:
    st.title(TL("nav_ml"))
    show({"en": "Pick an algorithm, choose data, change the parameters and run it. Every number and chart below is "
                "computed by the real algorithm on your data.",
          "ru": "Algorithm chunein, data chunein, parameters badlein aur chalayein. Neeche har number aur chart asli "
                "algorithm ne aap ke data par nikala hai."})
    names = list(pg.ALGORITHMS)
    default = st.session_state.get("ml_algorithm", "linear_regression")
    algo = st.selectbox(TL("algorithm"), names, index=names.index(default) if default in names else 0,
                        format_func=lambda a: S(pg.ALGORITHMS[a]), key="ml_algo_select")
    st.session_state["ml_algorithm"] = algo
    task = pg.ALGORITHMS[algo]["task"]
    topics = load_topics()
    lesson_id = LESSON_FOR.get(algo)
    if lesson_id in topics:
        st.button(f"\U0001F4D8 {TL('lesson_link')}", on_click=open_topic, args=(topics[lesson_id]["title"]["en"],),
                  key="ml_lesson")

    left, right = st.columns([1, 2], gap="large")
    out, n_features, error = None, 2, None
    with left:
        st.markdown(f"#### {esc(TL('dataset'))}")
        try:
            if task in ("regression", "classification", "clustering"):
                data = _dataset(task, f"ds_{algo}")
            elif algo == "tfidf":
                docs = st.text_area("One document per line", "cats like milk\ndogs like bones\ncats chase dogs", height=140)
                data = {"docs": docs.splitlines()}
            elif algo == "cosine_similarity":
                vecs = st.text_area("One vector per line (comma separated)", "1, 2\n2, 4\n2, -1", height=120)
                data = {"vectors": [np.array([float(x) for x in line.split(",")]) for line in vecs.splitlines() if line.strip()]}
            else:
                sents = st.text_area("One sentence per line",
                                     "Overfitting is when a model memorises the training data.\n"
                                     "A model that learns noise performs badly on new data.\n"
                                     "TCP uses a three way handshake.", height=140)
                data = {"sentences": sents.splitlines()}
        except (pg.DataError, ValueError) as e:
            data, error = None, str(e)

        st.markdown(f"#### {esc(TL('parameters'))}")
        params = {}
        if algo == "gradient_descent":
            params["lr"] = st.select_slider("learning rate", [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1], value=0.01)
            params["epochs"] = st.slider("epochs", 10, 2000, 300, 10)
        elif algo == "logistic_regression":
            params["C"] = st.select_slider("C (inverse regularisation)", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        elif algo == "knn":
            params["k"] = st.slider("k", 1, 15, 3)
        elif algo == "decision_tree":
            params["max_depth"] = st.slider("max_depth", 1, 10, 3)
            params["criterion"] = st.radio("criterion", ["gini", "entropy"], horizontal=True)
        elif algo == "svm":
            params["kernel"] = st.radio("kernel", ["linear", "rbf", "poly"], horizontal=True)
            params["C"] = st.select_slider("C", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        elif algo == "kmeans":
            params["k"] = st.slider("k (clusters)", 1, 8, 3)
            params["init"] = st.radio("initialisation", ["k-means++", "random"], horizontal=True)
            params["seed"] = st.number_input("random start (seed)", 0, 999, 0)
        if algo in ("knn", "naive_bayes") and data is not None and data.get("X") is not None and data["X"].shape[1] == 2:
            c1, c2 = st.columns(2)
            params["query"] = [c1.number_input("query x1", value=float(np.round(data["X"][:, 0].mean(), 2))),
                               c2.number_input("query x2", value=float(np.round(data["X"][:, 1].mean(), 2)))]
        run = st.button("▶ " + TL("run"), type="primary", width="stretch")

    with right:
        if error:
            st.error(T("data_error", err=error))
        elif data is None:
            st.info(TL("upload_csv"))
        elif run or st.session_state.get("ml_last") == algo:
            try:
                kw = {"features": data.get("features", ("x1", "x2"))} if "X" in data else {}
                if task in ("regression", "classification") and data.get("target"):
                    kw["target"] = data["target"]
                if algo == "linear_regression":
                    out = pg.run_linear_regression(data["X"], data["y"], **kw)
                elif algo == "gradient_descent":
                    out = pg.run_gradient_descent(data["X"], data["y"], **params, **kw)
                elif algo == "logistic_regression":
                    out = pg.run_logistic_regression(data["X"], data["y"], **params, **kw)
                elif algo == "knn":
                    out = pg.run_knn(data["X"], data["y"], **params, **kw)
                elif algo == "naive_bayes":
                    out = pg.run_naive_bayes(data["X"], data["y"], **params, **kw)
                elif algo == "decision_tree":
                    out = pg.run_decision_tree(data["X"], data["y"], **params, **kw)
                elif algo == "svm":
                    out = pg.run_svm(data["X"], data["y"], **params, **kw)
                elif algo == "kmeans":
                    out = pg.run_kmeans(data["X"], **params, features=data["features"])
                elif algo == "tfidf":
                    out = pg.run_tfidf(data["docs"])
                elif algo == "cosine_similarity":
                    out = pg.run_cosine(data["vectors"])
                else:
                    with st.spinner("Sentence-BERT ..."):
                        out = pg.run_sentence_bert(data["sentences"])
                if run:
                    get_progress().record_playground_run()
                st.session_state["ml_last"] = algo
                n_features = data["X"].shape[1] if "X" in data else 0
                _show_output(out, f"out_{algo}")
                if n_features:
                    _predict_box(out, n_features, f"pred_{algo}")
            except (pg.DataError, ValueError) as e:
                st.error(T("data_error", err=e))
        else:
            show({"en": "Press “Run algorithm” to see the results.",
                  "ru": "Natije dekhne ke liye “Algorithm chalayein” dabayein."})
