"""Mathematics Playground page."""

import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

from .. import mathsolve as M
from ..i18n import UI
from ..knowledge import load_topics
from .common import activity, badges, en as S, esc, open_topic, ui as T, ui as TL
from .common import content_html as bi_html, show_content as show

LAYOUT = dict(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF", height=380,
              margin=dict(l=40, r=20, t=40, b=40), font=dict(color="#243447"))
RELATED_TOPIC = {"gaussian_elimination": "systems_of_equations", "matrix_multiply": "matrices", "determinant": "matrices", "inverse": "matrices", "eigen": "eigenvalues",
                 "vectors": "vectors", "statistics": "statistics_basics", "bayes": "probability_bayes",
                 "binomial": "probability_bayes", "derivative": "derivatives", "integral": "derivatives",
                 "optimize": "derivatives", "equation": "derivatives", "gradient_descent": "gradient_descent"}


def _function_plot(f, extra=None, title="f(x)", x_range=None):
    fn = sp.lambdify(M.X, f, "numpy")
    lo, hi = x_range or (-6, 6)
    xs = np.linspace(lo, hi, 400)
    with np.errstate(all="ignore"):
        ys = np.asarray(fn(xs), dtype=float) * np.ones_like(xs)
    ys[~np.isfinite(ys)] = np.nan
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)", line=dict(color="#4A90E2", width=3)))
    for tr in extra or []:
        fig.add_trace(tr)
    fig.update_layout(title=title, **LAYOUT)
    return fig


def _visual(res):
    kind = res["kind"]
    if kind == "derivative" and isinstance(res.get("expr"), sp.Basic):
        f, df = res["expr"], sp.diff(res["expr"], M.X)
        fig = _function_plot(f, title="f(x) and f'(x)")
        xs = np.linspace(-6, 6, 400)
        dfn = sp.lambdify(M.X, df, "numpy")
        with np.errstate(all="ignore"):
            fig.add_trace(go.Scatter(x=xs, y=np.asarray(dfn(xs), float) * np.ones_like(xs), mode="lines",
                                     name="f'(x)", line=dict(color="#F5A623", width=2, dash="dash")))
        return fig
    if kind == "optimize" and isinstance(res.get("expr"), sp.Basic):
        pts = res["answer"]
        xs_pts = [p["x"] for p in pts]
        rng = (min(xs_pts + [0]) - 5, max(xs_pts + [0]) + 5)
        extra = [go.Scatter(x=xs_pts, y=[p["f"] for p in pts], mode="markers+text", name="critical points",
                            text=[p["type"] for p in pts], textposition="top center",
                            marker=dict(size=12, color="#D0021B"))]
        return _function_plot(res["expr"], extra, "Critical points", rng)
    if kind == "integral" and res.get("bounds"):
        a, b = res["bounds"]
        fn = sp.lambdify(M.X, res["expr"], "numpy")
        xs = np.linspace(a, b, 200)
        fill = go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[np.asarray(fn(xs), float) * np.ones_like(xs), np.zeros_like(xs)],
                          fill="toself", fillcolor="rgba(74,144,226,0.25)", line=dict(width=0), name="area")
        return _function_plot(res["expr"], [fill], "Area under the curve", (min(a, b) - 2, max(a, b) + 2))
    if kind == "gradient_descent":
        h = res["history"]
        xs = [p["x"] for p in h]
        lo, hi = min(xs + [0]) - 1, max(xs + [0]) + 1
        if hi - lo > 1e4:
            lo, hi = -10, 10
        path = go.Scatter(x=xs, y=[p["f"] for p in h], mode="lines+markers", name="steps",
                          marker=dict(color="#D0021B", size=8), line=dict(color="#D0021B", dash="dot"))
        return _function_plot(res["expr"], [path], "Gradient descent path on f(x)", (lo, hi))
    if kind == "vectors" and len(res["u"]) == 2:
        u, v = res["u"], res["v"]
        fig = go.Figure()
        for vec, name, color in [(u, "u", "#4A90E2"), (v, "v", "#F5A623")]:
            fig.add_trace(go.Scatter(x=[0, vec[0]], y=[0, vec[1]], mode="lines+markers+text", name=name,
                                     text=["", name], textposition="top center", line=dict(color=color, width=4)))
        lim = float(max(np.abs(np.r_[u, v]).max(), 1)) * 1.3
        fig.update_layout(title="Vectors", xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim], scaleanchor="x"),
                          **LAYOUT)
        return fig
    if kind == "statistics":
        data = res["data"]
        a = res["answer"]
        fig = go.Figure(go.Scatter(x=data, y=np.zeros_like(data), mode="markers", marker=dict(size=14, color="#4A90E2"),
                                   name="data"))
        fig.add_vline(x=a["mean"], line=dict(color="#D0021B", width=3), annotation_text="mean")
        fig.add_vrect(x0=a["mean"] - a["std_pop"], x1=a["mean"] + a["std_pop"], fillcolor="rgba(74,144,226,0.15)",
                      line_width=0, annotation_text="± 1 std")
        fig.update_layout(title="Data, mean and one standard deviation", yaxis=dict(visible=False), **{**LAYOUT, "height": 260})
        return fig
    if kind in ("matrix_multiply", "inverse") and isinstance(res["answer"], np.ndarray):
        z = res["answer"]
        fig = go.Figure(go.Heatmap(z=z[::-1], colorscale="Blues", text=np.round(z[::-1], 3), texttemplate="%{text}",
                                   showscale=False))
        fig.update_layout(title="Result matrix", xaxis=dict(visible=False), yaxis=dict(visible=False),
                          **{**LAYOUT, "height": 120 + 60 * z.shape[0]})
        return fig
    return None


def show_solution(res: dict) -> None:
    badges("computed")
    st.markdown(f"#### {esc(S(res['title']))}")
    show(res["concept"], "cs-muted")
    st.markdown(f"**{esc(TL('steps'))}**")
    for i, step in enumerate(res["steps"], 1):
        if "latex" in step:
            st.latex(step["latex"])
        st.markdown(f"<div class='cs-card'><b>{i}.</b> {bi_html(step)}</div>", unsafe_allow_html=True)
    st.success(f"{TL('final_answer')}: {res['answer_text']}")
    if "MISMATCH" in res["verified"]:
        st.error("Verification failed - please report this input.")
    else:
        st.caption("✔ " + T("verified", lib=res["verified"]))
    fig = _visual(res)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    tid = RELATED_TOPIC.get(res["kind"])
    topics = load_topics()
    if tid in topics:
        st.button(f"\U0001F4D8 {S(topics[tid]['title'])}", key=f"math_lesson_{res['kind']}",
                  on_click=open_topic, args=(tid, topics[tid]["subject"]))


def _form_inputs(kind: str):
    """Inputs for each problem type; returns a callable that solves it."""
    if kind in ("matrix_multiply",):
        a = st.text_area("A (rows separated by ;)", "1 2; 3 4", key="m_a")
        b = st.text_area("B", "5 6; 7 8", key="m_b")
        return lambda: M.matrix_multiply(M.parse_matrix(a), M.parse_matrix(b))
    if kind == "gaussian_elimination":
        a = st.text_area("Augmented matrix [A | b] (rows separated by ;, last column = b)",
                         "2 1 -1 8; -3 -1 2 -11; -2 1 2 -3", key="m_gauss")
        return lambda: M.gaussian_elimination(M.parse_matrix(a))
    if kind in ("determinant", "inverse", "eigen"):
        a = st.text_area("A (rows separated by ;)", "2 1; 1 3" if kind != "determinant" else "2 0 1; 1 3 2; 1 1 1",
                         key=f"m_{kind}")
        fn = {"determinant": M.determinant, "inverse": M.inverse, "eigen": M.eigen2}[kind]
        return lambda: fn(M.parse_matrix(a))
    if kind == "vectors":
        u = st.text_input("u", "3, 4", key="v_u")
        v = st.text_input("v", "4, 3", key="v_v")
        return lambda: M.vectors(M.parse_vector(u), M.parse_vector(v))
    if kind == "statistics":
        d = st.text_input("data", "2, 4, 4, 4, 5, 5, 7, 9", key="s_d")
        return lambda: M.statistics(M.parse_vector(d))
    if kind == "bayes":
        c = st.columns(3)
        prior = c[0].number_input("P(H) prior", 0.0, 1.0, 0.01, 0.01, format="%.3f")
        lik = c[1].number_input("P(E|H)", 0.0, 1.0, 0.99, 0.01, format="%.3f")
        fp = c[2].number_input("P(E|not H)", 0.0, 1.0, 0.05, 0.01, format="%.3f")
        return lambda: M.bayes(prior, lik, fp)
    if kind == "binomial":
        c = st.columns(3)
        n = c[0].number_input("n", 1, 100, 5)
        k = c[1].number_input("k", 0, 100, 3)
        p = c[2].number_input("p", 0.0, 1.0, 0.5, 0.05)
        return lambda: M.binomial(int(n), int(k), float(p))
    if kind == "derivative":
        e = st.text_input("f(x)", "x^3 - 3x^2 + 2", key="d_f")
        at = st.text_input("evaluate at x = (optional)", "", key="d_at")
        return lambda: M.derivative(e, at=float(at) if at.strip() else None)
    if kind == "integral":
        e = st.text_input("f(x)", "x^2", key="i_f")
        c = st.columns(2)
        a = c[0].text_input("from a (optional)", "0", key="i_a")
        b = c[1].text_input("to b (optional)", "3", key="i_b")
        return lambda: M.integral(e, float(a) if a.strip() else None, float(b) if b.strip() else None)
    if kind == "optimize":
        e = st.text_input("f(x)", "x^2 - 6x + 10", key="o_f")
        return lambda: M.optimize(e)
    if kind == "equation":
        e = st.text_input("equation", "x^2 - 5x + 6 = 0", key="e_f")
        return lambda: M.solve_equation(e)
    if kind == "gradient_descent":
        e = st.text_input("f(x)", "x^2", key="g_f")
        c = st.columns(3)
        x0 = c[0].number_input("x0", value=4.0)
        lr = c[1].number_input("learning rate", value=0.1, min_value=0.0001, format="%.4f")
        n = c[2].number_input("steps", 1, 200, 15)
        return lambda: M.gradient_descent(e, x0, lr, int(n))
    raise ValueError(kind)


def math_page() -> None:
    st.title(TL("nav_math"))
    st.caption("Every answer is calculated and double-checked with SymPy / NumPy / SciPy and explained step by step "
               "in your response language.")
    tab_solve, tab_free, tab_practice = st.tabs([TL("solve"), "⌨️ " + S({"en": "Type a question", "ru": "Sawal likhein"}),
                                                 "\U0001F3AF " + TL("practice_problem")])
    with tab_solve:
        kind = st.selectbox(TL("problem_type"), list(M.KINDS), format_func=lambda k: S(M.KINDS[k]), key="math_kind")
        solver = _form_inputs(kind)
        if st.button(TL("solve"), type="primary", key="math_solve"):
            try:
                st.session_state["math_result"] = solver()
            except (M.MathInputError, ValueError) as e:
                st.session_state["math_result"] = None
                st.error(T("math_parse_error", err=e))
        res = st.session_state.get("math_result")
        if res and res["kind"] == kind:
            show_solution(res)

    with tab_free:
        st.caption(UI["free_text_math"]["en"])
        text = st.text_input("question", "derivative of x^3 + 2x at x = 2", label_visibility="collapsed", key="math_free")
        if st.button(TL("solve"), key="math_free_btn", type="primary"):
            try:
                res = M.detect_and_solve(text)
                st.caption(T("detected_type", kind=S(res["title"])))
                show_solution(res)
            except (M.MathInputError, ValueError) as e:
                st.error(T("math_parse_error", err=e))

    with tab_practice:
        pkind = st.selectbox(TL("problem_type"), M.PRACTICE_KINDS, format_func=lambda k: S(M.KINDS[k]), key="pr_kind")
        current = st.session_state.get("pr_problem")
        if st.button("\U0001F504 " + TL("new_problem"), key="pr_new") or current is None or current["kind"] != pkind:
            seed = int(np.random.default_rng().integers(1_000_000))
            st.session_state["pr_problem"] = M.practice_problem(pkind, seed)
            st.session_state["pr_state"] = {"tries": 0, "hint": False, "solved": False, "show": False, "n": seed}
        prob, state = st.session_state["pr_problem"], st.session_state["pr_state"]
        st.markdown(f"<div class='cs-card'>{bi_html(prob['q'])}</div>", unsafe_allow_html=True)
        ans = st.text_input(TL("your_answer"), key=f"pr_ans_{state['n']}")
        c1, c2, c3 = st.columns(3)
        if c1.button(TL("check_answer"), type="primary", key="pr_check") and ans.strip():
            res = M.check_practice(prob, ans)
            state["tries"] += 1
            state["solved"] = res["status"] == "correct"
            activity().add_practice(st.session_state.get("subject") or "Linear Algebra", "maths", prob["q"]["en"],
                                    state["solved"])
            state["last"] = res["status"]
        if c2.button(TL("hint"), key="pr_hint"):
            state["hint"] = True
        if c3.button(TL("show_solution"), key="pr_show", disabled=state["tries"] == 0):
            state["show"] = True
        if state.get("last") == "correct":
            st.success(T("correct"))
        elif state.get("last") == "incorrect":
            st.error(T("incorrect") + " " + S({"en": "Use a hint or try again.", "ru": "Hint lein ya dobara koshish karein."}))
            state["hint"] = True
        if state["hint"]:
            st.markdown(f"<div class='cs-quote'>\U0001F4A1 {bi_html(prob['hint'])}</div>", unsafe_allow_html=True)
        if state["solved"] or state["show"]:
            show_solution(prob["solution"])
