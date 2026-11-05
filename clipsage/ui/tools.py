"""Algorithm practice tools and numerical practice for each subject (English interface)."""

import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from .. import mathsolve as M
from .. import practice_gen as G
from .. import simulators as S
from ..knowledge import load_topics
from .common import activity, badges, content, esc, show_content, store

LAYOUT = dict(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#F5F9FF", margin=dict(l=40, r=20, t=40, b=40),
              font=dict(color="#243447"))
PALETTE = ["#4A90E2", "#F5A623", "#7ED321", "#BD10E0", "#D0021B", "#50E3C2", "#9013FE", "#417505"]

TOOL_NAMES = {
    "os_scheduling": "CPU scheduling simulator", "os_page_replacement": "Page replacement simulator",
    "os_paging": "Paging address translation", "os_bankers": "Banker's algorithm",
    "ca_numbers": "Number representation", "ca_cache": "Cache simulator", "ca_pipeline": "Pipeline hazards",
    "ca_performance": "Performance calculator", "cc_lexer": "Lexical analyzer", "cc_first_follow": "FIRST / FOLLOW and LL(1)",
    "cc_shift_reduce": "SLR shift-reduce parser", "cc_tac": "Three-address code generator",
    "la_solver": "Matrix and vector solver", "la_elimination": "Gaussian elimination", "ml_playground": "ML algorithm playground",
    "ml_math": "Maths for ML solver", "web_live": "Live HTML / CSS / JavaScript editor", "web_http": "HTTP and URL explorer",
}


def save_result_button(subject: str, key: str, title: str, text: str) -> None:
    if st.button("Save result", key=f"save_{key}", help="Saves this result to the subject's Saved tab."):
        store(subject).save_material("practice result", title, text)
        st.success(f"Saved in {subject} > Saved.")


def _df(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _ints(text: str):
    return [int(x, 0) for x in re.split(r"[,\s]+", (text or "").strip()) if x]


def _matrix(text: str):
    return [[int(v) for v in re.split(r"[,\s]+", row.strip()) if v] for row in (text or "").strip().split(";") if row.strip()]


# ---------------------------------------------------------------------------
# Operating Systems
# ---------------------------------------------------------------------------
def gantt_figure(gantt, title):
    fig = go.Figure()
    names = sorted({g[0] for g in gantt if g[0] != "idle"})
    for name, a, b in gantt:
        color = "#D9DEE5" if name == "idle" else PALETTE[names.index(name) % len(PALETTE)]
        fig.add_trace(go.Bar(x=[b - a], base=[a], y=["CPU"], orientation="h", marker=dict(color=color, line=dict(color="white", width=2)),
                             text=[name], textposition="inside", showlegend=False,
                             hovertemplate=f"{name}: {a}-{b}<extra></extra>"))
    fig.update_layout(barmode="overlay", height=170, title=title, xaxis_title="time", yaxis=dict(visible=False), **LAYOUT)
    return fig


def tool_os_scheduling(subject):
    c1, c2 = st.columns([2, 1])
    text = c1.text_area("Processes (name, arrival, burst, priority)", "P1, 0, 5, 2\nP2, 1, 3, 1\nP3, 2, 1, 3\nP4, 4, 2, 2", height=130)
    algo = c2.selectbox("Algorithm", ["FCFS", "SJF", "SRTF", "PRIORITY", "RR"])
    q = c2.number_input("Time quantum (RR)", 1, 10, 2)
    compare = c2.checkbox("Compare all algorithms")
    try:
        procs = S.parse_processes(text)
    except S.SimError as e:
        st.error(str(e)); return
    r = S.schedule(procs, algo, q)
    badges("computed")
    st.plotly_chart(gantt_figure(r["gantt"], f"Gantt chart - {algo}"), width="stretch")
    m = st.columns(4)
    m[0].metric("Average waiting", f"{r['avg_waiting']:.2f}")
    m[1].metric("Average turnaround", f"{r['avg_turnaround']:.2f}")
    m[2].metric("Average response", f"{r['avg_response']:.2f}")
    m[3].metric("Context switches", r["context_switches"])
    _df(r["rows"])
    with st.expander("Dry run (decisions step by step)"):
        st.code("\n".join(r["log"]) or "(no choices to make)", language="text")
    if compare:
        rows = [{"algorithm": a, "avg waiting": round(S.schedule(procs, a, q)["avg_waiting"], 2),
                 "avg turnaround": round(S.schedule(procs, a, q)["avg_turnaround"], 2)} for a in ["FCFS", "SJF", "SRTF", "PRIORITY", "RR"]]
        _df(rows)
    save_result_button(subject, "sched", f"{algo} scheduling", f"{text}\n-> avg waiting {r['avg_waiting']:.2f}, "
                       f"avg turnaround {r['avg_turnaround']:.2f}\nGantt: {r['gantt']}")
    activity().add_practice(subject, "os_scheduling", f"{algo} avg WT {r['avg_waiting']:.2f}")


def tool_os_page_replacement(subject):
    c1, c2 = st.columns([2, 1])
    refs_text = c1.text_input("Reference string", "7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1")
    frames = c2.number_input("Frames", 1, 8, 3)
    algo = c2.selectbox("Algorithm", ["FIFO", "LRU", "OPT"])
    try:
        refs = _ints(refs_text)
        r = S.page_replacement(refs, int(frames), algo)
    except (ValueError, S.SimError) as e:
        st.error(f"Input problem: {e}"); return
    badges("computed")
    st.metric("Page faults", f"{r['faults']} / {len(refs)}", help=f"fault rate {r['fault_rate']:.0%}")
    table = {f"{i + 1}: {s['ref']}": [("" if f is None else f) for f in s["frames"]] + [s["result"]] for i, s in enumerate(r["steps"])}
    st.dataframe(pd.DataFrame(table, index=[f"frame {i + 1}" for i in range(int(frames))] + ["result"]), width="stretch")
    comp = {a: [S.page_replacement(refs, f, a)["faults"] for f in range(1, 8)] for a in ["FIFO", "LRU", "OPT"]}
    fig = go.Figure([go.Scatter(x=list(range(1, 8)), y=v, mode="lines+markers", name=a,
                                line=dict(color=PALETTE[i], width=3)) for i, (a, v) in enumerate(comp.items())])
    fig.update_layout(title="Faults vs number of frames (look for Belady's anomaly in FIFO)", xaxis_title="frames",
                      yaxis_title="page faults", height=330, **LAYOUT)
    st.plotly_chart(fig, width="stretch")
    save_result_button(subject, "pr", f"{algo} page replacement", f"refs {refs}, {frames} frames -> {r['faults']} faults")


def tool_os_paging(subject):
    c1, c2 = st.columns(2)
    size = c1.selectbox("Page size (bytes)", [256, 512, 1024, 2048, 4096], index=2)
    bits = c1.number_input("Logical address bits", 8, 32, 16)
    table_text = c2.text_input("Page table (page:frame)", "0:3, 1:7, 2:5, 3:1")
    addr_text = c2.text_input("Logical addresses", "2500, 100, 5000")
    try:
        table = {int(a): int(b) for a, b in (p.split(":") for p in re.split(r"[,\s]+", table_text.strip()) if p)}
        r = S.translate_addresses(int(size), table, _ints(addr_text), int(bits))
    except (ValueError, S.SimError) as e:
        st.error(f"Input problem: {e}"); return
    badges("computed")
    st.write(f"Offset bits = log2({size}) = **{r['offset_bits']}**, page-number bits = {bits} - {r['offset_bits']} = "
             f"**{r['page_bits']}** ({r['pages']} pages)")
    _df(r["rows"])
    save_result_button(subject, "paging", "Address translation", str(r["rows"]))


def tool_os_bankers(subject):
    c1, c2 = st.columns(2)
    avail = c1.text_input("Available", "3 3 2")
    mx = c1.text_area("Max (rows separated by ;)", "7 5 3; 3 2 2; 9 0 2; 2 2 2; 4 3 3")
    alloc = c2.text_area("Allocation", "0 1 0; 2 0 0; 3 0 2; 2 1 1; 0 0 2")
    req = c2.text_input("Optional request 'process: r1 r2 r3' (e.g. 1: 1 0 2)", "")
    try:
        request = None
        if req.strip():
            pid, vals = req.split(":")
            request = (int(pid), _ints(vals))
        r = S.bankers(_ints(avail), _matrix(mx), _matrix(alloc), request)
    except (ValueError, S.SimError) as e:
        st.error(f"Input problem: {e}"); return
    badges("computed")
    for n in r["notes"]:
        st.info(n)
    st.markdown("**Need = Max - Allocation**")
    _df([{"process": f"P{i}", "need": " ".join(map(str, row))} for i, row in enumerate(r["need"])])
    _df([{"step": i + 1, "process": s["process"], "need": str(s["need"]), "work before": str(s["work_before"]),
          "work after": str(s["work_after"])} for i, s in enumerate(r["steps"])])
    (st.success if r["safe"] else st.error)(("SAFE - sequence: " + " -> ".join(r["sequence"])) if r["safe"]
                                            else "UNSAFE - no safe sequence exists")
    save_result_button(subject, "banker", "Banker's algorithm", f"safe={r['safe']} sequence={r['sequence']}")


# ---------------------------------------------------------------------------
# Computer Architecture
# ---------------------------------------------------------------------------
def tool_ca_numbers(subject):
    tab1, tab2, tab3 = st.tabs(["Decimal to two's complement", "Binary to decimal", "IEEE 754 single precision"])
    with tab1:
        c1, c2 = st.columns(2)
        v = c1.number_input("Integer", value=-13, step=1)
        bits = c2.selectbox("Bits", [4, 8, 16, 32], index=1)
        try:
            r = S.integer_representation(int(v), int(bits))
            badges("computed")
            st.code("\n".join(r["steps"]) + f"\n\nbinary {r['grouped']}  hex {r['hex']}  range {r['range']}", language="text")
        except S.SimError as e:
            st.error(str(e))
    with tab2:
        pattern = st.text_input("Binary pattern", "11110011")
        try:
            d = S.decode_binary(pattern)
            st.code(f"unsigned = {d['unsigned']}\nsigned (two's complement) = {d['signed']}\nhex = {d['hex']}", language="text")
        except S.SimError as e:
            st.error(str(e))
    with tab3:
        x = st.number_input("Real number", value=5.75, format="%.6f")
        r = S.ieee754_single(float(x))
        st.markdown(f"<span class='cs-chip'>sign {r['sign']}</span><span class='cs-chip'>exponent {r['exponent']}</span>"
                    f"<span class='cs-chip'>fraction {r['fraction']}</span><span class='cs-chip'>{r['hex']}</span>",
                    unsafe_allow_html=True)
        st.code("\n".join(r["steps"]) + f"\nstored value {r['stored_value']!r} (rounding error {r['rounding_error']:.3g})",
                language="text")
        save_result_button(subject, "ieee", "IEEE 754", f"{x} -> {r['hex']}")


def tool_ca_cache(subject):
    c = st.columns(4)
    bits = c[0].number_input("Address bits", 8, 32, 16)
    size = c[1].selectbox("Cache size (bytes)", [64, 128, 256, 1024, 4096, 65536], index=0)
    block = c[2].selectbox("Block size (bytes)", [4, 8, 16, 32, 64], index=2)
    ways = c[3].selectbox("Associativity (ways)", [1, 2, 4, 8])
    addrs = st.text_input("Address sequence (decimal or 0x hex)", "0, 4, 16, 64, 0, 128, 4")
    h1, h2 = st.columns(2)
    hit_t = h1.number_input("Hit time (ns)", value=1.0)
    pen = h2.number_input("Miss penalty (ns)", value=100.0)
    try:
        r = S.cache_simulate(int(bits), int(size), int(block), int(ways), _ints(addrs), hit_t, pen)
    except (ValueError, S.SimError) as e:
        st.error(f"Input problem: {e}"); return
    badges("computed")
    st.markdown(f"<span class='cs-chip'>tag {r['tag_bits']} bits</span><span class='cs-chip'>index {r['index_bits']} bits "
                f"({r['sets']} sets)</span><span class='cs-chip'>offset {r['offset_bits']} bits</span>", unsafe_allow_html=True)
    _df(r["rows"])
    m = st.columns(3)
    m[0].metric("Hit rate", f"{r['hit_rate']:.0%}")
    m[1].metric("Misses", r["misses"])
    m[2].metric("AMAT (ns)", f"{r['amat']:.2f}")
    save_result_button(subject, "cache", "Cache simulation", f"{r['hits']} hits / {len(r['rows'])}, AMAT {r['amat']:.2f}")


def tool_ca_pipeline(subject):
    text = st.text_area("Instructions (one per line)", "LW R1, 0(R2)\nADD R3, R1, R4\nSUB R5, R3, R6\nSW R5, 8(R2)", height=120)
    fwd = st.toggle("Forwarding (bypassing)", value=True)
    try:
        r = S.pipeline_simulate(S.parse_instructions(text), fwd)
    except S.SimError as e:
        st.error(str(e)); return
    badges("computed")
    grid = pd.DataFrame(r["grid"], index=[x["instr"] for x in r["rows"]], columns=[f"c{i + 1}" for i in range(r["total_cycles"])])
    colors = {"IF": "#EAF4FF", "ID": "#D6E9FB", "EX": "#B9D7F7", "MEM": "#9BC4F2", "WB": "#7CB9F7", "stall": "#FCE4E4"}
    st.dataframe(grid.style.map(lambda v: f"background-color: {colors.get(v, '#FFFFFF')}"), width="stretch")
    _df([{"instruction": x["instr"], "stalls": x["stalls"], "waits for": x["reason"]} for x in r["rows"]])
    m = st.columns(4)
    m[0].metric("Total cycles", r["total_cycles"])
    m[1].metric("Ideal cycles", r["ideal_cycles"])
    m[2].metric("Stalls", r["stalls"])
    m[3].metric("CPI", f"{r['cpi']:.2f}")
    save_result_button(subject, "pipe", "Pipeline", f"forwarding={fwd}: {r['total_cycles']} cycles, {r['stalls']} stalls")


def tool_ca_performance(subject):
    t1, t2, t3 = st.tabs(["CPU time", "CPI from an instruction mix", "Amdahl's law"])
    with t1:
        c = st.columns(3)
        ic = c[0].number_input("Instruction count", value=1e9, format="%e")
        cpi = c[1].number_input("CPI", value=2.0)
        ghz = c[2].number_input("Clock (GHz)", value=2.0)
        st.code(S.cpu_performance(ic, cpi, ghz * 1e9)["steps"][0], language="text")
    with t2:
        mix = st.data_editor(pd.DataFrame([{"class": "ALU", "fraction": 0.5, "CPI": 1.0}, {"class": "load/store", "fraction": 0.3, "CPI": 2.0},
                                           {"class": "branch", "fraction": 0.2, "CPI": 3.0}]), num_rows="dynamic", width="stretch")
        try:
            r = S.weighted_cpi([(row["class"], float(row["fraction"]), float(row["CPI"])) for _, row in mix.iterrows()])
            st.code(f"CPI = {r['steps'][0]}", language="text")
        except (S.SimError, ValueError) as e:
            st.error(str(e))
    with t3:
        c = st.columns(2)
        f = c[0].slider("Fraction improved", 0.0, 1.0, 0.8)
        s = c[1].number_input("Speed-up of that part", value=4.0, min_value=0.1)
        r = S.amdahl(f, s)
        st.code("\n".join(r["steps"]), language="text")
        xs = np.linspace(1, 50, 100)
        fig = go.Figure(go.Scatter(x=xs, y=[S.amdahl(f, x)["speedup"] for x in xs], mode="lines", line=dict(color=PALETTE[0], width=3)))
        if np.isfinite(r["limit"]):
            fig.add_hline(y=r["limit"], line_dash="dash", annotation_text="limit")
        fig.update_layout(title="Overall speedup vs speed-up of the improved part", xaxis_title="speed-up factor",
                          yaxis_title="overall speedup", height=320, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        save_result_button(subject, "amdahl", "Amdahl's law", "\n".join(r["steps"]))


# ---------------------------------------------------------------------------
# Compiler Construction
# ---------------------------------------------------------------------------
def tool_cc_lexer(subject):
    src = st.text_area("Source code", "int count = 0;\nwhile (count <= 10) {\n  total = total + count * 2; // add\n  count = count + 1;\n}", height=150)
    r = S.tokenize(src)
    badges("computed")
    st.metric("Tokens", len(r["tokens"]))
    c1, c2 = st.columns([3, 2])
    with c1:
        _df(r["tokens"])
    with c2:
        st.markdown("**Symbol table**")
        _df(r["symbols"])
    for e in r["errors"]:
        st.error(e)
    save_result_button(subject, "lex", "Lexical analysis", " ".join(f"{t['token']}({t['lexeme']})" for t in r["tokens"]))


DEFAULT_GRAMMAR = "E -> T E'\nE' -> + T E' | eps\nT -> F T'\nT' -> * F T' | eps\nF -> ( E ) | id"


def tool_cc_first_follow(subject):
    c1, c2 = st.columns([3, 2])
    gtext = c1.text_area("Grammar (one rule per line, | for alternatives, eps for epsilon)", DEFAULT_GRAMMAR, height=150)
    inp = c2.text_input("Input tokens to parse (space separated)", "id + id * id")
    try:
        start, g = S.parse_grammar(gtext)
        ff = S.first_follow(start, g)
    except S.SimError as e:
        st.error(str(e)); return
    badges("computed")
    _df([{"non-terminal": A, "FIRST": "{ " + ", ".join(ff["first"][A]) + " }", "FOLLOW": "{ " + ", ".join(ff["follow"][A]) + " }"} for A in g])
    table = {a: [" | ".join(f"{A} -> {' '.join(p) or 'eps'}" for p in ff["table"].get((A, a), [])) for A in g] for a in ff["terminals"]}
    st.markdown("**LL(1) parsing table**")
    st.dataframe(pd.DataFrame(table, index=list(g)), width="stretch")
    if ff["is_ll1"]:
        st.success("The grammar is LL(1): every cell has at most one production.")
        r = S.ll1_parse(start, g, inp.split())
        st.markdown("**Parse trace**")
        _df(r["steps"])
        (st.success if r["accepted"] else st.error)("Input accepted" if r["accepted"] else "Syntax error")
    else:
        st.error("Not LL(1): conflicts in " + ", ".join(f"M[{A}, {a}]" for A, a in ff["conflicts"]))
    save_result_button(subject, "ff", "FIRST/FOLLOW", str({"FIRST": ff["first"], "FOLLOW": ff["follow"]}))


def tool_cc_shift_reduce(subject):
    c1, c2 = st.columns([3, 2])
    gtext = c1.text_area("Grammar", "E -> E + T | T\nT -> T * F | F\nF -> ( E ) | id", height=110)
    inp = c2.text_input("Input tokens", "id * id + id")
    try:
        start, g = S.parse_grammar(gtext)
        r = S.slr_parse(start, g, inp.split())
    except S.SimError as e:
        st.error(str(e)); return
    badges("computed")
    st.caption(f"SLR(1) automaton with {r['states']} states" + (f"; conflicts: {r['conflicts']}" if r["conflicts"] else "; no conflicts"))
    _df(r["steps"])
    (st.success if r["accepted"] else st.error)("Input accepted" if r["accepted"] else "Syntax error")
    save_result_button(subject, "slr", "Shift-reduce parse", "\n".join(f"{s['stack']} | {s['input']} | {s['action']}" for s in r["steps"]))


def tool_cc_tac(subject):
    c1, c2 = st.columns([3, 1])
    expr = c1.text_input("Arithmetic expression", "b * c + b * c - 2 * 3")
    target = c2.text_input("Assign to", "a")
    try:
        r = S.three_address_code(expr, target or "x")
    except S.SimError as e:
        st.error(str(e)); return
    badges("computed")
    c1, c2 = st.columns(2)
    c1.markdown("**Three-address code**")
    c1.code("\n".join(r["tac"]), language="text")
    c2.markdown("**After optimization**")
    c2.code("\n".join(r["optimized"]), language="text")
    for n in r["notes"]:
        c2.caption(n)
    st.markdown("**Quadruples**")
    _df(r["quadruples"])
    save_result_button(subject, "tac", "Three-address code", "\n".join(r["tac"]))


# ---------------------------------------------------------------------------
# Linear Algebra and ML maths (reuse the Mathematics playground solvers)
# ---------------------------------------------------------------------------
def _solver(subject, kinds, key):
    from .math_page import _form_inputs, show_solution

    kind = st.selectbox("Problem type", kinds, format_func=lambda k: M.KINDS[k]["en"], key=f"{key}_kind")
    solve = _form_inputs(kind)
    if st.button("Solve step by step", type="primary", key=f"{key}_solve"):
        try:
            st.session_state[f"{key}_res"] = solve()
        except (M.MathInputError, ValueError) as e:
            st.session_state[f"{key}_res"] = None
            st.error(f"Could not solve: {e}")
    res = st.session_state.get(f"{key}_res")
    if res and res["kind"] == kind:
        show_solution(res)
        save_result_button(subject, f"{key}_save", M.KINDS[kind]["en"],
                           "\n".join(s["en"] for s in res["steps"]) + f"\nAnswer: {res['answer_text']}")


def tool_la_solver(subject):
    _solver(subject, ["gaussian_elimination", "matrix_multiply", "determinant", "inverse", "eigen", "vectors"], "la")


def tool_ml_math(subject):
    _solver(subject, ["statistics", "bayes", "binomial", "derivative", "integral", "optimize", "gradient_descent"], "mlm")


def tool_ml_playground(subject):
    from .ml_page import ml_page

    ml_page()


# ---------------------------------------------------------------------------
# Web Engineering
# ---------------------------------------------------------------------------
class _Checker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.issues, self.labels, self.inputs = [], [], set(), []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag)
        if tag == "img" and not a.get("alt"):
            self.issues.append("An <img> has no alt text.")
        if tag == "label" and a.get("for"):
            self.labels.add(a["for"])
        if tag == "input" and a.get("type") not in ("submit", "button", "hidden"):
            self.inputs.append(a.get("id"))


def tool_web_live(subject):
    topics = load_topics()
    examples = {t["title"]["en"]: t["code"] for t in topics.values()
                if t["subject"] == "Web Engineering" and t.get("code_language") == "html" and t.get("code")}
    pick = st.selectbox("Start from an example", list(examples))
    code = st.text_area("Your HTML / CSS / JavaScript", examples[pick], height=260, key=f"web_code_{pick}")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Live preview** (runs in a sandboxed frame)")
        components.html(f"<div style='font-family:sans-serif;color:#243447'>{code}</div>", height=320, scrolling=True)
    with c2:
        st.markdown("**Automatic checks**")
        chk = _Checker()
        try:
            chk.feed(code)
        except Exception as e:  # malformed markup
            st.error(f"Could not parse the HTML: {e}")
        semantic = [t for t in chk.tags if t in ("header", "nav", "main", "section", "article", "aside", "footer")]
        headings = [t for t in chk.tags if re.fullmatch(r"h[1-6]", t)]
        unlabeled = [i for i in chk.inputs if i not in chk.labels]
        badges("computed")
        st.write(f"Elements: {len(chk.tags)} - semantic: {', '.join(sorted(set(semantic))) or 'none'}")
        st.write(f"Headings: {' > '.join(headings) or 'none'}")
        for issue in chk.issues + [f"Input '{i}' has no matching <label for>." for i in unlabeled if i] + \
                ([f"{len([i for i in unlabeled if not i])} input(s) have no id/label."] if any(i is None for i in unlabeled) else []):
            st.warning(issue)
        if not chk.issues and not unlabeled:
            st.success("No accessibility problems found by these checks.")
    save_result_button(subject, "web", "Web practice code", code)


def tool_web_http(subject):
    from http import HTTPStatus
    from urllib.parse import parse_qs, urlparse

    t1, t2, t3 = st.tabs(["URL anatomy", "Build a request", "Status codes"])
    with t1:
        url = st.text_input("URL", "https://uni.edu:443/courses/search?subject=web&level=3#top")
        u = urlparse(url)
        _df([{"part": k, "value": str(v)} for k, v in [("scheme", u.scheme), ("host", u.hostname), ("port", u.port or ("443" if u.scheme == "https" else "80")),
                                                       ("path", u.path), ("query", parse_qs(u.query)), ("fragment", u.fragment)]])
    with t2:
        c = st.columns([1, 3])
        method = c[0].selectbox("Method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
        path = c[1].text_input("Path", "/api/v1/courses")
        body = st.text_area("JSON body", '{"code": "CS301", "title": "Web Engineering"}' if method in ("POST", "PUT", "PATCH") else "", height=80)
        raw = f"{method} {path} HTTP/1.1\nHost: uni.edu\nAccept: application/json\n"
        if body.strip():
            raw += f"Content-Type: application/json\nContent-Length: {len(body.encode())}\n\n{body}"
        st.code(raw, language="http")
        props = {"GET": (True, True), "POST": (False, False), "PUT": (False, True), "PATCH": (False, False), "DELETE": (False, True)}
        st.write(f"Safe: {props[method][0]} - Idempotent: {props[method][1]}")
    with t3:
        code = st.number_input("Status code", 100, 599, 404)
        try:
            s = HTTPStatus(int(code))
            cls = {1: "informational", 2: "success", 3: "redirection", 4: "client error", 5: "server error"}[int(code) // 100]
            st.success(f"{int(code)} {s.phrase} - {cls}. {s.description}")
        except ValueError:
            st.warning("Not a standard status code.")


TOOLS = {
    "os_scheduling": tool_os_scheduling, "os_page_replacement": tool_os_page_replacement, "os_paging": tool_os_paging,
    "os_bankers": tool_os_bankers, "ca_numbers": tool_ca_numbers, "ca_cache": tool_ca_cache, "ca_pipeline": tool_ca_pipeline,
    "ca_performance": tool_ca_performance, "cc_lexer": tool_cc_lexer, "cc_first_follow": tool_cc_first_follow,
    "cc_shift_reduce": tool_cc_shift_reduce, "cc_tac": tool_cc_tac, "la_solver": tool_la_solver,
    "la_elimination": tool_la_solver, "ml_playground": tool_ml_playground, "ml_math": tool_ml_math,
    "web_live": tool_web_live, "web_http": tool_web_http,
}


def algorithm_practice(subject: dict) -> None:
    tools = [t for t in subject["tools"] if t in TOOLS]
    if "la_elimination" in tools and "la_solver" in tools:
        tools.remove("la_elimination")
    choice = st.radio("Tool", tools, format_func=TOOL_NAMES.get, horizontal=True, key=f"tool_{subject['id']}")
    st.markdown(f"#### {TOOL_NAMES[choice]}")
    TOOLS[choice](subject["name"])


# ---------------------------------------------------------------------------
# Numerical practice
# ---------------------------------------------------------------------------
def numerical_practice(subject: dict) -> None:
    name = subject["name"]
    kinds = G.SUBJECT_KINDS.get(name, [])
    if not kinds:
        st.info("No numerical problems for this subject.")
        return
    kind = st.selectbox("Problem type", kinds, format_func=G.title, key=f"np_kind_{subject['id']}")
    key = f"np_{subject['id']}"
    current = st.session_state.get(key)
    if st.button("New problem", key=f"{key}_new") or current is None or current["kind"] != kind:
        current = G.generate(kind, int(np.random.default_rng().integers(1_000_000)))
        current["state"] = {"tries": 0, "hint": False, "show": False, "last": None, "n": np.random.randint(1e9)}
        st.session_state[key] = current
    p, state = current, current["state"]
    st.markdown(f"<div class='cs-card'>{esc(content(p['q']))}</div>", unsafe_allow_html=True)
    ans = st.text_input("Your answer", key=f"{key}_ans_{state['n']}")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Check answer", type="primary", key=f"{key}_check") and ans.strip():
        res = G.check(p, ans)
        state["tries"] += 1
        state["last"] = res["status"]
        activity().add_practice(name, kind, p["q"]["en"], res["status"] == "correct")
    if c2.button("Show hint", key=f"{key}_hint"):
        state["hint"] = True
    if c3.button("Show solution", key=f"{key}_sol", disabled=state["tries"] == 0):
        state["show"] = True
    if c4.button("Save problem", key=f"{key}_save"):
        store(name).save_material("numerical problem", G.title(kind), p["q"]["en"] + "\nAnswer: " + str(p["answer"]) +
                                  "\n" + "\n".join(p["steps"]))
        st.success(f"Saved in {name} > Saved.")
    if state["last"] == "correct":
        st.success(content({"en": "Correct! Well done.", "ru": "Sahi jawab! Shabash."}))
    elif state["last"] == "incorrect":
        st.error(content({"en": "Not correct yet - here is a hint.", "ru": "Abhi sahi nahi - yeh hint dekhein."}))
        state["hint"] = True
    if state["hint"]:
        st.markdown(f"<div class='cs-quote'>Hint: {esc(content(p['hint']))}</div>", unsafe_allow_html=True)
    if state["show"] or state["last"] == "correct":
        badges("computed")
        st.code("\n".join(p["steps"]) + f"\n\nAnswer: {p['answer']:g}", language="text")
