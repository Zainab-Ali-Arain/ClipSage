"""
Evaluation plan (proposal section 10).

Sub-commands (run them in this order the first time):

  pool       Build the relevance-labelling sheet: top-10 videos of BOTH methods
             for each of the 30 test queries, shuffled so the labeller cannot
             tell which method found them -> evaluation/relevance_labels.csv.
             The team marks relevant = 1 / 0 by hand.
  retrieval  Precision@5, Recall@10, MRR for TF-IDF vs Sentence-BERT
             (+ Wilcoxon signed-rank test on the per-query scores).
  summary    ROUGE-1/2/L of every explanation method against the reference
             summaries, and a sheet for the manual 1-5 ratings.
  keywords   Sheet for the manual check of extracted key terms (TF-IDF vs KeyBERT).
  sources    Automatic check that every reference link + timestamp points to
             the content it is quoted from (add --online to also confirm the
             videos still exist through the YouTube API).
  speed      Time from query to result for all test queries (target < 5 s).
  report     Collects everything (including the filled-in manual sheets and the
             usability survey) into reports/evaluation_report.md + charts.
  all        retrieval + summary + sources + speed + report.

Usage:
    python -m clipsage.evaluate pool
    python -m clipsage.evaluate all
"""

import argparse
import csv
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config
from .utils import read_json, write_json

EVAL_DIR = config.ROOT_DIR / "evaluation"
QUERIES_FILE = EVAL_DIR / "queries.json"
LABELS_FILE = EVAL_DIR / "relevance_labels.csv"
REFERENCE_SUMMARIES = EVAL_DIR / "reference_summaries.json"
RATINGS_FILE = EVAL_DIR / "summary_ratings.csv"
KEYWORD_SHEET = EVAL_DIR / "keyword_review.csv"
SURVEY_FILE = EVAL_DIR / "survey_responses.csv"

OUT = config.REPORTS_DIR
METHODS = ["tfidf", "sbert"]


def load_queries() -> List[dict]:
    return read_json(QUERIES_FILE)["queries"]


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _engine():
    from .retrieve import get_engine

    return get_engine()


def _available_methods() -> List[str]:
    from .index import index_info

    return METHODS if index_info().get("sbert_model") else ["tfidf"]


# ---------------------------------------------------------------------------
# Retrieval: Precision@5, Recall@10, MRR
# ---------------------------------------------------------------------------
def precision_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return sum(1 for v in ranked[:k] if v in relevant) / k


def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    return sum(1 for v in ranked[:k] if v in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: List[str], relevant: set) -> float:
    for i, v in enumerate(ranked, 1):
        if v in relevant:
            return 1.0 / i
    return 0.0


def pool(depth: int = 10, seed: int = 42) -> None:
    engine = _engine()
    existing = {(r["query_id"], r["video_id"]): r for r in read_csv(LABELS_FILE)}
    rows = list(existing.values())
    rng = random.Random(seed)
    added = 0
    for q in load_queries():
        found: Dict[str, dict] = {}
        for method in _available_methods():
            for v in engine.rank_videos(q["query"], method, k=depth):
                found.setdefault(v["video_id"], v)
        new = [v for vid, v in found.items() if (q["id"], vid) not in existing]
        rng.shuffle(new)
        for v in new:
            rows.append({
                "query_id": q["id"], "query": q["query"], "video_id": v["video_id"],
                "title": v["title"], "channel": v["channel_title"],
                "link": f"https://www.youtube.com/watch?v={v['video_id']}&t={int(v['start'])}s",
                "relevant": "",
            })
            added += 1
    rows.sort(key=lambda r: (int(r["query_id"]), r["video_id"]))
    write_csv(LABELS_FILE, rows, ["query_id", "query", "video_id", "title", "channel", "link", "relevant"])
    print(f"{added} new (query, video) pairs added -> {LABELS_FILE}")
    print("Open it in Excel/Sheets and put 1 (relevant) or 0 (not relevant) in the 'relevant' column.\n"
          "A video is relevant if it actually explains the query topic. Then run:"
          " python -m clipsage.evaluate retrieval")


def load_relevance() -> Dict[str, set]:
    rel: Dict[str, set] = {}
    for q in load_queries():
        rel[str(q["id"])] = set(q.get("relevant_videos", []))
    for row in read_csv(LABELS_FILE):
        if str(row.get("relevant", "")).strip() in {"1", "yes", "y", "true"}:
            rel.setdefault(str(row["query_id"]), set()).add(row["video_id"])
    return rel


def evaluate_retrieval() -> Optional[dict]:
    engine = _engine()
    relevance = load_relevance()
    queries = [q for q in load_queries() if relevance.get(str(q["id"]))]
    if not queries:
        print(f"No relevance labels yet. Run `python -m clipsage.evaluate pool` and fill in {LABELS_FILE}.")
        return None

    per_query = []
    summary = {}
    for method in _available_methods():
        p5s, r10s, rrs = [], [], []
        for q in queries:
            relevant = relevance[str(q["id"])]
            ranked = [v["video_id"] for v in engine.rank_videos(q["query"], method, k=100, pool=500)]
            p5 = precision_at_k(ranked, relevant, 5)
            r10 = recall_at_k(ranked, relevant, 10)
            rr = reciprocal_rank(ranked, relevant)
            p5s.append(p5), r10s.append(r10), rrs.append(rr)
            per_query.append({"method": method, "query_id": q["id"], "query": q["query"],
                              "n_relevant": len(relevant), "P@5": p5, "R@10": r10, "RR": rr})
        summary[method] = {"P@5": float(np.mean(p5s)), "R@10": float(np.mean(r10s)), "MRR": float(np.mean(rrs)),
                           "per_query": {"P@5": p5s, "R@10": r10s, "RR": rrs}}

    tests = {}
    if len(summary) == 2:
        from scipy.stats import wilcoxon

        for metric in ["P@5", "R@10", "RR"]:
            a = summary["tfidf"]["per_query"][metric]
            b = summary["sbert"]["per_query"][metric]
            if np.allclose(a, b):  # identical results - the test is undefined
                tests[metric] = {"statistic": None, "p_value": 1.0}
                continue
            try:
                stat, p = wilcoxon(b, a, alternative="greater")
                tests[metric] = {"statistic": float(stat), "p_value": float(p)}
            except ValueError:  # all differences are zero
                tests[metric] = {"statistic": None, "p_value": 1.0}

    result = {"n_queries": len(queries),
              "methods": {m: {k: v for k, v in s.items() if k != "per_query"} for m, s in summary.items()},
              "wilcoxon_sbert_gt_tfidf": tests}
    write_json(OUT / "retrieval_results.json", result)
    write_csv(OUT / "retrieval_per_query.csv", per_query,
              ["method", "query_id", "query", "n_relevant", "P@5", "R@10", "RR"])

    print(f"\nRetrieval on {len(queries)} labelled queries")
    print(f"{'Method':<12}{'P@5':>8}{'R@10':>8}{'MRR':>8}")
    for m, s in result["methods"].items():
        print(f"{m:<12}{s['P@5']:>8.3f}{s['R@10']:>8.3f}{s['MRR']:>8.3f}")
    for metric, t in tests.items():
        print(f"Wilcoxon (SBERT > TF-IDF) on {metric}: p = {t['p_value']:.4f}")
    _retrieval_chart(result)
    return result


def _retrieval_chart(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["P@5", "R@10", "MRR"]
    methods = list(result["methods"])
    x = np.arange(len(metrics))
    width = 0.38
    colors = {"tfidf": "#94a3b8", "sbert": "#2563eb"}
    labels = {"tfidf": "TF-IDF (baseline)", "sbert": "Sentence-BERT"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(methods):
        vals = [result["methods"][m][k] for k in metrics]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=labels[m], color=colors[m])
        ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Retrieval quality ({result['n_queries']} test queries)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "retrieval_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary quality: ROUGE + manual rating sheet
# ---------------------------------------------------------------------------
SUMMARY_CONFIGS = [
    ("baseline", "tfidf", "top_sentences"),
    ("sbert+top_sentences", "sbert", "top_sentences"),
    ("sbert+textrank", "sbert", "textrank"),
    ("sbert+embedding", "sbert", "embedding"),
]


def evaluate_summaries(include_abstractive: bool = False) -> Optional[dict]:
    from rouge_score import rouge_scorer

    from .pipeline import ClipSage

    refs = read_json(REFERENCE_SUMMARIES)["summaries"]
    sage = ClipSage()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    methods = _available_methods()
    configs = [c for c in SUMMARY_CONFIGS if c[1] in methods]

    scores: Dict[str, Dict[str, List[float]]] = {}
    sheet = []
    for ref in refs:
        for name, method, summarizer in configs:
            r = sage.ask(ref["query"], method=method, summarizer=summarizer, with_subtopics=False)
            text = " ".join(s["text"] for s in r["explanation"])
            s = scorer.score(ref["reference"], text) if text else None
            bucket = scores.setdefault(name, {"rouge1": [], "rouge2": [], "rougeL": []})
            for k in bucket:
                bucket[k].append(s[k].fmeasure if s else 0.0)
            sheet.append({"query": ref["query"], "system": name, "explanation": text,
                          "relevance_1to5": "", "clarity_1to5": ""})
            if include_abstractive and name == "sbert+textrank" and r["covered"]:
                from .summarize import abstractive_summary

                a = abstractive_summary(ref["query"], r["explanation"], r["hits"])
                if a and a.get("text"):
                    s = scorer.score(ref["reference"], a["text"])
                    bucket = scores.setdefault("sbert+bart", {"rouge1": [], "rouge2": [], "rougeL": []})
                    for k in bucket:
                        bucket[k].append(s[k].fmeasure)
                    sheet.append({"query": ref["query"], "system": "sbert+bart", "explanation": a["text"],
                                  "relevance_1to5": "", "clarity_1to5": ""})

    result = {"n_references": len(refs),
              "rouge_f1": {n: {k: float(np.mean(v)) for k, v in b.items()} for n, b in scores.items()}}
    write_json(OUT / "summary_results.json", result)

    # keep ratings already entered by the team
    old = {(r["query"], r["system"]): r for r in read_csv(RATINGS_FILE)}
    for row in sheet:
        prev = old.get((row["query"], row["system"]))
        if prev and prev.get("explanation") == row["explanation"]:
            row["relevance_1to5"], row["clarity_1to5"] = prev.get("relevance_1to5", ""), prev.get("clarity_1to5", "")
    rng = random.Random(1)
    rng.shuffle(sheet)
    write_csv(RATINGS_FILE, sheet, ["query", "system", "explanation", "relevance_1to5", "clarity_1to5"])

    print(f"\nROUGE F1 against {len(refs)} reference summaries")
    print(f"{'System':<22}{'R-1':>8}{'R-2':>8}{'R-L':>8}")
    for n, s in result["rouge_f1"].items():
        print(f"{n:<22}{s['rouge1']:>8.3f}{s['rouge2']:>8.3f}{s['rougeL']:>8.3f}")
    print(f"Manual rating sheet: {RATINGS_FILE} (fill relevance_1to5 and clarity_1to5)")
    return result


# ---------------------------------------------------------------------------
# Key-term quality (manual)
# ---------------------------------------------------------------------------
def keyword_sheet(n_queries: int = 10, n_terms: int = 10) -> None:
    """One query per subject in turn, so the sheet stays small enough to label by hand."""
    from .keywords import extract_keywords, keybert_available

    engine = _engine()
    old = {(r["query"], r["method"], r["term"]): r.get("relevant", "") for r in read_csv(KEYWORD_SHEET)}
    queries = load_queries()
    by_subject: Dict[str, List[dict]] = {}
    for q in queries:
        by_subject.setdefault(q.get("subject", ""), []).append(q)
    picked = []
    while len(picked) < min(n_queries, len(queries)):
        for qs in by_subject.values():
            if qs and len(picked) < n_queries:
                picked.append(qs.pop(0))
    rows = []
    for q in picked:
        hits = engine.search(q["query"], method=_available_methods()[-1])
        texts = [h["text"] for h in hits]
        for kw_method in ["tfidf", "keybert"]:
            terms = extract_keywords(texts, q["query"], kw_method, n=n_terms, engine=engine)
            if kw_method == "keybert" and not keybert_available():
                print("KeyBERT is not available - the sheet only contains TF-IDF terms.")
                break
            for term, _ in terms:
                rows.append({"query": q["query"], "method": kw_method, "term": term,
                             "relevant": old.get((q["query"], kw_method, term), "")})
    write_csv(KEYWORD_SHEET, rows, ["query", "method", "term", "relevant"])
    print(f"{len(rows)} key terms written to {KEYWORD_SHEET}. Mark relevant = 1 / 0, then run `report`.")


# ---------------------------------------------------------------------------
# Source accuracy (automatic)
# ---------------------------------------------------------------------------
def evaluate_sources(online: bool = False) -> dict:
    import os
    import re

    from .collect import load_existing_videos
    from .pipeline import ClipSage
    from .preprocess import clean_snippets

    videos = load_existing_videos()
    sage = ClipSage()
    transcripts: Dict[str, List[dict]] = {}
    checked = valid = 0
    problems = []
    cited_ids = set()
    link_re = re.compile(r"^https://www\.youtube\.com/watch\?v=[\w-]{6,20}&t=\d+s$")

    def norm(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower())

    for q in load_queries():
        r = sage.ask(q["query"], method=_available_methods()[-1], with_subtopics=False)
        for s in r["explanation"]:
            checked += 1
            vid, start = s["video_id"], s["start"]
            cited_ids.add(vid)
            reasons = []
            if not link_re.match(s["url"]):
                reasons.append("bad link format")
            if vid not in videos:
                reasons.append("video not in dataset")
            else:
                duration = float(videos[vid].get("duration_seconds") or 0)
                if not (0 <= start <= duration + 1):
                    reasons.append("timestamp outside video")
            if vid not in transcripts:
                path = config.TRANSCRIPT_DIR / f"{vid}.json"
                transcripts[vid] = clean_snippets(read_json(path)["snippets"]) if path.exists() else []
            window = " ".join(x["text"] for x in transcripts[vid] if start - 10 <= x["start"] <= s["end"] + 10)
            head = " ".join(norm(s["text"]).split()[:6])
            if head and head not in norm(window):
                reasons.append("quoted text not found at that time")
            if reasons:
                problems.append({"query": q["query"], "video_id": vid, "start": start,
                                 "text": s["text"][:80], "problem": "; ".join(reasons)})
            else:
                valid += 1

    online_result = None
    if online:
        from .collect import YouTubeCollector

        key = os.environ.get(config.YOUTUBE_API_KEY_ENV)
        if key:
            cited = sorted(cited_ids)
            found = YouTubeCollector(key).video_details(cited)
            online_result = {"videos_checked": len(cited), "still_available": len(found)}
        else:
            print("No API key - skipping the online check.")

    result = {"sentences_checked": checked, "valid": valid,
              "accuracy": valid / checked if checked else 0.0,
              "online": online_result, "problems": problems}
    write_json(OUT / "source_results.json", result)
    print(f"\nSource accuracy: {valid}/{checked} cited sentences point to the right video + time "
          f"({result['accuracy']:.1%})")
    for p in problems[:10]:
        print(f"  ! {p['query']}: {p['video_id']} @ {p['start']}: {p['problem']}")
    return result


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
def evaluate_speed(repeats: int = 1) -> dict:
    from .pipeline import ClipSage

    sage = ClipSage()
    queries = load_queries()
    result = {}
    for method in _available_methods():
        # warm-up so model loading / imports are not counted
        sage.ask(sage.engine.segments[0]["title"], method=method)
        times = []
        for _ in range(repeats):
            for q in queries:
                t0 = time.perf_counter()
                sage.ask(q["query"], method=method)
                times.append(time.perf_counter() - t0)
        times.sort()
        result[method] = {
            "queries": len(times), "mean_s": statistics.mean(times), "median_s": statistics.median(times),
            "p95_s": times[min(len(times) - 1, int(0.95 * len(times)))], "max_s": max(times),
            "under_5s": sum(t < 5 for t in times) / len(times),
        }
    write_json(OUT / "speed_results.json", result)
    print("\nTime from query to result (seconds)")
    for m, r in result.items():
        print(f"  {m:<6} mean {r['mean_s']:.2f}  median {r['median_s']:.2f}  p95 {r['p95_s']:.2f}  "
              f"max {r['max_s']:.2f}  under 5 s: {r['under_5s']:.0%}")
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _mean_of(rows, field):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def report() -> Path:
    lines = ["# ClipSage - Evaluation Report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}", ""]

    def load(name):
        p = OUT / name
        return read_json(p) if p.exists() else None

    r = load("retrieval_results.json")
    lines += ["## 1. Retrieval quality (target: embedding method clearly better than baseline)", ""]
    if r:
        lines += [f"{r['n_queries']} hand-labelled test queries.", "",
                  "| Method | Precision@5 | Recall@10 | MRR |", "|---|---|---|---|"]
        for m, s in r["methods"].items():
            name = "TF-IDF (baseline)" if m == "tfidf" else "Sentence-BERT (improved)"
            lines.append(f"| {name} | {s['P@5']:.3f} | {s['R@10']:.3f} | {s['MRR']:.3f} |")
        if r.get("wilcoxon_sbert_gt_tfidf"):
            lines += ["", "Wilcoxon signed-rank test (H1: SBERT > TF-IDF): " + ", ".join(
                f"{k} p = {v['p_value']:.4f}" for k, v in r["wilcoxon_sbert_gt_tfidf"].items())]
        lines += ["", "![Retrieval comparison](retrieval_comparison.png)", ""]
    else:
        lines += ["_Not run yet (needs relevance labels)._", ""]

    s = load("summary_results.json")
    ratings = read_csv(RATINGS_FILE)
    lines += ["## 2. Summary quality (target: average manual rating >= 4)", ""]
    if s:
        lines += [f"ROUGE F1 against {s['n_references']} reference summaries:", "",
                  "| System | ROUGE-1 | ROUGE-2 | ROUGE-L | Relevance (1-5) | Clarity (1-5) |",
                  "|---|---|---|---|---|---|"]
        for name, v in s["rouge_f1"].items():
            rows = [x for x in ratings if x["system"] == name]
            rel, _ = _mean_of(rows, "relevance_1to5")
            cla, _ = _mean_of(rows, "clarity_1to5")
            fmt = lambda x: f"{x:.2f}" if x is not None else "-"
            lines.append(f"| {name} | {v['rouge1']:.3f} | {v['rouge2']:.3f} | {v['rougeL']:.3f} | "
                         f"{fmt(rel)} | {fmt(cla)} |")
        lines.append("")
    else:
        lines += ["_Not run yet._", ""]

    kw = [x for x in read_csv(KEYWORD_SHEET) if str(x.get("relevant", "")).strip() in {"0", "1"}]
    lines += ["## 3. Key-term / concept-map quality (target: >= 80% relevant terms)", ""]
    if kw:
        lines += ["| Method | Terms checked | Relevant |", "|---|---|---|"]
        for m in ["tfidf", "keybert"]:
            rows = [x for x in kw if x["method"] == m]
            if rows:
                share = sum(x["relevant"].strip() == "1" for x in rows) / len(rows)
                lines.append(f"| {m} | {len(rows)} | {share:.0%} |")
        lines.append("")
    else:
        lines += [f"_Fill in {KEYWORD_SHEET.name} first._", ""]

    src = load("source_results.json")
    lines += ["## 4. Source accuracy (target: 100% valid links)", ""]
    if src:
        lines.append(f"{src['valid']}/{src['sentences_checked']} cited sentences link to the right video "
                     f"and timestamp: **{src['accuracy']:.1%}**.")
        if src.get("online"):
            o = src["online"]
            lines.append(f"Online check: {o['still_available']}/{o['videos_checked']} videos still available.")
        lines.append("")
    else:
        lines += ["_Not run yet._", ""]

    survey = read_csv(SURVEY_FILE)
    lines += ["## 5. Usability survey (10-15 students)", ""]
    if survey:
        fields = [f for f in survey[0] if f.startswith("q") and f[1:].split("_")[0].isdigit()]
        lines += [f"{len(survey)} responses (1 = strongly disagree, 5 = strongly agree).", "",
                  "| Question | Mean |", "|---|---|"]
        for f in fields:
            m, n = _mean_of(survey, f)
            if n:
                lines.append(f"| {f} | {m:.2f} |")
        lines.append("")
    else:
        lines += [f"_Collect responses in {SURVEY_FILE.name} (see docs/usability_survey.md)._", ""]

    sp = load("speed_results.json")
    lines += ["## 6. Speed (target: under 5 seconds)", ""]
    if sp:
        lines += ["| Method | Mean (s) | Median (s) | 95th pct (s) | Max (s) | Under 5 s |", "|---|---|---|---|---|---|"]
        for m, v in sp.items():
            lines.append(f"| {m} | {v['mean_s']:.2f} | {v['median_s']:.2f} | {v['p95_s']:.2f} | "
                         f"{v['max_s']:.2f} | {v['under_5s']:.0%} |")
        lines.append("")
    else:
        lines += ["_Not run yet._", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "evaluation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClipSage evaluation")
    parser.add_argument("command", choices=["pool", "retrieval", "summary", "keywords", "sources",
                                            "speed", "report", "all"])
    parser.add_argument("--online", action="store_true", help="sources: also check videos via the API")
    parser.add_argument("--abstractive", action="store_true", help="summary: also score BART")
    args = parser.parse_args(argv)

    if args.command == "pool":
        pool()
    elif args.command == "retrieval":
        evaluate_retrieval()
    elif args.command == "summary":
        evaluate_summaries(args.abstractive)
    elif args.command == "keywords":
        keyword_sheet()
    elif args.command == "sources":
        evaluate_sources(args.online)
    elif args.command == "speed":
        evaluate_speed()
    elif args.command == "report":
        report()
    elif args.command == "all":
        evaluate_retrieval()
        evaluate_summaries(args.abstractive)
        evaluate_sources(args.online)
        evaluate_speed()
        report()


if __name__ == "__main__":
    main()
