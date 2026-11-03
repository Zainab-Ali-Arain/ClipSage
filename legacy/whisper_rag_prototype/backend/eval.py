"""
eval.py
Evaluation harness for the ML course presentation.

Measures two things that matter for this project's core claim
("find the exact lecture moment that answers a question"):

1. Retrieval accuracy (Recall@1 and Recall@3) - for each test question,
   does the correct lecture segment show up in the top-1 / top-3 chunks
   returned by semantic search?
2. Timestamp precision - how far off (in seconds) is the top-ranked
   chunk's start time from the true start time of that concept in the
   video?

You provide the "ground truth" in eval_questions.json: for each test
question, the video it belongs to and the true start/end time where the
answer is actually taught (you get this by watching the video once and
noting the timestamp). This is a standard way to evaluate a retrieval
system when you don't have a large public benchmark - a small,
hand-labeled test set that's honest about being small.

Usage:
    python eval.py
    python eval.py my_other_questions.json

Outputs:
    - A results table printed to the console
    - eval_results.json (raw per-question results)
    - eval_report.png (bar chart + timestamp error histogram, for slides)
"""

import sys
import json
import statistics

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save the image
import matplotlib.pyplot as plt

from rag import retrieve_chunks

TOP_K = 3


def overlaps(a_start, a_end, b_start, b_end):
    """True if two time ranges overlap at all."""
    return a_start <= b_end and b_start <= a_end


def evaluate(questions_path: str):
    with open(questions_path, "r", encoding="utf-8-sig") as f:
        test_cases = json.load(f)

    if not test_cases:
        print("No test questions found in", questions_path)
        return

    results = []

    for case in test_cases:
        question = case["question"]
        expected_video = case["video_name"]
        expected_start = case["expected_start"]
        expected_end = case["expected_end"]

        retrieved = retrieve_chunks(question, top_k=TOP_K)

        def is_correct(chunk):
            return chunk["video_name"] == expected_video and overlaps(
                chunk["start"], chunk["end"], expected_start, expected_end
            )

        hit_at_1 = bool(retrieved) and is_correct(retrieved[0])
        hit_at_3 = any(is_correct(c) for c in retrieved)

        timestamp_error = None
        if retrieved:
            timestamp_error = abs(retrieved[0]["start"] - expected_start)

        results.append({
            "question": question,
            "expected_start": expected_start,
            "retrieved_start": retrieved[0]["start"] if retrieved else None,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "timestamp_error_sec": timestamp_error,
        })

    # ---- aggregate metrics ----
    n = len(results)
    recall_at_1 = sum(r["hit_at_1"] for r in results) / n
    recall_at_3 = sum(r["hit_at_3"] for r in results) / n
    errors = [r["timestamp_error_sec"] for r in results if r["timestamp_error_sec"] is not None]
    mean_error = statistics.mean(errors) if errors else None
    median_error = statistics.median(errors) if errors else None

    # ---- print report ----
    print(f"\n{'Question':<45} {'Hit@1':<7} {'Hit@3':<7} {'Time Err (s)':<12}")
    print("-" * 75)
    for r in results:
        q = (r["question"][:42] + "...") if len(r["question"]) > 42 else r["question"]
        err = f"{r['timestamp_error_sec']:.2f}" if r["timestamp_error_sec"] is not None else "n/a"
        print(f"{q:<45} {str(r['hit_at_1']):<7} {str(r['hit_at_3']):<7} {err:<12}")

    print("-" * 75)
    print(f"Recall@1: {recall_at_1:.0%}   Recall@3: {recall_at_3:.0%}")
    if mean_error is not None:
        print(f"Mean timestamp error: {mean_error:.2f}s   Median: {median_error:.2f}s")
    print(f"\n({n} test question{'s' if n != 1 else ''} evaluated - see eval_questions.json to add more)")

    # ---- save raw results ----
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "mean_timestamp_error_sec": mean_error,
            "median_timestamp_error_sec": median_error,
            "per_question": results,
        }, f, indent=2)

    # ---- chart for slides ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Recall@1", "Recall@3"], [recall_at_1 * 100, recall_at_3 * 100], color=["#d9a441", "#a67c2e"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("%")
    axes[0].set_title("Retrieval Accuracy")
    for i, v in enumerate([recall_at_1 * 100, recall_at_3 * 100]):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center")

    if errors:
        axes[1].hist(errors, bins=min(10, len(errors)), color="#d9a441", edgecolor="#1a1608")
        axes[1].set_xlabel("Timestamp error (seconds)")
        axes[1].set_ylabel("Number of questions")
        axes[1].set_title("Timestamp Precision")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("eval_report.png", dpi=150)
    print("\nSaved eval_results.json and eval_report.png (use the PNG in your slides).")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_questions.json"
    evaluate(path)
