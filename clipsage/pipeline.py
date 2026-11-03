"""
End-to-end query pipeline: topic -> explanation + references + visuals.

    from clipsage.pipeline import ClipSage
    result = ClipSage().ask("overfitting")

Command line:
    python -m clipsage.pipeline "overfitting" --method sbert
"""

import time
from typing import Dict, List, Optional

from . import config
from .keywords import extract_keywords
from .retrieve import get_engine
from .summarize import abstractive_summary, extractive_summary
from .topics import subtopics
from .utils import format_timestamp, youtube_link
from .visualize import build_concept_graph


def build_references(explanation: List[dict], hits: List[dict]) -> List[dict]:
    """
    One reference per video, numbered in order of first use in the explanation,
    followed by the other retrieved videos.
    """
    refs: Dict[str, dict] = {}

    def add(video_id, title, channel, start):
        if video_id not in refs:
            refs[video_id] = {"n": len(refs) + 1, "video_id": video_id, "title": title,
                              "channel_title": channel, "timestamps": []}
        ts = refs[video_id]["timestamps"]
        if all(abs(t["start"] - start) > 20 for t in ts):
            ts.append({"start": start, "label": format_timestamp(start), "url": youtube_link(video_id, start)})

    for s in explanation:
        add(s["video_id"], s["title"], s["channel_title"], s["start"])
    for h in hits:
        add(h["video_id"], h["title"], h["channel_title"], h["start"])
    for r in refs.values():
        r["timestamps"].sort(key=lambda t: t["start"])
        r["used_in_explanation"] = any(s["video_id"] == r["video_id"] for s in explanation)
    return list(refs.values())


class ClipSage:
    def __init__(self):
        self.engine = get_engine()

    def ask(self, query: str, method: str = "sbert", summarizer: str = "textrank",
            keyword_method: Optional[str] = None, top_k: int = config.TOP_K_SEGMENTS,
            n_sentences: int = config.SUMMARY_SENTENCES, abstractive: bool = False,
            with_subtopics: bool = True) -> dict:
        timings = {}
        t_total = t0 = time.perf_counter()
        keyword_method = keyword_method or ("keybert" if method == "sbert" else "tfidf")

        hits = self.engine.search(query, method=method, top_k=top_k)
        timings["retrieval"] = time.perf_counter() - t0

        result = {"query": query, "method": method, "summarizer": summarizer,
                  "keyword_method": keyword_method, "hits": hits}

        if not self.engine.is_covered(hits, method):
            result.update(covered=False, explanation=[], references=[], key_terms=[],
                          concept_graph=None, subtopics=[],
                          suggestions=self.engine.suggest_topics(query, method))
            timings["total"] = time.perf_counter() - t_total
            result["timings"] = timings
            return result

        t0 = time.perf_counter()
        explanation = extractive_summary(query, hits, summarizer, method, self.engine, n_sentences)
        timings["explanation"] = time.perf_counter() - t0

        if abstractive:
            t0 = time.perf_counter()
            result["abstractive"] = abstractive_summary(query, explanation, hits)
            timings["abstractive"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        texts = [h["text"] for h in hits]
        key_terms = extract_keywords(texts, query, keyword_method, engine=self.engine)
        graph = build_concept_graph(query, key_terms, hits)
        timings["key_terms_and_map"] = time.perf_counter() - t0

        groups = []
        if with_subtopics:
            t0 = time.perf_counter()
            vectors = self.engine.segment_vectors(method, [h["row"] for h in hits])
            groups = subtopics(hits, vectors)
            timings["subtopics"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_total
        result.update(covered=True, explanation=explanation,
                      references=build_references(explanation, hits),
                      key_terms=key_terms, concept_graph=graph, subtopics=groups,
                      suggestions=[], timings=timings)
        return result


def print_result(r: dict) -> None:
    print(f"\nQuery: {r['query']}   (retrieval: {r['method']}, summary: {r['summarizer']})\n")
    if not r["covered"]:
        print("This topic is not covered by the collected lectures.")
        if r["suggestions"]:
            print("Related topics you can try:", ", ".join(r["suggestions"]))
        return
    ref_no = {ref["video_id"]: ref["n"] for ref in r["references"]}
    print("Explanation:")
    for s in r["explanation"]:
        print(f"  {s['text']} [{ref_no[s['video_id']]} @ {s['timestamp']}]")
    if r.get("abstractive") and r["abstractive"].get("text"):
        print(f"\nAbstractive version ({r['abstractive']['model']}):\n  {r['abstractive']['text']}")
    print("\nReferences:")
    for ref in r["references"]:
        stamps = ", ".join(t["label"] for t in ref["timestamps"])
        print(f"  [{ref['n']}] {ref['title']} - {ref['channel_title']} ({stamps})")
        print(f"      {ref['timestamps'][0]['url']}")
    print("\nKey terms:", ", ".join(t for t, _ in r["key_terms"]))
    if r["subtopics"]:
        print("Sub-topics:", " | ".join(g["label"] for g in r["subtopics"]))
    print(f"\nTime: {r['timings']['total']:.2f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ask ClipSage from the command line")
    parser.add_argument("query")
    parser.add_argument("--method", choices=["sbert", "tfidf"], default="sbert")
    parser.add_argument("--summarizer", choices=["textrank", "embedding", "top_sentences"], default="textrank")
    parser.add_argument("--abstractive", action="store_true")
    args = parser.parse_args()
    print_result(ClipSage().ask(args.query, args.method, args.summarizer, abstractive=args.abstractive))


if __name__ == "__main__":
    main()
