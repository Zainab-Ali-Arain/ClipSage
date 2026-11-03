"""
Dataset exploration and documentation (week 3 milestone, Member 1).

Writes reports/dataset_report.md with the size of the collected dataset and
charts of videos per subject/channel, video durations and segment lengths.

Usage:
    python -m clipsage.explore
"""

import csv
from collections import Counter

import numpy as np

from . import config
from .collect import load_existing_videos
from .utils import read_jsonl


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    videos = load_existing_videos()
    if not videos:
        raise SystemExit("No videos collected yet - run `python -m clipsage.collect` first.")
    segments = list(read_jsonl(config.SEGMENTS_FILE)) if config.SEGMENTS_FILE.exists() else []
    log = {}
    if config.COLLECTION_LOG.exists():
        with open(config.COLLECTION_LOG, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                log[row["video_id"]] = row["status"]

    with_transcript = {v for v in videos if (config.TRANSCRIPT_DIR / f"{v}.json").exists()}
    used = {s["video_id"] for s in segments}
    durations = np.array([float(videos[v]["duration_seconds"] or 0) / 60 for v in used]) if used else np.array([])
    seg_len = np.array([s["end"] - s["start"] for s in segments]) if segments else np.array([])
    seg_words = np.array([len(s["text"].split()) for s in segments]) if segments else np.array([])
    by_subject = Counter(videos[v]["subject"] for v in used)
    by_channel = Counter(videos[v]["channel_title"] for v in used)
    kinds = Counter(s["transcript_kind"] for s in {s["video_id"]: s for s in segments}.values())
    status = Counter(log.values())

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if by_subject:
        names, counts = zip(*by_subject.most_common())
        axes[0, 0].barh(names, counts, color="#2563eb")
    axes[0, 0].set_title("Videos per subject")
    if by_channel:
        names, counts = zip(*by_channel.most_common(10))
        axes[0, 1].barh(names, counts, color="#0d9488")
    axes[0, 1].set_title("Videos per channel (top 10)")
    if durations.size:
        axes[1, 0].hist(durations, bins=20, color="#7c3aed")
    axes[1, 0].set_title("Video duration (minutes)")
    if seg_len.size:
        axes[1, 1].hist(seg_len, bins=20, color="#c2410c")
    axes[1, 1].set_title("Segment length (seconds)")
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "dataset_overview.png", dpi=150)
    plt.close(fig)

    hours = sum(float(videos[v]["duration_seconds"] or 0) for v in used) / 3600
    lines = [
        "# ClipSage dataset", "",
        "| | |", "|---|---|",
        f"| Videos with metadata | {len(videos)} |",
        f"| Videos with English transcript | {len(with_transcript)} |",
        f"| Videos used (non-empty after cleaning) | {len(used)} |",
        f"| Total lecture time | {hours:.1f} hours |",
        f"| Segments | {len(segments)} |",
    ]
    if seg_len.size:
        lines += [
            f"| Segment length | mean {seg_len.mean():.1f} s, min {seg_len.min():.1f} s, max {seg_len.max():.1f} s |",
            f"| Words per segment | mean {seg_words.mean():.0f} |",
        ]
    lines += ["", "## Videos per subject", "", "| Subject | Videos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in by_subject.most_common()]
    lines += ["", "## Channels", "", "| Channel | Videos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in by_channel.most_common()]
    lines += ["", "## Transcript types", "", "| Type | Videos |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in kinds.most_common()]
    if status:
        lines += ["", "## Transcript collection status", "", "| Status | Videos |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in status.most_common()]
    lines += ["", "![Dataset overview](dataset_overview.png)", ""]
    out = config.REPORTS_DIR / "dataset_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
