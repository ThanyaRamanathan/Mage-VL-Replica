from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BUDGETS = (1568, 3136, 6272)


def window_starts(length: int, clip_frames: int, stride: int) -> list[int]:
    if length <= clip_frames:
        return [0]
    starts = list(range(0, length - clip_frames + 1, stride))
    if starts[-1] != length - clip_frames:
        starts.append(length - clip_frames)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dense-anchor counts in cached clips")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--codec-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/anchor_audit.json"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--grid-height", type=int, default=14)
    parser.add_argument("--grid-width", type=int, default=14)
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if args.split != "all" and row["split"] != args.split:
                continue
            groups[row["video_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["frame_idx"]))

    audit_rows: list[dict[str, object]] = []
    for video_id, rows in sorted(groups.items()):
        with np.load(args.codec_scores / f"{video_id}.npz") as cache:
            lookup = {int(frame): i for i, frame in enumerate(cache["frame_idx"])}
            flags = np.asarray(cache["keyframes"], dtype=bool)
            for start in window_starts(len(rows), args.clip_frames, args.clip_stride):
                window = rows[start : start + args.clip_frames]
                indices = np.asarray([lookup[int(row["frame_idx"])] for row in window])
                native_count = int(flags[indices].sum())
                for budget in BUDGETS:
                    dense_tokens = args.grid_height * args.grid_width
                    allowed = int((budget * args.anchor_fraction) // dense_tokens)
                    fallback = native_count == 0 and allowed > 0
                    actual = min(native_count if native_count else int(fallback), allowed)
                    audit_rows.append(
                        {
                            "split": window[len(window) // 2]["split"],
                            "video_id": video_id,
                            "window_start_sample": start,
                            "budget": budget,
                            "allowed_dense_anchors": allowed,
                            "native_keyframes": native_count,
                            "actual_dense_anchors": actual,
                            "fallback_frame0": fallback,
                        }
                    )

    summaries = []
    for budget in BUDGETS:
        selected = [row for row in audit_rows if row["budget"] == budget]
        counts = [int(row["actual_dense_anchors"]) for row in selected]
        native = [int(row["native_keyframes"]) for row in selected]
        fallback = sum(bool(row["fallback_frame0"]) for row in selected)
        summaries.append(
            {
                "budget": budget,
                "windows": len(selected),
                "allowed_dense_anchors": int(selected[0]["allowed_dense_anchors"]),
                "actual_anchor_mean": statistics.mean(counts),
                "actual_anchor_median": statistics.median(counts),
                "actual_anchor_min": min(counts),
                "actual_anchor_max": max(counts),
                "native_keyframe_mean": statistics.mean(native),
                "fallback_windows": fallback,
                "fallback_fraction": fallback / len(selected),
                "actual_anchor_histogram": dict(sorted(Counter(counts).items())),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "clip_frames": args.clip_frames,
        "clip_stride": args.clip_stride,
        "anchor_fraction": args.anchor_fraction,
        "summaries": summaries,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
