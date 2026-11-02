from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from surgical_tokens.codec import iter_codec, patch_importance


def main():
    parser = argparse.ArgumentParser(description="Precompute codec patch scores for training")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--motion-weight", type=float, default=0.5)
    parser.add_argument("--sample-step", type=int, default=25,
                        help="cache every Nth decoded source frame (25 = 1 fps for Cataract-101)")
    parser.add_argument("--manifest", type=Path,
                        help="cache the exact source frame_idx values listed for each video")
    args = parser.parse_args()
    requested = defaultdict(set)
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                requested[row["video_id"]].add(int(row["frame_idx"]))
    args.output.mkdir(parents=True, exist_ok=True)
    videos = [path for path in args.videos.rglob("*.mp4")
              if "__MACOSX" not in path.parts and not path.name.startswith("._")]
    for video in sorted(videos):
        score_rows, frame_indices, pts_seconds, keyframes = [], [], [], []
        wanted = requested.get(video.stem)
        if args.manifest and not wanted:
            raise ValueError(f"manifest contains no source frames for {video.stem}")
        for index, frame in iter_codec(video):
            requested_out = wanted is not None and index not in wanted
            stride_out = wanted is None and index % args.sample_step
            if requested_out or stride_out:
                continue
            score_rows.append(patch_importance(frame, args.patch, args.motion_weight))
            frame_indices.append(index)
            pts_seconds.append(frame.pts_seconds)
            keyframes.append(frame.keyframe)
        scores = np.stack(score_rows)
        if wanted is not None and set(frame_indices) != wanted:
            missing = sorted(wanted - set(frame_indices))
            raise ValueError(f"video {video.stem} is missing requested source frames: {missing[:10]}")
        np.savez_compressed(
            args.output / f"{video.stem}.npz", scores=scores,
            frame_idx=np.asarray(frame_indices, dtype=np.int64),
            pts_seconds=np.asarray(pts_seconds, dtype=np.float64),
            keyframes=np.asarray(keyframes, dtype=bool),
            residual_signal="motion_compensated_l1_proxy",
        )
        print(f"cached {video.name}: {scores.shape}", flush=True)


if __name__ == "__main__":
    main()
