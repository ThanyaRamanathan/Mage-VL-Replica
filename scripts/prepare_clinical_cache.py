from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from surgical_tokens.model import ClinicalImportancePredictor


def main():
    parser = argparse.ArgumentParser(description="Convert a clinical scorer checkpoint to per-video patch scores")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    num_phases = int(checkpoint.get("num_phases", 10)) if isinstance(checkpoint, dict) else 10
    model = ClinicalImportancePredictor(num_phases=num_phases).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    rows = defaultdict(list)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row.get("video_id", Path(row["frame_path"]).parent.name)].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for video_id, video_rows in rows.items():
            ordered = sorted(video_rows, key=lambda item: int(item.get("frame_idx", 0)))
            scores = []
            for start in range(0, len(ordered), args.batch_size):
                frames = []
                for row in ordered[start:start + args.batch_size]:
                    with Image.open(row["frame_path"]) as image:
                        array = np.asarray(
                            image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
                    frames.append(torch.from_numpy(array).permute(2, 0, 1))
                batch = torch.stack(frames).to(device, non_blocking=True)
                score, _, _ = model(batch)
                scores.extend(score.cpu().numpy())
            np.savez_compressed(
                args.output / f"{video_id}.npz", scores=np.stack(scores),
                frame_idx=np.asarray([int(row.get("frame_idx", 0)) for row in ordered]),
                scorer_checkpoint=str(args.checkpoint.resolve()),
            )
            print(f"cached {video_id}: {len(scores)} frames", flush=True)


if __name__ == "__main__":
    main()
