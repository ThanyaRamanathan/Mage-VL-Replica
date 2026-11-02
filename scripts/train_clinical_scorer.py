from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from surgical_tokens.model import ClinicalImportancePredictor


class FrameLabels(Dataset):
    def __init__(self, manifest: Path, split: str, resolution=(224, 224)):
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
        self.resolution = resolution

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["frame_path"]).convert("RGB").resize(self.resolution)
        frame = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
        phase = int(row["phase_id"])
        transition = float(row.get("near_transition", 0))
        return frame, phase, transition


class ClipLabels(Dataset):
    """Complete clips make score offsets identifiable across time."""

    def __init__(self, manifest: Path, split: str, clip_frames: int = 32,
                 clip_stride: int = 32, resolution=(224, 224)):
        groups = defaultdict(list)
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == split:
                    groups[row["video_id"]].append(row)
        self.records = []
        for rows in groups.values():
            rows.sort(key=lambda row: int(row["frame_idx"]))
            starts = list(range(0, max(1, len(rows) - clip_frames + 1), clip_stride))
            tail = max(0, len(rows) - clip_frames)
            if not starts or starts[-1] != tail:
                starts.append(tail)
            for start in starts:
                window = rows[start:start + clip_frames]
                if len(window) == clip_frames:
                    self.records.append(window)
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rows = self.records[index]
        frames = []
        for row in rows:
            with Image.open(row["frame_path"]) as image:
                array = np.asarray(image.convert("RGB").resize(self.resolution), dtype=np.float32) / 255
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        center = rows[len(rows) // 2]
        return (torch.stack(frames), int(center["phase_id"]),
                float(center.get("near_transition", 0)))


def main():
    parser = argparse.ArgumentParser(description="Train the weakly supervised clinical patch scorer")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-phases", type=int, default=10)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--selection-rate", type=float, default=0.25)
    parser.add_argument("--training-mode", choices=("frame", "clip"), default="frame")
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--clip-stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = (ClipLabels(args.manifest, "train", args.clip_frames, args.clip_stride)
               if args.training_mode == "clip" else FrameLabels(args.manifest, "train"))
    loader = DataLoader(dataset, args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = ClinicalImportancePredictor(num_phases=args.num_phases).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, examples = 0.0, 0
        for frames, phase, transition in loader:
            frames, phase = frames.to(device), phase.to(device)
            transition = transition.to(device)
            scores, phase_logits, transition_logits = model(frames)
            phase_loss = torch.nn.functional.cross_entropy(phase_logits, phase)
            transition_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                transition_logits, transition)
            probabilities = scores.sigmoid()
            if probabilities.ndim == 4:
                tv = ((probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()
                      + (probabilities[:, :, 1:] - probabilities[:, :, :-1]).abs().mean()
                      + (probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]).abs().mean())
            else:
                tv = ((probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()
                      + (probabilities[:, :, 1:] - probabilities[:, :, :-1]).abs().mean())
            sparsity = (probabilities.mean() - args.selection_rate).square()
            loss = (phase_loss + args.transition_weight * transition_loss
                    + args.tv_weight * tv + args.sparsity_weight * sparsity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * frames.shape[0]
            examples += frames.shape[0]
        print(f"epoch={epoch} loss={total_loss / max(1, examples):.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_phases": args.num_phases, "seed": args.seed,
                "objective": "phase_transition_tv_sparsity",
                "training_mode": args.training_mode}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
