from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgical_tokens.data import SurgicalClipDataset, pad_batch
from surgical_tokens.model import SparseTemporalClassifier
from train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a saved checkpoint and export per-clip predictions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-scores", type=Path)
    parser.add_argument("--clinical-scores", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    selector = checkpoint["selector"]
    budget = int(checkpoint["visual_tokens"])
    resolution = tuple(config["data"].get("resolution", [224, 224]))
    model_config = config["model"]

    dataset = SurgicalClipDataset(
        manifest=args.manifest,
        split="test",
        budget=budget,
        selector=selector,
        clip_frames=int(config["data"].get("clip_frames", 32)),
        clip_stride=int(config["data"].get("clip_stride", 8)),
        resolution=resolution,
        patch_size=int(config.get("codec", {}).get("patch_size", 16)),
        codec_scores=args.codec_scores,
        clinical_scores=args.clinical_scores,
        anchor_fraction=float(config.get("codec", config.get("importance", {})).get(
            "anchor_fraction", 0.5)),
        score_normalization=str(config.get("selection", {}).get(
            "score_normalization", "none")),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False,
        num_workers=args.workers, collate_fn=pad_batch, pin_memory=device.type == "cuda")
    model = SparseTemporalClassifier(
        model_config["backbone"], int(model_config.get("num_classes", 10)),
        max_frames=int(config["data"].get("clip_frames", 32)),
        tune_backbone=not bool(model_config.get("frozen", True)),
        architecture=str(model_config.get("architecture", "frame_pool")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, rows = evaluate(model, loader, device, collect_predictions=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reevaluation.json").write_text(
        json.dumps({"selector": selector, "visual_tokens": budget, **metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"selector": selector, "visual_tokens": budget,
                      "predictions": len(rows), **metrics}, indent=2))


if __name__ == "__main__":
    main()
