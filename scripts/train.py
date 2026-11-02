from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from surgical_tokens.data import SurgicalClipDataset, pad_batch
from surgical_tokens.model import SparseTemporalClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    predictions, targets, transitions, elapsed, examples = [], [], [], 0.0, 0
    with torch.inference_mode():
        for batch_index, (patches, coords, padding, phase, near_transition) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            patches, coords, padding = patches.to(device), coords.to(device), padding.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(patches, coords, padding)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(phase.tolist())
            transitions.extend(near_transition.tolist())
            examples += phase.numel()
    precision, recall, per_f1, support = precision_recall_fscore_support(
        targets, predictions, labels=list(range(model.head.out_features)), zero_division=0)
    transition_indices = [i for i, value in enumerate(transitions) if value >= 0.5]
    transition_accuracy = (float(accuracy_score(
        [targets[i] for i in transition_indices], [predictions[i] for i in transition_indices]))
        if transition_indices else None)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "latency_ms": float(elapsed * 1000 / max(1, examples)),
        "transition_window_accuracy": transition_accuracy,
        "per_class": [{"class_id": i, "precision": float(precision[i]),
                       "recall": float(recall[i]), "f1": float(per_f1[i]),
                       "support": int(support[i])} for i in range(len(support))],
    }


def main():
    parser = argparse.ArgumentParser(description="Train one matched-budget surgical selector")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="override data.manifest from the config")
    parser.add_argument("--selector", choices=("uniform", "codec", "clinical"))
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec-scores", type=Path)
    parser.add_argument("--clinical-scores", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    selector = args.selector or config["selector"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 17))
    seed_everything(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    resolution = tuple(config["data"].get("resolution", [224, 224]))
    dataset_args = dict(
        manifest=args.manifest or config["data"]["manifest"], split="train", budget=args.budget, selector=selector,
        clip_frames=int(config["data"].get("clip_frames", 32)), resolution=resolution,
        clip_stride=int(config["data"].get("clip_stride", 8)),
        patch_size=int(config.get("codec", {}).get("patch_size", 16)), codec_scores=args.codec_scores,
        clinical_scores=args.clinical_scores,
        anchor_fraction=float(config.get("codec", config.get("importance", {})).get(
            "anchor_fraction", 0.5)),
    )
    train_loader = DataLoader(SurgicalClipDataset(**dataset_args), batch_size=int(config["train"]["batch_size"]),
                              shuffle=True, num_workers=args.workers, collate_fn=pad_batch,
                              pin_memory=device.type == "cuda")
    val_args = {**dataset_args, "split": "val"}
    val_loader = DataLoader(SurgicalClipDataset(**val_args), batch_size=int(config["train"]["batch_size"]),
                            shuffle=False, num_workers=args.workers, collate_fn=pad_batch,
                            pin_memory=device.type == "cuda")
    model_config = config["model"]
    model = SparseTemporalClassifier(
        model_config["backbone"], int(model_config.get("num_classes", 10)),
        max_frames=int(config["data"].get("clip_frames", 32)),
        tune_backbone=not bool(model_config.get("frozen", True)))
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(config["train"]["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = args.epochs or int(config["train"]["epochs"])
    args.output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_index, (patches, coords, padding, phase, _) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            patches, coords, padding, phase = patches.to(device), coords.to(device), padding.to(device), phase.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(patches, coords, padding), phase)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics = evaluate(model, val_loader, device, args.max_eval_batches)
        metrics.update(epoch=epoch, loss=float(loss.detach().cpu()))
        print(json.dumps(metrics), flush=True)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "config": config, "selector": selector,
                        "visual_tokens": args.budget}, args.output / "best.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_args = {**dataset_args, "split": "test"}
    test_loader = DataLoader(SurgicalClipDataset(**test_args), batch_size=int(config["train"]["batch_size"]),
                             shuffle=False, num_workers=args.workers, collate_fn=pad_batch,
                             pin_memory=device.type == "cuda")
    test_metrics = evaluate(model, test_loader, device, args.max_eval_batches)
    (args.output / "results.json").write_text(json.dumps({
        "dataset": "cataract101", "selector": selector, "visual_tokens": args.budget,
        "seed": seed, **test_metrics, "validation_best_macro_f1": best,
    }, indent=2))


if __name__ == "__main__":
    main()
