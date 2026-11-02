from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASE_NAMES = (
    "Incision",
    "Viscous agent injection",
    "Rhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation and aspiration",
    "Capsule polishing",
    "Lens implant setting-up",
    "Viscous agent removal",
    "Tonifying and antibiotics",
)


def normalize(score: np.ndarray) -> np.ndarray:
    low, high = np.percentile(score, (5, 95))
    if high <= low:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - low) / (high - low), 0, 1).astype(np.float32)


def resize_score(score: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((normalize(score) * 255).astype(np.uint8))
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255


def top_mask(score: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    flat = score.ravel()
    count = max(1, round(flat.size * fraction))
    indices = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(score.shape)


def heatmap_overlay(image: Image.Image, score: np.ndarray) -> Image.Image:
    width, height = image.size
    values = resize_score(score, (width, height))
    heat = np.zeros((height, width, 4), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = (180 * values).astype(np.uint8)
    heat[..., 3] = (175 * values).astype(np.uint8)
    overlay = Image.fromarray(heat, "RGBA")
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_title(image: Image.Image, title: str) -> Image.Image:
    bar = 42
    canvas = Image.new("RGB", (image.width, image.height + bar), "white")
    canvas.paste(image, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 11), title, fill="#172033", font=ImageFont.load_default(size=18))
    return canvas


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    frame_root = project / "cataract101_frames"
    codec_root = project / "cataract101_codec_scores"
    clinical_root = project / "cataract101_clinical_scores_seed17"
    output = project / "reports" / "qualitative"
    output.mkdir(parents=True, exist_ok=True)

    metadata: dict[tuple[str, int], dict[str, str]] = {}
    with (project / "cataract101.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metadata[(row["video_id"], int(row["frame_idx"]))] = row

    candidates = []
    for case_dir in sorted(frame_root.glob("case_*")):
        frames = sorted(case_dir.glob("*.jpg"))
        codec_path = codec_root / f"{case_dir.name}.npz"
        clinical_path = clinical_root / f"{case_dir.name}.npz"
        if not frames or not codec_path.exists() or not clinical_path.exists():
            continue
        codec = np.load(codec_path)
        clinical = np.load(clinical_path)
        codec_scores = codec["scores"]
        clinical_scores = clinical["scores"]
        keyframes = codec["keyframes"]
        codec_lookup = {int(value): i for i, value in enumerate(codec["frame_idx"])}
        clinical_lookup = {int(value): i for i, value in enumerate(clinical["frame_idx"])}
        for frame_path in frames:
            frame_idx = int(frame_path.stem)
            if frame_idx not in codec_lookup or frame_idx not in clinical_lookup:
                continue
            ci, li = codec_lookup[frame_idx], clinical_lookup[frame_idx]
            if bool(keyframes[ci]):
                continue
            codec_small = resize_score(codec_scores[ci], (14, 14))
            clinical_small = normalize(clinical_scores[li])
            codec_mask, clinical_mask = top_mask(codec_small), top_mask(clinical_small)
            union = np.logical_or(codec_mask, clinical_mask).sum()
            overlap = np.logical_and(codec_mask, clinical_mask).sum() / max(1, union)
            candidates.append(
                {
                    "case": case_dir.name,
                    "frame_idx": frame_idx,
                    "path": frame_path,
                    "codec_index": ci,
                    "clinical_index": li,
                    "top20_jaccard": float(overlap),
                    "metadata": metadata.get((case_dir.name, frame_idx), {}),
                }
            )

    if not candidates:
        raise SystemExit("No aligned non-keyframes were available for qualitative export")

    # Choose the strongest selector disagreement from three different procedures.
    chosen = []
    used_cases = set()
    for candidate in sorted(candidates, key=lambda item: item["top20_jaccard"]):
        if candidate["case"] in used_cases:
            continue
        chosen.append(candidate)
        used_cases.add(candidate["case"])
        if len(chosen) == 3:
            break

    rows = []
    for candidate in chosen:
        codec = np.load(codec_root / f"{candidate['case']}.npz")
        clinical = np.load(clinical_root / f"{candidate['case']}.npz")
        codec_score = codec["scores"][candidate["codec_index"]]
        clinical_score = clinical["scores"][candidate["clinical_index"]]
        source = Image.open(candidate["path"]).convert("RGB")
        source.thumbnail((640, 480), Image.Resampling.LANCZOS)
        panels = [
            add_title(source.copy(), "Decoded frame"),
            add_title(heatmap_overlay(source, codec_score), "Codec motion + residual proxy"),
            add_title(heatmap_overlay(source, clinical_score), "Learned clinical importance"),
        ]
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), height), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        name = f"{candidate['case']}_frame{candidate['frame_idx']:08d}.jpg"
        canvas.save(output / name, quality=92)
        meta = candidate["metadata"]
        phase_id = int(meta["phase_id"]) if meta.get("phase_id", "") != "" else -1
        rows.append(
            {
                "image": name,
                "video_id": candidate["case"],
                "frame_idx": candidate["frame_idx"],
                "time_seconds": meta.get("time_seconds", ""),
                "phase_id": phase_id,
                "phase": PHASE_NAMES[phase_id] if 0 <= phase_id < len(PHASE_NAMES) else "",
                "near_transition": meta.get("near_transition", ""),
                "top20_jaccard": candidate["top20_jaccard"],
                "selection_note": "Low overlap means the selectors prioritize different regions.",
            }
        )

    with (output / "selection_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} selector-disagreement triptychs to {output}")


if __name__ == "__main__":
    main()
