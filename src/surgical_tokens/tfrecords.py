"""Convert the official TF-Cholec80 TFRecords into explicit PNGs and a CSV."""
from __future__ import annotations

import csv
import re
from pathlib import Path

PHASES = (
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderRetraction", "CleaningCoagulation",
    "GallbladderPackaging",
)
TOOLS = ("Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag")


def split_for_video(video_number: int, validation=(5, 12, 18, 24, 27, 31, 35, 39)) -> str:
    """Procedure-level first-40/last-40 split; video numbers are one-based."""
    if not 1 <= video_number <= 80:
        raise ValueError("video number must be in [1, 80]")
    if video_number > 40:
        return "test"
    return "val" if video_number in validation else "train"


def convert_tfrecords(source: Path, output: Path, manifest: Path) -> int:
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required only for TFRecord conversion") from error

    natural_key = lambda path: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]
    records = sorted((p for p in source.iterdir() if p.is_file()), key=natural_key)
    if len(records) != 80:
        raise ValueError(f"expected 80 per-video TFRecords, found {len(records)} in {source}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["video_id", "frame_path", "frame_idx", "time_seconds", "phase", "tools", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video_number, record in enumerate(records, start=1):
            frame_dir = output / f"video{video_number:02d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for raw in tf.data.TFRecordDataset(str(record)):
                example = tf.train.Example.FromString(bytes(raw.numpy()))
                feature = example.features.feature
                frame_idx = int(feature["frame_id"].int64_list.value[0])
                phase_idx = int(feature["phase"].int64_list.value[0])
                tool_bits = list(feature["instruments"].int64_list.value)
                png = bytes(feature["frame"].bytes_list.value[0])
                path = frame_dir / f"{frame_idx:06d}.png"
                path.write_bytes(png)
                writer.writerow({
                    "video_id": f"{video_number:02d}", "frame_path": str(path.resolve()),
                    "frame_idx": frame_idx, "time_seconds": float(frame_idx),
                    "phase": PHASES[phase_idx],
                    "tools": ";".join(x for x, active in zip(TOOLS, tool_bits) if active),
                    "split": split_for_video(video_number),
                })
                count += 1
    return count
