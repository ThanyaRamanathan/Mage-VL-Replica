from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .budget import allocate_exact
from .selectors import budgeted_anchor_mask, selected_coordinates, uniform_mask

@dataclass(frozen=True)
class ClipRecord:
    video_id: str
    rows: tuple[dict, ...]
    center: int


def _read_rows(path: Path, split: str) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if split != "all" and row.get("split") != split:
            continue
        video_id = row.get("video_id") or Path(row["frame_path"]).parent.name
        groups.setdefault(video_id, []).append(row)
    for video_rows in groups.values():
        video_rows.sort(key=lambda row: int(row.get("frame_idx", 0)))
    if not groups:
        raise ValueError(f"no rows for split {split!r} in {path}")
    return groups


def _window_starts(length: int, clip_frames: int, stride: int) -> list[int]:
    if length <= clip_frames:
        return [0]
    starts = list(range(0, length - clip_frames + 1, stride))
    if starts[-1] != length - clip_frames:
        starts.append(length - clip_frames)
    return starts


def _resize_score_grid(scores: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    tensor = torch.as_tensor(scores, dtype=torch.float32)[:, None]
    return F.interpolate(tensor, (grid_h, grid_w), mode="area")[:, 0].numpy()


class SurgicalClipDataset(Dataset):
    """Sliding temporal clips labeled by the center-frame surgical phase."""

    def __init__(self, manifest: str | Path, split: str, budget: int, selector: str,
                 clip_frames: int = 32, clip_stride: int = 8,
                 resolution: tuple[int, int] = (224, 224), patch_size: int = 16,
                 codec_scores: str | Path | None = None,
                 clinical_scores: str | Path | None = None,
                 anchor_fraction: float = 0.5):
        groups = _read_rows(Path(manifest), split)
        self.records = []
        for video_id, rows in sorted(groups.items()):
            for start in _window_starts(len(rows), clip_frames, clip_stride):
                window = tuple(rows[start:start + clip_frames])
                self.records.append(ClipRecord(video_id, window, len(window) // 2))
        self.budget, self.selector = budget, selector
        self.resolution, self.patch_size = resolution, patch_size
        self.codec_scores = Path(codec_scores) if codec_scores else None
        self.clinical_scores = Path(clinical_scores) if clinical_scores else None
        self.anchor_fraction = anchor_fraction
        capacity = clip_frames * (resolution[0] // patch_size) * (resolution[1] // patch_size)
        if budget > capacity:
            raise ValueError(f"budget {budget} exceeds clip capacity {capacity}")

    def __len__(self):
        return len(self.records)

    def _load_frames(self, rows: tuple[dict, ...]) -> torch.Tensor:
        frames = []
        pil_size = (self.resolution[1], self.resolution[0])
        for row in rows:
            with Image.open(row["frame_path"]) as image:
                array = np.asarray(image.convert("RGB").resize(pil_size), dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames)

    @staticmethod
    def _cache_indices(cache, rows: tuple[dict, ...]) -> np.ndarray:
        if "frame_idx" in cache:
            lookup = {int(value): i for i, value in enumerate(cache["frame_idx"])}
            try:
                return np.asarray([lookup[int(row["frame_idx"])] for row in rows])
            except KeyError as error:
                raise ValueError(f"frame {error.args[0]} is absent from score cache") from error
        return np.asarray([int(row["frame_idx"]) for row in rows]).clip(0, len(cache["scores"]) - 1)

    def _cached_mask(self, directory: Path, record: ClipRecord, grid_h: int,
                     grid_w: int, require_anchors: bool,
                     anchor_directory: Path | None = None) -> np.ndarray:
        path = directory / f"{record.video_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing score cache: {path}")
        with np.load(path) as cache:
            indices = self._cache_indices(cache, record.rows)
            scores = _resize_score_grid(cache["scores"][indices], grid_h, grid_w)
            mandatory = None
            if require_anchors:
                anchor_path = (anchor_directory or directory) / f"{record.video_id}.npz"
                with np.load(anchor_path) as anchors:
                    anchor_indices = self._cache_indices(anchors, record.rows)
                    if "keyframes" not in anchors:
                        raise ValueError(f"codec cache lacks keyframes required for dense anchors: {anchor_path}")
                    mandatory = budgeted_anchor_mask(
                        anchors["keyframes"][anchor_indices], grid_h, grid_w,
                        self.budget, self.anchor_fraction)
        return allocate_exact(scores, self.budget, mandatory)

    def _mask(self, record: ClipRecord) -> np.ndarray:
        grid_h = self.resolution[0] // self.patch_size
        grid_w = self.resolution[1] // self.patch_size
        if self.selector == "uniform":
            return uniform_mask(len(record.rows), grid_h, grid_w, self.budget)
        if self.selector == "codec":
            if self.codec_scores is None:
                raise ValueError("codec selection requires --codec-scores directory")
            return self._cached_mask(self.codec_scores, record, grid_h, grid_w, True)
        if self.selector == "clinical":
            if self.clinical_scores is None:
                raise ValueError("clinical selection requires --clinical-scores directory")
            if self.codec_scores is None:
                raise ValueError("clinical selection requires codec cache for the matched anchor policy")
            return self._cached_mask(self.clinical_scores, record, grid_h, grid_w, True,
                                     anchor_directory=self.codec_scores)
        raise ValueError(f"unknown selector: {self.selector}")

    def __getitem__(self, index: int):
        record = self.records[index]
        frames = self._load_frames(record.rows)
        coordinates = selected_coordinates(self._mask(record))
        patches = []
        for time, row, column in coordinates:
            y, x = row * self.patch_size, column * self.patch_size
            patches.append(frames[time, :, y:y + self.patch_size, x:x + self.patch_size])
        target_row = record.rows[record.center]
        phase = int(target_row["phase_id"])
        near_transition = float(target_row.get("near_transition", 0))
        return (torch.stack(patches), torch.tensor(coordinates, dtype=torch.long), phase,
                near_transition)


def pad_batch(batch):
    patches, coordinates, phases, transitions = zip(*batch)
    max_tokens = max(item.shape[0] for item in patches)
    padded = torch.zeros(len(batch), max_tokens, *patches[0].shape[1:], dtype=patches[0].dtype)
    coords = torch.zeros(len(batch), max_tokens, 3, dtype=torch.long)
    padding = torch.ones(len(batch), max_tokens, dtype=torch.bool)
    for index, (item, location) in enumerate(zip(patches, coordinates)):
        padded[index, :item.shape[0]] = item
        coords[index, :item.shape[0]] = location
        padding[index, :item.shape[0]] = False
    return (padded, coords, padding, torch.as_tensor(phases, dtype=torch.long),
            torch.as_tensor(transitions, dtype=torch.float32))
