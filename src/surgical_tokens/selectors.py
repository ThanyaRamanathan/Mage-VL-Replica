from __future__ import annotations

import numpy as np

from .budget import allocate_exact
from .codec import CodecFrame, patch_importance


def budgeted_anchor_mask(keyframes: np.ndarray, grid_h: int, grid_w: int,
                         budget: int, anchor_fraction: float = 0.5) -> np.ndarray:
    """Choose a spread subset of dense codec anchors while reserving sparse budget."""
    flags = np.asarray(keyframes, dtype=bool)
    if flags.ndim != 1:
        raise ValueError("keyframes must be a one-dimensional array")
    if not 0 <= anchor_fraction < 1:
        raise ValueError("anchor_fraction must be in [0, 1)")
    dense_tokens = grid_h * grid_w
    maximum = int((budget * anchor_fraction) // dense_tokens)
    candidates = np.flatnonzero(flags)
    if not len(candidates):
        candidates = np.asarray([0])
    count = min(len(candidates), maximum)
    chosen = (candidates[np.linspace(0, len(candidates) - 1, count).round().astype(int)]
              if count else np.asarray([], dtype=int))
    anchors = np.zeros((len(flags), grid_h, grid_w), dtype=bool)
    anchors[chosen] = True
    return anchors


def uniform_mask(num_frames: int, grid_h: int, grid_w: int, budget: int) -> np.ndarray:
    """Distribute exact tokens over time; dense frames when budget permits."""
    scores = np.full((num_frames, grid_h, grid_w), -1.0)
    frame_order = np.linspace(0, num_frames - 1, min(num_frames, max(1, budget // (grid_h * grid_w))))
    for rank, idx in enumerate(np.unique(np.rint(frame_order).astype(int))):
        scores[idx] = 1.0 - rank * 1e-9
    # Remaining partial-frame tokens are deterministic and spatially spread.
    yy, xx = np.mgrid[:grid_h, :grid_w]
    scores += ((yy * grid_w + xx) % 17)[None] * 1e-12
    return allocate_exact(scores, budget)


def codec_mask(frames: list[CodecFrame], budget: int, patch: int = 16,
               motion_weight: float = 0.5, anchor_fraction: float = 0.5):
    if not frames:
        raise ValueError("empty clip")
    h, w = frames[0].rgb.shape[:2]
    gh, gw = h // patch, w // patch
    scores = np.stack([patch_importance(f, patch, motion_weight) for f in frames])
    anchors = budgeted_anchor_mask(
        np.asarray([frame.keyframe for frame in frames]), gh, gw, budget, anchor_fraction)
    return allocate_exact(scores, budget, anchors), scores


def selected_coordinates(mask: np.ndarray) -> np.ndarray:
    """Return [N,3] (time,row,column) coordinates in deterministic order."""
    return np.argwhere(mask)


def clinical_mask(scores: np.ndarray, budget: int) -> np.ndarray:
    """Select an exact token budget from learned clinical importance scores."""
    if scores.ndim != 3 or not scores.size:
        raise ValueError("scores must have shape [time, grid_h, grid_w]")
    return allocate_exact(scores.astype(np.float32, copy=False), budget)
