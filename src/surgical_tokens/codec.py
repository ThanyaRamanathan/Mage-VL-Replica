from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CodecFrame:
    rgb: np.ndarray
    pts_seconds: float
    keyframe: bool
    motion: np.ndarray  # H x W x 2, pixels in source-frame coordinates
    residual: np.ndarray  # H x W, absolute motion-compensated prediction error


def _motion_field(frame, height: int, width: int) -> np.ndarray:
    field = np.zeros((height, width, 2), dtype=np.float32)
    found = False
    for side in frame.side_data:
        if "MOTION_VECTORS" not in str(side.type):
            continue
        found = True
        for mv in side.to_ndarray():
            # PyAV/FFmpeg AVMotionVector structured field names are stable.
            names = mv.dtype.names or ()
            get = lambda key, default=0: mv[key].item() if key in names else default
            x, y = int(get("dst_x")), int(get("dst_y"))
            w, h = max(1, int(get("w", 16))), max(1, int(get("h", 16)))
            scale = max(1, int(get("motion_scale", 1)))
            dx = float(get("motion_x")) / scale
            dy = float(get("motion_y")) / scale
            x0, x1 = max(0, x - w // 2), min(width, x + (w + 1) // 2)
            y0, y1 = max(0, y - h // 2), min(height, y + (h + 1) // 2)
            field[y0:y1, x0:x1] = (dx, dy)
    if not found and not frame.key_frame:
        raise RuntimeError(
            "Decoder exposed no motion vectors. Re-encode with H.264/HEVC and use an "
            "FFmpeg/PyAV build supporting AV_FRAME_DATA_MOTION_VECTORS."
        )
    return field


def _prediction_error(rgb: np.ndarray, previous: np.ndarray | None, motion: np.ndarray) -> np.ndarray:
    if previous is None:
        return np.zeros(rgb.shape[:2], np.float32)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    sx = np.clip(np.rint(xx - motion[..., 0]).astype(int), 0, w - 1)
    sy = np.clip(np.rint(yy - motion[..., 1]).astype(int), 0, h - 1)
    predicted = previous[sy, sx]
    return np.abs(rgb.astype(np.float32) - predicted.astype(np.float32)).mean(-1)


def iter_codec(path: str | Path, limit: int | None = None):
    """Yield RGB, I/P status, codec MVs, and a motion-compensated residual proxy."""
    import av

    container = av.open(str(path), options={"flags2": "+export_mvs"})
    stream = container.streams.video[0]
    stream.codec_context.options = {**stream.codec_context.options, "flags2": "+export_mvs"}
    previous = None
    for i, frame in enumerate(container.decode(stream)):
        if limit is not None and i >= limit:
            break
        rgb = frame.to_ndarray(format="rgb24")
        h, w = rgb.shape[:2]
        motion = _motion_field(frame, h, w)
        residual = _prediction_error(rgb, previous, motion)
        pts = float(frame.pts * frame.time_base) if frame.pts is not None else i / float(stream.average_rate)
        yield i, CodecFrame(rgb, pts, bool(frame.key_frame), motion, residual)
        previous = rgb


def decode_codec(path: str | Path, limit: int | None = None) -> list[CodecFrame]:
    return [frame for _, frame in iter_codec(path, limit)]


def patch_importance(frame: CodecFrame, patch: int = 16, motion_weight: float = 0.5):
    """Aggregate normalized motion magnitude and residual energy per patch."""
    h, w = frame.rgb.shape[:2]
    gh, gw = h // patch, w // patch
    motion = np.linalg.norm(frame.motion[: gh * patch, : gw * patch], axis=-1)
    residual = frame.residual[: gh * patch, : gw * patch]

    def pool(x):
        return x.reshape(gh, patch, gw, patch).mean((1, 3))

    m, r = pool(motion), pool(residual)
    robust = lambda x: x / (np.quantile(x, 0.95) + 1e-6)
    return motion_weight * robust(m) + (1.0 - motion_weight) * robust(r)
