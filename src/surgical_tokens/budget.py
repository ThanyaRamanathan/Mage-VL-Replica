from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    total: int
    grid_h: int
    grid_w: int

    @property
    def dense_frame_tokens(self) -> int:
        return self.grid_h * self.grid_w

    @classmethod
    def equivalent_dense_frames(cls, frames: int, height: int, width: int, patch: int = 16):
        gh, gw = height // patch, width // patch
        return cls(frames * gh * gw, gh, gw)


def allocate_exact(scores, budget: int, mandatory=None):
    """Return a boolean mask with exactly budget entries, respecting mandatory tokens."""
    import numpy as np

    x = np.asarray(scores, dtype=np.float64)
    if budget < 0 or budget > x.size:
        raise ValueError(f"budget {budget} outside [0, {x.size}]")
    required = np.zeros(x.shape, dtype=bool) if mandatory is None else np.asarray(mandatory, bool)
    if required.shape != x.shape:
        raise ValueError("mandatory and scores must have the same shape")
    n_required = int(required.sum())
    if n_required > budget:
        raise ValueError(f"{n_required} mandatory tokens exceed budget {budget}")
    out = required.copy()
    remaining = budget - n_required
    if remaining:
        candidates = np.flatnonzero(~required.ravel())
        order = np.argsort(-x.ravel()[candidates], kind="stable")[:remaining]
        out.ravel()[candidates[order]] = True
    return out
