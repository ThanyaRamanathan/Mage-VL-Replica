import csv

import numpy as np
from PIL import Image

from surgical_tokens.data import (
    SurgicalClipDataset,
    _normalize_score_grid,
    _resize_score_grid,
    _window_starts,
)


def test_window_starts_covers_video_tail():
    assert _window_starts(10, 4, 3) == [0, 3, 6]
    assert _window_starts(11, 4, 3) == [0, 3, 6, 7]
    assert _window_starts(2, 4, 3) == [0]


def test_score_grid_resampling_preserves_time_axis():
    scores = np.stack([np.zeros((4, 6)), np.ones((4, 6))])
    resized = _resize_score_grid(scores, 2, 3)
    assert resized.shape == (2, 2, 3)
    np.testing.assert_allclose(resized[0], 0)
    np.testing.assert_allclose(resized[1], 1)


def test_per_frame_rank_normalization_removes_offsets():
    scores = np.asarray([[[0.0, 2.0], [1.0, 3.0]],
                         [[100.0, 102.0], [101.0, 103.0]]])
    normalized = _normalize_score_grid(scores, "per_frame_rank")
    np.testing.assert_allclose(normalized[0], normalized[1])
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_dataset_builds_windows_and_uses_center_label(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["video_id", "frame_path", "frame_idx", "phase", "phase_id",
              "near_transition", "split"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(6):
            image = tmp_path / f"{index}.png"
            Image.fromarray(np.full((32, 32, 3), index, np.uint8)).save(image)
            writer.writerow({
                "video_id": "01", "frame_path": image, "frame_idx": index,
                "phase": "Incision" if index < 2 else "Rhexis",
                "phase_id": 0 if index < 2 else 2, "near_transition": 0, "split": "train",
            })
    dataset = SurgicalClipDataset(
        manifest, "train", budget=4, selector="uniform", clip_frames=4,
        clip_stride=2, resolution=(32, 32), patch_size=16,
    )
    assert len(dataset) == 2
    patches, coordinates, phase, _ = dataset[0]
    assert len(patches) == len(coordinates) == 4
    assert phase == 2


def test_codec_cache_enforces_dense_keyframe(tmp_path):
    manifest = tmp_path / "manifest.csv"
    image = tmp_path / "frame.png"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(image)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "video_id", "frame_path", "frame_idx", "phase", "phase_id",
            "near_transition", "split"
        ])
        writer.writeheader()
        for index in range(2):
            writer.writerow({"video_id": "01", "frame_path": image, "frame_idx": index,
                             "phase": "Incision", "phase_id": 0,
                             "near_transition": 0, "split": "train"})
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "01.npz", scores=np.zeros((2, 2, 2)), frame_idx=[0, 1],
             keyframes=[True, False])
    dataset = SurgicalClipDataset(manifest, "train", 5, "codec", clip_frames=2,
                                  resolution=(32, 32), patch_size=16, codec_scores=cache)
    _, coordinates, _, _ = dataset[0]
    assert (coordinates[:, 0] == 0).sum() == 4
