import sys
from types import SimpleNamespace

import torch
from torch import nn

from surgical_tokens.model import ClinicalImportancePredictor, SparseTemporalClassifier


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (16, 16)
        self.proj = nn.Conv2d(3, 8, 16, 16)


class _FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.num_features = 8
        self.num_prefix_tokens = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 8))
        self.blocks = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()
        self.pos_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.default_cfg = {
            "input_size": (3, 32, 32),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.25, 0.25, 0.25),
        }


def test_clinical_predictor_supports_frame_and_clip_training():
    model = ClinicalImportancePredictor(num_phases=10, hidden=8)

    frame_scores, frame_phase, frame_transition = model(torch.rand(2, 3, 64, 64))
    assert frame_scores.shape == (2, 14, 14)
    assert frame_phase.shape == (2, 10)
    assert frame_transition.shape == (2,)

    clip_scores, clip_phase, clip_transition = model(torch.rand(2, 4, 3, 64, 64))
    assert clip_scores.shape == (2, 4, 14, 14)
    assert clip_phase.shape == (2, 10)
    assert clip_transition.shape == (2,)


def test_sparse_classifier_supports_batched_frame_pool_and_joint(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "timm",
        SimpleNamespace(create_model=lambda *args, **kwargs: _FakeViT()),
    )
    patches = torch.rand(2, 4, 3, 16, 16)
    coords = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
    ])
    padding = torch.zeros(2, 4, dtype=torch.bool)

    frame_model = SparseTemporalClassifier("fake", 3, max_frames=2)
    assert frame_model(patches, coords, padding).shape == (2, 3)
    assert frame_model.last_encoder_calls == 1
    assert frame_model.last_present_frames == 4

    joint_model = SparseTemporalClassifier("fake", 3, max_frames=2,
                                           architecture="joint_sparse")
    assert joint_model(patches, coords, padding).shape == (2, 3)
    assert joint_model.last_encoder_calls == 1
