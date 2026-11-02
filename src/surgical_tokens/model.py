from __future__ import annotations

import math

import torch
from torch import nn


class ClinicalImportance(nn.Module):
    """Cheap patch scorer trained with phase/tool auxiliary supervision."""

    def __init__(self, channels: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 5, 4, 2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.GELU(),
            nn.Conv2d(hidden, 1, 2, 2),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # B,T,C,H,W -> B,T,H/16,W/16
        b, t, c, h, w = frames.shape
        return self.net(frames.reshape(b * t, c, h, w)).reshape(b, t, h // 16, w // 16)


class ClinicalImportancePredictor(nn.Module):
    """Weakly supervised frame scorer used to build clinical score caches."""

    def __init__(self, num_phases: int = 10, hidden: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 5, 4, 2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1), nn.GELU(),
        )
        self.grid = nn.AdaptiveAvgPool2d((14, 14))
        self.score = nn.Conv2d(hidden, 1, 1)
        self.phase = nn.Linear(hidden, num_phases)
        self.transition = nn.Linear(hidden, 1)

    def forward(self, frames: torch.Tensor):
        if frames.ndim == 4:
            features = self.grid(self.features(frames))
            scores = self.score(features).squeeze(1)
            weights = scores.flatten(1).softmax(1).reshape_as(scores)
            pooled = (features * weights[:, None]).flatten(2).sum(-1)
            return scores, self.phase(pooled), self.transition(pooled).squeeze(1)
        if frames.ndim != 5:
            raise ValueError("clinical scorer expects [B,C,H,W] or [B,T,C,H,W]")
        batch, time, channels, height, width = frames.shape
        features = self.grid(self.features(
            frames.reshape(batch * time, channels, height, width)))
        feature_shape = features.shape
        features = features.reshape(batch, time, feature_shape[1],
                                    feature_shape[2], feature_shape[3])
        scores = self.score(features.reshape(batch * time, *feature_shape[1:]))
        scores = scores.reshape(batch, time, feature_shape[2], feature_shape[3])
        weights = scores.flatten(1).softmax(1).reshape_as(scores)
        pooled = (features * weights[:, :, None]).sum((1, 3, 4))
        return scores, self.phase(pooled), self.transition(pooled).squeeze(1)


class SparseTemporalClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, max_frames: int = 64,
                 tune_backbone=False, encoder_chunk: int = 256,
                 architecture: str = "frame_pool"):
        super().__init__()
        import timm

        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        if not hasattr(self.encoder, "patch_embed") or not hasattr(self.encoder.patch_embed, "proj"):
            raise ValueError(f"{backbone} is not a supported ViT with patch_embed.proj")
        dim = self.encoder.num_features
        for p in self.encoder.parameters():
            p.requires_grad = tune_backbone
        self.time = nn.Embedding(max_frames, dim)
        self.tune_backbone = tune_backbone
        if architecture not in {"frame_pool", "joint_sparse"}:
            raise ValueError(f"unknown architecture: {architecture}")
        self.architecture = architecture
        input_size = self.encoder.default_cfg.get("input_size", (3, 224, 224))
        self.image_size = tuple(input_size[-2:])
        patch_size = self.encoder.patch_embed.patch_size
        self.patch_size = tuple(patch_size) if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        mean = self.encoder.default_cfg.get("mean", (0.485, 0.456, 0.406))
        std = self.encoder.default_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 4, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.tune_backbone:
            self.encoder.eval()
        return self

    def _spatial_positions(self, coordinates: torch.Tensor) -> torch.Tensor:
        num_prefix = int(getattr(self.encoder, "num_prefix_tokens", 1))
        spatial = self.encoder.pos_embed[:, num_prefix:]
        grid_h = self.image_size[0] // self.patch_size[0]
        grid_w = self.image_size[1] // self.patch_size[1]
        if spatial.shape[1] != grid_h * grid_w:
            raise ValueError("backbone positional grid does not match configured input/patch size")
        indices = coordinates[:, 1] * grid_w + coordinates[:, 2]
        return spatial[0, indices]

    def _encode_frame_batch(self, patches: torch.Tensor,
                            coordinates: torch.Tensor) -> torch.Tensor:
        """Encode equal-length frame fragments in one ViT call.

        Frames remain attention-independent, preserving the v0.2 architecture,
        but grouping by token count removes the per-frame Python/encoder-call
        artifact from latency measurements.
        """
        if patches.ndim != 5 or coordinates.ndim != 3:
            raise ValueError("batched frame patches/coordinates must be [B,N,...]")
        batch, tokens, channels, height, width = patches.shape
        x = patches.reshape(batch * tokens, channels, height, width)
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.encoder.patch_embed.proj(x).flatten(1).reshape(batch, tokens, -1)
        positions = self._spatial_positions(coordinates.reshape(batch * tokens, 3))
        x = x + positions.reshape(batch, tokens, -1)
        if hasattr(self.encoder, "pos_drop"):
            x = self.encoder.pos_drop(x)
        if hasattr(self.encoder, "norm_pre"):
            x = self.encoder.norm_pre(x)
        context = torch.enable_grad() if self.tune_backbone else torch.no_grad()
        with context:
            for block in self.encoder.blocks:
                x = block(x)
            x = self.encoder.norm(x)
        return x.mean(1)

    def _encode_frame(self, patches: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return self._encode_frame_batch(patches.unsqueeze(0), coordinates.unsqueeze(0))[0]

    @staticmethod
    def _sinusoidal_time(frame_ids: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        scale = -math.log(10000.0) / max(1, half - 1)
        frequencies = torch.exp(torch.arange(half, device=frame_ids.device) * scale)
        angles = frame_ids.to(torch.float32).unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < dim:
            embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
        return embedding

    def _forward_joint(self, patches: torch.Tensor, coords: torch.Tensor,
                       padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is not None and padding_mask.any():
            raise ValueError("joint_sparse requires equal exact token budgets within a batch")
        batch, tokens, channels, height, width = patches.shape
        x = patches.reshape(batch * tokens, channels, height, width)
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.encoder.patch_embed.proj(x).flatten(1).reshape(batch, tokens, -1)
        spatial = self._spatial_positions(coords.reshape(batch * tokens, 3))
        x = x + spatial.reshape(batch, tokens, -1)
        x = x + self._sinusoidal_time(
            coords[..., 0].clamp_max(self.time.num_embeddings - 1), x.shape[-1])
        if hasattr(self.encoder, "pos_drop"):
            x = self.encoder.pos_drop(x)
        if hasattr(self.encoder, "norm_pre"):
            x = self.encoder.norm_pre(x)
        context = torch.enable_grad() if self.tune_backbone else torch.no_grad()
        with context:
            for block in self.encoder.blocks:
                x = block(x)
            x = self.encoder.norm(x)
        self.last_encoder_calls = 1
        self.last_present_frames = int(coords[..., 0].unique().numel())
        return self.head(x.mean(1))

    def _forward_frame_pool(self, patches: torch.Tensor, coords: torch.Tensor,
                            padding_mask=None):
        b, _, _, h, w = patches.shape
        if (h, w) != self.patch_size:
            raise ValueError(f"selected patches are {(h, w)}, backbone requires {self.patch_size}")
        frame_ids = coords[..., 0].clamp_max(self.time.num_embeddings - 1)
        token_valid = ~padding_mask if padding_mask is not None else torch.ones(
            coords.shape[:2], dtype=torch.bool, device=patches.device)
        clip_length = int(frame_ids[token_valid].max().item()) + 1
        frame_z = patches.new_zeros((b, clip_length, self.encoder.num_features))
        frame_present = torch.zeros((b, clip_length), dtype=torch.bool, device=patches.device)
        groups: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        for batch_index in range(b):
            for frame_id in frame_ids[batch_index][token_valid[batch_index]].unique(sorted=True):
                selected = token_valid[batch_index] & (frame_ids[batch_index] == frame_id)
                token_count = int(selected.sum().item())
                groups.setdefault(token_count, []).append(
                    (batch_index, int(frame_id.item()), selected))
        for items in groups.values():
            grouped_patches = torch.stack(
                [patches[batch_index, selected] for batch_index, _, selected in items])
            grouped_coords = torch.stack(
                [coords[batch_index, selected] for batch_index, _, selected in items])
            encoded = self._encode_frame_batch(grouped_patches, grouped_coords)
            for item_index, (batch_index, frame_id, _) in enumerate(items):
                frame_z[batch_index, frame_id] = encoded[item_index]
                frame_present[batch_index, frame_id] = True
        self.last_encoder_calls = len(groups)
        self.last_present_frames = int(frame_present.sum().item())
        frame_z = frame_z + self.time.weight[:clip_length].unsqueeze(0)
        z = self.temporal(frame_z, src_key_padding_mask=~frame_present)
        pooled = (z * frame_present[..., None]).sum(1) / frame_present.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

    def forward(self, patches: torch.Tensor, coords: torch.Tensor, padding_mask=None):
        _, _, _, h, w = patches.shape
        if (h, w) != self.patch_size:
            raise ValueError(f"selected patches are {(h, w)}, backbone requires {self.patch_size}")
        if self.architecture == "joint_sparse":
            return self._forward_joint(patches, coords, padding_mask)
        return self._forward_frame_pool(patches, coords, padding_mask)
