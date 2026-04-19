"""PTv3 backbone wrapper.

PointTransformerV3 is vendored as a git submodule under
``third_party/PointTransformerV3`` and is not a pip-installable package.
We prepend its directory to ``sys.path`` at import time and import the
``PointTransformerV3`` class directly from ``model.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PTV3_PATH = _REPO_ROOT / "third_party" / "PointTransformerV3"
if str(_PTV3_PATH) not in sys.path:
    sys.path.insert(0, str(_PTV3_PATH))

from model import PointTransformerV3  # noqa: E402  (import after sys.path munge)


class SubdetectorEmbedding(nn.Module):
    """Per-subdetector linear embedding + learned subdetector-type token.

    Each hit carries ``subdet_id`` identifying its source subsystem
    (tracker, ECAL, HCAL, RICH, ...). We project each subdetector's raw
    features into a common width with its own ``nn.Linear`` and add a
    learned per-subdetector embedding. The output is a flat (N, D)
    feature tensor suitable for PTv3's ``feat`` input.
    """

    def __init__(
        self,
        in_features_per_subdet: list[int] | int,
        embed_dim: int,
        n_subdetectors: int,
    ) -> None:
        super().__init__()
        if isinstance(in_features_per_subdet, int):
            in_features_per_subdet = [in_features_per_subdet] * n_subdetectors
        assert len(in_features_per_subdet) == n_subdetectors
        self.n_subdetectors = n_subdetectors
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList(
            [nn.Linear(f, embed_dim) for f in in_features_per_subdet]
        )
        self.type_embed = nn.Embedding(n_subdetectors, embed_dim)

    def forward(self, feat: torch.Tensor, subdet_id: torch.Tensor) -> torch.Tensor:
        out = feat.new_zeros((feat.shape[0], self.embed_dim))
        for s, proj in enumerate(self.projs):
            mask = subdet_id == s
            if mask.any():
                out[mask] = proj(feat[mask])
        out = out + self.type_embed(subdet_id)
        return out


class PTv3Backbone(nn.Module):
    """Thin wrapper around the vendored PointTransformerV3.

    Forwards a dict with ``coord``, ``feat``, ``offset`` (plus the
    optional ``grid_size``) into PTv3 and returns the resulting ``Point``
    dict so downstream heads can consume ``point.feat``.

    TODO: wire in the subdetector embedding as a pre-PTv3 step when the
    real dataset is available.
    """

    def __init__(
        self,
        in_channels: int = 32,
        order: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: tuple[int, ...] = (2, 2, 2, 2),
        enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2),
        enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        enc_num_head: tuple[int, ...] = (2, 4, 8, 16, 32),
        enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: tuple[int, ...] = (2, 2, 2, 2),
        dec_channels: tuple[int, ...] = (64, 64, 128, 256),
        dec_num_head: tuple[int, ...] = (4, 4, 8, 16),
        dec_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024),
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
        enable_flash: bool = True,
        shuffle_orders: bool = True,
        grid_size: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.out_channels = dec_channels[0]
        self.net = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            enable_flash=enable_flash,
            shuffle_orders=shuffle_orders,
            **kwargs,
        )

    def forward(self, data: dict[str, torch.Tensor]) -> Any:
        data = dict(data)
        data.setdefault("grid_size", self.grid_size)
        return self.net(data)
