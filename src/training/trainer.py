"""Minimal training loop scaffold.

TODO: replace with a proper trainer (Lightning / accelerate / custom)
once the data pipeline is stable. For now this is a thin wrapper that
runs forward/backward on dummy batches so downstream code can be tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        backbone: torch.nn.Module,
        heads: torch.nn.Module,
        loss_fn: torch.nn.Module,
        config: TrainerConfig | None = None,
    ) -> None:
        self.backbone = backbone
        self.heads = heads
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        params = list(backbone.parameters()) + list(heads.parameters())
        self.optim = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

    def step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = self.config.device
        data = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        point = self.backbone(data)
        preds = self.heads(point["feat"])
        targets = {"object_id": data["object_id"]}
        losses = self.loss_fn(preds, targets)
        self.optim.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            self.config.grad_clip,
        )
        self.optim.step()
        return losses

    def fit(self, loader: DataLoader) -> None:
        self.backbone.train().to(self.config.device)
        self.heads.train().to(self.config.device)
        for epoch in range(self.config.max_epochs):
            for i, batch in enumerate(loader):
                losses = self.step(batch)
                if i % self.config.log_every == 0:
                    scalars = {k: float(v.detach()) for k, v in losses.items() if torch.is_tensor(v)}
                    print(f"[epoch {epoch} step {i}] {scalars}")
