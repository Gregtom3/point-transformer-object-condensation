"""Training loop with TensorBoard instrumentation.

The Trainer is minimal on purpose — it wires together backbone, heads,
and the OC loss, plus TB logging for scalars, gradient norms, the
learned cluster-coord embedding, and prediction-vs-truth canvas images.

Training loop: one optimizer step per batch. Validation runs once per
epoch. Checkpoints are dumped per epoch to ``ckpt_dir``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.inference.cluster import beta_threshold_cluster
from src.utils.tb_logging import (
    log_oc_embedding,
    log_prediction_canvas,
    log_scalars,
)


@dataclass
class TrainerConfig:
    max_epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TBConfig:
    log_dir: str = "runs/default"
    image_every: int = 200
    embedding_every: int = 500
    inference_t_beta: float = 0.1
    inference_t_d: float = 0.4


class Trainer:
    def __init__(
        self,
        backbone: torch.nn.Module,
        heads: torch.nn.Module,
        loss_fn: torch.nn.Module,
        config: TrainerConfig | None = None,
        tb_config: TBConfig | None = None,
        ckpt_dir: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.heads = heads
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        self.tb_cfg = tb_config or TBConfig()
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        params = list(backbone.parameters()) + list(heads.parameters())
        self.optim = torch.optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._step = 0
        self.writer = self._make_writer()

    def _make_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=self.tb_cfg.log_dir)
        except Exception:
            print("WARN: tensorboard not available; scalars will be printed only")
            return None

    # ---- per-step mechanics ------------------------------------------------

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self.config.device
        return {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

    def _forward(self, batch: dict[str, Any]) -> tuple[dict, dict]:
        data = {
            "coord": batch["coord"],
            "feat": batch["feat"],
            "offset": batch["offset"],
        }
        point = self.backbone(data)
        preds = self.heads(point["feat"])
        return preds, point

    def _loss(self, preds: dict, batch: dict) -> dict[str, torch.Tensor]:
        targets = {
            "object_id": batch["object_id"],
        }
        for k in ("shape_id_per_hit", "width_per_hit", "height_per_hit",
                  "energy_per_hit", "momentum_per_hit"):
            if k in batch and torch.is_tensor(batch[k]):
                targets[k] = batch[k]
        return self.loss_fn(preds, targets)

    def step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        preds, _ = self._forward(batch)
        losses = self._loss(preds, batch)

        self.optim.zero_grad()
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            self.config.grad_clip,
        )
        self.optim.step()

        if self.writer is not None:
            scalars = {k: float(v.detach()) for k, v in losses.items()
                       if torch.is_tensor(v) and v.dim() == 0}
            scalars["grad_norm"] = float(grad_norm)
            log_scalars(self.writer, "train", scalars, self._step)
            if self._step % self.tb_cfg.image_every == 0:
                self._log_canvas(batch, preds, tag="train/canvas")
            if self._step % self.tb_cfg.embedding_every == 0 and self._step > 0:
                log_oc_embedding(
                    self.writer, "train/oc_space",
                    preds["x"], batch["object_id"], self._step,
                )
            self.writer.add_histogram("train/beta", preds["beta"].detach(), self._step)
        self._step += 1
        return losses

    def _log_canvas(self, batch: dict, preds: dict, tag: str) -> None:
        # split batch back into events via offset; log first event only
        offset = batch["offset"].cpu().tolist()
        end = offset[0]
        coord = batch["coord"][:end]
        object_id = batch["object_id"][:end]
        x = preds["x"][:end]
        beta = preds["beta"][:end]
        pred_cluster = beta_threshold_cluster(
            beta, x, t_beta=self.tb_cfg.inference_t_beta,
            t_d=self.tb_cfg.inference_t_d,
        )
        frame = tuple(int(v) for v in batch["frame"][0].cpu().tolist())
        log_prediction_canvas(
            self.writer, tag, coord, object_id, pred_cluster, frame, self._step,
        )

    # ---- full loop ---------------------------------------------------------

    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.backbone.eval()
        self.heads.eval()
        sums: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                preds, _ = self._forward(batch)
                losses = self._loss(preds, batch)
                for k, v in losses.items():
                    if torch.is_tensor(v) and v.dim() == 0:
                        sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
        self.backbone.train()
        self.heads.train()
        return {k: v / max(1, n) for k, v in sums.items()}

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "heads": self.heads.state_dict(),
                "optim": self.optim.state_dict(),
                "step": self._step,
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        self.backbone.train().to(self.config.device)
        self.heads.train().to(self.config.device)
        for epoch in range(self.config.max_epochs):
            for i, batch in enumerate(train_loader):
                losses = self.step(batch)
                if i % self.config.log_every == 0:
                    scalars = {
                        k: float(v.detach())
                        for k, v in losses.items()
                        if torch.is_tensor(v) and v.dim() == 0
                    }
                    print(f"[epoch {epoch} step {i}] {scalars}")
            if val_loader is not None:
                val = self.validate(val_loader)
                print(f"[epoch {epoch}] val: {val}")
                if self.writer is not None:
                    log_scalars(self.writer, "val", val, self._step)
            if self.ckpt_dir is not None:
                self.save_checkpoint(self.ckpt_dir / f"epoch_{epoch:03d}.pt")
