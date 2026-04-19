"""Object Condensation loss wrapper.

Wraps :func:`object_condensation.pytorch.losses.condensation_loss_tiger`
and adds optional payload regression / classification heads. Hits with
``object_id <= noise_threshold`` are excluded from payload losses.

All payload losses are hit-level: we regress / classify every foreground
hit directly against its broadcast per-object truth. This is simple and
works well for the shapes pseudo-dataset; for a real detector you would
typically weight by ``beta`` around the condensation points.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from object_condensation.pytorch.losses import condensation_loss_tiger
except ImportError:  # pragma: no cover
    _OC_PATH = Path(__file__).resolve().parents[2] / "third_party" / "object_condensation" / "src"
    if str(_OC_PATH) not in sys.path:
        sys.path.insert(0, str(_OC_PATH))
    from object_condensation.pytorch.losses import condensation_loss_tiger


class ObjectCondensationLoss(nn.Module):
    def __init__(
        self,
        q_min: float = 1.0,
        noise_threshold: int = 0,
        max_n_rep: int = 0,
        payload_weight: float = 1.0,
        shape_id_weight: float = 1.0,
        width_weight: float = 0.1,
        height_weight: float = 0.1,
        energy_weight: float = 1.0,
        momentum_weight: float = 1.0,
        torch_compile: bool = False,
    ) -> None:
        super().__init__()
        self.q_min = q_min
        self.noise_threshold = noise_threshold
        self.max_n_rep = max_n_rep
        self.payload_weight = payload_weight
        self.shape_id_weight = shape_id_weight
        self.width_weight = width_weight
        self.height_weight = height_weight
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight
        self.torch_compile = torch_compile

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        oc = condensation_loss_tiger(
            beta=preds["beta"],
            x=preds["x"],
            object_id=targets["object_id"],
            q_min=self.q_min,
            noise_threshold=self.noise_threshold,
            max_n_rep=self.max_n_rep,
            torch_compile=self.torch_compile,
        )
        losses: dict[str, torch.Tensor] = {f"oc_{k}": v for k, v in oc.items()}
        # Individual terms can be NaN when their denominator is 0 — e.g.
        # ``oc_noise`` when the event has no noise hits (as in the shapes
        # dataset, where background is dropped rather than labeled 0).
        # Treat those as contributing nothing.
        total = sum(
            torch.where(torch.isfinite(v), v, torch.zeros_like(v))
            for k, v in oc.items()
            if k != "n_rep" and torch.is_tensor(v)
        )
        losses["oc_total"] = total

        mask = targets["object_id"] > self.noise_threshold
        if mask.any() and self.payload_weight > 0:
            if "pid_logits" in preds and "shape_id_per_hit" in targets:
                ce = F.cross_entropy(
                    preds["pid_logits"][mask], targets["shape_id_per_hit"][mask]
                )
                losses["shape_id"] = ce
                total = total + self.payload_weight * self.shape_id_weight * ce

            if "width" in preds and "width_per_hit" in targets:
                mse_w = F.mse_loss(preds["width"][mask], targets["width_per_hit"][mask])
                losses["width"] = mse_w
                total = total + self.payload_weight * self.width_weight * mse_w

            if "height" in preds and "height_per_hit" in targets:
                mse_h = F.mse_loss(preds["height"][mask], targets["height_per_hit"][mask])
                losses["height"] = mse_h
                total = total + self.payload_weight * self.height_weight * mse_h

            # legacy (physics detector use)
            if "energy" in preds and "energy_per_hit" in targets:
                mse_e = F.mse_loss(preds["energy"][mask], targets["energy_per_hit"][mask])
                losses["energy"] = mse_e
                total = total + self.payload_weight * self.energy_weight * mse_e

            if "momentum" in preds and "momentum_per_hit" in targets:
                mse_p = F.mse_loss(preds["momentum"][mask], targets["momentum_per_hit"][mask])
                losses["momentum"] = mse_p
                total = total + self.payload_weight * self.momentum_weight * mse_p

        losses["total"] = total
        return losses
