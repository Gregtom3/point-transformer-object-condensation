"""Object Condensation loss wrapper.

This wraps :func:`object_condensation.pytorch.losses.condensation_loss_tiger`
from the vendored submodule, adding optional payload regression terms
(energy/momentum/PID) that are weighted by beta around truth particles.

The submodule is installed in editable mode via
``pip install -e third_party/object_condensation[pytorch]`` so the import
below resolves normally. If it is not installed we fall back to adding
the submodule ``src`` dir to ``sys.path``.
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
except ImportError:  # pragma: no cover - fallback for non-editable installs
    _OC_PATH = Path(__file__).resolve().parents[2] / "third_party" / "object_condensation" / "src"
    if str(_OC_PATH) not in sys.path:
        sys.path.insert(0, str(_OC_PATH))
    from object_condensation.pytorch.losses import condensation_loss_tiger


class ObjectCondensationLoss(nn.Module):
    """Condensation loss + optional payload losses.

    Args:
        q_min: minimum charge for condensation (see OC paper).
        noise_threshold: ``object_id <= noise_threshold`` are treated as noise.
        max_n_rep: cap on repulsive pair sampling (0 = no cap).
        payload_weight: scalar multiplier on payload regression losses.
        torch_compile: let the OC package torch-compile its kernel.
    """

    def __init__(
        self,
        q_min: float = 1.0,
        noise_threshold: int = 0,
        max_n_rep: int = 0,
        payload_weight: float = 1.0,
        torch_compile: bool = False,
    ) -> None:
        super().__init__()
        self.q_min = q_min
        self.noise_threshold = noise_threshold
        self.max_n_rep = max_n_rep
        self.payload_weight = payload_weight
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
        total = sum(v for k, v in oc.items() if k != "n_rep")

        losses: dict[str, torch.Tensor] = {f"oc_{k}": v for k, v in oc.items()}
        losses["oc_total"] = total

        # TODO: weight payload losses by beta-per-truth-object; this is
        # a naive placeholder that treats non-noise hits equally.
        mask = targets["object_id"] > self.noise_threshold
        if mask.any() and self.payload_weight > 0:
            if "energy" in preds and "energy_per_hit" in targets:
                losses["energy"] = F.mse_loss(preds["energy"][mask], targets["energy_per_hit"][mask])
                total = total + self.payload_weight * losses["energy"]
            if "momentum" in preds and "momentum_per_hit" in targets:
                losses["momentum"] = F.mse_loss(
                    preds["momentum"][mask], targets["momentum_per_hit"][mask]
                )
                total = total + self.payload_weight * losses["momentum"]
            if "pid_logits" in preds and "pid_per_hit" in targets:
                losses["pid"] = F.cross_entropy(
                    preds["pid_logits"][mask], targets["pid_per_hit"][mask]
                )
                total = total + self.payload_weight * losses["pid"]

        losses["total"] = total
        return losses
