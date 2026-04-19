"""Object Condensation prediction heads.

Three per-hit heads are attached to the PTv3 backbone output:

* ``beta`` — condensation likelihood, squashed into (0, 1) with sigmoid.
* ``x``    — cluster coordinates in a learned low-dim space.
* payload — per-hit prediction of per-particle quantities (energy,
  momentum, PID logits). At inference these get aggregated around
  condensation points using the ``inference/cluster.py`` procedure.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObjectCondensationHeads(nn.Module):
    """Three-headed OC output module."""

    def __init__(
        self,
        in_dim: int,
        cluster_dim: int = 4,
        hidden_dim: int = 128,
        n_pid_classes: int = 5,
        predict_momentum: bool = True,
    ) -> None:
        super().__init__()
        self.cluster_dim = cluster_dim
        self.n_pid_classes = n_pid_classes
        self.predict_momentum = predict_momentum

        self.beta_head = MLP(in_dim, hidden_dim, 1)
        self.coord_head = MLP(in_dim, hidden_dim, cluster_dim)
        self.energy_head = MLP(in_dim, hidden_dim, 1)
        if predict_momentum:
            self.momentum_head = MLP(in_dim, hidden_dim, 3)
        self.pid_head = MLP(in_dim, hidden_dim, n_pid_classes)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        beta_logit = self.beta_head(feat).squeeze(-1)
        out = {
            "beta": torch.sigmoid(beta_logit),
            "beta_logit": beta_logit,
            "x": self.coord_head(feat),
            "energy": self.energy_head(feat).squeeze(-1),
            "pid_logits": self.pid_head(feat),
        }
        if self.predict_momentum:
            out["momentum"] = self.momentum_head(feat)
        return out
