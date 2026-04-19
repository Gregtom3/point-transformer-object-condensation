"""Object Condensation prediction heads.

Per-hit outputs on top of the PTv3 decoder:
    * ``beta``          - condensation likelihood (sigmoid of logit).
    * ``x``             - cluster coordinates in a learned low-dim space.
    * ``pid_logits``    - per-class logits (shape class for the shapes
                           dataset, particle class for a physics detector).
    * optional regressors: ``width`` / ``height`` / ``energy`` /
      ``momentum`` depending on flags.
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
    def __init__(
        self,
        in_dim: int,
        cluster_dim: int = 4,
        hidden_dim: int = 128,
        n_pid_classes: int = 5,
        predict_width_height: bool = True,
        predict_energy: bool = False,
        predict_momentum: bool = False,
    ) -> None:
        super().__init__()
        self.cluster_dim = cluster_dim
        self.n_pid_classes = n_pid_classes
        self.predict_width_height = predict_width_height
        self.predict_energy = predict_energy
        self.predict_momentum = predict_momentum

        self.beta_head = MLP(in_dim, hidden_dim, 1)
        self.coord_head = MLP(in_dim, hidden_dim, cluster_dim)
        self.pid_head = MLP(in_dim, hidden_dim, n_pid_classes)

        if predict_width_height:
            self.width_head = MLP(in_dim, hidden_dim, 1)
            self.height_head = MLP(in_dim, hidden_dim, 1)
        if predict_energy:
            self.energy_head = MLP(in_dim, hidden_dim, 1)
        if predict_momentum:
            self.momentum_head = MLP(in_dim, hidden_dim, 3)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        beta_logit = self.beta_head(feat).squeeze(-1)
        out = {
            "beta": torch.sigmoid(beta_logit),
            "beta_logit": beta_logit,
            "x": self.coord_head(feat),
            "pid_logits": self.pid_head(feat),
        }
        if self.predict_width_height:
            out["width"] = self.width_head(feat).squeeze(-1)
            out["height"] = self.height_head(feat).squeeze(-1)
        if self.predict_energy:
            out["energy"] = self.energy_head(feat).squeeze(-1)
        if self.predict_momentum:
            out["momentum"] = self.momentum_head(feat)
        return out
