"""Inference-time OC clustering.

Standard Object Condensation inference procedure (Kieseler 2020):

1. sort hits by beta (descending).
2. Walk the sorted list; the highest-beta hit above ``t_beta`` becomes a
   condensation point. Claim all unassigned hits within euclidean radius
   ``t_d`` in cluster-coord space ``x``.
3. Repeat until no hits with beta > t_beta remain. Unclaimed hits are
   considered noise.

This is the simple, per-event version; for large events a KDTree / torch
cdist chunking would be used in practice.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def beta_threshold_cluster(
    beta: torch.Tensor,
    x: torch.Tensor,
    t_beta: float = 0.1,
    t_d: float = 1.0,
) -> torch.Tensor:
    """Return a (N,) long tensor of cluster ids (0 = noise).

    TODO: vectorize / batch over events (this assumes a single event).
    """
    n = beta.shape[0]
    cluster = torch.zeros(n, dtype=torch.long, device=beta.device)
    unassigned = beta > t_beta
    next_id = 1
    while unassigned.any():
        idx = torch.argmax(beta * unassigned.float())
        if beta[idx] <= t_beta:
            break
        center = x[idx]
        d = torch.linalg.norm(x - center, dim=-1)
        members = (d < t_d) & unassigned
        cluster[members] = next_id
        unassigned = unassigned & ~members
        next_id += 1
    return cluster
