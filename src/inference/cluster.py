"""Inference-time OC clustering.

Standard Object Condensation inference procedure (Kieseler 2020):

1. Order all hits by their learned beta.
2. Take the highest unassigned beta; if it is above ``t_beta`` it becomes
   a condensation point and all still-unassigned hits within euclidean
   radius ``t_d`` in cluster-coord space ``x`` are claimed as cluster
   members (regardless of their own beta).
3. Repeat; stop when the highest remaining unassigned beta falls below
   ``t_beta``. Anything still unassigned is noise (cluster id 0).

The defaults ``t_beta=0.5, t_d=0.28`` match the values used in the paper.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def beta_threshold_cluster(
    beta: torch.Tensor,
    x: torch.Tensor,
    t_beta: float = 0.5,
    t_d: float = 0.28,
) -> torch.Tensor:
    """Return a (N,) long tensor of cluster ids (0 = noise).

    Member eligibility: any hit not yet assigned AND within ``t_d`` of the
    current center, irrespective of the member's own beta. Only the cluster
    *center* requires ``beta > t_beta``.
    """
    n = beta.shape[0]
    device = beta.device
    cluster = torch.zeros(n, dtype=torch.long, device=device)
    assigned = torch.zeros(n, dtype=torch.bool, device=device)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=beta.dtype)
    next_id = 1
    while True:
        eligible_center = (~assigned) & (beta > t_beta)
        if not eligible_center.any():
            break
        scored = torch.where(eligible_center, beta, neg_inf)
        idx = int(torch.argmax(scored))
        center = x[idx]
        d = torch.linalg.norm(x - center, dim=-1)
        members = (~assigned) & (d < t_d)
        cluster[members] = next_id
        assigned = assigned | members
        next_id += 1
    return cluster
