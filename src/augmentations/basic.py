"""Ready-to-use augmentations that work for any OC task with the
standard per-hit keys (``coord``, ``feat``, ``object_id``, and friends).

These are deliberately minimal reference implementations — good for the
shapes pseudo-dataset and as templates to copy into task-specific
augmentations (e.g. a detector-specific smear).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .base import Augmentation


class RandomRotation2D(Augmentation):
    """Rotate the (x, y) coordinates of each hit by a random angle.

    Intended for 2D image-like point clouds (``coord`` shape ``(N, 3)``
    with channel 2 unused or zero). Assumes ``coord`` is in [0, 1]-
    normalized units; the rotation is applied around (0.5, 0.5) so the
    canvas center is invariant. Hits that fall outside [0, 1] after
    rotation are clipped.

    Args:
        max_angle_deg: samples the rotation uniformly from
            ``[-max_angle_deg, max_angle_deg]``.
        seed: optional seed for reproducibility; when None uses a
            worker-local generator.
    """

    def __init__(self, max_angle_deg: float = 15.0, seed: int | None = None) -> None:
        self.max_angle_deg = float(max_angle_deg)
        self._rng = np.random.default_rng(seed)

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        coord = event["coord"]
        if coord.shape[0] == 0:
            return event
        ang = self._rng.uniform(-self.max_angle_deg, self.max_angle_deg)
        theta = math.radians(ang)
        c, s = math.cos(theta), math.sin(theta)

        xy = coord[:, :2] - 0.5
        rot = torch.tensor([[c, -s], [s, c]], dtype=xy.dtype)
        new_xy = xy @ rot.T + 0.5
        new_xy = new_xy.clamp_(0.0, 1.0)

        new_coord = coord.clone()
        new_coord[:, :2] = new_xy
        event = dict(event)
        event["coord"] = new_coord
        return event

    def __repr__(self) -> str:
        return f"RandomRotation2D(max_angle_deg={self.max_angle_deg})"


class RandomColorJitter(Augmentation):
    """Perturb per-hit RGB features.

    Adds i.i.d. Gaussian noise with standard deviation ``sigma`` to the
    ``feat`` tensor and clips back to ``[0, 1]``. A zero per-shape
    offset is sampled once per event so hits of the same shape drift
    together (helps the model see color-wiggled but still coherent
    shapes rather than noisy per-pixel speckle).

    Args:
        sigma: per-hit Gaussian noise std.
        shape_sigma: per-shape offset std. Requires ``object_id`` in the
            event (skipped if missing).
    """

    def __init__(self, sigma: float = 0.02, shape_sigma: float = 0.05,
                 seed: int | None = None) -> None:
        self.sigma = float(sigma)
        self.shape_sigma = float(shape_sigma)
        self._rng = np.random.default_rng(seed)

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        feat = event["feat"]
        if feat.shape[0] == 0:
            return event
        delta = torch.from_numpy(
            self._rng.normal(0.0, self.sigma, size=feat.shape).astype("float32")
        )
        new_feat = feat + delta

        if self.shape_sigma > 0 and "object_id" in event:
            obj = event["object_id"].detach().cpu().numpy()
            uniq = np.unique(obj)
            if uniq.size:
                shape_offsets = self._rng.normal(
                    0.0, self.shape_sigma, size=(uniq.size, feat.shape[1])
                ).astype("float32")
                id_to_off = {int(u): shape_offsets[i] for i, u in enumerate(uniq)}
                offs = np.stack([id_to_off[int(o)] for o in obj], axis=0)
                new_feat = new_feat + torch.from_numpy(offs)

        new_feat = new_feat.clamp_(0.0, 1.0)
        event = dict(event)
        event["feat"] = new_feat
        return event

    def __repr__(self) -> str:
        return (f"RandomColorJitter(sigma={self.sigma}, "
                f"shape_sigma={self.shape_sigma})")


class RandomHitDropout(Augmentation):
    """Drop a random fraction of hits per event.

    Every per-hit tensor in the event (any tensor whose first dim
    matches ``N``) is subselected with the same index mask. Non-hit
    tensors (``frame``, ``batch_size``) are left alone.

    Args:
        p: fraction of hits to drop (0 disables).
        min_keep: never drop below this many hits (keeps the event from
            collapsing to zero on tiny samples).
    """

    def __init__(self, p: float = 0.1, min_keep: int = 8,
                 seed: int | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = float(p)
        self.min_keep = int(min_keep)
        self._rng = np.random.default_rng(seed)

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.p == 0.0:
            return event
        n = int(event["coord"].shape[0])
        if n <= self.min_keep:
            return event
        keep_n = max(self.min_keep, int(round(n * (1.0 - self.p))))
        idx = self._rng.choice(n, size=keep_n, replace=False)
        idx.sort()
        idx_t = torch.as_tensor(idx, dtype=torch.long)

        new_event: dict[str, Any] = {}
        for k, v in event.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
                new_event[k] = v.index_select(0, idx_t)
            else:
                new_event[k] = v
        return new_event

    def __repr__(self) -> str:
        return f"RandomHitDropout(p={self.p}, min_keep={self.min_keep})"
