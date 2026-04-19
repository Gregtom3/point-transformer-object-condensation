"""TensorBoard logging helpers for OC training runs."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


def log_scalars(writer: "SummaryWriter", tag_prefix: str,
                scalars: dict[str, float | torch.Tensor], step: int) -> None:
    for k, v in scalars.items():
        if torch.is_tensor(v):
            v = float(v.detach())
        writer.add_scalar(f"{tag_prefix}/{k}", v, global_step=step)


@torch.no_grad()
def log_oc_embedding(
    writer: "SummaryWriter",
    tag: str,
    x: torch.Tensor,
    object_id: torch.Tensor,
    step: int,
    max_points: int = 4096,
) -> None:
    """Log the learned cluster-coord embedding to TB's projector.

    ``x`` is (N, D). We log each hit's coord labeled by its object_id so
    the projector's "color by label" shows condensation structure.
    """
    n = x.shape[0]
    if n > max_points:
        sel = torch.randperm(n, device=x.device)[:max_points]
        x = x[sel]
        object_id = object_id[sel]
    labels = [str(int(oid)) for oid in object_id.detach().cpu().tolist()]
    writer.add_embedding(
        mat=x.detach().cpu(),
        metadata=labels,
        tag=tag,
        global_step=step,
    )


@torch.no_grad()
def log_prediction_canvas(
    writer: "SummaryWriter",
    tag: str,
    coord: torch.Tensor,
    object_id: torch.Tensor,
    pred_cluster: torch.Tensor,
    frame: tuple[int, int],
    step: int,
) -> None:
    """Paint truth vs. predicted cluster assignments on a white canvas.

    ``coord`` is (N, 3); the first two dims are (x, y) in [0, 1] if
    coords were normalized by ShapeDataset (otherwise pixel coords).
    ``object_id`` is truth, ``pred_cluster`` is the OC-clustered label.
    """
    fw, fh = frame
    truth = _paint(coord, object_id, fw, fh)
    pred = _paint(coord, pred_cluster, fw, fh)
    # TB expects (N, 3, H, W) or (3, H, W)
    img = np.concatenate([truth, pred], axis=2)  # side-by-side along width
    writer.add_image(tag, img, global_step=step, dataformats="CHW")


def _paint(coord: torch.Tensor, labels: torch.Tensor, fw: int, fh: int) -> np.ndarray:
    xy = coord[:, :2].detach().cpu().numpy()
    if xy.max() <= 1.5:  # normalized
        xy = xy.copy()
        xy[:, 0] *= fw
        xy[:, 1] *= fh
    xy = np.clip(xy.astype(np.int64), [[0, 0]], [[fw - 1, fh - 1]])
    canvas = np.full((3, fh, fw), 255, dtype=np.uint8)
    lab = labels.detach().cpu().numpy()
    # deterministic color table
    rng = np.random.default_rng(0)
    palette = rng.integers(30, 230, size=(2048, 3), dtype=np.uint8)
    for (x, y), l in zip(xy, lab):
        if l <= 0:
            continue
        c = palette[int(l) % palette.shape[0]]
        canvas[:, int(y), int(x)] = c
    return canvas
