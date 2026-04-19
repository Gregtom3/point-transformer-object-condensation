"""CalorimeterTask — an :class:`OCTask` for the toy left/right calorimeter.

Key differences vs. :class:`ShapesTask` (a good way to see what's
task-specific vs. domain-specific):

* ``coord`` is truly 3D (``x = ±wall_x``), not a 2D image with z=0.
* ``feat`` is physics-ish: ``(log10(E_cell), t, subdet_id)`` with
  subdet_id ∈ {0, 1} marking left vs right wall.
* Rendering: truth / pred panels plot the two walls side-by-side as
  (y, z) scatter panels since an event is naturally two 2D planes,
  not one.
* Payload target: a single per-hit regression (``energy_per_hit``),
  not shape-id / width / height.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from src.augmentations.base import Augmentation
from src.tasks.base import (
    BG_COLOR,
    OCTask,
    UNCLAIMED_COLOR,
    UNMATCHED_COLOR,
    match_pred_to_truth,
)


class CalorimeterTask(OCTask):
    """HDF5-backed toy left/right calorimeter task."""

    def __init__(
        self,
        path: str | Path,
        normalize_coords: bool = True,
        max_hits: int = 0,
        panel_hw: tuple[int, int] = (192, 192),
        projection: str = "pca",
        augmentation: Augmentation | None = None,
    ) -> None:
        self.path = Path(path)
        self.normalize_coords = normalize_coords
        self.max_hits = max_hits
        self.panel_hw = panel_hw
        if projection not in ("pca", "umap"):
            raise ValueError(f"projection must be 'pca' or 'umap', got {projection!r}")
        self.projection = projection
        self.augmentation = augmentation

        with h5py.File(self.path, "r") as f:
            self.n_events = int(f["meta"].attrs["n_events"])
            self.wall_x = float(f["meta"].attrs["wall_x"])
            self.wall_yz_half = float(f["meta"].attrs["wall_yz_half"])
            self.frame = tuple(int(x) for x in f["meta"].attrs["frame"])
        self._h5: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    # ----- data ----------------------------------------------------------

    def __len__(self) -> int:
        return self.n_events

    def __repr__(self) -> str:
        return (f"CalorimeterTask(path={self.path}, n_events={self.n_events}, "
                f"wall_x=±{self.wall_x}, yz_half={self.wall_yz_half})")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self._file()[f"event_{idx:06d}"]
        coord = np.asarray(g["coord"])
        feat = np.asarray(g["feat"])
        object_id = np.asarray(g["object_id"])
        energy = np.asarray(g["energy_per_hit"])

        if self.max_hits and coord.shape[0] > self.max_hits:
            sel = np.random.choice(coord.shape[0], size=self.max_hits, replace=False)
            sel.sort()
            coord, feat, object_id, energy = (a[sel] for a in (coord, feat, object_id, energy))

        if self.normalize_coords:
            # map (x, y, z) into [0, 1] — PTv3's sparse voxelizer expects a
            # bounded, positive-quadrant grid.
            coord = coord.copy()
            coord[:, 0] = (coord[:, 0] + self.wall_x) / (2.0 * self.wall_x)
            coord[:, 1] = (coord[:, 1] + self.wall_yz_half) / (2.0 * self.wall_yz_half)
            coord[:, 2] = (coord[:, 2] + self.wall_yz_half) / (2.0 * self.wall_yz_half)

        event = {
            "coord": torch.from_numpy(coord).float(),
            "feat": torch.from_numpy(feat).float(),
            "object_id": torch.from_numpy(object_id).long(),
            "energy_per_hit": torch.from_numpy(energy).float(),
            "frame": torch.tensor(self.frame, dtype=torch.long),
        }
        if self.augmentation is not None:
            event = self.augmentation(event)
        return event

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys_flat = ["coord", "feat", "object_id", "energy_per_hit"]
        out: dict[str, Any] = {k: [] for k in keys_flat}
        sizes: list[int] = []
        frames: list[torch.Tensor] = []
        for item in batch:
            for k in keys_flat:
                out[k].append(item[k])
            sizes.append(item["coord"].shape[0])
            frames.append(item["frame"])
        for k in keys_flat:
            out[k] = torch.cat(out[k], dim=0)
        out["offset"] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)
        out["frame"] = torch.stack(frames, dim=0)
        out["batch_size"] = len(batch)
        return out

    # ----- rendering primitives -----------------------------------------

    def _event_yz(self, event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return (yz pixel coords, side_mask) where side_mask is 0 for left,
        1 for right. ``coord`` is expected in [0, 1]."""
        coord = event["coord"].detach().cpu().numpy()
        if coord.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        side = (x > 0.5).astype(np.int64)  # 0 = left, 1 = right
        fh, fw = self.panel_hw
        # each side panel is half the width
        half_w = fw // 2
        py = np.clip((z * (fh - 1)).astype(np.int32), 0, fh - 1)
        px_side = np.clip((y * (half_w - 1)).astype(np.int32), 0, half_w - 1)
        px = np.where(side == 0, px_side, px_side + half_w)
        return np.stack([px, py], axis=1).astype(np.int64), side

    def _paint_detector_panel(
        self,
        event: dict[str, Any],
        colors: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Left wall | Right wall side-by-side scatter; z on y-axis, y on x-axis."""
        fh, fw = self.panel_hw
        canvas = np.empty((fh, fw, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        # vertical divider between the two walls
        mid = fw // 2
        cv2.line(canvas, (mid, 0), (mid, fh - 1), (180, 180, 180), 1)
        cv2.putText(canvas, "L", (4, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
        cv2.putText(canvas, "R", (mid + 4, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        xy, _side = self._event_yz(event)
        if xy.shape[0] == 0:
            return canvas
        if mask is None:
            mask = np.ones(xy.shape[0], dtype=bool)
        radius = 2
        for (x, y), c, m in zip(xy, colors, mask):
            if not m:
                continue
            cv2.circle(canvas, (int(x), int(y)), radius,
                       tuple(int(v) for v in c), thickness=-1, lineType=cv2.LINE_AA)
        return canvas

    def _id_palette(self, ids: np.ndarray) -> dict[int, np.ndarray]:
        """Assign a stable per-event color to each truth object id using a
        lightweight hash so truth / pred panels can be visually matched
        without relying on feat (this task's feat is physics-y, not a color)."""
        uniq = [int(i) for i in np.unique(ids) if i > 0]
        rng = np.random.default_rng(0xC01DBEEF)
        base = rng.integers(40, 230, size=(len(uniq), 3), dtype=np.uint8)
        # jitter with each id so nearby ids don't collide visually
        return {oid: base[i] for i, oid in enumerate(uniq)}

    # ----- plot_truth / plot_pred / plot_oc -----------------------------

    def plot_truth(self, event: dict[str, Any],
                   size_hw: tuple[int, int]) -> np.ndarray:
        self.panel_hw = size_hw
        obj = event["object_id"].detach().cpu().numpy()
        id_to_color = self._id_palette(obj)
        colors = np.zeros((obj.shape[0], 3), dtype=np.uint8)
        mask = obj > 0
        for i, o in enumerate(obj):
            if o > 0:
                colors[i] = id_to_color[int(o)]
        return self._paint_detector_panel(event, colors, mask)

    def plot_pred(self, event: dict[str, Any], pred_cluster: np.ndarray,
                  size_hw: tuple[int, int]) -> np.ndarray:
        self.panel_hw = size_hw
        truth = event["object_id"].detach().cpu().numpy()
        id_to_color = self._id_palette(truth)
        mapping = match_pred_to_truth(truth, pred_cluster)
        n = pred_cluster.shape[0]
        colors = np.zeros((n, 3), dtype=np.uint8)
        mask = np.zeros(n, dtype=bool)
        for i, p in enumerate(pred_cluster):
            p = int(p)
            if p <= 0:
                colors[i] = UNCLAIMED_COLOR
                mask[i] = True
                continue
            t = mapping.get(p, -1)
            if t > 0 and t in id_to_color:
                colors[i] = id_to_color[t]
            else:
                colors[i] = UNMATCHED_COLOR
            mask[i] = True
        return self._paint_detector_panel(event, colors, mask)

    def plot_oc(self, event: dict[str, Any], oc_xy: np.ndarray,
                beta: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Standard OC scatter — colored by per-hit id-palette (since this
        task's feat isn't a color). Alpha by β."""
        h, w = size_hw
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        n = oc_xy.shape[0]
        if n == 0:
            return canvas

        obj = event["object_id"].detach().cpu().numpy()
        id_to_color = self._id_palette(obj)
        rgb = np.full((n, 3), UNCLAIMED_COLOR, dtype=np.uint8)
        for i, o in enumerate(obj):
            if o > 0:
                rgb[i] = id_to_color[int(o)]

        margin = max(8, int(min(h, w) * 0.06))
        xmin, ymin = oc_xy.min(axis=0)
        xmax, ymax = oc_xy.max(axis=0)
        xr = max(float(xmax - xmin), 1e-6)
        yr = max(float(ymax - ymin), 1e-6)
        px = (((oc_xy[:, 0] - xmin) / xr) * (w - 2 * margin) + margin).astype(np.int32)
        py = (((oc_xy[:, 1] - ymin) / yr) * (h - 2 * margin) + margin).astype(np.int32)

        alpha = np.clip(beta, 0.0, 1.0).astype(np.float32)
        radius = max(2, int(min(h, w) * 0.012))
        order = np.argsort(alpha, kind="stable")
        for i in order:
            cx, cy = int(px[i]), int(py[i])
            a = float(alpha[i])
            color = tuple(int(v) for v in rgb[i])
            x0 = max(0, cx - radius - 1); y0 = max(0, cy - radius - 1)
            x1 = min(w, cx + radius + 2); y1 = min(h, cy + radius + 2)
            region = canvas[y0:y1, x0:x1]
            overlay = region.copy()
            cv2.circle(overlay, (cx - x0, cy - y0), radius,
                       color, thickness=-1, lineType=cv2.LINE_AA)
            canvas[y0:y1, x0:x1] = cv2.addWeighted(overlay, a, region, 1 - a, 0)
        return canvas
