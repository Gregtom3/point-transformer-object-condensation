"""Concrete :class:`OCTask` for the shapes pseudo-dataset.

``ShapesTask`` owns both the HDF5 reader and the cv2-based rendering
for the three panels (TRUTH / PRED / OC). The OC panel is a real
scatter: each hit is placed at its (optionally PCA-reduced) cluster
coordinate on its own canvas, colored by the hit's input RGB feature
and alpha-blended by beta. A small inset of the shape image is pasted
in the corner of the OC panel for visual cross-reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from .base import (
    BG_COLOR,
    OCTask,
    PALETTE,
    UNCLAIMED_COLOR,
    UNMATCHED_COLOR,
    match_pred_to_truth,
)


class ShapesTask(OCTask):
    """HDF5-backed shapes task.

    Args:
        path: path to ``train.h5`` / ``val.h5`` / ``test.h5``.
        normalize_coords: divide (x, y) by frame size so coords live in
            [0, 1] (helps PTv3's voxelization with small grid_size).
        max_hits: optional per-event cap; ``0`` disables.
        panel_hw: (H, W) the three rendered panels should target. Set
            independently from the source frame so the OC scatter gets a
            roomy canvas regardless of the input resolution.
    """

    def __init__(
        self,
        path: str | Path,
        normalize_coords: bool = True,
        max_hits: int = 0,
        panel_hw: tuple[int, int] = (192, 192),
    ) -> None:
        self.path = Path(path)
        self.normalize_coords = normalize_coords
        self.max_hits = max_hits
        self.panel_hw = panel_hw

        with h5py.File(self.path, "r") as f:
            self.n_events = int(f["meta"].attrs["n_events"])
            self.frame = tuple(int(x) for x in f["meta"].attrs["frame"])
            self.n_shape_classes = int(f["meta"].attrs["n_shape_classes"])
            self.shape_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f["meta"].attrs["shape_names"]
            ]
            self.object_id_range = tuple(
                int(x) for x in f["meta"].attrs["object_id_range"]
            )
        self._h5: h5py.File | None = None

    # ----- data -----------------------------------------------------------

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return self.n_events

    def __repr__(self) -> str:
        return (
            f"ShapesTask(path={self.path}, n_events={self.n_events}, "
            f"frame={self.frame}, shapes={self.shape_names})"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = self._file()[f"event_{idx:06d}"]
        coord = np.asarray(g["coord"])
        feat = np.asarray(g["feat"])
        object_id = np.asarray(g["object_id"])
        shape_id = np.asarray(g["shape_id_per_hit"])
        width = np.asarray(g["width_per_hit"])
        height = np.asarray(g["height_per_hit"])

        if self.max_hits and coord.shape[0] > self.max_hits:
            sel = np.random.choice(coord.shape[0], size=self.max_hits, replace=False)
            sel.sort()
            coord = coord[sel]
            feat = feat[sel]
            object_id = object_id[sel]
            shape_id = shape_id[sel]
            width = width[sel]
            height = height[sel]

        if self.normalize_coords:
            fw, fh = self.frame
            coord = coord.copy()
            coord[:, 0] /= float(fw)
            coord[:, 1] /= float(fh)

        return {
            "coord": torch.from_numpy(coord).float(),
            "feat": torch.from_numpy(feat).float(),
            "object_id": torch.from_numpy(object_id).long(),
            "shape_id_per_hit": torch.from_numpy(shape_id).long(),
            "width_per_hit": torch.from_numpy(width).float(),
            "height_per_hit": torch.from_numpy(height).float(),
            "frame": torch.tensor(self.frame, dtype=torch.long),
        }

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys_flat = [
            "coord", "feat", "object_id", "shape_id_per_hit",
            "width_per_hit", "height_per_hit",
        ]
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

    def _slice_event(self, batch: dict[str, Any], start: int, end: int,
                     event_idx: int) -> dict[str, Any]:
        event = super()._slice_event(batch, start, end, event_idx)
        # per-event frame (events can in principle have different frame sizes)
        if "frame" in batch and torch.is_tensor(batch["frame"]):
            event["frame"] = batch["frame"][event_idx]
        return event

    # ----- rendering primitives ------------------------------------------

    def _event_pixel_xy(self, event: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
        """Return (N, 2) int pixel coords and the source (fw, fh)."""
        coord = event["coord"].detach().cpu().numpy()
        if "frame" in event:
            fw, fh = (int(v) for v in event["frame"].detach().cpu().numpy().tolist())
        else:
            fw, fh = self.frame
        xy = coord[:, :2].astype(np.float32)
        if xy.size and xy.max() <= 1.5:
            xy = xy.copy()
            xy[:, 0] *= fw
            xy[:, 1] *= fh
        xy = np.clip(xy.astype(np.int64), [[0, 0]], [[fw - 1, fh - 1]])
        return xy, (fw, fh)

    @staticmethod
    def _blank_canvas(size_hw: tuple[int, int]) -> np.ndarray:
        h, w = size_hw
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = BG_COLOR
        return canvas

    @staticmethod
    def _scale_xy(xy: np.ndarray, src: tuple[int, int],
                  dst: tuple[int, int]) -> np.ndarray:
        """Scale pixel coords from source (w, h) to destination (w, h)."""
        if xy.size == 0:
            return xy.astype(np.int32)
        sw, sh = src
        dw, dh = dst
        x = xy[:, 0].astype(np.float32) * ((dw - 1) / max(1, sw - 1))
        y = xy[:, 1].astype(np.float32) * ((dh - 1) / max(1, sh - 1))
        out = np.stack([x, y], axis=1)
        out = np.clip(out, [0, 0], [dw - 1, dh - 1])
        return out.astype(np.int32)

    def _paint_pixel_panel(
        self,
        event: dict[str, Any],
        size_hw: tuple[int, int],
        colors: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Shared body for TRUTH and PRED: paint one color per hit."""
        canvas = self._blank_canvas(size_hw)
        xy_src, (fw, fh) = self._event_pixel_xy(event)
        if xy_src.shape[0] == 0:
            return canvas
        h, w = size_hw
        xy = self._scale_xy(xy_src, (fw, fh), (w, h))
        if mask is None:
            mask = np.ones(xy.shape[0], dtype=bool)
        for (x, y), c, m in zip(xy, colors, mask):
            if not m:
                continue
            cv2.rectangle(canvas, (int(x), int(y)), (int(x), int(y)),
                          tuple(int(v) for v in c), thickness=-1)
        return canvas

    # ----- plot_truth / plot_pred / plot_oc ------------------------------

    def plot_truth(self, event: dict[str, Any],
                   size_hw: tuple[int, int]) -> np.ndarray:
        obj = event["object_id"].detach().cpu().numpy()
        colors = np.zeros((obj.shape[0], 3), dtype=np.uint8)
        mask = obj > 0
        if mask.any():
            ids = obj[mask] % PALETTE.shape[0]
            colors[mask] = PALETTE[ids]
        return self._paint_pixel_panel(event, size_hw, colors, mask)

    def plot_pred(self, event: dict[str, Any], pred_cluster: np.ndarray,
                  size_hw: tuple[int, int]) -> np.ndarray:
        truth = event["object_id"].detach().cpu().numpy()
        mapping = match_pred_to_truth(truth, pred_cluster)
        n = pred_cluster.shape[0]
        colors = np.zeros((n, 3), dtype=np.uint8)
        mask = np.zeros(n, dtype=bool)
        for i, p in enumerate(pred_cluster):
            p = int(p)
            if p <= 0:
                colors[i] = UNCLAIMED_COLOR  # light gray: dropped by OC inference
                mask[i] = True
                continue
            t = mapping.get(p, -1)
            colors[i] = PALETTE[t % PALETTE.shape[0]] if t > 0 else UNMATCHED_COLOR
            mask[i] = True
        return self._paint_pixel_panel(event, size_hw, colors, mask)

    def plot_oc(self, event: dict[str, Any], oc_xy: np.ndarray,
                beta: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        canvas = self._blank_canvas(size_hw)
        h, w = size_hw
        feat = event["feat"].detach().cpu().numpy()
        n = oc_xy.shape[0]
        if n > 0:
            margin = max(8, int(min(h, w) * 0.06))
            xmin = float(oc_xy[:, 0].min())
            ymin = float(oc_xy[:, 1].min())
            xmax = float(oc_xy[:, 0].max())
            ymax = float(oc_xy[:, 1].max())
            xr = max(xmax - xmin, 1e-6)
            yr = max(ymax - ymin, 1e-6)
            px = ((oc_xy[:, 0] - xmin) / xr) * (w - 2 * margin) + margin
            py = ((oc_xy[:, 1] - ymin) / yr) * (h - 2 * margin) + margin
            px = px.astype(np.int32)
            py = py.astype(np.int32)

            rgb = (np.clip(feat, 0.0, 1.0) * 255.0).astype(np.uint8)
            alpha = np.clip(beta, 0.0, 1.0).astype(np.float32)
            radius = max(2, int(min(h, w) * 0.012))

            # draw low-beta first so high-beta lands on top
            order = np.argsort(alpha, kind="stable")
            for i in order:
                cx, cy = int(px[i]), int(py[i])
                a = float(alpha[i])
                color = tuple(int(v) for v in rgb[i])
                x0 = max(0, cx - radius - 1)
                y0 = max(0, cy - radius - 1)
                x1 = min(w, cx + radius + 2)
                y1 = min(h, cy + radius + 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                region = canvas[y0:y1, x0:x1]
                overlay = region.copy()
                cv2.circle(overlay, (cx - x0, cy - y0), radius,
                           color, thickness=-1, lineType=cv2.LINE_AA)
                canvas[y0:y1, x0:x1] = cv2.addWeighted(overlay, a, region, 1 - a, 0)

        inset = self._paint_shape_inset(event, inset_frac=0.28, canvas_hw=(h, w))
        ih, iw = inset.shape[:2]
        pad = 2
        y0, y1 = pad, pad + ih
        x1 = w - pad
        x0 = x1 - iw
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1, y1), (0, 0, 0), 1)
        canvas[y0:y1, x0:x1] = inset
        return canvas

    def _paint_shape_inset(self, event: dict[str, Any], inset_frac: float,
                           canvas_hw: tuple[int, int]) -> np.ndarray:
        h, w = canvas_hw
        ih = max(16, int(h * inset_frac))
        iw = max(16, int(w * inset_frac))
        inset = self._blank_canvas((ih, iw))
        xy_src, (fw, fh) = self._event_pixel_xy(event)
        if xy_src.shape[0] == 0:
            return inset
        xy = self._scale_xy(xy_src, (fw, fh), (iw, ih))
        feat = event["feat"].detach().cpu().numpy()
        rgb = (np.clip(feat, 0.0, 1.0) * 255.0).astype(np.uint8)
        for (x, y), c in zip(xy, rgb):
            cv2.rectangle(inset, (int(x), int(y)), (int(x), int(y)),
                          tuple(int(v) for v in c), thickness=-1)
        return inset
