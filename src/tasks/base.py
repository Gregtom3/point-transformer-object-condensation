"""Base class for Object-Condensation tasks.

An :class:`OCTask` bundles the responsibilities that vary per data domain:

* iterating the dataset (``__len__`` / ``__getitem__`` / ``__repr__``),
* collating per-event dicts into PTv3's offset-batched layout
  (``collate``),
* rendering a single event's truth / prediction / OC-space panels
  (``plot_truth`` / ``plot_pred`` / ``plot_oc``).

The base class owns the parts that generalize across tasks: a shared
palette, greedy majority-vote matching of predicted clusters to truth
ids, PCA-to-2D for high-dim cluster coordinates, label strips, and grid
composition of per-event cells. Subclasses only implement the methods
marked ``@abstractmethod``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


# Shared color conventions so matched clusters look identical across panels.
PALETTE = np.random.default_rng(42).integers(40, 230, size=(2048, 3), dtype=np.uint8)
UNMATCHED_COLOR = np.array([160, 160, 160], dtype=np.uint8)  # pred has no truth overlap
UNCLAIMED_COLOR = np.array([220, 220, 220], dtype=np.uint8)  # OC did not claim this hit
BG_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def match_pred_to_truth(truth: np.ndarray, pred: np.ndarray) -> dict[int, int]:
    """Greedy majority-vote mapping from predicted cluster id to truth id.

    For each predicted cluster `p > 0` pick the truth id with the most
    shared hits. Returns `{p: best_truth_id}` (or `-1` if none).
    """
    mapping: dict[int, int] = {}
    for p in np.unique(pred):
        if p <= 0:
            continue
        overlap = truth[pred == p]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            mapping[int(p)] = -1
            continue
        ids, counts = np.unique(overlap, return_counts=True)
        mapping[int(p)] = int(ids[counts.argmax()])
    return mapping


def pca_to_2d(x: np.ndarray) -> np.ndarray:
    """Linear projection of ``x`` (N, D) to (N, 2) via centered SVD.

    ``D == 1`` is zero-padded; ``D == 2`` is returned as-is so axes stay
    stable across steps (PCA on 2D would only rotate).
    """
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[1] == 1:
        return np.concatenate([x, np.zeros_like(x)], axis=1).astype(np.float32)
    if x.shape[1] == 2:
        return x.astype(np.float32)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return (xc @ vt.T[:, :2]).astype(np.float32)


def umap_to_2d(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Non-linear projection of ``x`` (N, D) to (N, 2) via UMAP.

    Falls back to :func:`pca_to_2d` when D ≤ 2 (nothing to reduce), when
    there are too few points for UMAP to be meaningful, or when
    ``umap-learn`` isn't installed — so enabling UMAP in the config can't
    crash a run that otherwise works.
    """
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[1] <= 2 or x.shape[0] < 10:
        return pca_to_2d(x)
    try:
        import umap  # type: ignore
    except ImportError:
        return pca_to_2d(x)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, x.shape[0] - 1)),
        min_dist=0.1,
        random_state=random_state,
        # UMAP warns on tiny datasets; silence for our per-event render.
        verbose=False,
    )
    try:
        return reducer.fit_transform(x).astype(np.float32)
    except Exception:
        return pca_to_2d(x)


def project_2d(x: np.ndarray, mode: str = "pca") -> np.ndarray:
    """Dispatch 2D projection by string name ("pca" | "umap")."""
    if mode == "umap":
        return umap_to_2d(x)
    return pca_to_2d(x)


# ---------------------------------------------------------------------------
# Label strip helpers (PIL text rendering over an RGB bar).
# ---------------------------------------------------------------------------

def _make_label_strip(width: int, labels: list[tuple[int, str]],
                      bar_px: int = 14) -> np.ndarray:
    strip = np.full((bar_px, width, 3), 30, dtype=np.uint8)
    pil = Image.fromarray(strip)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for x, text in labels:
        draw.text((x, 1), text, fill=(240, 240, 240), font=font)
    return np.array(pil)


def _upscale_nn(img_hw3: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img_hw3
    return np.repeat(np.repeat(img_hw3, factor, axis=0), factor, axis=1)


# ---------------------------------------------------------------------------
# OCTask
# ---------------------------------------------------------------------------

class OCTask(Dataset, ABC):
    """Abstract OC task: data iteration + per-event rendering.

    ``panel_hw`` is the default (height, width) the base class asks each
    ``plot_*`` method to produce. Subclasses may use it as a guideline or
    return a different-sized image (the grid composer pads to the max).

    ``projection`` chooses how >2D cluster coordinates are reduced to 2D
    for the OC panel: "pca" (fast, linear, stable axes across steps) or
    "umap" (non-linear; better for separated clusters when cluster_dim is
    large). With ``cluster_dim <= 2`` both modes are identity.
    """

    panel_hw: tuple[int, int] = (128, 128)
    projection: str = "pca"

    # ----- data iteration -------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate events into PTv3's flat ``coord``/``feat``/``offset`` layout."""

    # ----- rendering ------------------------------------------------------

    @abstractmethod
    def plot_truth(self, event: dict[str, Any],
                   size_hw: tuple[int, int]) -> np.ndarray:
        """Return an (H, W, 3) uint8 image of the truth panel."""

    @abstractmethod
    def plot_pred(self, event: dict[str, Any], pred_cluster: np.ndarray,
                  size_hw: tuple[int, int]) -> np.ndarray:
        """Return an (H, W, 3) uint8 image of the predicted-cluster panel.

        Predicted cluster colors should be matched to truth object colors
        so a correctly reconstructed object keeps the same color across
        TRUTH and PRED panels.
        """

    @abstractmethod
    def plot_oc(self, event: dict[str, Any], oc_xy: np.ndarray,
                beta: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Return an (H, W, 3) uint8 image of the OC-space panel."""

    # ----- multi-event composition ---------------------------------------

    def render_cell(
        self,
        event: dict[str, Any],
        pred_cluster: np.ndarray,
        oc_xy: np.ndarray,
        beta: np.ndarray,
        event_idx: int,
        upscale: int = 1,
    ) -> np.ndarray:
        """Compose the three panels for one event into a single cell."""
        size_hw = self.panel_hw
        truth_img = self.plot_truth(event, size_hw)
        pred_img = self.plot_pred(event, pred_cluster, size_hw)
        oc_img = self.plot_oc(event, oc_xy, beta, size_hw)

        fh, fw = size_hw
        pad_px = 2
        v_pad = np.full((fh, pad_px, 3), 0, dtype=np.uint8)
        row = np.concatenate([truth_img, v_pad, pred_img, v_pad, oc_img], axis=1)
        total_w = row.shape[1]

        n_truth = int(np.sum(np.unique(event["object_id"].detach().cpu().numpy()) > 0))
        mapping = match_pred_to_truth(
            event["object_id"].detach().cpu().numpy(), pred_cluster,
        )
        n_pred = int(np.sum(np.unique(pred_cluster) > 0))
        matched = sum(1 for v in mapping.values() if v > 0)

        x_truth = 2
        x_pred = fw + pad_px + 2
        x_oc = 2 * (fw + pad_px) + 2
        sub_strip = _make_label_strip(
            total_w,
            [(x_truth, "TRUTH"), (x_pred, "PRED"), (x_oc, "OC")],
            bar_px=12,
        )
        header = f"event {event_idx}   truth n={n_truth}   pred k={n_pred}   matched={matched}"
        top_strip = _make_label_strip(total_w, [(3, header)], bar_px=14)

        cell = np.concatenate([top_strip, sub_strip, row], axis=0)
        return _upscale_nn(cell, upscale)

    def render_grid(
        self,
        batch: dict[str, Any],
        preds: dict[str, torch.Tensor],
        pred_cluster: torch.Tensor,
        grid: tuple[int, int] = (4, 4),
        upscale: int = 3,
        sep_px: int = 4,
    ) -> np.ndarray:
        """Return a (3, H, W) uint8 grid image for ``writer.add_image``.

        The batch is split back into per-event dicts using ``batch['offset']``
        and each event is rendered via :meth:`render_cell`.
        """
        offsets = batch["offset"].detach().cpu().numpy().tolist()
        pred_np = pred_cluster.detach().cpu().numpy()
        beta_np = preds["beta"].detach().cpu().numpy()
        oc_raw = preds["x"].detach().cpu().numpy()

        rows, cols = grid
        n_wanted = rows * cols
        n_events = min(len(offsets), n_wanted)
        starts = [0] + offsets[:-1]

        cells: list[np.ndarray] = []
        for i in range(n_events):
            s, e = starts[i], offsets[i]
            event = self._slice_event(batch, s, e, event_idx=i)
            # Project cluster coords per event so each OC panel shows its
            # own embedding structure (shared axes across events aren't
            # meaningful once we consider non-linear modes like UMAP).
            oc_xy_event = project_2d(oc_raw[s:e], mode=self.projection)
            cell = self.render_cell(
                event,
                pred_cluster=pred_np[s:e],
                oc_xy=oc_xy_event,
                beta=beta_np[s:e],
                event_idx=i,
                upscale=upscale,
            )
            cells.append(cell)

        if not cells:
            return np.zeros((3, 32, 32), dtype=np.uint8)

        max_h = max(c.shape[0] for c in cells)
        max_w = max(c.shape[1] for c in cells)
        padded: list[np.ndarray] = []
        for c in cells:
            pad_h = max_h - c.shape[0]
            pad_w = max_w - c.shape[1]
            if pad_h or pad_w:
                c = np.pad(c, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
            padded.append(c)
        while len(padded) < n_wanted:
            padded.append(np.full_like(padded[0], 255))

        v_sep = np.full((max_h, sep_px, 3), 0, dtype=np.uint8)
        h_sep_width = cols * max_w + (cols - 1) * sep_px
        h_sep = np.full((sep_px, h_sep_width, 3), 0, dtype=np.uint8)

        row_imgs: list[np.ndarray] = []
        for r in range(rows):
            row_cells = padded[r * cols : (r + 1) * cols]
            parts: list[np.ndarray] = []
            for i, cell in enumerate(row_cells):
                parts.append(cell)
                if i < cols - 1:
                    parts.append(v_sep)
            row_imgs.append(np.concatenate(parts, axis=1))

        parts: list[np.ndarray] = []
        for i, row_img in enumerate(row_imgs):
            parts.append(row_img)
            if i < rows - 1:
                parts.append(h_sep)
        grid_hw3 = np.concatenate(parts, axis=0)
        return np.transpose(grid_hw3, (2, 0, 1))

    # ----- helpers subclasses can override -------------------------------

    def _slice_event(self, batch: dict[str, Any], start: int, end: int,
                     event_idx: int) -> dict[str, Any]:
        """Per-event slice of a batch. Default handles the common keys;
        subclasses override when they need additional per-event state
        (e.g. per-event frame sizes)."""
        event: dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == batch["offset"][-1]:
                event[k] = v[start:end]
        return event
