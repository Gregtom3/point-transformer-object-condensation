"""TensorBoard helpers for OC training runs.

Three kinds of artefacts are logged here:

1. Scalars (via :func:`log_scalars`) — straight key/value pairs.
2. Prediction grid (via :func:`log_prediction_grid`) — a 4x4 layout of
   events, each cell stacking three panels with text labels:

       +-------------------------+
       | TRUTH    (object_id)    |
       +-------------------------+
       | PRED     (matched)      |
       +-------------------------+
       | OC       (x, alpha=beta)|
       +-------------------------+

   Predicted-cluster colors are matched to truth object colors via
   majority-vote assignment, so the same shape keeps the same color
   across panels when the model gets it right. The OC panel scatters
   per-hit cluster coordinates (PCA-reduced to 2D if
   ``cluster_dim > 2``), colored by each hit's input RGB feature and
   alpha-blended by beta (higher beta → more opaque). A small inset of
   the original shape image is pasted in the corner for reference.

3. Cluster-coord projector (via :func:`log_oc_embedding`) — raw OC
   coordinates per hit, labeled by truth object id.

A one-shot :func:`log_run_description` dumps a markdown block to the
Text tab so a reader landing on the run knows what every panel means.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


# A deterministic, high-contrast palette of 2048 colors. Truth object ids
# and (matched) predicted cluster ids both index into this same array.
_PALETTE = np.random.default_rng(42).integers(40, 230, size=(2048, 3), dtype=np.uint8)
_UNMATCHED_COLOR = np.array([160, 160, 160], dtype=np.uint8)  # gray for FPs
_BG_COLOR = np.array([255, 255, 255], dtype=np.uint8)         # white canvas


def log_scalars(writer: "SummaryWriter", tag_prefix: str,
                scalars: dict[str, float | torch.Tensor], step: int) -> None:
    for k, v in scalars.items():
        if torch.is_tensor(v):
            v = float(v.detach())
        writer.add_scalar(f"{tag_prefix}/{k}", v, global_step=step)


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------


def _match_pred_to_truth(truth: np.ndarray, pred: np.ndarray) -> dict[int, int]:
    """Greedy majority-vote mapping from predicted cluster id to truth id.

    For each predicted cluster `p > 0`, find the truth id that has the
    most hits shared with `p`. Returns a dict `{p: best_truth_id}`; if a
    predicted cluster has no truth overlap, it's mapped to `-1`.
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


def _paint_cluster_canvas(
    coord_xy: np.ndarray,
    labels: np.ndarray,
    fw: int,
    fh: int,
    label_to_color: dict[int, np.ndarray] | None = None,
    unclaimed_color: np.ndarray | None = None,
) -> np.ndarray:
    """Return an (H, W, 3) uint8 canvas. ``coord_xy`` is (N, 2) int pixel coords.

    ``unclaimed_color`` (e.g. light gray) is used for hits with ``label <= 0``
    so dropped-by-OC hits are distinguishable from "no hit here" white.
    """
    canvas = np.broadcast_to(_BG_COLOR, (fh, fw, 3)).copy()
    for (x, y), l in zip(coord_xy, labels):
        il = int(l)
        if il <= 0:
            if unclaimed_color is not None:
                canvas[int(y), int(x)] = unclaimed_color
            continue
        if label_to_color is not None and il in label_to_color:
            color = label_to_color[il]
        else:
            color = _PALETTE[il % _PALETTE.shape[0]]
        canvas[int(y), int(x)] = color
    return canvas


def _pca_to_2d(x: np.ndarray) -> np.ndarray:
    """Linear projection of ``x`` (N, D) to (N, 2) via centered SVD.

    D == 1 is zero-padded; D == 2 is returned as-is (PCA is identity
    up to rotation and we want the axes stable across steps).
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


def _paint_shape_inset(
    coord_xy: np.ndarray,
    feat_rgb: np.ndarray,
    src_fw: int,
    src_fh: int,
    inset_w: int,
    inset_h: int,
) -> np.ndarray:
    """Scaled-down copy of the raw shape image (points colored by feat)."""
    inset = np.broadcast_to(_BG_COLOR, (inset_h, inset_w, 3)).copy()
    if coord_xy.shape[0] == 0:
        return inset
    sx = (inset_w - 1) / max(1, src_fw - 1)
    sy = (inset_h - 1) / max(1, src_fh - 1)
    rgb = (np.clip(feat_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    ix = np.clip((coord_xy[:, 0] * sx).astype(np.int64), 0, inset_w - 1)
    iy = np.clip((coord_xy[:, 1] * sy).astype(np.int64), 0, inset_h - 1)
    inset[iy, ix] = rgb
    return inset


def _paint_oc_scatter(
    oc_xy: np.ndarray,
    feat_rgb: np.ndarray,
    beta: np.ndarray,
    fw: int,
    fh: int,
    image_xy_px: np.ndarray,
    margin_px: int = 6,
    dot_size: int = 2,
    inset_frac: float = 0.28,
) -> np.ndarray:
    """Scatter of (PCA-reduced) OC coords, colored by feat, alpha by beta.

    Points with higher ``beta`` are rendered more opaque against the
    white canvas; points near beta ≈ 0 fade into the background.
    A small inset of the original shape image is pasted top-right for
    visual cross-reference.
    """
    canvas = np.broadcast_to(_BG_COLOR, (fh, fw, 3)).copy().astype(np.float32)
    if oc_xy.shape[0] > 0:
        xmin, ymin = oc_xy.min(axis=0)
        xmax, ymax = oc_xy.max(axis=0)
        xr = max(float(xmax - xmin), 1e-6)
        yr = max(float(ymax - ymin), 1e-6)
        px = ((oc_xy[:, 0] - xmin) / xr) * (fw - 2 * margin_px) + margin_px
        py = ((oc_xy[:, 1] - ymin) / yr) * (fh - 2 * margin_px) + margin_px

        rgb = np.clip(feat_rgb, 0.0, 1.0) * 255.0
        alpha = np.clip(beta, 0.0, 1.0)[:, None]
        blended = alpha * rgb + (1.0 - alpha) * 255.0

        half = dot_size // 2
        for (cx, cy), c in zip(zip(px, py), blended):
            x0 = max(0, int(cx) - half)
            y0 = max(0, int(cy) - half)
            x1 = min(fw, x0 + dot_size)
            y1 = min(fh, y0 + dot_size)
            canvas[y0:y1, x0:x1] = c

    canvas = canvas.astype(np.uint8)

    # Inset: top-right corner with 1px black border.
    inset_h = max(8, int(fh * inset_frac))
    inset_w = max(8, int(fw * inset_frac))
    inset = _paint_shape_inset(image_xy_px, feat_rgb, fw, fh, inset_w, inset_h)
    pad = 1
    y0 = pad
    y1 = y0 + inset_h
    x1 = fw - pad
    x0 = x1 - inset_w
    canvas[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = 0  # border
    canvas[y0:y1, x0:x1] = inset
    return canvas


def _make_label_strip(width: int, labels: list[tuple[int, str]], bar_px: int = 14) -> np.ndarray:
    """Return a (bar_px, width, 3) strip with ``labels`` drawn at given x offsets."""
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


def _add_label_strip(img_hw3: np.ndarray, text: str, bar_px: int = 14) -> np.ndarray:
    strip = _make_label_strip(img_hw3.shape[1], [(3, text)], bar_px=bar_px)
    return np.concatenate([strip, img_hw3], axis=0)


def _upscale_nn(img_hw3: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img_hw3
    return np.repeat(np.repeat(img_hw3, factor, axis=0), factor, axis=1)


def _denormalize_coords(coord: np.ndarray, fw: int, fh: int) -> np.ndarray:
    xy = coord[:, :2].astype(np.float32)
    if xy.max() <= 1.5:
        xy = xy.copy()
        xy[:, 0] *= fw
        xy[:, 1] *= fh
    xy = np.clip(xy.astype(np.int64), [[0, 0]], [[fw - 1, fh - 1]])
    return xy


def _render_one_event(
    coord: np.ndarray,
    feat: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    beta: np.ndarray,
    oc_x: np.ndarray,
    frame: tuple[int, int],
    event_idx: int,
    upscale: int,
) -> np.ndarray:
    fw, fh = frame
    xy = _denormalize_coords(coord, fw, fh)
    mapping = _match_pred_to_truth(truth, pred)

    pred_color_table = {
        p: (_PALETTE[m % _PALETTE.shape[0]] if m > 0 else _UNMATCHED_COLOR)
        for p, m in mapping.items()
    }

    # Light-gray for hits the OC inference didn't claim — makes "model
    # dropped these as noise" visually distinct from "no hit at this pixel".
    unclaimed = np.array([220, 220, 220], dtype=np.uint8)
    truth_img = _paint_cluster_canvas(xy, truth, fw, fh)
    pred_img = _paint_cluster_canvas(
        xy, pred, fw, fh,
        label_to_color=pred_color_table,
        unclaimed_color=unclaimed,
    )
    oc_xy = _pca_to_2d(oc_x)
    oc_img = _paint_oc_scatter(oc_xy, feat, beta, fw, fh, xy)

    # optional per-panel labels, then upscale the whole cell together so
    # labels scale with the image.
    n_truth = int(np.sum(np.unique(truth) > 0))
    n_pred = int(np.sum(np.unique(pred) > 0))
    matched = sum(1 for v in mapping.values() if v > 0)

    pad_px = 2
    v_pad = np.full((fh, pad_px, 3), 0, dtype=np.uint8)
    row = np.concatenate([truth_img, v_pad, pred_img, v_pad, oc_img], axis=1)
    total_w = row.shape[1]

    # Sub-header: one label per panel, positioned at each panel's left edge.
    x_truth = 2
    x_pred = fw + pad_px + 2
    x_oc = 2 * (fw + pad_px) + 2
    sub_labels = [(x_truth, "TRUTH"), (x_pred, "PRED"), (x_oc, "OC")]
    sub_strip = _make_label_strip(total_w, sub_labels, bar_px=12)

    # Top header spans the whole cell.
    header = f"event {event_idx}   truth n={n_truth}   pred k={n_pred}   matched={matched}"
    top_strip = _make_label_strip(total_w, [(3, header)], bar_px=14)

    cell = np.concatenate([top_strip, sub_strip, row], axis=0)
    return _upscale_nn(cell, upscale)


def render_prediction_grid(
    coords: list[np.ndarray],
    feats: list[np.ndarray],
    truths: list[np.ndarray],
    preds: list[np.ndarray],
    betas: list[np.ndarray],
    oc_xs: list[np.ndarray],
    frames: list[tuple[int, int]],
    grid: tuple[int, int] = (4, 4),
    upscale: int = 3,
    sep_px: int = 4,
) -> np.ndarray:
    """Compose per-event cells into a (rows x cols) grid image (CHW uint8).

    ``coords[i]`` is (N_i, 3) pixel/normalized xy; ``feats[i]`` is (N_i, 3)
    RGB in [0,1]; ``oc_xs[i]`` is (N_i, D) cluster coords; the remaining
    lists are (N_i,).
    """
    rows, cols = grid
    n_events = min(len(coords), rows * cols)
    cells = []
    for i in range(n_events):
        cell = _render_one_event(
            coords[i], feats[i], truths[i], preds[i], betas[i], oc_xs[i],
            frames[i], i, upscale,
        )
        cells.append(cell)
    if not cells:
        return np.zeros((3, 32, 32), dtype=np.uint8)

    # pad all cells to same H/W (events may have different frame sizes)
    max_h = max(c.shape[0] for c in cells)
    max_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        pad_h = max_h - c.shape[0]
        pad_w = max_w - c.shape[1]
        if pad_h or pad_w:
            c = np.pad(c, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
        padded.append(c)
    while len(padded) < rows * cols:
        padded.append(np.full_like(padded[0], 255))

    v_sep = np.full((max_h, sep_px, 3), 0, dtype=np.uint8)
    h_sep_width = cols * max_w + (cols - 1) * sep_px
    h_sep = np.full((sep_px, h_sep_width, 3), 0, dtype=np.uint8)

    row_imgs = []
    for r in range(rows):
        row_cells = padded[r * cols : (r + 1) * cols]
        parts = []
        for i, cell in enumerate(row_cells):
            parts.append(cell)
            if i < cols - 1:
                parts.append(v_sep)
        row_imgs.append(np.concatenate(parts, axis=1))

    parts = []
    for i, row_img in enumerate(row_imgs):
        parts.append(row_img)
        if i < rows - 1:
            parts.append(h_sep)
    grid_hw3 = np.concatenate(parts, axis=0)
    return np.transpose(grid_hw3, (2, 0, 1))  # HWC -> CHW


def log_prediction_grid(
    writer: "SummaryWriter",
    tag: str,
    batch: dict[str, Any],
    preds: dict[str, torch.Tensor],
    pred_cluster: torch.Tensor,
    step: int,
    grid: tuple[int, int] = (4, 4),
    upscale: int = 3,
) -> None:
    """Split a concatenated batch back into events and render the grid."""
    offsets = batch["offset"].detach().cpu().numpy().tolist()
    frames = batch["frame"].detach().cpu().numpy().tolist()
    coord = batch["coord"].detach().cpu().numpy()
    feat = batch["feat"].detach().cpu().numpy()
    truth = batch["object_id"].detach().cpu().numpy()
    pred = pred_cluster.detach().cpu().numpy()
    beta = preds["beta"].detach().cpu().numpy()
    oc_x = preds["x"].detach().cpu().numpy()

    starts = [0] + offsets[:-1]
    coords, feats, truths, preds_l, betas, oc_xs, fr = [], [], [], [], [], [], []
    for i, (s, e) in enumerate(zip(starts, offsets)):
        coords.append(coord[s:e])
        feats.append(feat[s:e])
        truths.append(truth[s:e])
        preds_l.append(pred[s:e])
        betas.append(beta[s:e])
        oc_xs.append(oc_x[s:e])
        fr.append(tuple(int(x) for x in frames[i]))

    img = render_prediction_grid(
        coords, feats, truths, preds_l, betas, oc_xs, fr,
        grid=grid, upscale=upscale,
    )
    writer.add_image(tag, img, global_step=step, dataformats="CHW")


# ---------------------------------------------------------------------------
# OC-space projector
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_oc_embedding(
    writer: "SummaryWriter",
    tag: str,
    x: torch.Tensor,
    object_id: torch.Tensor,
    step: int,
    max_points: int = 4096,
) -> None:
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


# ---------------------------------------------------------------------------
# Run description (the "readme" inside TB)
# ---------------------------------------------------------------------------

_RUN_README = """\
# Run overview

This run trains a Point Transformer V3 backbone and Object Condensation heads
on the shapes pseudo-dataset. Every non-white pixel in a canvas is a point;
the model learns to cluster pixels into shapes and predict each shape's
class plus its bounding-box width and height.

---

## SCALARS tab

All scalars are logged under two prefixes:

- **`train/*`** — logged every step.
- **`val/*`** — logged once per epoch (mean across the val split).

Key scalars:

| tag | what it is | what to watch for |
|---|---|---|
| `*/total` | weighted sum of every loss component — the quantity AdamW actually minimizes | should decrease; any sudden spike usually means a bad batch or LR too high |
| `*/oc_attractive` | OC attractive potential: pulls same-object hits toward the condensation point | should drop fastest early in training |
| `*/oc_repulsive` | OC repulsive potential: pushes different-object hits apart | drops more slowly; floor is set by `q_min` |
| `*/oc_coward` | OC "coward" penalty on low-beta hits in real objects | keeps beta from collapsing to zero |
| `*/oc_noise` | penalty on noise hits with high beta (NaN/0 if no noise in the event) | usually flat/zero on the shapes dataset |
| `*/shape_id` | cross-entropy for shape-class prediction over foreground hits | drops once clustering starts working |
| `*/width`, `*/height` | MSE of per-hit width/height regression (pixels²) | expect ~10–50 early, <5 when converged |
| `*/grad_norm` | L2 norm of the full-model gradient before clipping | spikes indicate an exploding gradient; clip threshold is `grad_clip` in config |

---

## IMAGES tab

**`viz/grid`** — a 4x4 panel of **fixed** val-set events (same events every step,
so you can watch them converge). Each cell stacks three rows:

1. **TRUTH** — ground-truth clustering. Each truth object id has a deterministic
   color from a shared palette (seed 42).
2. **PRED** — predicted clusters from running OC inference (β > `t_β`, then
   radius `t_d` in cluster-coord space). Colors are **matched** to truth:
   each predicted cluster gets the color of the truth object it most overlaps
   with, so a correctly-reconstructed shape shows the **same** color on both
   rows. Predictions with no truth overlap (false positives) are drawn in
   medium gray. Hits the OC inference did NOT claim (β below threshold, or
   outside every condensation ball in cluster-coord space) are rendered in
   **light gray** so you can tell "dropped by the model" apart from "no hit
   at this pixel" (white). Early in training, expect most of each shape to
   be light gray — β hasn't spiked yet. As training progresses those light
   gray regions should fill in with matched color.
3. **OC** — scatter of the learned cluster coordinates `x` for every hit
   in the event. Each point is colored by its **input RGB feature** (i.e.
   the color of the shape it belongs to in the source image) and alpha-
   blended against a white background by β — high-β hits render fully
   opaque, low-β hits fade out. If `cluster_dim > 2`, coordinates are
   reduced to 2D with a PCA projection (no t-SNE / UMAP). A small inset
   of the original shape image is pasted top-right for reference, so you
   can visually tie each OC-space cluster back to the shape that produced
   it.

Cell caption format: `event <i>  TRUTH  n=<n_truth_shapes>` and
`PRED  k=<n_pred_clusters>  matched=<k_matched>`. A shape is "matched" if its
predicted cluster has at least one overlapping truth hit.

---

## HISTOGRAMS tab

**`train/beta`** — distribution of predicted β across every hit in the step's
training batch.

What to look for as training progresses:
- Early: β clusters around 0.5 (random init).
- Mid: β becomes bimodal — a narrow spike above `t_β` for condensation-point
  hits, a broad mass below it.
- Late: the low-β mass is pushed toward 0, the high-β spike sharpens.

---

## PROJECTOR tab

**`viz/oc_space`** — a scatter of the learned **cluster coordinates** `x` for a
subsample of hits from the fixed viz batch. Each point is one hit.

- The scatter lives in `cluster_dim` dimensions (see `model.heads.cluster_dim`
  in the config). Stick to the **PCA** tab in the projector page — t-SNE /
  UMAP are not used by this run.
- Each point is labeled with its truth `object_id`. Click **"color by label"**
  to color by object id — ideally each object forms a tight, isolated cloud
  (attractive term) that is far from other objects (repulsive term).
- Use the "neighbors" slider to check if a hit's nearest neighbors in OC
  space share its object id.

**What failure modes look like:**
- All points in one blob → attractive term dominates, nothing is separated.
- Points scattered uniformly → repulsive term dominates, no condensation.
- Multiple objects share the same cloud → the model is merging them.

---

## Config snapshot

The exact run config is written to `<log_dir>/config.yaml`. The architecture
report (parameter counts + Mermaid flowchart + torchinfo layer table) is in
`<log_dir>/architecture/architecture.md`.
"""


def log_run_description(writer: "SummaryWriter", config_dump: str = "") -> None:
    """Write the human-readable run overview to the TB Text tab."""
    text = _RUN_README
    if config_dump:
        text = text + "\n\n## Config (verbatim)\n\n```yaml\n" + config_dump + "\n```\n"
    writer.add_text("0_overview", text, global_step=0)
