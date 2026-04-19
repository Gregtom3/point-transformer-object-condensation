"""TensorBoard helpers for OC training runs.

Rendering of the per-event TRUTH / PRED / OC panels is delegated to an
:class:`~src.tasks.base.OCTask`, so swapping data domains only requires
a new task subclass — the trainer and this logger are generic.

Three kinds of artefacts are logged here:

1. Scalars (via :func:`log_scalars`) — key/value pairs.
2. Prediction grid (via :func:`log_prediction_grid`) — a rows × cols grid
   of per-event cells composed by ``task.render_grid(...)``. Each cell
   stacks TRUTH, PRED, and OC panels with header / sublabel strips.
3. Cluster-coord projector (via :func:`log_oc_embedding`) — raw cluster
   coordinates per hit, labeled by truth object id. View in PCA mode.

A one-shot :func:`log_run_description` dumps a markdown block into the
Text tab so a reader landing on the run knows what every panel means.
"""
from __future__ import annotations

from typing import Any

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


def log_prediction_grid(
    writer: "SummaryWriter",
    tag: str,
    task: Any,
    batch: dict[str, Any],
    preds: dict[str, torch.Tensor],
    pred_cluster: torch.Tensor,
    step: int,
    grid: tuple[int, int] = (4, 4),
    upscale: int = 3,
) -> None:
    """Render the fixed-batch grid via the task and log it to TB."""
    img = task.render_grid(batch, preds, pred_cluster, grid=grid, upscale=upscale)
    writer.add_image(tag, img, global_step=step, dataformats="CHW")


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
on the configured task. Rendering of the per-event panels is provided by the
task class itself (see ``src/tasks/``), so what you see in the IMAGES tab is
always aligned with the data format being trained on.

---

## SCALARS tab

Scalars are grouped so loss components sit under their own section and
diagnostic quantities like the gradient norm stay separate:

- **`train/loss/*`** — objective components, logged every step.
- **`val/loss/*`** — same components averaged over the val split, once per epoch.
- **`train/grad_norm`** — L2 norm of the full-model gradient before clipping;
  spikes signal an exploding gradient (clip threshold: `grad_clip` in config).

Loss components (same keys under both `train/loss/` and `val/loss/`):

| tag | what it is | what to watch for |
|---|---|---|
| `.../total` | weighted sum of every loss component — the quantity AdamW actually minimizes | should decrease; any sudden spike usually means a bad batch or LR too high |
| `.../oc_attractive` | OC attractive potential: pulls same-object hits toward the condensation point | should drop fastest early in training |
| `.../oc_repulsive` | OC repulsive potential: pushes different-object hits apart | drops more slowly; floor is set by `q_min` |
| `.../oc_coward` | OC "coward" penalty on low-beta hits in real objects | keeps beta from collapsing to zero |
| `.../oc_noise` | penalty on noise hits with high beta (NaN/0 if no noise in the event) | usually flat/zero on the shapes dataset |
| `.../shape_id` | cross-entropy for shape-class prediction over foreground hits | drops once clustering starts working |
| `.../width`, `.../height` | MSE of per-hit width/height regression in frame-normalized units | expect ~0.1 early, <0.01 when converged |

---

## IMAGES tab

**`viz/grid`** — a rows × cols panel of **fixed** val-set events (same events
every step, so you can watch them converge). Each cell stacks three panels:

1. **TRUTH** — ground-truth clustering, colored per truth object id from a
   shared deterministic palette (seed 42).
2. **PRED** — clusters produced by the paper-standard OC inference:
   hits are ordered by β, the highest unassigned β above `t_β` becomes a
   condensation point, and every still-unassigned hit within radius `t_d`
   in cluster-coord space joins that cluster (regardless of the joiner's
   own β). Repeat; stop when no unassigned β exceeds `t_β`. Anything left
   is **light gray** (noise). Colors are matched to TRUTH via majority
   vote, so a correctly reconstructed object keeps its color across panels;
   predicted clusters with no truth overlap render in **medium gray**.
3. **OC** — scatter of the learned cluster coordinates `x` for every hit
   in the event, on its own canvas (not the image pixel grid). Each point
   is colored by its **input RGB feature** (the color the shape has in the
   source image) and alpha-blended against white by β — high-β hits are
   fully opaque, low-β hits fade out. If `cluster_dim > 2`, coordinates are
   reduced to 2D by the projection selected in `viz.oc_projection` (`pca`
   default, or `umap` for non-linear reduction on well-separated clusters).
   A small inset of the shape image is pasted top-right so you can tie each
   OC-space cluster back to the shape that produced it.

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
subsample of hits from the fixed viz batch.

- The scatter lives in `cluster_dim` dimensions (see `model.heads.cluster_dim`
  in the config). Stick to the **PCA** tab — t-SNE / UMAP are not used.
- Each point is labeled with its truth `object_id`. Click **"color by label"**
  to color by object id — ideally each object forms a tight, isolated cloud
  (attractive term) that is far from other objects (repulsive term).

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
