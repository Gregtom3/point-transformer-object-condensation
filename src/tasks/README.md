# `src/tasks/` — the OCTask interface

This folder defines the **data-domain boundary** of the pipeline. Every
data format the pipeline handles (detector hits, shape images, LiDAR
scans, ...) is a concrete subclass of [`OCTask`](base.py).

> [!TIP]
> If you're adding your own data, start by reading
> [`../../docs/custom_task.md`](../../docs/custom_task.md). It walks you
> through the interface end-to-end with a concrete scenario.

## What an `OCTask` owns

| Responsibility | Abstract method | Why the task owns it |
|---|---|---|
| Dataset size | `__len__` | Varies per source (file count, row count, ...). |
| One event | `__getitem__(idx) -> dict[str, Tensor]` | Source I/O and unit conventions are domain-specific. |
| Sanity print | `__repr__` | So `print(task)` is informative in logs. |
| Batching | `collate(batch)` | Flattening to PTv3's `coord`/`feat`/`offset` depends on which per-hit keys exist. |
| Truth panel | `plot_truth(event, size_hw)` | Domain picks the visualization (image, 3D scatter, ...). |
| Pred panel | `plot_pred(event, pred_cluster, size_hw)` | Same — and colors should be *matched* to truth (helper provided). |
| OC-space panel | `plot_oc(event, oc_xy, beta, size_hw)` | Domain decides the scatter style (inset, axes, ...). |

The base class owns everything that generalizes across tasks:

* shared color palette (`PALETTE`, `UNMATCHED_COLOR`, `UNCLAIMED_COLOR`)
* `match_pred_to_truth(truth, pred)` majority-vote helper
* `pca_to_2d(x)` / `umap_to_2d(x)` / `project_2d(x, mode)` reducers
* label strips and grid composition in `render_cell` / `render_grid`

## Contract for `__getitem__`

Return a `dict[str, torch.Tensor]` where every per-hit tensor has the
same first dimension `N`. Required keys:

| key | dtype | shape | meaning |
|---|---|---|---|
| `coord` | float32 | (N, 3) | Hit positions. Channels are x, y, z. 2D tasks leave z = 0. |
| `feat` | float32 | (N, F) | Per-hit input features. RGB for the shapes task; anything for a detector task (dE/dx, time, ...). |
| `object_id` | int64 | (N,) | Ground-truth object assignment. `0` or negative = noise. IDs must be **globally unique across all events in the split** (see `docs/data_format.md`). |

Optional keys for payload regression / classification heads:

| key | dtype | shape | consumed by |
|---|---|---|---|
| `shape_id_per_hit` | int64 | (N,) | PID cross-entropy (`loss.shape_id`) |
| `width_per_hit`, `height_per_hit` | float32 | (N,) | width/height MSE |
| `energy_per_hit` | float32 | (N,) | legacy energy regression |
| `momentum_per_hit` | float32 | (N, 3) | legacy momentum regression |
| `frame` | int64 | (2,) | image-like tasks use this for rendering |

## Contract for `collate`

Return a `dict` with PTv3's flat layout:

| key | shape | meaning |
|---|---|---|
| `coord` | (sum N_i, 3) | all events concatenated along hit axis |
| `feat` | (sum N_i, F) | same |
| *(any per-hit truth key)* | (sum N_i, ...) | same |
| `offset` | (B,) | cumulative sum of per-event hit counts; backbone reads this to know where one event ends and the next begins |
| `batch_size` | int | `B` |

See [`shapes.py`](shapes.py) for a reference implementation.

## Rendering: what each method returns

All three `plot_*` methods return an `(H, W, 3)` **uint8** RGB array.
The base class composes them into a grid cell with a labeled header,
then tiles cells into the final `(3, H_total, W_total)` image that
`render_grid` hands to TensorBoard.

### `plot_truth(event, size_hw)`
Per-hit coloring of the event. The reference `ShapesTask` colors each
hit by its input `feat` (so the panel reproduces the source image) and
paints background as white.

### `plot_pred(event, pred_cluster, size_hw)`
Same layout, but colors come from the **predicted** clusters. Use
`match_pred_to_truth` to majority-vote each predicted cluster to the
truth id it overlaps most, then color matched clusters the same way
you colored the TRUTH panel so correctly reconstructed objects keep
their color. Convention:

* `UNCLAIMED_COLOR` (light gray) for hits the OC inference did not claim
  (cluster_id ≤ 0). Distinguishes "dropped by the model" from
  "no hit here" (white).
* `UNMATCHED_COLOR` (medium gray) for predicted clusters with no truth
  overlap (false positives).

### `plot_oc(event, oc_xy, beta, size_hw)`
Scatter of the (already 2D-reduced) cluster coordinates. The reference
`ShapesTask` colors each point by its `feat` and alpha-blends by β so
high-β hits are opaque and low-β hits fade out. A shape-image inset in
the corner anchors the viewer so OC-space clusters can be visually
tied back to the source image.

## Adding a new task

1. Subclass `OCTask`.
2. Implement the six abstract methods.
3. Optionally override `_slice_event` if your events carry per-event
   metadata other than the standard keys.
4. Export it from `src/tasks/__init__.py` so `scripts/train.py` can
   import it.

See [`../../docs/custom_task.md`](../../docs/custom_task.md) for the
full walkthrough.
