# Bringing your own data

This guide walks through **slotting a new data format into the pipeline**
end-to-end: from raw files to training, evaluation, ONNX export, and
writing predictions back into your source data with full provenance. We
use the [shapes pseudo-dataset](../data/generate_shapes.py) as a
working reference the whole way through — every piece you need to write
has an analog in `ShapesTask`.

> [!TIP]
> Work top-down. The first thing the pipeline needs is a **Task**; most
> of the other pieces fall out of it.

## A concrete scenario

You have your own data. For the rest of this guide we'll assume:

* It lives on disk in some format of your choosing (HDF5 / ROOT / Parquet / numpy / ...).
* Each "event" is a set of points with features — positions, feature
  vectors, and a per-point label that says which object they belong
  to. 0 means noise / background.
* You want to predict per-object things (a class label, maybe a
  regression target like a size or an energy).
* You want to train a model, evaluate it, and end up with **a copy of
  your data that has the predicted features appended**, tagged with the
  checkpoint / config / timestamp that produced them.

Here's the shortest possible map from that scenario to this repo:

| You need ... | You write ... | Reference |
|---|---|---|
| The pipeline to read your data | A `MyTask(OCTask)` subclass | [`src/tasks/shapes.py`](../src/tasks/shapes.py) |
| A config (runcard) | A YAML in `configs/train/` | [`configs/train/shapes.yaml`](../configs/train/shapes.yaml) |
| Entry points | `scripts/train.py`, `scripts/evaluate.py` already work — they just need to import your task | [`scripts/train.py`](../scripts/train.py) |
| Train-time noise / rotations | A `MyAug(Augmentation)` subclass | [`src/augmentations/`](../src/augmentations/) |
| Predictions written back to HDF5 | `scripts/export_predictions.py` | [`scripts/export_predictions.py`](../scripts/export_predictions.py) |

---

## Step 1: write your task class

> [!NOTE]
> **What this step does:** teaches the pipeline how to read your data
> and how to visualize one event's truth / prediction / OC-space panel.

An [`OCTask`](../src/tasks/base.py) is a `torch.utils.data.Dataset` with
six extra methods. Subclass it and fill in the abstract methods. The
full contract and method-by-method breakdown is in
[`src/tasks/README.md`](../src/tasks/README.md); here we sketch the
skeleton.

```python
# src/tasks/mytask.py
from __future__ import annotations

import h5py  # or whatever you need
import numpy as np
import torch

from src.augmentations.base import Augmentation
from .base import OCTask, UNCLAIMED_COLOR, UNMATCHED_COLOR, match_pred_to_truth


class MyTask(OCTask):
    def __init__(self, path, augmentation: Augmentation | None = None, ...) -> None:
        self.path = path
        self.augmentation = augmentation
        # open source, cache per-event offsets, etc.

    # --- data -----------------------------------------------------------

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        # Required keys on return:
        #   coord:     float32 (N, 3)   — positions (2D tasks leave z=0)
        #   feat:      float32 (N, F)   — per-hit features
        #   object_id: int64   (N,)     — truth id (0 = noise); globally unique
        # Optional per-hit payload targets:
        #   shape_id_per_hit, width_per_hit, height_per_hit, energy_per_hit, momentum_per_hit
        event = {
            "coord": torch.as_tensor(..., dtype=torch.float32),
            "feat": torch.as_tensor(..., dtype=torch.float32),
            "object_id": torch.as_tensor(..., dtype=torch.long),
            # ... your payload targets ...
        }
        if self.augmentation is not None:
            event = self.augmentation(event)
        return event

    def __repr__(self) -> str:
        return f"MyTask(path={self.path}, n_events={self.n_events})"

    def collate(self, batch: list[dict]) -> dict:
        # Flatten along the hit axis and emit PTv3's offset layout.
        # See ShapesTask.collate for the canonical pattern.
        ...

    # --- rendering -----------------------------------------------------

    def plot_truth(self, event, size_hw):  ...
    def plot_pred(self, event, pred_cluster, size_hw):  ...
    def plot_oc(self, event, oc_xy, beta, size_hw):  ...
```

### Per-hit alignment rule

Every per-hit tensor in `event` must share the same first dimension
`N`. If you drop/filter hits in `__getitem__`, drop them from every
per-hit tensor. The collate function relies on this.

### What the three panels mean

The base class composes a `(TRUTH | PRED | OC)` cell per event and tiles
them into the TensorBoard grid. Each method returns an `(H, W, 3)` uint8
RGB array. Key conventions:

| Panel | What it shows | Coloring |
|---|---|---|
| TRUTH | Ground-truth clustering. | Shapes task colors each hit by its input `feat` so the panel reproduces the source image. Any scheme you like is fine as long as it's deterministic. |
| PRED | Predicted clusters after OC inference. | **Match colors to TRUTH** — use `match_pred_to_truth()` and paint matched clusters the same color as their truth counterpart. Use `UNCLAIMED_COLOR` (light gray) for cluster_id ≤ 0 and `UNMATCHED_COLOR` (medium gray) for false positives. |
| OC | Scatter of `oc_xy` (already 2D-reduced). | Show condensation structure. Shapes task colors by `feat` and alpha-blends by β (high β = opaque). |

See [`src/tasks/shapes.py`](../src/tasks/shapes.py) for a reference.

### Register your task

```python
# src/tasks/__init__.py
from .shapes import ShapesTask  # noqa: F401
from .mytask import MyTask      # noqa: F401
```

---

## Step 2: write a runcard

> [!NOTE]
> **What this step does:** tells the trainer how to build your
> backbone, heads, loss, task, and TB logging for one run.

Copy `configs/train/shapes.yaml` and edit it in place. The schema:

```yaml
data:
  root: path/to/your/data       # task-specific; MyTask reads from here
  train_file: train.h5          # or whatever your task needs
  val_file: val.h5
  test_file: test.h5
  normalize_coords: true
  max_hits: 0                   # 0 = no cap

model:
  backbone:
    in_channels: <F>            # match MyTask.feat's last dim
    enable_flash: false
    enc_depths: [2, 2, 2, 2]
    enc_channels: [32, 64, 128, 256]
    # ... (see configs/model/ptv3_base.yaml for all knobs)
  heads:
    cluster_dim: 2              # 2 = directly plottable
    hidden_dim: 128
    n_pid_classes: <C>          # number of per-object classes you predict
    predict_width_height: true  # set false if you don't need the regressor

loss:
  q_min: 1.0
  noise_threshold: 0
  payload_weight: 1.0
  shape_id_weight: 1.0          # cross-entropy weight on shape_id_per_hit
  width_weight: 1.0             # MSE weight on width_per_hit (normalized)
  height_weight: 1.0

train:
  batch_size: 2
  num_workers: 0
  log_dir: runs/mytask
  ckpt_dir: outputs/mytask
  tb_image_every: 200
  tb_embedding_every: 500
  trainer:
    max_epochs: 10
    lr: 3.0e-4
    weight_decay: 1.0e-4
    grad_clip: 1.0
    log_every: 20

inference:
  t_beta: 0.5                   # OC paper defaults
  t_d: 0.28

viz:
  oc_projection: pca            # pca | umap (applies when cluster_dim > 2)
```

> [!IMPORTANT]
> **`in_channels` must match your `feat` width.** If your `feat` has
> 8 channels, `model.backbone.in_channels: 8`. Mismatches surface as a
> cryptic shape error on the first forward pass.

---

## Step 3: wire your task into train / eval

The existing entrypoints already do everything you need — you just need
to import your task instead of (or alongside) `ShapesTask`. The
simplest change:

```python
# scripts/train.py
# replace or add:
from src.tasks import MyTask as TaskCls
# ...
train_task = TaskCls(root / cfg.data.train_file, ...)
val_task   = TaskCls(root / cfg.data.val_file, ...)
```

If you want to keep both working, dispatch on a config field:

```yaml
# configs/train/mytask.yaml
data:
  task_class: MyTask            # or ShapesTask
  root: ...
```

```python
TASK_REGISTRY = {"ShapesTask": ShapesTask, "MyTask": MyTask}
TaskCls = TASK_REGISTRY[cfg.data.task_class]
```

Then:

```bash
python scripts/train.py --config configs/train/mytask.yaml
```

The trainer opens TensorBoard under `<log_dir>`:

* `train/loss/*` and `val/loss/*` — objective components (see
  [`src/utils/tb_logging.py`](../src/utils/tb_logging.py) for the list).
* `train/grad_norm` — clipped L2 gradient.
* `viz/grid` — fixed set of val events, refreshed every `tb_image_every`
  steps, using your task's `plot_*` methods.
* `viz/oc_space` — TB Projector scatter of raw `x` cluster coordinates.
* Text tab — a run README describing every panel, regenerated per run.

---

## Step 4: (optional) augmentations

> [!NOTE]
> **What this step does:** adds train-time noise so the model doesn't
> overfit to pixel-perfect inputs.

Augmentations are callables with the signature

```python
def __call__(self, event: dict) -> dict: ...
```

Everything else is in [`src/augmentations/README.md`](../src/augmentations/README.md).
The reference augmentations in [`src/augmentations/basic.py`](../src/augmentations/basic.py) are a template you can copy:

```python
from src.augmentations import Compose, RandomRotation2D, RandomColorJitter, RandomHitDropout

aug = Compose([
    RandomRotation2D(max_angle_deg=15),
    RandomColorJitter(sigma=0.02),
    RandomHitDropout(p=0.1),
])

train_task = MyTask(..., augmentation=aug)
val_task   = MyTask(...)  # augmentation=None is the default
```

> [!WARNING]
> Keep augmentations off for val / test. The trainer's viz panels
> pull from val, and a noisy val set makes TB loss curves wobble for
> the wrong reason.

---

## Step 5: evaluate

`scripts/evaluate.py` runs OC inference over a split and prints /
dumps aggregate metrics (purity, efficiency, shape accuracy, width /
height MAE). It already uses the task's `collate`, so if Step 3 hooked
your task in, eval just works:

```bash
python scripts/evaluate.py --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt --split test
```

Output is a JSON dumped to `args.out` (default
`outputs/mytask/eval.json`).

To add your own metrics, edit `scripts/evaluate.py` — it's intentionally
small and reads straight. Aggregate over events in the main loop, add
the new field to the final JSON dict.

---

## Step 6: append predictions back into your data

> [!NOTE]
> **What this step does:** runs inference over a split and writes
> per-hit predictions + run metadata back into the source HDF5 file
> so later analyses can load them without re-running the model.

The reference script is
[`scripts/export_predictions.py`](../scripts/export_predictions.py).
Run it:

```bash
python scripts/export_predictions.py \
    --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt \
    --split test \
    --output-group predictions_v1 \
    [--overwrite]
```

What it writes (per event):

```
event_000000/
├── coord, feat, object_id, ...     # your original data, untouched
└── predictions_v1/                 # new group
    ├── beta         (N,)  float32
    ├── cluster_id   (N,)  int64
    ├── pid_pred     (N,)  int64
    ├── width_pred   (N,)  float32
    ├── height_pred  (N,)  float32
    ├── x_cluster    (N, cluster_dim)  float32
    └── .attrs: t_beta, t_d
```

And at the file root:

```
predictions_meta/predictions_v1/.attrs:
    checkpoint:    absolute path of the .pt used
    config_path:   absolute path of the YAML used
    config_yaml:   entire runcard, verbatim
    timestamp_utc: ISO 8601 timestamp
    git_sha:       `git rev-parse HEAD` at export time
    hostname:      machine that ran it
    t_beta, t_d:   inference thresholds
    split:         train / val / test
```

> [!TIP]
> The `--output-group` flag lets you keep side-by-side runs in the same
> file: `predictions_v1`, `predictions_ablation_noise`, etc. Pick a
> descriptive name; future-you will thank you.

**Adapting the keys you write.** The script hardcodes five per-hit
fields because that's what the shapes heads produce. To write different
keys (e.g. your `energy_pred`, your `subdet_id_pred`), edit
`_write_predictions` — the function is ~20 lines and clearly marks
what's arbitrary.

---

## Step 7: ONNX export

`scripts/export_onnx.py` exports **the OC heads only**. The PTv3
backbone uses sparse-conv ops (spconv) that don't trace cleanly to
ONNX; keeping it in PyTorch while exporting just the heads is the
pragmatic split: feature extraction stays flexible, the "what does
this hit predict?" part is a clean dense MLP that onnx-runtimes can
consume.

```bash
python scripts/export_onnx.py --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt \
    --out outputs/mytask/heads.onnx
```

If you need the backbone in ONNX too: the honest answer is "export the
backbone separately with a custom op registration for spconv, or swap
it for a dense backbone". The repo doesn't solve that for you; the
heads export is what's tractable out of the box.

---

## Cheatsheet

```bash
# 1. one-time: install deps + submodules
CUDA=cu121 bash setup.sh

# 2. write src/tasks/mytask.py + configs/train/mytask.yaml

# 3. train
python scripts/train.py --config configs/train/mytask.yaml

# 4. watch
tensorboard --logdir runs/mytask

# 5. evaluate
python scripts/evaluate.py --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt --split test

# 6. write predictions + provenance back into HDF5
python scripts/export_predictions.py --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt --split test \
    --output-group predictions_v1

# 7. ONNX export of the heads
python scripts/export_onnx.py --config configs/train/mytask.yaml \
    --checkpoint outputs/mytask/epoch_009.pt --out outputs/mytask/heads.onnx
```

---

## Further reading

* [`src/tasks/README.md`](../src/tasks/README.md) — the `OCTask`
  interface in detail, with contracts for `__getitem__`, `collate`, and
  each `plot_*` method.
* [`src/augmentations/README.md`](../src/augmentations/README.md) — the
  augmentation contract and how to add a new one.
* [`docs/data_format.md`](data_format.md) — per-event and batch-level
  schema, HDF5 layout produced by the shapes generator, invariants.
* [`docs/architecture.md`](architecture.md) — model flowchart and OC
  inference procedure (β-threshold + radius clustering).
