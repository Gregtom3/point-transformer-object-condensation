# Tutorial: plug your own data into the pipeline

> [!NOTE]
> Audience: first-year grad student / first-time contributor. If you've
> never touched `torch.utils.data.Dataset` before, that's fine — this
> tutorial walks through it end-to-end and tells you exactly what to type.

You have some data. You want to train the PTv3 + Object Condensation
model on it, watch it converge in TensorBoard, evaluate, and save
predictions back next to your inputs. This guide shows you how,
using a **toy "left / right calorimeter" detector** as a worked example
(runnable code lives in [`examples/calorimeter/`](../examples/calorimeter/)).

At the end you'll have done every piece you need to do for your real
data — just with different file formats and different physics.

---

## The 30-second mental model

The pipeline has three moving parts that are yours:

```
┌──────────┐     ┌───────────┐     ┌──────────┐
│   your   │     │   your    │     │   your   │
│   data   │ ──▶ │   Task    │ ──▶ │  runcard │
│ on disk  │     │   class   │     │  (yaml)  │
└──────────┘     └───────────┘     └──────────┘
                       │                │
                       ▼                ▼
                 everything else runs unchanged:
                 backbone, OC heads, loss, trainer,
                 TensorBoard logging, evaluation, ONNX export
```

The **Task class** is the contract. Once it's written, the rest of the
repo stops caring what your data looks like.

---

## The checklist

Copy this into a scratch file and tick items as you go. If a step feels
stuck, read the linked section.

- [ ] **0.** Decide the shape of one event: how many per-hit tensors do
  you have, and which one is the truth clustering?
  ([→ Step 0](#step-0-pin-down-what-one-event-looks-like))
- [ ] **1.** Put your data on disk in a format you can read per-event.
  HDF5 is the path of least resistance in this repo.
  ([→ Step 1](#step-1-put-your-data-somewhere-you-can-read-it))
- [ ] **2.** Write a `Task` class that subclasses `OCTask`.
  Implement: `__len__`, `__getitem__`, `__repr__`, `collate`,
  `plot_truth`, `plot_pred`, `plot_oc`.
  ([→ Step 2](#step-2-write-the-task-class))
- [ ] **3.** Register the task name in `src/tasks/__init__.py`.
  ([→ Step 3](#step-3-register-the-task))
- [ ] **4.** Write a runcard YAML — copy an existing one and edit.
  ([→ Step 4](#step-4-write-the-runcard))
- [ ] **5.** Smoke-test it before training: does `task[0]` return a dict?
  Does `task.collate([task[0], task[1]])` work? Does
  `task.render_grid(...)` produce a non-blank image?
  ([→ Step 5](#step-5-smoke-test-before-training))
- [ ] **6.** Train. Watch TensorBoard. If the loss explodes, read
  [troubleshooting](#troubleshooting).
- [ ] **7.** Evaluate + export predictions back to your data with full
  provenance.
  ([→ Step 7](#step-7-evaluate-and-export-predictions))
- [ ] **8.** (optional) ONNX-export the heads.

---

## The running example: left/right calorimeter

> [!TIP]
> The complete, runnable implementation is at
> [`examples/calorimeter/`](../examples/calorimeter/). Try it:
> `python examples/calorimeter/generate.py --out data/calo_v1` then
> `python scripts/train.py --config examples/calorimeter/config.yaml`.

**The physics (one paragraph).** Imagine two flat calorimeter walls, one
at `x = -50 cm`, one at `x = +50 cm`. A particle originating near the
origin can fly outward and hit either wall. When it does, its energy
doesn't go into a single cell — it spreads over a small cluster ("a
shower") of nearby cells. Our dataset has many events; each event has a
few particles; each particle produced maybe 6–14 cells of energy. We
want the model to:

1. cluster cells back into their source particle (**OC**), and
2. predict each particle's total energy from its cells
   (**per-hit regression**).

**In OC terms:**

| OC concept | calorimeter concept |
|---|---|
| A hit | A calorimeter cell with nonzero energy deposit |
| `coord` | Cell's 3D position `(x, y, z)` in cm |
| `feat` | `(log10(E_cell), arrival_time, subdet_id)` — `subdet_id` is 0 for left, 1 for right |
| `object_id` | Unique ID of the particle that produced this cell |
| `noise` | Zero cells (never emitted, so there are none) |
| Payload target | `energy_per_hit` — total particle energy, broadcast per cell |

---

## Step 0: pin down what one event looks like

> [!IMPORTANT]
> Before you write any code, write two or three sentences describing
> your data in the table above. What's one hit? What are its features?
> What is the truth clustering called? What (if anything) are you
> regressing or classifying per-object?

If you can't fill that table in yet, you don't know your data well
enough to train a model on it. Go read your detector's docs, open one
event in a notebook, whatever it takes.

For the calorimeter, the table above is all we need.

---

## Step 1: put your data somewhere you can read it

The pipeline expects **one event per group in HDF5**, and that's the
path we recommend for new tasks — it's what `ShapesTask` and
`CalorimeterTask` both use. The advantage: your `__getitem__` is three
lines, and `scripts/export_predictions.py` can write results back into
the same file.

Target layout:

```
data/your_task_v1/
├── train.h5
│   └── meta/                     # attributes describing the split
│   └── event_000000/
│       ├── coord          (N, 3)  float32
│       ├── feat           (N, F)  float32
│       ├── object_id      (N,)    int64   # 0 = noise; globally unique across events
│       └── <your payload targets>
│   └── event_000001/
│   └── ...
├── val.h5
└── test.h5
```

> [!IMPORTANT]
> **`object_id` must be globally unique across every event in a
> split**. If event 0 has particles 1, 2, 3, then event 1's particles
> start at 4, not back at 1. The OC loss relies on this. See
> `data/generate_shapes.py` or `examples/calorimeter/generate.py` for
> how both generators keep a running counter.

If your data isn't in HDF5 yet, **write a generator script once** and
be done with it. The calorimeter generator is ~150 lines of
straightforward numpy:

```bash
# look at the reference implementation
less examples/calorimeter/generate.py

# run it
python examples/calorimeter/generate.py --out data/calo_v1 \
    --n-train 400 --n-val 50 --n-test 50 --seed 0
```

After it runs:

```
data/calo_v1/
├── train.h5        (400 events)
├── val.h5          (50 events)
├── test.h5         (50 events)
└── metadata.json   (counts, config, paths)
```

---

## Step 2: write the Task class

> [!NOTE]
> Every concept in this section has a working example in
> [`examples/calorimeter/task.py`](../examples/calorimeter/task.py).
> Read it alongside this walkthrough — it's ~250 lines of
> mostly-boilerplate.

Your task is a single class that subclasses `OCTask`. The class lives
wherever is convenient (`src/tasks/your_task.py`, or
`examples/your_task/task.py` while iterating).

### 2a. Data methods

```python
from src.tasks.base import OCTask

class CalorimeterTask(OCTask):
    def __init__(self, path, normalize_coords=True, max_hits=0,
                 panel_hw=(192, 192), projection="pca", augmentation=None):
        self.path = Path(path)
        self.normalize_coords = normalize_coords
        self.max_hits = max_hits
        self.panel_hw = panel_hw
        self.projection = projection
        self.augmentation = augmentation
        # Read whatever you need from meta ONCE in __init__ — DataLoader
        # workers will copy this object to each process.
        with h5py.File(self.path, "r") as f:
            self.n_events = int(f["meta"].attrs["n_events"])
            self.wall_x = float(f["meta"].attrs["wall_x"])
            self.frame = tuple(int(x) for x in f["meta"].attrs["frame"])
        self._h5 = None  # lazy-open in worker: see _file() below

    def _file(self):
        # Lazily open the HDF5 file. This matters for multi-worker
        # DataLoaders: each worker needs its own file handle.
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __len__(self):
        return self.n_events

    def __repr__(self):
        return f"CalorimeterTask(path={self.path}, n_events={self.n_events})"

    def __getitem__(self, idx):
        g = self._file()[f"event_{idx:06d}"]
        coord = np.asarray(g["coord"])
        feat = np.asarray(g["feat"])
        object_id = np.asarray(g["object_id"])
        energy = np.asarray(g["energy_per_hit"])

        if self.normalize_coords:
            # Whatever normalization puts coord into [0, 1]. PTv3's sparse
            # voxelizer wants a bounded positive-quadrant grid.
            coord = coord.copy()
            coord[:, 0] = (coord[:, 0] + self.wall_x) / (2 * self.wall_x)
            coord[:, 1] = (coord[:, 1] + 40) / 80.0
            coord[:, 2] = (coord[:, 2] + 40) / 80.0

        event = {
            "coord":          torch.from_numpy(coord).float(),          # (N, 3)
            "feat":           torch.from_numpy(feat).float(),           # (N, F)
            "object_id":      torch.from_numpy(object_id).long(),       # (N,)
            "energy_per_hit": torch.from_numpy(energy).float(),         # (N,)
            "frame":          torch.tensor(self.frame, dtype=torch.long),
        }
        if self.augmentation is not None:
            event = self.augmentation(event)
        return event
```

> [!IMPORTANT]
> **Every per-hit tensor must share the same first dimension `N`.** If
> you drop a row anywhere, drop it from every per-hit tensor. The
> collate step concatenates them and assumes they're aligned.

### 2b. Collate

`collate` takes a list of event dicts and emits the flat-batched layout
that PTv3 expects. It's almost always the same boilerplate — copy it
from `ShapesTask.collate` or `CalorimeterTask.collate` and edit the
`keys_flat` list.

```python
def collate(self, batch):
    keys_flat = ["coord", "feat", "object_id", "energy_per_hit"]
    out = {k: [] for k in keys_flat}
    sizes, frames = [], []
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
```

> [!NOTE]
> `offset` is PTv3's "cumulative hit count per event in this batch". The
> backbone reads it to know where one event ends and the next begins.
> You don't need to understand the internals — just emit it.

### 2c. The three plot methods

These methods each take an event (dict) and a target `(H, W)` size and
return an `(H, W, 3)` uint8 RGB image. The base class composes them
into a grid cell with TRUTH / PRED / OC labels.

**`plot_truth(event, size_hw)`** — how do you want to visualize the
ground-truth clustering? For the calorimeter, we show the two walls
side-by-side as (y, z) scatter panels, color each hit by its
`object_id`:

```python
def plot_truth(self, event, size_hw):
    obj = event["object_id"].cpu().numpy()
    id_to_color = self._id_palette(obj)   # small helper: id -> stable RGB
    colors = np.zeros((obj.shape[0], 3), dtype=np.uint8)
    for i, o in enumerate(obj):
        if o > 0:
            colors[i] = id_to_color[int(o)]
    return self._paint_detector_panel(event, colors, mask=(obj > 0))
```

**`plot_pred(event, pred_cluster, size_hw)`** — same layout, but use
**matched colors** (each predicted cluster colored the same as its
best-overlapping truth cluster). The base class gives you
`match_pred_to_truth` to do the assignment:

```python
from src.tasks.base import match_pred_to_truth, UNCLAIMED_COLOR, UNMATCHED_COLOR

def plot_pred(self, event, pred_cluster, size_hw):
    truth = event["object_id"].cpu().numpy()
    id_to_color = self._id_palette(truth)
    mapping = match_pred_to_truth(truth, pred_cluster)  # {pred_id: truth_id or -1}
    colors = np.zeros((len(pred_cluster), 3), dtype=np.uint8)
    mask = np.ones(len(pred_cluster), dtype=bool)
    for i, p in enumerate(pred_cluster):
        p = int(p)
        if p <= 0:
            colors[i] = UNCLAIMED_COLOR      # light gray: OC dropped this hit
        else:
            t = mapping.get(p, -1)
            colors[i] = id_to_color[t] if t > 0 else UNMATCHED_COLOR
    return self._paint_detector_panel(event, colors, mask)
```

**`plot_oc(event, oc_xy, beta, size_hw)`** — scatter of the learned
cluster coordinates. `oc_xy` has already been reduced to 2D (via PCA or
UMAP, per `viz.oc_projection`). Color by truth id, alpha by β so
high-β points are opaque and low-β points fade. The pattern:

```python
def plot_oc(self, event, oc_xy, beta, size_hw):
    # normalize oc_xy into [margin, size - margin], draw cv2.circle per point
    # with alpha = beta, color = id_to_color[truth_id]
    # see examples/calorimeter/task.py for the full implementation
    ...
```

You can copy that method from the calorimeter example verbatim if your
task looks similar.

> [!TIP]
> You don't need to write these panels from scratch. For 2D image-like
> tasks, copy from `ShapesTask`. For 3D scatter-like tasks, copy from
> `CalorimeterTask`. The patterns generalize.

---

## Step 3: register the task

Add your class to `src/tasks/__init__.py` so the runcard can name it:

```python
# src/tasks/__init__.py
from .your_task import YourTask  # noqa: F401

TASK_REGISTRY = {
    "ShapesTask": ShapesTask,
    "CalorimeterTask": _load_calorimeter_task,
    "YourTask": YourTask,          # ← add this line
}
```

`get_task_class("YourTask")` now returns your class. `scripts/train.py`,
`scripts/evaluate.py`, and `scripts/export_predictions.py` call this
function — you don't edit those scripts.

---

## Step 4: write the runcard

Copy [`configs/train/shapes.yaml`](../configs/train/shapes.yaml) or
[`examples/calorimeter/config.yaml`](../examples/calorimeter/config.yaml)
to `configs/train/your_task.yaml` and edit. The fields you'll almost
certainly change:

| Field | Set it to |
|---|---|
| `data.task_class` | `"YourTask"` (the name you registered in Step 3) |
| `data.root` | Path to the directory that has `train.h5` / `val.h5` / `test.h5` |
| `model.backbone.in_channels` | The width `F` of your `feat` tensor |
| `model.heads.n_pid_classes` | Number of per-object classes if you have a classification target; 2 if unused |
| `model.heads.predict_width_height` | `false` if you don't have width/height targets |
| `model.heads.predict_energy` | `true` if you have `energy_per_hit` |
| `loss.*_weight` | Per-head weight; start at 1.0 and adjust after seeing loss magnitudes |
| `train.log_dir` / `train.ckpt_dir` | `runs/your_task`, `outputs/your_task` |

> [!WARNING]
> **`in_channels` must match `feat` width exactly.** If `feat` has 3
> columns, set `in_channels: 3`. Mismatches surface as a cryptic shape
> error on the first forward pass. Open `examples/calorimeter/task.py`
> and `config.yaml` side by side to see this constraint honored.

---

## Step 5: smoke-test before training

Training runs are slow to fail. Catch bugs in 10 seconds instead:

```python
# scratch.py — run with python scratch.py
import torch
from src.tasks import get_task_class

TaskCls = get_task_class("CalorimeterTask")  # or YourTask
task = TaskCls("data/calo_v1/train.h5")

# (a) one event parses
ev = task[0]
for k, v in ev.items():
    if torch.is_tensor(v):
        print(f"  {k}: {tuple(v.shape)}  {v.dtype}")

# (b) per-hit tensors are aligned
Ns = {k: v.shape[0] for k, v in ev.items()
      if torch.is_tensor(v) and v.dim() >= 1 and k not in ("frame",)}
assert len(set(Ns.values())) == 1, f"per-hit N mismatch: {Ns}"

# (c) collate produces PTv3's flat layout
batch = task.collate([task[i] for i in range(4)])
assert batch["coord"].shape[0] == batch["offset"][-1]

# (d) the grid renders
fake_preds = {
    "beta": torch.rand(batch["coord"].shape[0]),
    "x": torch.randn(batch["coord"].shape[0], 2),
}
fake_cluster = torch.zeros(batch["coord"].shape[0], dtype=torch.long)
img = task.render_grid(batch, fake_preds, fake_cluster, grid=(2, 2), upscale=2)
print(f"grid image: {img.shape}  dtype={img.dtype}  (expect CHW uint8)")
```

If any of those assertions fail, fix them before you start training —
they'd just fail *inside* training with noisier error messages.

---

## Step 6: train and watch TensorBoard

```bash
python scripts/train.py --config configs/train/your_task.yaml
```

Then in another terminal:

```bash
tensorboard --logdir runs/your_task
```

What to look at, in order:

1. **Scalars → `train/loss/*`** — every loss component should be the
   same order of magnitude (typically O(0.1) to O(10)). If one is 100×
   bigger than the others, your weight / target normalization is off.
   See [troubleshooting](#troubleshooting).
2. **Scalars → `train/grad_norm`** — stays O(1). Spikes mean an
   exploding gradient.
3. **Images → `viz/grid`** — side-by-side TRUTH / PRED / OC panels on a
   fixed set of val events. Watch them evolve. The first few steps
   will show random PRED; a well-wired model starts recovering shape
   outlines within a few hundred steps.
4. **Text tab** — a human-readable run overview with the exact config
   dumped verbatim.

Checkpoints land in `outputs/your_task/epoch_*.pt` per epoch.

---

## Step 7: evaluate and export predictions

**Aggregate metrics** (purity, efficiency, MAE, ...):

```bash
python scripts/evaluate.py --config configs/train/your_task.yaml \
    --checkpoint outputs/your_task/epoch_009.pt --split test
# prints JSON to stdout and writes outputs/your_task/eval.json
```

**Write per-hit predictions back into the HDF5** with full provenance
(checkpoint path, config YAML verbatim, UTC timestamp, git SHA,
hostname, thresholds):

```bash
python scripts/export_predictions.py \
    --config configs/train/your_task.yaml \
    --checkpoint outputs/your_task/epoch_009.pt \
    --split test \
    --output-group predictions_v1
```

After that runs, every event in the file has a new subgroup:

```
event_000000/
├── coord, feat, object_id, ...      # untouched inputs
└── predictions_v1/
    ├── beta         (N,)   float32
    ├── cluster_id   (N,)   int64
    ├── x_cluster    (N, 2) float32
    ├── pid_pred     (N,)   int64    # if your task predicts class
    ├── energy_pred  (N,)   float32  # if your task predicts energy
    └── width_pred / height_pred     # if your task predicts those
```

And at the file root:

```
predictions_meta/predictions_v1/.attrs:
    checkpoint:    .../epoch_009.pt
    config_yaml:   <full yaml>
    timestamp_utc: 2026-04-19T15:04:05+00:00
    git_sha:       abc123...
    hostname:      yourbox.lab.example.com
    t_beta: 0.5
    t_d: 0.28
    split: test
    fields_written: [beta, cluster_id, ...]
```

Two months from now you'll be able to open the HDF5 and answer "which
model produced these numbers?" without guessing.

> [!TIP]
> Pick a descriptive `--output-group` name per run (e.g.
> `predictions_v1`, `predictions_noisier_aug`, `predictions_epoch50`).
> The script errors if you'd overwrite an existing group unless you
> also pass `--overwrite`.

---

## Troubleshooting

### One loss term dominates `train/loss/total`

Your targets aren't in the same numeric range as everything else.
Example: if `energy_per_hit` is in GeV with values ~5.0, MSE at init
is ~25 per hit while the OC terms are O(1). Fix: normalize the target
in `__getitem__` (divide by a reasonable scale), then report back in
the original units in your evaluator.

### `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

Almost always `model.backbone.in_channels` in your runcard disagreeing
with the last dimension of `feat` in your `__getitem__`. Open the
runcard and the task side-by-side.

### `viz/grid` panel is blank / one solid color

Your `plot_truth` / `plot_pred` / `plot_oc` isn't returning a
well-formed `(H, W, 3)` uint8. Re-read Step 5 — the smoke test would
have caught this.

### Loss plateaus at a high value, never drops

Check the β histogram in TensorBoard. If it's stuck around 0.5 with
no bimodal structure after several epochs: your OC attractive /
repulsive terms aren't pushing. Try `q_min=0.5` or a smaller learning
rate. Verify `object_id` is globally unique across events.

### `Augmentation changed N but not every per-hit tensor`

You wrote an augmentation that subselects `coord` but forgot one of
your payload targets. Re-read
[`src/augmentations/README.md`](../src/augmentations/README.md) §
"Rules the framework relies on". The rule: if you change `N`, change
it for **every** per-hit tensor consistently.

---

## What to read next

* [`src/tasks/README.md`](../src/tasks/README.md) — the `OCTask`
  interface reference, once you've got the shape of it.
* [`src/augmentations/README.md`](../src/augmentations/README.md) —
  how to add train-time noise / rotations / dropout.
* [`examples/calorimeter/`](../examples/calorimeter/) — the full
  runnable example behind this tutorial.
* [`src/tasks/shapes.py`](../src/tasks/shapes.py) — a second, 2D
  reference task. Diff it against the calorimeter to see what's
  task-specific.
* [`docs/architecture.md`](architecture.md) — what PTv3 + OC is doing
  under the hood, if you want to know *why* the interface looks the
  way it does.
