# Toy calorimeter task

A second, **physics-flavored** reference task. Read this alongside the
shapes pseudo-dataset to understand the `OCTask` interface in two
different settings.

**The setup.** Two calorimeter walls at `x = ±50` (left / right). Each
event has a handful of particles, each firing at one wall and
depositing energy in a small cluster of cells (Gaussian shower). You
want to:

1. cluster the cells back into their source particles (**OC**), and
2. predict each particle's total energy (**per-hit regression**).

## Contents

| File | What it is |
|---|---|
| [`generate.py`](generate.py) | Toy data generator → HDF5 with per-hit coord / feat / object_id / energy_per_hit |
| [`task.py`](task.py) | `CalorimeterTask(OCTask)` — reads the HDF5 and renders the three panels |
| [`config.yaml`](config.yaml) | Runcard (copy of `configs/train/shapes.yaml`, edited for this task) |

## Try it

```bash
# 1. generate a tiny dataset (< 1 second; writes under data/calo_v1/)
python examples/calorimeter/generate.py --out data/calo_v1

# 2. train (not many steps — just proving the wiring)
python scripts/train.py --config examples/calorimeter/config.yaml

# 3. evaluate
python scripts/evaluate.py --config examples/calorimeter/config.yaml \
    --checkpoint outputs/calo/epoch_009.pt --split test

# 4. append predictions back to the HDF5 with provenance
python scripts/export_predictions.py \
    --config examples/calorimeter/config.yaml \
    --checkpoint outputs/calo/epoch_009.pt \
    --split test --output-group predictions_v1
```

## Where to look when building your own task

* [`../../docs/custom_task.md`](../../docs/custom_task.md) — the
  step-by-step tutorial that uses this calorimeter as its running
  example.
* [`../../src/tasks/README.md`](../../src/tasks/README.md) — the
  `OCTask` interface reference, contracts for every method.
* [`../../src/tasks/shapes.py`](../../src/tasks/shapes.py) — a second
  reference implementation (2D image-like data instead of 3D).

## Registering the task

`scripts/train.py` looks up tasks by string name from
`src/tasks/__init__.py`. The calorimeter task is re-exported there as
`CalorimeterTask`. If you want your runcard to pick a task by name
instead of hard-coding `ShapesTask` in `scripts/train.py`, see the
"wiring your task into train / eval" section of the tutorial.
