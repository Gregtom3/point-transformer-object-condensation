# `src/augmentations/` — per-event augmentations

Augmentations are callables that run inside `Task.__getitem__` to
produce a new view of each event every epoch. They're how you add
training-time noise / rotations / dropout without touching your data
files.

> [!NOTE]
> Augmentations only run on the training split. Pass `augmentation=None`
> (the default) to your val / test task so evaluation is deterministic.

## The contract

Every augmentation is a callable:

```python
def __call__(self, event: dict[str, Any]) -> dict[str, Any]: ...
```

`event` is the dict your task's `__getitem__` just built. The
augmentation returns an event with the same keys.

Rules the framework relies on:

| # | Rule |
|---|------|
| 1 | **Keep the key set.** If `event` had `coord`, `feat`, `object_id`, the return must still have all of them. Adding extra keys is allowed. |
| 2 | **Keep per-hit tensors aligned.** Every per-hit tensor must share the first dimension `N` on return. If you drop a hit you must drop it from every per-hit tensor — `coord`, `feat`, `object_id`, `shape_id_per_hit`, `width_per_hit`, `height_per_hit`, etc. |
| 3 | **Don't touch non-hit tensors.** `frame`, `batch_size`, and any other metadata-shaped tensor must pass through unchanged. |
| 4 | **Be picklable.** DataLoader workers pickle the task (and its augmentation) to each worker process. Store RNGs / config as plain fields; don't close over unpicklable objects. |
| 5 | **Be cheap.** One augmentation call runs once per sample per epoch. Expensive ops (big FFTs, network calls) will bottleneck training. |

Shape-changing augmentations are fine — a dropout that turns `N=500`
into `N=450` is valid as long as all per-hit tensors are subselected
the same way. See [`RandomHitDropout`](basic.py) for the reference
implementation.

## Composition

`Compose([A, B, C])(event)` runs `A` then `B` then `C`:

```python
from src.augmentations import Compose, RandomRotation2D, RandomColorJitter, RandomHitDropout

aug = Compose([
    RandomRotation2D(max_angle_deg=20),
    RandomColorJitter(sigma=0.02, shape_sigma=0.05),
    RandomHitDropout(p=0.1, min_keep=16),
])
```

Identity (`Identity()`) is the do-nothing augmentation — useful as a
default off-switch or as a placeholder in a factory.

## Built-in augmentations

| Class | What it does | Relies on |
|---|---|---|
| [`RandomRotation2D`](basic.py) | Rotates `coord[:, :2]` by a uniform angle around (0.5, 0.5). | 2D (z unused), `coord` in [0, 1] units. |
| [`RandomColorJitter`](basic.py) | Adds Gaussian noise to `feat`, plus an optional per-shape offset so all hits of one shape drift together. | 3-channel `feat`, values in [0, 1]. |
| [`RandomHitDropout`](basic.py) | Randomly drops a fraction of hits; keeps every per-hit tensor aligned. | None (works for any event). |

## Writing your own

Minimal skeleton:

```python
import numpy as np
import torch
from src.augmentations.base import Augmentation


class MyAugmentation(Augmentation):
    def __init__(self, strength: float = 0.1, seed: int | None = None) -> None:
        self.strength = float(strength)
        self._rng = np.random.default_rng(seed)

    def __call__(self, event):
        # Do not mutate the input in place — callers may be holding the
        # original dict. Shallow-copy first.
        event = dict(event)

        # Modify one or more per-hit tensors. Example: jitter coord.
        coord = event["coord"]
        noise = self._rng.normal(0, self.strength, size=coord.shape).astype("float32")
        event["coord"] = coord + torch.from_numpy(noise)
        return event

    def __repr__(self) -> str:
        return f"MyAugmentation(strength={self.strength})"
```

Attach it to your task:

```python
from src.tasks import ShapesTask
train_task = ShapesTask("data/shapes_v1/train.h5", augmentation=MyAugmentation(0.05))
val_task   = ShapesTask("data/shapes_v1/val.h5")  # no augmentation
```

> [!IMPORTANT]
> Augmentations that change `coord` **must preserve the normalization
> convention** the rest of the pipeline assumes. `ShapesTask` normalizes
> `coord` to [0, 1]; a rotation / jitter that pushes coords outside
> that range will break PTv3's voxelization at small `grid_size`. Clamp
> or reject the sample.

## Testing a new augmentation

A useful smoke test:

```python
import torch
from src.tasks import ShapesTask

task = ShapesTask("data/shapes_v1/train.h5", augmentation=MyAugmentation())
ev = task[0]
# per-hit tensors must all agree on N
Ns = {k: v.shape[0] for k, v in ev.items()
      if torch.is_tensor(v) and v.dim() >= 1 and k not in ("frame",)}
assert len(set(Ns.values())) == 1, f"per-hit N mismatch: {Ns}"

# forward through the collate
batch = task.collate([task[i] for i in range(4)])
assert batch["coord"].shape[0] == batch["offset"][-1]
```

If you add it to a training run, eyeball the `viz/grid` image in
TensorBoard for the first few epochs — an augmentation that silently
corrupts object ids shows up immediately as scrambled colors.
