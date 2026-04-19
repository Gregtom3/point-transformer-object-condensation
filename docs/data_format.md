# Data format

This document describes exactly what the training loop expects a batch
to contain. Any dataset that produces the keys below with the right
shapes will work with `scripts/train.py`.

## Event-level schema

An "event" is a single point cloud (one image for the shapes
pseudo-dataset, one physics event for the detector case). Per-event
tensors:

| key | shape | dtype | meaning |
|---|---|---|---|
| `coord` | `(N, 3)` | `float32` | 3D hit coordinates. For 2D data (shapes), set `z=0`. |
| `feat` | `(N, F)` | `float32` | Per-hit input features (RGB for shapes, native subdetector features for physics). |
| `object_id` | `(N,)` | `int64` | Truth cluster id. **Globally unique across the whole dataset.** `0` means background/noise. |
| `shape_id_per_hit` | `(N,)` | `int64` | Per-hit shape/particle class. Values `< 0` are ignored by the loss. |
| `width_per_hit` | `(N,)` | `float32` | Shape's bbox width, broadcast to every hit in that shape. |
| `height_per_hit` | `(N,)` | `float32` | Shape's bbox height, broadcast to every hit in that shape. |
| `frame` | `(2,)` | `int64` | `(W, H)` of the original canvas — used only for TB image logging. |

## Batch-level schema (after `collate_shapes`)

PTv3 expects a single flat point cloud per batch, with an `offset`
tensor giving the cumulative sizes:

| key | shape | dtype | meaning |
|---|---|---|---|
| `coord` | `(sum(N_b), 3)` | `float32` | All events concatenated along the hit axis. |
| `feat` | `(sum(N_b), F)` | `float32` | Same. |
| `offset` | `(B,)` | `int64` | Cumulative sum of per-event hit counts — event `b` occupies `coord[offset[b-1]:offset[b]]`. |
| `object_id`, `shape_id_per_hit`, `width_per_hit`, `height_per_hit` | `(sum(N_b), ...)` | matches event | Concatenated truth. |
| `frame` | `(B, 2)` | `int64` | Stacked per-event frame sizes. |

## Required invariants

1. **`object_id` global uniqueness.** Two different shapes anywhere in
   the dataset must not share an `object_id`. Object Condensation's
   repulsive term uses these labels to decide which pairs of hits must
   be pushed apart; reusing an id merges those shapes into one cluster.
2. **`object_id = 0` ⇒ background / noise.** The OC loss skips these
   hits when `noise_threshold=0` (the default). If your dataset has
   explicit noise hits, give them `object_id = 0`.
3. **Coordinates should live in a bounded range.** PTv3 voxelizes via
   `grid_coord = floor((coord - coord.min()) / grid_size)` and then
   builds a sparse tensor of that shape. Normalize pixel/distance units
   so `grid_size` produces a reasonable voxel count (O(10^2–10^3) per
   axis). For shapes we divide pixel coords by frame size, putting them
   in `[0, 1]`.

## HDF5 on-disk layout (shapes dataset)

Produced by `data/generate_shapes.py`:

```
train.h5
├── meta/ (attrs: split, frame, n_shape_classes, shape_names, n_events, object_id_range)
├── event_000000/
│   ├── coord                (N, 3)  float32
│   ├── feat                 (N, 3)  float32
│   ├── object_id            (N,)    int64
│   ├── shape_id_per_hit     (N,)    int64
│   ├── width_per_hit        (N,)    float32
│   └── height_per_hit       (N,)    float32
├── event_000001/ ...
```

Each dataset is gzip-compressed (level 4). `ShapeDataset` opens the
file lazily per DataLoader worker via `swmr` mode, so multi-worker
loading is safe.

## Wiring a real detector dataset

Implement a `torch.utils.data.Dataset` whose `__getitem__` returns the
event-level dict above. Keep:

- `coord` in the units your `grid_size` expects.
- `feat` sized to whatever `model.backbone.in_channels` is set to.
- `object_id` globally unique, `0` reserved for noise.

Everything else (loss, inference, metrics) is already generic.
