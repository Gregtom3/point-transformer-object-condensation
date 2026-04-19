"""Generate a pseudo-dataset of geometric shapes on white canvases.

Each image contains a variable number of shapes (square, circle, triangle,
rectangle, ring) placed without overlap. Every non-white pixel becomes a
"hit" in the resulting point cloud. Each shape carries:

    * RGB color        (3 features)
    * unique_id        (globally unique across the entire dataset)
    * width, height    (2 features, per shape)
    * shape_id         (square=0, circle=1, triangle=2, rectangle=3, ring=4)

Output: three HDF5 files (``train.h5``, ``val.h5``, ``test.h5``) under
the chosen output directory, plus ``metadata.json`` summarising the run.

HDF5 layout per split:

    /event_000000/
        coord             (N, 3) float32    # (x, y, 0)
        feat              (N, 3) float32    # RGB in [0, 1]
        object_id         (N,)   int64      # 0 = background (none here; white is dropped)
        shape_id_per_hit  (N,)   int64
        width_per_hit     (N,)   float32
        height_per_hit    (N,)   float32
    /meta/ ... (frame_size, n_shape_classes, total_objects, split_name)

CLI usage (see ``--help`` for all options):

    python data/generate_shapes.py --out data/shapes_v1 \\
        --n-train 800 --n-val 100 --n-test 100 \\
        --frame 128 128 --shapes-per-image 1 6 \\
        --shape-size 10 40 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

SHAPE_NAMES = ["square", "circle", "triangle", "rectangle", "ring"]
SHAPE_ID = {name: i for i, name in enumerate(SHAPE_NAMES)}

WHITE = (255, 255, 255)


@dataclass
class GenConfig:
    out: Path
    n_train: int = 800
    n_val: int = 100
    n_test: int = 100
    frame: tuple[int, int] = (128, 128)
    shapes_per_image: tuple[int, int] = (1, 6)
    shape_size: tuple[int, int] = (10, 40)
    ring_thickness: tuple[int, int] = (2, 5)
    shapes: tuple[str, ...] = tuple(SHAPE_NAMES)
    max_place_attempts: int = 100
    # Minimum fraction of a new shape's pixels that must overlap with already-
    # placed shapes (ignored for the first shape in an image). Non-zero values
    # force dense, visually ambiguous packings that are harder to segment.
    # Overlapping pixels belong to the first-placed shape (FCFS); the new
    # shape only emits hits for pixels that weren't already claimed.
    min_overlap: float = 0.1
    seed: int = 0
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shape rasterizers: each returns a boolean mask of pixels that belong to the
# shape inside the given bbox (x0, y0, w, h). Masks are local to the bbox.
# ---------------------------------------------------------------------------

def _mask_square(w: int, h: int) -> np.ndarray:
    s = min(w, h)
    m = np.zeros((h, w), dtype=bool)
    m[:s, :s] = True
    return m


def _mask_rectangle(w: int, h: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mask_circle(w: int, h: int) -> np.ndarray:
    d = min(w, h)
    yy, xx = np.ogrid[:d, :d]
    r = d / 2
    mask = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 <= r * r
    out = np.zeros((h, w), dtype=bool)
    out[:d, :d] = mask
    return out


def _mask_ring(w: int, h: int, thickness: int) -> np.ndarray:
    d = min(w, h)
    yy, xx = np.ogrid[:d, :d]
    r = d / 2
    d2 = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2
    inner = max(1.0, r - thickness)
    mask = (d2 <= r * r) & (d2 >= inner * inner)
    out = np.zeros((h, w), dtype=bool)
    out[:d, :d] = mask
    return out


def _mask_triangle(w: int, h: int) -> np.ndarray:
    # isoceles triangle with apex at top-center, base along the bottom.
    m = np.zeros((h, w), dtype=bool)
    for row in range(h):
        frac = row / max(1, h - 1)  # 0 at top, 1 at bottom
        half_w = frac * (w / 2)
        cx = w / 2
        lo = int(math.ceil(cx - half_w))
        hi = int(math.floor(cx + half_w))
        lo = max(0, lo)
        hi = min(w - 1, hi)
        if hi >= lo:
            m[row, lo:hi + 1] = True
    return m


def make_shape_mask(shape: str, w: int, h: int, rng: np.random.Generator,
                    ring_thickness: tuple[int, int]) -> np.ndarray:
    if shape == "square":
        return _mask_square(w, h)
    if shape == "rectangle":
        return _mask_rectangle(w, h)
    if shape == "circle":
        return _mask_circle(w, h)
    if shape == "triangle":
        return _mask_triangle(w, h)
    if shape == "ring":
        t_lo, t_hi = ring_thickness
        t = int(rng.integers(t_lo, t_hi + 1))
        return _mask_ring(w, h, thickness=t)
    raise ValueError(f"unknown shape {shape}")


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def _random_color(rng: np.random.Generator) -> tuple[int, int, int]:
    # avoid pure white so the foreground/background separation stays clean
    while True:
        c = rng.integers(0, 255, size=3, dtype=np.int64)
        if tuple(int(x) for x in c) != WHITE:
            return int(c[0]), int(c[1]), int(c[2])


def _random_bbox(
    frame_w: int,
    frame_h: int,
    size_range: tuple[int, int],
    rng: np.random.Generator,
    is_rectangle: bool,
) -> tuple[int, int, int, int]:
    lo, hi = size_range
    if is_rectangle:
        w = int(rng.integers(lo, hi + 1))
        h = int(rng.integers(lo, hi + 1))
    else:
        s = int(rng.integers(lo, hi + 1))
        w = h = s
    w = min(w, frame_w)
    h = min(h, frame_h)
    x0 = int(rng.integers(0, frame_w - w + 1))
    y0 = int(rng.integers(0, frame_h - h + 1))
    return x0, y0, w, h


def generate_event(
    cfg: GenConfig,
    rng: np.random.Generator,
    next_object_id: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Generate a single image as a point-cloud dict. Returns the dict and
    the updated next ``next_object_id`` so IDs stay globally unique."""
    frame_w, frame_h = cfg.frame
    n_lo, n_hi = cfg.shapes_per_image
    n_shapes = int(rng.integers(n_lo, n_hi + 1))

    occupancy = np.zeros((frame_h, frame_w), dtype=bool)
    # per-hit buffers (filled as we place shapes)
    coord_list: list[np.ndarray] = []
    feat_list: list[np.ndarray] = []
    obj_list: list[np.ndarray] = []
    shape_id_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    h_list: list[np.ndarray] = []

    for shape_idx in range(n_shapes):
        placed = False
        for _ in range(cfg.max_place_attempts):
            shape = str(rng.choice(cfg.shapes))
            x0, y0, bw, bh = _random_bbox(
                frame_w, frame_h, cfg.shape_size, rng,
                is_rectangle=(shape == "rectangle"),
            )
            local = make_shape_mask(shape, bw, bh, rng, cfg.ring_thickness)
            region = occupancy[y0:y0 + bh, x0:x0 + bw]
            if region.shape != local.shape:  # edge clipping; retry
                continue
            new_claim = local & ~region
            overlap = local & region
            total = int(local.sum())
            n_new = int(new_claim.sum())
            if n_new == 0:
                # fully occluded by earlier shapes — nothing visible to emit
                continue
            if shape_idx > 0 and cfg.min_overlap > 0.0:
                # enforce dense packing: require enough shared pixels with
                # existing shapes. First shape always passes.
                if int(overlap.sum()) < cfg.min_overlap * total:
                    continue
            occupancy[y0:y0 + bh, x0:x0 + bw] |= local
            color = _random_color(rng)
            ys, xs = np.nonzero(new_claim)
            n_hits = xs.size
            if n_hits == 0:
                continue
            coord = np.stack(
                [xs + x0, ys + y0, np.zeros_like(xs)], axis=1
            ).astype(np.float32)
            feat = np.tile(
                np.array([c / 255.0 for c in color], dtype=np.float32), (n_hits, 1)
            )
            coord_list.append(coord)
            feat_list.append(feat)
            obj_list.append(np.full((n_hits,), next_object_id, dtype=np.int64))
            shape_id_list.append(np.full((n_hits,), SHAPE_ID[shape], dtype=np.int64))
            w_list.append(np.full((n_hits,), bw, dtype=np.float32))
            h_list.append(np.full((n_hits,), bh, dtype=np.float32))
            next_object_id += 1
            placed = True
            break
        if not placed:
            # couldn't fit another shape; stop trying for this image
            break

    if not coord_list:
        # degenerate: empty image. Emit a single background "hit" so HDF5
        # doesn't have a zero-length dataset (PTv3 wouldn't consume it
        # anyway; this path should be rare with sane configs).
        coord_list = [np.zeros((1, 3), dtype=np.float32)]
        feat_list = [np.ones((1, 3), dtype=np.float32)]
        obj_list = [np.zeros((1,), dtype=np.int64)]
        shape_id_list = [np.full((1,), -1, dtype=np.int64)]
        w_list = [np.zeros((1,), dtype=np.float32)]
        h_list = [np.zeros((1,), dtype=np.float32)]

    event = {
        "coord": np.concatenate(coord_list, axis=0),
        "feat": np.concatenate(feat_list, axis=0),
        "object_id": np.concatenate(obj_list, axis=0),
        "shape_id_per_hit": np.concatenate(shape_id_list, axis=0),
        "width_per_hit": np.concatenate(w_list, axis=0),
        "height_per_hit": np.concatenate(h_list, axis=0),
    }
    return event, next_object_id


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_split(path: Path, cfg: GenConfig, n_events: int,
                 rng: np.random.Generator, start_object_id: int,
                 split: str) -> tuple[int, int]:
    """Returns (next_object_id, n_shapes_written)."""
    total_shapes = 0
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["split"] = split
        meta.attrs["frame"] = np.array(cfg.frame, dtype=np.int64)
        meta.attrs["n_shape_classes"] = len(SHAPE_NAMES)
        meta.attrs["shape_names"] = np.array(SHAPE_NAMES, dtype="S16")
        meta.attrs["n_events"] = n_events
        obj_id = start_object_id
        for i in range(n_events):
            event, new_obj_id = generate_event(cfg, rng, obj_id)
            total_shapes += new_obj_id - obj_id
            obj_id = new_obj_id
            g = f.create_group(f"event_{i:06d}")
            for k, v in event.items():
                g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            g.attrs["n_hits"] = event["coord"].shape[0]
            g.attrs["n_shapes"] = int(np.unique(event["object_id"]).size)
        meta.attrs["object_id_range"] = np.array(
            [start_object_id, obj_id], dtype=np.int64
        )
    return obj_id, total_shapes


def parse_args() -> GenConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--frame", type=int, nargs=2, default=(128, 128),
                    metavar=("W", "H"), help="canvas width and height in pixels")
    ap.add_argument("--shapes-per-image", type=int, nargs=2, default=(1, 6),
                    metavar=("LO", "HI"), help="inclusive range of shapes per image")
    ap.add_argument("--shape-size", type=int, nargs=2, default=(10, 40),
                    metavar=("LO", "HI"), help="inclusive range of shape side lengths in pixels")
    ap.add_argument("--ring-thickness", type=int, nargs=2, default=(2, 5),
                    metavar=("LO", "HI"), help="range of ring thickness in pixels")
    ap.add_argument("--shapes", nargs="+", default=SHAPE_NAMES,
                    choices=SHAPE_NAMES, help="shape types to sample from")
    ap.add_argument("--max-place-attempts", type=int, default=100)
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=0.1,
        help="Minimum fraction of a new shape's area that must overlap "
             "existing shapes (0 disables, 0.1 default → dense packings).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return GenConfig(
        out=args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        frame=tuple(args.frame),
        shapes_per_image=tuple(args.shapes_per_image),
        shape_size=tuple(args.shape_size),
        ring_thickness=tuple(args.ring_thickness),
        shapes=tuple(args.shapes),
        max_place_attempts=args.max_place_attempts,
        min_overlap=args.min_overlap,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    next_obj = 1  # 0 reserved for background / noise
    stats = {}
    for split, n_events in [("train", cfg.n_train), ("val", cfg.n_val), ("test", cfg.n_test)]:
        path = cfg.out / f"{split}.h5"
        next_obj, n_shapes = _write_split(path, cfg, n_events, rng, next_obj, split)
        stats[split] = {"n_events": n_events, "n_shapes": int(n_shapes), "path": str(path)}
        print(f"[{split}] {n_events} events, {n_shapes} shapes -> {path}")

    stats["total_objects"] = int(next_obj - 1)
    stats["config"] = {
        "frame": list(cfg.frame),
        "shapes_per_image": list(cfg.shapes_per_image),
        "shape_size": list(cfg.shape_size),
        "ring_thickness": list(cfg.ring_thickness),
        "shapes": list(cfg.shapes),
        "min_overlap": cfg.min_overlap,
        "seed": cfg.seed,
    }
    meta_path = cfg.out / "metadata.json"
    meta_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
