"""Toy left/right calorimeter data generator.

Each event has ``n_particles`` particles; each particle fires at a
random (y, z) on one of the two calorimeter walls (``x = -wall_x`` or
``x = +wall_x``) and deposits energy in a small Gaussian shower of
cells around its hit point. Every cell becomes one "hit":

    coord           (N, 3)  float32    # (x, y, z) in cm
    feat            (N, 3)  float32    # (log10(E_cell), t, subdet_id)
    object_id       (N,)    int64      # globally unique particle id (>= 1)
    energy_per_hit  (N,)    float32    # broadcast per-particle total GeV

Data is written to ``<out>/train.h5`` / ``val.h5`` / ``test.h5`` using
the same layout as the shapes generator, so ``CalorimeterTask`` can
consume it without any plumbing tricks.

Usage:
    python examples/calorimeter/generate.py --out data/calo_v1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


SUBDET_LEFT = 0
SUBDET_RIGHT = 1


@dataclass
class GenConfig:
    out: Path
    n_train: int = 400
    n_val: int = 50
    n_test: int = 50
    n_particles: tuple[int, int] = (3, 7)     # inclusive range per event
    wall_x: float = 50.0                      # cm, |x| of each calorimeter wall
    wall_yz_half: float = 40.0                # cm, half-extent of the y/z wall
    shower_sigma: float = 2.0                 # cm, Gaussian width of a shower
    shower_cells: tuple[int, int] = (6, 14)   # cells per shower (inclusive)
    energy_range: tuple[float, float] = (0.5, 5.0)  # GeV per particle
    time_jitter: float = 0.3                  # ns, per-cell time noise
    seed: int = 0
    stats: dict = field(default_factory=dict)


def _one_particle(
    rng: np.random.Generator, cfg: GenConfig, pid: int,
) -> dict[str, np.ndarray]:
    """Generate the hits for a single particle. Returns a dict of (M, ...) arrays."""
    side = int(rng.integers(0, 2))
    x_wall = -cfg.wall_x if side == SUBDET_LEFT else +cfg.wall_x

    # center of the shower on the wall
    y0 = float(rng.uniform(-cfg.wall_yz_half, cfg.wall_yz_half))
    z0 = float(rng.uniform(-cfg.wall_yz_half, cfg.wall_yz_half))

    # total energy & arrival time
    e_total = float(rng.uniform(*cfg.energy_range))
    t0 = float(rng.uniform(0.0, 5.0))  # event-relative ns

    # sample cell positions as a Gaussian blob around (y0, z0)
    n_cells = int(rng.integers(cfg.shower_cells[0], cfg.shower_cells[1] + 1))
    yz = rng.normal(loc=[y0, z0], scale=cfg.shower_sigma, size=(n_cells, 2))
    yz = np.clip(yz, -cfg.wall_yz_half, cfg.wall_yz_half)

    # split the total energy across cells (Dirichlet so fractions sum to 1)
    fracs = rng.dirichlet(np.ones(n_cells))
    e_cells = (fracs * e_total).astype(np.float32)

    t_cells = (t0 + rng.normal(0.0, cfg.time_jitter, size=n_cells)).astype(np.float32)

    coord = np.stack(
        [np.full(n_cells, x_wall, dtype=np.float32),
         yz[:, 0].astype(np.float32),
         yz[:, 1].astype(np.float32)], axis=1,
    )
    feat = np.stack(
        [np.log10(np.clip(e_cells, 1e-4, None)),  # log energy — PTv3 likes bounded
         t_cells,
         np.full(n_cells, float(side), dtype=np.float32)],
        axis=1,
    )
    object_id = np.full(n_cells, pid, dtype=np.int64)
    energy_per_hit = np.full(n_cells, e_total, dtype=np.float32)
    return {
        "coord": coord, "feat": feat,
        "object_id": object_id, "energy_per_hit": energy_per_hit,
    }


def generate_event(rng: np.random.Generator, cfg: GenConfig,
                   next_object_id: int) -> tuple[dict[str, np.ndarray], int]:
    n_p = int(rng.integers(cfg.n_particles[0], cfg.n_particles[1] + 1))
    parts: list[dict[str, np.ndarray]] = []
    for _ in range(n_p):
        p = _one_particle(rng, cfg, pid=next_object_id)
        next_object_id += 1
        parts.append(p)
    event = {
        k: np.concatenate([p[k] for p in parts], axis=0)
        for k in ("coord", "feat", "object_id", "energy_per_hit")
    }
    return event, next_object_id


def _write_split(path: Path, cfg: GenConfig, n_events: int,
                 rng: np.random.Generator, start_object_id: int,
                 split: str) -> tuple[int, int]:
    total_particles = 0
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["split"] = split
        meta.attrs["n_events"] = n_events
        meta.attrs["wall_x"] = cfg.wall_x
        meta.attrs["wall_yz_half"] = cfg.wall_yz_half
        # ``frame`` is shaped like the shapes generator's so plot_* can reuse
        # the same idea of a source canvas size for scaling.
        meta.attrs["frame"] = np.array(
            [int(2 * cfg.wall_yz_half), int(2 * cfg.wall_yz_half)],
            dtype=np.int64,
        )
        obj_id = start_object_id
        for i in range(n_events):
            event, new_obj_id = generate_event(rng, cfg, obj_id)
            total_particles += new_obj_id - obj_id
            obj_id = new_obj_id
            g = f.create_group(f"event_{i:06d}")
            for k, v in event.items():
                g.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            g.attrs["n_hits"] = event["coord"].shape[0]
            g.attrs["n_particles"] = int(np.unique(event["object_id"]).size)
        meta.attrs["object_id_range"] = np.array(
            [start_object_id, obj_id], dtype=np.int64,
        )
    return obj_id, total_particles


def parse_args() -> GenConfig:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-particles", type=int, nargs=2, default=(3, 7),
                    metavar=("LO", "HI"))
    ap.add_argument("--wall-x", type=float, default=50.0)
    ap.add_argument("--wall-yz-half", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return GenConfig(
        out=args.out,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_particles=tuple(args.n_particles),
        wall_x=args.wall_x, wall_yz_half=args.wall_yz_half, seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    cfg.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    next_obj = 1  # 0 reserved for noise
    stats: dict = {}
    for split, n_events in [("train", cfg.n_train), ("val", cfg.n_val),
                            ("test", cfg.n_test)]:
        path = cfg.out / f"{split}.h5"
        next_obj, n_parts = _write_split(path, cfg, n_events, rng, next_obj, split)
        stats[split] = {"n_events": n_events, "n_particles": int(n_parts),
                        "path": str(path)}
        print(f"[{split}] {n_events} events, {n_parts} particles -> {path}")

    stats["total_particles"] = int(next_obj - 1)
    stats["config"] = {
        "n_particles": list(cfg.n_particles),
        "wall_x": cfg.wall_x,
        "wall_yz_half": cfg.wall_yz_half,
        "seed": cfg.seed,
    }
    (cfg.out / "metadata.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {cfg.out / 'metadata.json'}")


if __name__ == "__main__":
    main()
