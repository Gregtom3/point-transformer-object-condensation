"""HDF5-backed dataset for the shapes pseudo-dataset.

One event per HDF5 group (see ``data/generate_shapes.py``). We lazily
open the file per worker so multi-process DataLoaders work, and read a
full event at ``__getitem__``.

Collation produces a single flat batch in PTv3's ``coord``/``feat``/
``offset`` layout, with all truth tensors concatenated in the same order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):
    """Reads a single split's HDF5 file produced by ``generate_shapes.py``.

    Args:
        path: path to ``train.h5`` / ``val.h5`` / ``test.h5``.
        normalize_coords: if True, divide (x, y) by frame size so coords
            live in [0, 1]. Helps PTv3's voxelization with small grid_size.
        max_hits: optional cap; events larger than this are randomly
            subsampled (useful for memory). ``0`` disables.
    """

    def __init__(
        self,
        path: str | Path,
        normalize_coords: bool = True,
        max_hits: int = 0,
    ) -> None:
        self.path = Path(path)
        self.normalize_coords = normalize_coords
        self.max_hits = max_hits

        # Open once to read meta; actual per-event reads happen lazily.
        with h5py.File(self.path, "r") as f:
            self.n_events = int(f["meta"].attrs["n_events"])
            self.frame = tuple(int(x) for x in f["meta"].attrs["frame"])
            self.n_shape_classes = int(f["meta"].attrs["n_shape_classes"])
            self.shape_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f["meta"].attrs["shape_names"]
            ]
            self.object_id_range = tuple(
                int(x) for x in f["meta"].attrs["object_id_range"]
            )
        self._h5: h5py.File | None = None

    def _file(self) -> h5py.File:
        # open lazily so DataLoader workers each get their own handle
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        g = self._file()[f"event_{idx:06d}"]
        coord = np.asarray(g["coord"])
        feat = np.asarray(g["feat"])
        object_id = np.asarray(g["object_id"])
        shape_id = np.asarray(g["shape_id_per_hit"])
        width = np.asarray(g["width_per_hit"])
        height = np.asarray(g["height_per_hit"])

        if self.max_hits and coord.shape[0] > self.max_hits:
            sel = np.random.choice(coord.shape[0], size=self.max_hits, replace=False)
            sel.sort()
            coord = coord[sel]
            feat = feat[sel]
            object_id = object_id[sel]
            shape_id = shape_id[sel]
            width = width[sel]
            height = height[sel]

        if self.normalize_coords:
            fw, fh = self.frame
            coord = coord.copy()
            coord[:, 0] /= float(fw)
            coord[:, 1] /= float(fh)

        return {
            "coord": torch.from_numpy(coord).float(),
            "feat": torch.from_numpy(feat).float(),
            "object_id": torch.from_numpy(object_id).long(),
            "shape_id_per_hit": torch.from_numpy(shape_id).long(),
            "width_per_hit": torch.from_numpy(width).float(),
            "height_per_hit": torch.from_numpy(height).float(),
            "frame": torch.tensor(self.frame, dtype=torch.long),
        }


def collate_shapes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Concat events along the hit axis and emit PTv3's offset layout."""
    keys_flat = [
        "coord",
        "feat",
        "object_id",
        "shape_id_per_hit",
        "width_per_hit",
        "height_per_hit",
    ]
    out: dict[str, Any] = {k: [] for k in keys_flat}
    sizes: list[int] = []
    frames: list[torch.Tensor] = []
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
