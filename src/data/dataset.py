"""Hit-level dataset for multi-subdetector particle reconstruction.

Each event is a heterogeneous set of detector hits originating from
different subsystems (tracker, calorimeter, RICH, ...). Hits from each
subdetector have their own native features, so we embed them with
per-subdetector encoders before concatenating them into a single token
stream that is consumed by the PTv3 backbone.

The expected per-event sample is a dict with:
    coord:        (N, 3) float tensor of 3D positions (mm or cm, consistent units)
    feat:         (N, F) float tensor of per-hit features (may be padded per subdetector)
    subdet_id:    (N,)   long tensor with subdetector type (0=tracker, 1=ecal, ...)
    object_id:    (N,)   long tensor with ground-truth particle id (<=0 means noise)
    truth:        dict of per-object payload targets, e.g. energy, momentum, pid
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


SUBDET_TRACKER = 0
SUBDET_ECAL = 1
SUBDET_HCAL = 2
SUBDET_RICH = 3

SUBDET_NAMES = {
    SUBDET_TRACKER: "tracker",
    SUBDET_ECAL: "ecal",
    SUBDET_HCAL: "hcal",
    SUBDET_RICH: "rich",
}


class HitDataset(Dataset):
    """Placeholder hit-level dataset.

    TODO: wire this up to a real data source (ROOT files via uproot,
    awkward arrays, HDF5, ...). For now it just returns dummy tensors so
    the rest of the pipeline can be exercised.
    """

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        n_events: int = 8,
        n_hits: int = 500,
        n_subdetectors: int = 4,
        n_features: int = 6,
    ) -> None:
        self.root = root
        self.split = split
        self.n_events = n_events
        self.n_hits = n_hits
        self.n_subdetectors = n_subdetectors
        self.n_features = n_features

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(idx)
        n = self.n_hits
        coord = torch.rand((n, 3), generator=g)
        feat = torch.randn((n, self.n_features), generator=g)
        subdet_id = torch.randint(0, self.n_subdetectors, (n,), generator=g)
        n_objects = 8
        object_id = torch.randint(0, n_objects + 1, (n,), generator=g)  # 0 => noise
        truth = {
            "energy": torch.rand((n_objects,), generator=g) * 10.0,
            "momentum": torch.randn((n_objects, 3), generator=g),
            "pid": torch.randint(0, 5, (n_objects,), generator=g),
        }
        return {
            "coord": coord,
            "feat": feat,
            "subdet_id": subdet_id,
            "object_id": object_id,
            "truth": truth,
        }


def collate_hits(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate per-event hit tensors into PTv3's offset-batched layout.

    PTv3 expects a single flat `coord`/`feat` with an `offset` tensor of
    cumulative sizes.
    """
    coords, feats, subdet_ids, object_ids, sizes, truths = [], [], [], [], [], []
    for item in batch:
        coords.append(item["coord"])
        feats.append(item["feat"])
        subdet_ids.append(item["subdet_id"])
        object_ids.append(item["object_id"])
        sizes.append(item["coord"].shape[0])
        truths.append(item["truth"])
    offset = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)
    return {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "subdet_id": torch.cat(subdet_ids, dim=0),
        "object_id": torch.cat(object_ids, dim=0),
        "offset": offset,
        "truth": truths,
    }
