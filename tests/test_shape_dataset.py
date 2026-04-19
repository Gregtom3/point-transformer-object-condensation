"""Exercise the shapes generator + HDF5 dataset + loss on a tiny run.

No CUDA required — this skips the PTv3 forward pass entirely and just
verifies that the generated data flows through ShapeDataset, the OC
loss, and the head forward pass with the right shapes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _have(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


needs_h5py = pytest.mark.skipif(not _have("h5py"), reason="h5py not installed")


@pytest.fixture(scope="module")
def shapes_dir(tmp_path_factory) -> Path:
    if not _have("h5py"):
        pytest.skip("h5py not installed")
    out = tmp_path_factory.mktemp("shapes")
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, str(repo / "data" / "generate_shapes.py"),
        "--out", str(out),
        "--n-train", "3", "--n-val", "2", "--n-test", "2",
        "--frame", "48", "48",
        "--shapes-per-image", "2", "3",
        "--shape-size", "6", "16",
        "--seed", "0",
    ]
    subprocess.run(cmd, check=True)
    return out


@needs_h5py
def test_generator_produces_globally_unique_object_ids(shapes_dir):
    import h5py
    seen = set()
    for split in ("train.h5", "val.h5", "test.h5"):
        with h5py.File(shapes_dir / split, "r") as f:
            for key in f:
                if key == "meta":
                    continue
                ids = set(int(x) for x in f[key]["object_id"][:] if int(x) > 0)
                assert not (ids & seen), f"duplicate object id across dataset: {ids & seen}"
                seen.update(ids)
    assert len(seen) > 0


@needs_h5py
def test_shape_dataset_and_collate(shapes_dir):
    from src.data.shape_dataset import ShapeDataset, collate_shapes
    ds = ShapeDataset(shapes_dir / "train.h5", normalize_coords=True)
    batch = collate_shapes([ds[0], ds[1]])
    assert batch["coord"].ndim == 2 and batch["coord"].shape[1] == 3
    assert batch["feat"].shape == (batch["coord"].shape[0], 3)
    assert batch["offset"].shape == (2,)
    assert batch["offset"][-1] == batch["coord"].shape[0]
    assert batch["coord"].max() <= 1.0 and batch["coord"].min() >= 0.0
    assert (batch["object_id"] >= 0).all()


@needs_h5py
def test_heads_and_loss_on_shapes_batch(shapes_dir):
    from src.data.shape_dataset import ShapeDataset, collate_shapes
    from src.losses.oc_loss import ObjectCondensationLoss
    from src.models.heads import ObjectCondensationHeads

    ds = ShapeDataset(shapes_dir / "train.h5", normalize_coords=True)
    batch = collate_shapes([ds[0]])
    n = batch["coord"].shape[0]
    feat = torch.randn(n, 32, requires_grad=True)
    heads = ObjectCondensationHeads(
        in_dim=32, cluster_dim=3, n_pid_classes=5,
        predict_width_height=True, predict_momentum=False,
    )
    preds = heads(feat)
    assert preds["beta"].shape == (n,)
    assert preds["x"].shape == (n, 3)
    assert preds["pid_logits"].shape == (n, 5)
    assert preds["width"].shape == (n,)
    assert preds["height"].shape == (n,)

    loss_fn = ObjectCondensationLoss(q_min=1.0, payload_weight=1.0)
    losses = loss_fn(preds, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
