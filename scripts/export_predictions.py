"""Run a trained model over a split and append its per-hit predictions
back into the source HDF5 file, along with a ``predictions/`` metadata
group that records what model / config / timestamp produced them.

This is the reference pattern for "my pipeline produced features; I want
to keep them alongside the input data for later analysis without
re-running inference". Copy and adapt for task-specific outputs — the
set of per-hit prediction keys written here (``beta``, ``cluster_id``,
``pid_pred``, ``width_pred``, ``height_pred``) is arbitrary.

Usage:
    python scripts/export_predictions.py --config configs/train/shapes.yaml \\
        --checkpoint outputs/shapes/epoch_009.pt --split test \\
        [--output-group predictions] [--overwrite]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.inference.cluster import beta_threshold_cluster
from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads
from src.tasks import ShapesTask


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _write_predictions(
    group: h5py.Group,
    *,
    beta: np.ndarray,
    cluster_id: np.ndarray,
    pid_pred: np.ndarray,
    width_pred: np.ndarray,
    height_pred: np.ndarray,
    x_cluster: np.ndarray,
    overwrite: bool,
) -> None:
    """Write per-hit predictions as gzip-compressed datasets."""
    fields = {
        "beta": beta,
        "cluster_id": cluster_id,
        "pid_pred": pid_pred,
        "width_pred": width_pred,
        "height_pred": height_pred,
        "x_cluster": x_cluster,
    }
    for k, v in fields.items():
        if k in group:
            if not overwrite:
                raise FileExistsError(
                    f"dataset {group.name}/{k} already exists (pass --overwrite to replace)"
                )
            del group[k]
        group.create_dataset(k, data=v, compression="gzip", compression_opts=4)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train/shapes.yaml"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--output-group", type=str, default="predictions",
                    help="subgroup inside each event_* group to write into")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing predictions/* datasets")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_file = getattr(cfg.data, f"{args.split}_file")
    h5_path = Path(cfg.data.root) / split_file

    task = ShapesTask(
        h5_path, normalize_coords=cfg.data.normalize_coords,
        max_hits=cfg.data.max_hits,
    )
    loader = DataLoader(task, batch_size=1, collate_fn=task.collate)

    backbone = PTv3Backbone(**OmegaConf.to_container(cfg.model.backbone, resolve=True)).to(device)
    heads = ObjectCondensationHeads(
        in_dim=backbone.out_channels,
        **OmegaConf.to_container(cfg.model.heads, resolve=True),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    heads.load_state_dict(ckpt["heads"])
    backbone.eval()
    heads.eval()

    t_beta = float(cfg.inference.t_beta)
    t_d = float(cfg.inference.t_d)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # Write predictions into the HDF5 file in-place. swmr=False and "a"
    # so writers can create groups.
    with h5py.File(h5_path, "a") as f, torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            data = {"coord": batch["coord"], "feat": batch["feat"],
                    "offset": batch["offset"]}
            point = backbone(data)
            preds = heads(point["feat"])
            cluster = beta_threshold_cluster(
                preds["beta"], preds["x"], t_beta=t_beta, t_d=t_d,
            )

            event_group = f[f"event_{i:06d}"]
            if args.output_group in event_group:
                if not args.overwrite:
                    raise FileExistsError(
                        f"event_{i:06d}/{args.output_group} already exists; "
                        f"pass --overwrite to replace"
                    )
                del event_group[args.output_group]
            pgroup = event_group.create_group(args.output_group)
            _write_predictions(
                pgroup,
                beta=preds["beta"].cpu().numpy().astype(np.float32),
                cluster_id=cluster.cpu().numpy().astype(np.int64),
                pid_pred=preds["pid_logits"].argmax(-1).cpu().numpy().astype(np.int64),
                width_pred=preds["width"].cpu().numpy().astype(np.float32),
                height_pred=preds["height"].cpu().numpy().astype(np.float32),
                x_cluster=preds["x"].cpu().numpy().astype(np.float32),
                overwrite=args.overwrite,
            )
            pgroup.attrs["t_beta"] = t_beta
            pgroup.attrs["t_d"] = t_d

        # Global provenance: one shared group at the file root.
        meta_root = f.require_group(f"predictions_meta/{args.output_group}")
        meta_root.attrs["checkpoint"] = str(args.checkpoint.resolve())
        meta_root.attrs["config_path"] = str(args.config.resolve())
        meta_root.attrs["config_yaml"] = OmegaConf.to_yaml(cfg)
        meta_root.attrs["timestamp_utc"] = timestamp
        meta_root.attrs["git_sha"] = _git_sha()
        meta_root.attrs["hostname"] = socket.gethostname()
        meta_root.attrs["t_beta"] = t_beta
        meta_root.attrs["t_d"] = t_d
        meta_root.attrs["split"] = args.split

    summary = {
        "h5_file": str(h5_path),
        "output_group": args.output_group,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "t_beta": t_beta,
        "t_d": t_d,
        "timestamp_utc": timestamp,
        "git_sha": _git_sha(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
