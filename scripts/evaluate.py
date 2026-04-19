"""Evaluation entrypoint.

Loads a trained checkpoint, runs inference over the validation set, and
runs the OC clustering to produce per-hit cluster assignments that can
be compared against ground-truth particles.

Usage:
    python scripts/evaluate.py --config configs/train/default.yaml --checkpoint path/to.ckpt

TODO:
    * load checkpoint
    * compute per-event efficiency / purity / LRP against truth
    * dump plots / JSON summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import HitDataset, collate_hits
from src.inference.cluster import beta_threshold_cluster
from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train/default.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--t-beta", type=float, default=0.1)
    ap.add_argument("--t-d", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    dataset = HitDataset(**OmegaConf.to_container(cfg.data, resolve=True))
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_hits)

    backbone = PTv3Backbone(**OmegaConf.to_container(cfg.model.backbone, resolve=True))
    heads = ObjectCondensationHeads(
        in_dim=backbone.out_channels,
        **OmegaConf.to_container(cfg.model.heads, resolve=True),
    )
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        backbone.load_state_dict(ckpt["backbone"])
        heads.load_state_dict(ckpt["heads"])

    backbone.eval()
    heads.eval()
    with torch.no_grad():
        for batch in loader:
            data = {k: v for k, v in batch.items() if torch.is_tensor(v)}
            point = backbone(data)
            preds = heads(point["feat"])
            clusters = beta_threshold_cluster(
                preds["beta"], preds["x"], t_beta=args.t_beta, t_d=args.t_d
            )
            print(f"n_clusters={clusters.max().item()}")


if __name__ == "__main__":
    main()
