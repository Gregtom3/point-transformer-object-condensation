"""Training entrypoint.

Currently a no-op stub that loads a Hydra/OmegaConf config and runs a
short loop over the dummy dataset so the plumbing can be checked
end-to-end. Replace the dataset with a real one and the Trainer with a
real training harness when ready.

Usage:
    python scripts/train.py --config configs/train/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import HitDataset, collate_hits
from src.losses.oc_loss import ObjectCondensationLoss
from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads
from src.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train/default.yaml"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    dataset = HitDataset(**OmegaConf.to_container(cfg.data, resolve=True))
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_hits,
        num_workers=0,
    )

    backbone = PTv3Backbone(**OmegaConf.to_container(cfg.model.backbone, resolve=True))
    heads = ObjectCondensationHeads(
        in_dim=backbone.out_channels,
        **OmegaConf.to_container(cfg.model.heads, resolve=True),
    )
    loss_fn = ObjectCondensationLoss(**OmegaConf.to_container(cfg.loss, resolve=True))

    trainer = Trainer(
        backbone, heads, loss_fn,
        config=TrainerConfig(**OmegaConf.to_container(cfg.train.trainer, resolve=True)),
    )
    trainer.fit(loader)


if __name__ == "__main__":
    main()
