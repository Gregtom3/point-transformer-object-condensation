"""Training entrypoint.

Loads an OmegaConf config, builds the PTv3 backbone + OC heads + loss,
writes an architecture report into ``<log_dir>/architecture/``, and runs
the training loop with TensorBoard logging.

Usage:
    python scripts/train.py --config configs/train/shapes.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.shape_dataset import ShapeDataset, collate_shapes
from src.losses.oc_loss import ObjectCondensationLoss
from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads
from src.training.trainer import TBConfig, Trainer, TrainerConfig
from src.utils.model_summary import count_params, write_architecture_report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train/shapes.yaml"))
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    return ap.parse_args()


def _loader(ds: ShapeDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_shapes,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    torch.manual_seed(int(cfg.train.seed))

    root = Path(cfg.data.root)
    train_ds = ShapeDataset(
        root / cfg.data.train_file,
        normalize_coords=cfg.data.normalize_coords,
        max_hits=cfg.data.max_hits,
    )
    val_ds = ShapeDataset(
        root / cfg.data.val_file,
        normalize_coords=cfg.data.normalize_coords,
        max_hits=cfg.data.max_hits,
    )
    train_loader = _loader(train_ds, cfg.train.batch_size, True, cfg.train.num_workers)
    val_loader = _loader(val_ds, cfg.train.batch_size, False, cfg.train.num_workers)

    backbone = PTv3Backbone(**OmegaConf.to_container(cfg.model.backbone, resolve=True))
    heads = ObjectCondensationHeads(
        in_dim=backbone.out_channels,
        **OmegaConf.to_container(cfg.model.heads, resolve=True),
    )
    loss_fn = ObjectCondensationLoss(**OmegaConf.to_container(cfg.loss, resolve=True))

    trainer = Trainer(
        backbone, heads, loss_fn,
        config=TrainerConfig(**OmegaConf.to_container(cfg.train.trainer, resolve=True)),
        tb_config=TBConfig(
            log_dir=cfg.train.log_dir,
            image_every=cfg.train.tb_image_every,
            embedding_every=cfg.train.tb_embedding_every,
            inference_t_beta=cfg.inference.t_beta,
            inference_t_d=cfg.inference.t_d,
        ),
        ckpt_dir=cfg.train.ckpt_dir,
    )

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        backbone.load_state_dict(ckpt["backbone"])
        heads.load_state_dict(ckpt["heads"])
        if "optim" in ckpt:
            trainer.optim.load_state_dict(ckpt["optim"])
        trainer._step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} at step {trainer._step}")

    # Architecture report + config dump into the log dir
    log_dir = Path(cfg.train.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / "config.yaml")

    # Head summary only: full PTv3 + spconv doesn't play nicely with
    # torchinfo's tracing, but the heads are pure MLPs and export fine.
    n_total, n_train = count_params(torch.nn.Sequential(backbone, heads))
    print(f"parameters: {n_total:,} total ({n_train:,} trainable)")
    write_architecture_report(
        heads,
        out_dir=log_dir / "architecture",
        example_input=torch.zeros(4, backbone.out_channels),
        extra_note=(
            f"Full model (backbone + heads): **{n_total:,}** parameters.\n\n"
            "The PTv3 backbone uses spconv sparse ops that torchinfo cannot "
            "trace, so the layer table below covers the OC heads only. "
            "See `configs/train/shapes.yaml` for the full backbone spec."
        ),
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
