"""Evaluation entrypoint.

Loads a trained checkpoint, runs OC inference over the test split, and
prints per-event cluster/purity metrics plus shape-id and width/height
summaries.

Metrics per event:
    * n_truth_shapes        - unique object ids > 0 in the event
    * n_pred_clusters       - clusters returned by beta-threshold clustering
    * purity / efficiency   - simple matching via majority vote
    * shape_accuracy        - of foreground hits, fraction with correct shape id
    * width_mae / height_mae

Usage:
    python scripts/evaluate.py --config configs/train/shapes.yaml \
        --checkpoint outputs/shapes/epoch_009.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.inference.cluster import beta_threshold_cluster
from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads
from src.tasks import ShapesTask, get_task_class  # noqa: F401


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train/shapes.yaml"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out", type=Path, default=Path("outputs/shapes/eval.json"))
    return ap.parse_args()


@torch.no_grad()
def match_clusters(truth: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    """Return (purity, efficiency) via majority-vote matching.

    For every predicted cluster id > 0 pick the most-common truth id >= 1
    and score how many hits agree (purity). Efficiency is the symmetric
    version: for each truth id, the best-matching predicted cluster's hit
    coverage.
    """
    device = truth.device
    mask = (truth > 0) & (pred > 0)
    if not mask.any():
        return 0.0, 0.0
    t = truth[mask]
    p = pred[mask]
    # purity
    pur_num = 0
    pur_den = 0
    for c in p.unique():
        m = p == c
        _, counts = t[m].unique(return_counts=True)
        pur_num += int(counts.max())
        pur_den += int(m.sum())
    purity = pur_num / max(1, pur_den)
    # efficiency
    eff_num = 0
    eff_den = 0
    for o in t.unique():
        m = t == o
        _, counts = p[m].unique(return_counts=True)
        eff_num += int(counts.max())
        eff_den += int(m.sum())
    efficiency = eff_num / max(1, eff_den)
    return purity, efficiency


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(cfg.data.root)
    split_file = getattr(cfg.data, f"{args.split}_file")
    task_name = getattr(cfg.data, "task_class", "ShapesTask")
    TaskCls = get_task_class(task_name)
    task = TaskCls(
        root / split_file,
        normalize_coords=cfg.data.normalize_coords,
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

    t_beta = cfg.inference.t_beta
    t_d = cfg.inference.t_d

    totals = {
        "events": 0,
        "n_truth_shapes": 0,
        "n_pred_clusters": 0,
        "purity_sum": 0.0,
        "efficiency_sum": 0.0,
        "shape_correct": 0,
        "shape_total": 0,
        "width_abs_err": 0.0,
        "height_abs_err": 0.0,
        "fg_hits": 0,
    }

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            data = {"coord": batch["coord"], "feat": batch["feat"], "offset": batch["offset"]}
            point = backbone(data)
            preds = heads(point["feat"])
            cluster = beta_threshold_cluster(preds["beta"], preds["x"], t_beta=t_beta, t_d=t_d)
            truth = batch["object_id"]

            pur, eff = match_clusters(truth, cluster)
            totals["events"] += 1
            totals["n_truth_shapes"] += int((truth.unique() > 0).sum())
            totals["n_pred_clusters"] += int(cluster.max())
            totals["purity_sum"] += pur
            totals["efficiency_sum"] += eff

            mask = truth > 0
            if mask.any():
                pred_shape = preds["pid_logits"][mask].argmax(-1)
                totals["shape_correct"] += int((pred_shape == batch["shape_id_per_hit"][mask]).sum())
                totals["shape_total"] += int(mask.sum())
                totals["fg_hits"] += int(mask.sum())
                if "width" in preds:
                    totals["width_abs_err"] += float(
                        (preds["width"][mask] - batch["width_per_hit"][mask]).abs().sum()
                    )
                if "height" in preds:
                    totals["height_abs_err"] += float(
                        (preds["height"][mask] - batch["height_per_hit"][mask]).abs().sum()
                    )

    n = max(1, totals["events"])
    fg = max(1, totals["fg_hits"])
    # width/height live in frame-normalized units during training; report
    # them back in pixels for readability.
    fw, fh = task.frame
    out = {
        "split": args.split,
        "events": totals["events"],
        "mean_n_truth_shapes": totals["n_truth_shapes"] / n,
        "mean_n_pred_clusters": totals["n_pred_clusters"] / n,
        "purity": totals["purity_sum"] / n,
        "efficiency": totals["efficiency_sum"] / n,
        "shape_accuracy": totals["shape_correct"] / max(1, totals["shape_total"]),
        "width_mae_px": totals["width_abs_err"] / fg * fw,
        "height_mae_px": totals["height_abs_err"] / fg * fh,
    }
    print(json.dumps(out, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
