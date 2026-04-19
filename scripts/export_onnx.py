"""Export OC heads to ONNX.

Why only the heads: PTv3's backbone uses spconv's sparse-conv kernels
(``MaskImplicitGemm``), which don't map to any ONNX op set. Exporting
the whole model requires either a dense rewrite of the backbone or
going through spconv's TensorRT path; neither is one-liner work. The
heads, in contrast, are plain MLPs over per-hit features and export
cleanly.

Intended use: run the PTv3 backbone in PyTorch to get per-hit features,
then serve the three heads through the exported ONNX graph.

Usage:
    python scripts/export_onnx.py --config configs/train/shapes.yaml \\
        --checkpoint outputs/shapes/epoch_009.pt --out outputs/shapes/heads.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.backbone import PTv3Backbone
from src.models.heads import ObjectCondensationHeads


class HeadsExportWrapper(torch.nn.Module):
    """Produces a flat tuple output so ONNX can represent it cleanly."""

    def __init__(self, heads: ObjectCondensationHeads) -> None:
        super().__init__()
        self.heads = heads
        self.predict_width_height = heads.predict_width_height
        self.predict_energy = heads.predict_energy
        self.predict_momentum = heads.predict_momentum

    def forward(self, feat: torch.Tensor):
        out = self.heads(feat)
        parts = [out["beta"], out["x"], out["pid_logits"]]
        if self.predict_width_height:
            parts += [out["width"], out["height"]]
        if self.predict_energy:
            parts.append(out["energy"])
        if self.predict_momentum:
            parts.append(out["momentum"])
        return tuple(parts)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train/shapes.yaml"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n-hits", type=int, default=1024,
                    help="dummy N for the dynamic-axes trace")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    backbone = PTv3Backbone(**OmegaConf.to_container(cfg.model.backbone, resolve=True))
    heads = ObjectCondensationHeads(
        in_dim=backbone.out_channels,
        **OmegaConf.to_container(cfg.model.heads, resolve=True),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    heads.load_state_dict(ckpt["heads"])
    heads.eval()

    wrapper = HeadsExportWrapper(heads).eval()
    dummy = torch.zeros(args.n_hits, backbone.out_channels)

    output_names = ["beta", "x", "pid_logits"]
    if heads.predict_width_height:
        output_names += ["width", "height"]
    if heads.predict_energy:
        output_names.append("energy")
    if heads.predict_momentum:
        output_names.append("momentum")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.out),
        input_names=["feat"],
        output_names=output_names,
        dynamic_axes={"feat": {0: "n_hits"}, **{n: {0: "n_hits"} for n in output_names}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KiB)")
    print("NOTE: this export covers the OC heads only; the PTv3 backbone "
          "still needs to run in PyTorch (spconv is not ONNX-compatible).")


if __name__ == "__main__":
    main()
