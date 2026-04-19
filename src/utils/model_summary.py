"""Emit a structured architecture report for a training run.

Writes three artefacts to ``out_dir``:
    * ``summary.txt``     - torchinfo layer table (parameter counts, shapes).
    * ``architecture.md`` - Mermaid flowchart of the high-level blocks plus
                            a short description of the forward data flow.
    * ``params.json``     - parameter counts per top-level module.

Designed to be called once at the start of training so the log dir is
self-describing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def count_params(module: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _torchinfo_summary(module: nn.Module, input_data: Any | None) -> str:
    try:
        from torchinfo import summary
    except Exception:
        return "torchinfo not installed; `pip install torchinfo`"
    try:
        s = summary(
            module,
            input_data=input_data,
            depth=5,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
        )
        return str(s)
    except Exception as e:  # pragma: no cover
        return f"torchinfo.summary() failed: {e}"


_MERMAID = """```mermaid
flowchart LR
    in[("hit cloud<br/>coord (N,3)<br/>feat (N,F_in)<br/>offset (B,)")] --> emb["Embedding<br/>(Linear + GN + act)"]
    emb --> enc["PTv3 encoder<br/>serialized attention<br/>(U-Net down)"]
    enc --> dec["PTv3 decoder<br/>(U-Net up)"]
    dec --> pt(("point features<br/>(N, D_out)"))
    pt --> beta["beta head<br/>sigmoid"]
    pt --> x["cluster-coord head<br/>(N, cluster_dim)"]
    pt --> pid["shape-class head<br/>softmax over C"]
    pt --> wh["width/height heads<br/>scalar regression"]
    beta --> oc{{Object Condensation<br/>inference:<br/>beta>t_b, ||x-x_c||<t_d}}
    x --> oc
    oc --> out[("predicted clusters<br/>+ per-cluster w,h,pid")]
```
"""


def write_architecture_report(
    model: nn.Module,
    out_dir: str | Path,
    example_input: Any | None = None,
    extra_note: str = "",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total, trainable = count_params(model)
    per_top: dict[str, dict[str, int]] = {}
    for name, child in model.named_children():
        t, tr = count_params(child)
        per_top[name] = {"total": t, "trainable": tr}

    (out / "params.json").write_text(
        json.dumps({"total": total, "trainable": trainable, "per_module": per_top}, indent=2)
    )

    summary_txt = _torchinfo_summary(model, example_input)
    (out / "summary.txt").write_text(summary_txt)

    md = []
    md.append("# Architecture\n\n")
    md.append(f"**Total parameters:** {total:,} ({trainable:,} trainable)\n\n")
    md.append("## Top-level modules\n\n")
    md.append("| module | params |\n|---|---:|\n")
    for name, counts in per_top.items():
        md.append(f"| `{name}` | {counts['total']:,} |\n")
    md.append("\n## Data flow\n\n")
    md.append(_MERMAID)
    md.append("\n")
    if extra_note:
        md.append("\n## Notes\n\n")
        md.append(extra_note)
        md.append("\n")
    md.append("\n## Layer table\n\n```\n")
    md.append(summary_txt)
    md.append("\n```\n")
    (out / "architecture.md").write_text("".join(md))
