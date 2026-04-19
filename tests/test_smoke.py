"""Smoke tests: imports + one forward pass + one OC loss call.

Designed to run on CPU so they pass in minimal dev environments.
PTv3's flash attention is disabled and patch sizes shrunk below 500 so
the network fits the dummy batch without serialization crashes.
"""
from __future__ import annotations

import math

import pytest
import torch


def _have(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


needs_torch_scatter = pytest.mark.skipif(
    not _have("torch_scatter"), reason="torch-scatter not installed"
)
needs_spconv = pytest.mark.skipif(not _have("spconv"), reason="spconv not installed")
needs_addict = pytest.mark.skipif(not _have("addict"), reason="addict not installed")
needs_timm = pytest.mark.skipif(not _have("timm"), reason="timm not installed")


def test_import_wrappers():
    """Backbone and heads modules import without errors."""
    from src.models import heads  # noqa: F401
    from src.inference import cluster  # noqa: F401


def test_oc_loss_tiger_finite():
    """condensation_loss_tiger returns finite scalars on dummy data."""
    try:
        from object_condensation.pytorch.losses import condensation_loss_tiger
    except ImportError:
        pytest.skip("object_condensation not installed; run setup.sh first")

    torch.manual_seed(0)
    n = 256
    beta = torch.sigmoid(torch.randn(n))
    x = torch.randn(n, 4)
    # object ids: ~1/4 noise (0), the rest spread over 5 objects
    object_id = torch.randint(0, 6, (n,))
    out = condensation_loss_tiger(
        beta=beta, x=x, object_id=object_id, q_min=1.0, noise_threshold=0, max_n_rep=0
    )
    assert isinstance(out, dict)
    for k, v in out.items():
        if k == "n_rep":
            continue
        assert torch.isfinite(v).all(), f"non-finite loss {k}={v}"
        assert math.isfinite(float(v))


@needs_torch_scatter
@needs_spconv
@needs_addict
@needs_timm
def test_ptv3_forward_smoke():
    """PTv3 backbone runs forward on a ~500-point dummy batch.

    spconv's implicit-gemm sparse convolution is CUDA-only, so this test
    is skipped on machines without a CUDA device.
    """
    if not torch.cuda.is_available():
        pytest.skip("spconv's implicit-gemm kernel requires CUDA")

    from src.models.backbone import PTv3Backbone

    device = torch.device("cuda")
    torch.manual_seed(0)
    in_ch = 8
    backbone = PTv3Backbone(
        in_channels=in_ch,
        enc_depths=(1, 1, 1),
        enc_channels=(16, 32, 64),
        enc_num_head=(2, 2, 4),
        enc_patch_size=(64, 64, 64),
        stride=(2, 2),
        dec_depths=(1, 1),
        dec_channels=(16, 32),
        dec_num_head=(2, 2),
        dec_patch_size=(64, 64),
        mlp_ratio=2,
        drop_path=0.0,
        enable_flash=False,
        shuffle_orders=False,
        grid_size=0.02,
    ).to(device)
    n = 500
    coord = torch.rand(n, 3, device=device)
    feat = torch.randn(n, in_ch, device=device)
    offset = torch.tensor([n], dtype=torch.long, device=device)
    data = {"coord": coord, "feat": feat, "offset": offset, "grid_size": 0.02}

    point = backbone(data)
    assert "feat" in point
    assert point["feat"].shape[0] == n
    assert point["feat"].shape[1] == backbone.out_channels


def test_heads_forward():
    """OC heads produce correctly-shaped tensors from dummy backbone output."""
    from src.models.heads import ObjectCondensationHeads

    n, d = 128, 64
    feat = torch.randn(n, d)
    heads = ObjectCondensationHeads(in_dim=d, cluster_dim=4, n_pid_classes=5)
    out = heads(feat)
    assert out["beta"].shape == (n,)
    assert out["x"].shape == (n, 4)
    assert out["energy"].shape == (n,)
    assert out["momentum"].shape == (n, 3)
    assert out["pid_logits"].shape == (n, 5)
    assert torch.all((out["beta"] >= 0) & (out["beta"] <= 1))
