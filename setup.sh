#!/usr/bin/env bash
# Bootstrap script for the PTv3 + Object Condensation environment.
#
# Installs in the required order: torch first, then the CUDA-linked
# extensions (torch-scatter / torch-geometric / spconv), then flash-attn,
# then everything else, then the object_condensation submodule in
# editable mode.
#
# Usage:
#   CUDA=cu121 bash setup.sh   # default
#   CUDA=cu118 bash setup.sh
set -euo pipefail

CUDA="${CUDA:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"

case "$CUDA" in
  cu121) SPCONV_PKG="spconv-cu121" ;;
  cu118) SPCONV_PKG="spconv-cu118" ;;
  *) echo "Unknown CUDA tag: $CUDA (expected cu121 or cu118)"; exit 1 ;;
esac

echo "[1/5] Installing torch (${TORCH_VERSION}+${CUDA})"
pip install --extra-index-url "https://download.pytorch.org/whl/${CUDA}" "torch>=${TORCH_VERSION}"

echo "[2/5] Installing torch-scatter / torch-geometric"
pip install --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA}.html" torch-scatter torch-geometric

echo "[3/5] Installing sparse conv: ${SPCONV_PKG}"
pip install "${SPCONV_PKG}"

echo "[4/5] Installing flash-attn (optional; skip with SKIP_FLASH=1)"
if [[ "${SKIP_FLASH:-0}" != "1" ]]; then
  pip install flash-attn --no-build-isolation || {
    echo "  flash-attn install failed — disable it via enable_flash=False in the model config"
  }
fi

echo "[5/5] Installing remaining requirements + object_condensation"
pip install numpy scipy pandas pyyaml omegaconf hydra-core timm addict wandb tensorboard pytest
pip install -e "third_party/object_condensation[pytorch]"

echo "Done. Run 'pytest tests/' to verify the install."
