"""Model package.

The backbone wrapper is imported lazily because its heavy CUDA
dependencies (spconv, torch-scatter, flash-attn) aren't always
available in dev environments. Use ``from src.models.backbone import
PTv3Backbone`` explicitly when you need it.
"""
from .heads import ObjectCondensationHeads  # noqa: F401
