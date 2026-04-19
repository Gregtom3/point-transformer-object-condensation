"""Backwards-compatibility shim.

The shapes dataset lives in :mod:`src.tasks.shapes` as
:class:`ShapesTask` now — it pairs data iteration with the task-specific
rendering used by the trainer and TB logger. This module re-exports the
old ``ShapeDataset`` / ``collate_shapes`` names so existing imports keep
working.
"""
from __future__ import annotations

from typing import Any

from src.tasks.shapes import ShapesTask as ShapeDataset  # noqa: F401


def collate_shapes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy free-function collate — delegates to :meth:`ShapesTask.collate`.

    Kept so scripts that imported ``collate_shapes`` directly still work.
    The collate logic itself has no per-instance state, so a throwaway
    ``ShapesTask.__new__`` is sufficient.
    """
    stub = ShapeDataset.__new__(ShapeDataset)
    return stub.collate(batch)
