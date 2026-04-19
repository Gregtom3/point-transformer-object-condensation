"""Augmentation base class and composition helpers.

Every augmentation is a callable with the signature

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]: ...

It receives one event — the same dict shape a task yields from
``__getitem__`` — and returns an event with the same keys. Shape- /
dtype-changing augmentations (e.g. hit dropout changes ``N``) are fine
as long as every per-hit tensor is kept consistent.

Contract every augmentation must honor:

* It MUST preserve the set of per-hit tensors: if the event contains
  ``coord`` it must still contain ``coord`` on return, and every
  per-hit tensor must share the same first dimension ``N`` (possibly
  a new ``N`` if the augmentation drops/duplicates hits).
* It MAY mutate the event in place or return a new dict; tasks treat
  the return value as authoritative.
* It MUST NOT touch per-object or metadata tensors whose shape is not
  ``(N, ...)`` (e.g. ``frame``, ``batch_size``).
* It SHOULD be cheap — it runs inside ``__getitem__`` in each DataLoader
  worker, once per sample per epoch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Augmentation(ABC):
    """Base class for per-event augmentations.

    Subclasses override :meth:`__call__`. They may store state (e.g. a
    ``numpy.random.Generator``) but must be picklable so DataLoader
    worker processes can recreate them.
    """

    @abstractmethod
    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class Identity(Augmentation):
    """No-op augmentation. Useful as a default / off-switch."""

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        return event


class Compose(Augmentation):
    """Apply a sequence of augmentations left-to-right.

    ``Compose([A, B, C])(event)`` is ``C(B(A(event)))``.
    """

    def __init__(self, augs: list[Augmentation]) -> None:
        self.augs = list(augs)

    def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        for aug in self.augs:
            event = aug(event)
        return event

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.augs)
        return f"Compose([{inner}])"
