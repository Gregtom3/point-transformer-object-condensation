"""Per-event augmentations for OC tasks.

Augmentations are callables that take an event dict (the same dict a task
returns from ``__getitem__``) and return a possibly-modified event dict.
They compose, and a task can thread one (or a :class:`Compose` of many)
through its ``__getitem__`` to sample a new augmented view each iteration.

See :mod:`~src.augmentations.base` for the contract and
:mod:`~src.augmentations.basic` for ready-to-use augmentations.
"""
from .base import Augmentation, Compose, Identity  # noqa: F401
from .basic import RandomColorJitter, RandomHitDropout, RandomRotation2D  # noqa: F401
