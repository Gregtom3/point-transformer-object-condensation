"""Task abstractions for Object Condensation pipelines.

Each task couples a data source with visualization logic appropriate to
the data's domain. The base :class:`OCTask` defines the interface the
trainer / TB logger / evaluator use; concrete subclasses (e.g.
:class:`ShapesTask`) implement the per-domain pieces.
"""
from .base import OCTask  # noqa: F401
from .shapes import ShapesTask  # noqa: F401
