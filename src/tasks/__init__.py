"""Task abstractions for Object Condensation pipelines.

Each task couples a data source with visualization logic appropriate to
the data's domain. The base :class:`OCTask` defines the interface the
trainer / TB logger / evaluator use; concrete subclasses (e.g.
:class:`ShapesTask`) implement the per-domain pieces.
"""
from .base import OCTask  # noqa: F401
from .shapes import ShapesTask  # noqa: F401


def _load_calorimeter_task():
    """Lazy import so the calorimeter example doesn't need to import
    unless the user actually asks for it."""
    from examples.calorimeter.task import CalorimeterTask  # noqa: WPS433
    return CalorimeterTask


TASK_REGISTRY: dict[str, object] = {
    "ShapesTask": ShapesTask,
    # Example / tutorial task — resolved lazily via __getitem__ below.
    "CalorimeterTask": _load_calorimeter_task,
}


def get_task_class(name: str):
    """Resolve a task class by its string name (for runcards).

    Values in :data:`TASK_REGISTRY` may be the class itself or a zero-arg
    factory that returns the class — the factory form lets examples
    defer their imports.
    """
    try:
        entry = TASK_REGISTRY[name]
    except KeyError as err:
        known = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(
            f"unknown task_class {name!r}; known: {known}"
        ) from err
    return entry() if callable(entry) and not isinstance(entry, type) else entry
