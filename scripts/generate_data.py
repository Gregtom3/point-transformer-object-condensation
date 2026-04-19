"""Thin wrapper over ``data/generate_shapes.py`` so the generator sits
next to the other scripts. Forwards all CLI args unchanged."""
from __future__ import annotations

import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = "data/generate_shapes.py"
    runpy.run_path("data/generate_shapes.py", run_name="__main__")
