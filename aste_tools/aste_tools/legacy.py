"""Compatibility helpers for notebook-era ``an_helper_functions`` modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


LEGACY_DIR = Path(__file__).resolve().parents[1] / "an_helper_functions"


def load_legacy_module(name: str) -> ModuleType:
    """Load one module from ``an_helper_functions`` without mutating ``sys.path``."""

    path = LEGACY_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"legacy helper module not found: {path}")
    spec = importlib.util.spec_from_file_location(f"aste_tools.legacy_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load legacy helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

