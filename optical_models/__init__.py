# -*- coding: utf-8 -*-
"""
PADDOL: Python Advanced Design & Dispersion Optimization Lab
Copyright (c) 2025 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

# my_dispersion/__init__.py   (excerpt – replace the discovery section)

import importlib
import pkgutil
import pathlib
from types import ModuleType
from typing import Dict

_MODEL_REGISTRY: Dict[str, object] = {}

def _discover_models() -> None:
    """Discover and register all public callables in the package.

    The function imports each sub‑module, then adds every non‑private
    callable (function or class) that originates from that module to
    ``_MODEL_REGISTRY``.  It is idempotent: a second call does nothing.
    """
    if _MODEL_REGISTRY:
        return

    # Resolve package name & path
    if __package__:
        pkg_name = __package__
        pkg_path = pathlib.Path(importlib.import_module(pkg_name).__file__).parent
    else:  # executed as script
        pkg_name = __spec__.name.split(".")[0]          # e.g. "my_dispersion"
        pkg_path = pathlib.Path(__file__).resolve().parent

    for _, mod_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        full_mod_name = f"{pkg_name}.{mod_name}"
        mod: ModuleType = importlib.import_module(full_mod_name)

        for name, obj in vars(mod).items():
            if (
                callable(obj)                      # functions *and* classes
                and not name.startswith("_")
                and getattr(obj, "__module__", None) == mod.__name__
            ):
                _MODEL_REGISTRY[name] = obj

_discover_models()
globals().update(_MODEL_REGISTRY)
__all__ = list(_MODEL_REGISTRY.keys())
