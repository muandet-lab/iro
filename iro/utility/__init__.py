"""Utility exports for IRO training logic.

Keep imports lazy so top-level package import stays lightweight and avoids
forcing optional heavy dependencies at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ARM_Regression",
    "Nonparametric",
    "get_grad_norm",
    "FHatNetwork",
    "HyperNetwork",
    "FiLMLayer",
    "FiLMClassifierHead",
]


def __getattr__(name: str) -> Any:
    if name == "ARM_Regression":
        return import_module("iro.utility.arm_regression").ARM_Regression
    if name in {"Nonparametric", "get_grad_norm"}:
        module = import_module("iro.utility.kde")
        return getattr(module, name)
    if name in {"FHatNetwork", "HyperNetwork", "FiLMLayer", "FiLMClassifierHead"}:
        module = import_module("iro.utility.networks")
        return getattr(module, name)
    raise AttributeError(f"module 'iro.utility' has no attribute '{name}'")
