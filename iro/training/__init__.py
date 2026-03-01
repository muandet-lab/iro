"""Training and evaluation entrypoints for CMNIST."""

from __future__ import annotations

from .train_cmnist import eval_cmnist_iro, train_cmnist_iro

__all__ = [
    "train_cmnist_iro",
    "eval_cmnist_iro",
]
