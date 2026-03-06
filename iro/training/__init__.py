"""Training and evaluation entrypoints for supported experiments."""

from __future__ import annotations

from .train_cmnist import eval_cmnist_iro, train_cmnist_iro
from .train_iwildcam import eval_iwildcam_iro, train_iwildcam_iro

__all__ = [
    "train_cmnist_iro",
    "eval_cmnist_iro",
    "train_iwildcam_iro",
    "eval_iwildcam_iro",
]
