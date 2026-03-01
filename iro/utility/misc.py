"""Small utility helpers used by the original CMNIST training loop."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch


class Tee:
    """Mirror stdout/stderr to a file."""

    def __init__(self, fname: str, mode: str = "a", stream=None):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.file = open(fname, mode, encoding="utf-8")
        self.stdout = stream if stream is not None else sys.__stdout__

    def write(self, data: str) -> None:
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


def print_row(row: list[Any], colwidth: int = 12) -> None:
    out = []
    for item in row:
        if isinstance(item, float):
            text = f"{item:.6f}"
        else:
            text = str(item)
        out.append(text[:colwidth].ljust(colwidth))
    print(" ".join(out))


def _predict(algorithm, x: torch.Tensor, alpha: float | None = None):
    if alpha is None:
        return algorithm.predict(x)
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return algorithm.predict(x, alpha_t)


@torch.no_grad()
def accuracy(algorithm, loader, device: str, alpha: float | None = None) -> float:
    algorithm.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = _predict(algorithm, x, alpha=alpha)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            preds = (logits.view(-1) > 0).long()
            targets = y.view(-1).long()
        else:
            preds = logits.argmax(dim=1)
            targets = y.view(-1).long()

        correct += int((preds == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1)


@torch.no_grad()
def loss(algorithm, loader, loss_fn, device: str, alpha: float | None = None) -> float:
    algorithm.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = _predict(algorithm, x, alpha=alpha)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            val = loss_fn(logits.view(-1), y.view(-1).float())
        else:
            val = loss_fn(logits, y.view(-1).long())
        losses.append(float(val.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def cvar(algorithm, loaders, loss_fn, device: str, alphas, invariant: bool = False) -> None:
    """Lightweight CVaR print helper for compatibility logs."""
    env_losses = []
    for i, loader in enumerate(loaders):
        alpha = None if invariant else float(alphas[i])
        env_losses.append(loss(algorithm, loader, loss_fn, device, alpha=alpha))
    if env_losses:
        print(f"env_loss_mean={float(np.mean(env_losses)):.6f} env_loss_max={float(np.max(env_losses)):.6f}")
