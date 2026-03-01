"""Minimal utilities required by DGIL-style algorithms.

The original project used a non-parametric risk estimator. For the current
minimal IRO codebase we keep a lightweight quantile-based approximation.
"""

from __future__ import annotations

import torch


def get_grad_norm(model: torch.nn.Module) -> float:
    """Compute L2 norm of all available gradients."""

    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is None:
            continue
        total = total + torch.sum(p.grad.detach() ** 2)
    return float(torch.sqrt(total).item())


class Nonparametric:
    """Simple empirical distribution helper based on observed risk values."""

    def __init__(self):
        self._samples = None

    def estimate_parameters(self, risks: torch.Tensor) -> None:
        risks = risks.detach().view(-1).float()
        if risks.numel() == 0:
            raise ValueError("Nonparametric estimator requires at least one sample.")
        self._samples = torch.sort(risks).values

    def icdf(self, alpha: torch.Tensor | float) -> torch.Tensor:
        if self._samples is None:
            raise RuntimeError("Call estimate_parameters before icdf.")

        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=self._samples.dtype, device=self._samples.device)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return torch.quantile(self._samples, alpha)
