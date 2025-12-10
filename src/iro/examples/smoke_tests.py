"""
Lightweight smoke tests for IRO components.

Usage (from repo root):
  PYTHONPATH=src python -m iro.examples.smoke_tests
or install editable (`pip install -e .`) then:
  python -m iro.examples.smoke_tests
"""
import sys
from pathlib import Path

import torch
from torch import nn

# Ensure local src/ is on path when running without installation
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iro.aggregation.aggregators import AggregationFunction
from iro.utility import algorithms


def check_aggregators():
    risks = torch.linspace(0, 1, steps=20)
    cvar = AggregationFunction("cvar").aggregate(risks, alpha=0.8)
    evar = AggregationFunction("evar").aggregate(risks, alpha=0.8)
    esrm = AggregationFunction("exponential").aggregate(risks, gamma=1.0)
    print(f"CVaR(0.8)={cvar:.4f}  EVaR(0.8)={evar:.4f}  ESRM(gamma=1)={esrm:.4f}")


def check_erm_training():
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    bce = nn.BCEWithLogitsLoss()

    def loss_fn(preds, target):
        return bce(preds.squeeze(1), target)

    hparams = {
        "lr": 1e-2,
        "weight_decay": 0.0,
        "erm_pretrain_iters": 0,
        "lr_factor_reduction": 1.0,
        "penalty_weight": 1.0,
        "groupdro_eta": 1.0,
    }
    alg = algorithms.ERM(model, hparams, loss_fn)
    minibatches = [(x, y)]
    for _ in range(10):
        alg.update(minibatches)
    with torch.no_grad():
        logits = alg.predict(x).squeeze(1)
        acc = ((torch.sigmoid(logits) > 0.5) == y).float().mean().item()
    print(f"ERM smoke accuracy on synthetic data: {acc:.3f}")


def main():
    check_aggregators()
    check_erm_training()
    print("Smoke tests completed.")


if __name__ == "__main__":
    main()
