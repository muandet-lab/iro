
"""
Utility functions and classes for IRO.

This subpackage collects reusable building blocks (data loaders, KDE helpers,
risk aggregators, etc.) that are shared across experiments and training
scripts (including CMNIST_spectral).
"""

from .arm_regression import ARM_Regression
from .data_generator_1d import data_generator_1D
from .fast_data_loader import FastDataLoader, InfiniteDataLoader
from .iro_utils import (
    aggregation_function,
    ArrowPrattDistribution,
    EntropicDistribution,
    Pareto_distribution,
)
from .kde import get_grad_norm, Nonparametric
from . import misc

__all__ = [
    "ARM_Regression",
    "ArrowPrattDistribution",
    "EntropicDistribution",
    "FastDataLoader",
    "InfiniteDataLoader",
    "Nonparametric",
    "Pareto_distribution",
    "aggregation_function",
    "data_generator_1D",
    "get_grad_norm",
    "misc",
]
