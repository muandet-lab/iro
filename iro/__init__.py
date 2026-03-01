"""Top-level package for Imprecise Risk Optimization."""

from __future__ import annotations

from .core import run_evaluation, run_training
from .iro import ARM_Regression, AggregationFunction, Pareto_distribution, aggregation_function
from .utility.networks import FHatNetwork, FiLMClassifierHead, FiLMLayer, HyperNetwork

__author__ = "Joseph C. Sheils"
__email__ = "joseph.sheils@example.com"

__all__ = [
    "aggregation_function",
    "AggregationFunction",
    "Pareto_distribution",
    "ARM_Regression",
    "FHatNetwork",
    "HyperNetwork",
    "FiLMLayer",
    "FiLMClassifierHead",
    "run_training",
    "run_evaluation",
]
