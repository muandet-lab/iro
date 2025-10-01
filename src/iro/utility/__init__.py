
"""
Utility functions and classes for IRO.

This subpackage contains helper modules such as:
- ARM_Regression: Aggregated Risk Minimization regression model.
- data_generator_1D: Synthetic data generator for 1D experiments.
"""

from .arm_regression import ARM_Regression
from .data_generator_1d import data_generator_1D

__all__ = ["ARM_Regression", "data_generator_1D"]