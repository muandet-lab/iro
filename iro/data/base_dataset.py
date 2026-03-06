"""Base abstractions for dataset wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Abstract base class for datasets used in IRO."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self, split: str = "train"):
        """Load a dataset split (train/val/test)."""

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """Preprocess dataset (e.g. normalization, domain splits)."""
