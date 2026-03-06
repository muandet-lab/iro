"""Template for user-defined datasets."""

from __future__ import annotations

from .base_dataset import BaseDataset


class CustomDataset(BaseDataset):
    """Template for user-defined datasets."""

    def __init__(self, name: str, data_source: str):
        super().__init__(name)
        self.data_source = data_source

    def load(self, split: str = "train"):
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError
