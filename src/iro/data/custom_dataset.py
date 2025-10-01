from .base_dataset import BaseDataset

class CustomDataset(BaseDataset):
    """Template for user-defined datasets."""

    def __init__(self, name: str, data_source: str):
        super().__init__(name)
        self.data_source = data_source

    def load(self, split: str = "train"):
        # user loads their dataset here
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        # user-specific preprocessing
        raise NotImplementedError