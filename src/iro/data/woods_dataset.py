from .base_dataset import BaseDataset
import woods  

class WoodsDataset(BaseDataset):
    """WOODS Benchmark dataset wrapper."""

    def __init__(self, name: str, root: str = "./data"):
        super().__init__(name)
        self.root = root

    def load(self, split: str = "train"):
        return woods.get_dataset(self.name, split=split, root_dir=self.root)

    def preprocess(self, *args, **kwargs):
        # optional preprocessing hook
        pass