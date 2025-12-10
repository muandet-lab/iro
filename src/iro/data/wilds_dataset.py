from .base_dataset import BaseDataset
import wilds  # pip install wilds

class WildsDataset(BaseDataset):
    """WILDS dataset wrapper."""

    def __init__(self, name: str, root: str = "./data"):
        super().__init__(name)
        self.root = root
        self.dataset = wilds.get_dataset(dataset=name, download=True, root_dir=root)

    def load(self, split: str = "train"):
        return self.dataset.get_subset(split)

    def preprocess(self, *args, **kwargs):
        # optional preprocessing hook
        pass