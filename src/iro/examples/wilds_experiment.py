import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from iro.models import ResNetClassifier
from iro.aggregation.aggregators import AggregationFunction
from iro.trainer import Trainer
from iro.data.wilds_dataset import WildsDataset

def main():
    # 1. Load dataset
    dataset = WildsDataset("camelyon17", root="./data")
    train_data = dataset.load("train")
    val_data = dataset.load("val")

    # 2. Wrap into dataloaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # 3. Define model (ResNet backbone)
    model = ResNetClassifier(num_classes=2, pretrained=False)

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Risk aggregators to try
    agg_names = ["cvar", "entropic", "worst_case"]
    for name in agg_names:
        print(f"\n=== Training with {name} aggregator ===")
        agg = AggregationFunction(name)
        trainer = Trainer(
            model,
            optimizer,
            loss_fn=nn.CrossEntropyLoss(reduction="none"),
            aggregator=agg
        )

        trainer.fit(train_loader, epochs=2, alpha=0.1, eta=1.0)

        # 6. Evaluate on validation set
        val_loss = trainer.evaluate(val_loader, alpha=0.1, eta=1.0)
        print(f"Validation {name} risk = {val_loss:.4f}")

if __name__ == "__main__":
    main()