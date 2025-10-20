import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from iro.models import MLP
from iro.aggregation.aggregators import AggregationFunction
from iro.trainer import Trainer

# Synthetic dataset (for quick check)
X = torch.randn(500, 20)
y = torch.randint(0, 2, (500,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = MLP(input_size=20, hidden_size=64, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Try different aggregation functions
agg_names = ["cvar", "entropic", "var", "ph"]
for name in agg_names:
    print(f"\n=== Training with {name} aggregator ===")
    agg = AggregationFunction(name)
    trainer = Trainer(model, optimizer, loss_fn=nn.CrossEntropyLoss(reduction="none"), aggregator=agg)
    trainer.fit(train_loader, epochs=2, alpha=0.1)