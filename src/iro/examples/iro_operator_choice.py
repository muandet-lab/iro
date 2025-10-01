import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from iro.models import MLP
from iro.aggregation.aggregators import AggregationFunction
from iro.trainer import Trainer

# Step 1: synthetic multi-domain dataset
def make_domains():
    X1, y1 = torch.randn(200, 20), torch.randint(0, 2, (200,))
    X2, y2 = torch.randn(200, 20) * 1.5, torch.randint(0, 2, (200,))
    X3, y3 = torch.randn(200, 20) + 1.0, torch.randint(0, 2, (200,))
    datasets = [TensorDataset(X1, y1), TensorDataset(X2, y2), TensorDataset(X3, y3)]
    return datasets

domains = make_domains()
loaders = [DataLoader(d, batch_size=32, shuffle=True) for d in domains]

# Step 2: train under credal set (multi-risk objective)
model = MLP(input_size=20, hidden_size=64, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

credal_aggregators = [AggregationFunction("cvar"), AggregationFunction("entropic"), AggregationFunction("ph")]

trainer = Trainer(model, optimizer, loss_fn=nn.CrossEntropyLoss(reduction="none"),
                  aggregator=None)  # aggregator will be applied manually

epochs = 2
for e in range(epochs):
    for batches in zip(*loaders):
        losses = []
        for (inputs, targets) in batches:
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
            # apply multiple aggregators
            risks = [agg.aggregate(loss, alpha=0.1, eta=1.0) for agg in credal_aggregators]
            combined_risk = torch.stack(risks).max()  # hedge against credal set
            losses.append(combined_risk)
        total_risk = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_risk.backward()
        optimizer.step()

# Step 3: test-time operator choice
operator_aggs = [
    ("cvar", {"alpha": 0.1}),
    ("entropic", {"eta": 1.0}),
    ("variance", {}),
    ("worst_case", {}),
    ("ph", {"xi": 0.5}),
]
for name, params in operator_aggs:
    agg = AggregationFunction(name)
    all_risks = []
    for loader in loaders:
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
            risk = agg.aggregate(loss, **params)
            all_risks.append(risk.item())
    print(f"Operator choice {name} with params {params}: avg test risk = {sum(all_risks)/len(all_risks):.4f}")