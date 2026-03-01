import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)