import torch.nn as nn

class TransformerClassifier(nn.Module):
    """Simple Transformer encoder classifier for tabular/text-like input.
    
    Expects input of shape (batch, seq_len, dim).
    """
    def __init__(self, input_dim: int, num_classes: int, num_heads: int = 2, num_layers: int = 2, hidden_dim: int = 64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        h = self.encoder(x)         # (batch, seq_len, dim)
        h = h.mean(dim=1)           # mean-pool over sequence
        return self.classifier(h)