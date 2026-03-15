"""Neural architectures used by IRO training utilities.

This module contains:
- Tabular conditioning networks from the original DGIL UCI-Bike-Rental workflow.
- Shared FiLM conditioning layers for representation-based tasks (e.g., images).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class FHatNetwork(nn.Module):
    """Feed-forward network used both as a base predictor and hypernetwork block."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] | None,
        output_size: int,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_sizes = list(hidden_sizes or [])
        layers: list[nn.Module] = []

        if not hidden_sizes:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.PReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                layers.append(nn.PReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HyperNetwork(nn.Module):
    """Hypernetwork that conditions a linear predictor on CVaR level `alpha`.

    This generalizes the original notebook implementation:
    - `alpha` is passed through an MLP (`FHatNetwork`)
    - the output is converted to linear weights (and optional bias)
    - prediction is done as conditioned linear map of tabular input `x`
    """

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int = 1,
        hidden_sizes: Sequence[int] | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_bias = bool(use_bias)

        param_dim = self.output_dim * self.input_dim
        if self.use_bias:
            param_dim += self.output_dim

        self.hyper_layer = FHatNetwork(1, hidden_sizes or [], param_dim)

    def conditioned_parameters(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if alpha.ndim == 0:
            alpha = alpha.view(1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(-1, 1)

        raw = torch.tanh(self.hyper_layer(alpha))
        raw = raw.mean(dim=0, keepdim=True)

        if self.use_bias:
            w_flat = raw[:, : self.output_dim * self.input_dim]
            b_flat = raw[:, self.output_dim * self.input_dim :]
            weight = w_flat.view(self.output_dim, self.input_dim)
            bias = b_flat.view(self.output_dim)
        else:
            weight = raw.view(self.output_dim, self.input_dim)
            bias = None

        return weight, bias

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        else:
            alpha = alpha.to(device=x.device, dtype=x.dtype)

        weight, bias = self.conditioned_parameters(alpha)
        out = x @ weight.t()
        if bias is not None:
            out = out + bias
        return out


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation conditioned on the risk level `alpha`.

    The conditioner predicts per-feature scale and shift:
        h' = (1 + tanh(gamma(alpha))) * h + tanh(beta(alpha))
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_sizes: Sequence[int] | None = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.conditioner = FHatNetwork(1, hidden_sizes or [], 2 * self.feature_dim)

    def _alpha_tensor(self, alpha: torch.Tensor | float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=device, dtype=dtype)
        else:
            alpha = alpha.to(device=device, dtype=dtype)

        if alpha.ndim == 0:
            alpha = alpha.view(1, 1)
        elif alpha.ndim == 1:
            alpha = alpha.view(-1, 1)
        return alpha

    def forward(self, features: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        alpha_t = self._alpha_tensor(alpha, device=features.device, dtype=features.dtype)
        params = self.conditioner(alpha_t).mean(dim=0, keepdim=True)
        gamma, beta = torch.split(params, self.feature_dim, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        beta = torch.tanh(beta)
        return features * gamma + beta


class FiLMClassifierHead(nn.Module):
    """Generic FiLM-conditioned classifier head for feature vectors."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_sizes: Sequence[int] | None = None,
    ):
        super().__init__()
        self.film = FiLMLayer(feature_dim=feature_dim, hidden_sizes=hidden_sizes)
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, features: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        modulated = self.film(features, alpha)
        return self.classifier(modulated)


class MLP(nn.Module):
    """Compatibility MLP for CMNIST/DGIL-style algorithms."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(n_classes)),
        )

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | float | None = None) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class FiLMedMLP(nn.Module):
    """FiLM-conditioned MLP where `alpha` modulates hidden representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        *,
        dropout: float = 0.0,
        film_dim: int = 1,
    ):
        super().__init__()
        self.film_dim = int(film_dim)
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.film = FiLMLayer(feature_dim=int(hidden_dim), hidden_sizes=[max(16, int(hidden_dim // 4))])
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), int(n_classes))

    def _alpha_batch(self, x: torch.Tensor, alpha: torch.Tensor | float | None) -> torch.Tensor:
        if alpha is None:
            return torch.zeros((x.size(0), self.film_dim), device=x.device, dtype=x.dtype)
        if not torch.is_tensor(alpha):
            return torch.full((x.size(0), self.film_dim), float(alpha), device=x.device, dtype=x.dtype)
        alpha = alpha.to(device=x.device, dtype=x.dtype)
        if alpha.ndim == 0:
            return alpha.view(1, 1).expand(x.size(0), self.film_dim)
        if alpha.ndim == 1:
            if alpha.size(0) == x.size(0):
                return alpha.view(-1, 1).expand(-1, self.film_dim)
            return alpha.view(1, -1).expand(x.size(0), -1)
        return alpha

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | float | None = None) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = self.film(h, self._alpha_batch(h, alpha))
        h = self.dropout(h)
        return self.fc2(h)


class CNN(nn.Module):
    """Small CNN compatibility model for CMNIST experiments."""

    def __init__(self, input_shape: Sequence[int], n_classes: int):
        super().__init__()
        c = int(input_shape[0])
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, int(n_classes))

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | float | None = None) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avg(x).view(x.size(0), -1)
        return self.fc(x)


class FiLMedResNetClassifier(nn.Module):
    """ResNet classifier with optional FiLM modulation of pooled features."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = False,
        film_hidden_sizes: Sequence[int] | None = None,
        backbone_name: str = "resnet18",
    ):
        super().__init__()
        backbone_key = str(backbone_name).lower()
        if backbone_key == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        elif backbone_key == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported FiLMedResNet backbone '{backbone_name}'.")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = int(backbone.fc.in_features)
        self.film = FiLMLayer(
            feature_dim=self.feature_dim,
            hidden_sizes=film_hidden_sizes or [max(64, self.feature_dim // 8)],
        )
        self.classifier = nn.Linear(self.feature_dim, int(num_classes))

    def _alpha_batch(self, x: torch.Tensor, alpha: torch.Tensor | float | None) -> torch.Tensor:
        if alpha is None:
            return torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        if not torch.is_tensor(alpha):
            return torch.full((x.size(0), 1), float(alpha), device=x.device, dtype=x.dtype)
        alpha = alpha.to(device=x.device, dtype=x.dtype)
        if alpha.ndim == 0:
            return alpha.view(1, 1).expand(x.size(0), 1)
        if alpha.ndim == 1:
            if alpha.size(0) == x.size(0):
                return alpha.view(-1, 1)
            return alpha.view(1, -1)[:, :1].expand(x.size(0), 1)
        if alpha.size(0) == 1 and x.size(0) > 1:
            return alpha[:, :1].expand(x.size(0), 1)
        return alpha[:, :1]

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | float | None = None) -> torch.Tensor:
        features = self.feature_extractor(x).flatten(start_dim=1)
        if alpha is not None:
            features = self.film(features, self._alpha_batch(features, alpha))
        return self.classifier(features)
