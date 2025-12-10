# src/iro/trainer.py
import time
from typing import Optional, Callable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .aggregation.aggregators import AggregationFunction

class Trainer:
    """Simple Trainer orchestrating dataset -> model -> risk aggregation -> optimizer.

    Core idea:
      - collect per-sample losses (reduction='none')
      - optionally split losses by domain (if dataset yields domain labels)
      - aggregate per-domain losses (e.g. mean per domain or cvar per domain)
      - combine domain risks using an AggregationFunction (can use cvar/ph/etc across domains)
      - backpropagate the final scalar risk
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable = None,
        device: torch.device = None,
        aggregator: AggregationFunction = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction="none")
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.aggregator = aggregator or AggregationFunction("mean")

        self.model.to(self.device)

    def _extract_domain(self, batch):
        """
        Expect batch to be either:
          - (x, y)  -> no domains
          - (x, y, domain)  -> domain present (domain can be tensor of ints or list)
        """
        if len(batch) == 2:
            x, y = batch
            domain = None
        elif len(batch) == 3:
            x, y, domain = batch
        else:
            raise ValueError("Unsupported batch format. Expected (x,y) or (x,y,domain).")
        return x, y, domain

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        alpha: float = 0.05,
        eta: float = 1.0,
        weights = None,
        xi: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, float]:
        self.model.train()
        losses = []
        start = time.time()
        for batch in dataloader:
            x, y, domain = self._extract_domain(batch)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            per_sample_losses = self.loss_fn(logits, y)  # tensor shape [B, ...] -> reduction='none' expected
            # flatten per-sample loss to 1D
            per_sample_losses = per_sample_losses.view(per_sample_losses.size(0), -1).mean(dim=1)

            if domain is None:
                # simple case: aggregate across batch directly
                risk = self.aggregator.aggregate(per_sample_losses, alpha=alpha, eta=eta, weights=weights, xi=xi, backend="torch")
            else:
                # domain-aware: expect domain as tensor of ints with same batch size
                if not isinstance(domain, torch.Tensor):
                    domain = torch.as_tensor(domain, device=self.device)
                domain = domain.to(self.device)
                unique_domains = torch.unique(domain)
                domain_risks = []
                for d in unique_domains:
                    mask = domain == d
                    if mask.sum().item() == 0:
                        continue
                    d_losses = per_sample_losses[mask]
                    # aggregate per domain: use simple mean (domain-level risk), or you may choose cvar per-domain by setting aggregator name
                    # Here we compute mean as domain internal aggregation, then we will combine across domains via aggregator
                    d_risk = torch.mean(d_losses)
                    domain_risks.append(d_risk)
                domain_risks_tensor = torch.stack(domain_risks)
                # combine domain risks using aggregator (this keeps autograd)
                risk = self.aggregator.aggregate(domain_risks_tensor, alpha=alpha, eta=eta, weights=weights, xi=xi, backend="torch")

            # backward and step
            risk.backward()
            self.optimizer.step()

            losses.append(risk.item() if isinstance(risk, float) or not torch.is_tensor(risk) else risk.detach().cpu().item())

        took = time.time() - start
        avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
        if verbose:
            print(f"Epoch {epoch}: avg aggregated risk = {avg_loss:.4f} (took {took:.1f}s)")
        return {"avg_aggregated_risk": float(avg_loss)}

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        **train_kwargs,
    ):
        for e in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch=e, **train_kwargs)