import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from .kde import Nonparametric
from scipy.stats import beta
import copy


class Quantile(torch.autograd.Function):

    @staticmethod
    def forward(ctx, risks, alpha):
        q = torch.quantile(risks, alpha)
        ctx.save_for_backward(risks, alpha, q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        risks, alpha, q = ctx.saved_tensors
        diff = 1e-5
        grad_risks = grad_output * (risks == q).int()
        if alpha < diff:
            local_grad_alpha = (torch.quantile(risks, alpha+diff) - torch.quantile(risks, alpha))/diff
        elif alpha + diff > 1.0:
            local_grad_alpha = (torch.quantile(risks, alpha) - torch.quantile(risks, alpha-diff))/diff
        else:
            local_grad_alpha = (torch.quantile(risks, alpha+diff) - torch.quantile(risks, alpha-diff)) / (2.0 * diff)
        grad_alpha = grad_output * local_grad_alpha
        return grad_risks, grad_alpha

class AggregationFunction:
    """Aggregate risks with legacy and generic risk measures."""

    def __init__(self, name: str, custom_func=None):
        self.name = name.lower()
        self.custom_func = custom_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _to_tensor(risks):
        if not torch.is_tensor(risks):
            return torch.tensor(risks, dtype=torch.float32)
        return risks.to(dtype=torch.float32)

    @staticmethod
    def _alpha_tensor(alpha, risks: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(alpha):
            return alpha.to(device=risks.device, dtype=risks.dtype)
        return torch.tensor(float(alpha), dtype=risks.dtype, device=risks.device)

    def aggregate(self, risks, alpha: float = 0.05, eta: float = 1.0, weights=None, xi: float = 1.0, **kwargs):
        risks = self._to_tensor(risks)

        if self.name == "custom" and self.custom_func is not None:
            return self.custom_func(risks, **kwargs)
        if self.name == "cvar":
            return self.cvar(risks, alpha)
        if self.name == "var":
            return self.var(risks, alpha)
        if self.name == "cvar-full":
            return self.cvar_full(risks, alpha)
        if self.name == "cvar-diff":
            return self.cvar_diff(risks, alpha)
        if self.name == "cvar-dist":
            return self.cvar_dist(risks, alpha)
        if self.name == "entropic":
            return self.entropic(risks, eta)
        if self.name == "mean":
            if weights is None:
                return self.mean(risks)
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=risks.dtype, device=risks.device)
            else:
                weights = weights.to(dtype=risks.dtype, device=risks.device)
            if weights.shape != risks.shape:
                raise ValueError("Weights and risks must have the same shape.")
            return torch.dot(weights, risks)
        if self.name == "worst_case":
            return self.worst_case(risks)
        if self.name == "median":
            return self.median(risks)
        if self.name == "variance":
            return self.variance(risks)
        if self.name == "ph":
            return self.ph(risks, xi)
        if self.name == "wang":
            return self.wang(risks, alpha)
        raise NotImplementedError(f"Aggregation function '{self.name}' is not implemented.")

    def var(self, risks, alpha) -> torch.Tensor:
        alpha_t = self._alpha_tensor(alpha, risks)
        if torch.any((alpha_t < 0) | (alpha_t > 1)):
            raise ValueError("Alpha must be between 0 and 1 (inclusive).")
        return torch.quantile(risks, alpha_t, interpolation="linear")

    def cvar(self, risks, alpha) -> torch.Tensor:
        var = self.var(risks, alpha)
        selected = risks[risks >= var]
        if selected.numel() == 0:
            return var
        return selected.mean()

    def cvar_full(self, risks, alpha) -> torch.Tensor:
        alpha_t = self._alpha_tensor(alpha, risks)
        var = Quantile.apply(risks, alpha_t)
        cvar_plus = risks[risks >= var].mean()
        lambda_alpha = ((risks <= var).sum().div(len(risks)) - alpha_t) / (1 - alpha_t)
        return lambda_alpha * var + (1 - lambda_alpha) * cvar_plus

    def cvar_diff(self, risks, base_alpha) -> torch.Tensor:
        base_alpha_t = self._alpha_tensor(base_alpha, risks)
        number_of_points = 5
        alphas = [(1 - base_alpha_t) * (i / number_of_points) + base_alpha_t for i in range(number_of_points)]
        quantiles = torch.stack([Quantile.apply(risks, alpha) for alpha in alphas])
        return quantiles.mean()

    def cvar_dist(self, risks, alpha) -> torch.Tensor:
        alpha_value = float(self._alpha_tensor(alpha, risks).detach().cpu().item())
        if 0 <= alpha_value <= 0.2:
            return self.cvar(risks, alpha_value)
        np_dist = Nonparametric()
        np_dist.estimate_parameters(risks)
        if 0.95 <= alpha_value <= 1:
            return np_dist.icdf(-1000)
        obser = torch.arange(alpha_value, 1.0, 10)
        return torch.stack([np_dist.icdf(percentile) for percentile in obser]).mean()

    def entropic(self, risks, eta: float) -> torch.Tensor:
        return (1.0 / eta) * torch.log(torch.mean(torch.exp(eta * risks)))

    def mean(self, risks) -> torch.Tensor:
        return risks.mean()

    def worst_case(self, risks) -> torch.Tensor:
        return risks.max()

    def median(self, risks) -> torch.Tensor:
        return torch.quantile(risks, 0.5, interpolation="linear")

    def variance(self, risks) -> torch.Tensor:
        return torch.mean((risks - risks.mean()) ** 2)

    def ph(self, risks, xi: float) -> torch.Tensor:
        return torch.mean(risks ** xi)

    def wang(self, risks, alpha: float) -> torch.Tensor:
        def norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))

        def norm_ppf(p):
            return torch.sqrt(torch.tensor(2.0, dtype=p.dtype, device=p.device)) * torch.erfinv(2 * p - 1)

        alpha_t = self._alpha_tensor(alpha, risks)
        risks_sorted, _ = torch.sort(risks)
        n = risks_sorted.numel()
        i = torch.arange(1, n + 1, dtype=risks.dtype, device=risks.device)
        weights = norm_cdf(norm_ppf(i / (n + 1)) + norm_ppf(1 - alpha_t))
        weights = weights / weights.sum()
        return torch.sum(weights * risks_sorted)

class IcdfBetaScaler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(beta.ppf(x.item(), a.item(), b.item())).float().to(device)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff = 1e-5
        x = x.item()
        a = a.item()
        b = b.item()
        if x < diff:
            local_grad_x = torch.tensor([(beta.ppf(x+diff, a, b) - beta.ppf(x, a, b)) / diff]).float()
        elif x + diff > 1.0:
            local_grad_x = torch.tensor([(beta.ppf(x, a, b) - beta.ppf(x-diff, a, b)) / diff]).float()
        else:
            local_grad_x = torch.tensor([(beta.ppf(x+diff, a, b) - beta.ppf(x-diff, a, b)) / (2.0 * diff)]).float()

        if a < diff:
            local_grad_a = torch.tensor([(beta.ppf(x, a+diff, b) - beta.ppf(x, a, b)) / diff]).float()
        else:
            local_grad_a = torch.tensor([(beta.ppf(x, a+diff, b) - beta.ppf(x, a-diff, b)) / (2.0 * diff)]).float()

        if b < diff:
            local_grad_b = torch.tensor([(beta.ppf(x, a, b+diff) - beta.ppf(x, a, b)) / diff]).float()
        else:
            local_grad_b = torch.tensor([(beta.ppf(x, a, b+diff) - beta.ppf(x, a, b-diff)) / (2.0 * diff)]).float()
        grad_x = grad_output * local_grad_x.to(device)
        grad_a = grad_output * local_grad_a.to(device)
        grad_b = grad_output * local_grad_b.to(device)
        return grad_x, grad_a, grad_b

class Pareto_distribution:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.aggregator = aggregation_function(name="cvar-diff")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_param = torch.tensor([1.0,1.0], requires_grad=True, device=self.device, dtype=torch.float32)
    
    def aggregated_objective(self, model, minibatches, num_samples=5):
        ### reparameterization needed here.
        uniform_samples = dist.Uniform(0,1).sample((num_samples,))
        uniform_samples.requires_grad=True
        alphas = []
        for t_unif in uniform_samples:
            alphas.append(IcdfBetaScaler.apply(t_unif, self.dist_param[0], self.dist_param[1]))
        cvar_estimates = []
        for alpha in alphas:
            risks = []
            for x, y in minibatches:
                t_alpha = torch.tile(alpha,(x.shape[0],1))
                risks.append(self.loss_fn(model(x,t_alpha), y).reshape(1))
            risks = torch.cat(risks)
            cvar_estimates.append(self.aggregator.aggregate(risks, alpha))
        cvar_estimates = torch.stack(cvar_estimates)
        average_cvar = torch.mean(cvar_estimates)
        return average_cvar
    def update(self, model, minibatches):
        avg_cvar = self.aggregated_objective(model, minibatches)
        params = [p for p in model.parameters()]
        grads = torch.autograd.grad(avg_cvar, params, retain_graph=True, create_graph=True)[0]
        grad_norms = [torch.norm(grad, p=2) for grad in grads if grad is not None]
        total_norm = torch.norm(torch.stack(grad_norms), p=2)
        obj_grad = torch.autograd.grad(total_norm, self.dist_param)[0]
        # Adjust distribution parameters based on accumulated gradients
        self.dist_param = self.dist_param - 0.000001 * obj_grad
        return self.dist_param.detach().cpu().numpy()


# Backward-compatible alias used across DGIL-derived codepaths.
aggregation_function = AggregationFunction
