import copy
import torch
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from scipy.stats import beta

from .kde import Nonparametric


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
            local_grad_alpha = (torch.quantile(risks, alpha + diff) - torch.quantile(risks, alpha)) / diff
        elif alpha + diff > 1.0:
            local_grad_alpha = (torch.quantile(risks, alpha) - torch.quantile(risks, alpha - diff)) / diff
        else:
            local_grad_alpha = (torch.quantile(risks, alpha + diff) - torch.quantile(risks, alpha - diff)) / (2.0 * diff)
        grad_alpha = grad_output * local_grad_alpha
        return grad_risks, grad_alpha


class aggregation_function:
    """Aggregate risks via CVaR / ESRM / EVaR etc."""

    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evar_num_t = 64

    def aggregate(self, risks, alpha) -> float:
        if self.name == "cvar":
            return self.cvar(risks, alpha)
        if self.name == "cvar-full":
            return self.cvar_full(risks, alpha)
        if self.name == "cvar-diff":
            return self.cvar_diff(risks, alpha)
        if self.name == "cvar-dist":
            return self.cvar_dist(risks, alpha)
        if self.name == "exponential":
            return self.exponential(risks, alpha)
        if self.name == "evar":
            return self.evar(risks, alpha)
        raise NotImplementedError("Currently, only CVaR/ESRM/EVaR are implemented.")

    def cvar_full(self, risks, alpha) -> float:
        var = Quantile.apply(risks, alpha)
        cvar_plus = risks[risks >= var].mean()
        lambda_alpha = ((risks <= var).sum().div(len(risks)) - alpha) / (1 - alpha)
        cvar = lambda_alpha * var + (1 - lambda_alpha) * cvar_plus
        return cvar

    def cvar_diff(self, risks, base_alpha) -> float:
        number_of_points = 5
        alphas = [(1 - base_alpha) * (i / number_of_points) + base_alpha for i in range(number_of_points)]
        quantiles = torch.stack([Quantile.apply(risks, alpha) for alpha in alphas])
        return quantiles.mean()

    def cvar(self, risks, alpha) -> float:
        var = torch.quantile(risks, alpha, interpolation="linear")
        cvar_val = risks[risks >= var].mean()
        return cvar_val

    def cvar_dist(self, risks, alpha) -> float:
        if 0 <= alpha <= 0.2:
            return self.cvar(risks, alpha)
        dist_np = Nonparametric()
        dist_np.estimate_parameters(risks)
        if 0.95 <= alpha <= 1:
            return dist_np.icdf(-1000)
        obser = torch.arange(alpha, 1.0, 10)
        cvar_val = torch.stack([dist_np.icdf(percentile) for percentile in obser]).mean()
        return cvar_val

    def exponential(self, risks, gamma) -> float:
        """Exponential spectral risk with Arrow-Pratt coefficient gamma > 0."""
        device = risks.device
        gamma = torch.as_tensor(gamma, dtype=risks.dtype, device=device)
        gamma = torch.clamp(gamma, min=1e-6)
        sorted_risks, _ = torch.sort(risks)
        n = sorted_risks.numel()
        if n == 0:
            raise ValueError("Risks tensor must contain at least one element.")
        u = (torch.arange(n, device=device, dtype=risks.dtype) + 0.5) / n
        numerator = gamma * torch.exp(-gamma * (1.0 - u))
        denom = torch.clamp(1.0 - torch.exp(-gamma), min=1e-6)
        phi = numerator / denom
        esrm = torch.dot(sorted_risks, phi) / n
        return esrm

    def evar(self, risks, alpha) -> float:
        """Entropic Value-at-Risk using a log-space grid over tilting parameter t."""
        if alpha >= 1.0:
            raise ValueError("Alpha must be < 1 for EVaR.")
        device = risks.device
        dtype = risks.dtype
        flat = risks.view(-1)
        if flat.numel() == 0:
            raise ValueError("Risks tensor must contain at least one element.")
        tail = torch.clamp(1.0 - alpha, min=1e-6)
        log_tail = torch.log(tail)
        log_n = torch.log(torch.tensor(float(flat.numel()), device=device, dtype=dtype))
        t_vals = torch.logspace(-2, 1, steps=self.evar_num_t, device=device, dtype=dtype)
        scaled = t_vals.view(-1, 1) * flat.view(1, -1)
        log_mgf = torch.logsumexp(scaled, dim=1) - log_n
        objs = (log_mgf - log_tail) / t_vals
        val, _ = torch.min(objs, dim=0)
        return val


class IcdfBetaScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(beta.ppf(x.item(), a.item(), b.item())).float().to(device)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff = 1e-5
        x_val = x.item()
        a_val = a.item()
        b_val = b.item()
        if x_val < diff:
            local_grad_x = torch.tensor([(beta.ppf(x_val + diff, a_val, b_val) - beta.ppf(x_val, a_val, b_val)) / diff]).float()
        elif x_val + diff > 1.0:
            local_grad_x = torch.tensor([(beta.ppf(x_val, a_val, b_val) - beta.ppf(x_val - diff, a_val, b_val)) / diff]).float()
        else:
            local_grad_x = torch.tensor([(beta.ppf(x_val + diff, a_val, b_val) - beta.ppf(x_val - diff, a_val, b_val)) / (2.0 * diff)]).float()

        if a_val < diff:
            local_grad_a = torch.tensor([(beta.ppf(x_val, a_val + diff, b_val) - beta.ppf(x_val, a_val, b_val)) / diff]).float()
        else:
            local_grad_a = torch.tensor([(beta.ppf(x_val, a_val + diff, b_val) - beta.ppf(x_val, a_val - diff, b_val)) / (2.0 * diff)]).float()

        if b_val < diff:
            local_grad_b = torch.tensor([(beta.ppf(x_val, a_val, b_val + diff) - beta.ppf(x_val, a_val, b_val)) / diff]).float()
        else:
            local_grad_b = torch.tensor([(beta.ppf(x_val, a_val, b_val + diff) - beta.ppf(x_val, a_val, b_val - diff)) / (2.0 * diff)]).float()
        grad_x = grad_output * local_grad_x.to(device)
        grad_a = grad_output * local_grad_a.to(device)
        grad_b = grad_output * local_grad_b.to(device)
        return grad_x, grad_a, grad_b


class Pareto_distribution:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.aggregator = aggregation_function(name="cvar-diff")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_param = torch.tensor([1.0, 1.0], requires_grad=True, device=self.device, dtype=torch.float32)

    def aggregated_objective(self, model, minibatches, num_samples=5):
        uniform_samples = dist.Uniform(0, 1).sample((num_samples,))
        uniform_samples.requires_grad = True
        alphas = []
        for t_unif in uniform_samples:
            alphas.append(IcdfBetaScaler.apply(t_unif, self.dist_param[0], self.dist_param[1]))
        cvar_estimates = []
        for alpha in alphas:
            risks = []
            for x, y in minibatches:
                t_alpha = torch.tile(alpha, (x.shape[0], 1))
                risks.append(self.loss_fn(model(x, t_alpha), y).reshape(1))
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
        self.dist_param = self.dist_param - 0.000001 * obj_grad
        return self.dist_param.detach().cpu().numpy()


class ArrowPrattDistribution:
    """Positive parameter distribution for exponential spectral risk measure."""

    def __init__(self, loss_fn, step_size=1e-6):
        self.loss_fn = loss_fn
        self.aggregator = aggregation_function(name="exponential")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_param = torch.tensor([0.0, 0.0], requires_grad=True, device=self.device, dtype=torch.float32)
        self.step_size = step_size

    def _current_params(self):
        loc = self.dist_param[0]
        scale = F.softplus(self.dist_param[1]) + 1e-3
        return loc, scale

    def aggregated_objective(self, model, minibatches, num_samples=5):
        loc, scale = self._current_params()
        gamma_dist = dist.LogNormal(loc, scale)
        gamma_samples = gamma_dist.rsample((num_samples, 1))
        esrm_estimates = []
        for gamma in gamma_samples:
            risks = []
            for x, y in minibatches:
                t_gamma = gamma.repeat(x.shape[0], 1)
                risks.append(self.loss_fn(model(x, t_gamma), y).reshape(1))
            risks = torch.cat(risks)
            esrm_estimates.append(self.aggregator.aggregate(risks, gamma))
        esrm_estimates = torch.stack(esrm_estimates)
        return esrm_estimates.mean()

    def update(self, model, minibatches):
        avg_esrm = self.aggregated_objective(model, minibatches)
        params = [p for p in model.parameters()]
        grads = torch.autograd.grad(avg_esrm, params, retain_graph=True, create_graph=True)
        grad_norms = [torch.norm(grad, p=2) for grad in grads if grad is not None]
        total_norm = torch.norm(torch.stack(grad_norms), p=2)
        obj_grad = torch.autograd.grad(total_norm, self.dist_param)[0]
        with torch.no_grad():
            self.dist_param -= self.step_size * obj_grad
        self.dist_param.requires_grad_(True)
        loc, scale = self._current_params()
        return float(loc.detach().cpu()), float(scale.detach().cpu())


class EntropicDistribution:
    """Beta-distributed CVaR-style sampler for EVaR risk aversion."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.aggregator = aggregation_function(name="evar")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_param = torch.tensor([1.0, 1.0], requires_grad=True, device=self.device, dtype=torch.float32)

    def aggregated_objective(self, model, minibatches, num_samples=5):
        uniform_samples = dist.Uniform(0, 1).sample((num_samples,))
        uniform_samples.requires_grad = True
        alphas = []
        for t_unif in uniform_samples:
            alphas.append(IcdfBetaScaler.apply(t_unif, self.dist_param[0], self.dist_param[1]))
        evar_estimates = []
        for alpha in alphas:
            risks = []
            for x, y in minibatches:
                t_alpha = torch.tile(alpha, (x.shape[0], 1))
                risks.append(self.loss_fn(model(x, t_alpha), y).reshape(1))
            risks = torch.cat(risks)
            evar_estimates.append(self.aggregator.aggregate(risks, alpha))
        evar_estimates = torch.stack(evar_estimates)
        return torch.mean(evar_estimates)

    def update(self, model, minibatches):
        avg_evar = self.aggregated_objective(model, minibatches)
        params = [p for p in model.parameters()]
        grads = torch.autograd.grad(avg_evar, params, retain_graph=True, create_graph=True)[0]
        grad_norms = [torch.norm(grad, p=2) for grad in grads if grad is not None]
        total_norm = torch.norm(torch.stack(grad_norms), p=2)
        obj_grad = torch.autograd.grad(total_norm, self.dist_param)[0]
        self.dist_param = self.dist_param - 0.000001 * obj_grad
        return self.dist_param.detach().cpu().numpy()
