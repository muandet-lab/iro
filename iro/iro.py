import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import beta
import copy

from iro.utility.iro_utils import AggregationFunction as aggregation_function

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
    def __init__(self, env_dict, risk_measure="cvar-diff"):
        self.env_dict = env_dict
        self.loss_fn = torch.nn.MSELoss()
        self.aggregator = aggregation_function(name=risk_measure)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_norm(self, model):
        # Calculate norm of gradients
        total_norm = 0
        for param in model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm**2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def aggregated_objective(self, model, a, b, num_samples=5):
        uniform_samples = dist.Uniform(0, 1).sample((num_samples,))
        alphas = []
        for unif_sample in uniform_samples:
            t_unif = unif_sample.detach().to(self.device).requires_grad_(True)
            alphas.append(IcdfBetaScaler.apply(t_unif, a, b))
        risk_estimates = []
        for alpha in alphas:
            risks = []
            for e in self.env_dict.keys():
                x, y = self.env_dict[e]['x'].to(self.device), self.env_dict[e]['y'].to(self.device)
                x.requires_grad, y.requires_grad = False, False
                risks.append(self.loss_fn(y, model(x, alpha)))
            risks = torch.stack(risks)
            risk_estimates.append(self.aggregator.aggregate(risks, alpha))
        risk_estimates = torch.stack(risk_estimates)
        average_risk = torch.mean(risk_estimates)
        return average_risk
    
    def optimize(self, model, risk_measure="cvar"):
        # Allow dynamic choice of risk measure
        self.aggregator = aggregation_function(name=risk_measure)
        
        for param in model.parameters():
            param.requires_grad = False
        a = torch.tensor([1.0], requires_grad=True, device=self.device, dtype=torch.float32)
        b = torch.tensor([1.0], requires_grad=True, device=self.device, dtype=torch.float32)
        optimizer_dist = torch.optim.Adam([a, b], lr=0.01)
        num_epochs = 10
        for epoch in range(num_epochs):
            avg_risk = self.aggregated_objective(model, a, b)  # Use the selected risk measure
            avg_risk.backward()
            optimizer_dist.step()
            optimizer_dist.zero_grad()
        return a.detach().item(), b.detach().item()

class ARM_Regression:
    def __init__(self, name, experiment="1D_linear", risk_measure="cvar"):
        # Allow dynamic choice of risk measure
        self.aggregator = aggregation_function(name=risk_measure)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_risk_h(self, alpha, h, env_dict):
        # Generalize to compute risk based on the selected measure
        loss_fn = torch.nn.MSELoss()        
        risks = []
        for e in env_dict.keys():
            output = h(env_dict[e]['x'].to(self.device), alpha.to(self.device))
            risks.append(loss_fn(output, env_dict[e]['y'].to(self.device)))
        risks = torch.stack(risks)
        risk = self.aggregator.aggregate(risks, alpha)
        return risk

    def evaluate_cvar(self, alpha, h, env_dict):
        """Evaluate aggregated risk on held-out environments."""
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(float(alpha), dtype=torch.float32, device=self.device)
        else:
            alpha = alpha.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            risk = self.compute_risk_h(alpha, h, env_dict)
        return float(risk.detach().cpu().item())
    
    def fit_h(self, h, env_dict, a, b, num_epochs=30, risk_measure="cvar"):
        # Allow dynamic choice of risk measure during fitting
        self.aggregator = aggregation_function(name=risk_measure)
        loss_fn = torch.nn.MSELoss()
        alphas = np.random.beta(a=a, b=b, size=5)
        alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)
        learning_rate = 0.1
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            avg_risk = torch.mean(torch.stack([self.compute_risk_h(alpha, h, env_dict) for alpha in alphas]))
            avg_risk.backward()
            optimizer.step()
            optimizer.zero_grad()
        return 
    
    def fit_f(self, f, env_dict, alpha, num_epochs=100):        
        learning_rate = 0.1
        loss_fn = torch.nn.MSELoss()        
        optimizer = torch.optim.Adam(f.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            risks = []
            for e in env_dict.keys():
                x, y = env_dict[e]['x'].to(self.device), env_dict[e]['y'].to(self.device) 
                risks.append(loss_fn(f(x),y))
            risks = torch.stack(risks)
            cvar = self.aggregator.aggregate(risks, alpha)
            cvar.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {cvar.item()}")
        return 
    
    def fit_h_as_f(self, h, env_dict, alpha, num_epochs=100, risk_measure="cvar"): 
        # Allow dynamic choice of risk measure
        self.aggregator = aggregation_function(name=risk_measure)
        t_alpha = torch.tensor(alpha).to(self.device)
        learning_rate = 0.1
        loss_fn = torch.nn.MSELoss()        
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            risks = []
            for e in env_dict.keys():
                x, y = env_dict[e]['x'].to(self.device), env_dict[e]['y'].to(self.device) 
                risks.append(loss_fn(h(x, t_alpha), y))
            risks = torch.stack(risks)
            avg_risk = self.aggregator.aggregate(risks, alpha)
            avg_risk.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_risk.item()}")
        return
    
    def fit_h_pareto(self, h, env_dict, num_epochs=30, risk_measure="cvar"):
        # Allow dynamic choice of risk measure
        self.aggregator = aggregation_function(name=risk_measure)
        loss_fn = torch.nn.MSELoss()
        learning_rate = 0.1
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        p_min = Pareto_distribution(env_dict, risk_measure=risk_measure)
        for epoch in range(num_epochs):
            a, b = p_min.optimize(copy.deepcopy(h), risk_measure=risk_measure)
            alphas = np.random.beta(a, b, size=5)
            alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)
            avg_risk = torch.mean(torch.stack([self.compute_risk_h(alpha, h, env_dict) for alpha in alphas]))
            avg_risk.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_risk.item()}")
        return


# Backward-compatible alias used by package-level imports.
AggregationFunction = aggregation_function
