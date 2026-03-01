from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from iro.utility.iro_utils import AggregationFunction as aggregation_function

class ARM_Regression(BaseEstimator, RegressorMixin):
    def __init__(self, name, experiment="1D_linear"):
        self.aggregator = aggregation_function(name=name)
        self.experiment = experiment
        self.coef = None

    def _apply_f(self, f, x, theta):
        x = np.asarray(x).reshape(-1, x.shape[-1])
        theta = np.atleast_1d(theta).reshape(-1, 1)
        y_pred = x @ theta
        return y_pred.reshape(-1)

    def fit(self, f, env_dict, alpha=None, eta=None):
        """
        Fit ARM regression by minimizing the chosen risk aggregation.

        Parameters
        ----------
        f : callable
            Hypothesis function.
        env_dict : dict
            Dictionary of training environments, each with 'x' and 'y'.
        alpha : float, optional
            Tail probability (for CVaR, VaR).
        eta : float, optional
            Risk aversion (for entropic risk).
        """
        d = env_dict[0]['x'].shape[1]

        def return_risks(env_dict):
            return lambda coefs: [
                mean_squared_error(
                    env_dict[e]['y'].reshape(-1),
                    self._apply_f(None, env_dict[e]['x'], coefs).reshape(-1)
                )
                for e in env_dict.keys()
            ]

        parameterized_risks = return_risks(env_dict)

        def aggregated_risk_fn(coefs):
            return self.aggregator.aggregate(
                parameterized_risks(coefs),
                alpha=alpha if alpha is not None else 0.05,
                eta=eta if eta is not None else 1.0
            )

        self.coef = minimize(
            aggregated_risk_fn,
            x0=np.random.uniform(0, 1, d)
        ).x

        return self.coef

    def fit_grad(self, f, env_dict, alpha=None, eta=None):
        """
        Gradient-based alternative using PyTorch autograd.
        """
        learning_rate = 0.01
        num_epochs = 200
        d = env_dict[0]['x'].shape[1]
        loss_fn = torch.nn.MSELoss()

        theta = torch.rand(d, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

        for epoch in range(num_epochs):
            risks = torch.stack([
                loss_fn(
                    torch.tensor(env_dict[e]['y'], dtype=torch.float32).view(-1),
                    f(torch.tensor(env_dict[e]['x'], dtype=torch.float32),
                      theta if theta.ndim > 0 else theta.unsqueeze(0)).view(-1)
                )
                for e in env_dict.keys()
            ])
            agg_risk = self.aggregator.aggregate(
                risks.detach().numpy(),
                alpha=alpha if alpha is not None else 0.05,
                eta=eta if eta is not None else 1.0
            )
            agg_risk = torch.tensor(agg_risk, requires_grad=True)

            optimizer.zero_grad()
            agg_risk.backward()
            optimizer.step()
            scheduler.step()

        return theta.detach().numpy()
