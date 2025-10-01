import torch

class AggregationFunction:
    """ This class aggregates the risks using different risk measures. """

    def __init__(self, name: str, custom_func: callable = None):
        """
        Construct the aggregation function.

        Parameters
        ----------
        name : str
            the name of the function to aggregate the risks
            (options: "cvar", "var", "entropic", "mean", 
                      "worst_case", "median", "variance",
                      "tail_var", "ph", "g_entropic", "wang", "superhedging", "custom")
        custom_func : callable, optional
            A user-defined function to aggregate risks when name is "custom".
            The function should accept risks as the first argument and any additional
            keyword arguments as needed.
        """       
        self.name = name.lower()
        self.custom_func = custom_func
    
    def aggregate(self, risks, alpha: float = 0.05, eta: float = 1.0, weights=None, xi: float = 1.0, **kwargs) -> float:
        """
        Aggregates a list of risks according to an aggregation function rho. 

        Parameters
        ----------
        risks : list or array or torch.Tensor
            List of risk values.
        alpha : float, optional
            Confidence/risk level (for VaR, CVaR, Tail VaR).
        eta : float, optional
            Risk aversion parameter (for entropic risk).
        weights : list or array or torch.Tensor, optional
            Weights for linear aggregation (for mean).
        xi : float, optional
            Parameter for Proportional Hazard risk measure.
        **kwargs : additional keyword arguments
            Additional parameters passed to custom aggregation functions.

        Returns
        -------
        aggregated_risks : float

        Examples
        --------
        >>> aggregation_function(name="cvar").aggregate([1,2,3,4], alpha=0.1)
        """
        if not torch.is_tensor(risks):
            risks = torch.tensor(risks, dtype=torch.float32)
        else:
            risks = risks.to(dtype=torch.float32)

        if self.name == "custom" and self.custom_func is not None:
            return self.custom_func(risks, **kwargs)
        if self.name == "cvar":
            return self.__cvar__(risks, alpha)
        elif self.name == "var":
            return self.__var__(risks, alpha)
        elif self.name == "entropic":
            return self.__entropic__(risks, eta)
        elif self.name == "mean":
            if weights is None:
                weights = torch.ones_like(risks) / risks.numel()
            else:
                if not torch.is_tensor(weights):
                    weights = torch.tensor(weights, dtype=torch.float32)
                else:
                    weights = weights.to(dtype=torch.float32)
            if weights.shape != risks.shape:
                raise ValueError("Weights and risks must have the same shape.")
            return torch.dot(weights, risks)
        elif self.name == "worst_case":
            return self.__worst_case__(risks)
        elif self.name == "median":
            return self.__median__(risks)
        elif self.name == "variance":
            return self.__variance__(risks)
        elif self.name == "tail_var":
            return self.__tail_var__(risks, alpha)
        elif self.name == "ph":
            return self.__ph__(risks, xi)
        elif self.name == "g_entropic":
            return self.__g_entropic__(risks, eta)
        elif self.name == "wang":
            return self.__wang__(risks, alpha)
        elif self.name == "soft_cvar":
            return self.__soft_cvar__(risks, alpha, eta)
        else:
            raise NotImplementedError(f"Aggregation function '{self.name}' not implemented.")
    
    def __cvar__(self, risks: torch.Tensor, alpha: float) -> torch.Tensor:
        """Conditional Value-at-Risk (CVaR)."""
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1 (inclusive).")
        var = self.__var__(risks, alpha)
        mask = risks >= var
        selected = risks[mask]
        if selected.numel() == 0:
            return var
        return selected.mean()

    def __var__(self, risks: torch.Tensor, alpha: float) -> torch.Tensor:
        """Value-at-Risk (VaR)."""
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1 (inclusive).")
        k = int(alpha * risks.numel())
        if k == 0:
            k = 1
        sorted_risks, _ = torch.sort(risks)
        return sorted_risks[k-1]

    def __entropic__(self, risks: torch.Tensor, eta: float) -> torch.Tensor:
        """Entropic risk measure: (1/eta) * log( E[exp(eta * risk)] )."""
        return (1.0 / eta) * torch.log(torch.mean(torch.exp(eta * risks)))

    def __mean__(self, risks: torch.Tensor) -> torch.Tensor:
        """Mean risk (expected value)."""
        return risks.mean()

    def __worst_case__(self, risks: torch.Tensor) -> torch.Tensor:
        """Worst-case risk (maximal loss)."""
        return risks.max()

    def __median__(self, risks: torch.Tensor) -> torch.Tensor:
        """Median risk (robust alternative to mean)."""
        sorted_risks, _ = torch.sort(risks)
        n = sorted_risks.numel()
        if n % 2 == 1:
            return sorted_risks[n // 2]
        else:
            return 0.5 * (sorted_risks[n // 2 - 1] + sorted_risks[n // 2])

    def __variance__(self, risks: torch.Tensor) -> torch.Tensor:
        """Variance as a risk measure (penalizing spread)."""
        mean = risks.mean()
        return ((risks - mean) ** 2).mean()

    def __tail_var__(self, risks: torch.Tensor, alpha: float) -> torch.Tensor:
        """Tail Value at Risk using g(x) = min(x/alpha, 1)."""
        g = torch.minimum(risks / alpha, torch.tensor(1.0, dtype=risks.dtype, device=risks.device))
        return torch.mean(g * risks)

    def __ph__(self, risks: torch.Tensor, xi: float) -> torch.Tensor:
        """Proportional Hazard risk measure with transform g(x) = x^xi."""
        return torch.mean(risks ** xi)

    def __g_entropic__(self, risks: torch.Tensor, eta: float) -> torch.Tensor:
        """Placeholder for g-entropic risk measures."""
        # Placeholder implementation; user should override or extend
        return (1.0 / eta) * torch.log(torch.mean(torch.exp(eta * risks)))

    def __wang__(self, risks: torch.Tensor, alpha: float) -> torch.Tensor:
        """Wang risk measure using standard normal CDF and inverse."""
        # Implement standard normal CDF and inverse using torch.erf and erfinv
        def norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))

        def norm_ppf(p):
            # p should be in (0,1)
            return torch.sqrt(torch.tensor(2.0, dtype=p.dtype, device=p.device)) * torch.erfinv(2 * p - 1)

        risks_sorted, _ = torch.sort(risks)
        n = risks_sorted.numel()
        i = torch.arange(1, n + 1, dtype=risks.dtype, device=risks.device)
        weights = norm_cdf(norm_ppf(i / (n + 1)) + norm_ppf(1 - alpha))
        weights = weights / weights.sum()
        return torch.sum(weights * risks_sorted)
    
    def __soft_cvar__(self, risks: torch.Tensor, alpha: float, eta: float) -> torch.Tensor:
        """Differentiable smooth CVaR approximation using softmax weights."""
        n = risks.numel()
        k = max(1, int(alpha * n))
        # Sort risks ascending
        risks_sorted, _ = torch.sort(risks)
        # Use negative eta to emphasize larger risks (tail)
        weights_raw = torch.zeros_like(risks_sorted)
        weights_raw[:k] = eta * risks_sorted[:k]
        weights_raw[k:] = eta * risks_sorted[k:].min() - 1000.0  # very small weight for tail beyond k
        weights = torch.softmax(weights_raw, dim=0)
        return torch.sum(weights * risks_sorted)
    

    # want users to be able to define their own aggregation functions (could use self.name=="custom" and pass a function handle)