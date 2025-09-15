import numpy as np

class aggregation_function:
    """ This class aggregates the risks using different risk measures. """

    def __init__(self, name: str):
        """
        Construct the aggregation function.

        Parameters
        ----------
        name : str
            the name of the function to aggregate the risks
            (options: "cvar", "var", "entropic", "mean", 
                      "worst_case", "median", "variance")
        """       
        self.name = name.lower()
    
    def aggregate(self, risks: list, alpha: float = 0.05, eta: float = 1.0) -> float:
        """
        Aggregates a list of risks according to an aggregation function rho. 

        Parameters
        ----------
        risks : list
            List of risk values.
        alpha : float, optional
            Confidence/risk level (for VaR, CVaR).
        eta : float, optional
            Risk aversion parameter (for entropic risk).

        Returns
        -------
        aggregated_risks : float

        Examples
        --------
        >>> aggregation_function(name="cvar").aggregate([1,2,3,4], alpha=0.1)
        """
        risks = np.array(risks)

        if self.name == "cvar":
            return self.__cvar__(risks, alpha)
        elif self.name == "var":
            return self.__var__(risks, alpha)
        elif self.name == "entropic":
            return self.__entropic__(risks, eta)
        elif self.name == "mean":
            return self.__mean__(risks)
        elif self.name == "worst_case":
            return self.__worst_case__(risks)
        elif self.name == "median":
            return self.__median__(risks)
        elif self.name == "variance":
            return self.__variance__(risks)
        else:
            raise NotImplementedError(f"Aggregation function '{self.name}' not implemented.")
    
    def __cvar__(self, risks: np.ndarray, alpha: float) -> float:
        """Conditional Value-at-Risk (CVaR)."""
        var = np.percentile(risks, alpha * 100)
        return risks[risks >= var].mean()

    def __var__(self, risks: np.ndarray, alpha: float) -> float:
        """Value-at-Risk (VaR)."""
        return np.percentile(risks, alpha * 100)

    def __entropic__(self, risks: np.ndarray, eta: float) -> float:
        """Entropic risk measure: (1/eta) * log( E[exp(eta * risk)] )."""
        return (1.0 / eta) * np.log(np.mean(np.exp(eta * risks)))

    def __mean__(self, risks: np.ndarray) -> float:
        """Mean risk (expected value)."""
        return risks.mean()

    def __worst_case__(self, risks: np.ndarray) -> float:
        """Worst-case risk (maximal loss)."""
        return risks.max()

    def __median__(self, risks: np.ndarray) -> float:
        """Median risk (robust alternative to mean)."""
        return np.median(risks)

    def __variance__(self, risks: np.ndarray) -> float:
        """Variance as a risk measure (penalizing spread)."""
        return risks.var()