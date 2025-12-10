"""
Risk aggregation demo on Iris across many synthetic domains.

Highlights:
  - Generates subtle domain shifts
  - Trains per-domain models
  - Aggregates OOD risks using CVaR / ESRM / EVaR with sensible parameter sampling
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Ensure local repo src/ is on path (avoid picking up old installs)
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iro.aggregation.aggregators import AggregationFunction

NUM_DOMAINS = 40
SAMPLES_PER_DOMAIN = 12
MAX_CLASSES = 3
ALPHA_GRID = np.linspace(0.6, 0.98, 8)
GAMMA_GRID = np.exp(np.linspace(-3, 1.5, 8))  # Arrow-Pratt coeffs


def generate_domain_data(iris_data, n_samples, domain_id):
    """Subsample and nudge features to simulate domain shift."""
    X_original, y = iris_data.data, iris_data.target
    X_sub, _, y_sub, _ = train_test_split(X_original, y, train_size=n_samples, random_state=domain_id)
    shift = np.sin(domain_id / 7.0) * 0.12
    noise = np.random.normal(loc=shift, scale=0.04, size=X_sub.shape)
    return X_sub + noise, y_sub


def load_and_split_data(n_domains, n_samples):
    iris = load_iris()
    domain_data = {}
    for i in range(n_domains):
        X, y = generate_domain_data(iris, n_samples, i)
        domain_data[f"D{i}"] = {"X": X, "y": y}
    return domain_data


def train_domain_models(domain_data):
    """Train a simple logistic regression on each domain."""
    domain_models = {}
    for domain_name, data in tqdm(domain_data.items(), desc="Training domain models"):
        model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        model.fit(data["X"], data["y"])
        domain_models[domain_name] = model
    return domain_models


def calculate_risk_matrix(domain_data, domain_models):
    """Risk matrix R[i,j] = MSE(model_i on domain_j)."""
    domain_names = list(domain_data.keys())
    num_domains = len(domain_names)
    risk_matrix = np.zeros((num_domains, num_domains))

    for i in tqdm(range(num_domains), desc="Computing risk matrix"):
        model_i = domain_models[domain_names[i]]
        for j in range(num_domains):
            data_j = domain_data[domain_names[j]]
            y_pred_proba = model_i.predict_proba(data_j["X"])
            num_samples_j = data_j["X"].shape[0]
            y_pred_full = np.zeros((num_samples_j, MAX_CLASSES))
            predicted_class_indices = model_i.classes_.astype(int)
            y_pred_full[:, predicted_class_indices] = y_pred_proba
            y_true_one_hot = np.eye(MAX_CLASSES)[data_j["y"]]
            risk_matrix[i, j] = mean_squared_error(y_true_one_hot, y_pred_full)

    return torch.from_numpy(risk_matrix).float()


def sample_risk_params(num_samples=32):
    """Draw CVaR alphas (Beta prior) and ESRM gammas (log-normal prior)."""
    alphas = np.random.beta(a=2.0, b=5.0, size=num_samples)
    gammas = np.random.lognormal(mean=-0.5, sigma=0.8, size=num_samples)
    return alphas, gammas


def summarize_aggregations(risk_vector):
    """Return a dataframe of aggregated risks under multiple measures."""
    records = []
    alphas, gammas = sample_risk_params(num_samples=24)
    for alpha in alphas:
        agg = AggregationFunction("cvar").aggregate(risk_vector, alpha=alpha).item()
        records.append({"measure": "cvar", "param": alpha, "value": agg})
    for gamma in gammas:
        agg = AggregationFunction("exponential").aggregate(risk_vector, gamma=gamma).item()
        records.append({"measure": "esrm", "param": gamma, "value": agg})
    # EVaR: reuse alpha slot as tail level
    for alpha in alphas:
        agg = AggregationFunction("evar").aggregate(risk_vector, alpha=alpha).item()
        records.append({"measure": "evar", "param": alpha, "value": agg})
    return pd.DataFrame.from_records(records)


def visualize_tradeoffs(risk_vector):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    alphas = ALPHA_GRID
    gammas = GAMMA_GRID
    agg_cvar = [AggregationFunction("cvar").aggregate(risk_vector, alpha=a).item() for a in alphas]
    agg_esrm = [AggregationFunction("exponential").aggregate(risk_vector, gamma=g).item() for g in gammas]
    agg_evar = [AggregationFunction("evar").aggregate(risk_vector, alpha=a).item() for a in alphas]
    ax.plot(alphas, agg_cvar, label="CVaR (alpha)", marker="o")
    ax.plot(alphas, agg_evar, label="EVaR (alpha)", marker="x")
    ax2 = ax.twinx()
    ax2.plot(gammas, agg_esrm, label="ESRM (gamma)", color="tab:red", marker="s")
    ax.set_xlabel("alpha (CVaR/EVaR)")
    ax.set_ylabel("aggregated risk")
    ax2.set_ylabel("aggregated risk (ESRM)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Risk aggregation trade-offs on one model's OOD risks")
    plt.tight_layout()
    plt.show()


def main():
    domain_data = load_and_split_data(NUM_DOMAINS, SAMPLES_PER_DOMAIN)
    domain_models = train_domain_models(domain_data)
    risk_matrix = calculate_risk_matrix(domain_data, domain_models)
    # pick a model's OOD risks
    risk_vector = risk_matrix[0]
    df_summary = summarize_aggregations(risk_vector)
    print(df_summary.groupby("measure")["value"].describe())
    visualize_tradeoffs(risk_vector)


if __name__ == "__main__":
    main()
