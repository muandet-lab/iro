import argparse
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from iro.cmnist_exp.vis.common import (
    load_algorithm_from_ckpt,
    parse_probabilities,
    peek_loss_name,
)
from iro.cmnist_exp.lib.iro_utils import Pareto_distribution, ArrowPrattDistribution
from iro.cmnist_exp.datasets import get_cmnist_datasets


def build_minibatches(envs, device):
    minibatches = []
    for env in envs:
        x = env.tensors[0].to(device)
        y = env.tensors[1].to(device)
        minibatches.append((x, y))
    return minibatches


def main():
    parser = argparse.ArgumentParser(description="Visualize learned risk-parameter distributions.")
    parser.add_argument("--iro_ckpt", required=True)
    parser.add_argument("--esrm_ckpt", required=True)
    parser.add_argument("--train_envs", type=str,
                        default="0.01,0.12,0.0,0.0,0.99,0.5,0.7,0.01,0.0,0.0,0.14")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_risk_distributions.png")
    args = parser.parse_args()

    train_ps = parse_probabilities(args.train_envs)
    device = torch.device(args.device)

    loss_name = peek_loss_name(args.iro_ckpt)
    train_envs = get_cmnist_datasets(
        args.data_dir,
        train_envs=tuple(train_ps),
        test_envs=tuple(),
        label_noise_rate=args.label_noise,
        cuda=(device.type == "cuda"),
        int_target=(loss_name != "nll"),
        subsample=not args.full_resolution
    )
    minibatches = build_minibatches(train_envs, device)
    input_shape = train_envs[0].tensors[0].size()[1:]

    iro_algorithm, _, iro_loss_fn, _, _ = load_algorithm_from_ckpt(args.iro_ckpt, device, input_shape)
    esrm_algorithm, _, esrm_loss_fn, _, _ = load_algorithm_from_ckpt(args.esrm_ckpt, device, input_shape)

    pareto = Pareto_distribution(iro_loss_fn)
    a, b = pareto.update(copy.deepcopy(iro_algorithm.network).to(device), minibatches)
    beta_samples = np.random.beta(a, b, size=args.samples)

    arrow = ArrowPrattDistribution(esrm_loss_fn)
    loc, scale = arrow.update(copy.deepcopy(esrm_algorithm.network).to(device), minibatches)
    gamma_samples = np.random.lognormal(mean=loc, sigma=scale, size=args.samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(beta_samples, bins=30, color="C0", alpha=0.8)
    axes[0].set_title(f"IRO Beta(a={a:.2f}, b={b:.2f})")
    axes[0].set_xlabel("CVaR alpha")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(gamma_samples, bins=30, color="C3", alpha=0.8)
    axes[1].set_title(f"ESRM LogNormal(μ={loc:.2f}, σ={scale:.2f})")
    axes[1].set_xlabel("Arrow-Pratt gamma")

    fig.suptitle("Learned risk-parameter distributions (re-estimated post-training)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved distribution histograms to {args.output}")


if __name__ == "__main__":
    main()
