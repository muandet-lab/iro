import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from iro.cmnist_exp.vis.common import (
    collect_metrics,
    get_env_loaders,
    load_algorithm_from_ckpt,
    parse_probabilities,
    peek_loss_name,
)


def parse_list(spec):
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def compute_front(algorithm, loaders, device, params, loss_fn):
    avg_accs, worst_losses = [], []
    for val in params:
        accs, losses = collect_metrics(algorithm, loaders, device, val, loss_fn)
        avg_accs.append(accs.mean())
        worst_losses.append(losses.max())
    return np.array(avg_accs), np.array(worst_losses)


def main():
    parser = argparse.ArgumentParser(description="Pareto front between accuracy and worst-case loss.")
    parser.add_argument("--iro_ckpt", required=True)
    parser.add_argument("--esrm_ckpt", required=True)
    parser.add_argument("--iro_alphas", type=str, default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--esrm_gammas", type=str, default="0.2,0.5,1.0,2.0,5.0")
    parser.add_argument("--env_grid", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_pareto.png")
    args = parser.parse_args()

    env_ps = parse_probabilities(args.env_grid)
    if len(env_ps) == 0:
        raise ValueError("Specify at least one evaluation environment via --env_grid.")

    device = torch.device(args.device)
    loss_name = peek_loss_name(args.iro_ckpt)
    if peek_loss_name(args.esrm_ckpt) != loss_name:
        raise ValueError("Checkpoints use different loss functions; cannot compare directly.")

    loaders, input_shape = get_env_loaders(
        args.data_dir, env_ps, device,
        loss_name=loss_name,
        batch_size=args.batch_size,
        label_noise=args.label_noise,
        full_resolution=args.full_resolution,
        use_test_set=True
    )

    iro_algorithm, _, iro_loss_fn, _, _ = load_algorithm_from_ckpt(args.iro_ckpt, device, input_shape)
    esrm_algorithm, _, esrm_loss_fn, _, _ = load_algorithm_from_ckpt(args.esrm_ckpt, device, input_shape)

    iro_params = parse_list(args.iro_alphas)
    esrm_params = parse_list(args.esrm_gammas)
    if len(iro_params) == 0 or len(esrm_params) == 0:
        raise ValueError("Provide non-empty parameter lists for both algorithms.")

    iro_acc, iro_worst = compute_front(iro_algorithm, loaders, device, iro_params, iro_loss_fn)
    esrm_acc, esrm_worst = compute_front(esrm_algorithm, loaders, device, esrm_params, esrm_loss_fn)

    plt.figure(figsize=(7, 5))
    plt.scatter(iro_worst, iro_acc, c="C0", label="IRO")
    for x, y, p in zip(iro_worst, iro_acc, iro_params):
        plt.annotate(f"{p:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.scatter(esrm_worst, esrm_acc, c="C3", label="ESRM")
    for x, y, p in zip(esrm_worst, esrm_acc, esrm_params):
        plt.annotate(f"{p:.2f}", (x, y), textcoords="offset points", xytext=(4, -8), fontsize=8, color="C3")
    plt.xlabel("Worst-case loss across envs")
    plt.ylabel("Average accuracy across envs")
    plt.title("Pareto frontier: accuracy vs worst-case loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved Pareto front to {args.output}")


if __name__ == "__main__":
    main()
